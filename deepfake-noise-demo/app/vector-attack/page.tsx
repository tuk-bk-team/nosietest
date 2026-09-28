"use client";

import { useCallback, useRef, useState } from "react";
import {
  loadFaceModels,
  detectFaceBox,
  cropFaceToTensor,
  craftAdversarialFace,
  tensorToCanvas,
  canvasToImageData,
  compositeCropIntoImage,
  computePSNR,
  type FaceBox,
  type AttackParams,
} from "@/lib/faceVector";

// --------------------------------------------------------------------------
// 파라미터 프리셋
// --------------------------------------------------------------------------
// DEFAULT: 이전에 파이썬(adversarial_face_protect.py)으로 튜닝했던,
//          "눈에 안 보이면서 효과적인" 균형점입니다.
const DEFAULT_PARAMS: AttackParams = { epsilon: 0.03, stepSize: 0.0025, steps: 50, smoothing: true };

type Scenario = {
  key: string;
  title: string;
  paramLabel: string;
  description: string;
  params: AttackParams;
};

// 각 파라미터를 하나씩 극단으로 바꿔가며 그 파라미터의 역할을 보여주는 실험 목록입니다.
// (나머지 값은 DEFAULT_PARAMS를 그대로 사용합니다)
const SCENARIOS: Scenario[] = [
  {
    key: "epsilon-min",
    title: "epsilon 최솟값",
    paramLabel: "epsilon = 0.005 (기본값 0.03)",
    description: "픽셀 변화 한도를 아주 작게 주면, 벡터를 충분히 멀리 밀어내지 못해 보호 효과가 약합니다.",
    params: { ...DEFAULT_PARAMS, epsilon: 0.005 },
  },
  {
    key: "epsilon-max",
    title: "epsilon 최댓값",
    paramLabel: "epsilon = 0.15 (기본값 0.03)",
    description: "픽셀 변화 한도를 크게 풀어주면 보호 효과는 강해지지만, 그만큼 눈에도 잘 띕니다.",
    params: { ...DEFAULT_PARAMS, epsilon: 0.15 },
  },
  {
    key: "stepsize-min",
    title: "step_size 최솟값",
    paramLabel: "step_size = 0.0008 (기본값 0.0025)",
    description: "한 걸음이 너무 작으면, 같은 반복 횟수(steps) 안에 epsilon 한도까지 도달하지 못합니다.",
    params: { ...DEFAULT_PARAMS, stepSize: 0.0008 },
  },
  {
    key: "stepsize-max",
    title: "step_size 최댓값",
    paramLabel: "step_size = 0.02 (기본값 0.0025)",
    description: "한 걸음을 크게 주면 몇 스텝 만에 epsilon 한도에 도달해, 적은 반복으로도 효과가 빨리 나타납니다.",
    params: { ...DEFAULT_PARAMS, stepSize: 0.02 },
  },
  {
    key: "steps-min",
    title: "steps 최솟값",
    paramLabel: "steps = 5 (기본값 50)",
    description: "반복 횟수가 너무 적으면 최적화가 끝나기 전에 멈춰서, 보호 효과가 충분히 나타나지 않습니다.",
    params: { ...DEFAULT_PARAMS, steps: 5 },
  },
  {
    key: "steps-max",
    title: "steps 최댓값",
    paramLabel: "steps = 150 (기본값 50)",
    description: "반복을 아주 많이 해도, epsilon 한도 안에서는 결과가 어느 시점부터 거의 수렴합니다.",
    params: { ...DEFAULT_PARAMS, steps: 150 },
  },
  {
    key: "smoothing-off",
    title: "가우시안 스무딩 끔",
    paramLabel: "smoothing = off (기본값 on)",
    description: "스무딩을 끄면 픽셀 단위로 날카로운 섭동이 생겨, 같은 epsilon이어도 더 거칠어 보일 수 있습니다.",
    params: { ...DEFAULT_PARAMS, smoothing: false },
  },
];

type ScenarioResult = {
  key: string;
  title: string;
  paramLabel: string;
  description: string;
  beforeDataUrl: string;
  afterDataUrl: string;
  cosineAfter: number;
  psnr: number;
};

export default function VectorAttackPage() {
  const [status, setStatus] = useState<string>("사진을 올리면 시작합니다.");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<{ label: string; step: number; total: number } | null>(null);

  const [originalFullUrl, setOriginalFullUrl] = useState<string | null>(null);
  const [protectedFullUrl, setProtectedFullUrl] = useState<string | null>(null);
  const [mainCosine, setMainCosine] = useState<{ before: number; after: number } | null>(null);
  const [mainPsnr, setMainPsnr] = useState<number | null>(null);

  const [scenarioResults, setScenarioResults] = useState<ScenarioResult[]>([]);
  const modelsReady = useRef(false);

  const handleFile = useCallback(async (file: File) => {
    setIsProcessing(true);
    setScenarioResults([]);
    setProtectedFullUrl(null);
    setMainCosine(null);
    setMainPsnr(null);

    try {
      if (!modelsReady.current) {
        setStatus("얼굴 인식 모델 불러오는 중... (처음 한 번만 시간이 걸립니다)");
        await loadFaceModels();
        modelsReady.current = true;
      }

      // --- 이미지 엘리먼트 준비 ---
      const dataUrl = await fileToDataUrl(file);
      const img = await loadImage(dataUrl);
      setOriginalFullUrl(dataUrl);

      // --- 얼굴 검출 & 크롭 ---
      setStatus("얼굴 검출 중...");
      const box: FaceBox = await detectFaceBox(img);
      const { tensor: cropTensor, cropBox } = cropFaceToTensor(img, box);

      // --- 1) 메인 결과: 기본(추천) 파라미터로 보호 ---
      setStatus("보호된 이미지 계산 중 (기본 파라미터)...");
      const mainResult = await craftAdversarialFace(cropTensor, DEFAULT_PARAMS, (step, total, cos) => {
        setProgress({ label: "메인 결과", step, total });
      });
      const mainCanvas = await tensorToCanvas(mainResult.tensor);
      const originalCropCanvas = await tensorToCanvas(cropTensor);
      const psnr = computePSNR(canvasToImageData(originalCropCanvas), canvasToImageData(mainCanvas));

      setProtectedFullUrl(compositeCropIntoImage(img, cropBox, mainCanvas));
      setMainCosine({ before: mainResult.cosineSimilarityBefore, after: mainResult.cosineSimilarityAfter });
      setMainPsnr(psnr);
      mainResult.tensor.dispose();

      // --- 2) 파라미터를 하나씩 최고/최저로 바꾼 실험들 ---
      const results: ScenarioResult[] = [];
      for (let i = 0; i < SCENARIOS.length; i++) {
        const scenario = SCENARIOS[i];
        setStatus(`파라미터 실험 중... (${i + 1}/${SCENARIOS.length}: ${scenario.title})`);
        const result = await craftAdversarialFace(cropTensor, scenario.params, (step, total) => {
          setProgress({ label: scenario.title, step, total });
        });
        const afterCanvas = await tensorToCanvas(result.tensor);
        const scenarioPsnr = computePSNR(canvasToImageData(originalCropCanvas), canvasToImageData(afterCanvas));

        results.push({
          key: scenario.key,
          title: scenario.title,
          paramLabel: scenario.paramLabel,
          description: scenario.description,
          beforeDataUrl: originalCropCanvas.toDataURL("image/png"),
          afterDataUrl: afterCanvas.toDataURL("image/png"),
          cosineAfter: result.cosineSimilarityAfter,
          psnr: scenarioPsnr,
        });
        result.tensor.dispose();
        setScenarioResults([...results]);
      }

      cropTensor.dispose();
      setStatus("완료되었습니다.");
    } catch (err) {
      console.error(err);
      setStatus(err instanceof Error ? `오류: ${err.message}` : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsProcessing(false);
      setProgress(null);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="border-b border-neutral-800 px-6 py-5">
        <h1 className="text-xl font-semibold">LowKey/Fawkes 원리 데모 — 벡터 기반 적대적 섭동</h1>
        <p className="mt-1 max-w-3xl text-sm text-neutral-400">
          사진을 올리면 실제 얼굴 인식 신경망(브라우저 안에서 TensorFlow.js로 직접 실행)이 뽑아내는
          특징 벡터를 계산하고, 그 벡터가 최대한 멀어지도록 픽셀을 미세하게 바꿉니다. 서버 없이
          이 브라우저 안에서 모든 계산이 이루어집니다.
        </p>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {/* 업로드 영역 */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="mb-6 flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-neutral-700 bg-neutral-900 px-6 py-10 text-center"
        >
          <p className="mb-3 text-sm text-neutral-400">사진을 이 영역에 끌어다 놓거나, 아래 버튼으로 선택하세요. (얼굴이 잘 보이는 사진)</p>
          <label className="cursor-pointer rounded-lg bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white">
            사진 선택
            <input type="file" accept="image/*" className="hidden" onChange={handleFileInput} disabled={isProcessing} />
          </label>
        </div>

        {/* 상태 표시 */}
        <div className="mb-8 text-sm text-neutral-400">
          <p data-testid="status-text">{status}</p>
          {progress && (
            <div className="mt-2">
              <div className="h-1.5 w-full max-w-md overflow-hidden rounded-full bg-neutral-800">
                <div
                  className="h-full bg-emerald-500 transition-all"
                  style={{ width: `${(progress.step / progress.total) * 100}%` }}
                />
              </div>
              <p className="mt-1 text-xs">
                {progress.label}: {progress.step}/{progress.total} 스텝
              </p>
            </div>
          )}
        </div>

        {/* 메인 결과 */}
        {originalFullUrl && (
          <section className="mb-12">
            <h2 className="mb-3 text-lg font-semibold text-neutral-100">메인 결과 (기본 파라미터)</h2>
            <p className="mb-4 text-xs text-neutral-500">
              epsilon=0.03, step_size=0.0025, steps=50, 가우시안 스무딩=켬 — 이전 파이썬 실험에서 찾은
              &quot;눈에 안 보이면서 효과적인&quot; 균형점입니다.
            </p>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-medium text-neutral-400">원본 사진</p>
                <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={originalFullUrl} alt="원본 사진" className="w-full" />
                </div>
              </div>
              <div>
                <p className="mb-1 text-xs font-medium text-neutral-400">보호된 사진 (사람 눈에는 거의 같아 보임)</p>
                <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
                  {protectedFullUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={protectedFullUrl} alt="보호된 사진" className="w-full" />
                  ) : (
                    <div className="flex aspect-square items-center justify-center text-xs text-neutral-600">계산 중...</div>
                  )}
                </div>
              </div>
            </div>
            {mainCosine && mainPsnr !== null && (
              <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-3">
                <StatCard label="벡터 코사인 유사도 (전)" value={mainCosine.before.toFixed(4)} hint="1.0 = 완전 동일 인물" />
                <StatCard label="벡터 코사인 유사도 (후)" value={mainCosine.after.toFixed(4)} hint="0에 가까울수록 다른 사람" highlight />
                <StatCard label="이미지 PSNR" value={`${mainPsnr.toFixed(1)} dB`} hint="40dB 이상 = 육안 구분 거의 불가" />
              </div>
            )}
          </section>
        )}

        {/* 파라미터별 실험 */}
        {scenarioResults.length > 0 && (
          <section>
            <h2 className="mb-1 text-lg font-semibold text-neutral-100">파라미터별 효과 실험</h2>
            <p className="mb-6 text-xs text-neutral-500">
              각 파라미터를 하나씩 최솟값/최댓값으로 바꿔보고, 나머지는 기본값을 유지했을 때의 차이를 보여줍니다.
              (아래 카드들은 얼굴 부분만 잘라서 비교합니다)
            </p>
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {scenarioResults.map((r) => (
                <div key={r.key} className="rounded-xl border border-neutral-800 bg-neutral-900 p-3">
                  <p className="text-sm font-semibold text-neutral-100">{r.title}</p>
                  <p className="mb-2 text-[11px] text-neutral-500">{r.paramLabel}</p>
                  <div className="mb-2 grid grid-cols-2 gap-2">
                    <div>
                      <p className="mb-1 text-[10px] text-neutral-500">원본</p>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={r.beforeDataUrl} alt="원본 크롭" className="w-full rounded-md" />
                    </div>
                    <div>
                      <p className="mb-1 text-[10px] text-neutral-500">변형 후</p>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={r.afterDataUrl} alt="변형된 크롭" className="w-full rounded-md" />
                    </div>
                  </div>
                  <p className="text-[11px] text-neutral-400">
                    코사인 유사도: <span className="font-medium text-neutral-200">{r.cosineAfter.toFixed(3)}</span> · PSNR:{" "}
                    <span className="font-medium text-neutral-200">{r.psnr.toFixed(1)}dB</span>
                  </p>
                  <p className="mt-2 text-[11px] leading-snug text-neutral-500">{r.description}</p>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

// --------------------------------------------------------------------------
// 유틸리티
// --------------------------------------------------------------------------

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function StatCard({
  label,
  value,
  hint,
  highlight,
}: {
  label: string;
  value: string;
  hint: string;
  highlight?: boolean;
}) {
  return (
    <div className={`rounded-lg border p-3 ${highlight ? "border-emerald-700 bg-emerald-950/30" : "border-neutral-800 bg-neutral-900"}`}>
      <p className="text-[11px] text-neutral-500">{label}</p>
      <p className="text-lg font-semibold text-neutral-100">{value}</p>
      <p className="text-[11px] text-neutral-500">{hint}</p>
    </div>
  );
}
