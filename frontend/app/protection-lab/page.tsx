"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  compositeCropIntoImage,
  craftAdversarialFace,
  cropFaceToTensor,
  detectFaceBox,
  loadFaceModels,
  tensorToCanvas,
  type AttackParams,
  type FaceBox,
} from "@/lib/faceVector";
import { computePixelDelta, computePSNR, computeSSIM } from "@/lib/metrics";
import {
  addAttributionWatermark,
  applyGenerationInterferenceProxy,
  applyTrainingInterferenceProxy,
} from "@/lib/protectionLab";
import { fileToImageData, imageDataToDataUrl } from "@/lib/noise";

type Result = {
  key: string;
  title: string;
  badge: string;
  note: string;
  image: ImageData;
  faceCosineBefore?: number;
  faceCosineAfter?: number;
};

type FaceAttackOutput = {
  image: ImageData;
  cosineBefore: number;
  cosineAfter: number;
};

const FACE_PARAMS: AttackParams = { epsilon: 0.025, stepSize: 0.002, steps: 24, smoothing: true };

function loadImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("이미지를 불러오지 못했습니다."));
    image.src = dataUrl;
  });
}

async function runFacePerturbation(imageData: ImageData): Promise<FaceAttackOutput> {
  const dataUrl = imageDataToDataUrl(imageData);
  const image = await loadImage(dataUrl);
  const box: FaceBox = await detectFaceBox(image);
  const { tensor, cropBox } = cropFaceToTensor(image, box);
  try {
    const attacked = await craftAdversarialFace(tensor, FACE_PARAMS);
    try {
      const faceCanvas = await tensorToCanvas(attacked.tensor);
      const compositeUrl = compositeCropIntoImage(image, cropBox, faceCanvas);
      const composite = await loadImage(compositeUrl);
      const canvas = document.createElement("canvas");
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("이미지 캔버스를 만들 수 없습니다.");
      ctx.drawImage(composite, 0, 0, canvas.width, canvas.height);
      return {
        image: ctx.getImageData(0, 0, canvas.width, canvas.height),
        cosineBefore: attacked.cosineSimilarityBefore,
        cosineAfter: attacked.cosineSimilarityAfter,
      };
    } finally {
      attacked.tensor.dispose();
    }
  } finally {
    tensor.dispose();
  }
}

function ResultCard({
  result,
  original,
  onImageClick,
}: {
  result: Result;
  original: ImageData;
  onImageClick: (title: string, dataUrl: string) => void;
}) {
  const dataUrl = imageDataToDataUrl(result.image);
  const psnr = computePSNR(original, result.image);
  const ssim = computeSSIM(original, result.image);
  const delta = computePixelDelta(original, result.image);
  return (
    <article className="overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-900">
      <div className="relative aspect-[4/3] bg-neutral-950">
        <button
          type="button"
          className="h-full w-full cursor-zoom-in"
          onClick={() => onImageClick(result.title, dataUrl)}
          aria-label={`${result.title} 크게 보기`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={dataUrl} alt={result.title} className="h-full w-full object-contain" />
        </button>
      </div>
      <div className="space-y-2 p-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-neutral-100">{result.title}</h3>
          <span className="shrink-0 rounded-full border border-neutral-700 px-2 py-1 text-[10px] text-neutral-300">{result.badge}</span>
        </div>
        <p className="min-h-10 text-xs leading-5 text-neutral-400">{result.note}</p>
        <dl className="grid grid-cols-2 gap-2 pt-1">
          <Metric label="원본 유사도 · PSNR" value={psnr === Infinity ? "∞" : `${psnr.toFixed(1)} dB`} />
          <Metric label="구조 유사도 · SSIM" value={ssim.toFixed(3)} />
          <Metric label="평균 픽셀 변화 · MAE" value={delta.meanAbsoluteError.toFixed(2)} />
          <Metric label="변화한 픽셀" value={`${delta.changedPixelPercent.toFixed(1)}%`} />
          <Metric
            label="최대 채널 변화 · 0–255"
            value={delta.maxAbsoluteError.toFixed(0)}
          />
          <Metric
            label="보호 성능 검증"
            value={result.faceCosineAfter === undefined ? "아직 미측정" : "얼굴 임베딩만"}
          />
        </dl>
        {result.faceCosineAfter !== undefined && (
          <div className="rounded-lg border border-sky-900/70 bg-sky-950/30 p-3 text-xs">
            <p className="font-medium text-sky-200">얼굴 임베딩 코사인 유사도</p>
            <p className="mt-1 text-neutral-300">
              원본 기준 {result.faceCosineBefore?.toFixed(4) ?? "—"} → 변형 후 {result.faceCosineAfter.toFixed(4)}
            </p>
            <p className="mt-1 leading-5 text-neutral-500">낮아지면 이 브라우저 모델의 얼굴 특징 벡터가 달라졌다는 뜻입니다. 실제 인식 성공률이나 다른 모델의 성능을 나타내지는 않습니다.</p>
          </div>
        )}
        <a
          href={dataUrl}
          download={`${result.key}.png`}
          className="inline-flex rounded-lg border border-neutral-700 px-3 py-2 text-xs text-neutral-200 hover:border-neutral-500"
        >
          PNG 저장
        </a>
      </div>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-neutral-950/70 px-3 py-2">
      <dt className="text-[10px] leading-4 text-neutral-500">{label}</dt>
      <dd className="mt-1 text-sm font-medium tabular-nums text-neutral-200">{value}</dd>
    </div>
  );
}

export default function ProtectionLabPage() {
  const [original, setOriginal] = useState<ImageData | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [status, setStatus] = useState("사진을 선택해 실험을 시작하세요.");
  const [busy, setBusy] = useState(false);
  const [lightbox, setLightbox] = useState<{ title: string; dataUrl: string } | null>(null);

  useEffect(() => {
    if (!lightbox) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [lightbox]);

  async function run(file: File) {
    setBusy(true);
    setResults([]);
    setStatus("사진을 준비하고 있습니다. 미리보기와 계산은 512px 이하로 진행됩니다.");
    try {
      const image = await fileToImageData(file, 512);
      setOriginal(image);

      const training = applyTrainingInterferenceProxy(image);
      const generation = addAttributionWatermark(applyGenerationInterferenceProxy(image));
      setResults([
        {
          key: "training-proxy",
          title: "학습 방해 실험",
          badge: "대리 실험",
          note: "다중 주파수 섭동을 적용한 비교용 출력입니다. 학습 모델을 최적화하지 않으므로 Unlearnable Examples/Fawkes 구현은 아닙니다.",
          image: training,
        },
        {
          key: "generation-proxy",
          title: "생성·편집 방해 실험",
          badge: "대리 실험",
          note: "고주파 패턴과 눈에 보이는 출처 워터마크를 적용합니다. Stable Diffusion VAE에 대한 PhotoGuard 최적화는 포함하지 않습니다.",
          image: generation,
        },
      ]);

      setStatus("얼굴 인식 회피 결과를 계산하고 있습니다. 브라우저에서 얼굴 모델을 처음 불러오면 시간이 걸립니다.");
      let faceResult: FaceAttackOutput | null = null;
      let faceError = "";
      try {
        await loadFaceModels();
        faceResult = await runFacePerturbation(image);
        setResults((previous) => [
          ...previous,
          {
            key: "recognition-evasion",
            title: "인식 회피 실험",
            badge: "얼굴 모델 기반",
            note: "브라우저 얼굴 임베딩의 특징 벡터를 바꾸는 기존 실험을 적용했습니다. 상용 인식 서비스에 대한 효과를 보장하지 않습니다.",
            image: faceResult!.image,
            faceCosineBefore: faceResult!.cosineBefore,
            faceCosineAfter: faceResult!.cosineAfter,
          },
        ]);
      } catch (error) {
        faceError = error instanceof Error ? error.message : "얼굴 인식 회피 처리를 완료하지 못했습니다.";
      }

      setStatus("세 기능을 순서대로 적용한 통합 출력을 만들고 있습니다.");
      const combinedBase = addAttributionWatermark(
        applyGenerationInterferenceProxy(applyTrainingInterferenceProxy(image))
      );
      let combined = combinedBase;
      let combinedNote = "학습 방해·생성 방해 대리 실험과 출처 워터마크를 차례로 적용했습니다.";
      let faceApplied = false;
      let combinedFaceCosine: FaceAttackOutput | null = null;
      try {
        // Models are loaded above; re-run the face objective on the combined image.
        combinedFaceCosine = await runFacePerturbation(combinedBase);
        combined = combinedFaceCosine.image;
        faceApplied = true;
        combinedNote += " 얼굴 임베딩 섭동도 추가했습니다.";
      } catch {
        combinedNote += " 얼굴을 찾지 못해 인식 회피 단계는 제외했습니다.";
      }
      setResults((previous) => [
        ...previous,
        {
          key: "combined-prototype",
          title: "통합 실험 버전",
          badge: faceApplied ? "3단계 적용" : "부분 적용",
          note: `${combinedNote} 두 대리 실험은 해당 논문의 모델 기반 구현이 아닙니다.`,
          image: combined,
          faceCosineBefore: combinedFaceCosine?.cosineBefore,
          faceCosineAfter: combinedFaceCosine?.cosineAfter,
        },
      ]);
      setStatus(faceError ? `완료. 얼굴 인식 실험: ${faceError}` : "완료. 원본과 각 결과를 비교하고 PNG로 저장할 수 있습니다.");
    } catch (error) {
      setStatus(error instanceof Error ? `오류: ${error.message}` : "처리 중 오류가 발생했습니다.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="border-b border-neutral-800 px-6 py-6">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4">
          <div>
            <Link href="/" className="text-xs text-neutral-500 hover:text-neutral-300">← 노이즈 실험 홈</Link>
            <h1 className="mt-2 text-2xl font-semibold">이미지 보호 통합 실험실</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-neutral-400">
              한 장의 사진으로 인식 회피, 학습 방해, 생성·편집 방해 출력과 통합 결과를 나란히 비교합니다.
            </p>
          </div>
          <label className={`cursor-pointer rounded-xl px-5 py-3 text-sm font-medium ${busy ? "bg-neutral-700 text-neutral-400" : "bg-white text-neutral-950 hover:bg-neutral-200"}`}>
            {busy ? "실험 중…" : "사진 선택 및 실험"}
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void run(file);
                event.currentTarget.value = "";
              }}
            />
          </label>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-6">
        <div className="mb-6 rounded-xl border border-amber-900/70 bg-amber-950/30 px-4 py-3 text-xs leading-5 text-amber-100/80">
          학습·생성 방해는 현재 모델이 없는 상태에서 비교를 위한 대리 섭동으로 동작합니다. 이 결과만으로 학습 방해나 딥페이크 편집 차단 효과를 주장할 수 없습니다. 논문 방식의 검증에는 학습 모델과 Stable Diffusion 계열 VAE가 필요합니다.
        </div>
        <p role="status" className="mb-6 text-sm text-neutral-400">{status}</p>

        {original && (
          <section className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
            <article className="overflow-hidden rounded-2xl border border-neutral-800 bg-neutral-900">
              <div className="relative aspect-[4/3] bg-neutral-950">
                <button
                  type="button"
                  className="h-full w-full cursor-zoom-in"
                  onClick={() => setLightbox({ title: "원본 사진", dataUrl: imageDataToDataUrl(original) })}
                  aria-label="원본 사진 크게 보기"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imageDataToDataUrl(original)} alt="업로드한 원본" className="h-full w-full object-contain" />
                </button>
              </div>
              <div className="p-4"><h2 className="text-sm font-semibold">원본 사진</h2><p className="mt-2 text-xs text-neutral-500">모든 결과와 비교하는 기준 이미지</p></div>
            </article>
            {results.map((result) => (
              <ResultCard
                key={result.key}
                result={result}
                original={original}
                onImageClick={(title, dataUrl) => setLightbox({ title, dataUrl })}
              />
            ))}
          </section>
        )}

        {!original && (
          <div className="rounded-2xl border border-dashed border-neutral-700 bg-neutral-900/60 px-6 py-16 text-center text-sm text-neutral-500">
            사진을 선택하면 원본, 각 실험 결과, 통합 출력이 이곳에 표시됩니다.
          </div>
        )}

        <section className="mt-10 grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-neutral-800 p-4"><h2 className="text-sm font-semibold">인식 회피</h2><p className="mt-2 text-xs leading-5 text-neutral-500">기존 브라우저 얼굴 임베딩 실험을 재사용합니다.</p></div>
          <div className="rounded-xl border border-neutral-800 p-4"><h2 className="text-sm font-semibold">학습 방해</h2><p className="mt-2 text-xs leading-5 text-neutral-500">현재는 대리 노이즈 단계입니다. 실제 Unlearnable Examples는 대상 학습 모델에 대한 손실 최적화가 추가되어야 합니다.</p></div>
          <div className="rounded-xl border border-neutral-800 p-4"><h2 className="text-sm font-semibold">생성·편집 방해</h2><p className="mt-2 text-xs leading-5 text-neutral-500">현재는 고주파 대리 섭동과 출처 워터마크입니다. PhotoGuard는 VAE 또는 확산 모델을 연결해야 합니다.</p></div>
        </section>
      </div>

      {lightbox && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`${lightbox.title} 크게 보기`}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-4 sm:p-8"
          onClick={() => setLightbox(null)}
        >
          <button
            type="button"
            autoFocus
            aria-label="크게 보기 닫기"
            className="absolute right-4 top-4 z-10 flex h-11 w-11 items-center justify-center rounded-full border border-white/20 bg-neutral-900/80 text-2xl text-white hover:bg-neutral-700 sm:right-7 sm:top-7"
            onClick={() => setLightbox(null)}
          >
            ×
          </button>
          <div className="flex max-h-full max-w-full flex-col items-center gap-3" onClick={(event) => event.stopPropagation()}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={lightbox.dataUrl}
              alt={lightbox.title}
              className="max-h-[86vh] max-w-[94vw] object-contain"
            />
            <p className="text-sm text-neutral-300">{lightbox.title}</p>
          </div>
        </div>
      )}
    </main>
  );
}
