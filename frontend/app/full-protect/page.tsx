"use client";

import { useCallback, useEffect, useState } from "react";

// 백엔드(FastAPI) 주소. GCP VM에 올려서 돌리는 걸 기준으로 합니다.
// (로컬에서 backend/ 를 직접 띄워서 쓰려면 아래 줄을 "http://localhost:8000"으로 바꾸면 됩니다.)
const BACKEND_URL = "http://35.238.45.143:8000";

type HealthState = {
  status: "checking" | "online" | "offline";
  faceModels: string[];
  photoguardAvailable: boolean;
  photoguardError: string | null;
};

type ProtectMetrics = {
  face_cosine_before: Record<string, number>;
  face_cosine_after: Record<string, number>;
  photoguard_used: boolean;
  photoguard_latent_distance: number | null;
  psnr: number;
  ssim: number;
};

type ProtectResponse = {
  protected_image_base64: string;
  metrics: ProtectMetrics;
  warnings: string[];
  elapsed_seconds: number;
};

export default function FullProtectPage() {
  const [health, setHealth] = useState<HealthState>({
    status: "checking",
    faceModels: [],
    photoguardAvailable: false,
    photoguardError: null,
  });

  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [epsilon, setEpsilon] = useState(0.04);
  const [steps, setSteps] = useState(30);
  const [usePhotoguard, setUsePhotoguard] = useState(true);

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProtectResponse | null>(null);

  // 페이지가 열리면 백엔드가 떠있는지 먼저 확인합니다.
  const checkHealth = useCallback(async () => {
    setHealth((prev) => ({ ...prev, status: "checking" }));
    try {
      const res = await fetch(`${BACKEND_URL}/health`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      setHealth({
        status: "online",
        faceModels: data.face_models ?? [],
        photoguardAvailable: !!data.photoguard_available,
        photoguardError: data.photoguard_error ?? null,
      });
    } catch {
      setHealth({
        status: "offline",
        faceModels: [],
        photoguardAvailable: false,
        photoguardError: null,
      });
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const handleFile = useCallback((file: File) => {
    setSelectedFile(file);
    setFileName(file.name);
    setResult(null);
    setError(null);
    setOriginalUrl(URL.createObjectURL(file));
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

  const runProtection = useCallback(async () => {
    if (!selectedFile) return;
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("epsilon", String(epsilon));
      form.append("steps", String(steps));
      form.append("use_photoguard", String(usePhotoguard));

      const res = await fetch(`${BACKEND_URL}/protect`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        throw new Error(errBody?.error ?? `서버 오류 (status ${res.status})`);
      }

      const data: ProtectResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "알 수 없는 오류가 발생했습니다.");
    } finally {
      setIsRunning(false);
    }
  }, [selectedFile, epsilon, steps, usePhotoguard]);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="border-b border-neutral-800 px-6 py-5">
        <h1 className="text-xl font-semibold">앙상블 LowKey + PhotoGuard 결합 보호</h1>
        <p className="mt-1 text-sm text-neutral-400">
          얼굴 인식 모델 2개를 동시에 공격하는 앙상블과, 이미지 생성/편집 AI를 방해하는
          PhotoGuard를 하나의 섭동으로 결합해서 실제 사진 전체에 적용합니다. 백엔드(FastAPI)
          서버가 로컬에서 실행 중이어야 합니다.
        </p>
        <a
          href="/"
          className="mt-3 inline-block rounded-lg border border-neutral-700 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:border-neutral-500 hover:text-white"
        >
          ← 메인으로
        </a>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {/* 백엔드 상태 */}
        <div className="mb-6 rounded-lg border border-neutral-800 bg-neutral-900 px-4 py-3 text-sm">
          {health.status === "checking" && <p className="text-neutral-400">백엔드 서버 상태 확인 중...</p>}
          {health.status === "offline" && (
            <div className="text-amber-300">
              <p className="font-medium">백엔드 서버에 연결할 수 없습니다.</p>
              <p className="mt-1 text-xs text-amber-200/80">
                터미널에서 backend/ 폴더로 이동한 뒤 아래 명령으로 서버를 먼저 띄워주세요:
              </p>
              <pre className="mt-2 overflow-x-auto rounded bg-black/40 px-3 py-2 text-xs text-neutral-300">
{`cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000`}
              </pre>
              <button
                onClick={checkHealth}
                className="mt-2 rounded-md border border-amber-700 px-2 py-1 text-xs text-amber-200 hover:border-amber-500"
              >
                다시 확인
              </button>
            </div>
          )}
          {health.status === "online" && (
            <div>
              <p className="text-emerald-300">
                ✓ 백엔드 연결됨 — 얼굴 인식 모델 {health.faceModels.length}개 ({health.faceModels.join(", ")})
              </p>
              <p className="mt-1 text-xs text-neutral-400">
                PhotoGuard(생성모델 편집 방해):{" "}
                {health.photoguardAvailable ? (
                  <span className="text-emerald-300">사용 가능</span>
                ) : (
                  <span className="text-neutral-500">
                    사용 불가 (앙상블 공격만 적용됩니다{health.photoguardError ? " — 네트워크/버전 문제" : ""})
                  </span>
                )}
              </p>
            </div>
          )}
        </div>

        {/* 업로드 영역 */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="mb-6 flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-neutral-700 bg-neutral-900 px-6 py-10 text-center"
        >
          <p className="mb-3 text-sm text-neutral-400">
            보호하고 싶은 사진을 끌어다 놓거나, 아래 버튼으로 선택하세요.
          </p>
          <label className="cursor-pointer rounded-lg bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white">
            사진 선택
            <input type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
          </label>
          {fileName && <p className="mt-3 text-xs text-neutral-500">선택한 파일: {fileName}</p>}
        </div>

        {/* 옵션 */}
        {originalUrl && (
          <div className="mb-6 grid grid-cols-1 gap-4 rounded-xl border border-neutral-800 bg-neutral-900 p-4 sm:grid-cols-3">
            <label className="flex flex-col gap-1 text-xs text-neutral-400">
              섭동 크기 (epsilon): {epsilon.toFixed(3)}
              <input
                type="range"
                min={0.01}
                max={0.08}
                step={0.005}
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              />
              <span className="text-neutral-500">클수록 공격 효과↑, 화질 저하↑</span>
            </label>
            <label className="flex flex-col gap-1 text-xs text-neutral-400">
              반복 횟수 (steps): {steps}
              <input
                type="range"
                min={5}
                max={80}
                step={5}
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value, 10))}
              />
              <span className="text-neutral-500">클수록 효과↑, 처리 시간↑ (CPU 기준)</span>
            </label>
            <label className="flex items-center gap-2 text-xs text-neutral-400">
              <input
                type="checkbox"
                checked={usePhotoguard}
                disabled={!health.photoguardAvailable}
                onChange={(e) => setUsePhotoguard(e.target.checked)}
              />
              PhotoGuard도 같이 적용 {!health.photoguardAvailable && "(현재 사용 불가)"}
            </label>
          </div>
        )}

        {originalUrl && (
          <button
            onClick={runProtection}
            disabled={isRunning || health.status !== "online"}
            className="mb-8 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-neutral-700 disabled:text-neutral-400"
          >
            {isRunning ? "처리 중... (CPU에서는 수십 초~몇 분 걸릴 수 있어요)" : "보호 처리 시작"}
          </button>
        )}

        {error && (
          <div className="mb-8 rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-200">
            오류: {error}
          </div>
        )}

        {originalUrl && (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
            <div>
              <h2 className="mb-2 text-sm font-semibold text-neutral-300">원본 사진</h2>
              <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={originalUrl} alt="원본 사진" className="w-full" />
              </div>
            </div>

            <div>
              <h2 className="mb-2 text-sm font-semibold text-neutral-300">보호된 사진</h2>
              <div className="flex min-h-[200px] items-center justify-center overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
                {result ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:image/png;base64,${result.protected_image_base64}`}
                    alt="보호된 사진"
                    className="w-full"
                  />
                ) : (
                  <p className="px-6 text-center text-xs text-neutral-500">
                    {isRunning ? "처리 중입니다..." : "보호 처리를 시작하면 여기에 결과가 표시됩니다."}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {result && (
          <section className="mt-8 rounded-xl border border-neutral-800 bg-neutral-900 p-5">
            <h2 className="mb-3 text-sm font-semibold text-neutral-200">결과 지표</h2>

            <div className="mb-4">
              <p className="mb-1 text-xs font-medium text-neutral-400">
                얼굴 인식 앙상블 — 코사인 유사도 (1.0 = 완전히 같은 사람으로 인식, 낮을수록 회피 성공)
              </p>
              <ul className="space-y-1 text-xs text-neutral-300">
                {Object.keys(result.metrics.face_cosine_after).map((name) => (
                  <li key={name}>
                    <span className="text-neutral-400">{name}:</span>{" "}
                    {result.metrics.face_cosine_before[name]?.toFixed(4)} →{" "}
                    <span className="font-medium text-emerald-300">
                      {result.metrics.face_cosine_after[name]?.toFixed(4)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mb-4 text-xs text-neutral-300">
              <p className="font-medium text-neutral-400">PhotoGuard (생성모델 편집 방해)</p>
              {result.metrics.photoguard_used ? (
                <p>
                  latent 거리:{" "}
                  <span className="font-medium text-emerald-300">
                    {result.metrics.photoguard_latent_distance?.toFixed(4)}
                  </span>{" "}
                  (클수록 편집 방해 효과가 큽니다)
                </p>
              ) : (
                <p className="text-neutral-500">적용 안 됨</p>
              )}
            </div>

            <div className="mb-4 text-xs text-neutral-300">
              <p className="font-medium text-neutral-400">전체 사진 화질 (원본과 비교)</p>
              <p>
                PSNR: {result.metrics.psnr.toFixed(2)} dB · SSIM: {result.metrics.ssim.toFixed(4)}
              </p>
              <p className="mt-0.5 text-neutral-500">
                PSNR 40dB 이상, SSIM 0.95 이상이면 사람 눈에는 거의 원본과 같아 보입니다.
              </p>
            </div>

            {result.warnings.length > 0 && (
              <div className="mb-2 rounded-md border border-amber-800 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
                {result.warnings.map((w, i) => (
                  <p key={i}>⚠ {w}</p>
                ))}
              </div>
            )}

            <p className="text-xs text-neutral-500">처리 시간: {result.elapsed_seconds.toFixed(1)}초</p>
          </section>
        )}
      </main>
    </div>
  );
}
