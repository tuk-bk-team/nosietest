"use client";

import { useCallback, useState } from "react";
import {
  fileToImageData,
  imageDataToDataUrl,
  testNoiseLevels,
  testChannelNoise,
  testFrequencyNoise,
  type NoiseResult,
} from "@/lib/noise";
import { computePSNR, computeSSIM } from "@/lib/metrics";

// 화면에 보여줄 결과 카드 1개의 형태
type ResultCard = {
  label: string;
  dataUrl: string;
  psnr: number;
  ssim: number;
};

type ResultGroup = {
  title: string;
  description: string;
  cards: ResultCard[];
};

// 처리 도중 브라우저 화면이 멈추지 않도록 한 프레임 양보하는 헬퍼
function yieldToBrowser() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function toCards(results: NoiseResult[], original: ImageData): ResultCard[] {
  return results.map((r) => ({
    label: r.label,
    dataUrl: imageDataToDataUrl(r.imageData),
    psnr: computePSNR(original, r.imageData),
    ssim: computeSSIM(original, r.imageData),
  }));
}

export default function Home() {
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [groups, setGroups] = useState<ResultGroup[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fileName, setFileName] = useState<string>("");

  const handleFile = useCallback(async (file: File) => {
    setIsProcessing(true);
    setGroups([]);
    setFileName(file.name);

    try {
      const original = await fileToImageData(file, 512);
      setOriginalUrl(imageDataToDataUrl(original));
      await yieldToBrowser();

      // 1) 노이즈 강도별 테스트
      const levelResults = testNoiseLevels(original, [2, 5, 10, 20, 40]);
      const levelCards = toCards(levelResults, original);
      setGroups((prev) => [
        ...prev,
        {
          title: "1. 노이즈 강도별 테스트",
          description: "같은 방식의 노이즈를 강도(sigma)만 다르게 준 결과입니다. 숫자가 클수록 노이즈가 강합니다.",
          cards: levelCards,
        },
      ]);
      await yieldToBrowser();

      // 2) 채널별 테스트
      const channelResults = testChannelNoise(original, 15);
      const channelCards = toCards(channelResults, original);
      setGroups((prev) => [
        ...prev,
        {
          title: "2. 채널별(R/G/B) 노이즈 테스트",
          description: "R, G, B 색상 채널 중 하나에만 노이즈(sigma=15)를 넣어본 결과입니다.",
          cards: channelCards,
        },
      ]);
      await yieldToBrowser();

      // 3) 고주파 대역 노이즈 테스트
      const freqResults = testFrequencyNoise(original, [200, 1000, 4000]);
      const freqCards = toCards(freqResults, original);
      setGroups((prev) => [
        ...prev,
        {
          title: "3. 고주파 대역 노이즈 테스트 (FFT)",
          description: "이미지를 주파수 영역으로 변환해 고주파(디테일) 영역에만 노이즈를 넣은 결과입니다. 실제 적대적 섭동 기법과 원리가 유사합니다.",
          cards: freqCards,
        },
      ]);
    } finally {
      setIsProcessing(false);
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
        <h1 className="text-xl font-semibold">딥페이크 방지 노이즈 실험 데모</h1>
        <p className="mt-1 text-sm text-neutral-400">
          사진을 올리면 여러 종류/강도의 노이즈를 적용한 결과를 바로 비교해볼 수 있습니다. 모든 계산은 서버 없이 브라우저에서 처리됩니다.
        </p>
        <a
          href="/vector-attack"
          className="mt-3 inline-block rounded-lg border border-neutral-700 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:border-neutral-500 hover:text-white"
        >
          → LowKey/Fawkes 원리(벡터 기반) 데모 보러 가기
        </a>
        <a
          href="/protection-lab"
          className="ml-2 mt-3 inline-block rounded-lg border border-emerald-800 bg-emerald-950/40 px-3 py-1.5 text-xs font-medium text-emerald-200 hover:border-emerald-600"
        >
          → 통합 보호 실험실 (3가지 비교)
        </a>
        <a
          href="/full-protect"
          className="ml-2 mt-3 inline-block rounded-lg border border-sky-800 bg-sky-950/40 px-3 py-1.5 text-xs font-medium text-sky-200 hover:border-sky-600"
        >
          → 앙상블 + PhotoGuard 실제 적용 (백엔드 연동)
        </a>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* 업로드 영역 */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="mb-8 flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-neutral-700 bg-neutral-900 px-6 py-10 text-center"
        >
          <p className="mb-3 text-sm text-neutral-400">
            사진을 이 영역에 끌어다 놓거나, 아래 버튼으로 선택하세요.
          </p>
          <label className="cursor-pointer rounded-lg bg-neutral-100 px-4 py-2 text-sm font-medium text-neutral-900 hover:bg-white">
            사진 선택
            <input type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
          </label>
          {fileName && <p className="mt-3 text-xs text-neutral-500">선택한 파일: {fileName}</p>}
        </div>

        {isProcessing && (
          <p className="mb-6 text-sm text-neutral-400">노이즈 계산 중입니다... (고주파 테스트는 몇 초 걸릴 수 있어요)</p>
        )}

        {originalUrl && (
          <div className="grid grid-cols-1 gap-8 lg:grid-cols-[320px_1fr]">
            {/* 왼쪽: 원본 사진 */}
            <div className="lg:sticky lg:top-8 lg:self-start">
              <h2 className="mb-2 text-sm font-semibold text-neutral-300">원본 사진</h2>
              <div className="overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={originalUrl} alt="원본 사진" className="w-full" />
              </div>
            </div>

            {/* 오른쪽: 결과물들 */}
            <div className="flex flex-col gap-10">
              {groups.map((group) => (
                <section key={group.title}>
                  <h2 className="text-sm font-semibold text-neutral-200">{group.title}</h2>
                  <p className="mb-3 mt-1 text-xs text-neutral-500">{group.description}</p>
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
                    {group.cards.map((card) => (
                      <div key={card.label} className="overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={card.dataUrl} alt={card.label} className="w-full" />
                        <div className="space-y-0.5 px-2 py-2 text-[11px] text-neutral-400">
                          <p className="truncate font-medium text-neutral-200">{card.label}</p>
                          <p>PSNR: {card.psnr === Infinity ? "∞" : card.psnr.toFixed(1)} dB</p>
                          <p>SSIM: {card.ssim.toFixed(3)}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
