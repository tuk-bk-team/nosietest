/**
 * noise.ts
 * --------------------------------------------------------
 * 파이썬 스크립트(noise_test.py)에서 했던 3가지 실험을
 * 브라우저에서 그대로 재현하는 함수들입니다.
 *   1) 노이즈 강도별 테스트 (가우시안 노이즈)
 *   2) 채널별(R/G/B) 노이즈 테스트
 *   3) 고주파 대역 노이즈 테스트 (FFT 기반)
 *
 * 모든 계산은 서버 없이 "브라우저 안에서" 끝나기 때문에,
 * Vercel에 정적 사이트로 배포해도 그대로 동작합니다.
 */

import { fft2d, nextPowerOfTwo } from "./fft";

export type NoiseResult = {
  label: string; // 화면에 표시할 이름 (예: "sigma=10")
  imageData: ImageData;
};

// ------------------------------------------------------------
// 이미지 파일을 브라우저에서 다루기 쉬운 형태(ImageData)로 변환
// ------------------------------------------------------------

/**
 * 업로드된 File을 ImageData로 변환합니다.
 * maxSize보다 큰 사진은 비율을 유지한 채 줄여서, 브라우저에서도
 * 빠르게 계산될 수 있도록 합니다. (원본 그대로면 FFT 계산이 느려질 수 있음)
 */
export async function fileToImageData(file: File, maxSize = 512): Promise<ImageData> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = reject;
    el.src = dataUrl;
  });

  let { width, height } = img;
  if (width > maxSize || height > maxSize) {
    const scale = maxSize / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}

export function imageDataToDataUrl(imageData: ImageData): string {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function cloneImageData(src: ImageData): ImageData {
  return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
}

// 표준정규분포(가우시안) 난수를 생성합니다 (Box-Muller 변환).
function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ------------------------------------------------------------
// 1) 노이즈 강도별 테스트
// ------------------------------------------------------------

/**
 * 이미지 전체에 가우시안 노이즈를 더합니다.
 * sigma가 클수록 노이즈가 강해집니다. (알파 채널은 그대로 둡니다)
 */
export function addGaussianNoise(src: ImageData, sigma: number): ImageData {
  const out = cloneImageData(src);
  const data = out.data;
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const noisy = data[i + c] + gaussianRandom() * sigma;
      data[i + c] = Math.min(255, Math.max(0, noisy));
    }
  }
  return out;
}

export function testNoiseLevels(src: ImageData, sigmas: number[]): NoiseResult[] {
  return sigmas.map((sigma) => ({
    label: `노이즈 강도 sigma=${sigma}`,
    imageData: addGaussianNoise(src, sigma),
  }));
}

// ------------------------------------------------------------
// 2) 채널별(R/G/B) 노이즈 테스트
// ------------------------------------------------------------

export function addChannelNoise(src: ImageData, channel: 0 | 1 | 2, sigma: number): ImageData {
  const out = cloneImageData(src);
  const data = out.data;
  for (let i = 0; i < data.length; i += 4) {
    const noisy = data[i + channel] + gaussianRandom() * sigma;
    data[i + channel] = Math.min(255, Math.max(0, noisy));
  }
  return out;
}

export function testChannelNoise(src: ImageData, sigma: number): NoiseResult[] {
  const names = ["R(빨강)", "G(초록)", "B(파랑)"] as const;
  return ([0, 1, 2] as const).map((ch) => ({
    label: `${names[ch]} 채널만 sigma=${sigma}`,
    imageData: addChannelNoise(src, ch, sigma),
  }));
}

// ------------------------------------------------------------
// 3) 고주파 대역 노이즈 테스트 (FFT 기반)
// ------------------------------------------------------------

/**
 * 이미지를 주파수 영역(FFT)으로 변환한 뒤, "저주파(전체적인 형태/색)"는
 * 그대로 두고 "고주파(경계선, 디테일)" 영역에만 노이즈를 추가합니다.
 * 사람 눈은 고주파 변화에 둔감해서, 실제 적대적 섭동 기법들이 이런 원리를
 * 활용하는 경우가 많습니다. (원리는 noise_test.py의 3번 테스트와 동일)
 *
 * lowFreqRadius: 저주파로 취급할 범위(값이 클수록 저주파 영역이 넓어짐)
 */
export function addHighFrequencyNoise(src: ImageData, sigma: number, lowFreqRadius = 20): ImageData {
  const { width, height, data } = src;

  // FFT는 2의 거듭제곱 크기에서만 동작하므로, 가장자리 픽셀을 복제해서
  // 2의 거듭제곱 크기로 패딩합니다.
  const padW = nextPowerOfTwo(width);
  const padH = nextPowerOfTwo(height);

  const out = cloneImageData(src);

  for (let channel = 0; channel < 3; channel++) {
    // --- 패딩된 실수부 배열 준비 ---
    const re: Float64Array[] = [];
    const im: Float64Array[] = [];
    for (let y = 0; y < padH; y++) {
      re.push(new Float64Array(padW));
      im.push(new Float64Array(padW));
    }
    for (let y = 0; y < padH; y++) {
      const sy = Math.min(y, height - 1);
      for (let x = 0; x < padW; x++) {
        const sx = Math.min(x, width - 1);
        re[y][x] = data[(sy * width + sx) * 4 + channel];
      }
    }

    // --- 정방향 FFT ---
    fft2d(re, im, false);

    // --- 고주파 영역에만 노이즈 추가 ---
    // 주파수 0(저주파)은 배열의 (0,0) 모서리와 그 반대편 끝에 걸쳐 있습니다
    // (fftshift를 안 했을 때의 배치). 그래서 "모서리에서의 거리"로 저주파/고주파를 구분합니다.
    for (let y = 0; y < padH; y++) {
      const dy = Math.min(y, padH - y);
      for (let x = 0; x < padW; x++) {
        const dx = Math.min(x, padW - x);
        const isLowFreq = dy < lowFreqRadius && dx < lowFreqRadius;
        if (!isLowFreq) {
          re[y][x] += (Math.random() * 2 - 1) * sigma;
          im[y][x] += (Math.random() * 2 - 1) * sigma;
        }
      }
    }

    // --- 역방향 FFT (다시 이미지로 복원) ---
    fft2d(re, im, true);

    // --- 원래 크기로 잘라서 결과에 반영 ---
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const value = Math.min(255, Math.max(0, re[y][x]));
        out.data[(y * width + x) * 4 + channel] = value;
      }
    }
  }

  return out;
}

export function testFrequencyNoise(src: ImageData, sigmas: number[]): NoiseResult[] {
  return sigmas.map((sigma) => ({
    label: `고주파 노이즈 sigma=${sigma}`,
    imageData: addHighFrequencyNoise(src, sigma),
  }));
}
