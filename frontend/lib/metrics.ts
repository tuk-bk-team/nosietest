/**
 * metrics.ts
 * --------------------------------------------------------
 * 원본 사진과 노이즈 낀 사진이 "사람 눈에 얼마나 비슷해 보이는지"를
 * 수치로 계산합니다. noise_test.py에서 쓴 PSNR/SSIM과 같은 지표입니다.
 *
 * - PSNR: 높을수록 원본과 비슷함 (노이즈가 안 보임). 보통 40dB 이상이면
 *         육안으로 거의 구분이 안 된다고 봅니다.
 * - SSIM: 1에 가까울수록 구조적으로 원본과 비슷함. 0.95 이상이면 꽤 비슷한 편.
 *
 * 참고: 여기 SSIM은 8x8 블록 단위로 계산하는 단순화된 버전입니다.
 * (scikit-image의 정확한 구현과 소수점 단위까지 똑같지는 않지만,
 *  "더 비슷하다/덜 비슷하다"를 비교하는 용도로는 충분합니다)
 */

export function computePSNR(a: ImageData, b: ImageData): number {
  const da = a.data;
  const db = b.data;
  let sumSquaredError = 0;
  let count = 0;

  for (let i = 0; i < da.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const diff = da[i + c] - db[i + c];
      sumSquaredError += diff * diff;
      count++;
    }
  }

  const mse = sumSquaredError / count;
  if (mse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}

export type PixelDeltaMetrics = {
  meanAbsoluteError: number;
  maxAbsoluteError: number;
  changedPixelPercent: number;
};

/** Pixel-level change statistics; these describe image difference, not protection efficacy. */
export function computePixelDelta(a: ImageData, b: ImageData): PixelDeltaMetrics {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error("비교할 두 이미지의 크기가 같아야 합니다.");
  }

  let absoluteSum = 0;
  let maxAbsoluteError = 0;
  let changedPixels = 0;
  const pixelCount = a.width * a.height;
  for (let i = 0; i < a.data.length; i += 4) {
    let pixelChanged = false;
    for (let c = 0; c < 3; c++) {
      const difference = Math.abs(a.data[i + c] - b.data[i + c]);
      absoluteSum += difference;
      maxAbsoluteError = Math.max(maxAbsoluteError, difference);
      pixelChanged ||= difference > 0;
    }
    if (pixelChanged) changedPixels++;
  }

  return {
    meanAbsoluteError: absoluteSum / (pixelCount * 3),
    maxAbsoluteError,
    changedPixelPercent: pixelCount ? (changedPixels / pixelCount) * 100 : 0,
  };
}

function toGrayscale(img: ImageData): Float64Array {
  const { width, height, data } = img;
  const gray = new Float64Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    // 사람 눈이 초록색에 더 민감한 것을 반영한 표준 가중치
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return gray;
}

export function computeSSIM(a: ImageData, b: ImageData): number {
  const width = a.width;
  const height = a.height;
  const grayA = toGrayscale(a);
  const grayB = toGrayscale(b);

  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const blockSize = 8;

  let totalSSIM = 0;
  let blockCount = 0;

  for (let by = 0; by + blockSize <= height; by += blockSize) {
    for (let bx = 0; bx + blockSize <= width; bx += blockSize) {
      let meanA = 0;
      let meanB = 0;
      const n = blockSize * blockSize;

      for (let y = 0; y < blockSize; y++) {
        for (let x = 0; x < blockSize; x++) {
          const idx = (by + y) * width + (bx + x);
          meanA += grayA[idx];
          meanB += grayB[idx];
        }
      }
      meanA /= n;
      meanB /= n;

      let varA = 0;
      let varB = 0;
      let covAB = 0;
      for (let y = 0; y < blockSize; y++) {
        for (let x = 0; x < blockSize; x++) {
          const idx = (by + y) * width + (bx + x);
          const da = grayA[idx] - meanA;
          const db = grayB[idx] - meanB;
          varA += da * da;
          varB += db * db;
          covAB += da * db;
        }
      }
      varA /= n - 1;
      varB /= n - 1;
      covAB /= n - 1;

      const numerator = (2 * meanA * meanB + C1) * (2 * covAB + C2);
      const denominator = (meanA * meanA + meanB * meanB + C1) * (varA + varB + C2);
      totalSSIM += numerator / denominator;
      blockCount++;
    }
  }

  return blockCount > 0 ? totalSSIM / blockCount : 1;
}
