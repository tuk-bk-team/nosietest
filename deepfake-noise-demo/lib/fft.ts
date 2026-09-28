/**
 * fft.ts
 * --------------------------------------------------------
 * 아주 작은 2D FFT(고속 푸리에 변환) 구현체입니다.
 * 파이썬 스크립트(noise_test.py)의 numpy.fft.fft2와 같은 역할을
 * 브라우저(자바스크립트)에서 하기 위해 직접 구현했습니다.
 *
 * 주의: 가로/세로 크기가 반드시 "2의 거듭제곱"(256, 512, 1024...)
 * 이어야 동작합니다. 그래서 noise.ts 쪽에서 이미지를 처리하기 전에
 * 2의 거듭제곱 크기로 맞춰(padding) 줍니다.
 */

// 1차원 FFT (한 줄에 대한 변환). re/im 배열을 직접 수정합니다(in-place).
// invert=false 면 정방향 변환, true면 역방향 변환입니다.
export function fft1d(re: Float64Array, im: Float64Array, invert: boolean) {
  const n = re.length;

  // 1) 비트 반전(bit-reversal) 순서로 재배열 - FFT 알고리즘의 필수 전처리 단계
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; (j & bit) !== 0; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tRe = re[i]; re[i] = re[j]; re[j] = tRe;
      const tIm = im[i]; im[i] = im[j]; im[j] = tIm;
    }
  }

  // 2) 버터플라이 연산 반복 (Cooley-Tukey 알고리즘)
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (invert ? -1 : 1);
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;

        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;

        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }

  // 3) 역변환일 때는 크기(n)로 나눠서 정규화 (numpy의 ifft와 동일한 방식)
  if (invert) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

// 2차원 FFT: 모든 행에 대해 1D FFT를 하고, 그 다음 모든 열에 대해 1D FFT를 합니다.
export function fft2d(re: Float64Array[], im: Float64Array[], invert: boolean) {
  const h = re.length;
  const w = re[0].length;

  // 행(가로) 방향 변환
  for (let y = 0; y < h; y++) {
    fft1d(re[y], im[y], invert);
  }

  // 열(세로) 방향 변환
  const colRe = new Float64Array(h);
  const colIm = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      colRe[y] = re[y][x];
      colIm[y] = im[y][x];
    }
    fft1d(colRe, colIm, invert);
    for (let y = 0; y < h; y++) {
      re[y][x] = colRe[y];
      im[y][x] = colIm[y];
    }
  }
}

// 2의 거듭제곱으로 올림한 값을 반환 (예: 300 -> 512)
export function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}
