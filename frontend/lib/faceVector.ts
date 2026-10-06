/**
 * faceVector.ts
 * --------------------------------------------------------------------------
 * LowKey/Fawkes 원리(적대적 섭동으로 얼굴 특징 벡터를 밀어내기)를
 * 브라우저 안에서(TensorFlow.js + face-api.js) 그대로 재현한 코드입니다.
 *
 * 백엔드에서 만들었던 adversarial_face_protect.py(PyTorch + facenet-pytorch)와
 * 원리는 100% 동일하고, 신경망만 다릅니다.
 *   - 파이썬 버전: FaceNet(InceptionResnetV1, VGGFace2로 훈련, 512차원 벡터)
 *   - 이 버전(브라우저): face-api.js의 FaceRecognitionNet
 *     (dlib 계열 ResNet, 128차원 벡터) — 서버 없이 브라우저에서 바로 돌리기
 *     위해 선택했습니다.
 *
 * 핵심 아이디어는 똑같습니다.
 *   1) 얼굴을 찾아서 잘라낸다 (얼굴 검출)
 *   2) 신경망에 통과시켜 벡터(임베딩)를 얻는다
 *   3) 벡터를 얻는 신경망에 역전파를 걸어서, "벡터가 멀어지는 방향"으로
 *      픽셀을 아주 조금씩(epsilon 이내로) 반복해서 이동시킨다 (signed gradient
 *      descent/ascent — 이 파일에서는 코사인 유사도를 '최소화'하는 방향으로
 *      내려갑니다)
 *   4) 결과: 사람 눈에는 원본과 거의 같지만, 벡터는 많이 달라진 이미지
 */

"use client";

import * as faceapi from "face-api.js";

// face-api.js가 내부적으로 쓰는 tfjs-core를 그대로 재사용합니다.
// (버전 충돌을 피하기 위해 별도로 @tensorflow/tfjs-core를 import하지 않습니다)
const tf = faceapi.tf;

// ==========================================================================
// 0. 모델 로딩
// ==========================================================================

let modelsLoadedPromise: Promise<void> | null = null;

/** 얼굴 검출 + 얼굴 임베딩(특징 벡터) 모델을 public/models 에서 불러옵니다. */
export function loadFaceModels(): Promise<void> {
  if (!modelsLoadedPromise) {
    modelsLoadedPromise = (async () => {
      // 일반적인 사용자 브라우저에서는 webgl(GPU 가속)이 훨씬 빠르지만,
      // webgl을 쓸 수 없는 환경(오래된 브라우저, 일부 헤드리스 환경 등)도 있어서
      // 실패하면 자동으로 cpu 백엔드로 넘어가도록 명시적으로 처리합니다.
      try {
        await tf.setBackend("webgl");
        await tf.ready();
      } catch {
        console.warn("webgl 백엔드를 사용할 수 없어 cpu 백엔드로 전환합니다.");
        await tf.setBackend("cpu");
        await tf.ready();
      }
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
      ]);
    })();
  }
  return modelsLoadedPromise;
}

// ==========================================================================
// 1. 얼굴 검출 & 크롭 (텐서로 변환)
// ==========================================================================

export type FaceBox = { x: number; y: number; width: number; height: number };

/** 이미지에서 얼굴 위치(바운딩 박스)를 찾습니다. */
export async function detectFaceBox(img: HTMLImageElement): Promise<FaceBox> {
  const detection = await faceapi.detectSingleFace(
    img,
    new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.4 })
  );
  if (!detection) {
    throw new Error(
      "이미지에서 얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 정면 사진으로 다시 시도해주세요."
    );
  }
  const { x, y, width, height } = detection.box;
  return { x, y, width, height };
}

/**
 * 얼굴 박스 주변을 여유 있게(margin) 잘라내어 float32 텐서(0~255 범위, [H,W,3])로 반환합니다.
 * 원본 해상도를 그대로 유지합니다 (모델에 넣을 때 필요한 150x150 리사이즈는
 * face-api.js의 forwardInput 내부에서 자동으로, 미분 가능한 방식으로 처리됩니다).
 */
export function cropFaceToTensor(
  img: HTMLImageElement,
  box: FaceBox,
  margin = 0.35
): { tensor: import("@tensorflow/tfjs-core").Tensor3D; cropBox: FaceBox } {
  return tf.tidy(() => {
    const full = tf.browser.fromPixels(img).toFloat(); // [H, W, 3], 0~255
    const imgH = full.shape[0];
    const imgW = full.shape[1];

    const mx = box.width * margin;
    const my = box.height * margin;
    const x0 = Math.max(0, Math.round(box.x - mx));
    const y0 = Math.max(0, Math.round(box.y - my));
    const x1 = Math.min(imgW, Math.round(box.x + box.width + mx));
    const y1 = Math.min(imgH, Math.round(box.y + box.height + my));
    const w = Math.max(1, x1 - x0);
    const h = Math.max(1, y1 - y0);

    const cropped = tf.slice(full, [y0, x0, 0], [h, w, 3]);
    return { tensor: cropped, cropBox: { x: x0, y: y0, width: w, height: h } };
  });
}

// ==========================================================================
// 2. 특징 벡터(임베딩) 계산
// ==========================================================================

function computeDescriptor(x: import("@tensorflow/tfjs-core").Tensor3D) {
  // NetInput은 tf.Tensor를 그대로 받아 (미분 가능하게) 150x150으로
  // 패딩+리사이즈한 뒤 신경망에 통과시킵니다.
  const netInput = new faceapi.NetInput([x]);
  return faceapi.nets.faceRecognitionNet.forwardInput(netInput); // [1, 128]
}

function cosineSimilarityToOriginal(
  x: import("@tensorflow/tfjs-core").Tensor3D,
  originalDescriptor: import("@tensorflow/tfjs-core").Tensor2D
) {
  return tf.tidy(() => {
    const desc = computeDescriptor(x);
    const descNorm = desc.div(desc.norm("euclidean", 1, true).add(1e-8));
    const origNorm = originalDescriptor.div(
      originalDescriptor.norm("euclidean", 1, true).add(1e-8)
    );
    return tf.sum(descNorm.mul(origNorm)) as import("@tensorflow/tfjs-core").Scalar;
  });
}

// ==========================================================================
// 3. 가우시안 스무딩 (LowKey 논문 6.2절과 동일한 아이디어)
//    gradient에 블러를 적용해 섭동을 더 부드럽고 자연스럽게 만듭니다.
// ==========================================================================

function makeGaussianKernel(size = 7, sigma = 3) {
  return tf.tidy(() => {
    const ax = tf.range(0, size).sub(tf.scalar((size - 1) / 2));
    const xx = ax.reshape([size, 1]).tile([1, size]);
    const yy = ax.reshape([1, size]).tile([size, 1]);
    const kernel2d = tf.exp(
      xx.square().add(yy.square()).neg().div(2 * sigma * sigma)
    );
    const normalized = kernel2d.div(kernel2d.sum());
    // depthwiseConv2d 필터 형태: [h, w, inChannels(3), channelMultiplier(1)]
    return normalized.reshape([size, size, 1, 1]).tile([1, 1, 3, 1]);
  });
}

function smoothGradient(
  grad: import("@tensorflow/tfjs-core").Tensor3D,
  kernel: import("@tensorflow/tfjs-core").Tensor4D
) {
  return tf.tidy(() => {
    const batched = grad.expandDims(0) as import("@tensorflow/tfjs-core").Tensor4D;
    const smoothed = tf.depthwiseConv2d(batched, kernel, 1, "same");
    return smoothed.squeeze([0]) as import("@tensorflow/tfjs-core").Tensor3D;
  });
}

// ==========================================================================
// 4. 핵심: 적대적 섭동 계산 (signed gradient descent)
// ==========================================================================

export type AttackParams = {
  /** 픽셀 변화량 한도 (0~1 스케일, 실제 픽셀 0~255에 곱해서 사용). 클수록 강하지만 눈에 띔 */
  epsilon: number;
  /** 한 스텝에서 픽셀을 얼마나 움직일지 (0~1 스케일) */
  stepSize: number;
  /** 반복 횟수 */
  steps: number;
  /** 가우시안 스무딩 사용 여부 */
  smoothing: boolean;
};

export type AttackResult = {
  tensor: import("@tensorflow/tfjs-core").Tensor3D;
  cosineSimilarityBefore: number;
  cosineSimilarityAfter: number;
};

/**
 * 원본 얼굴 텐서로부터, 벡터가 최대한 멀어지도록 만든 "보호된" 얼굴 텐서를 계산합니다.
 * (== adversarial_face_protect.py 의 craft_adversarial_face 와 동일한 역할)
 */
export async function craftAdversarialFace(
  originalTensor: import("@tensorflow/tfjs-core").Tensor3D,
  params: AttackParams,
  onProgress?: (step: number, totalSteps: number, cosineSim: number) => void
): Promise<AttackResult> {
  const epsilonPx = params.epsilon * 255;
  const stepPx = Math.max(params.stepSize * 255, 0.1);

  const original = tf.keep(originalTensor.clone()) as import("@tensorflow/tfjs-core").Tensor3D;
  const originalDescriptor = tf.keep(
    tf.tidy(() => computeDescriptor(original))
  ) as import("@tensorflow/tfjs-core").Tensor2D;

  const kernel = params.smoothing ? tf.keep(makeGaussianKernel(7, 3)) : null;

  let x = tf.keep(original.clone()) as import("@tensorflow/tfjs-core").Tensor3D;

  const valueAndGradFn = tf.valueAndGrad((t: import("@tensorflow/tfjs-core").Tensor3D) =>
    cosineSimilarityToOriginal(t, originalDescriptor)
  );

  let cosineBefore = 1;
  let cosineAfter = 1;
  const steps = Math.max(1, Math.round(params.steps));

  for (let step = 0; step < steps; step++) {
    const { value, grad } = valueAndGradFn(x);
    let g = grad as import("@tensorflow/tfjs-core").Tensor3D;

    if (kernel) {
      const smoothed = smoothGradient(g, kernel as import("@tensorflow/tfjs-core").Tensor4D);
      g.dispose();
      g = smoothed;
    }

    const nextX = tf.tidy(() => {
      const updated = x.sub(tf.sign(g).mul(stepPx));
      const perturbation = updated.sub(original).clipByValue(-epsilonPx, epsilonPx);
      return original.add(perturbation).clipByValue(0, 255) as import("@tensorflow/tfjs-core").Tensor3D;
    });

    x.dispose();
    g.dispose();
    x = tf.keep(nextX) as import("@tensorflow/tfjs-core").Tensor3D;

    const cosVal = (await value.data())[0];
    value.dispose();
    if (step === 0) cosineBefore = cosVal;
    cosineAfter = cosVal;
    onProgress?.(step + 1, steps, cosVal);

    // 브라우저가 멈추지 않도록 가끔 한 프레임 양보
    if (step % 5 === 0) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  original.dispose();
  originalDescriptor.dispose();
  kernel?.dispose();

  return { tensor: x, cosineSimilarityBefore: cosineBefore, cosineSimilarityAfter: cosineAfter };
}

// ==========================================================================
// 5. 텐서 <-> 이미지 변환 유틸리티
// ==========================================================================

/** float32 [0,255] 텐서를 캔버스로 그립니다. */
export async function tensorToCanvas(
  tensor: import("@tensorflow/tfjs-core").Tensor3D
): Promise<HTMLCanvasElement> {
  const canvas = document.createElement("canvas");
  canvas.width = tensor.shape[1];
  canvas.height = tensor.shape[0];
  const intTensor = tf.tidy(
    () => tensor.round().clipByValue(0, 255).cast("int32") as import("@tensorflow/tfjs-core").Tensor3D
  );
  await tf.browser.toPixels(intTensor, canvas);
  intTensor.dispose();
  return canvas;
}

export function canvasToImageData(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext("2d")!;
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/** 보호된 얼굴 크롭을, 원본 전체 사진의 원래 위치에 다시 합성합니다. */
export function compositeCropIntoImage(
  fullImg: HTMLImageElement,
  cropBox: FaceBox,
  cropCanvas: HTMLCanvasElement
): string {
  const canvas = document.createElement("canvas");
  canvas.width = fullImg.naturalWidth;
  canvas.height = fullImg.naturalHeight;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(fullImg, 0, 0);
  ctx.drawImage(cropCanvas, cropBox.x, cropBox.y, cropBox.width, cropBox.height);
  return canvas.toDataURL("image/png");
}

// ==========================================================================
// 6. 이미지 품질 지표 (noise_test.py / metrics.ts와 동일한 방식)
// ==========================================================================

export function computePSNR(a: ImageData, b: ImageData): number {
  let sumSquaredError = 0;
  let count = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const diff = a.data[i + c] - b.data[i + c];
      sumSquaredError += diff * diff;
      count++;
    }
  }
  const mse = sumSquaredError / count;
  if (mse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}
