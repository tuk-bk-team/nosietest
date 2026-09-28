"""
adversarial_face_protect.py
--------------------------------------------------------------------------
LowKey / Fawkes와 "같은 원리"로 동작하는 적대적 섭동(adversarial perturbation)
스크립트입니다.

기존에 만들었던 noise_test.py 는 "사람 눈에 안 보이면서 화질을 크게 안 깎는
잡음"을 무작위로 탐색하는 실험이었습니다 (얼굴 인식 모델을 전혀 쓰지 않음).

반면 이 스크립트는 실제 "얼굴 인식 모델"(FaceNet, InceptionResnetV1)을 안에
불러와서, 그 모델이 뽑아내는 얼굴 특징 벡터(embedding)가 원본 사진에서
최대한 멀어지도록 픽셀을 미세하게, 반복적으로(gradient 기반) 수정합니다.
LowKey/Fawkes 논문에서 말하는 "signed gradient ascent"를 그대로 구현한
것이라고 보시면 됩니다.

------------------------------------------------------------------------
전체 흐름 (LowKey 논문 3장, 우리가 번역했던 그 부분과 1:1로 대응됩니다)
------------------------------------------------------------------------
1) 얼굴 검출 & 정렬 (face detection & alignment)
   - MTCNN 모델로 사진 속 얼굴 위치를 찾고, 160x160 크기로 잘라 정렬합니다.
   - LowKey 논문에서 "A"라고 부르는 단계와 동일합니다.

2) 특징 벡터 추출 (feature extraction)
   - InceptionResnetV1 (VGGFace2 데이터셋으로 훈련된 FaceNet 계열 모델)에
     얼굴 이미지를 통과시켜 512차원 벡터를 얻습니다.
   - 논문에서 f_i(x) 라고 부르는 것이 바로 이 벡터입니다.

3) 적대적 섭동 계산 (adversarial perturbation / signed gradient ascent)
   - 목표: "원본 벡터와 최대한 멀어지는 픽셀 변화"를 찾는 것.
   - 방법: 모델에 대해 역전파(backpropagation)를 걸어서, 벡터 사이의 거리를
     늘리는 방향으로 픽셀의 gradient(기울기)를 계산하고, 그 부호(sign)만큼
     아주 조금씩 픽셀을 반복해서 이동시킵니다.
   - 사람 눈에 티가 나지 않도록, 픽셀 변화량은 epsilon 이라는 값 이내로
     제한합니다(L-infinity 제약). 논문은 이 자리에 LPIPS라는 지각적 손실
     함수를 추가로 사용하지만,  우리는 더 간단하고 표준적인 epsilon 제약(L-infinity
     ball)을 사용합니다 — 결과적으로 목표(눈에는 안 보이면서 벡터는 크게
     이동)는 동일하게 달성됩니다.
   - "가우시안 스무딩(Gaussian smoothing)"도 옵션으로구현했습니다. 
     gradient 에 가우시안 블러를 적용하면 섭동이 날카로운
     선이 아니라 부드러운 패턴이 되어 더 자연스러워 보입니다
     (원문 부록 8.6: sigma=3, window=7 을 그대로 사용)

4) 검증
   - 적대적 섭동 전/후로 벡터 사이의 거리(L2)와 코사인 유사도를 계산해서
     "실제로 다른 사람처럼 보이게 되었는지"를 확인합니다.
   - 원본과 보호된 이미지 사이의 PSNR/SSIM 도 계산해서 "사람 눈에는 얼마나
     비슷해 보이는지"도 함께 확인합니다.

------------------------------------------------------------------------
주의할 점 (연구 범위를 정확히 알기 위해 꼭 읽어주세요)
------------------------------------------------------------------------
- LowKey 논문은 4개의 서로 다른 모델(ResNet-50/152, IR-50/152 x ArcFace/
  CosFace)을 앙상블로 묶어서 공격합니다. 이렇게 해야 "본 적 없는" 상용 API
  (Amazon Rekognition 등)에도 전이(transfer)가 잘 됩니다. 이 스크립트는
  이해하기 쉽도록 모델 1개(FaceNet/VGGFace2)만 사용합니다. 즉, 이 자체로
  상용 얼굴 인식 API를 반드시 속인다는 보장은 없고, "같은 원리가 실제로
  작동하는지"를 확인하는 교육/실험용 프로토타입입니다.
- 상용 API에 대한 전이성을 높이려면, 서로 다른 아키텍처의 모델 여러 개를
  같은 방식으로 준비해서 앙상블로 묶는 확장이 다음 단계가 됩니다.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# ==========================================================================
# 0. 유틸리티: PSNR / SSIM (noise_test.py 에서 쓴 것과 동일한 방식)
#    -> "사람 눈에 얼마나 비슷해 보이는지"를 확인하기 위한 지표입니다.
# ==========================================================================

def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """두 이미지(0~255, uint8 또는 float) 사이의 PSNR(dB)을 계산합니다."""
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def compute_ssim_simple(img_a: np.ndarray, img_b: np.ndarray, block=8) -> float:
    """8x8 블록 단위의 간단한 SSIM 근사치를 계산합니다 (grayscale 기준)."""
    def to_gray(img):
        if img.ndim == 3:
            return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return img

    a = to_gray(img_a.astype(np.float64))
    b = to_gray(img_b.astype(np.float64))
    h, w = a.shape
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    total, count = 0.0, 0
    for by in range(0, h - block + 1, block):
        for bx in range(0, w - block + 1, block):
            pa = a[by:by + block, bx:bx + block]
            pb = b[by:by + block, bx:bx + block]
            mu_a, mu_b = pa.mean(), pb.mean()
            var_a, var_b = pa.var(), pb.var()
            cov = ((pa - mu_a) * (pb - mu_b)).mean()
            ssim = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / (
                (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
            )
            total += ssim
            count += 1
    return total / count if count else 1.0


# ==========================================================================
# 1. 가우시안 블러 커널 (논문 6.2절 "가우시안 스무딩" 구현용)
#    gradient 에 이 블러를 적용하면 섭동이 더 부드럽고 자연스러워집니다.
# ==========================================================================

def make_gaussian_kernel(window: int = 7, sigma: float = 3.0) -> torch.Tensor:
    """3채널(RGB)에 동일하게 적용할 수 있는 2D 가우시안 커널을 만듭니다."""
    ax = torch.arange(window, dtype=torch.float32) - (window - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    # (out_channels=3, in_channels/groups=1, k, k) 형태로 만들어 depthwise conv에 사용
    return kernel.view(1, 1, window, window).repeat(3, 1, 1, 1)


def smooth_gradient(grad: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """grad: (1, 3, H, W) 텐서에 depthwise 가우시안 블러를 적용합니다."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(grad, kernel, padding=pad, groups=3)


# 2. 핵심: 적대적 섭동 계산 (signed gradient ascent, = LowKey/Fawkes 핵심 원리)
def craft_adversarial_face(
    face_tensor: torch.Tensor, #이 텐서의 픽셀 값을 조금씩 바꿔가면서 "보호된 얼굴"을 만들어내는 함수
    model: torch.nn.Module, # 이 픽셀 변화가 벡터를 얼마나 움직이는지" 계산해주는 도구
    epsilon: float = 0.12, # 픽셀 변화량 제한 (0~1, 클수록 강한 보호 + 더 눈에 띔)
    step_size: float = 0.01, # 한 번 반복할 때 픽셀을 얼마나 이동시킬지 (0~1, 클수록 빠르게 이동하지만 눈에 띔)
    steps: int = 60, # 반복 횟수 (많을수록 벡터는 더 멀어지지만 눈에 띔)
    use_smoothing: bool = True, # gradient 에 가우시안 블러를 적용할지 여부 (True=부드러운 섭동, False=날카로운 섭동)
) -> torch.Tensor:
    """
    face_tensor: (1, 3, 160, 160), 값 범위 [-1, 1] (facenet-pytorch 표준 정규화)
    model: 사전 훈련된 얼굴 임베딩 모델 (InceptionResnetV1), eval + gradient 계산 가능 상태

    반환값: 원본과 시각적으로는 거의 같지만, 임베딩 벡터는 최대한 멀어진
            "보호된" 얼굴 이미지 텐서.
    """
    original = face_tensor.clone().detach()
    adv = face_tensor.clone().detach()

    with torch.no_grad():
        original_embedding = model(original)
        original_embedding = F.normalize(original_embedding, dim=1)

    gaussian_kernel = make_gaussian_kernel(window=7, sigma=3.0) if use_smoothing else None

    for step in range(steps):
        adv.requires_grad_(True)

        adv_embedding = model(adv)
        adv_embedding = F.normalize(adv_embedding, dim=1)

        # 논문 식(1)의 핵심: 두 벡터 사이의 거리를 "최대화"하려는 것이므로,
        # 코사인 유사도는 "최소화"하면 됩니다 (유사도가 낮을수록 벡터가 멀어짐).
        cosine_similarity = (original_embedding * adv_embedding).sum(dim=1)
        loss = cosine_similarity.sum()  # 이 값을 줄이는 방향으로 gradient 사용

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = adv.grad.detach()
        if gaussian_kernel is not None:
            grad = smooth_gradient(grad, gaussian_kernel)

        with torch.no_grad():
            # 부호 있는 경사 "하강"으로 코사인 유사도를 줄인다
            # = 벡터를 원본에서 최대한 멀리 밀어내는 것과 동일한 효과
            adv = adv - step_size * grad.sign()

            # epsilon 이내로 픽셀 변화량을 제한 (L-infinity 제약, 눈에 안 보이게)
            perturbation = torch.clamp(adv - original, min=-epsilon, max=epsilon)
            adv = torch.clamp(original + perturbation, min=-1.0, max=1.0)

        adv = adv.detach()

        if (step + 1) % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                cur_emb = F.normalize(model(adv), dim=1)
                cur_cos = (original_embedding * cur_emb).sum(dim=1).item()
            print(f"  step {step + 1:3d}/{steps} | 코사인 유사도: {cur_cos:.4f} (1.0=완전히 동일, 낮을수록 좋음)")

    return adv.detach()


# ==========================================================================
# 3. 텐서 <-> 이미지 변환 helper
# ==========================================================================

def tensor_to_uint8(face_tensor: torch.Tensor) -> np.ndarray:
    """facenet-pytorch의 [-1, 1] 텐서를 0~255 uint8 numpy 이미지로 변환합니다."""
    img = face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) / 2.0 * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def save_side_by_side(original: np.ndarray, adversarial: np.ndarray, out_path: str):
    """원본 | 보호된 이미지 | 차이(증폭) 를 나란히 붙여 저장합니다."""
    diff = np.abs(original.astype(np.int16) - adversarial.astype(np.int16))
    diff_amplified = np.clip(diff * 8, 0, 255).astype(np.uint8)  # 8배 증폭해야 육안으로 보임

    combined = np.concatenate([original, adversarial, diff_amplified], axis=1)
    Image.fromarray(combined).save(out_path)


# ==========================================================================
# 4. 메인 파이프라인
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="LowKey/Fawkes 원리의 적대적 섭동 데모")
    parser.add_argument("--image", type=str, default=None, help="입력 이미지 경로 (얼굴이 포함된 사진)")
    parser.add_argument("--epsilon", type=float, default=0.03, help="섭동 크기 제한 (0~1, 클수록 강한 보호 + 더 눈에 띔). 0.03 근처가 '눈에 안 보이면서 효과적인' 균형점입니다.")
    parser.add_argument("--steps", type=int, default=50, help="반복 횟수")
    parser.add_argument("--no-smoothing", action="store_true", help="가우시안 스무딩 끄기")
    parser.add_argument("--outdir", type=str, default="results", help="결과 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- 입력 이미지 준비 ---
    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"입력 이미지: {args.image}")
    else:
        # 이미지가 지정되지 않으면 scikit-image에 내장된 표준 테스트 사진을 사용합니다
        # (네트워크 다운로드 없이 로컬에 내장되어 있는 공개 테스트용 이미지입니다).
        from skimage import data
        img = Image.fromarray(data.astronaut())
        print("입력 이미지가 지정되지 않아 scikit-image의 기본 테스트 이미지를 사용합니다.")
        print("본인 사진으로 테스트하려면: python3 adversarial_face_protect.py --image 내사진.jpg")

    # --- 1) 얼굴 검출 & 정렬 (MTCNN) ---
    print("\n[1/4] 얼굴 검출 및 정렬 중...")
    mtcnn = MTCNN(image_size=160, margin=20, post_process=True)
    face_tensor, prob = mtcnn(img, return_prob=True)
    if face_tensor is None:
        raise RuntimeError("이미지에서 얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 사진으로 다시 시도해주세요.")
    print(f"  얼굴 검출 성공 (신뢰도: {prob:.3f})")
    face_tensor = face_tensor.unsqueeze(0)  # (1, 3, 160, 160)

    # --- 2) 얼굴 임베딩 모델 로드 ---
    print("\n[2/4] 얼굴 임베딩 모델(FaceNet, VGGFace2) 로드 중...")
    model = InceptionResnetV1(pretrained="vggface2").eval()
    for p in model.parameters():
        p.requires_grad_(False)  # 모델 가중치는 고정, 이미지 픽셀만 최적화 대상

    # --- 3) 적대적 섭동 계산 ---
    print(f"\n[3/4] 적대적 섭동 계산 중 (epsilon={args.epsilon}, steps={args.steps}, "
          f"가우시안 스무딩={'끔' if args.no_smoothing else '켬'})...")
    adv_face_tensor = craft_adversarial_face(
        face_tensor,
        model,
        epsilon=args.epsilon,
        step_size=max(args.epsilon / 12, 0.005),
        steps=args.steps,
        use_smoothing=not args.no_smoothing,
    )

    # --- 4) 검증: 벡터 거리 + 이미지 품질(PSNR/SSIM) ---
    print("\n[4/4] 결과 검증 중...")
    with torch.no_grad():
        orig_emb = F.normalize(model(face_tensor), dim=1)
        adv_emb = F.normalize(model(adv_face_tensor), dim=1)
        cosine_sim = (orig_emb * adv_emb).sum(dim=1).item()
        l2_dist = torch.norm(orig_emb - adv_emb, p=2).item()

    original_img = tensor_to_uint8(face_tensor)
    adversarial_img = tensor_to_uint8(adv_face_tensor)
    psnr = compute_psnr(original_img, adversarial_img)
    ssim = compute_ssim_simple(original_img, adversarial_img)

    out_path = os.path.join(args.outdir, "adversarial_comparison.png")
    save_side_by_side(original_img, adversarial_img, out_path)

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"임베딩 코사인 유사도 : {cosine_sim:.4f}  (1.0=완전 동일 인물, 0에 가까울수록 다른 사람)")
    print(f"임베딩 L2 거리       : {l2_dist:.4f}  (클수록 서로 다른 벡터)")
    print(f"이미지 PSNR          : {psnr:.2f} dB  (40dB 이상이면 육안으로 거의 구분 안 됨)")
    print(f"이미지 SSIM          : {ssim:.4f}  (1.0에 가까울수록 구조적으로 동일)")
    print("-" * 60)
    # FaceNet(VGGFace2) 계열에서 흔히 쓰이는 대략적인 판단 기준입니다.
    # (정확한 임계값은 사용하는 시스템/데이터셋마다 다를 수 있어 참고용입니다.)
    if cosine_sim < 0.4:
        print("=> 코사인 유사도가 충분히 낮아, 이 벡터만 보면 '다른 사람'으로 판단될 가능성이 높습니다.")
    else:
        print("=> 아직 '동일 인물'로 판단될 가능성이 있습니다. --epsilon 이나 --steps 값을 높여보세요.")
    print(f"\n비교 이미지 저장됨: {out_path}")
    print("  (왼쪽: 원본 얼굴 / 가운데: 보호된 얼굴 / 오른쪽: 차이를 8배 증폭한 이미지)")


if __name__ == "__main__":
    main()
