"""
app/attacks/recognition_ensemble.py
--------------------------------------------------------------------------
LowKey/Fawkes 원리의 "앙상블 공격" — 얼굴 인식 모델 여러 개를 동시에 공격.

기존 adversarial_face_protect.py의 craft_adversarial_face()와 원리는
완전히 같습니다 (signed gradient descent로 코사인 유사도를 낮추는 것).
달라진 점은 모델이 1개가 아니라 "리스트"라는 것 뿐입니다 - 매 스텝마다
모델 하나하나에 대해 손실(loss)을 구하고, 그걸 전부 더해서 하나의 총
손실로 만든 다음, 그 총 손실 기준으로 이미지(얼굴)를 한 번만 업데이트합니다.

이 파일은 두 가지 용도로 쓸 수 있게 나눠져 있습니다:
  1) ensemble_cosine_loss()  - "이번 스텝의 손실값"만 계산하는 함수.
     PhotoGuard 손실과 합쳐서 하나의 결합 섭동을 만들 때 (attacks/combined.py)
     이 함수만 가져다 씁니다.
  2) craft_adversarial_face_ensemble() - 위 손실 함수를 이용해 처음부터
     끝까지 공격을 완료하는 "독립 실행형" 함수. 앙상블 공격만 단독으로
     테스트하고 싶을 때 씁니다 (예: scripts/run_cli.py --mode recognition).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from app.models.face_models import FaceModel


def make_gaussian_kernel(window: int = 7, sigma: float = 3.0) -> torch.Tensor:
    """3채널(RGB)에 동일하게 적용할 수 있는 2D 가우시안 커널을 만듭니다.
    (LowKey 논문의 가우시안 스무딩용 - 이미지 자체를 블러 처리하는 데 씁니다.)
    """
    ax = torch.arange(window, dtype=torch.float32) - (window - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window, window).repeat(3, 1, 1, 1)


def gaussian_blur_image(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """img: (1, 3, H, W) 텐서를 블러 처리합니다 (미분 가능한 conv2d 연산)."""
    pad = kernel.shape[-1] // 2
    return F.conv2d(img, kernel.to(img.device), padding=pad, groups=3)


@torch.no_grad()
def compute_original_embeddings(
    face_tensor: torch.Tensor, face_models: List[FaceModel]
) -> List[torch.Tensor]:
    """공격 시작 전, 원본 얼굴에 대한 "각 모델의" 임베딩을 한 번씩 미리 구해둡니다."""
    embeddings = []
    for fm in face_models:
        emb = F.normalize(fm.model(face_tensor), dim=1)
        embeddings.append(emb)
    return embeddings


def ensemble_cosine_loss(
    face_tensor: torch.Tensor,
    original_embeddings: List[torch.Tensor],
    face_models: List[FaceModel],
    gaussian_kernel: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    "이번 스텝"의 앙상블 손실값 1개를 계산합니다 (여러 모델의 코사인 유사도 합).

    face_tensor: 지금 시점의 섭동이 적용된 얼굴 텐서 (1,3,160,160), requires_grad=True 상태여야 함
    original_embeddings: compute_original_embeddings()로 미리 구해둔 "원본" 임베딩들
    face_models: 공격할 모델 리스트
    gaussian_kernel: None이 아니면, 각 모델마다 "원본 섭동 이미지"와 "블러 버전"
        두 가지의 코사인 유사도를 더합니다 (LowKey 논문 방식 - 블러를 당해도
        공격 효과가 유지되도록 하는 견고성 기법).

    반환값: (총 손실 텐서, {모델이름: 현재 코사인 유사도} 딕셔너리 - 로그/진행상황 표시용)
    """
    total_loss = 0.0
    per_model_cos = {}

    face_blurred = gaussian_blur_image(face_tensor, gaussian_kernel) if gaussian_kernel is not None else None

    for fm, orig_emb in zip(face_models, original_embeddings):
        adv_emb = F.normalize(fm.model(face_tensor), dim=1)
        cos_sharp = (orig_emb * adv_emb).sum(dim=1)
        per_model_cos[fm.name] = cos_sharp.item()

        if face_blurred is not None:
            adv_emb_blur = F.normalize(fm.model(face_blurred), dim=1)
            cos_blur = (orig_emb * adv_emb_blur).sum(dim=1)
            total_loss = total_loss + cos_sharp.sum() + cos_blur.sum()
        else:
            total_loss = total_loss + cos_sharp.sum()

    return total_loss, per_model_cos


def craft_adversarial_face_ensemble(
    face_tensor: torch.Tensor,
    face_models: List[FaceModel],
    epsilon: float = 0.03,
    step_size: float = 0.0025,
    steps: int = 50,
    use_smoothing: bool = True,
) -> torch.Tensor:
    """
    여러 얼굴 인식 모델을 동시에 공격하는 독립 실행형 함수
    (adversarial_face_protect.py의 craft_adversarial_face와 사용법 동일,
    model 하나 대신 face_models 리스트를 받는다는 점만 다릅니다).
    """
    original = face_tensor.clone().detach()
    adv = face_tensor.clone().detach()

    original_embeddings = compute_original_embeddings(original, face_models)
    gaussian_kernel = make_gaussian_kernel(window=7, sigma=3.0) if use_smoothing else None

    for step in range(steps):
        adv.requires_grad_(True)

        loss, per_model_cos = ensemble_cosine_loss(adv, original_embeddings, face_models, gaussian_kernel)

        for fm in face_models:
            fm.model.zero_grad(set_to_none=True)
        loss.backward()

        grad = adv.grad.detach()

        with torch.no_grad():
            adv = adv - step_size * grad.sign()
            perturbation = torch.clamp(adv - original, min=-epsilon, max=epsilon)
            adv = torch.clamp(original + perturbation, min=-1.0, max=1.0)

        adv = adv.detach()

        if (step + 1) % 10 == 0 or step == steps - 1:
            cos_str = ", ".join(f"{name}={cos:.3f}" for name, cos in per_model_cos.items())
            print(f"  step {step + 1:3d}/{steps} | {cos_str}")

    return adv.detach()
