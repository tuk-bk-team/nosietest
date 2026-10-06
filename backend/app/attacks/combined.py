"""
app/attacks/combined.py
--------------------------------------------------------------------------
"하나의 사진에 LowKey(얼굴 인식 회피) + PhotoGuard(생성모델 편집 방해)를
동시에 적용" 하는 결합 파이프라인.

핵심 아이디어 (대화에서 정리했던 내용 그대로 구현):
  - 모델은 합치지 않습니다. 얼굴 인식 모델 앙상블(2개)과 PhotoGuard용 VAE,
    총 3개의 서로 다른 사전학습 모델을 "그대로" 불러와서 동시에 켜놓습니다.
  - 최적화 대상은 모델이 아니라 "사진 전체에 걸리는 섭동(delta) 하나"입니다.
  - 매 스텝마다: 섭동이 적용된 사진에서
      ① 얼굴 부분만 잘라 앙상블 모델들에 통과시켜 "인식 회피" 손실을 구하고
      ② 사진 전체를 VAE에 통과시켜 "편집 방해" 손실을 구한 뒤
    두 손실을 가중합(weighted sum)해서 하나의 loss로 만들고, 그 loss 기준으로
    delta 하나만 업데이트합니다.

해상도에 대한 중요한 메모:
  얼굴 인식 모델은 160x160 "얼굴 크롭"만 보고, PhotoGuard의 VAE는 보통
  512x512 "전체 사진"을 봅니다. 이 둘을 한 섭동으로 동시에 공격하기 위해,
  전체 사진을 적당한 "작업 해상도"(기본 768px)로 리사이즈한 뒤 그 위에서
  섭동을 계산하고, 얼굴 영역은 그 안에서 미분 가능하게(슬라이싱 + 리사이즈)
  잘라내 앙상블에 넣고, 전체 이미지는 512x512로 미분 가능하게 리사이즈해서
  VAE에 넣습니다. 슬라이싱/리사이즈 모두 PyTorch 연산이라 역전파가 끊기지
  않고 delta까지 그대로 흘러갑니다.

PhotoGuard(VAE)가 네트워크 문제로 로드되지 않으면, 자동으로 "앙상블 공격만"
수행하는 모드로 내려가고 그 사실을 결과에 명시합니다 - 전체 파이프라인이
중간에 죽지 않고 항상 결과를 내주도록 하기 위함입니다.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN

from app.attacks.recognition_ensemble import (
    compute_original_embeddings,
    ensemble_cosine_loss,
    gaussian_blur_image,
    make_gaussian_kernel,
)
from app.attacks.generation_photoguard import (
    VAE_WORKING_SIZE,
    encode_latent_mean,
)
from app.face_detect import crop_face_with_box
from app.metrics import compute_psnr, compute_ssim_simple
from app.models.face_models import FaceModel

WORKING_MAX_SIDE = 768  # 섭동을 계산할 "작업 해상도"의 최대 변 길이


@dataclass
class CombinedResult:
    protected_image: Image.Image          # 최종 보호된 전체 사진 (원본 해상도)
    face_cosine_before: dict               # 공격 전 각 모델의 코사인 유사도 (전부 1.0)
    face_cosine_after: dict                # 공격 후 각 모델의 코사인 유사도
    photoguard_used: bool                  # PhotoGuard(VAE)가 실제로 적용됐는지
    photoguard_latent_distance: Optional[float] = None
    psnr: float = 0.0
    ssim: float = 0.0
    warnings: List[str] = field(default_factory=list)


def _resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    """긴 변 기준으로 max_side를 넘지 않게 비율을 유지하며 축소합니다 (확대는 하지 않음)."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img.copy()
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def _pil_to_tensor_pm1(img: Image.Image) -> torch.Tensor:
    """PIL(RGB) -> (1,3,H,W), [-1,1] 텐서."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def _tensor_pm1_to_pil(tensor: torch.Tensor) -> Image.Image:
    """(1,3,H,W), [-1,1] 텐서 -> PIL(RGB)."""
    arr = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = (arr + 1.0) / 2.0 * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def craft_combined_protection(
    original_img: Image.Image,
    face_models: List[FaceModel],
    vae=None,  # None이면 PhotoGuard 단계를 건너뜀 (앙상블만 수행)
    mtcnn: Optional[MTCNN] = None,
    epsilon: float = 0.04,
    steps: int = 30,
    face_weight: float = 1.0,
    vae_weight: float = 1.0,
    use_smoothing: bool = True,
    target_cosine: Optional[float] = 0.5,
) -> CombinedResult:
    """
    target_cosine: 각 모델의 코사인 유사도가 전부 이 값 이하로 떨어지면 steps를
    다 채우지 않고 조기 종료합니다 (기본 0.5 - 상용 얼굴 인식 시스템은 대체로
    이보다 높으면 "같은 사람"으로 판단하기 때문에, 이 이하면 이미 충분히 회피된
    것으로 봅니다). None으로 주면 옛날처럼 steps를 끝까지 채웁니다.

    왜 필요한가: 코사인 유사도를 0에 가깝게 밀어붙일수록 섭동이 커져서 "필요
    이상으로" 화질이 깨집니다 (특히 저해상도 사진의 작은 얼굴 크롭에서는 눈에
    확연히 보일 정도로). 목표치만 넘기면 바로 멈추는 게 화질/효과의 균형에
    더 유리합니다.
    """
    warnings: List[str] = []

    if mtcnn is None:
        mtcnn = MTCNN(image_size=160, margin=20, post_process=True)

    # --- 1) 작업 해상도로 축소 (원본이 너무 크면 CPU에서 느려서) ---
    working_img = _resize_keep_aspect(original_img, WORKING_MAX_SIDE)
    working_tensor = _pil_to_tensor_pm1(working_img)

    # --- 2) 얼굴 위치 찾기 (작업 해상도 기준, 한 번만) ---
    crop_box, prob = crop_face_with_box(working_img, mtcnn)
    x1, y1, x2, y2 = crop_box

    # --- 3) 섭동(delta) 준비 ---
    delta = torch.zeros_like(working_tensor, requires_grad=True)
    step_size = max(epsilon / 12, 0.003)
    gaussian_kernel = make_gaussian_kernel(window=7, sigma=3.0) if use_smoothing else None

    # --- 4) 원본 기준값들 미리 계산 (공격 전 상태) ---
    with torch.no_grad():
        clean_face = F.interpolate(
            working_tensor[:, :, y1:y2, x1:x2], size=(160, 160), mode="bilinear", align_corners=False
        )
        original_embeddings = compute_original_embeddings(clean_face, face_models)
        face_cosine_before = {fm.name: 1.0 for fm in face_models}  # 자기 자신과의 유사도는 정의상 1.0

        original_latent_mean = None
        if vae is not None:
            vae_input_clean = F.interpolate(working_tensor, size=(VAE_WORKING_SIZE, VAE_WORKING_SIZE), mode="bilinear", align_corners=False)
            original_latent_mean = encode_latent_mean(vae, vae_input_clean)

    if vae is None:
        warnings.append("PhotoGuard(VAE)를 불러오지 못해 앙상블 얼굴 인식 공격만 적용했습니다.")

    # --- 5) 결합 최적화 루프 ---
    # PhotoGuard(VAE)가 켜져 있으면 스텝당 시간이 꽤 걸릴 수 있어서(CPU에서 512x512
    # 인코딩+역전파), 매 스텝마다 진행률/소요시간을 찍습니다 - 그래야 "멈춘 것처럼
    # 보여서" 중간에 취소하는 일이 없습니다. VAE 없이 앙상블만 할 때는 원래처럼
    # 10스텝마다만 찍습니다(그만큼 빨라서 로그가 너무 많이 찍히는 걸 막기 위함).
    print_every = 1 if vae is not None else 10
    for step in range(steps):
        step_t0 = time.time()
        adv = torch.clamp(working_tensor + delta, -1.0, 1.0)

        # ① 얼굴 인식 앙상블 손실 (미분 가능한 슬라이싱 + 리사이즈)
        face_region = adv[:, :, y1:y2, x1:x2]
        face_160 = F.interpolate(face_region, size=(160, 160), mode="bilinear", align_corners=False)
        loss_face, per_model_cos = ensemble_cosine_loss(face_160, original_embeddings, face_models, gaussian_kernel)

        total_loss = face_weight * loss_face

        # ② PhotoGuard 손실 (VAE가 로드됐을 때만)
        if vae is not None:
            vae_input = F.interpolate(adv, size=(VAE_WORKING_SIZE, VAE_WORKING_SIZE), mode="bilinear", align_corners=False)
            latent = vae.encode(vae_input).latent_dist.mean
            vae_distance = F.mse_loss(latent, original_latent_mean)
            loss_vae = -vae_distance  # 거리를 최대화 = loss는 최소화
            total_loss = total_loss + vae_weight * loss_vae

        for fm in face_models:
            fm.model.zero_grad(set_to_none=True)
        if vae is not None:
            vae.zero_grad(set_to_none=True)
        if delta.grad is not None:
            delta.grad.zero_()
        total_loss.backward()

        with torch.no_grad():
            grad = delta.grad.detach()
            delta -= step_size * grad.sign()
            delta.clamp_(-epsilon, epsilon)
        delta.requires_grad_(True)

        step_elapsed = time.time() - step_t0
        if (step + 1) % print_every == 0 or step == steps - 1:
            cos_str = ", ".join(f"{k}={v:.3f}" for k, v in per_model_cos.items())
            extra = ""
            if vae is not None:
                extra = f" | latent 거리: {vae_distance.item():.4f}"
            remaining = (steps - step - 1) * step_elapsed
            print(
                f"  [combined] step {step + 1:3d}/{steps} | {cos_str}{extra} "
                f"| {step_elapsed:.1f}초/스텝 (남은 시간 약 {remaining:.0f}초)"
            )

        # 목표 코사인 유사도 이하로 떨어졌으면 더 돌리지 않고 멈춥니다
        # (불필요하게 더 공격해서 화질만 깎아먹는 걸 막기 위함).
        if target_cosine is not None and max(per_model_cos.values()) <= target_cosine:
            msg = (
                f"목표 코사인 유사도({target_cosine:.2f}) 이하로 떨어져 "
                f"{step + 1}/{steps} 스텝에서 조기 종료했습니다 (화질 저하를 줄이기 위함)."
            )
            print(f"  [combined] {msg}")
            warnings.append(msg)
            break

    # --- 6) 최종 결과 정리 ---
    with torch.no_grad():
        final_adv = torch.clamp(working_tensor + delta, -1.0, 1.0)
        final_face = F.interpolate(final_adv[:, :, y1:y2, x1:x2], size=(160, 160), mode="bilinear", align_corners=False)
        face_cosine_after = {}
        for fm, orig_emb in zip(face_models, original_embeddings):
            cur_emb = F.normalize(fm.model(final_face), dim=1)
            face_cosine_after[fm.name] = (orig_emb * cur_emb).sum(dim=1).item()

        photoguard_latent_distance = None
        if vae is not None:
            final_vae_input = F.interpolate(final_adv, size=(VAE_WORKING_SIZE, VAE_WORKING_SIZE), mode="bilinear", align_corners=False)
            final_latent = vae.encode(final_vae_input).latent_dist.mean
            photoguard_latent_distance = F.mse_loss(final_latent, original_latent_mean).item()

    protected_working_img = _tensor_pm1_to_pil(final_adv)
    # 작업 해상도 -> 원본 해상도로 다시 확대 (원본 크기 그대로 최종 결과물 제공)
    protected_full_img = protected_working_img.resize(original_img.size, Image.BILINEAR)

    original_arr = np.array(original_img.convert("RGB"))
    protected_arr = np.array(protected_full_img)
    psnr = compute_psnr(original_arr, protected_arr)
    ssim = compute_ssim_simple(original_arr, protected_arr)

    return CombinedResult(
        protected_image=protected_full_img,
        face_cosine_before=face_cosine_before,
        face_cosine_after=face_cosine_after,
        photoguard_used=(vae is not None),
        photoguard_latent_distance=photoguard_latent_distance,
        psnr=psnr,
        ssim=ssim,
        warnings=warnings,
    )
