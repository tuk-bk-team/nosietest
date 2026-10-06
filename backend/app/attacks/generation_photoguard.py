"""
app/attacks/generation_photoguard.py
--------------------------------------------------------------------------
PhotoGuard 원리 - "생성/편집 AI 방해" 공격.

LowKey/Fawkes가 "얼굴 인식 모델의 특징 벡터"를 공격한다면, PhotoGuard는
Stable Diffusion 같은 이미지 생성 모델이 사진을 편집하기 전 제일 먼저
거치는 단계인 "VAE 인코더"(사진을 latent라는 압축된 벡터로 바꾸는 부분)를
공격합니다. 이 latent가 망가지면, 그 뒤에 이어지는 디노이징/편집 과정 전체가
망가진 입력을 기반으로 진행되기 때문에 결과물이 심하게 왜곡됩니다.

MadryLab/photoguard (https://github.com/madrylab/photoguard) 저장소의
"encoder attack"(간단한 버전) 방식을 참고했습니다. 원문은 보통 "정해둔 타겟
이미지의 latent로 끌어당기는" 방식을 쓰지만, 여기서는 LowKey 쪽 공격과 일관된
형태로 맞추기 위해 "원본 latent에서 최대한 멀어지게 밀어내는" 더 단순한
버전을 사용합니다 - 목표(편집 파이프라인 입력을 망가뜨리는 것)는 동일합니다.

중요: VAE 가중치(stabilityai/sd-vae-ft-mse, 약 335MB)는 HuggingFace에서
받아옵니다. 이 환경(샌드박스/디바이스 브릿지)에서는 huggingface.co 접속이
막혀 있는 걸 확인했지만, 사용자가 본인 터미널에서 직접 실행할 때는 네트워크
제약이 다를 수 있습니다. load_vae()가 실패하면 그 이유를 명확한 한국어
메시지로 알려줍니다.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


# Stable Diffusion 계열이 공통으로 쓰는 VAE (비교적 가벼운 선택지).
DEFAULT_VAE_REPO = "stabilityai/sd-vae-ft-mse"

# PhotoGuard/Stable Diffusion 관례상 VAE는 8의 배수 해상도를 기대합니다.
# 512는 SD 1.x 계열의 기본 해상도입니다 (전체 사진을 이 크기로 리사이즈해서 공격).
VAE_WORKING_SIZE = 512


def load_vae(device: str = "cpu"):
    """
    diffusers 라이브러리로 Stable Diffusion의 VAE(AutoencoderKL)를 불러옵니다.

    실패하면 RuntimeError로 "왜 실패했는지 + 어떻게 해보면 되는지"를
    한국어로 친절하게 설명합니다 (네트워크 문제가 제일 흔한 원인이기 때문).
    """
    try:
        from diffusers import AutoencoderKL
    except ImportError as e:
        raise RuntimeError(
            "diffusers 패키지가 설치되어 있지 않습니다. "
            "'pip install diffusers' 로 먼저 설치해주세요."
        ) from e
    except Exception as e:  # noqa: BLE001 - diffusers가 torch 버전 호환 문제 등으로
        # import 자체에서 깨지는 경우도 있어서(예: 너무 오래된 torch), ImportError가
        # 아닌 다른 예외도 폭넓게 잡아서 "PhotoGuard 없이 계속 진행"할 수 있게 합니다.
        raise RuntimeError(
            "diffusers 라이브러리를 불러오는 중 오류가 발생했습니다 (torch 버전 호환 문제일 "
            f"가능성이 높습니다 - 'pip install -U torch diffusers'로 업그레이드해보세요). 원래 에러: {e}"
        ) from e

    try:
        vae = AutoencoderKL.from_pretrained(DEFAULT_VAE_REPO)
    except Exception as e:  # noqa: BLE001 - 사용자에게 원인을 그대로 보여주기 위해 광범위하게 잡음
        raise RuntimeError(
            "Stable Diffusion VAE 가중치를 HuggingFace(huggingface.co)에서 받아오지 "
            "못했습니다. 보통 네트워크/방화벽 문제입니다. 터미널에서 다음 명령으로 "
            "먼저 접속이 되는지 확인해보세요:\n"
            "  curl -I https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json\n"
            "403/타임아웃이 뜨면 PhotoGuard 부분은 건너뛰고 앙상블 공격만 사용하세요 "
            f"(원래 에러: {e})"
        ) from e

    vae = vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def pil_to_vae_tensor(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """PIL 이미지를 VAE 입력 형태(1,3,512,512, [-1,1] 범위)로 변환합니다."""
    img_resized = img.convert("RGB").resize((VAE_WORKING_SIZE, VAE_WORKING_SIZE), Image.BILINEAR)
    arr = torch.from_numpy(
        __import__("numpy").array(img_resized, dtype="float32")
    ).permute(2, 0, 1) / 255.0  # (3, H, W), [0, 1]
    tensor = (arr * 2.0 - 1.0).unsqueeze(0).to(device)  # [-1, 1]
    return tensor


@torch.no_grad()
def encode_latent_mean(vae, image_tensor: torch.Tensor) -> torch.Tensor:
    """VAE로 이미지를 인코딩해서 latent 분포의 평균(mean)만 꺼냅니다 (그래디언트 불필요한 경우용)."""
    return vae.encode(image_tensor).latent_dist.mean


def photoguard_latent_loss(
    image_tensor: torch.Tensor,
    vae,
    original_latent_mean: torch.Tensor,
) -> torch.Tensor:
    """
    "이번 스텝"의 PhotoGuard 손실값을 계산합니다.

    image_tensor: 지금 시점의 섭동이 적용된 전체 이미지 텐서 (1,3,512,512), requires_grad=True
    original_latent_mean: 공격 시작 전 원본 이미지의 latent 평균 (미리 한 번 계산해둔 것)

    LowKey 쪽과 반대로, 여기서는 "코사인 유사도"가 아니라 "L2 거리"를 씁니다
    (latent 공간은 방향보다 위치 자체가 의미를 가지기 때문에 L2가 더 자연스럽습니다).
    손실을 "음수 거리"로 만들어서, 이 값을 작게 만들면(=거리가 커지면) 공격이
    되도록 합니다 - 바깥의 결합 루프(combined.py)는 항상 "loss를 줄이는 방향"으로
    동작하도록 통일했기 때문입니다.
    """
    latent_dist = vae.encode(image_tensor).latent_dist.mean
    distance = F.mse_loss(latent_dist, original_latent_mean)
    return -distance  # 거리를 "최대화"하고 싶으므로 음수를 취해 "최소화 loss"로 맞춤


def craft_photoguard_attack(
    image_tensor: torch.Tensor,
    vae,
    epsilon: float = 0.05,
    step_size: float = 0.005,
    steps: int = 50,
) -> Tuple[torch.Tensor, float]:
    """
    PhotoGuard 공격만 단독으로 돌리는 독립 실행형 함수.

    image_tensor: (1,3,512,512), [-1,1] 범위의 전체 이미지 텐서
    반환값: (보호된 이미지 텐서, 최종 latent L2 거리)
    """
    original = image_tensor.clone().detach()
    adv = image_tensor.clone().detach()

    original_latent_mean = encode_latent_mean(vae, original)

    for step in range(steps):
        adv.requires_grad_(True)
        loss = photoguard_latent_loss(adv, vae, original_latent_mean)

        vae.zero_grad(set_to_none=True)
        loss.backward()

        grad = adv.grad.detach()
        with torch.no_grad():
            adv = adv - step_size * grad.sign()
            perturbation = torch.clamp(adv - original, min=-epsilon, max=epsilon)
            adv = torch.clamp(original + perturbation, min=-1.0, max=1.0)
        adv = adv.detach()

        if (step + 1) % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                cur_latent = encode_latent_mean(vae, adv)
                cur_dist = F.mse_loss(cur_latent, original_latent_mean).item()
            print(f"  step {step + 1:3d}/{steps} | latent L2 거리: {cur_dist:.4f} (클수록 편집 방해 효과 큼)")

    with torch.no_grad():
        final_latent = encode_latent_mean(vae, adv)
        final_dist = F.mse_loss(final_latent, original_latent_mean).item()

    return adv.detach(), final_dist
