"""
app/face_detect.py
--------------------------------------------------------------------------
얼굴 검출/크롭/복원(합성) 관련 유틸리티.

기존 adversarial_face_protect.py 에 있던 함수들을 그대로 옮겨왔습니다.
얼굴 "검출"은 공격 로직과 무관하게 항상 똑같이 필요한 전처리 단계라서,
앙상블 공격이든 PhotoGuard든 공통으로 이 모듈을 가져다 씁니다.
"""

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_tensor as tv_to_tensor


def crop_face_with_box(img: Image.Image, mtcnn: MTCNN):
    """
    img 전체 사진에서 얼굴을 찾아, facenet-pytorch(MTCNN)가 내부적으로 쓰는 것과
    "완전히 동일한 공식"으로 margin을 적용한 크롭 박스를 계산합니다.

    이렇게 박스 좌표를 직접 들고 있어야, 나중에 적대적 섭동을 적용한 160x160
    얼굴 이미지를 다시 원본 전체 사진의 "정확히 같은 위치"에 되돌려 붙일 수
    있습니다 (mtcnn(img)만 쓰면 크롭된 얼굴만 얻고 좌표는 버려지기 때문).

    반환값: (crop_box [x1,y1,x2,y2], probability)
    """
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("이미지에서 얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 사진으로 다시 시도해주세요.")

    best_idx = int(np.argmax(probs))
    raw_box = boxes[best_idx]
    prob = float(probs[best_idx])

    # 아래 수식은 facenet_pytorch.extract_face() 내부 구현과 동일합니다.
    margin_px = mtcnn.margin
    image_size = mtcnn.image_size
    margin = [
        margin_px * (raw_box[2] - raw_box[0]) / (image_size - margin_px),
        margin_px * (raw_box[3] - raw_box[1]) / (image_size - margin_px),
    ]
    crop_box = [
        int(max(raw_box[0] - margin[0] / 2, 0)),
        int(max(raw_box[1] - margin[1] / 2, 0)),
        int(min(raw_box[2] + margin[0] / 2, img.size[0])),
        int(min(raw_box[3] + margin[1] / 2, img.size[1])),
    ]
    return crop_box, prob


def extract_face_tensor(img: Image.Image, crop_box, image_size: int = 160) -> torch.Tensor:
    """
    crop_box 영역을 잘라서 image_size x image_size로 리사이즈하고, facenet-pytorch
    모델이 기대하는 입력 형태인 [-1, 1] 범위의 텐서로 바꿔줍니다
    (fixed_image_standardization: (픽셀값 - 127.5) / 128.0).
    """
    face_crop = img.crop(crop_box).resize((image_size, image_size), Image.BILINEAR)
    face_raw = tv_to_tensor(np.float32(face_crop))  # (3, H, W), 대략 0~255 범위
    face_tensor = (face_raw - 127.5) / 128.0  # [-1, 1] 근처로 정규화
    return face_tensor.unsqueeze(0)  # (1, 3, H, W)


def composite_face_into_image(
    original_img: Image.Image, adv_face_uint8: np.ndarray, crop_box
) -> Image.Image:
    """
    적대적 섭동이 적용된 "160x160 얼굴 이미지"를, 원본 전체 사진에서 그 얼굴을
    잘라냈던 "정확히 그 위치"에 다시 붙여 넣습니다. 얼굴 이외의 배경은 원본
    그대로 유지됩니다.
    """
    x1, y1, x2, y2 = crop_box
    box_w, box_h = x2 - x1, y2 - y1

    adv_face_resized = Image.fromarray(adv_face_uint8).resize((box_w, box_h), Image.BILINEAR)

    composited = original_img.copy()
    composited.paste(adv_face_resized, (x1, y1))
    return composited


def tensor_to_uint8(face_tensor: torch.Tensor) -> np.ndarray:
    """facenet-pytorch의 [-1, 1] 텐서를 0~255 uint8 numpy 이미지로 변환합니다."""
    img = face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) / 2.0 * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def save_side_by_side(original: np.ndarray, adversarial: np.ndarray, out_path: str):
    """원본 | 보호된 이미지 | 차이(증폭) 를 나란히 붙여 저장합니다."""
    diff = np.abs(original.astype(np.int16) - adversarial.astype(np.int16))
    diff_amplified = np.clip(diff * 8, 0, 255).astype(np.uint8)
    combined = np.concatenate([original, adversarial, diff_amplified], axis=1)
    Image.fromarray(combined).save(out_path)
