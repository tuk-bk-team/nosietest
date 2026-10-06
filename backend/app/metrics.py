"""
app/metrics.py
--------------------------------------------------------------------------
"사람 눈에 얼마나 비슷해 보이는지"를 재는 화질 지표 두 가지 (PSNR, SSIM).

기존 adversarial_face_protect.py 에 있던 함수를 그대로 옮겨온 것입니다.
여러 모듈(얼굴 크롭 비교, 전체 사진 비교, 추후 다른 공격 모듈)에서
공통으로 재사용하기 위해 별도 파일로 분리했습니다.
"""

import numpy as np


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """두 이미지(0~255, uint8 또는 float) 사이의 PSNR(dB)을 계산합니다.

    값이 클수록(대략 40dB 이상) 사람 눈으로는 거의 구분이 안 되는 수준입니다.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255.0 ** 2) / mse)


def compute_ssim_simple(img_a: np.ndarray, img_b: np.ndarray, block: int = 8) -> float:
    """8x8 블록 단위의 간단한 SSIM 근사치를 계산합니다 (grayscale 기준).

    1.0에 가까울수록 두 이미지의 "구조"가 비슷하다는 뜻입니다.
    (scikit-image의 정식 SSIM보다는 단순화된 버전이지만, 상대 비교 용도로는 충분합니다.)
    """

    def to_gray(img: np.ndarray) -> np.ndarray:
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
