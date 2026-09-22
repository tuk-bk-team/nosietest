"""
noise_test.py
--------------
딥페이크 방지용 이미지 노이즈(섭동) 실험 스크립트

이 스크립트는 사진에 여러 종류/강도의 노이즈를 넣어보고,
- 눈으로 봤을 때 원본과 얼마나 비슷한지 (PSNR, SSIM 수치로 확인)
- 이미지를 어떤 "레이어(계층)"로 나눠서 노이즈를 줄 수 있는지
를 실험해볼 수 있게 만든 기본 틀입니다.

사용법:
    python noise_test.py --image 원본사진.jpg

필요한 라이브러리 (터미널에서 설치):
    pip install numpy pillow opencv-python scikit-image
"""

import argparse
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 0. 이미지를 나눌 수 있는 "레이어(계층)" 종류 정리
# ============================================================
#
# 1) 색상 채널 레이어 (RGB)
#    - 이미지는 원래 R(빨강), G(초록), B(파랑) 3개 채널이 겹쳐진 것.
#    - 채널마다 따로 노이즈를 넣어볼 수 있음 (예: G채널에만 노이즈를 세게)
#
# 2) 주파수 레이어 (저주파 vs 고주파) - FFT 기반
#    - 저주파 = 이미지의 전체적인 형태/색 (뭉뚱그려진 정보)
#    - 고주파 = 경계선, 디테일, 질감 (세밀한 정보)
#    - 사람 눈은 저주파 변화에 민감하고, 고주파 변화에는 둔감함
#
# 3) 노이즈 강도(epsilon) 레이어
#    - 같은 노이즈를 강도만 다르게 여러 단계로 만들어서 비교
#    - "어느 정도 강도부터 사람 눈에 보이기 시작하는지" 실험할 때 사용
#
# 참고: 딥러닝 모델 "내부"의 레이어(층)는 이것과 다른 개념입니다.
# (모델 안의 은닉층/특징맵을 말하는 것 - 이미지 파일 자체를 자르는 게 아님)
# 이 스크립트는 "이미지 파일 자체를 나누는" 1)~3)번 이야기입니다.


def load_image_as_array(image_path):
    """
    이미지 파일을 불러와서 numpy 배열로 변환합니다.
    반환값: (높이, 너비, 3) 형태의 배열, 값 범위는 0~255 (float32)
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img).astype(np.float32)


def save_array_as_image(array, save_path):
    """
    numpy 배열(0~255 float)을 다시 이미지 파일로 저장합니다.
    """
    array_uint8 = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array_uint8).save(save_path)


# ============================================================
# 1. 노이즈 강도별 테스트 (가장 기본)
# ============================================================
def add_gaussian_noise(image_array, sigma):
    """
    가우시안(정규분포) 노이즈를 이미지에 더합니다.

    image_array: (H, W, 3) numpy 배열, 0~255 범위
    sigma: 노이즈의 표준편차. 값이 클수록 노이즈가 강함
           (참고로 사람 눈에는 보통 sigma=5~10 정도부터 티가 나기 시작합니다)
    """
    noise = np.random.normal(loc=0, scale=sigma, size=image_array.shape)
    noisy = image_array + noise
    return np.clip(noisy, 0, 255)


def test_noise_levels(image_array, sigmas, output_dir):
    """
    여러 강도(sigma)의 노이즈를 각각 적용해보고 결과를 저장 + 수치로 비교합니다.
    """
    print("\n[1] 노이즈 강도별 테스트")
    print(f"{'강도(sigma)':>12} | {'PSNR(dB)':>10} | {'SSIM':>8}   <- PSNR/SSIM이 높을수록 원본과 비슷함(=노이즈가 덜 보임)")
    print("-" * 60)

    original_uint8 = np.clip(image_array, 0, 255).astype(np.uint8)

    for sigma in sigmas:
        noisy = add_gaussian_noise(image_array, sigma)
        noisy_uint8 = noisy.astype(np.uint8)

        # 원본과 비교해서 얼마나 달라졌는지 수치로 측정
        p = psnr(original_uint8, noisy_uint8, data_range=255)
        s = ssim(original_uint8, noisy_uint8, channel_axis=2, data_range=255)

        print(f"{sigma:>12} | {p:>10.2f} | {s:>8.4f}")

        save_path = os.path.join(output_dir, f"noise_sigma_{sigma}.png")
        save_array_as_image(noisy, save_path)

    print(f"\n결과 이미지가 {output_dir}/ 폴더에 저장되었습니다.")
    print("여러 장을 나란히 띄워놓고 '몇 단계부터 눈에 띄는지' 직접 확인해보세요.")


# ============================================================
# 2. 색상 채널별 테스트
# ============================================================
def test_channel_noise(image_array, sigma, output_dir):
    """
    R, G, B 채널 중 하나에만 노이즈를 넣어서 채널별로 눈에 띄는 정도가
    다른지 비교합니다. (사람 눈은 보통 초록색 변화에 가장 민감합니다)
    """
    print("\n[2] 채널별 노이즈 테스트")
    channel_names = ["R(빨강)", "G(초록)", "B(파랑)"]

    for ch_idx, ch_name in enumerate(channel_names):
        noisy = image_array.copy()
        noise = np.random.normal(0, sigma, image_array.shape[:2])
        noisy[:, :, ch_idx] = noisy[:, :, ch_idx] + noise
        noisy = np.clip(noisy, 0, 255)

        save_path = os.path.join(output_dir, f"noise_channel_{ch_idx}_{ch_name[0]}.png")
        save_array_as_image(noisy, save_path)
        print(f"  {ch_name} 채널에만 노이즈(sigma={sigma}) 적용 -> {save_path}")


# ============================================================
# 3. 주파수 영역(저주파/고주파) 분리 + 고주파에만 노이즈 넣기
# ============================================================
def add_high_frequency_noise(image_array, sigma, low_freq_radius=30):
    """
    이미지를 저주파/고주파로 나눈 뒤, 고주파 성분에만 노이즈를 추가합니다.
    (사람 눈은 고주파 변화에 둔감하므로, 실제 적대적 섭동 기법들이
     이 원리를 활용하는 경우가 많습니다)

    low_freq_radius: 저주파로 취급할 중심 영역의 반지름 (값이 클수록 저주파 범위가 넓어짐)
    """
    result = np.zeros_like(image_array)

    for ch in range(3):
        channel = image_array[:, :, ch]
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # 1) FFT로 주파수 영역으로 변환
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        # 2) 저주파/고주파를 나눌 마스크 생성 (중심 = 저주파)
        mask_low = np.zeros((rows, cols), dtype=np.float32)
        r0 = max(crow - low_freq_radius, 0)
        r1 = min(crow + low_freq_radius, rows)
        c0 = max(ccol - low_freq_radius, 0)
        c1 = min(ccol + low_freq_radius, cols)
        mask_low[r0:r1, c0:c1] = 1
        mask_high = 1 - mask_low

        # 3) 고주파 영역에만 노이즈를 추가 (주파수 도메인에서)
        noise = np.random.normal(0, sigma, fshift.shape)
        fshift_noisy = fshift + (noise * mask_high)

        # 4) 다시 이미지(공간 도메인)로 역변환
        f_ishift = np.fft.ifftshift(fshift_noisy)
        img_back = np.fft.ifft2(f_ishift)
        result[:, :, ch] = np.abs(img_back)

    return np.clip(result, 0, 255)


def test_frequency_noise(image_array, sigmas, output_dir):
    print("\n[3] 고주파 대역 노이즈 테스트 (실제 적대적 섭동과 유사한 방식)")
    original_uint8 = np.clip(image_array, 0, 255).astype(np.uint8)

    for sigma in sigmas:
        noisy = add_high_frequency_noise(image_array, sigma)
        noisy_uint8 = noisy.astype(np.uint8)

        p = psnr(original_uint8, noisy_uint8, data_range=255)
        s = ssim(original_uint8, noisy_uint8, channel_axis=2, data_range=255)
        print(f"  sigma={sigma:>6} | PSNR={p:.2f}dB | SSIM={s:.4f}")

        save_path = os.path.join(output_dir, f"noise_highfreq_sigma_{sigma}.png")
        save_array_as_image(noisy, save_path)


# ============================================================
# 실행부
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="딥페이크 방지 노이즈 실험 스크립트")
    parser.add_argument("--image", type=str, required=True, help="테스트할 원본 사진 경로")
    parser.add_argument("--outdir", type=str, default="noise_test_results", help="결과 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"원본 이미지 불러오는 중: {args.image}")
    image_array = load_image_as_array(args.image)

    # 원본도 같이 저장해두면 나중에 비교하기 편합니다
    save_array_as_image(image_array, os.path.join(args.outdir, "original.png"))

    # 1) 노이즈 강도 단계별 테스트
    test_noise_levels(image_array, sigmas=[2, 5, 10, 20, 40], output_dir=args.outdir)

    # 2) 채널별 테스트
    test_channel_noise(image_array, sigma=15, output_dir=args.outdir)

    # 3) 주파수 영역(고주파) 테스트
    # 참고: 주파수 도메인에 더하는 노이즈는 역변환을 거치면서 크기가 줄어들기 때문에,
    # 공간(픽셀) 도메인 노이즈와 "비슷한 세기"를 보려면 sigma를 훨씬 크게 줘야 합니다.
    # (아래 값들은 [1]번 테스트와 비슷한 PSNR 범위가 나오도록 미리 맞춰둔 값입니다)
    test_frequency_noise(image_array, sigmas=[500, 3000, 8000], output_dir=args.outdir)

    print("\n모든 테스트가 끝났습니다. 결과 폴더를 열어서 눈으로 직접 비교해보세요.")


if __name__ == "__main__":
    main()