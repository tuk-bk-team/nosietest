"""
scripts/run_cli.py
--------------------------------------------------------------------------
터미널에서 서버 없이 바로 돌려볼 수 있는 CLI (예전 adversarial_face_protect.py
와 같은 역할). 앙상블 공격 + (가능하면) PhotoGuard까지 한 번에 적용해서
results/ 폴더에 결과를 저장합니다.

사용법 (backend/ 폴더에서, venv 활성화한 상태로):
  python3 scripts/run_cli.py --image dex.jpeg
  python3 scripts/run_cli.py --image dex.jpeg --no-photoguard   # 앙상블만
  python3 scripts/run_cli.py --image dex.jpeg --epsilon 0.05 --steps 60
"""

import argparse
import os
import sys

# backend/ 를 기준으로 app 패키지를 import할 수 있도록 경로를 추가합니다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from app.attacks.combined import craft_combined_protection
from app.models.face_models import load_face_ensemble


def main():
    parser = argparse.ArgumentParser(description="앙상블 LowKey + PhotoGuard 결합 공격 CLI")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--epsilon", type=float, default=0.04, help="섭동 크기 제한")
    parser.add_argument("--steps", type=int, default=30, help="반복 횟수")
    parser.add_argument("--no-photoguard", action="store_true", help="PhotoGuard(VAE) 단계 건너뛰기")
    parser.add_argument(
        "--target-cosine",
        type=float,
        default=0.5,
        help="이 코사인 유사도 이하로 떨어지면 조기 종료 (기본 0.5, 화질 보존용). "
        "--target-cosine -1 처럼 음수를 주면 steps를 끝까지 다 채웁니다.",
    )
    parser.add_argument("--outdir", type=str, default="results", help="결과 저장 폴더")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    print(f"입력 이미지: {args.image} ({img.size[0]}x{img.size[1]})")

    print("\n[1/3] 얼굴 인식 앙상블 모델 불러오는 중...")
    face_models = load_face_ensemble(device="cpu")
    print(f"  -> {[m.name for m in face_models]}")

    vae = None
    if not args.no_photoguard:
        print("\n[2/3] PhotoGuard용 VAE 불러오는 중...")
        try:
            from app.attacks.generation_photoguard import load_vae
            vae = load_vae(device="cpu")
            print("  -> VAE 준비 완료")
        except Exception as e:  # noqa: BLE001
            print(f"  -> [경고] VAE 로드 실패, 앙상블 공격만 진행합니다: {e}")
    else:
        print("\n[2/3] --no-photoguard 옵션으로 PhotoGuard 단계는 건너뜁니다.")

    target_cosine = None if args.target_cosine < 0 else args.target_cosine
    print(
        f"\n[3/3] 결합 공격 실행 중 (epsilon={args.epsilon}, steps={args.steps}, "
        f"target_cosine={target_cosine})..."
    )
    result = craft_combined_protection(
        original_img=img,
        face_models=face_models,
        vae=vae,
        epsilon=args.epsilon,
        steps=args.steps,
        target_cosine=target_cosine,
    )

    out_path = os.path.join(args.outdir, "combined_protected.png")
    result.protected_image.save(out_path)

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    for name in result.face_cosine_after:
        print(f"[{name}] 코사인 유사도: {result.face_cosine_before[name]:.4f} -> {result.face_cosine_after[name]:.4f}")
    if result.photoguard_used:
        print(f"[PhotoGuard] latent L2 거리: {result.photoguard_latent_distance:.4f} (클수록 편집 방해 효과)")
    else:
        print("[PhotoGuard] 적용 안 됨")
    print(f"전체 사진 PSNR/SSIM: {result.psnr:.2f} dB / {result.ssim:.4f}")
    if result.warnings:
        print("경고:")
        for w in result.warnings:
            print(f"  - {w}")
    print(f"\n결과 저장됨: {out_path}")


if __name__ == "__main__":
    main()
