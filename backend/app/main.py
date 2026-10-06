"""
app/main.py
--------------------------------------------------------------------------
FastAPI 백엔드 서버. 프론트엔드(Next.js)에서 사진을 업로드하면, 이 서버가
"앙상블 LowKey 공격 + PhotoGuard 공격"을 결합해서 적용한 뒤 결과를 돌려줍니다.

실행 방법 (backend/ 폴더 기준):
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8000

그 다음 프론트엔드(frontend/)에서 http://localhost:8000 으로 요청을 보내면 됩니다.
(둘 다 로컬에서 돌리는 걸 기준으로, CORS는 localhost 전체를 허용해뒀습니다.)
"""

import base64
import io
import time
import traceback

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.attacks.combined import craft_combined_protection
from app.models.face_models import load_face_ensemble

app = FastAPI(title="딥페이크 방지 이미지 변환 서비스 API")

# 로컬 개발 단계라 모든 localhost 출처를 허용합니다.
# (배포할 때는 실제 프론트엔드 도메인만 넣는 식으로 좁혀야 합니다.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# 모델은 서버가 켜질 때 "딱 한 번만" 불러옵니다 (요청마다 다시 불러오면 매우 느림).
# --------------------------------------------------------------------
print("=" * 60)
print("서버 시작: 얼굴 인식 앙상블 모델 불러오는 중...")
FACE_MODELS = load_face_ensemble(device="cpu")
print(f"얼굴 인식 모델 {len(FACE_MODELS)}개 준비 완료: {[m.name for m in FACE_MODELS]}")

VAE = None
VAE_LOAD_ERROR = None
try:
    from app.attacks.generation_photoguard import load_vae
    print("PhotoGuard용 VAE 불러오는 중 (HuggingFace 다운로드, 처음엔 시간이 걸릴 수 있습니다)...")
    VAE = load_vae(device="cpu")
    print("VAE 준비 완료 - PhotoGuard 공격도 같이 적용됩니다.")
except Exception as e:  # noqa: BLE001
    VAE_LOAD_ERROR = str(e)
    print(f"[경고] VAE를 불러오지 못했습니다 - PhotoGuard 없이 앙상블 공격만 제공합니다.\n사유: {e}")
print("=" * 60)


@app.get("/health")
def health():
    """서버/모델 상태 확인용. 프론트엔드가 시작할 때 이걸 먼저 찔러보면 됩니다."""
    return {
        "status": "ok",
        "face_models": [m.name for m in FACE_MODELS],
        "photoguard_available": VAE is not None,
        "photoguard_error": VAE_LOAD_ERROR,
    }


@app.post("/protect")
async def protect(
    file: UploadFile = File(...),
    epsilon: float = Form(0.04),
    steps: int = Form(30),
    use_photoguard: bool = Form(True),
    target_cosine: float = Form(0.5),
):
    """
    사진을 업로드 받아서 보호 처리된 사진을 돌려줍니다.

    요청(form-data): file(이미지), epsilon(float, 기본 0.04), steps(int, 기본 30),
                      use_photoguard(bool, 기본 True - VAE가 없으면 자동으로 꺼짐),
                      target_cosine(float, 기본 0.5 - 이 값 이하로 코사인 유사도가
                      떨어지면 조기 종료해서 불필요한 화질 저하를 막습니다.
                      steps를 끝까지 다 채우고 싶으면 음수(예: -1)를 보내세요.)
    응답(JSON): { protected_image_base64, metrics: {...}, warnings: [...] }
    """
    t0 = time.time()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        vae_to_use = VAE if (use_photoguard and VAE is not None) else None
        target_cosine_arg = None if target_cosine < 0 else target_cosine

        result = craft_combined_protection(
            original_img=img,
            face_models=FACE_MODELS,
            vae=vae_to_use,
            epsilon=epsilon,
            steps=steps,
            target_cosine=target_cosine_arg,
        )

        buf = io.BytesIO()
        result.protected_image.save(buf, format="PNG")
        protected_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        elapsed = time.time() - t0
        return JSONResponse({
            "protected_image_base64": protected_b64,
            "metrics": {
                "face_cosine_before": result.face_cosine_before,
                "face_cosine_after": result.face_cosine_after,
                "photoguard_used": result.photoguard_used,
                "photoguard_latent_distance": result.photoguard_latent_distance,
                "psnr": result.psnr,
                "ssim": result.ssim,
            },
            "warnings": result.warnings,
            "elapsed_seconds": round(elapsed, 2),
        })
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
