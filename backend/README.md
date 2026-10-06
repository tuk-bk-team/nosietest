# 백엔드 (딥페이크 방지 이미지 변환 서비스)

## 구조

```
backend/
├── app/
│   ├── main.py                        FastAPI 서버 (프론트엔드가 호출하는 API)
│   ├── metrics.py                     PSNR/SSIM 계산
│   ├── face_detect.py                 얼굴 위치 탐지 + 크롭 + 원본에 다시 합성
│   ├── models/
│   │   └── face_models.py             얼굴 인식 앙상블 모델 로딩 (facenet-pytorch 2종)
│   └── attacks/
│       ├── recognition_ensemble.py    앙상블 LowKey 공격 (얼굴 인식 회피)
│       ├── generation_photoguard.py   PhotoGuard 공격 (생성모델 편집 방해, VAE)
│       └── combined.py                위 둘을 하나의 섭동으로 결합하는 파이프라인
├── scripts/
│   └── run_cli.py                     서버 없이 터미널에서 바로 테스트하는 CLI
├── requirements.txt
└── results/                            (실행하면 생성됨 - 결과 이미지 저장 위치)
```

## 설치

기존 venv를 그대로 써도 되고, 새로 만들어도 됩니다 (Python 3.12 권장 - 3.14는 pillow 빌드가 깨짐).

```bash
cd backend
source venv/bin/activate        # 기존 venv가 있다면
pip install -r requirements.txt
```

`diffusers`/`transformers`/`accelerate` (PhotoGuard용) 설치가 실패해도 나머지는 정상
동작합니다 - 앙상블 얼굴 인식 공격만 적용되는 걸로 자동 전환됩니다.

### GPU 없는 리눅스 서버(GCP VM 등)에 설치할 때

`pip install torch`를 그냥 실행하면 기본적으로 CUDA(GPU)용 버전이 깔리면서
`nvidia-cublas`, `cuda-toolkit` 같은 수 GB짜리 GPU 패키지를 같이 받으려고
합니다. GPU가 없는 서버에서는 전혀 필요 없고, 오히려 설치만 느려지고
의존성 충돌(numpy 버전이 자꾸 꼬이는 등)이 날 수 있습니다. 이런 환경에서는
CPU 전용 버전을 먼저 깔고 나머지를 설치하세요:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 터미널에서 바로 테스트 (서버 없이)

```bash
python3 scripts/run_cli.py --image dex.jpeg
python3 scripts/run_cli.py --image dex.jpeg --no-photoguard       # 앙상블만, 더 빠름
python3 scripts/run_cli.py --image dex.jpeg --epsilon 0.05 --steps 60
python3 scripts/run_cli.py --image dex.jpeg --target-cosine -1    # 조기 종료 없이 steps 끝까지
```

결과는 `results/combined_protected.png` 에 저장됩니다.

### 코사인 유사도를 많이 낮췄는데 화질이 눈에 띄게 깨진다면

기본적으로 코사인 유사도가 0.5 이하로 떨어지면(상용 얼굴 인식 시스템이 보통
"다른 사람"으로 판단하는 수준) 자동으로 공격을 멈춥니다(`--target-cosine`,
기본값 0.5). steps를 다 채워서 코사인 유사도를 0에 가깝게 만들수록 섭동이
커져서 화질이 더 깨지기 때문입니다 - 특히 저해상도 사진의 작은 얼굴 크롭일수록
눈에 잘 보입니다. `--target-cosine`를 더 낮추면(예: 0.3) 더 강하게 공격하고,
더 높이면(예: 0.6) 더 약하게/덜 눈에 띄게 공격합니다.

## 서버로 실행 (프론트엔드 연동)

```bash
uvicorn app.main:app --reload --port 8000
```

서버가 켜지면:
- `GET http://localhost:8000/health` - 모델 상태 확인 (앙상블 모델 목록, PhotoGuard 사용 가능 여부)
- `POST http://localhost:8000/protect` - 이미지 업로드 (`file`) + `epsilon`, `steps`,
  `use_photoguard` 폼 파라미터 → 보호된 이미지(base64)와 지표(JSON)를 반환

프론트엔드(`frontend/`)는 기본적으로 `http://localhost:8000` 으로 요청을 보냅니다.
둘 다 로컬에서 같이 띄워놓고 쓰는 걸 기준으로 만들었습니다.

## PhotoGuard(VAE)가 안 될 때

`stabilityai/sd-vae-ft-mse` 가중치를 HuggingFace에서 받아와야 하는데, 환경에 따라
(방화벽, 사내망 등) 접속이 막혀있을 수 있습니다. 서버/CLI를 실행했을 때 아래처럼 뜨면
네트워크 문제입니다 - 앙상블 공격은 그대로 정상 동작합니다.

```
[경고] VAE를 불러오지 못했습니다 - PhotoGuard 없이 앙상블 공격만 제공합니다.
```

직접 확인하려면:

```bash
curl -I https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
```

이게 403/타임아웃이면 PhotoGuard는 이 네트워크에서는 못 쓰는 거고, 200이 뜨면
`pip install -U torch diffusers` 로 버전을 맞춘 뒤 다시 시도해보면 됩니다
(diffusers는 비교적 최신 torch를 요구합니다).

## 앙상블에 모델 추가하기

`app/models/face_models.py`의 `FACE_MODEL_SPECS` 리스트에 항목을 추가하면 됩니다.
지금은 facenet-pytorch의 두 가지 사전학습 가중치(`vggface2`, `casia-webface`)를 씁니다.
