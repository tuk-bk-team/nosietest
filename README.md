## 진행 상황 (2026-10-06 기준)

### ✅ 완료

- **앙상블 얼굴인식 공격**: facenet-pytorch 기반 `facenet-vggface2`, `facenet-casia-webface` 두 모델을 동시에 공격하는 방식으로 구현 (`app/attacks/recognition_ensemble.py`)
- **PhotoGuard(VAE) 결합 공격**: Stable Diffusion VAE(`sd-vae-ft-mse`) 잠재공간 거리를 같이 최적화해서, 얼굴 인식 공격 + 생성모델 편집 방해를 하나의 섭동으로 합침 (`app/attacks/combined.py`)
- **`target_cosine` 조기종료**: 코사인 유사도가 목표치(기본 0.5) 밑으로 떨어지면 자동으로 멈춰서 불필요한 화질 저하 방지
- **백엔드 구조화**: FastAPI 서버로 재구성 (`/health`, `/protect` 엔드포인트), CLI(`scripts/run_cli.py`)도 별도 제공
- **프론트엔드 연동**: `/full-protect` 페이지에서 사진 업로드 → 결과(원본/보호된 사진, 코사인 유사도, PSNR/SSIM) 확인 가능
- **GCP 배포**: Compute Engine VM(e2-standard-4, 4vCPU/16GB)에 서버 배포 완료
  - CPU 전용 PyTorch 설치, numpy 버전 고정 등 의존성 이슈 해결
  - 초기 e2-medium(4GB RAM)에서는 PhotoGuard 연산 중 OOM(메모리 부족)으로 서버가 죽는 문제 발생 → 머신 사양 업그레이드로 해결
- **전이성(transferability) 검증**: `deepface`로 공격에 사용하지 않은 모델(VGG-Face, ArcFace, SFace)에 대해 교차 검증
  - 결과: SFace는 방어 성공, VGG-Face/ArcFace는 방어 실패 → 유사 아키텍처(FaceNet 계열) 안에서는 전이가 잘 되지만, 이종 아키텍처로는 전이성이 제한적이라는 한계 확인

### 🔲 진행 예정 / 남은 작업

- [ ] **동시 요청 처리**: 현재 `/protect`가 동기 방식이라 요청 처리 중 서버가 다른 요청(SSH 포함)을 못 받음 → 비동기/백그라운드 처리로 개선 필요
- [ ] **서버 상시 구동**: 지금은 `nohup`으로만 띄운 상태라 VM 재부팅 시 날아감 → systemd 서비스 등록 필요
- [ ] **전이성 개선**: 앙상블에 ArcFace 등 이종 아키텍처 모델 추가 검토 (현재 2개 모델이 전부 FaceNet 계열이라 전이 범위가 좁음)
- [ ] **상용 API 검증**: AWS Rekognition `CompareFaces` 등으로 블랙박스 상용 시스템 대상 추가 검증
- [ ] **워터마킹 기능**: 적대적 섭동 외에 워터마킹 삽입 기능은 아직 미구현
- [ ] **GitHub 정리**: `.gitignore`에 가상환경/캐시 폴더 제외 설정, 최신 백엔드 코드 커밋/푸시 마무리

## LowKey 논문과의 비교

이 프로젝트의 앙상블 공격은 LowKey(Cherepanova et al., ICLR 2021)의 핵심 원리(앙상블 공격 + 가우시안 스무딩 견고성)를 가져와 구현했지만, 세부적으로는 다음과 같은 차이가 있습니다.

| 항목 | LowKey (원 논문) | 본 프로젝트 |
|---|---|---|
| 앙상블 모델 수/다양성 | 4개, 서로 다른 아키텍처(IR-152, IR-50, ResNet-152, ResNet-50) × ArcFace/CosFace 헤드, MS-Celeb-1M으로 학습 | 2개, 같은 아키텍처(InceptionResnetV1) — 학습 데이터만 다름(vggface2 vs casia-webface) |
| 손실 함수 | 특징 공간 L2 거리 − LPIPS(지각적 유사도) 페널티 | 코사인 유사도 합 (LPIPS 없음) |
| 가우시안 블러 견고성 | σ=3, window=7 — 블러 적용/미적용 둘 다에 대해 공격 | σ=3.0, window=7 — 동일하게 구현 |
| 섭동 크기 제한 | 고정 epsilon 없음 — 사진마다 다른 크기로 수렴할 때까지 | epsilon으로 고정 제한 (보통 0.03~0.05) + `target_cosine` 조기종료(본 프로젝트만의 추가 기능) |
| 최적화 | signed gradient ascent, 50 iteration, lr 0.0025 | signed gradient descent, 유사한 구조 |
| 얼굴 검출/정렬 | 미분 가능한 얼굴 검출+정렬 파이프라인 (아무 크기/비율 사진 지원) | 별도 검출 파이프라인 없이 얼굴 영역을 고정 리사이즈 |
| 상용 API 검증 | Amazon Rekognition(순위-1 정확도 0.6%), Microsoft Azure(0.1%)까지 실제 검증 | 아직 미검증 (AWS Rekognition 테스트 예정) |
| PhotoGuard 결합 | 없음 (LowKey는 2021년 논문, PhotoGuard는 이후 별도 연구) | 본 프로젝트만의 추가 — LowKey에는 없는 조합 |

**전이성(transferability) 검증 결과**: 공격에 사용하지 않은 모델(VGG-Face, ArcFace, SFace)로 `deepface`를 이용해 교차 검증한 결과, SFace는 방어에 성공했지만 VGG-Face와 ArcFace는 여전히 동일인으로 인식했습니다. 이는 본 프로젝트의 앙상블이 LowKey보다 아키텍처 다양성이 부족한 것(1종류 vs 4종류)과 직접적으로 연관된 한계로 보입니다.

**요약**: 본 프로젝트는 LowKey의 핵심 원리를 차용하되, (1) PhotoGuard를 결합해 생성형 AI 편집 방어까지 확장했고, (2) 앙상블 다양성과 상용 API 검증은 리소스 제약으로 축소된 버전입니다.

참고 문헌: Cherepanova, V. et al. "LowKey: Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition." ICLR 2021. (https://arxiv.org/abs/2101.07922)
EOF
