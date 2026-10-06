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