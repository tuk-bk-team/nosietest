"""
app/models/face_models.py
--------------------------------------------------------------------------
"앙상블 공격"에 쓸 얼굴 인식 모델들을 불러오는 곳입니다.

LowKey 논문(3장)의 핵심 아이디어: 모델 1개만 공격하면 "그 모델에만" 통하는
섭동이 나올 수 있습니다. 서로 다른 데이터셋/가중치로 학습된 모델 여러 개를
동시에 공격하면, "한 번도 본 적 없는" 다른 얼굴 인식 시스템(상용 API 등)에도
어느 정도 전이(transfer)가 잘 되는 섭동을 만들 수 있습니다.

지금 추가한 2개 모델:
  1) FaceNet (InceptionResnetV1), VGGFace2로 학습
  2) FaceNet (InceptionResnetV1), CASIA-WebFace로 학습

둘 다 구조(InceptionResnetV1)는 같지만, "어떤 사진들을 보고 학습했는지"가
달라서 얼굴을 보는 기준이 조금씩 다릅니다 (완전히 다른 아키텍처는 아니지만,
이 환경에서 네트워크 제약 없이 안정적으로 받을 수 있는 조합입니다 - 둘 다
facenet-pytorch가 GitHub 릴리즈에서 받아오기 때문에 다운로드가 막힐 걱정이
없습니다). 논문처럼 ResNet/IR + ArcFace/CosFace 조합까지 확장하고 싶다면,
아래 FACE_MODEL_SPECS 리스트에 항목만 추가하면 됩니다 (구조 자체는 이미
"여러 모델 리스트"를 기준으로 동작하도록 만들어 뒀습니다).
"""

from dataclasses import dataclass
from typing import List

import torch
from facenet_pytorch import InceptionResnetV1


@dataclass
class FaceModel:
    """앙상블에 들어가는 모델 1개를 감싸는 작은 래퍼입니다."""
    name: str          # 로그/결과에 표시할 이름
    model: torch.nn.Module


# 여기에 항목을 추가하면 앙상블에 모델이 하나 더 들어갑니다.
# (예: 나중에 ArcFace/CosFace 계열 모델을 구할 수 있으면 여기 추가)
FACE_MODEL_SPECS = [
    {"name": "facenet-vggface2", "pretrained": "vggface2"},
    {"name": "facenet-casia-webface", "pretrained": "casia-webface"},
]


def load_face_ensemble(device: str = "cpu") -> List[FaceModel]:
    """FACE_MODEL_SPECS에 정의된 모델들을 전부 불러와서 리스트로 반환합니다.

    각 모델은 eval() 모드 + 가중치 고정(requires_grad_(False)) 상태로
    반환됩니다 - 우리가 최적화할 대상은 "모델"이 아니라 "이미지 픽셀"이기
    때문입니다 (지금까지와 동일한 원칙).
    """
    ensemble: List[FaceModel] = []
    for spec in FACE_MODEL_SPECS:
        print(f"  - {spec['name']} 모델 불러오는 중 (최초 1회는 다운로드 때문에 시간이 걸릴 수 있습니다)...")
        model = InceptionResnetV1(pretrained=spec["pretrained"]).eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        ensemble.append(FaceModel(name=spec["name"], model=model))
    return ensemble
