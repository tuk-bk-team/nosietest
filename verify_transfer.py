"""
verify_transfer.py
-------------------------------------------------------------
공격에 사용하지 않은(한 번도 안 본) 다른 얼굴인식 모델들로
"원본 vs 보호된 사진"이 같은 사람으로 인식되는지 확인합니다.
(LowKey/Fawkes 논문에서 말하는 "전이성(transferability)" 검증)
"""
from deepface import DeepFace

ORIGINAL = "original.jpeg"
PROTECTED = "protected.png"

# 공격에 쓴 facenet 계열 말고, 전혀 다른 아키텍처들로만 테스트
MODELS = ["VGG-Face", "ArcFace", "Dlib", "SFace"]

print(f"{'모델':<12} {'판정':<14} {'거리':<10} {'임계값'}")
print("-" * 50)
for model_name in MODELS:
    try:
        result = DeepFace.verify(
            img1_path=ORIGINAL,
            img2_path=PROTECTED,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",  # 섭동 때문에 얼굴 검출 실패할 수 있어서 꺼둠
        )
        verdict = "동일인 (방어실패)" if result["verified"] else "다른사람 (방어성공)"
        print(f"{model_name:<12} {verdict:<14} {result['distance']:<10.4f} {result['threshold']}")
    except Exception as e:
        print(f"{model_name:<12} 오류: {e}"); print(f"  -> 진짜 원인: {e.__cause__!r}")
