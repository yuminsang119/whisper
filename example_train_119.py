#!/usr/bin/env python3
"""
119 신고 전화 Whisper 파인튜닝 - 간단한 예제
"""

import pandas as pd
from datasets import Dataset, Audio
from fine_tune_whisper import fine_tune_whisper

def main():
    print("=" * 70)
    print("119 신고 전화 Whisper 파인튜닝")
    print("=" * 70)

    # ========================================================================
    # 1단계: CSV 파일에서 데이터 로드
    # ========================================================================
    print("\n📂 1단계: 데이터 로드 중...")

    # train.csv 형식:
    # file_path,transcription
    # data/train/audio_001.wav,일일구입니다 무엇을 도와드릴까요
    # data/train/audio_002.wav,화재 신고 접수합니다

    try:
        train_df = pd.read_csv("data/train.csv")
        print(f"   ✓ 학습 데이터: {len(train_df)}개")
    except FileNotFoundError:
        print("   ✗ data/train.csv 파일을 찾을 수 없습니다!")
        print("\n   CSV 파일 생성 예제:")
        print("   --------------------------------------------------")
        print("   file_path,transcription")
        print("   data/train/audio_001.wav,일일구입니다")
        print("   data/train/audio_002.wav,화재 신고 접수합니다")
        print("   --------------------------------------------------")
        return

    # 테스트 데이터 (선택사항)
    test_dataset = None
    try:
        test_df = pd.read_csv("data/test.csv")
        print(f"   ✓ 테스트 데이터: {len(test_df)}개")

        test_dataset = Dataset.from_dict({
            "audio": test_df["file_path"].tolist(),
            "sentence": test_df["transcription"].tolist()
        })
        test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    except FileNotFoundError:
        print("   ⚠ data/test.csv 없음 (검증 없이 학습)")

    # ========================================================================
    # 2단계: Dataset 생성
    # ========================================================================
    print("\n🔄 2단계: Dataset 생성 중...")

    train_dataset = Dataset.from_dict({
        "audio": train_df["file_path"].tolist(),
        "sentence": train_df["transcription"].tolist()
    })

    # 오디오 파일 자동 로드 및 16kHz로 리샘플링
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(f"   ✓ Dataset 생성 완료")

    # ========================================================================
    # 3단계: 모델 선택
    # ========================================================================
    print("\n🤖 3단계: 모델 선택")
    print("   사용 가능한 모델:")
    print("   1. openai/whisper-tiny   (39M params, ~4GB VRAM)")
    print("   2. openai/whisper-base   (74M params, ~4GB VRAM)")
    print("   3. openai/whisper-small  (244M params, ~8GB VRAM) [추천]")
    print("   4. openai/whisper-medium (769M params, ~16GB VRAM)")

    # DMR 무전 음성용 모델 선택
    # VRAM 4GB → whisper-base, 8GB → whisper-small (권장), 16GB → whisper-medium
    model_name = "openai/whisper-small"
    print(f"   ✓ 선택된 모델: {model_name}")

    # ========================================================================
    # 4단계: 하이퍼파라미터 설정
    # ========================================================================
    print("\n⚙️  4단계: 학습 설정")

    # DMR 무전 음성 특화 하이퍼파라미터
    # - learning_rate를 낮게 설정: 압축/노이즈 아티팩트 과적합 방지
    # - OOM 발생 시: batch_size=2, fine_tune_whisper.py의 gradient_accumulation_steps=2
    hyperparameters = {
        "num_epochs": 10,          # 데이터 300개 미만이면 5로 낮추기
        "batch_size": 4,
        "learning_rate": 5e-6,     # 무전 노이즈 과적합 방지 (기본 1e-5보다 낮게)
        "warmup_steps": 200,
        "save_steps": 500,
    }

    print(f"   - Epochs: {hyperparameters['num_epochs']}")
    print(f"   - Batch Size: {hyperparameters['batch_size']}")
    print(f"   - Learning Rate: {hyperparameters['learning_rate']}")

    # ========================================================================
    # 5단계: 파인튜닝 시작
    # ========================================================================
    print("\n🚀 5단계: 파인튜닝 시작!")
    print("   (학습 중에는 TensorBoard로 모니터링 가능)")
    print("   명령어: tensorboard --logdir ./whisper-finetuned-dmr/runs")
    print()

    trainer = fine_tune_whisper(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model_name=model_name,
        output_dir="./whisper-finetuned-dmr",
        **hyperparameters
    )

    # ========================================================================
    # 6단계: 완료
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ 파인튜닝 완료!")
    print("=" * 70)
    print(f"\n📁 모델 저장 위치: ./whisper-finetuned-dmr")
    print("\n💡 실시간 STT 실행 방법:")
    print("   --------------------------------------------------")
    print("   python realtime_dmr_stt.py --model_path ./whisper-finetuned-dmr")
    print("   --------------------------------------------------")


if __name__ == "__main__":
    main()
