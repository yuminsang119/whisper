#!/usr/bin/env python3
"""
무전 음성 데이터 정리 스크립트

사용법:
  python prepare_radio_data.py --data_dir ./raw_data --output_dir ./data

지원하는 입력 구조:
  1) 오디오 + 같은 이름의 .txt 파일
       raw_data/001.wav  +  raw_data/001.txt
  2) 오디오만 있는 경우 (전사 없음) → 파일 목록만 생성
  3) 기존 CSV (컬럼명이 달라도 매핑 가능)
  4) 폴더별 분류된 구조
       raw_data/class_A/001.wav
       raw_data/class_B/002.wav
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus"}


# ---------------------------------------------------------------------------
# 오디오 변환 (ffmpeg)
# ---------------------------------------------------------------------------

def convert_to_wav(src: Path, dst: Path) -> bool:
    """ffmpeg으로 16kHz 모노 WAV 변환. 성공 여부 반환."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(dst),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# 전사 텍스트 찾기
# ---------------------------------------------------------------------------

def find_transcript(audio_path: Path) -> str:
    """오디오 파일과 같은 이름의 .txt 파일에서 전사 텍스트를 읽는다."""
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        return text
    return ""


# ---------------------------------------------------------------------------
# 데이터 탐색
# ---------------------------------------------------------------------------

def collect_audio_files(data_dir: Path) -> list[Path]:
    """하위 폴더 포함 모든 오디오 파일 수집."""
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def validate(records: list[dict]) -> dict:
    """데이터 통계 및 문제 파일 보고."""
    stats = {
        "total": len(records),
        "has_transcript": 0,
        "missing_transcript": 0,
        "missing_audio": 0,
        "issues": [],
    }
    for r in records:
        audio_ok = Path(r["file_path"]).exists()
        has_text = bool(r["transcription"].strip())

        if not audio_ok:
            stats["missing_audio"] += 1
            stats["issues"].append(f"[파일 없음] {r['file_path']}")
        if has_text:
            stats["has_transcript"] += 1
        else:
            stats["missing_transcript"] += 1

    return stats


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="무전 음성 데이터 정리")
    parser.add_argument("--data_dir", required=True, help="원본 데이터 폴더")
    parser.add_argument("--output_dir", default="./data", help="출력 폴더 (기본: ./data)")
    parser.add_argument(
        "--convert_wav",
        action="store_true",
        help="모든 오디오를 16kHz WAV로 변환 (ffmpeg 필요)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.1,
        help="검증셋 비율 (기본: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--existing_csv",
        help="기존 CSV 파일 경로 (컬럼 재매핑 시 사용)",
    )
    parser.add_argument(
        "--audio_col",
        default="file_path",
        help="기존 CSV의 오디오 경로 컬럼명 (기본: file_path)",
    )
    parser.add_argument(
        "--text_col",
        default="transcription",
        help="기존 CSV의 전사 텍스트 컬럼명 (기본: transcription)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # ------------------------------------------------------------------
    # 경로 1: 기존 CSV 재매핑
    # ------------------------------------------------------------------
    if args.existing_csv:
        print(f"기존 CSV 로드: {args.existing_csv}")
        with open(args.existing_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "file_path": row[args.audio_col],
                    "transcription": row[args.text_col],
                })
        print(f"  → {len(records)}개 레코드 로드 완료")

    # ------------------------------------------------------------------
    # 경로 2: 폴더 탐색
    # ------------------------------------------------------------------
    else:
        print(f"오디오 파일 탐색 중: {data_dir}")
        audio_files = collect_audio_files(data_dir)
        print(f"  → {len(audio_files)}개 파일 발견")

        if not audio_files:
            print("오디오 파일을 찾지 못했습니다. --data_dir 경로를 확인하세요.")
            sys.exit(1)

        for audio_path in audio_files:
            # WAV 변환
            if args.convert_wav and audio_path.suffix.lower() != ".wav":
                rel = audio_path.relative_to(data_dir)
                wav_path = output_dir / "audio" / rel.with_suffix(".wav")
                print(f"  변환: {audio_path.name} → {wav_path.name}")
                if not convert_to_wav(audio_path, wav_path):
                    print(f"    [경고] 변환 실패: {audio_path}")
                    wav_path = audio_path  # 원본 사용
                final_path = wav_path
            else:
                final_path = audio_path

            transcript = find_transcript(audio_path)
            records.append({
                "file_path": str(final_path),
                "transcription": transcript,
            })

    # ------------------------------------------------------------------
    # 검증 보고
    # ------------------------------------------------------------------
    print("\n[검증 결과]")
    stats = validate(records)
    print(f"  전체 파일 수:         {stats['total']}")
    print(f"  전사 있음:            {stats['has_transcript']}")
    print(f"  전사 없음 (빈 칸):    {stats['missing_transcript']}")
    print(f"  오디오 파일 없음:     {stats['missing_audio']}")

    if stats["issues"]:
        print("\n  문제 파일 (처음 10개):")
        for issue in stats["issues"][:10]:
            print(f"    {issue}")

    if stats["missing_transcript"] > 0:
        print(
            f"\n[주의] {stats['missing_transcript']}개 파일에 전사 텍스트가 없습니다.\n"
            f"  전사 없는 파일 목록 → {output_dir}/missing_transcripts.txt"
        )
        missing_path = output_dir / "missing_transcripts.txt"
        with open(missing_path, "w", encoding="utf-8") as f:
            for r in records:
                if not r["transcription"].strip():
                    f.write(r["file_path"] + "\n")

    # ------------------------------------------------------------------
    # 전사 있는 데이터만 분리 → train / test 분할
    # ------------------------------------------------------------------
    labeled = [r for r in records if r["transcription"].strip()]
    unlabeled = [r for r in records if not r["transcription"].strip()]

    if labeled:
        import random
        random.seed(42)
        random.shuffle(labeled)

        split_idx = max(1, int(len(labeled) * (1 - args.split)))
        train_records = labeled[:split_idx]
        test_records = labeled[split_idx:]

        def write_csv(path: Path, data: list[dict]):
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["file_path", "transcription"])
                writer.writeheader()
                writer.writerows(data)

        train_csv = output_dir / "train.csv"
        write_csv(train_csv, train_records)
        print(f"\n[출력]")
        print(f"  학습 데이터 ({len(train_records)}개) → {train_csv}")

        if test_records:
            test_csv = output_dir / "test.csv"
            write_csv(test_csv, test_records)
            print(f"  검증 데이터 ({len(test_records)}개) → {test_csv}")

    if unlabeled:
        unlabeled_csv = output_dir / "unlabeled.csv"
        with open(unlabeled_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "transcription"])
            writer.writeheader()
            writer.writerows(unlabeled)
        print(f"  전사 없는 파일 ({len(unlabeled)}개) → {unlabeled_csv}")

    # ------------------------------------------------------------------
    # 전체 목록 (원시 데이터 백업)
    # ------------------------------------------------------------------
    all_csv = output_dir / "all_records.csv"
    with open(all_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcription"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  전체 목록            → {all_csv}")

    print("\n완료!")
    if stats["missing_transcript"] > 0:
        print(
            "\n다음 단계: missing_transcripts.txt의 파일들에 전사를 추가한 뒤\n"
            "다시 실행하거나 unlabeled.csv를 직접 편집하세요."
        )


if __name__ == "__main__":
    main()
