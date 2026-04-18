#!/usr/bin/env python3
"""
무전 음성 데이터 정리 스크립트

사용법:
  python prepare_radio_data.py --data_dir ./raw_data --output_dir ./data

지원하는 입력 구조:
  1) 오디오 + 같은 이름의 .txt 파일 (자동 매칭)
       raw_data/001.wav  +  raw_data/001.txt
  2) 오디오 폴더 / 전사 폴더가 분리된 경우
       --audio_dir ./audio  --transcript_dir ./transcripts
  3) 기존 CSV / Excel (컬럼명이 달라도 매핑 가능)
       --existing_csv ./data.csv --audio_col 경로 --text_col 전사
  4) JSON 파일 (리스트 or 키-값)
       --existing_json ./data.json --audio_key audio --text_key text
  5) 오디오만 있는 경우 → 전사 없는 목록 생성
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus"}
TRANSCRIPT_EXTENSIONS = {".txt", ".lab"}


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
# 전사 파일 자동 탐색
# ---------------------------------------------------------------------------

def find_transcript_file(audio_path: Path, transcript_dir: Path | None = None) -> str:
    """
    오디오 경로로부터 전사 텍스트를 찾는다.
    1순위: 같은 폴더 내 같은 이름 .txt/.lab
    2순위: transcript_dir 안에서 같은 이름 탐색
    """
    # 같은 폴더
    for ext in TRANSCRIPT_EXTENSIONS:
        candidate = audio_path.with_suffix(ext)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()

    # 분리된 전사 폴더
    if transcript_dir and transcript_dir.exists():
        for ext in TRANSCRIPT_EXTENSIONS:
            candidate = transcript_dir / (audio_path.stem + ext)
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip()
        # 하위 폴더까지 탐색
        matches = list(transcript_dir.rglob(audio_path.stem + ".*"))
        for m in matches:
            if m.suffix.lower() in TRANSCRIPT_EXTENSIONS:
                return m.read_text(encoding="utf-8").strip()

    return ""


# ---------------------------------------------------------------------------
# 오디오 파일 수집
# ---------------------------------------------------------------------------

def collect_audio_files(data_dir: Path) -> list[Path]:
    """하위 폴더 포함 모든 오디오 파일 수집."""
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# 소스별 레코드 로더
# ---------------------------------------------------------------------------

def load_from_csv(csv_path: Path, audio_col: str, text_col: str) -> list[dict]:
    records = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        # 컬럼명 자동 추론 (대소문자 무시)
        def find_col(target: str) -> str:
            if target in headers:
                return target
            for h in headers:
                if h.lower() == target.lower():
                    return h
            return target  # 없으면 원래 이름 (KeyError 발생)

        real_audio = find_col(audio_col)
        real_text = find_col(text_col)
        for row in reader:
            records.append({
                "file_path": row.get(real_audio, "").strip(),
                "transcription": row.get(real_text, "").strip(),
            })
    return records


def load_from_excel(xlsx_path: Path, audio_col: str, text_col: str) -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        print("[오류] Excel 파일을 읽으려면 openpyxl이 필요합니다: pip install openpyxl")
        sys.exit(1)
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    headers = [str(c.value).strip() for c in next(ws.iter_rows(min_row=1, max_row=1))]
    audio_idx = headers.index(audio_col) if audio_col in headers else 0
    text_idx = headers.index(text_col) if text_col in headers else 1
    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        records.append({
            "file_path": str(row[audio_idx] or "").strip(),
            "transcription": str(row[text_idx] or "").strip(),
        })
    return records


def load_from_json(json_path: Path, audio_key: str, text_key: str) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # {"audio": [...], "text": [...]} 형식
        audio_list = data.get(audio_key, [])
        text_list = data.get(text_key, [])
        return [
            {"file_path": str(a), "transcription": str(t)}
            for a, t in zip(audio_list, text_list)
        ]
    # [{"audio": "...", "text": "..."}, ...] 형식
    return [
        {"file_path": str(item.get(audio_key, "")), "transcription": str(item.get(text_key, ""))}
        for item in data
    ]


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def validate(records: list[dict]) -> dict:
    stats = {
        "total": len(records),
        "has_transcript": 0,
        "missing_transcript": 0,
        "missing_audio": 0,
        "issues": [],
    }
    for r in records:
        audio_ok = Path(r["file_path"]).exists() if r["file_path"] else False
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
# CSV 쓰기
# ---------------------------------------------------------------------------

def write_csv(path: Path, data: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "transcription"])
        writer.writeheader()
        writer.writerows(data)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="무전 음성 데이터 정리 — 다양한 형식 자동 지원"
    )
    # 폴더 탐색 모드
    parser.add_argument("--data_dir", help="오디오(+전사) 원본 폴더")
    parser.add_argument("--audio_dir", help="오디오 전용 폴더 (전사 폴더와 분리된 경우)")
    parser.add_argument("--transcript_dir", help="전사 파일(.txt/.lab) 폴더")
    # 기존 파일 모드
    parser.add_argument("--existing_csv", help="기존 CSV 파일 경로")
    parser.add_argument("--existing_excel", help="기존 Excel(.xlsx) 파일 경로")
    parser.add_argument("--existing_json", help="기존 JSON 파일 경로")
    # 컬럼/키 매핑
    parser.add_argument("--audio_col", default="file_path", help="오디오 경로 컬럼명")
    parser.add_argument("--text_col", default="transcription", help="전사 텍스트 컬럼명")
    parser.add_argument("--audio_key", default="audio", help="JSON 오디오 키")
    parser.add_argument("--text_key", default="text", help="JSON 전사 키")
    # 공통 옵션
    parser.add_argument("--output_dir", default="./data", help="출력 폴더 (기본: ./data)")
    parser.add_argument("--convert_wav", action="store_true", help="16kHz WAV로 변환 (ffmpeg)")
    parser.add_argument("--split", type=float, default=0.1, help="검증셋 비율 (기본: 0.1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # ── 소스 선택 ────────────────────────────────────────────────────────
    if args.existing_csv:
        print(f"CSV 로드: {args.existing_csv}")
        records = load_from_csv(Path(args.existing_csv), args.audio_col, args.text_col)

    elif args.existing_excel:
        print(f"Excel 로드: {args.existing_excel}")
        records = load_from_excel(Path(args.existing_excel), args.audio_col, args.text_col)

    elif args.existing_json:
        print(f"JSON 로드: {args.existing_json}")
        records = load_from_json(Path(args.existing_json), args.audio_key, args.text_key)

    else:
        # 폴더 탐색 모드
        audio_root = Path(args.audio_dir) if args.audio_dir else Path(args.data_dir) if args.data_dir else None
        transcript_dir = Path(args.transcript_dir) if args.transcript_dir else None

        if audio_root is None:
            parser.error("--data_dir 또는 --audio_dir 중 하나는 반드시 지정해야 합니다.")

        print(f"오디오 파일 탐색: {audio_root}")
        audio_files = collect_audio_files(audio_root)
        print(f"  → {len(audio_files)}개 파일 발견")

        if not audio_files:
            print("오디오 파일을 찾지 못했습니다. 경로를 확인하세요.")
            sys.exit(1)

        for audio_path in audio_files:
            final_path = audio_path

            if args.convert_wav and audio_path.suffix.lower() != ".wav":
                rel = audio_path.relative_to(audio_root)
                wav_path = output_dir / "audio" / rel.with_suffix(".wav")
                print(f"  변환: {audio_path.name} → {wav_path.name}", end=" ")
                ok = convert_to_wav(audio_path, wav_path)
                print("✓" if ok else "[실패, 원본 사용]")
                final_path = wav_path if ok else audio_path

            transcript = find_transcript_file(audio_path, transcript_dir)
            records.append({
                "file_path": str(final_path.resolve()),
                "transcription": transcript,
            })

    print(f"  → 총 {len(records)}개 레코드")

    # ── 검증 ─────────────────────────────────────────────────────────────
    print("\n[검증 결과]")
    stats = validate(records)
    print(f"  전체:             {stats['total']}개")
    print(f"  전사 있음:        {stats['has_transcript']}개")
    print(f"  전사 없음:        {stats['missing_transcript']}개")
    print(f"  오디오 파일 없음: {stats['missing_audio']}개")

    if stats["issues"]:
        print("\n  문제 파일 (처음 10개):")
        for issue in stats["issues"][:10]:
            print(f"    {issue}")

    if stats["missing_transcript"] > 0:
        missing_path = output_dir / "missing_transcripts.txt"
        with open(missing_path, "w", encoding="utf-8") as f:
            for r in records:
                if not r["transcription"].strip():
                    f.write(r["file_path"] + "\n")
        print(f"\n[주의] 전사 없는 파일 목록 → {missing_path}")

    # ── 분할 & 저장 ────────────────────────────────────────────────────
    labeled = [r for r in records if r["transcription"].strip()]
    unlabeled = [r for r in records if not r["transcription"].strip()]

    print("\n[출력]")

    if labeled:
        import random
        random.seed(42)
        random.shuffle(labeled)
        split_idx = max(1, int(len(labeled) * (1 - args.split)))
        train_records = labeled[:split_idx]
        test_records = labeled[split_idx:]

        train_csv = output_dir / "train.csv"
        write_csv(train_csv, train_records)
        print(f"  학습 데이터 ({len(train_records)}개) → {train_csv}")

        if test_records:
            test_csv = output_dir / "test.csv"
            write_csv(test_csv, test_records)
            print(f"  검증 데이터 ({len(test_records)}개) → {test_csv}")

    if unlabeled:
        unlabeled_csv = output_dir / "unlabeled.csv"
        write_csv(unlabeled_csv, unlabeled)
        print(f"  전사 없는 파일 ({len(unlabeled)}개) → {unlabeled_csv}")

    all_csv = output_dir / "all_records.csv"
    write_csv(all_csv, records)
    print(f"  전체 목록 → {all_csv}")

    print("\n완료!")
    if labeled:
        print("\n다음 단계:")
        print("  python example_train_119.py")
        print("  (data/train.csv 와 data/test.csv 가 자동으로 사용됩니다)")


if __name__ == "__main__":
    main()
