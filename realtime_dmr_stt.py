#!/usr/bin/env python3
"""
DMR 무전기 실시간 STT 파이프라인

사용법:
  python realtime_dmr_stt.py --model_path ./whisper-finetuned-dmr

장치 목록 확인:
  python -m sounddevice

의존성:
  pip install -r requirements_dmr.txt
"""

import argparse
import datetime
import queue
import sys
import threading
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice가 없습니다: pip install sounddevice")
    sys.exit(1)

try:
    from transformers import pipeline
except ImportError:
    print("transformers가 없습니다: pip install transformers")
    sys.exit(1)

import whisper
from whisper.audio import N_SAMPLES, pad_or_trim

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# 링버퍼: 항상 최근 30초를 보관하는 원형 버퍼
# ---------------------------------------------------------------------------

class RingBuffer:
    """
    고정 크기 원형 버퍼. 가득 차면 가장 오래된 샘플을 덮어씀.
    30초 = 480,000 샘플 @ 16kHz
    """

    def __init__(self, max_samples: int = N_SAMPLES):
        self._buf = np.zeros(max_samples, dtype=np.float32)
        self._max = max_samples
        self._write = 0
        self._count = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray) -> None:
        with self._lock:
            n = len(chunk)
            if n >= self._max:
                self._buf[:] = chunk[-self._max :]
                self._write = 0
                self._count = self._max
                return
            end = self._write + n
            if end <= self._max:
                self._buf[self._write : end] = chunk
            else:
                split = self._max - self._write
                self._buf[self._write :] = chunk[:split]
                self._buf[: n - split] = chunk[split:]
            self._write = end % self._max
            self._count = min(self._count + n, self._max)

    def get(self) -> np.ndarray:
        """누적된 오디오를 시간 순서대로 반환."""
        with self._lock:
            if self._count < self._max:
                return self._buf[: self._count].copy()
            # 원형 버퍼를 시간 순서로 재정렬
            return np.concatenate([self._buf[self._write :], self._buf[: self._write]])

    def clear(self) -> None:
        with self._lock:
            self._buf[:] = 0.0
            self._write = 0
            self._count = 0

    def duration_s(self) -> float:
        with self._lock:
            return self._count / SAMPLE_RATE


# ---------------------------------------------------------------------------
# 침묵 감지
# ---------------------------------------------------------------------------

def is_speech(chunk: np.ndarray, threshold_db: float = -40.0) -> bool:
    """
    RMS 에너지를 dB로 변환해 임계값과 비교.
    DMR은 PTT 해제 시 완전한 디지털 묵음이므로 단순 에너지 비교로 충분.
    """
    rms = np.sqrt(np.mean(chunk ** 2) + 1e-10)
    db = 20 * np.log10(rms)
    return db > threshold_db


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str | None = None):
    """
    fine_tune_whisper.py가 HuggingFace 포맷으로 저장하므로
    transformers.pipeline으로 로드.
    """
    import torch

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    print(f"모델 로드 중: {model_path}")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        generate_kwargs={
            "condition_on_previous_text": False,  # PTT마다 독립 발화
            "temperature": 0.0,                   # 결정적 디코딩 (빠름)
            "no_speech_threshold": 0.4,           # DMR 묵음은 진짜 묵음
            "compression_ratio_threshold": 2.0,   # 무전 반복 패턴 필터링
        },
    )
    print("모델 로드 완료")
    return pipe


# ---------------------------------------------------------------------------
# 청크 추론
# ---------------------------------------------------------------------------

def transcribe_chunk(
    pipe,
    audio: np.ndarray,
    language: str = "ko",
    initial_prompt: str | None = None,
) -> str:
    """
    pad_or_trim()으로 30초 패딩 후 pipeline 호출.
    파일 저장 없이 numpy 배열 직접 처리.
    """
    audio = pad_or_trim(audio.astype(np.float32), N_SAMPLES)
    result = pipe(
        {"raw": audio, "sampling_rate": SAMPLE_RATE},
        generate_kwargs={"language": language}
        | ({"initial_prompt": initial_prompt} if initial_prompt else {}),
    )
    return result["text"].strip()


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

def run(
    model_path: str = "./whisper-finetuned-dmr",
    input_device: int | str | None = None,
    language: str = "ko",
    initial_prompt: str | None = None,
    silence_threshold_db: float = -40.0,
    silence_trigger_s: float = 1.0,
    min_speech_s: float = 0.5,
    output_log: str | None = None,
) -> None:
    """
    메인 루프 (Ctrl+C로 종료).

    sounddevice → 20ms 블록 → is_speech() → RingBuffer
    PTT 해제(침묵) 감지 → transcribe_chunk() → print + 로그
    """
    pipe = load_model(model_path)

    buf = RingBuffer()
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    block_size = int(SAMPLE_RATE * 0.02)  # 20ms 블록
    silence_blocks_needed = int(silence_trigger_s / 0.02)

    log_file = open(output_log, "a", encoding="utf-8") if output_log else None

    def audio_callback(indata: np.ndarray, frames: int, time, status):
        if status:
            print(f"[오디오 경고] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    speech_active = False
    silence_count = 0

    print("\n실시간 STT 시작 (Ctrl+C로 종료)\n" + "-" * 40)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=block_size,
            device=input_device,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                chunk = audio_q.get()
                speaking = is_speech(chunk, silence_threshold_db)

                if speaking:
                    speech_active = True
                    silence_count = 0
                    buf.push(chunk)
                elif speech_active:
                    buf.push(chunk)
                    silence_count += 1

                    if silence_count >= silence_blocks_needed:
                        # PTT 해제 — 누적 오디오 추론
                        if buf.duration_s() >= min_speech_s:
                            audio_data = buf.get()
                            text = transcribe_chunk(
                                pipe, audio_data, language, initial_prompt
                            )
                            if text:
                                ts = datetime.datetime.now().strftime("%H:%M:%S")
                                line = f"[{ts}] {text}"
                                print(line)
                                if log_file:
                                    log_file.write(line + "\n")
                                    log_file.flush()

                        buf.clear()
                        speech_active = False
                        silence_count = 0

                # 30초 꽉 차면 강제 추론 (긴 발화 대응)
                if buf.duration_s() >= 29.5:
                    audio_data = buf.get()
                    text = transcribe_chunk(pipe, audio_data, language, initial_prompt)
                    if text:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        line = f"[{ts}] {text}"
                        print(line)
                        if log_file:
                            log_file.write(line + "\n")
                            log_file.flush()
                    buf.clear()
                    silence_count = 0

    except KeyboardInterrupt:
        print("\n종료합니다.")
    finally:
        if log_file:
            log_file.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DMR 무전기 실시간 STT")
    parser.add_argument(
        "--model_path",
        default="./whisper-finetuned-dmr",
        help="파인튜닝된 모델 경로 (기본: ./whisper-finetuned-dmr)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="오디오 입력 장치 번호 또는 이름 (기본: 시스템 기본값). "
             "'python -m sounddevice'로 목록 확인",
    )
    parser.add_argument("--language", default="ko", help="언어 코드 (기본: ko)")
    parser.add_argument(
        "--prompt",
        default=None,
        help="도메인 힌트 (예: '무전 교신. 콜사인, 출동코드, 지명 포함.')",
    )
    parser.add_argument(
        "--silence_db",
        type=float,
        default=-40.0,
        help="침묵 판단 임계값 dB (기본: -40). 잡음 환경에선 -35로 올리기",
    )
    parser.add_argument(
        "--silence_s",
        type=float,
        default=1.0,
        help="PTT 해제 판단 침묵 지속시간(초) (기본: 1.0)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="텍스트 로그 파일 경로 (선택)",
    )
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"모델을 찾을 수 없습니다: {args.model_path}")
        print("먼저 파인튜닝을 완료하세요: python example_train_119.py")
        sys.exit(1)

    run(
        model_path=args.model_path,
        input_device=args.device,
        language=args.language,
        initial_prompt=args.prompt,
        silence_threshold_db=args.silence_db,
        silence_trigger_s=args.silence_s,
        output_log=args.log,
    )


if __name__ == "__main__":
    main()
