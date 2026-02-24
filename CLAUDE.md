# CLAUDE.md — AI Assistant Guide for openai-whisper

This file provides context for AI assistants (like Claude) working in this repository. It covers the codebase structure, development workflows, conventions, and key patterns.

---

## Project Overview

This is a fork of [OpenAI Whisper](https://github.com/openai/whisper), a general-purpose speech recognition model. The fork adds:

- Korean fine-tuning support (for 119 emergency call transcription)
- Fine-tuning scripts and documentation (`fine_tune_whisper.py`, `example_train_119.py`, `FINETUNING_GUIDE.md`)

**Core capability**: Multilingual automatic speech recognition (ASR) and speech-to-text translation across 98 languages, using a transformer-based encoder-decoder architecture trained on large-scale weakly-supervised data.

---

## Repository Structure

```
whisper/                     # Main Python package
├── __init__.py              # Public API: load_model(), available_models()
├── __main__.py              # CLI entry point (python -m whisper)
├── version.py               # Package version string
├── model.py                 # Transformer architecture (encoder + decoder)
├── audio.py                 # Audio loading, mel-spectrogram extraction
├── transcribe.py            # Main transcription pipeline + CLI
├── decoding.py              # Beam search / greedy decoding, language detection
├── tokenizer.py             # Tiktoken-based tokenizer, 98-language support
├── timing.py                # Word-level timestamp extraction via DTW
├── utils.py                 # Compression ratio, output writers (SRT/VTT/JSON/TXT)
├── triton_ops.py            # Triton CUDA kernels (DTW, median filter)
└── normalizers/
    ├── basic.py             # Unicode normalization, diacritic removal
    └── english.py           # English-specific: numbers, currency, dates

tests/                       # pytest test suite
├── conftest.py              # Fixtures (random seed, cuda marker)
├── test_audio.py
├── test_transcribe.py
├── test_tokenizer.py
├── test_normalizer.py
├── test_timing.py
└── jfk.flac                 # Reference audio: JFK "Ask Not" speech

notebooks/                   # Jupyter notebooks (usage examples)
data/                        # Dataset documentation (README.md only)
.github/workflows/           # CI: test.yml, python-publish.yml

# Fine-tuning additions (Korean / 119 emergency calls)
fine_tune_whisper.py
example_train_119.py
FINETUNING_GUIDE.md
requirements_finetuning.txt
```

---

## Architecture

### Model Design

Whisper follows a standard transformer encoder-decoder sequence-to-sequence architecture:

- **AudioEncoder** (`model.py`): Conv1d front-end → positional embeddings → N transformer layers. Processes log-mel spectrograms.
- **TextDecoder** (`model.py`): Auto-regressive transformer. Attends to encoder hidden states via cross-attention. Predicts tokens including special control tokens (language, task, timestamps).
- **Multitask via special tokens**: Language ID tokens (`<|en|>`, `<|ko|>`, …), task tokens (`<|transcribe|>`, `<|translate|>`), and timestamp tokens are part of the vocabulary — no separate classifiers needed.

### Key Constants (`audio.py`)

```python
SAMPLE_RATE   = 16000   # Hz
CHUNK_LENGTH  = 30      # seconds per chunk
N_FFT         = 400
HOP_LENGTH    = 160
N_FRAMES      = 3000    # mel frames per 30s chunk
```

### Decoding Strategy (`decoding.py`, `transcribe.py`)

1. Audio is chunked into 30-second windows with sliding context.
2. Beam search (or greedy) decoding generates tokens.
3. Quality checks filter bad segments:
   - **Compression ratio**: gzip compression of output text; high ratio = probable repetition/hallucination.
   - **Log-probability threshold**: low avg logprob → low confidence.
   - **No-speech probability**: model's own estimate.
4. On failure, temperature is increased (0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0) until a segment passes or max temperature is reached.

---

## Available Models

| Model      | Parameters | Notes                         |
|------------|-----------|-------------------------------|
| tiny       | 39M       | Fastest, lowest accuracy      |
| tiny.en    | 39M       | English-only variant          |
| base       | 74M       |                               |
| base.en    | 74M       | English-only                  |
| small      | 244M      |                               |
| small.en   | 244M      | English-only                  |
| medium     | 769M      |                               |
| medium.en  | 769M      | English-only                  |
| large-v1   | 1.55B     |                               |
| large-v2   | 1.55B     | Dec 2022                      |
| large-v3   | 1.55B     | Nov 2023                      |
| large      | 1.55B     | Alias for large-v3            |
| turbo      | 809M      | Sept 2024, recommended default|
| large-v3-turbo | 809M  | Alias for turbo               |

Models are downloaded automatically from Azure Blob Storage on first use.

---

## Python API

```python
import whisper

# High-level transcription
model = whisper.load_model("turbo")              # or "base", "small", etc.
result = model.transcribe("audio.mp3")
print(result["text"])

# With options
result = model.transcribe(
    "audio.mp3",
    language="ko",           # force language (skip detection)
    task="translate",        # "transcribe" (default) or "translate" to English
    word_timestamps=True,    # extract per-word timestamps
    verbose=True,
)

# Low-level API
audio = whisper.load_audio("audio.mp3")          # returns float32 numpy array at 16kHz
audio = whisper.pad_or_trim(audio)               # pad/trim to 30s chunk
mel   = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)            # {lang_code: probability}
options  = whisper.DecodingOptions(language="en", fp16=False)
result   = whisper.decode(model, mel, options)   # DecodingResult
```

### Transcription Result Schema

```python
{
  "text": str,              # full concatenated transcript
  "language": str,          # detected/forced language code
  "segments": [
    {
      "id": int,
      "seek": int,          # position in audio (centiseconds)
      "start": float,       # segment start time (seconds)
      "end": float,
      "text": str,
      "tokens": list[int],
      "temperature": float,
      "avg_logprob": float,
      "compression_ratio": float,
      "no_speech_prob": float,
      "words": [            # only when word_timestamps=True
        {"word": str, "start": float, "end": float, "probability": float}
      ]
    }
  ]
}
```

---

## CLI Usage

```bash
# Basic transcription (auto-detects language)
whisper audio.mp3

# Specify model and language
whisper audio.mp3 --model turbo --language Korean

# Translate to English
whisper audio.mp3 --model medium --task translate

# Output formats: txt, srt, vtt, tsv, json (comma-separated for multiple)
whisper audio.mp3 --output_format srt --output_dir ./output

# Word-level timestamps
whisper audio.mp3 --word_timestamps True

# Process multiple files
whisper audio1.mp3 audio2.wav audio3.flac --model base
```

---

## Development Setup

### Requirements

- Python 3.8–3.13
- PyTorch (1.10.1+, recommend 2.x)
- ffmpeg installed and on PATH (used for audio decoding)

```bash
# Install for development
pip install -e ".[dev]"

# Or install with fine-tuning extras
pip install -r requirements_finetuning.txt
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Tools configured:
- **black** (24.10.0) — code formatting, line length 88
- **isort** (5.13.2) — import sorting, black-compatible profile
- **flake8** (7.1.1) — linting; ignores E203, E501, W503, W504

Always run `pre-commit run --all-files` before committing.

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_tokenizer.py -v

# Skip CUDA tests (in CPU-only environments)
pytest tests/ -m "not requires_cuda"
```

### Test Fixtures

- `tests/jfk.flac` — Reference audio for transcription tests (JFK speech). Must not be modified or deleted.
- Random seed is set to 42 globally in `conftest.py`.
- `@pytest.mark.requires_cuda` marks tests that need a GPU.

### CI Matrix

The GitHub Actions pipeline (`test.yml`) tests across:
- Python: 3.8, 3.9, 3.10, 3.11, 3.12
- PyTorch: 1.10.1, 1.11.0, 1.12.1, 1.13.1, 2.0.0, 2.1.0, 2.2.0, 2.3.0, 2.4.0, 2.5.0

CUDA tests are excluded from CI (no GPU runners). GPU-specific functionality (Triton kernels) is tested manually.

---

## Code Conventions

### Style

- **Black** for formatting (88 char line length). No manual style decisions needed.
- **isort** for import order. Always group: stdlib → third-party → local.
- Type annotations throughout. Use them for new code.
- Dataclasses (`@dataclass`) for configuration objects (see `DecodingOptions`, `ModelDimensions`).

### Naming

- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Module-level config: prefer dataclasses over dicts for structured options

### Performance Patterns

- **Triton kernels** (`triton_ops.py`): Used for DTW and median filter on GPU. Gracefully falls back to CPU/numba implementations when Triton is unavailable.
- **SDPA** (Scaled Dot-Product Attention): Used when `torch.backends.cuda.flash_sdp_enabled()`. Use `model.disable_sdpa()` context manager to force standard attention.
- **KV caching**: Decoder caches key/value tensors in `MultiHeadAttention`. Set `kv_cache` tensor externally via hooks.
- **fp16 inference**: Default on CUDA. Pass `fp16=False` to `DecodingOptions` for CPU or debugging.

### Adding New Languages

Language support is configured in `tokenizer.py` via the `LANGUAGES` dict (ISO 639-1 codes → language names). Tokenizer token IDs for each language are implicit in the tiktoken BPE vocabulary — no changes needed for languages already in the training vocab.

### Adding Output Formats

Output format writers live in `utils.py`. Implement a class inheriting `ResultWriter` and register it in `get_writer()`.

---

## Fine-Tuning (Korean 119 Emergency Calls)

The repository includes tooling to fine-tune Whisper on domain-specific Korean data.

### Data Format

CSV file with columns:
```
audio_path,transcription
/path/to/audio.wav,텍스트 전사 내용
```

Audio: WAV files, 16kHz mono recommended.

### Training

```bash
# Install fine-tuning dependencies
pip install -r requirements_finetuning.txt

# Run fine-tuning
python fine_tune_whisper.py \
  --model_size small \
  --data_path ./data/train.csv \
  --output_dir ./output/whisper-small-ko-119 \
  --num_epochs 10 \
  --batch_size 8

# Example script for 119 call data
python example_train_119.py
```

### VRAM Requirements

| Model  | Minimum VRAM |
|--------|-------------|
| tiny   | 4GB         |
| base   | 4GB         |
| small  | 8GB         |
| medium | 16GB        |
| large  | 24GB        |

See `FINETUNING_GUIDE.md` for full details.

---

## Release Process

1. Update version string in `whisper/version.py`.
2. Update `CHANGELOG.md`.
3. Commit with message `Release vYYYYMMDD`.
4. The `python-publish.yml` workflow triggers on commits matching `Release [version]` and publishes to PyPI.

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `whisper/__init__.py` | Public API, model URLs, `load_model()` |
| `whisper/model.py` | Full transformer architecture |
| `whisper/audio.py` | Audio I/O and mel-spectrogram |
| `whisper/transcribe.py` | End-to-end pipeline and CLI |
| `whisper/decoding.py` | Beam search, language detection |
| `whisper/tokenizer.py` | BPE tokenizer, language registry |
| `whisper/timing.py` | Word-level timestamp extraction |
| `whisper/utils.py` | Output writers, text utilities |
| `whisper/normalizers/` | Text normalization for WER evaluation |
| `whisper/triton_ops.py` | CUDA kernels (Triton) |
| `tests/jfk.flac` | Reference audio for tests |
| `pyproject.toml` | Build config, dependencies, tool settings |
| `.pre-commit-config.yaml` | Linting and formatting hooks |
| `FINETUNING_GUIDE.md` | Korean fine-tuning guide |

---

## Common Pitfalls

- **ffmpeg must be on PATH** — `load_audio()` shells out to ffmpeg. Missing ffmpeg causes a runtime error, not an import error.
- **Triton only on x86_64 Linux with CUDA** — The `triton>=2` dependency is conditional. Don't assume Triton is available; the code has CPU fallbacks.
- **`pad_or_trim` is required before `log_mel_spectrogram`** — The mel encoder expects exactly `N_FRAMES` (3000) frames. Raw audio of arbitrary length must be padded/trimmed first.
- **`.en` models don't support `task=translate`** — English-only models lack the translation capability of multilingual models.
- **Word timestamps require alignment heads** — Only models with known alignment head metadata (stored in `__init__.py`) support `word_timestamps=True`. Check `whisper.get_alignment_heads(model)`.
- **Temperature fallback changes outputs non-deterministically** — If a segment fails quality checks, the model retries with higher temperature. Tests should not assume bitwise-identical outputs across runs unless temperature is fixed at 0.0.
