# CLAUDE.md

## Project Overview

OpenAI Whisper - a general-purpose speech recognition model. Supports multilingual transcription (99+ languages), speech translation, language identification, and voice activity detection. This fork includes fine-tuning scripts for Korean 119 emergency call data.

## Repository Structure

```
whisper/              # Main Python package
  __init__.py         # Public API: load_model(), transcribe(), decode()
  model.py            # Whisper model (Transformer encoder-decoder, PyTorch)
  transcribe.py       # Main transcription logic (30-sec sliding window)
  decoding.py         # Beam search decoding, language detection
  audio.py            # Audio loading (via FFmpeg), mel spectrogram
  tokenizer.py        # Multilingual BPE tokenizer (tiktoken)
  timing.py           # Word-level timestamp calculations
  utils.py            # Output format writers, math helpers
  assets/             # Tokenizer files, mel filter weights
  normalizers/        # Language-specific text normalization
tests/                # pytest test suite
notebooks/            # Jupyter examples (LibriSpeech, Multilingual ASR)
data/                 # Evaluation dataset docs
.github/workflows/    # CI (test matrix) and release (PyPI publish)
```

## Development Setup

```bash
pip install -e ".[dev]"
# Or manually:
pip install -r requirements.txt
pip install pytest black flake8 isort
```

Requires FFmpeg installed on the system for audio processing.

## Common Commands

### Running Tests

```bash
# Full test suite (requires GPU for some tests)
pytest

# CI subset (tiny models only, no CUDA)
pytest --durations=0 -vv -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' -m 'not requires_cuda'

# Single test file
pytest tests/test_audio.py -v
```

### Code Formatting & Linting

```bash
# Format code
black .
isort .

# Lint
flake8

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running Whisper

```bash
# CLI
whisper audio.mp3 --model turbo --language en

# Python
import whisper
model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
```

## Code Style Conventions

- **Formatter**: Black (line length 88)
- **Import sorting**: isort (profile: black)
- **Linter**: Flake8 (max line length 88, ignores E203/E501/W503/W504)
- **Pre-commit hooks** enforce all three automatically
- Python 3.8+ compatibility required (supports up to 3.13)
- No type annotations enforced (no mypy)

## Architecture

- **Model**: Transformer encoder-decoder trained on 680k hours of labeled audio
- **Audio pipeline**: 16kHz mono -> 30-sec chunks -> 80-bin log-mel spectrogram (400 FFT, 160 hop)
- **Tokenization**: tiktoken BPE with special tokens for tasks/languages
- **Decoding**: Temperature fallback (0.0 -> 0.2 -> 0.4 -> ... -> 1.0), beam search, compression ratio filtering
- **GPU optimization**: Triton kernels, PyTorch SDPA attention

## Available Models

| Model | Parameters | English-only | Multilingual |
|-------|-----------|:---:|:---:|
| tiny | 39M | tiny.en | tiny |
| base | 74M | base.en | base |
| small | 244M | small.en | small |
| medium | 769M | medium.en | medium |
| large | 1550M | - | large-v1/v2/v3 |
| turbo | 809M | - | turbo |

## Testing Notes

- Test audio fixture: `tests/jfk.flac` (JFK speech excerpt)
- `@pytest.mark.requires_cuda` marks GPU-only tests
- CI runs a matrix of Python 3.8-3.12 x PyTorch 1.10-2.5
- Model downloads happen on first use; CI only tests tiny/tiny.en

## Key Dependencies

- `torch` (>=1.10) - model inference
- `tiktoken` - fast BPE tokenizer
- `numba` - JIT-compiled DTW alignment
- `triton` (>=2.0, Linux x86_64) - GPU kernel optimization
- `torchaudio` - audio processing
- `ffmpeg` (system) - audio decoding

## Fine-tuning (Fork Addition)

This fork adds HuggingFace-based fine-tuning support:
- `fine_tune_whisper.py` - Template script
- `example_train_119.py` - Korean 119 emergency call example
- `FINETUNING_GUIDE.md` - Korean documentation
- Extra deps: `requirements_finetuning.txt`
