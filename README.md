# subtitler

Generate and translate subtitles from any audio or video file — powered by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
[deep-translator](https://github.com/nidhaloff/deep-translator).

## Features

- **Fast transcription** — up to 4× faster than OpenAI Whisper with the same
  accuracy, using CTranslate2.
- **Automatic language detection** — or specify the source language manually.
- **Translation** — translate subtitles to 100+ languages via Google Translate
  (free, no API key).
- **Multiple output formats** — SRT and WebVTT.
- **VAD filtering** — Silero VAD removes silence and reduces hallucination.
- **Model choice** — from `tiny` (fast, less accurate) to `large-v3`
  (slow, most accurate).
- **CPU & GPU** — works on both, with int8 quantisation for low-memory setups.
- **Packagable as .exe** — single-file Windows executable via PyInstaller.

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/simoneraffaelli/subtitler.git
cd subtitler
pip install -e ".[dev]"
```

### From PyPI (once published)

```bash
pip install subtitler
```

## Quick start

```bash
# Transcribe a video and generate subtitles (auto-detect language)
subtitler video.mp4

# Use a specific model and output format
subtitler video.mp4 -m large-v3 -f vtt

# Transcribe and translate to Italian
subtitler video.mp4 -t it

# Specify source language, translate to German, verbose output
subtitler podcast.mp3 -l en -t de -v

# Use CPU with int8 quantisation
subtitler interview.wav --device cpu --compute-type int8
```

## CLI reference

```
usage: subtitler [-h] [-o OUTPUT] [-f {srt,vtt}] [-m MODEL] [--device {auto,cpu,cuda}]
                 [--compute-type TYPE] [-l LANG] [--no-vad] [-t LANG] [-v] [--version]
                 [--list-languages]
                 input

positional arguments:
  input                 Path to an audio or video file.

options:
  -o, --output          Output subtitle file path (default: <input>.srt)
  -f, --format          Subtitle format: srt, vtt
  -v, --verbose         Increase verbosity (-v INFO, -vv DEBUG)
  --version             Show version and exit
  --list-languages      Print supported translation languages and exit

transcription:
  -m, --model           Whisper model size (default: medium)
  --device              auto | cpu | cuda (default: auto)
  --compute-type        Quantisation type (auto-selected if omitted)
  -l, --language        Source language code (auto-detected if omitted)
  --no-vad              Disable Voice Activity Detection

translation:
  -t, --translate LANG  Translate subtitles to this language code
```

## Python API

```python
from subtitler.transcriber import load_model, transcribe
from subtitler.translator import translate_segments
from subtitler.subtitle import write_subtitle_file, SubtitleFormat

# 1. Transcribe
model = load_model("medium", device="auto")
result = transcribe(model, "video.mp4")

# 2. Translate (optional)
translated = translate_segments(result.segments, source=result.language, target="it")

# 3. Write subtitle file
write_subtitle_file(translated, "video_it.srt")
```

## Building a Windows .exe

```bash
pip install ".[dev]"
pyinstaller subtitler.spec
```

The executable will be in `dist/subtitler.exe`.

> **Note:** The .exe does not bundle Whisper model weights. Models are downloaded
> on first run and cached in the default Hugging Face cache directory.

## Hugging Face token (optional)

On first run, Whisper model weights are downloaded from the Hugging Face Hub.
Without authentication you may see this warning:

> You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN
> to enable higher rate limits and faster downloads

This is **not an error** — the download still works, just at lower rate limits.
To silence the warning and get faster downloads:

1. Create a free account at <https://huggingface.co>.
2. Go to **Settings → Access Tokens** and generate a token.
3. Set the token before running subtitler:

```bash
# Linux / macOS
export HF_TOKEN="hf_your_token_here"

# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"
```

To make this permanent, add the variable to your shell profile or set it via
**System → Environment Variables** on Windows.

## Available models

| Model            | Parameters | Relative speed | VRAM   |
| ---------------- | ---------- | -------------- | ------ |
| `tiny`           | 39 M       | ~10×           | ~1 GB  |
| `base`           | 74 M       | ~7×            | ~1 GB  |
| `small`          | 244 M      | ~4×            | ~2 GB  |
| `medium`         | 769 M      | ~2×            | ~5 GB  |
| `large-v3`       | 1550 M     | 1×             | ~10 GB |
| `turbo`          | 809 M      | ~8×            | ~6 GB  |
| `distil-large-v3`| 756 M      | ~6×            | ~6 GB  |

### Choosing the right model

Not every model is the best choice for every situation. Here's a breakdown to
help you pick:

- **`tiny`** — Fastest model by far. Good for quick previews or testing your
  pipeline. Accuracy is noticeably lower, especially on non-English audio or
  noisy recordings. Use it when speed matters more than quality.
- **`base`** — A small step up from `tiny`. Slightly more accurate, still very
  fast. Suitable for clear speech in common languages.
- **`small`** — A solid mid-range option. Handles most languages well and runs
  comfortably on CPU. Good balance for everyday use when you don't have a GPU.
- **`medium`** — The default. Significantly more accurate than `small`,
  especially for accented speech, niche languages, and overlapping speakers.
  Slower on CPU, but a great choice with a GPU.
- **`large-v3`** — The most accurate model. Best for professional-quality
  subtitles, rare languages, or heavily accented audio. Requires a CUDA GPU
  with at least 10 GB VRAM for practical use.
- **`turbo`** — Near `large-v3` accuracy at roughly 8× the speed. This is the
  best "quality per second" option if you have a GPU with ≥6 GB VRAM.
- **`distil-large-v3`** — A distilled version of `large-v3`. Similar accuracy
  on English, slightly worse on other languages. Fast and memory-efficient.
  Best for English-heavy workloads on a GPU.

### Recommended commands

**Fastest result** — use `tiny` when you just need a rough draft quickly:

```bash
subtitler video.mp4 -m tiny
```

**Best result** — use `large-v3` (GPU required) for maximum accuracy:

```bash
subtitler video.mp4 -m large-v3
```

**Best compromise** — use `turbo` on GPU for near-best accuracy at high speed,
or `small` on CPU for a good quality-to-speed ratio:

```bash
# With a CUDA GPU (recommended)
subtitler video.mp4 -m turbo

# CPU only
subtitler video.mp4 -m small
```

> **Tip:** The device and compute type are auto-detected. If you have a CUDA
> GPU, subtitler will use it with `float16` automatically. On CPU it falls back
> to `int8` quantisation.

## Upgrading dependencies

```bash
pip install --upgrade faster-whisper deep-translator
```

## Contributing

1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Run tests: `python -m pytest`
4. Lint: `ruff check src/ tests/`
5. Open a pull request.

## License

[MIT](LICENSE)

## Acknowledgements

Built with the great help of [Claude Opus 4.6](https://www.anthropic.com/) by Anthropic.
