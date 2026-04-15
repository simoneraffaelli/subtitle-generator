# Architecture & Design

A technical overview of **asub** — how the project is structured, how the
pieces fit together, and why certain implementation choices were made.

---

## High-level pipeline

asub follows a simple linear pipeline:

```
Audio/Video file
      │
      ▼
┌─────────────┐
│  Transcribe  │  faster-whisper (CTranslate2)
└──────┬──────┘
       │  list[Segment]
       ▼
┌─────────────┐
│  Translate   │  deep-translator (Google Translate)   ← optional
└──────┬──────┘
       │  list[Segment]
       ▼
┌─────────────┐
│  Write file  │  SRT or WebVTT
└─────────────┘
```

Each stage is handled by a dedicated module. The CLI module (`cli.py`)
orchestrates the pipeline, calling each stage in sequence.

---

## Project structure

```
src/asub/
├── __init__.py       # Package metadata (version string)
├── __main__.py       # Entry point for `python -m asub`
├── cli.py            # Argument parsing, orchestration, user output
├── transcriber.py    # Whisper model loading and transcription
├── subtitle.py       # SRT/VTT generation and file writing
├── translator.py     # Segment translation via Google Translate
└── progress.py       # Terminal spinner utility
```

### Why this split?

Each module has **one responsibility**:

| Module           | Responsibility                              | External dependency       |
| ---------------- | ------------------------------------------- | ------------------------- |
| `transcriber.py` | Load a Whisper model and transcribe audio    | `faster-whisper`          |
| `translator.py`  | Translate text segments                      | `deep-translator`         |
| `subtitle.py`    | Format segments into SRT/VTT and write files | (none — pure Python)      |
| `progress.py`    | Show animated spinners in the terminal       | (none — pure Python)      |
| `cli.py`         | Parse arguments, wire the pipeline together  | All of the above          |

This separation means:

- **`subtitle.py` and `translator.py` never import `faster-whisper`** at
  runtime. They use `TYPE_CHECKING` guards to reference the `Segment` type only
  for static analysis. This keeps import time fast and avoids pulling
  CTranslate2 into modules that don't need it.
- Each module can be used independently as a **Python API** — you can call
  `transcribe()`, `translate_segments()`, or `write_subtitle_file()` from your
  own scripts without going through the CLI.
- Tests can target individual modules in isolation.

---

## Module deep dives

### `transcriber.py` — the transcription engine

This is the core of asub. It wraps `faster-whisper`, which is a CTranslate2
re-implementation of OpenAI's Whisper. CTranslate2 gives roughly 4× faster
inference than the original PyTorch Whisper, with lower memory usage.

**Key data types:**

```python
@dataclass(frozen=True, slots=True)
class Segment:
    start: float    # seconds
    end: float      # seconds
    text: str

@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    segments: list[Segment]
```

Both dataclasses use `frozen=True` (immutable) and `slots=True` (lower memory,
faster attribute access). `Segment` is the universal data type that flows
through the entire pipeline — transcriber produces them, translator transforms
their text, subtitle writer consumes them.

**Device and compute type auto-detection:**

```python
def _cuda_available() -> bool:
    import ctranslate2
    return "cuda" in ctranslate2.get_supported_compute_types("cuda")
```

Rather than depending on `torch.cuda.is_available()` (which would pull in the
entire PyTorch library), we query CTranslate2 directly — the library that
actually runs the model. This is more accurate and avoids a heavy transitive
dependency.

The resolution chain is:

1. `_resolve_device("auto")` → checks CUDA availability → returns `"cuda"` or
   `"cpu"`
2. `_resolve_compute_type(device, None)` → `"float16"` for CUDA, `"int8"` for
   CPU

This means users get optimal performance out of the box without needing to
understand quantisation types.

**Streaming callback:**

```python
on_segment: Callable[[int, Segment, float], None] | None = None
```

`faster-whisper` yields segments lazily via a generator. As each segment is
produced, the optional `on_segment` callback fires with `(index, segment,
audio_duration)`. The CLI uses this to update the progress spinner with a
percentage. The callback receives the full audio duration so it can compute
`segment.end / duration` as a progress estimate.

**Why `beam_size=5`?** This is the default value recommended by CTranslate2 for
a good accuracy-speed balance. Beam search explores multiple hypotheses in
parallel and picks the most likely one.

### `subtitle.py` — format generation

This module is pure Python with no external dependencies. It converts a list of
`Segment` objects into SRT or WebVTT strings.

**Two timestamp formatters:**

- SRT uses comma as the millisecond separator: `00:01:23,456`
- WebVTT uses a dot: `00:01:23.456`

Rather than a single formatter with a flag, there are two small functions
(`_format_timestamp_srt`, `_format_timestamp_vtt`). This is more readable and
avoids branching inside a hot loop.

**Format inference:** When the user doesn't specify `--format`, the output
extension is used to pick the format. This keeps the common case (`asub
video.mp4`) zero-config.

### `translator.py` — batched translation

Translation uses `deep-translator`'s `GoogleTranslator`, which calls Google
Translate's free web API. No API key needed.

**Batching strategy:**

Google Translate has a ~5000 character limit per request. The translator:

1. Groups segments into batches that fit under the limit.
2. Joins each batch with `\n` (newline) as a separator.
3. Sends one request per batch.
4. Splits the response back on `\n` to recover individual translations.

This dramatically reduces the number of HTTP requests. A 200-segment file might
need only 3–4 requests instead of 200.

**Fallback per-segment translation:** Google Translate sometimes merges or
splits lines in the response. When the returned line count doesn't match the
input count, the translator falls back to translating each segment individually
for that batch. This ensures we never lose or mix up segments.

**Lazy import of `Segment`:**

```python
from asub.transcriber import Segment as SegmentCls
```

This import is inside `translate_segments()`, not at module level. This avoids
creating a circular import path (`translator` → `transcriber` → heavy C
extensions) when the module is first loaded for lightweight operations like
`supported_languages()`.

### `progress.py` — terminal spinner

A minimal spinner implementation using only `threading` and `sys.stderr`.

**Design decisions:**

- **Braille characters** (`⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`) — these render as a smooth rotating
  animation in virtually all modern terminals. They're a single Unicode
  character each, so the cursor doesn't jump.
- **Background daemon thread** — the spinner runs in a separate thread so it
  keeps animating while the main thread is blocked on I/O (model loading,
  transcription, HTTP requests). Marked as `daemon=True` so it won't prevent
  process exit.
- **`threading.Event.wait(0.08)`** — used instead of `time.sleep()`. This lets
  the stop signal interrupt the wait immediately rather than sleeping the full
  80ms after the work is done.
- **Writes to stderr** — keeps spinner output separate from actual program
  output (which goes to stdout). This means `asub video.mp4 > output.srt`
  wouldn't capture spinner noise.
- **`update(message)`** — allows changing the spinner text mid-operation. The
  transcription progress uses this to show a live segment count and percentage.
- **Line clearing on exit** — when the `with` block ends, the spinner erases
  its line cleanly so the next `print()` starts fresh.

### `cli.py` — the orchestrator

The CLI module is the only place where all other modules come together. It:

1. Parses arguments with `argparse`.
2. Validates input (file existence, required args).
3. Configures logging verbosity.
4. Runs the pipeline: load model → transcribe → translate → write file.
5. Wraps each long-running step in a `Spinner` for user feedback.

**`nargs="?"` for the input argument:**

The input positional is declared with `nargs="?"` (optional) rather than being
strictly required. This allows `asub --list-languages` to work without
providing an input file. Validation is done manually after parsing:

```python
if input_path is None:
    parser.error("the following arguments are required: input")
```

**Two-level verbosity:**

- `-v` sets logging to `INFO` — shows model parameters, language detection,
  segment counts.
- `-vv` sets logging to `DEBUG` — additionally logs every individual segment
  with timestamps.
- Default is `WARNING` — only unexpected issues show up.

This uses `logging.basicConfig()` which only takes effect on the first call,
so third-party libraries that configure their own loggers aren't affected.

---

## Data flow

Here's how a `Segment` flows through the system:

```
faster-whisper generator
    │
    │  raw segment (start, end, text)
    ▼
transcriber.transcribe()
    │  strips whitespace from text
    │  fires on_segment callback
    │  collects into list
    ▼
TranscriptionResult.segments: list[Segment]
    │
    ├── (no translation) ─────────────────────┐
    │                                          │
    ▼                                          │
translator.translate_segments()                │
    │  batches text, sends to Google Translate │
    │  creates NEW Segment objects             │
    │  with same start/end, translated text    │
    ▼                                          │
list[Segment]  ◄───────────────────────────────┘
    │
    ▼
subtitle.write_subtitle_file()
    │  formats as SRT or VTT
    │  writes UTF-8 file
    ▼
output.srt / output.vtt
```

The `Segment` dataclass is frozen (immutable), so the translator creates new
`Segment` instances rather than mutating the originals. This means you still
have access to the untranslated result after translation if needed (relevant
for the Python API).

---

## Error handling philosophy

asub follows a **fail-fast, validate-at-boundaries** approach:

- **CLI layer** validates user input (file exists, required arguments). Errors
  here use `parser.error()` which prints a clean message and exits with code 2.
- **Library layer** uses descriptive `ValueError` exceptions for invalid
  parameters (e.g. unsupported device string). No silent defaults for obviously
  wrong input.
- **External failures** (model download, network errors in translation) are
  allowed to propagate naturally. The stack trace from `-vv` gives enough
  context for debugging.

There is no try/except wrapping around the main pipeline. If something fails,
the user sees the real error. This is intentional — swallowing exceptions makes
debugging harder.

---

## Testing strategy

Tests are in `tests/` and split by module:

- **`test_cli.py`** — tests the argument parser only. No model loading, no I/O.
  Verifies defaults, flag combinations, and verbosity levels.
- **`test_subtitle.py`** — tests SRT/VTT generation, timestamp formatting,
  output path inference, and file writing (using `tmp_path`).

Tests avoid importing `faster-whisper` or hitting the network. They exercise
the pure-logic parts of the codebase. The `Segment` dataclass is simple enough
to construct directly in test fixtures.

---

## Dependency choices

| Dependency         | Why                                                                 |
| ------------------ | ------------------------------------------------------------------- |
| `faster-whisper`   | CTranslate2 backend — 4× faster than OpenAI Whisper, lower VRAM    |
| `deep-translator`  | Free Google Translate — no API key, 100+ languages, zero setup      |
| `ruff`             | Linter + formatter in one tool, extremely fast (written in Rust)    |
| `pytest`           | Standard Python test runner, minimal boilerplate                    |
| `pyinstaller`      | Single-file .exe packaging for Windows distribution                 |

**Why not OpenAI's `whisper`?** The original uses PyTorch and is significantly
slower. `faster-whisper` uses the same model weights but runs them through
CTranslate2, which is optimised for inference with int8/float16 quantisation.

**Why not `googletrans` or the official Google Cloud API?** `googletrans` is
unmaintained and breaks frequently. The official API requires billing setup.
`deep-translator` wraps the free web endpoint reliably and supports batch
translation.

---

## Build and distribution

- **PyPI**: standard `pyproject.toml` with `setuptools` backend. The
  `[project.scripts]` entry point registers `asub` as a console command.
- **GitHub Actions**: `.github/workflows/python-publish.yml` uses trusted
  publishing (OIDC) to push to PyPI on GitHub releases — no API tokens stored
  in secrets.
- **PyInstaller**: `asub.spec` defines a single-file executable. Hidden imports
  are listed explicitly because PyInstaller can't always detect dynamic imports
  from `faster-whisper` and `ctranslate2`. Model weights are **not** bundled —
  they're downloaded on first run from Hugging Face Hub.

---

## Python version support

The project targets **Python 3.10+** (`requires-python = ">=3.10"`). This
allows use of:

- `X | Y` union syntax instead of `Union[X, Y]`
- `list[str]` instead of `List[str]`
- `from __future__ import annotations` for forward references

Every module includes `from __future__ import annotations` so that type hints
are treated as strings and don't cause runtime errors with older-style type
checking tools.
