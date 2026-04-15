"""Transcription engine powered by faster-whisper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Models ordered by size — users pick the trade-off between speed and accuracy.
AVAILABLE_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
    "distil-large-v3",
)

DEFAULT_MODEL = "medium"


@dataclass(frozen=True, slots=True)
class Segment:
    """A single transcribed segment with timing information."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Complete result of a transcription run."""

    language: str
    language_probability: float
    duration: float
    segments: list[Segment]


def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    try:
        import ctranslate2

        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    """Resolve the device, auto-detecting CUDA availability when needed."""
    if device == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if device in ("cpu", "cuda"):
        return device
    msg = f"Unsupported device: {device!r}. Use 'auto', 'cpu', or 'cuda'."
    raise ValueError(msg)


def _resolve_compute_type(device: str, compute_type: str | None) -> str:
    """Pick a sensible compute type when the caller doesn't specify one."""
    if compute_type is not None:
        return compute_type
    if device == "cuda":
        return "float16"
    return "int8"


def load_model(
    model_size: str = DEFAULT_MODEL,
    *,
    device: str = "auto",
    compute_type: str | None = None,
) -> WhisperModel:
    """Load a Whisper model for transcription.

    Parameters
    ----------
    model_size:
        One of :data:`AVAILABLE_MODELS`.
    device:
        ``"auto"`` (default), ``"cpu"``, or ``"cuda"``.
    compute_type:
        Quantisation type.  ``None`` picks a sensible default per device.

    """
    device = _resolve_device(device)
    compute_type = _resolve_compute_type(device, compute_type)

    logger.info("Loading Whisper model '%s' on %s (%s)…", model_size, device, compute_type)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(
    model: WhisperModel,
    audio_path: str | Path,
    *,
    language: str | None = None,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    on_segment: Callable[[int, Segment, float], None] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file and return timed segments.

    Parameters
    ----------
    model:
        A loaded :class:`WhisperModel`.
    audio_path:
        Path to an audio or video file (any format supported by FFmpeg / PyAV).
    language:
        ISO-639-1 code (e.g. ``"en"``).  ``None`` for auto-detection.
    vad_filter:
        Use Silero VAD to skip silence — reduces hallucination.
    word_timestamps:
        Request word-level timestamps (slower, but more precise).
    on_segment:
        Optional callback invoked after each segment is transcribed.
        Receives ``(segment_index, segment, audio_duration)``.

    """
    audio_path = str(Path(audio_path).resolve())
    logger.info("Transcribing '%s'…", audio_path)

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        beam_size=5,
    )

    logger.info(
        "Detected language: %s (probability %.2f%%)",
        info.language,
        info.language_probability * 100,
    )

    segments: list[Segment] = []
    for seg in segments_gen:
        segment = Segment(start=seg.start, end=seg.end, text=seg.text.strip())
        segments.append(segment)
        logger.debug("[%.2fs → %.2fs] %s", seg.start, seg.end, segment.text)
        if on_segment is not None:
            on_segment(len(segments), segment, info.duration)

    logger.info("Transcription complete — %d segments.", len(segments))
    return TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        duration=info.duration,
        segments=segments,
    )


def transcribe_file(
    audio_path: str | Path,
    *,
    model_size: str = DEFAULT_MODEL,
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    vad_filter: bool = True,
) -> TranscriptionResult:
    """Convenience wrapper: load a model, transcribe, and return the result."""
    model = load_model(model_size, device=device, compute_type=compute_type)
    return transcribe(model, audio_path, language=language, vad_filter=vad_filter)
