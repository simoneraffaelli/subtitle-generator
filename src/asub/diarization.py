"""Optional speaker diarization powered by WhisperX."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asub.transcriber import (
    DEFAULT_MODEL,
    Segment,
    TranscriptionResult,
    _resolve_compute_type,
    _resolve_device,
)

logger = logging.getLogger(__name__)

_WHISPERX_AUDIO_SAMPLE_RATE = 16000
_DEFAULT_ALIGNMENT_LANGUAGES = frozenset(
    {
        "ar",
        "ca",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "eu",
        "fa",
        "fi",
        "fr",
        "gl",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ka",
        "ko",
        "lv",
        "ml",
        "nl",
        "nn",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "te",
        "tl",
        "tr",
        "uk",
        "ur",
        "vi",
        "zh",
    }
)


class DiarizationUnavailableError(RuntimeError):
    """Raised when the optional diarization dependencies are not installed."""


@dataclass(slots=True)
class WhisperXDiarizer:
    """Reusable WhisperX transcription, alignment, and diarization engine."""

    whisperx: Any
    model: Any
    diarization_pipeline: Any
    device: str
    compute_type: str
    batch_size: int
    align_models: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        vad_filter: bool = True,
        speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_segment: Callable[[int, Segment, float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio/video file and attach anonymous speaker labels."""
        if not vad_filter:
            logger.warning("WhisperX manages VAD internally; --no-vad is ignored with --diarize.")

        audio_path = str(Path(audio_path).resolve())
        logger.info("Transcribing and diarizing '%s'...", audio_path)

        audio = self.whisperx.load_audio(audio_path)
        transcribe_kwargs: dict[str, Any] = {"batch_size": self.batch_size}
        if language is not None:
            transcribe_kwargs["language"] = language
        result = self.model.transcribe(audio, **transcribe_kwargs)

        detected_language = str(result.get("language") or language or "unknown")
        raw_segments = result.get("segments", [])
        if raw_segments:
            result = self._align_segments(raw_segments, result, audio, detected_language)
            result["language"] = detected_language

        diarize_segments = self.diarization_pipeline(
            audio,
            **_speaker_count_kwargs(
                speakers=speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            ),
        )
        result = self.whisperx.assign_word_speakers(diarize_segments, result)

        duration = _duration_seconds(audio)
        segments = _segments_from_whisperx_result(result)
        if duration <= 0:
            duration = max((seg.end for seg in segments), default=0.0)

        for index, segment in enumerate(segments, start=1):
            logger.debug(
                "[%.2fs -> %.2fs] %s%s",
                segment.start,
                segment.end,
                f"[{segment.speaker}] " if segment.speaker else "",
                segment.text,
            )
            if on_segment is not None:
                on_segment(index, segment, duration)

        logger.info("Diarized transcription complete - %d segments.", len(segments))
        return TranscriptionResult(
            language=detected_language,
            language_probability=float(result.get("language_probability", 0.0)),
            duration=duration,
            segments=segments,
        )

    def _load_align_model(self, language: str) -> tuple[Any, Any]:
        if language not in self.align_models:
            self.align_models[language] = self.whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
        return self.align_models[language]

    def _align_segments(
        self,
        raw_segments: Sequence[Any],
        result: dict[str, Any],
        audio: Any,
        language: str,
    ) -> dict[str, Any]:
        if not _has_default_alignment_model(language):
            logger.warning(
                "No default WhisperX alignment model is available for language '%s'; "
                "using segment-level speaker assignment.",
                language,
            )
            return result

        try:
            align_model, align_metadata = self._load_align_model(language)
        except ValueError as exc:
            if not _is_missing_default_alignment_model(exc):
                raise
            logger.warning(
                "No default WhisperX alignment model is available for language '%s'; "
                "using segment-level speaker assignment.",
                language,
            )
            return result

        return self.whisperx.align(
            raw_segments,
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )


def load_diarizer(
    model_size: str = DEFAULT_MODEL,
    *,
    device: str = "auto",
    compute_type: str | None = None,
    language: str | None = None,
    hf_token: str | None = None,
    batch_size: int = 16,
) -> WhisperXDiarizer:
    """Load a reusable WhisperX diarization engine."""
    whisperx, diarization_pipeline_cls = _load_whisperx()
    device = _resolve_device(device)
    compute_type = _resolve_compute_type(device, compute_type)
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        msg = (
            "Speaker diarization requires a Hugging Face token. Pass --hf-token or set "
            "HF_TOKEN after accepting the pyannote speaker-diarization model terms."
        )
        raise ValueError(msg)

    logger.info("Loading WhisperX model '%s' on %s (%s)...", model_size, device, compute_type)
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
    diarization_pipeline = diarization_pipeline_cls(token=token, device=device)
    return WhisperXDiarizer(
        whisperx=whisperx,
        model=model,
        diarization_pipeline=diarization_pipeline,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
    )


def transcribe_with_diarization(
    diarizer: WhisperXDiarizer,
    audio_path: str | Path,
    *,
    language: str | None = None,
    vad_filter: bool = True,
    speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    on_segment: Callable[[int, Segment, float], None] | None = None,
) -> TranscriptionResult:
    """Transcribe with a preloaded WhisperX diarizer."""
    return diarizer.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter,
        speakers=speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        on_segment=on_segment,
    )


def _load_whisperx() -> tuple[Any, Any]:
    try:
        whisperx = importlib.import_module("whisperx")
        diarize = importlib.import_module("whisperx.diarize")
    except ImportError as exc:
        msg = (
            "Speaker diarization requires the optional WhisperX dependencies. "
            'Install them with: pip install -e ".[diarization]"'
        )
        raise DiarizationUnavailableError(msg) from exc
    return whisperx, diarize.DiarizationPipeline


def _speaker_count_kwargs(
    *,
    speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, int]:
    if speakers is not None:
        return {"num_speakers": speakers}

    kwargs: dict[str, int] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    return kwargs


def _duration_seconds(audio: Any) -> float:
    try:
        return len(audio) / _WHISPERX_AUDIO_SAMPLE_RATE
    except TypeError:
        return 0.0


def _is_missing_default_alignment_model(exc: ValueError) -> bool:
    message = str(exc).casefold()
    return "no default align-model" in message or "no default alignment model" in message


def _has_default_alignment_model(language: str) -> bool:
    normalized = language.strip().casefold()
    if "-" in normalized:
        normalized = normalized.split("-", 1)[0]
    return normalized in _DEFAULT_ALIGNMENT_LANGUAGES


def _segments_from_whisperx_result(result: Mapping[str, Any]) -> list[Segment]:
    segments: list[Segment] = []
    for raw in result.get("segments", []):
        if isinstance(raw, Mapping):
            segments.extend(_segments_from_whisperx_segment(raw))
    return segments


def _segments_from_whisperx_segment(raw: Mapping[str, Any]) -> list[Segment]:
    words = raw.get("words")
    if isinstance(words, Sequence) and not isinstance(words, str):
        word_segments = _segments_from_words(words)
        if word_segments:
            return word_segments

    start = _float_value(raw.get("start"), default=0.0)
    end = _float_value(raw.get("end"), default=start)
    return [
        Segment(
            start=start,
            end=end,
            text=str(raw.get("text") or "").strip(),
            speaker=_speaker_label(raw.get("speaker")),
        )
    ]


def _segments_from_words(words: Sequence[Any]) -> list[Segment]:
    usable_words: list[Mapping[str, Any]] = []
    for word in words:
        if not isinstance(word, Mapping):
            return []
        text = str(word.get("word") or "").strip()
        if not text:
            continue
        if word.get("start") is None or word.get("end") is None or word.get("speaker") is None:
            return []
        usable_words.append(word)

    if not usable_words:
        return []

    segments: list[Segment] = []
    current_speaker = _speaker_label(usable_words[0].get("speaker"))
    current_words: list[Mapping[str, Any]] = []

    for word in usable_words:
        speaker = _speaker_label(word.get("speaker"))
        if speaker != current_speaker and current_words:
            segments.append(_segment_from_word_group(current_words, current_speaker))
            current_words = []
            current_speaker = speaker
        current_words.append(word)

    if current_words:
        segments.append(_segment_from_word_group(current_words, current_speaker))
    return segments


def _segment_from_word_group(words: Sequence[Mapping[str, Any]], speaker: str | None) -> Segment:
    start = _float_value(words[0].get("start"), default=0.0)
    end = _float_value(words[-1].get("end"), default=start)
    return Segment(start=start, end=end, text=_join_words(words), speaker=speaker)


def _join_words(words: Sequence[Mapping[str, Any]]) -> str:
    text = " ".join(str(word.get("word") or "").strip() for word in words).strip()
    for punctuation in (".", ",", "!", "?", ";", ":", "%", ")", "]", "}"):
        text = text.replace(f" {punctuation}", punctuation)
    for punctuation in ("(", "[", "{"):
        text = text.replace(f"{punctuation} ", punctuation)
    return text


def _speaker_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
