"""Translation layer using deep-translator (Google Translate by default)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deep_translator import GoogleTranslator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from asub.transcriber import Segment, TranscriptionResult

logger = logging.getLogger(__name__)

# Maximum characters Google Translate accepts per request.
_GOOGLE_CHAR_LIMIT = 5000


def supported_languages() -> dict[str, str]:
    """Return a ``{name: code}`` mapping of supported target languages."""
    return GoogleTranslator().get_supported_languages(as_dict=True)


def translate_text(text: str, *, source: str = "auto", target: str = "en") -> str:
    """Translate a single string."""
    if not text.strip():
        return text
    return GoogleTranslator(source=source, target=target).translate(text)


def translate_segments(
    segments: Sequence[Segment],
    *,
    source: str = "auto",
    target: str = "en",
) -> list[Segment]:
    """Translate every segment's text while preserving timestamps.

    Segments are batched to stay under the Google Translate character limit,
    then split back to keep one-to-one correspondence with the originals.

    Parameters
    ----------
    segments:
        The transcribed segments to translate.
    source:
        Source language code, or ``"auto"`` for auto-detection.
    target:
        Target language code (e.g. ``"it"``, ``"de"``, ``"fr"``).

    Returns
    -------
    A new list of :class:`~asub.transcriber.Segment` with translated text.

    """
    from asub.transcriber import Segment as SegmentCls

    if not segments:
        return []

    logger.info("Translating %d segments → %s…", len(segments), target)
    translator = GoogleTranslator(source=source, target=target)

    # Build batches that fit under the character limit.
    separator = "\n"
    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_length = 0

    for idx, seg in enumerate(segments):
        addition = len(seg.text) + len(separator)
        if current_length + addition > _GOOGLE_CHAR_LIMIT and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_length = 0
        current_batch.append(idx)
        current_length += addition

    if current_batch:
        batches.append(current_batch)

    # Translate each batch and map results back.
    translated_texts: list[str] = [""] * len(segments)

    for batch_indices in batches:
        combined = separator.join(segments[i].text for i in batch_indices)
        result = translator.translate(combined)
        parts = result.split("\n")

        # If the translator merges/splits lines, fall back to per-segment translation.
        if len(parts) != len(batch_indices):
            logger.debug("Batch split mismatch — falling back to per-segment translation.")
            for i in batch_indices:
                translated_texts[i] = translator.translate(segments[i].text)
        else:
            for i, part in zip(batch_indices, parts, strict=True):
                translated_texts[i] = part.strip()

    result_segments = [
        SegmentCls(start=seg.start, end=seg.end, text=translated_texts[i])
        for i, seg in enumerate(segments)
    ]
    logger.info("Translation complete.")
    return result_segments


def translate_result(
    result: TranscriptionResult,
    *,
    target: str,
    source: str | None = None,
) -> list[Segment]:
    """Translate a full :class:`TranscriptionResult`.

    If *source* is ``None``, the detected language from the transcription is used.
    """
    src = source if source is not None else result.language
    return translate_segments(result.segments, source=src, target=target)
