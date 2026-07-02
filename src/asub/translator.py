"""Translation layer using deep-translator (Google Translate by default)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from asub.transcriber import Segment, TranscriptionResult

logger = logging.getLogger(__name__)

# Maximum characters Google Translate accepts per request.
_GOOGLE_CHAR_LIMIT = 5000
_TRANSLATION_RETRY_ATTEMPTS = 3
_TRANSLATION_RETRY_DELAY_SECONDS = 1.0
_GOOGLE_TRANSLATE_LANGUAGE_ALIASES = {
    # faster-whisper reports Chinese as "zh"; GoogleTranslator requires a
    # regional variant. Default to Simplified Chinese unless the caller is more
    # specific.
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
    "zh-hans": "zh-CN",
    "zh-sg": "zh-CN",
    "zh-my": "zh-CN",
    "zh-tw": "zh-TW",
    "zh-hant": "zh-TW",
    "zh-hk": "zh-TW",
    "zh-mo": "zh-TW",
    # Google Translate still exposes Hebrew with its legacy code here.
    "he": "iw",
}


def _normalize_google_language(language: str) -> str:
    language = language.strip()
    return _GOOGLE_TRANSLATE_LANGUAGE_ALIASES.get(language.casefold(), language)


def supported_languages() -> dict[str, str]:
    """Return a ``{name: code}`` mapping of supported target languages."""
    return GoogleTranslator().get_supported_languages(as_dict=True)


def translate_text(text: str, *, source: str = "auto", target: str = "en") -> str:
    """Translate a single string."""
    if not text.strip():
        return text
    source = _normalize_google_language(source)
    target = _normalize_google_language(target)
    return GoogleTranslator(source=source, target=target).translate(text)


def _translate_with_retry(translator: GoogleTranslator, text: str) -> str:
    last_error: RequestError | None = None
    for attempt in range(_TRANSLATION_RETRY_ATTEMPTS):
        try:
            return translator.translate(text)
        except RequestError as exc:
            last_error = exc
            if attempt + 1 < _TRANSLATION_RETRY_ATTEMPTS:
                time.sleep(_TRANSLATION_RETRY_DELAY_SECONDS * (attempt + 1))

    assert last_error is not None
    raise last_error


def _translate_batch_texts(
    translator: GoogleTranslator,
    texts: list[str],
    *,
    separator: str,
) -> list[str]:
    combined = separator.join(texts)
    try:
        result = _translate_with_retry(translator, combined)
    except RequestError:
        if len(texts) == 1:
            raise

        midpoint = len(texts) // 2
        logger.debug("Batch translation failed; retrying as smaller batches.")
        return [
            *_translate_batch_texts(translator, texts[:midpoint], separator=separator),
            *_translate_batch_texts(translator, texts[midpoint:], separator=separator),
        ]

    parts = result.split(separator)
    if len(parts) != len(texts):
        if len(texts) == 1:
            return [result.strip()]

        midpoint = len(texts) // 2
        logger.debug("Batch split mismatch; retrying as smaller batches.")
        return [
            *_translate_batch_texts(translator, texts[:midpoint], separator=separator),
            *_translate_batch_texts(translator, texts[midpoint:], separator=separator),
        ]

    return [part.strip() for part in parts]


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

    source = _normalize_google_language(source)
    target = _normalize_google_language(target)
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
        batch_texts = [segments[i].text for i in batch_indices]
        parts = _translate_batch_texts(translator, batch_texts, separator=separator)
        for i, part in zip(batch_indices, parts, strict=True):
            translated_texts[i] = part

    result_segments = [
        SegmentCls(start=seg.start, end=seg.end, text=translated_texts[i], speaker=seg.speaker)
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
