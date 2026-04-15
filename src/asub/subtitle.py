"""Generate subtitle files (SRT, VTT) from transcription segments."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from asub.transcriber import Segment

logger = logging.getLogger(__name__)


class SubtitleFormat(Enum):
    """Supported subtitle output formats."""

    SRT = "srt"
    VTT = "vtt"


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT standard)."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = round((secs - int(secs)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm`` (WebVTT standard)."""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = round((secs - int(secs)) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}.{millis:03d}"


def generate_srt(segments: Sequence[Segment]) -> str:
    """Build an SRT-formatted string from segments."""
    lines: list[str] = []
    for index, seg in enumerate(segments, start=1):
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        lines.append(f"{index}")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")  # blank line between cues
    return "\n".join(lines)


def generate_vtt(segments: Sequence[Segment]) -> str:
    """Build a WebVTT-formatted string from segments."""
    lines: list[str] = ["WEBVTT", ""]
    for index, seg in enumerate(segments, start=1):
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        lines.append(f"{index}")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def generate(segments: Sequence[Segment], fmt: SubtitleFormat) -> str:
    """Generate subtitle content in the requested format."""
    generators = {
        SubtitleFormat.SRT: generate_srt,
        SubtitleFormat.VTT: generate_vtt,
    }
    return generators[fmt](segments)


def write_subtitle_file(
    segments: Sequence[Segment],
    output_path: str | Path,
    fmt: SubtitleFormat | None = None,
) -> Path:
    """Write segments to a subtitle file.

    Parameters
    ----------
    segments:
        Timed text segments.
    output_path:
        Destination file path.
    fmt:
        Subtitle format.  If ``None``, inferred from *output_path*'s extension.

    Returns
    -------
    The resolved :class:`Path` of the written file.

    """
    output_path = Path(output_path)

    if fmt is None:
        ext = output_path.suffix.lower().lstrip(".")
        try:
            fmt = SubtitleFormat(ext)
        except ValueError:
            msg = (
                f"Cannot infer subtitle format from extension '.{ext}'. "
                f"Use one of: {', '.join(f.value for f in SubtitleFormat)}."
            )
            raise ValueError(msg) from None

    content = generate(segments, fmt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    logger.info("Subtitle file written → %s", output_path)
    return output_path


def infer_output_path(
    input_path: str | Path,
    fmt: SubtitleFormat,
    *,
    suffix: str = "",
) -> Path:
    """Derive an output path from the input file.

    Example
    -------
    >>> infer_output_path("video.mp4", SubtitleFormat.SRT)
    PosixPath('video.srt')
    >>> infer_output_path("video.mp4", SubtitleFormat.SRT, suffix="_en")
    PosixPath('video_en.srt')

    """
    p = Path(input_path)
    return p.with_name(f"{p.stem}{suffix}.{fmt.value}")
