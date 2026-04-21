"""Command-line interface for asub."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from asub import __version__
from asub.progress import Spinner
from asub.subtitle import SubtitleFormat, infer_output_path, write_subtitle_file
from asub.transcriber import AVAILABLE_MODELS, DEFAULT_MODEL, Segment, load_model, transcribe
from asub.translator import translate_segments

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".aac",
        ".aiff",
        ".avi",
        ".flac",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asub",
        description="Generate and translate subtitles from audio/video files.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Path to an audio/video file, or a folder containing media files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output subtitle file path for a single input file, or an output directory "
            "when the input is a folder."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=[f.value for f in SubtitleFormat],
        default=None,
        help="Subtitle format (default: inferred from output extension, or srt).",
    )

    # --- Transcription options ---
    transcription = parser.add_argument_group("transcription")
    transcription.add_argument(
        "-m",
        "--model",
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL}).",
    )
    transcription.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help='Device to run inference on (default: "auto").',
    )
    transcription.add_argument(
        "--compute-type",
        default=None,
        help="Quantisation type (e.g. int8, float16). Auto-selected if omitted.",
    )
    transcription.add_argument(
        "-l",
        "--language",
        default=None,
        help="Source language code (e.g. en, it, de). Auto-detected if omitted.",
    )
    transcription.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection filter.",
    )

    # --- Translation options ---
    translation = parser.add_argument_group("translation")
    translation.add_argument(
        "-t",
        "--translate",
        metavar="LANG",
        default=None,
        help="Translate subtitles to this language code (e.g. it, de, fr, es).",
    )

    # --- General ---
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print supported translation languages and exit.",
    )
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _is_supported_media_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def _discover_directory_inputs(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.iterdir() if _is_supported_media_file(path)),
        key=lambda path: (path.name.lower(), path.name),
    )


def _resolve_inputs(
    parser: argparse.ArgumentParser,
    input_path: Path | None,
) -> tuple[list[Path], bool]:
    if input_path is None:
        parser.error("the following arguments are required: input")

    if not input_path.exists():
        parser.error(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path], False

    if not input_path.is_dir():
        parser.error(f"Input path is not a file or directory: {input_path}")

    inputs = _discover_directory_inputs(input_path)
    if not inputs:
        parser.error(f"No supported audio/video files found in directory: {input_path}")
    return inputs, True


def _resolve_output_plan(
    parser: argparse.ArgumentParser,
    input_files: list[Path],
    *,
    input_is_directory: bool,
    output_path: Path | None,
    fmt: SubtitleFormat | None,
    translate_to: str | None,
) -> dict[Path, Path]:
    target_fmt = fmt if fmt is not None else SubtitleFormat.SRT
    suffix = f"_{translate_to}" if translate_to else ""
    planned_outputs: dict[Path, Path] = {}

    batch_output_dir: Path | None = None
    if input_is_directory and output_path is not None:
        if output_path.exists() and not output_path.is_dir():
            parser.error("When the input is a directory, --output must be a directory path.")
        batch_output_dir = output_path

    for input_file in input_files:
        if input_is_directory:
            if batch_output_dir is None:
                planned = infer_output_path(input_file, target_fmt, suffix=suffix)
            else:
                planned = infer_output_path(
                    batch_output_dir / input_file.name,
                    target_fmt,
                    suffix=suffix,
                )
        elif output_path is not None:
            planned = output_path
        else:
            planned = infer_output_path(input_file, target_fmt, suffix=suffix)

        planned_outputs[input_file] = planned

    if input_is_directory:
        outputs_to_inputs: dict[Path, list[Path]] = {}
        for input_file, planned in planned_outputs.items():
            outputs_to_inputs.setdefault(planned.resolve(strict=False), []).append(input_file)

        collisions = {
            resolved_output: colliding_inputs
            for resolved_output, colliding_inputs in outputs_to_inputs.items()
            if len(colliding_inputs) > 1
        }
        if collisions:
            lines = ["Multiple input files would write to the same subtitle path:"]
            for resolved_output, colliding_inputs in sorted(
                collisions.items(),
                key=lambda item: str(item[0]).lower(),
            ):
                names = ", ".join(path.name for path in colliding_inputs)
                lines.append(f"  {resolved_output}: {names}")
            parser.error("\n".join(lines))

    return planned_outputs


def _format_file_label(input_path: Path, index: int, total: int) -> str:
    if total == 1:
        return input_path.name
    return f"[{index}/{total}] {input_path.name}"


def _process_input_file(
    input_path: Path,
    *,
    output_path: Path,
    fmt: SubtitleFormat | None,
    model: WhisperModel,
    language: str | None,
    translate_to: str | None,
    vad_filter: bool,
    index: int,
    total: int,
) -> Path:
    label = _format_file_label(input_path, index, total)

    with Spinner(f"{label} — transcribing") as spinner:

        def _on_segment(segment_index: int, seg: Segment, duration: float) -> None:
            pct = min(seg.end / duration * 100, 100.0) if duration > 0 else 0
            spinner.update(f"{label} — transcribing {segment_index} segments ({pct:.0f}%)")

        result = transcribe(
            model,
            input_path,
            language=language,
            vad_filter=vad_filter,
            on_segment=_on_segment,
        )

    segments = result.segments
    print(
        f"{label}: transcribed {len(segments)} segments "
        f"(detected language: {result.language}, "
        f"confidence: {result.language_probability:.0%})",
        flush=True,
    )

    if translate_to:
        if result.language.casefold() == translate_to.casefold():
            print(
                f"{label}: skipped translation because the detected language already matches "
                f"'{translate_to}'.",
                flush=True,
            )
        else:
            with Spinner(f"{label} — translating to '{translate_to}'"):
                segments = translate_segments(
                    segments,
                    source=result.language,
                    target=translate_to,
                )
            print(f"{label}: translated to '{translate_to}'.", flush=True)

    with Spinner(f"{label} — writing subtitle file"):
        written = write_subtitle_file(segments, output_path, fmt=fmt)
    print(f"{label}: saved -> {written}", flush=True)

    return written


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Returns
    -------
    Exit code: ``0`` on success, ``1`` when a batch run has per-file failures.
    Argument validation errors are handled by ``argparse`` with exit code ``2``.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- List languages and exit ---
    if args.list_languages:
        from asub.translator import supported_languages

        langs = supported_languages()
        for name, code in sorted(langs.items()):
            print(f"  {code:6s}  {name}")
        return 0

    # --- Validate input ---
    input_path: Path | None = args.input
    input_files, input_is_directory = _resolve_inputs(parser, input_path)

    _configure_logging(args.verbose)
    logger = logging.getLogger("asub")

    # --- Determine output path & format ---
    fmt: SubtitleFormat | None = None
    if args.format is not None:
        fmt = SubtitleFormat(args.format)

    planned_outputs = _resolve_output_plan(
        parser,
        input_files,
        input_is_directory=input_is_directory,
        output_path=args.output,
        fmt=fmt,
        translate_to=args.translate,
    )

    # --- Transcribe ---
    logger.info("Model: %s | Device: %s", args.model, args.device)
    with Spinner(f"Loading model '{args.model}'"):
        model = load_model(args.model, device=args.device, compute_type=args.compute_type)
    print(f"Model '{args.model}' loaded.", flush=True)

    total_files = len(input_files)
    if input_is_directory:
        print(f"Processing {total_files} file(s) from '{input_path}'.", flush=True)

    if not input_is_directory:
        _process_input_file(
            input_files[0],
            output_path=planned_outputs[input_files[0]],
            fmt=fmt,
            model=model,
            language=args.language,
            translate_to=args.translate,
            vad_filter=not args.no_vad,
            index=1,
            total=1,
        )
        return 0

    succeeded: list[tuple[Path, Path]] = []
    failed: list[tuple[Path, Exception]] = []

    for index, current_input in enumerate(input_files, start=1):
        try:
            written = _process_input_file(
                current_input,
                output_path=planned_outputs[current_input],
                fmt=fmt,
                model=model,
                language=args.language,
                translate_to=args.translate,
                vad_filter=not args.no_vad,
                index=index,
                total=total_files,
            )
            succeeded.append((current_input, written))
        except Exception as exc:  # pragma: no cover - exercised via tests with mocked failures.
            failed.append((current_input, exc))
            logger.debug("Detailed failure while processing '%s'.", current_input, exc_info=exc)
            print(
                f"{_format_file_label(current_input, index, total_files)}: failed ({exc})",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"Batch complete: {len(succeeded)} succeeded, {len(failed)} failed.",
        flush=True,
    )
    if failed:
        print("Failed files:", file=sys.stderr, flush=True)
        for failed_input, exc in failed:
            print(f"  {failed_input.name}: {exc}", file=sys.stderr, flush=True)
        return 1

    return 0
