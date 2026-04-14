"""Command-line interface for subtitler."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from subtitler import __version__
from subtitler.subtitle import SubtitleFormat, infer_output_path, write_subtitle_file
from subtitler.transcriber import AVAILABLE_MODELS, DEFAULT_MODEL, load_model, transcribe
from subtitler.translator import translate_segments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitler",
        description="Generate and translate subtitles from audio/video files.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Path to an audio or video file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output subtitle file path. Defaults to <input>.srt.",
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


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Returns
    -------
    Exit code: ``0`` on success, ``1`` on user error.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- List languages and exit ---
    if args.list_languages:
        from subtitler.translator import supported_languages

        langs = supported_languages()
        for name, code in sorted(langs.items()):
            print(f"  {code:6s}  {name}")
        return 0

    # --- Validate input ---
    input_path: Path | None = args.input
    if input_path is None:
        parser.error("the following arguments are required: input")
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    _configure_logging(args.verbose)
    logger = logging.getLogger("subtitler")

    # --- Determine output path & format ---
    fmt: SubtitleFormat | None = None
    if args.format is not None:
        fmt = SubtitleFormat(args.format)

    output_path: Path
    if args.output is not None:
        output_path = args.output
    else:
        target_fmt = fmt if fmt is not None else SubtitleFormat.SRT
        suffix = f"_{args.translate}" if args.translate else ""
        output_path = infer_output_path(input_path, target_fmt, suffix=suffix)

    # --- Transcribe ---
    logger.info("Model: %s | Device: %s", args.model, args.device)
    print(f"Loading model '{args.model}'…", flush=True)
    model = load_model(args.model, device=args.device, compute_type=args.compute_type)

    print(f"Transcribing '{input_path.name}'…", flush=True)

    def _on_segment(index: int, _seg: object) -> None:
        print(f"\r  Segments transcribed: {index}", end="", flush=True)

    result = transcribe(
        model,
        input_path,
        language=args.language,
        vad_filter=not args.no_vad,
        on_segment=_on_segment,
    )

    # Finish the progress line
    print()

    segments = result.segments
    print(
        f"Transcribed {len(segments)} segments "
        f"(detected language: {result.language}, "
        f"confidence: {result.language_probability:.0%})"
    )

    # --- Translate (optional) ---
    if args.translate:
        print(f"Translating to '{args.translate}'…", flush=True)
        segments = translate_segments(
            segments,
            source=result.language,
            target=args.translate,
        )
        print(f"Translated to '{args.translate}'.")

    # --- Write output ---
    written = write_subtitle_file(segments, output_path, fmt=fmt)
    print(f"Saved → {written}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
