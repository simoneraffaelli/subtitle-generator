"""Tests for the CLI argument parser (no model loading required)."""

from asub.cli import _build_parser


class TestParserDefaults:
    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3"])
        assert args.model == "medium"
        assert args.device == "auto"
        assert args.translate is None
        assert args.verbose == 0
        assert args.format is None
        assert args.no_vad is False

    def test_translate_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-t", "it"])
        assert args.translate == "it"

    def test_verbose_levels(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-vv"])
        assert args.verbose == 2

    def test_output_and_format(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-o", "out.vtt", "-f", "vtt"])
        assert str(args.output) == "out.vtt"
        assert args.format == "vtt"

    def test_model_selection(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-m", "tiny"])
        assert args.model == "tiny"
