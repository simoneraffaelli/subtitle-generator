"""Tests for the CLI argument parser (no model loading required)."""

import logging
import warnings

from asub.cli import _build_parser, _configure_logging


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
        assert args.diarize is False
        assert args.hf_token is None
        assert args.speakers is None
        assert args.min_speakers is None
        assert args.max_speakers is None
        assert args.diarization_batch_size == 16

    def test_translate_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-t", "it"])
        assert args.translate == "it"

    def test_verbose_levels(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-vv"])
        assert args.verbose == 2

    def test_single_dash_verbose_alias(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-verbose"])
        assert args.verbose == 1

    def test_output_and_format(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-o", "out.vtt", "-f", "vtt"])
        assert str(args.output) == "out.vtt"
        assert args.format == "vtt"

    def test_model_selection(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["audio.mp3", "-m", "tiny"])
        assert args.model == "tiny"

    def test_diarization_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "audio.mp3",
                "--diarize",
                "--hf-token",
                "hf_test",
                "--min-speakers",
                "2",
                "--max-speakers",
                "4",
                "--diarization-batch-size",
                "8",
            ]
        )

        assert args.diarize is True
        assert args.hf_token == "hf_test"
        assert args.min_speakers == 2
        assert args.max_speakers == 4
        assert args.diarization_batch_size == 8


class TestOutputConfiguration:
    def test_default_output_suppresses_dependency_warnings(self, capsys) -> None:
        _configure_logging(0)

        logging.getLogger("whisperx.asr").warning("hidden dependency log")
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("hidden dependency warning", UserWarning, stacklevel=1)

        captured = capsys.readouterr()
        assert "hidden dependency log" not in captured.err
        assert caught == []

    def test_verbose_output_shows_dependency_warnings(self, capsys) -> None:
        _configure_logging(1)

        logging.getLogger("whisperx.asr").warning("shown dependency log")
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("shown dependency warning", UserWarning, stacklevel=1)

        captured = capsys.readouterr()
        assert "shown dependency log" in captured.err
        assert len(caught) == 1
        assert str(caught[0].message) == "shown dependency warning"
