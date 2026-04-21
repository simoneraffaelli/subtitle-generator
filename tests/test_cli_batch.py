"""Tests for batch and directory processing in the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from asub import cli
from asub.subtitle import SubtitleFormat
from asub.transcriber import Segment, TranscriptionResult


class DummySpinner:
    def __init__(self, message: str = "") -> None:
        self.message = message

    def __enter__(self) -> DummySpinner:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, message: str) -> None:
        self.message = message


def _make_result(*, language: str = "en", text: str = "hello") -> TranscriptionResult:
    return TranscriptionResult(
        language=language,
        language_probability=0.99,
        duration=5.0,
        segments=[Segment(start=0.0, end=2.0, text=text)],
    )


def _configure_cli_mocks(monkeypatch, *, transcribe_behavior=None):
    model_calls: list[tuple[str, str, str | None]] = []
    transcribe_calls: list[tuple[str, str | None, bool]] = []
    translate_calls: list[tuple[str, str, list[str]]] = []
    write_calls: list[tuple[Path, SubtitleFormat | None, list[str]]] = []

    monkeypatch.setattr(cli, "Spinner", DummySpinner)

    def fake_load_model(model_size: str, *, device: str, compute_type: str | None):
        model_calls.append((model_size, device, compute_type))
        return object()

    def fake_transcribe(model, input_path, *, language, vad_filter, on_segment):
        path = Path(input_path)
        transcribe_calls.append((path.name, language, vad_filter))
        if transcribe_behavior is None:
            result = _make_result(text=path.stem)
        else:
            result = transcribe_behavior(path, language, vad_filter)
        if on_segment is not None and result.segments:
            on_segment(len(result.segments), result.segments[-1], result.duration)
        return result

    def fake_translate_segments(segments, *, source: str, target: str):
        translate_calls.append((source, target, [seg.text for seg in segments]))
        return [
            Segment(start=seg.start, end=seg.end, text=f"{seg.text}->{target}")
            for seg in segments
        ]

    def fake_write_subtitle_file(segments, output_path, fmt=None):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(seg.text for seg in segments), encoding="utf-8")
        write_calls.append((path, fmt, [seg.text for seg in segments]))
        return path

    monkeypatch.setattr(cli, "load_model", fake_load_model)
    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "translate_segments", fake_translate_segments)
    monkeypatch.setattr(cli, "write_subtitle_file", fake_write_subtitle_file)

    return model_calls, transcribe_calls, translate_calls, write_calls


class TestDirectoryInput:
    def test_batch_reuses_one_model_and_writes_to_output_directory(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        input_dir = tmp_path / "media"
        input_dir.mkdir()
        (input_dir / "beta.mp4").write_text("", encoding="utf-8")
        (input_dir / "alpha.mp3").write_text("", encoding="utf-8")
        output_dir = tmp_path / "subs"

        model_calls, transcribe_calls, _, write_calls = _configure_cli_mocks(monkeypatch)

        exit_code = cli.main([str(input_dir), "-o", str(output_dir), "-f", "vtt"])

        assert exit_code == 0
        assert model_calls == [("medium", "auto", None)]
        assert [name for name, _, _ in transcribe_calls] == ["alpha.mp3", "beta.mp4"]
        assert [path for path, _, _ in write_calls] == [
            output_dir / "alpha.vtt",
            output_dir / "beta.vtt",
        ]
        assert all(fmt == SubtitleFormat.VTT for _, fmt, _ in write_calls)

    def test_batch_uses_detected_language_per_file_for_translation(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        input_dir = tmp_path / "media"
        input_dir.mkdir()
        (input_dir / "english.mp3").write_text("", encoding="utf-8")
        (input_dir / "italian.mp3").write_text("", encoding="utf-8")

        def transcribe_behavior(
            path: Path,
            language: str | None,
            vad_filter: bool,
        ) -> TranscriptionResult:
            del language, vad_filter
            detected_language = "en" if path.stem == "english" else "it"
            return _make_result(language=detected_language, text=path.stem)

        _, _, translate_calls, write_calls = _configure_cli_mocks(
            monkeypatch,
            transcribe_behavior=transcribe_behavior,
        )

        exit_code = cli.main([str(input_dir), "-t", "de"])

        assert exit_code == 0
        assert translate_calls == [
            ("en", "de", ["english"]),
            ("it", "de", ["italian"]),
        ]
        assert [texts for _, _, texts in write_calls] == [["english->de"], ["italian->de"]]

    def test_batch_continues_after_file_failure_and_returns_one(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ) -> None:
        input_dir = tmp_path / "media"
        input_dir.mkdir()
        for name in ("a.mp3", "b.mp3", "c.mp3"):
            (input_dir / name).write_text("", encoding="utf-8")

        def transcribe_behavior(
            path: Path,
            language: str | None,
            vad_filter: bool,
        ) -> TranscriptionResult:
            del language, vad_filter
            if path.stem == "b":
                raise RuntimeError("decoder failed")
            return _make_result(text=path.stem)

        _, transcribe_calls, _, write_calls = _configure_cli_mocks(
            monkeypatch,
            transcribe_behavior=transcribe_behavior,
        )

        exit_code = cli.main([str(input_dir)])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert [name for name, _, _ in transcribe_calls] == ["a.mp3", "b.mp3", "c.mp3"]
        assert [path.name for path, _, _ in write_calls] == ["a.srt", "c.srt"]
        assert "Batch complete: 2 succeeded, 1 failed." in captured.out
        assert "[2/3] b.mp3: failed (decoder failed)" in captured.err
        assert "b.mp3: decoder failed" in captured.err

    def test_batch_rejects_directory_without_supported_media(self, tmp_path, capsys) -> None:
        input_dir = tmp_path / "docs"
        input_dir.mkdir()
        (input_dir / "notes.txt").write_text("hello", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(input_dir)])

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "No supported audio/video files found in directory" in captured.err

    def test_batch_rejects_duplicate_derived_output_paths(self, tmp_path, capsys) -> None:
        input_dir = tmp_path / "media"
        input_dir.mkdir()
        (input_dir / "clip.mp3").write_text("", encoding="utf-8")
        (input_dir / "clip.wav").write_text("", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(input_dir)])

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "Multiple input files would write to the same subtitle path" in captured.err

    def test_batch_requires_output_to_be_a_directory_path(self, tmp_path, capsys) -> None:
        input_dir = tmp_path / "media"
        input_dir.mkdir()
        (input_dir / "clip.mp3").write_text("", encoding="utf-8")
        output_file = tmp_path / "captions.srt"
        output_file.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(input_dir), "-o", str(output_file)])

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "--output must be a directory path" in captured.err
