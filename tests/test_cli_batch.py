"""Tests for batch and directory processing in the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from asub import cli
from asub.diarization import DiarizationUnavailableError
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

    def test_batch_diarization_reuses_one_engine(self, tmp_path, monkeypatch) -> None:
        input_dir = tmp_path / "media"
        input_dir.mkdir()
        (input_dir / "alpha.mp3").write_text("", encoding="utf-8")
        (input_dir / "beta.mp3").write_text("", encoding="utf-8")
        engine = object()
        load_calls: list[tuple[str, str, str | None, str | None, str | None, int]] = []
        transcribe_calls: list[tuple[str, str | None, bool, int | None]] = []
        write_calls: list[tuple[Path, list[str]]] = []

        monkeypatch.setattr(cli, "Spinner", DummySpinner)

        def fake_load_diarizer(
            model_size: str,
            *,
            device: str,
            compute_type: str | None,
            language: str | None,
            hf_token: str | None,
            batch_size: int,
        ):
            load_calls.append((model_size, device, compute_type, language, hf_token, batch_size))
            return engine

        def fake_transcribe_with_diarization(
            diarizer,
            input_path,
            *,
            language,
            vad_filter,
            speakers,
            min_speakers,
            max_speakers,
            on_segment,
        ):
            del min_speakers, max_speakers
            assert diarizer is engine
            path = Path(input_path)
            transcribe_calls.append((path.name, language, vad_filter, speakers))
            result = TranscriptionResult(
                language="en",
                language_probability=0.0,
                duration=2.0,
                segments=[Segment(start=0.0, end=1.0, text=path.stem, speaker="SPEAKER_00")],
            )
            if on_segment is not None:
                on_segment(1, result.segments[0], result.duration)
            return result

        def fake_write_subtitle_file(segments, output_path, fmt=None):
            del fmt
            path = Path(output_path)
            write_calls.append((path, [seg.speaker for seg in segments]))
            return path

        monkeypatch.setattr(cli, "load_diarizer", fake_load_diarizer)
        monkeypatch.setattr(cli, "transcribe_with_diarization", fake_transcribe_with_diarization)
        monkeypatch.setattr(cli, "write_subtitle_file", fake_write_subtitle_file)

        exit_code = cli.main(
            [
                str(input_dir),
                "--diarize",
                "--hf-token",
                "hf_test",
                "--speakers",
                "2",
                "-l",
                "zh",
            ]
        )

        assert exit_code == 0
        assert load_calls == [("medium", "auto", None, "zh", "hf_test", 16)]
        assert transcribe_calls == [
            ("alpha.mp3", "zh", True, 2),
            ("beta.mp3", "zh", True, 2),
        ]
        assert [speakers for _, speakers in write_calls] == [["SPEAKER_00"], ["SPEAKER_00"]]


class TestDiarizationValidation:
    def test_rejects_exact_and_range_speaker_counts(self, tmp_path, capsys) -> None:
        input_file = tmp_path / "audio.mp3"
        input_file.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                [
                    str(input_file),
                    "--diarize",
                    "--hf-token",
                    "hf_test",
                    "--speakers",
                    "2",
                    "--min-speakers",
                    "1",
                ]
            )

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "--speakers cannot be combined" in captured.err

    def test_rejects_min_speakers_greater_than_max_speakers(self, tmp_path, capsys) -> None:
        input_file = tmp_path / "audio.mp3"
        input_file.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                [
                    str(input_file),
                    "--diarize",
                    "--hf-token",
                    "hf_test",
                    "--min-speakers",
                    "4",
                    "--max-speakers",
                    "2",
                ]
            )

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "--min-speakers cannot be greater" in captured.err

    def test_diarize_without_optional_dependency_prints_install_hint(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ) -> None:
        input_file = tmp_path / "audio.mp3"
        input_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(cli, "Spinner", DummySpinner)

        def fake_load_diarizer(*args, **kwargs):
            del args, kwargs
            raise DiarizationUnavailableError('Install them with: pip install -e ".[diarization]"')

        monkeypatch.setattr(cli, "load_diarizer", fake_load_diarizer)

        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(input_file), "--diarize", "--hf-token", "hf_test"])

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "pip install" in captured.err
