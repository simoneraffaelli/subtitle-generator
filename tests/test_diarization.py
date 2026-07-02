"""Tests for the optional WhisperX diarization wrapper."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import pytest

from asub import diarization
from asub.diarization import DiarizationUnavailableError, WhisperXDiarizer
from asub.transcriber import Segment


class FakeWhisperX:
    def __init__(self) -> None:
        self.loaded_align_languages: list[str] = []

    def load_audio(self, audio_path: str) -> list[int]:
        assert audio_path
        return [0] * 32000

    def load_align_model(self, *, language_code: str, device: str):
        assert device == "cpu"
        self.loaded_align_languages.append(language_code)
        return object(), {"language": language_code}

    def align(
        self,
        segments,
        align_model,
        align_metadata,
        audio,
        device,
        *,
        return_char_alignments: bool,
    ):
        del segments, align_model, align_metadata, audio, device, return_char_alignments
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello there. Bye now.",
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.4},
                        {"word": "there.", "start": 0.5, "end": 0.9},
                        {"word": "Bye", "start": 1.0, "end": 1.4},
                        {"word": "now.", "start": 1.5, "end": 1.9},
                    ],
                }
            ]
        }

    def assign_word_speakers(self, diarize_segments, result):
        del diarize_segments
        words = result["segments"][0]["words"]
        words[0]["speaker"] = "SPEAKER_00"
        words[1]["speaker"] = "SPEAKER_00"
        words[2]["speaker"] = "SPEAKER_01"
        words[3]["speaker"] = "SPEAKER_01"
        return result


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        assert len(audio) == 32000
        return {"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": "ignored"}]}


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def __call__(self, audio, **kwargs):
        assert len(audio) == 32000
        self.calls.append(kwargs)
        return object()


def test_diarizer_splits_segments_on_word_speaker_changes(tmp_path) -> None:
    fake_whisperx = FakeWhisperX()
    fake_model = FakeModel()
    fake_pipeline = FakePipeline()
    diarizer = WhisperXDiarizer(
        whisperx=fake_whisperx,
        model=fake_model,
        diarization_pipeline=fake_pipeline,
        device="cpu",
        compute_type="int8",
        batch_size=4,
    )
    input_file = tmp_path / "audio.mp3"
    input_file.write_text("", encoding="utf-8")

    result = diarizer.transcribe(input_file, language="en", speakers=2)

    assert fake_model.calls == [{"batch_size": 4, "language": "en"}]
    assert fake_pipeline.calls == [{"num_speakers": 2}]
    assert fake_whisperx.loaded_align_languages == ["en"]
    assert result.language == "en"
    assert result.duration == 2.0
    assert result.segments == [
        Segment(start=0.0, end=0.9, text="Hello there.", speaker="SPEAKER_00"),
        Segment(start=1.0, end=1.9, text="Bye now.", speaker="SPEAKER_01"),
    ]


def test_diarizer_falls_back_to_segment_speakers_without_align_model(tmp_path, caplog) -> None:
    class FakeWhisperXWithoutThaiAlignment(FakeWhisperX):
        def load_align_model(self, *, language_code: str, device: str):
            del device
            self.loaded_align_languages.append(language_code)
            raise ValueError(f"No default align-model for language: {language_code}")

        def align(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("alignment should be skipped")

        def assign_word_speakers(self, diarize_segments, result):
            del diarize_segments
            result["segments"][0]["speaker"] = "SPEAKER_00"
            return result

    class FakeThaiModel(FakeModel):
        def transcribe(self, audio, **kwargs):
            self.calls.append(kwargs)
            assert len(audio) == 32000
            return {
                "language": "th",
                "segments": [{"start": 0.0, "end": 2.0, "text": "sawasdee"}],
            }

    fake_whisperx = FakeWhisperXWithoutThaiAlignment()
    fake_model = FakeThaiModel()
    fake_pipeline = FakePipeline()
    diarizer = WhisperXDiarizer(
        whisperx=fake_whisperx,
        model=fake_model,
        diarization_pipeline=fake_pipeline,
        device="cpu",
        compute_type="int8",
        batch_size=4,
    )
    input_file = tmp_path / "thai.mp3"
    input_file.write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="asub.diarization"):
        result = diarizer.transcribe(input_file)

    assert fake_whisperx.loaded_align_languages == ["th"]
    assert result.language == "th"
    assert result.segments == [
        Segment(start=0.0, end=2.0, text="sawasdee", speaker="SPEAKER_00")
    ]
    assert "segment-level speaker assignment" in caplog.text


def test_segment_conversion_falls_back_when_word_speakers_are_incomplete() -> None:
    result = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "speaker": "SPEAKER_00",
                "words": [{"word": "Hello", "start": 0.0, "end": 1.0}],
            }
        ]
    }

    assert diarization._segments_from_whisperx_result(result) == [
        Segment(start=0.0, end=1.0, text="Hello", speaker="SPEAKER_00")
    ]


def test_load_diarizer_reports_missing_optional_dependency(monkeypatch) -> None:
    def fake_import_module(name: str):
        if name == "whisperx":
            raise ImportError("missing")
        return importlib.import_module(name)

    monkeypatch.setattr(diarization.importlib, "import_module", fake_import_module)

    with pytest.raises(DiarizationUnavailableError, match="diarization"):
        diarization.load_diarizer(device="cpu", hf_token="hf_test")


def test_load_diarizer_passes_language_to_whisperx(monkeypatch) -> None:
    load_model_calls: list[dict[str, object]] = []

    class FakeWhisperX:
        def load_model(self, model_size, device, **kwargs):
            load_model_calls.append({"model_size": model_size, "device": device, **kwargs})
            return object()

    class FakeDiarizationPipeline:
        def __init__(self, *, token: str, device: str) -> None:
            assert token == "hf_test"
            assert device == "cpu"

    monkeypatch.setattr(
        diarization,
        "_load_whisperx",
        lambda: (FakeWhisperX(), FakeDiarizationPipeline),
    )

    diarization.load_diarizer(
        "tiny",
        device="cpu",
        compute_type="int8",
        language="zh",
        hf_token="hf_test",
    )

    assert load_model_calls == [
        {"model_size": "tiny", "device": "cpu", "compute_type": "int8", "language": "zh"}
    ]
