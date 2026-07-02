"""Tests for translation batching behavior."""

from __future__ import annotations

from deep_translator.exceptions import RequestError

from asub import translator
from asub.transcriber import Segment


def test_translate_segments_splits_failed_batches(monkeypatch) -> None:
    calls: list[str] = []

    class FakeGoogleTranslator:
        def __init__(self, *, source: str, target: str) -> None:
            assert source == "th"
            assert target == "en"

        def translate(self, text: str) -> str:
            calls.append(text)
            if "\n" in text:
                raise RequestError()
            return f"{text} translated"

    monkeypatch.setattr(translator, "GoogleTranslator", FakeGoogleTranslator)
    monkeypatch.setattr(translator.time, "sleep", lambda _: None)

    segments = [
        Segment(start=0.0, end=1.0, text="one"),
        Segment(start=1.0, end=2.0, text="two"),
        Segment(start=2.0, end=3.0, text="three"),
    ]

    translated = translator.translate_segments(segments, source="th", target="en")

    assert [seg.text for seg in translated] == [
        "one translated",
        "two translated",
        "three translated",
    ]
    assert any("\n" in call for call in calls)


def test_translate_segments_normalizes_google_language_aliases(monkeypatch) -> None:
    initializers: list[tuple[str, str]] = []

    class FakeGoogleTranslator:
        def __init__(self, *, source: str, target: str) -> None:
            initializers.append((source, target))

        def translate(self, text: str) -> str:
            return f"{text} translated"

    monkeypatch.setattr(translator, "GoogleTranslator", FakeGoogleTranslator)

    segments = [Segment(start=0.0, end=1.0, text="hello")]

    translated = translator.translate_segments(segments, source="zh", target="he")

    assert initializers == [("zh-CN", "iw")]
    assert [seg.text for seg in translated] == ["hello translated"]


def test_translate_segments_preserves_speaker_labels(monkeypatch) -> None:
    class FakeGoogleTranslator:
        def __init__(self, *, source: str, target: str) -> None:
            assert source == "en"
            assert target == "it"

        def translate(self, text: str) -> str:
            return f"{text} translated"

    monkeypatch.setattr(translator, "GoogleTranslator", FakeGoogleTranslator)

    segments = [Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00")]

    translated = translator.translate_segments(segments, source="en", target="it")

    assert translated == [
        Segment(start=0.0, end=1.0, text="hello translated", speaker="SPEAKER_00")
    ]


def test_translate_text_normalizes_chinese_target_casing(monkeypatch) -> None:
    initializers: list[tuple[str, str]] = []

    class FakeGoogleTranslator:
        def __init__(self, *, source: str, target: str) -> None:
            initializers.append((source, target))

        def translate(self, text: str) -> str:
            return f"{text} translated"

    monkeypatch.setattr(translator, "GoogleTranslator", FakeGoogleTranslator)

    translated = translator.translate_text("hello", source="auto", target="zh-tw")

    assert initializers == [("auto", "zh-TW")]
    assert translated == "hello translated"
