"""Tests for subtitle generation (SRT / VTT output)."""

from subtitler.subtitle import (
    SubtitleFormat,
    generate_srt,
    generate_vtt,
    infer_output_path,
    write_subtitle_file,
)
from subtitler.transcriber import Segment


def _sample_segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=2.5, text="Hello, world!"),
        Segment(start=3.0, end=5.8, text="This is a test."),
        Segment(start=6.2, end=10.0, text="Subtitles are great."),
    ]


class TestTimestampFormatting:
    def test_srt_format(self) -> None:
        srt = generate_srt(_sample_segments())
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "00:00:03,000 --> 00:00:05,800" in srt

    def test_vtt_format(self) -> None:
        vtt = generate_vtt(_sample_segments())
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in vtt
        assert "00:00:03.000 --> 00:00:05.800" in vtt


class TestSRTContent:
    def test_sequence_numbers(self) -> None:
        srt = generate_srt(_sample_segments())
        lines = srt.strip().split("\n")
        # First cue number
        assert lines[0] == "1"

    def test_contains_all_text(self) -> None:
        srt = generate_srt(_sample_segments())
        for seg in _sample_segments():
            assert seg.text in srt

    def test_empty_segments(self) -> None:
        srt = generate_srt([])
        assert srt == ""


class TestVTTContent:
    def test_header_present(self) -> None:
        vtt = generate_vtt(_sample_segments())
        assert vtt.startswith("WEBVTT\n")

    def test_empty_segments(self) -> None:
        vtt = generate_vtt([])
        assert vtt.startswith("WEBVTT")


class TestInferOutputPath:
    def test_srt_extension(self) -> None:
        path = infer_output_path("video.mp4", SubtitleFormat.SRT)
        assert str(path).endswith(".srt")
        assert path.stem == "video"

    def test_vtt_extension(self) -> None:
        path = infer_output_path("video.mp4", SubtitleFormat.VTT)
        assert str(path).endswith(".vtt")

    def test_suffix(self) -> None:
        path = infer_output_path("video.mp4", SubtitleFormat.SRT, suffix="_it")
        assert path.stem == "video_it"
        assert str(path).endswith(".srt")


class TestWriteFile:
    def test_write_srt(self, tmp_path) -> None:
        out = tmp_path / "output.srt"
        written = write_subtitle_file(_sample_segments(), out)
        assert written.exists()
        content = written.read_text(encoding="utf-8")
        assert "Hello, world!" in content
        assert "00:00:00,000" in content

    def test_write_vtt(self, tmp_path) -> None:
        out = tmp_path / "output.vtt"
        written = write_subtitle_file(_sample_segments(), out)
        content = written.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")

    def test_auto_creates_parent_dirs(self, tmp_path) -> None:
        out = tmp_path / "sub" / "dir" / "output.srt"
        written = write_subtitle_file(_sample_segments(), out)
        assert written.exists()
