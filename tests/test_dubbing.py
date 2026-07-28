"""Tests for omnivoice.cli.dubbing — SRT parsing, validation, non-speech detection, fallback logic."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Mock heavy ML dependencies that aren't available in test environment
# ---------------------------------------------------------------------------

def _mock_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]

_mock_module("torch")
_mock_module("torchaudio")
_mock_module("omnivoice")
_mock_module("omnivoice.models")
_mock_module("omnivoice.models.omnivoice")
_mock_module("omnivoice.utils")
_mock_module("omnivoice.utils.audio")
_mock_module("omnivoice.utils.common")
_mock_module("pydub")
_mock_module("pydub.silence")
_mock_module("soundfile")
_mock_module("numpy")

# Patch torch.inference_mode
sys.modules["torch"].inference_mode = lambda: (lambda f: f)

# Patch torch.Tensor for type annotations
sys.modules["torch"].Tensor = type("Tensor", (), {})
sys.modules["torch"].float16 = None

# Patch torchaudio.functional.resample
sys.modules["torchaudio"].functional = types.ModuleType("torchaudio.functional")

# Patch soundfile.write
sys.modules["soundfile"].write = lambda *a, **kw: None
sys.modules["soundfile"].read = lambda *a, **kw: (None, 24000)

# Patch numpy
import numpy as np
sys.modules["numpy"] = np

# Patch pydub.AudioSegment
class _MockAudioSegment:
    channels = 1
    frame_rate = 24000
    def set_channels(self, n): return self
    @staticmethod
    def from_mp3(path):
        return _MockAudioSegment()
    def __getitem__(self, s):
        return _MockAudioSegment()

sys.modules["pydub"].AudioSegment = _MockAudioSegment

# Patch omnivoice imports needed by dubbing.py
sys.modules["omnivoice.models.omnivoice"].OmniVoice = type("OmniVoice", (), {})
sys.modules["omnivoice.utils.audio"].concatenate_audio_with_silence = lambda chunks, sample_rate, silence_duration=0.3: np.array([])
sys.modules["omnivoice.utils.audio"].remove_silence = lambda audio, sampling_rate, mid_sil=300, lead_sil=100, trail_sil=300: audio
sys.modules["omnivoice.utils.common"].get_best_device = lambda: "cpu"

# ---------------------------------------------------------------------------
# Dynamic import (matching project convention from test_voice_clone_queue.py)
# ---------------------------------------------------------------------------
MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "omnivoice" / "cli" / "dubbing.py"
)
SPEC = importlib.util.spec_from_file_location("dubbing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dubbing = importlib.util.module_from_spec(SPEC)
sys.modules["dubbing"] = dubbing  # Needed for @dataclass decorator
SPEC.loader.exec_module(dubbing)

# Convenience aliases
parse_srt = dubbing.parse_srt
validate_srt_alignment = dubbing.validate_srt_alignment
is_non_speech = dubbing.is_non_speech
find_fallback_entry = dubbing.find_fallback_entry
SrtEntry = dubbing.SrtEntry
DubbingRequest = dubbing.DubbingRequest
DubbingResult = dubbing.DubbingResult
SrtAlignmentError = dubbing.SrtAlignmentError
DubbingError = dubbing.DubbingError
SegmentGenerationError = dubbing.SegmentGenerationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_srt(content: str) -> str:
    """Write SRT content to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", encoding="utf-8", delete=False
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _make_entry(
    index: int, start_sec: float, end_sec: float, text: str = "test"
) -> SrtEntry:
    return SrtEntry(index=index, start_sec=start_sec, end_sec=end_sec, text=text)


# ---------------------------------------------------------------------------
# Tests: parse_srt
# ---------------------------------------------------------------------------

class TestParseSrt(unittest.TestCase):
    """Tests for parse_srt()."""

    def test_two_entries(self):
        path = _write_srt(
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,200\nHow are you?\n"
        )
        result = parse_srt(path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].index, 1)
        self.assertAlmostEqual(result[0].start_sec, 1.0)
        self.assertAlmostEqual(result[0].end_sec, 3.5)
        self.assertEqual(result[0].text, "Hello world")
        self.assertEqual(result[1].index, 2)
        self.assertEqual(result[1].text, "How are you?")

    def test_multi_line_text(self):
        path = _write_srt(
            "1\n00:00:01,000 --> 00:00:03,000\nLine one\nLine two\n"
        )
        result = parse_srt(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Line one Line two")

    def test_empty_file(self):
        path = _write_srt("")
        result = parse_srt(path)
        self.assertEqual(len(result), 0)

    def test_single_entry(self):
        path = _write_srt("1\n00:00:00,500 --> 00:00:02,000\nHi\n")
        result = parse_srt(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hi")

    def test_unicode_text(self):
        path = _write_srt(
            "1\n00:00:01,000 --> 00:00:03,000\n"
            "Xin chao, ban khoe khong?\n"
        )
        result = parse_srt(path)
        self.assertEqual(len(result), 1)
        self.assertIn("Xin chao", result[0].text)

    def test_bom_tolerance(self):
        path = _write_srt(
            "\ufeff1\n00:00:01,000 --> 00:00:03,000\nHello\n"
        )
        result = parse_srt(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hello")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_srt("/nonexistent/path.srt")

    def test_timestamp_milliseconds(self):
        path = _write_srt(
            "1\n00:01:23,456 --> 00:02:45,789\nText\n"
        )
        result = parse_srt(path)
        self.assertAlmostEqual(result[0].start_sec, 83.456)
        self.assertAlmostEqual(result[0].end_sec, 165.789)


# ---------------------------------------------------------------------------
# Tests: is_non_speech
# ---------------------------------------------------------------------------

class TestIsNonSpeech(unittest.TestCase):
    """Tests for is_non_speech()."""

    def test_music_cue(self):
        self.assertTrue(is_non_speech("[music playing]"))

    def test_applause_cue(self):
        self.assertTrue(is_non_speech("(applause)"))

    def test_single_bracket_word(self):
        self.assertTrue(is_non_speech("[laughter]"))

    def test_normal_speech(self):
        self.assertFalse(is_non_speech("Hello, how are you?"))

    def test_inline_brackets_are_speech(self):
        self.assertFalse(is_non_speech("I said [hello] to him"))

    def test_empty_string(self):
        self.assertFalse(is_non_speech(""))

    def test_whitespace_only(self):
        self.assertFalse(is_non_speech("   "))

    def test_padded_brackets(self):
        self.assertTrue(is_non_speech("   [music]   "))

    def test_padded_parentheses(self):
        self.assertTrue(is_non_speech("  (laughs)  "))


# ---------------------------------------------------------------------------
# Tests: validate_srt_alignment
# ---------------------------------------------------------------------------

class TestValidateSrtAlignment(unittest.TestCase):
    """Tests for validate_srt_alignment()."""

    def test_valid_alignment(self):
        a = [_make_entry(1, 0.0, 2.0), _make_entry(2, 2.5, 4.0)]
        b = [_make_entry(1, 0.0, 2.0), _make_entry(2, 2.5, 4.0)]
        validate_srt_alignment(a, b)  # should not raise

    def test_small_timestamp_diff_within_tolerance(self):
        a = [_make_entry(1, 0.0, 2.0)]
        b = [_make_entry(1, 0.04, 2.03)]
        validate_srt_alignment(a, b)

    def test_mismatched_count(self):
        a = [_make_entry(1, 0.0, 1.0)]
        b = [_make_entry(1, 0.0, 1.0), _make_entry(2, 1.0, 2.0)]
        with self.assertRaises(SrtAlignmentError) as ctx:
            validate_srt_alignment(a, b)
        self.assertIn("count", str(ctx.exception).lower())

    def test_mismatched_index(self):
        a = [_make_entry(1, 0.0, 1.0)]
        b = [_make_entry(2, 0.0, 1.0)]
        with self.assertRaises(SrtAlignmentError):
            validate_srt_alignment(a, b)

    def test_timestamp_drift_beyond_tolerance(self):
        a = [_make_entry(1, 0.0, 2.0)]
        b = [_make_entry(1, 0.1, 2.1)]
        with self.assertRaises(SrtAlignmentError):
            validate_srt_alignment(a, b)

    def test_empty_lists(self):
        validate_srt_alignment([], [])


# ---------------------------------------------------------------------------
# Tests: find_fallback_entry
# ---------------------------------------------------------------------------

class TestFindFallbackEntry(unittest.TestCase):
    """Tests for find_fallback_entry()."""

    def setUp(self):
        self.entries = [
            _make_entry(1, 0.0, 2.5, "Long A"),
            _make_entry(2, 3.0, 3.5, "Short B"),
            _make_entry(3, 4.0, 6.5, "Long C"),
            _make_entry(4, 7.0, 7.3, "Short D"),
        ]

    def test_nearest_long_entry(self):
        result = find_fallback_entry(self.entries, 1, 1.5)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)

    def test_no_qualifying_entries(self):
        short_only = [_make_entry(1, 0.0, 0.5), _make_entry(2, 0.5, 0.8)]
        result = find_fallback_entry(short_only, 0, 1.5)
        self.assertIsNone(result)

    def test_self_is_fallback(self):
        result = find_fallback_entry(self.entries, 0, 1.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)

    def test_tie_breaker_prefers_earlier(self):
        entries = [
            _make_entry(1, 0.0, 3.0, "A"),
            _make_entry(2, 1.0, 1.5, "B"),
            _make_entry(3, 2.0, 5.0, "C"),
        ]
        result = find_fallback_entry(entries, 1, 1.5)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)

    def test_very_large_threshold(self):
        result = find_fallback_entry(self.entries, 0, 100.0)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests: Dataclasses
# ---------------------------------------------------------------------------

class TestSrtEntry(unittest.TestCase):
    """Tests for SrtEntry dataclass."""

    def test_creation(self):
        entry = SrtEntry(index=5, start_sec=1.5, end_sec=3.0, text="hello")
        self.assertEqual(entry.index, 5)
        self.assertEqual(entry.start_sec, 1.5)
        self.assertEqual(entry.end_sec, 3.0)
        self.assertEqual(entry.text, "hello")

    def test_frozen(self):
        entry = SrtEntry(1, 0.0, 1.0, "test")
        with self.assertRaises(Exception):
            entry.text = "modified"  # type: ignore[misc]


class TestDubbingRequest(unittest.TestCase):
    """Tests for DubbingRequest defaults."""

    def test_defaults(self):
        req = DubbingRequest(
            mp3_path="test.mp3",
            original_srt_path="orig.srt",
            translated_srt_path="trans.srt",
            language="Vietnamese",
        )
        self.assertEqual(req.padding_ms, 200)
        self.assertEqual(req.short_segment_threshold_s, 1.5)
        self.assertEqual(req.max_speed, 3.0)
        self.assertEqual(req.max_retries, 2)
        self.assertEqual(req.num_step, 32)
        self.assertEqual(req.guidance_scale, 2.0)
        self.assertTrue(req.denoise)
        self.assertTrue(req.postprocess_output)


class TestDubbingResult(unittest.TestCase):
    """Tests for DubbingResult dataclass."""

    def test_fields(self):
        result = DubbingResult(
            wav_path="/tmp/out.wav",
            total_segments=10,
            successful_segments=8,
            failed_indices=[3, 7],
            metadata={"lang": "vi"},
        )
        self.assertEqual(result.wav_path, "/tmp/out.wav")
        self.assertEqual(result.total_segments, 10)
        self.assertEqual(result.successful_segments, 8)
        self.assertEqual(result.failed_indices, [3, 7])
        self.assertEqual(result.metadata["lang"], "vi")


# ---------------------------------------------------------------------------
# Tests: Custom exceptions
# ---------------------------------------------------------------------------

class TestExceptions(unittest.TestCase):
    """Tests for custom exception classes."""

    def test_dubbing_error(self):
        with self.assertRaises(DubbingError):
            raise DubbingError("test")

    def test_srt_alignment_error_is_dubbing_error(self):
        err = SrtAlignmentError("mismatch")
        self.assertIsInstance(err, DubbingError)
        self.assertIsInstance(err, ValueError)

    def test_segment_generation_error(self):
        err = SegmentGenerationError(5, "model error")
        self.assertEqual(err.index, 5)
        self.assertIn("Segment 5", str(err))
        self.assertIsInstance(err, DubbingError)


# ---------------------------------------------------------------------------
# Integration test: parse -> validate workflow
# ---------------------------------------------------------------------------

class TestParseValidateWorkflow(unittest.TestCase):
    """End-to-end test of parse_srt -> validate_srt_alignment."""

    def test_parse_and_validate_identical_files(self):
        content = (
            "1\n00:00:01,000 --> 00:00:02,000\nLine 1\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nLine 2\n"
        )
        path_a = _write_srt(content)
        path_b = _write_srt(content)
        try:
            entries_a = parse_srt(path_a)
            entries_b = parse_srt(path_b)
            self.assertEqual(len(entries_a), 2)
            self.assertEqual(len(entries_b), 2)
            validate_srt_alignment(entries_a, entries_b)
        finally:
            Path(path_a).unlink(missing_ok=True)
            Path(path_b).unlink(missing_ok=True)

    def test_parse_and_detect_mismatch(self):
        path_a = _write_srt("1\n00:00:01,000 --> 00:00:02,000\nA\n")
        path_b = _write_srt(
            "1\n00:00:01,000 --> 00:00:02,000\nA\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nB\n"
        )
        try:
            entries_a = parse_srt(path_a)
            entries_b = parse_srt(path_b)
            with self.assertRaises(SrtAlignmentError):
                validate_srt_alignment(entries_a, entries_b)
        finally:
            Path(path_a).unlink(missing_ok=True)
            Path(path_b).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
