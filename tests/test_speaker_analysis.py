"""Tests for omnivoice.speaker — schema, cache, analyzer (unit-testable parts).

Uses dynamic import to avoid loading heavy ML dependencies.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic imports to bypass transformers/torch
# ---------------------------------------------------------------------------

SPEAKER_DIR = Path(__file__).resolve().parents[1] / "omnivoice" / "speaker"
DUBBING_PATH = Path(__file__).resolve().parents[1] / "omnivoice" / "cli" / "dubbing.py"


def _dynamic_import(module_name: str, file_path: Path) -> types.ModuleType:
    """Import a Python module without loading the parent package."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load speaker modules
schema_mod = _dynamic_import("omnivoice.speaker.schema", SPEAKER_DIR / "schema.py")
cache_mod = _dynamic_import("omnivoice.speaker.cache", SPEAKER_DIR / "cache.py")

# Mock heavy deps for dubbing module import
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch"].inference_mode = lambda: (lambda f: f)
sys.modules["torch"].Tensor = type("Tensor", (), {})
sys.modules["torch"].float16 = None
sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
sys.modules["torchaudio"].functional = types.ModuleType("torchaudio.functional")
sys.modules.setdefault("omnivoice", types.ModuleType("omnivoice"))
sys.modules.setdefault("omnivoice.models", types.ModuleType("omnivoice.models"))
sys.modules.setdefault("omnivoice.models.omnivoice", types.ModuleType("omnivoice.models.omnivoice"))
sys.modules["omnivoice.models.omnivoice"].OmniVoice = type("OmniVoice", (), {})
sys.modules.setdefault("omnivoice.utils", types.ModuleType("omnivoice.utils"))
sys.modules.setdefault("omnivoice.utils.audio", types.ModuleType("omnivoice.utils.audio"))
sys.modules["omnivoice.utils.audio"].remove_silence = lambda audio, sampling_rate, mid_sil=300, lead_sil=100, trail_sil=300: audio
sys.modules.setdefault("omnivoice.utils.common", types.ModuleType("omnivoice.utils.common"))
sys.modules["omnivoice.utils.common"].get_best_device = lambda: "cpu"
import numpy as np
sys.modules["numpy"] = np
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules["soundfile"].write = lambda *a, **kw: None
sys.modules["soundfile"].read = lambda *a, **kw: (None, 24000)
sys.modules.setdefault("pydub", types.ModuleType("pydub"))
sys.modules["pydub"].AudioSegment = type("AudioSegment", (), {
    "channels": 1,
    "frame_rate": 24000,
})

dubbing_mod = _dynamic_import("omnivoice.cli.dubbing", DUBBING_PATH)

# Setup for analyzer import
sys.modules["omnivoice.speaker"] = types.ModuleType("omnivoice.speaker")
analyzer_mod = _dynamic_import("omnivoice.speaker.analyzer", SPEAKER_DIR / "analyzer.py")

# ── Aliases ────────────────────────────────────────────────────────
make_voice_type = schema_mod.make_voice_type
parse_voice_type = schema_mod.parse_voice_type
parse_gemini_response = schema_mod.parse_gemini_response
ALL_VOICE_TYPES = schema_mod.ALL_VOICE_TYPES
GEMINI_RESPONSE_SCHEMA = schema_mod.GEMINI_RESPONSE_SCHEMA
Gender = schema_mod.Gender
Age = schema_mod.Age
SegmentVoiceAssignment = schema_mod.SegmentVoiceAssignment
VoiceTypeSummary = schema_mod.VoiceTypeSummary
WindowAnalysisResult = schema_mod.WindowAnalysisResult

_compute_srt_hash = cache_mod._compute_srt_hash
_result_to_dict = cache_mod._result_to_dict
_dict_to_result = cache_mod._dict_to_result
load_cache = cache_mod.load_cache
save_cache = cache_mod.save_cache

_format_timestamp = analyzer_mod._format_timestamp
_format_window_for_gemini = analyzer_mod._format_window_for_gemini
_group_into_windows = analyzer_mod._group_into_windows
merge_voice_types = analyzer_mod.merge_voice_types
build_segment_voice_map = analyzer_mod.build_segment_voice_map

SrtEntry = dubbing_mod.SrtEntry
parse_srt = dubbing_mod.parse_srt


def _write_srt(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", encoding="utf-8", delete=False
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestVoiceType(unittest.TestCase):
    def test_all_combinations(self):
        expected = {
            "male-child", "male-young", "male-middle-aged", "male-old",
            "female-child", "female-young", "female-middle-aged", "female-old",
        }
        self.assertEqual(set(ALL_VOICE_TYPES), expected)
        self.assertEqual(len(ALL_VOICE_TYPES), 8)

    def test_make(self):
        self.assertEqual(make_voice_type("male", "young"), "male-young")

    def test_parse_valid(self):
        g, a = parse_voice_type("male-middle-aged")
        self.assertEqual((g, a), ("male", "middle-aged"))

    def test_parse_invalid(self):
        with self.assertRaises(ValueError):
            parse_voice_type("robot-adult")

    def test_assignment_voice_type(self):
        s = SegmentVoiceAssignment(index=1, gender="female", age="child")
        self.assertEqual(s.voice_type, "female-child")


class TestEnums(unittest.TestCase):
    def test_gender(self):
        self.assertEqual(Gender.MALE.value, "male")
        self.assertEqual(Gender.FEMALE.value, "female")

    def test_age(self):
        self.assertEqual(Age.CHILD.value, "child")
        self.assertEqual(Age.OLD.value, "old")


class TestResponseSchema(unittest.TestCase):
    def test_structure(self):
        self.assertEqual(GEMINI_RESPONSE_SCHEMA["type"], "OBJECT")
        self.assertIn("segments", GEMINI_RESPONSE_SCHEMA["required"])


class TestParseGeminiResponse(unittest.TestCase):
    def test_valid(self):
        data = {
            "segments": [
                {"index": 1, "gender": "male", "age": "young"},
                {"index": 2, "gender": "female", "age": "middle-aged"},
            ],
            "voice_types_summary": [
                {"gender": "male", "age": "young", "segment_count": 2},
                {"gender": "female", "age": "middle-aged", "segment_count": 1},
            ],
        }
        result = parse_gemini_response(data)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].voice_type, "male-young")

    def test_empty(self):
        result = parse_gemini_response({"segments": [], "voice_types_summary": []})
        self.assertEqual(len(result.segments), 0)

    def test_missing_field(self):
        with self.assertRaises(KeyError):
            parse_gemini_response({"segments": [{"index": 1}]})


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCacheHash(unittest.TestCase):
    def test_same(self):
        self.assertEqual(_compute_srt_hash("hello"), _compute_srt_hash("hello"))

    def test_different(self):
        self.assertNotEqual(_compute_srt_hash("a"), _compute_srt_hash("b"))

    def test_hex(self):
        self.assertEqual(len(_compute_srt_hash("x")), 64)


class TestCacheRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="omnivoice_test_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_result(self, wi: int) -> WindowAnalysisResult:
        return WindowAnalysisResult(
            window_index=wi,
            segments=[SegmentVoiceAssignment(1, "male", "young")],
            summary=[VoiceTypeSummary("male", "young", 1)],
        )

    def test_save_and_load(self):
        srt = "SRT content here"
        results = [self._make_result(0), self._make_result(1)]
        save_cache(srt, results, self.tmpdir)
        loaded = load_cache(srt, self.tmpdir)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].window_index, 0)

    def test_miss(self):
        self.assertIsNone(load_cache("no-cache", self.tmpdir))

    def test_corrupt(self):
        srt = "corrupt me"
        h = _compute_srt_hash(srt)
        (self.tmpdir / f"{h}.json").write_text("{invalid")
        self.assertIsNone(load_cache(srt, self.tmpdir))


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        results = [
            WindowAnalysisResult(0,
                [SegmentVoiceAssignment(1, "male", "young")],
                [VoiceTypeSummary("male", "young", 1)],
            )
        ]
        restored = _dict_to_result(_result_to_dict(results))
        self.assertEqual(restored[0].segments[0].voice_type, "male-young")


# ---------------------------------------------------------------------------
# Analyzer unit tests
# ---------------------------------------------------------------------------

class TestFormatTimestamp(unittest.TestCase):
    def test_values(self):
        self.assertEqual(_format_timestamp(0.0), "00:00:00,000")
        self.assertEqual(_format_timestamp(1.5), "00:00:01,500")
        self.assertEqual(_format_timestamp(3661.123), "01:01:01,123")


class TestGroupIntoWindows(unittest.TestCase):
    def _e(self, idx, start, end) -> SrtEntry:
        return SrtEntry(index=idx, start_sec=start, end_sec=end, text="x")

    def test_single_short(self):
        w = _group_into_windows([self._e(1, 0, 5), self._e(2, 6, 10)])
        self.assertEqual(len(w), 1)
        self.assertEqual(len(w[0]), 2)

    def test_multi(self):
        entries = [self._e(i, i*30, i*30+20) for i in range(1, 4)]
        w = _group_into_windows(entries, target_duration_s=40, tolerance_s=15)
        self.assertGreater(len(w), 1)

    def test_empty(self):
        self.assertEqual(len(_group_into_windows([])), 0)

    def test_no_split(self):
        entries = [self._e(1, 0, 90), self._e(2, 91, 100)]
        w = _group_into_windows(entries, target_duration_s=60)
        self.assertEqual(sum(len(win) for win in w), 2)


class TestFormatWindowForGemini(unittest.TestCase):
    def test_basic(self):
        window = [SrtEntry(index=1, start_sec=1.0, end_sec=3.5, text="Hello")]
        fmt = _format_window_for_gemini(window)
        self.assertIn("00:00:01,000 --> 00:00:03,500", fmt)
        self.assertIn("Hello", fmt)


class TestMergeVoiceTypes(unittest.TestCase):
    def test_dedup_sort(self):
        results = [
            WindowAnalysisResult(0, [], [
                VoiceTypeSummary("male", "young", 10),
                VoiceTypeSummary("female", "old", 3),
            ]),
            WindowAnalysisResult(1, [], [
                VoiceTypeSummary("male", "young", 5),
                VoiceTypeSummary("female", "young", 7),
            ]),
        ]
        merged = merge_voice_types(results)
        self.assertEqual(merged, [
            ("male-young", 15), ("female-young", 7), ("female-old", 3),
        ])

    def test_empty(self):
        self.assertEqual(len(merge_voice_types([])), 0)


class TestBuildSegmentVoiceMap(unittest.TestCase):
    def test_build(self):
        results = [
            WindowAnalysisResult(0, [
                SegmentVoiceAssignment(1, "male", "young"),
                SegmentVoiceAssignment(2, "female", "old"),
            ], []),
            WindowAnalysisResult(1, [
                SegmentVoiceAssignment(3, "male", "young"),
            ], []),
        ]
        m = build_segment_voice_map(results)
        self.assertEqual(m[1], "male-young")
        self.assertEqual(m[2], "female-old")
        self.assertEqual(m[3], "male-young")


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

class TestSrtParsing(unittest.TestCase):
    def test_two_entries(self):
        path = _write_srt(
            "1\n00:00:01,000 --> 00:00:03,500\nHello world\n\n"
            "2\n00:00:04,000 --> 00:00:06,200\nHow are you?\n"
        )
        r = parse_srt(path)
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0].text, "Hello world")


if __name__ == "__main__":
    unittest.main()
