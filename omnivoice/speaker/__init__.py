"""Speaker analysis module for OmniVoice dubbing.

Provides Gemini-powered SRT analysis to identify distinct voice types
for voice-sample-based dubbing.

Public API:
- :func:`analyze_speakers` — Run full analysis pipeline
- :func:`merge_voice_types` — Merge and deduplicate results across windows
- :func:`build_segment_voice_map` — Build index → voice_type mapping
- :data:`ALL_VOICE_TYPES` — Canonical list of 8 voice type labels
- :func:`make_voice_type` — Combine gender + age into label
- :func:`parse_voice_type` — Parse label into (gender, age)
"""

from omnivoice.speaker.analyzer import (
    analyze_speakers,
    build_segment_voice_map,
    merge_voice_types,
)
from omnivoice.speaker.cache import load_cache, save_cache
from omnivoice.speaker.schema import (
    ALL_VOICE_TYPES,
    GEMINI_RESPONSE_SCHEMA,
    GEMINI_SYSTEM_PROMPT,
    Age,
    Gender,
    SegmentVoiceAssignment,
    VoiceTypeSummary,
    WindowAnalysisResult,
    make_voice_type,
    parse_voice_type,
)

__all__ = [
    "ALL_VOICE_TYPES",
    "Age",
    "GEMINI_RESPONSE_SCHEMA",
    "GEMINI_SYSTEM_PROMPT",
    "Gender",
    "SegmentVoiceAssignment",
    "VoiceTypeSummary",
    "WindowAnalysisResult",
    "analyze_speakers",
    "build_segment_voice_map",
    "load_cache",
    "make_voice_type",
    "merge_voice_types",
    "parse_voice_type",
    "save_cache",
]
