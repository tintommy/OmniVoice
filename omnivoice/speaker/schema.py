"""Schema definitions for Gemini speaker analysis.

Defines the VoiceType vocabulary, the Gemini response schema,
and helper functions for working with voice type labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Gender(str, Enum):
    """Speaker gender classification."""

    MALE = "male"
    FEMALE = "female"


class Age(str, Enum):
    """Speaker approximate age classification."""

    CHILD = "child"
    YOUNG = "young"
    MIDDLE_AGED = "middle-aged"
    OLD = "old"


#: Canonical ordered list of all 8 VoiceType combinations.
ALL_VOICE_TYPES: list[str] = [
    f"{g.value}-{a.value}" for g in Gender for a in Age
]


def make_voice_type(gender: str, age: str) -> str:
    """Combine gender and age into a VoiceType label.

    Args:
        gender: ``"male"`` or ``"female"``.
        age: ``"child"``, ``"young"``, ``"middle-aged"``, or ``"old"``.

    Returns:
        VoiceType label like ``"male-young"``.
    """
    return f"{gender}-{age}"


def parse_voice_type(label: str) -> tuple[str, str]:
    """Split a VoiceType label into (gender, age).

    Args:
        label: VoiceType label like ``"female-middle-aged"``.

    Returns:
        Tuple of ``(gender, age)`` strings.

    Raises:
        ValueError: If the label is not a valid VoiceType.
    """
    try:
        gender_str, age_str = label.split("-", 1)
    except ValueError:
        raise ValueError(f"Invalid voice type label: {label!r}") from None
    try:
        Gender(gender_str)
        Age(age_str)
    except ValueError:
        raise ValueError(f"Invalid voice type label: {label!r}") from None
    return gender_str, age_str


@dataclass(frozen=True)
class SegmentVoiceAssignment:
    """Gemini-assigned voice type for a single SRT segment."""

    index: int
    """SRT subtitle index (1-based, matching the SRT file)."""
    gender: str
    """``"male"`` or ``"female"``."""
    age: str
    """``"child"``, ``"young"``, ``"middle-aged"``, or ``"old"``."""

    @property
    def voice_type(self) -> str:
        """Combined label like ``"male-young"``."""
        return make_voice_type(self.gender, self.age)


@dataclass(frozen=True)
class VoiceTypeSummary:
    """Aggregated count for one VoiceType across an Analysis Window."""

    gender: str
    age: str
    segment_count: int

    @property
    def voice_type(self) -> str:
        """Combined label like ``"female-old"``."""
        return make_voice_type(self.gender, self.age)


@dataclass(frozen=True)
class WindowAnalysisResult:
    """Result of analyzing a single Analysis Window."""

    window_index: int
    """0-based index of this window."""
    segments: list[SegmentVoiceAssignment]
    """Per-segment voice type assignments."""
    summary: list[VoiceTypeSummary]
    """Aggregated voice type counts for this window."""


# ── Gemini response schema ──────────────────────────────────────────

# The schema describes the JSON object Gemini must return.
# It uses dict format compatible with google-genai's response_schema parameter.

GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "gender": {
                        "type": "STRING",
                        "enum": [g.value for g in Gender],
                    },
                    "age": {
                        "type": "STRING",
                        "enum": [a.value for a in Age],
                    },
                },
                "required": ["index", "gender", "age"],
            },
        },
        "voice_types_summary": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "gender": {
                        "type": "STRING",
                        "enum": [g.value for g in Gender],
                    },
                    "age": {
                        "type": "STRING",
                        "enum": [a.value for a in Age],
                    },
                    "segment_count": {"type": "INTEGER"},
                },
                "required": ["gender", "age", "segment_count"],
            },
        },
    },
    "required": ["segments", "voice_types_summary"],
}


# ── Prompt template ─────────────────────────────────────────────────

GEMINI_SYSTEM_PROMPT = """\
You are a dialogue analyst for film/TV subtitles. Your task is to analyze
an SRT subtitle segment and classify each spoken line by the speaker's
gender and approximate age.

## Rules
- Consecutive lines with little or no time gap are likely the same speaker.
- Different vocabulary, formality, or tone suggests a different speaker.
- Lines wrapped in brackets [...] or parentheses (...) are non-speech cues.
  Omit them entirely — do not include them in your output.
- Minimize distinct voices: only create a new voice type when a speaker
  is clearly different from all previously identified ones.
- Assign gender ("male"/"female") and age ("child"/"young"/"middle-aged"/"old")
  based on dialogue content and context.
- Age guidelines: child = under ~15, young = ~15–30, middle-aged = ~30–55,
  old = ~55+.

## Output Format
Return a JSON object with two fields:
1. "segments": array of {index, gender, age} — one per spoken SRT line
2. "voice_types_summary": array of {gender, age, segment_count} —
   aggregated count per voice type"""


def parse_gemini_response(data: dict[str, Any]) -> WindowAnalysisResult:
    """Parse the Gemini JSON response into typed dataclasses.

    Args:
        data: Parsed JSON dict from Gemini's ``response.parsed`` or
            ``json.loads(response.text)``.

    Returns:
        A fully typed :class:`WindowAnalysisResult`.

    Raises:
        ValueError: If the response format is invalid.
    """
    raw_segments: list[dict[str, Any]] = data.get("segments", [])
    raw_summary: list[dict[str, Any]] = data.get("voice_types_summary", [])

    segments = [
        SegmentVoiceAssignment(
            index=int(s["index"]),
            gender=str(s["gender"]),
            age=str(s["age"]),
        )
        for s in raw_segments
    ]

    summary = [
        VoiceTypeSummary(
            gender=str(s["gender"]),
            age=str(s["age"]),
            segment_count=int(s["segment_count"]),
        )
        for s in raw_summary
    ]

    return WindowAnalysisResult(
        window_index=0,  # caller sets this
        segments=segments,
        summary=summary,
    )
