"""Gemini-based speaker analysis for SRT subtitles.

Splits the original SRT into ~60-second Analysis Windows (bounded by
natural SRT line boundaries), sends each to Gemini for voice type
classification, and returns per-segment voice type assignments.

Handles rate limiting (15 RPM), retries with exponential backoff,
and caching via :mod:`omnivoice.speaker.cache`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from omnivoice.cli.dubbing import SrtEntry, is_non_speech, parse_srt
from omnivoice.speaker.cache import load_cache, save_cache
from omnivoice.speaker.schema import (
    GEMINI_RESPONSE_SCHEMA,
    GEMINI_SYSTEM_PROMPT,
    SegmentVoiceAssignment,
    VoiceTypeSummary,
    WindowAnalysisResult,
    make_voice_type,
    parse_gemini_response,
)

logger = logging.getLogger(__name__)

# ── Rate limit constants ────────────────────────────────────────────

RPM_LIMIT = 15  # requests per minute
MIN_INTERVAL_S = 60.0 / RPM_LIMIT  # 4 seconds between calls
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 5.0  # seconds
RETRY_MAX_DELAY_S = 120.0  # seconds

# ── Window constants ────────────────────────────────────────────────

TARGET_WINDOW_DURATION_S = 60.0  # target window duration
WINDOW_DURATION_TOLERANCE_S = 10.0  # allowed deviation (±10s)


# ── Window grouping ─────────────────────────────────────────────────


def _group_into_windows(
    entries: list[SrtEntry],
    target_duration_s: float = TARGET_WINDOW_DURATION_S,
    tolerance_s: float = WINDOW_DURATION_TOLERANCE_S,
) -> list[list[SrtEntry]]:
    """Group SRT entries into windows of approximately *target_duration_s*.

    Windows are bounded by natural SRT line boundaries — no line is split
    across two windows. Each window has total duration within
    ``[target - tolerance, target + tolerance]``.

    Args:
        entries: Parsed SRT entries in order.
        target_duration_s: Target window duration in seconds.
        tolerance_s: Allowed deviation from target.

    Returns:
        List of windows, each a list of SrtEntry.
    """
    windows: list[list[SrtEntry]] = []
    current_window: list[SrtEntry] = []
    current_duration = 0.0

    for entry in entries:
        entry_duration = entry.end_sec - entry.start_sec

        if current_duration + entry_duration > target_duration_s + tolerance_s and current_window:
            # Current window is full — close it
            windows.append(current_window)
            current_window = []
            current_duration = 0.0

        current_window.append(entry)
        current_duration += entry_duration

    # Don't forget the last window
    if current_window:
        windows.append(current_window)

    logger.info(
        "Grouped %d SRT entries into %d windows "
        "(avg %.1f entries/window, target %.0fs)",
        len(entries),
        len(windows),
        len(entries) / len(windows) if windows else 0,
        target_duration_s,
    )
    return windows


def _format_window_for_gemini(window: list[SrtEntry]) -> str:
    """Format a window of SRT entries as raw SRT text for Gemini analysis.

    Args:
        window: List of SRT entries in one window.

    Returns:
        Raw SRT text (subset of original format).
    """
    lines: list[str] = []
    for entry in window:
        start_str = _format_timestamp(entry.start_sec)
        end_str = _format_timestamp(entry.end_sec)
        lines.append(str(entry.index))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(entry.text)
        lines.append("")  # blank line between entries
    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp ``HH:MM:SS,mmm``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Gemini API call ─────────────────────────────────────────────────


def _call_gemini(
    client: Any,  # google.genai.Client
    window_text: str,
    window_index: int,
) -> WindowAnalysisResult:
    """Send one window to Gemini and parse the response.

    Args:
        client: A ``google.genai.Client`` instance.
        window_text: Formatted SRT text for this window.
        window_index: 0-based window index.

    Returns:
        Parsed analysis result.

    Raises:
        RuntimeError: If Gemini returns invalid or unparseable output.
    """
    from google.genai import types as genai_types

    prompt = (
        f"{GEMINI_SYSTEM_PROMPT}\n\n"
        f"Analyze the following SRT segment (window {window_index + 1}). "
        f"Return the voice type for each spoken line:\n\n{window_text}"
    )

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GEMINI_RESPONSE_SCHEMA,
        temperature=0.1,  # low temperature for consistent classification
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=config,
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        # Try response.parsed as fallback
        try:
            data = response.parsed
        except AttributeError:
            raise RuntimeError(
                f"Gemini response for window {window_index} is not valid JSON: "
                f"{response.text[:200]}..."
            ) from None

    result = parse_gemini_response(data)
    # Override window_index since parse_gemini_response defaults to 0
    return WindowAnalysisResult(
        window_index=window_index,
        segments=result.segments,
        summary=result.summary,
    )


def _call_gemini_with_retry(
    client: Any,
    window_text: str,
    window_index: int,
    max_retries: int = MAX_RETRIES,
) -> WindowAnalysisResult:
    """Call Gemini with retry and exponential backoff.

    Args:
        client: A ``google.genai.Client`` instance.
        window_text: Formatted SRT text for this window.
        window_index: 0-based window index.
        max_retries: Maximum retry attempts.

    Returns:
        Parsed analysis result.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return _call_gemini(client, window_text, window_index)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = min(
                    RETRY_BASE_DELAY_S * (2**attempt),
                    RETRY_MAX_DELAY_S,
                )
                logger.warning(
                    "Gemini call for window %d failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs...",
                    window_index,
                    attempt + 1,
                    max_retries + 1,
                    e,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Gemini call for window %d exhausted all %d retries: %s",
                    window_index,
                    max_retries + 1,
                    e,
                )

    raise RuntimeError(
        f"Gemini analysis failed for window {window_index} "
        f"after {max_retries + 1} attempts: {last_error}"
    ) from last_error


# ── Rate limiter ────────────────────────────────────────────────────


class _RateLimiter:
    """Simple rate limiter for Gemini API calls."""

    def __init__(self, min_interval_s: float = MIN_INTERVAL_S):
        self._min_interval = min_interval_s
        self._last_call = 0.0

    def wait(self) -> None:
        """Sleep if necessary to respect the rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


# ── Public API ──────────────────────────────────────────────────────


def analyze_speakers(
    srt_path: str,
    api_key: str,
    cache_dir: str | None = None,
    progress_callback: Any | None = None,
) -> list[WindowAnalysisResult]:
    """Analyze the original SRT to identify voice types per segment.

    This is the main entry point. It:
    1. Reads and parses the SRT file
    2. Checks the cache (keyed by SHA256 of SRT content)
    3. Groups entries into ~60s windows
    4. Sends each window to Gemini with rate limiting and retry
    5. Caches results for future runs

    Args:
        srt_path: Path to the original ``.srt`` file.
        api_key: Gemini API key.
        cache_dir: Optional custom cache directory path.
        progress_callback: Optional ``(fraction, message)`` callback.

    Returns:
        List of :class:`WindowAnalysisResult`, one per Analysis Window.

    Raises:
        FileNotFoundError: If the SRT file doesn't exist.
        ImportError: If ``google-genai`` is not installed.
        RuntimeError: If Gemini analysis fails for a window.
    """
    try:
        from google import genai as genai_module  # noqa: F401
    except ImportError:
        raise ImportError(
            "google-genai is required for speaker analysis. "
            "Install it with: pip install google-genai"
        ) from None

    # Read SRT content for caching
    srt_content = _read_srt_content(srt_path)

    # Check cache
    from pathlib import Path

    _cache_dir = Path(cache_dir) if cache_dir else None
    cached = load_cache(srt_content, _cache_dir)
    if cached is not None:
        if progress_callback:
            progress_callback(1.0, "Loaded from cache")
        return cached

    # Parse SRT
    entries = parse_srt(srt_path)
    if not entries:
        logger.warning("SRT file is empty: %s", srt_path)
        return []

    # Group into windows
    windows = _group_into_windows(entries)
    total_windows = len(windows)
    if total_windows == 0:
        return []

    if progress_callback:
        progress_callback(0.0, f"Analyzing {total_windows} windows...")

    # Initialize Gemini client
    client = genai_module.Client(api_key=api_key)
    rate_limiter = _RateLimiter()

    results: list[WindowAnalysisResult] = []
    failed_windows: list[int] = []

    for i, window in enumerate(windows):
        window_text = _format_window_for_gemini(window)

        try:
            rate_limiter.wait()
            result = _call_gemini_with_retry(client, window_text, i)
            results.append(result)
            logger.info(
                "Window %d/%d: found %d voice types",
                i + 1,
                total_windows,
                len(result.summary),
            )
        except RuntimeError as e:
            logger.warning("Window %d/%d failed: %s", i + 1, total_windows, e)
            failed_windows.append(i)

        if progress_callback:
            fraction = (i + 1) / total_windows
            progress_callback(
                fraction,
                f"Window {i + 1}/{total_windows}"
                + (f" ({len(failed_windows)} failed)" if failed_windows else ""),
            )

    if failed_windows:
        logger.warning(
            "%d/%d windows failed Gemini analysis; "
            "these will use per-segment audio cloning",
            len(failed_windows),
            total_windows,
        )

    # Save to cache
    if results:
        save_cache(srt_content, results, _cache_dir)

    if progress_callback:
        progress_callback(
            1.0,
            f"Analysis complete: {len(results)}/{total_windows} windows analyzed, "
            f"{len(failed_windows)} failed",
        )

    return results


def _read_srt_content(srt_path: str) -> str:
    """Read raw SRT file content for caching purposes."""
    from pathlib import Path

    return Path(srt_path).read_text(encoding="utf-8")


def merge_voice_types(
    results: list[WindowAnalysisResult],
) -> list[tuple[str, int]]:
    """Merge voice types across all windows, deduplicating and sorting by frequency.

    Args:
        results: Analysis results from :func:`analyze_speakers`.

    Returns:
        List of ``(voice_type_label, total_segment_count)`` sorted by count descending.
    """
    counts: dict[str, int] = {}
    for r in results:
        for s in r.summary:
            vt = make_voice_type(s.gender, s.age)
            counts[vt] = counts.get(vt, 0) + s.segment_count
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def build_segment_voice_map(
    results: list[WindowAnalysisResult],
) -> dict[int, str]:
    """Build a mapping from SRT index → voice_type for all segments.

    Args:
        results: Analysis results from :func:`analyze_speakers`.

    Returns:
        Dict mapping 1-based SRT index to voice_type label.
    """
    mapping: dict[int, str] = {}
    for r in results:
        for seg in r.segments:
            mapping[seg.index] = seg.voice_type
    return mapping
