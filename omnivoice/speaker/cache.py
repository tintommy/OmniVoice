"""Disk cache for Gemini speaker analysis results.

Caches results keyed by SHA256 hash of the original SRT content.
This avoids redundant API calls when re-dubbing the same video with
different translated SRT files.

Cache location: ``~/.omnivoice/cache/<hash>.json``
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from omnivoice.speaker.schema import WindowAnalysisResult

logger = logging.getLogger(__name__)

#: Default cache directory.
DEFAULT_CACHE_DIR = Path.home() / ".omnivoice" / "cache"


def _compute_srt_hash(srt_content: str) -> str:
    """Compute SHA256 hex digest of SRT content."""
    return hashlib.sha256(srt_content.encode("utf-8")).hexdigest()


def _cache_path(srt_hash: str, cache_dir: Path | None = None) -> Path:
    """Get the cache file path for a given SRT hash."""
    directory = cache_dir or DEFAULT_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{srt_hash}.json"


def _result_to_dict(results: list[WindowAnalysisResult]) -> list[dict[str, Any]]:
    """Serialize analysis results to JSON-compatible dicts."""
    serialized: list[dict[str, Any]] = []
    for r in results:
        serialized.append(
            {
                "window_index": r.window_index,
                "segments": [
                    {"index": s.index, "gender": s.gender, "age": s.age}
                    for s in r.segments
                ],
                "summary": [
                    {
                        "gender": v.gender,
                        "age": v.age,
                        "segment_count": v.segment_count,
                    }
                    for v in r.summary
                ],
            }
        )
    return serialized


def _dict_to_result(data: list[dict[str, Any]]) -> list[WindowAnalysisResult]:
    """Deserialize JSON dicts back to analysis results."""
    from omnivoice.speaker.schema import SegmentVoiceAssignment, VoiceTypeSummary

    results: list[WindowAnalysisResult] = []
    for w in data:
        segments = [
            SegmentVoiceAssignment(
                index=s["index"],
                gender=s["gender"],
                age=s["age"],
            )
            for s in w["segments"]
        ]
        summary = [
            VoiceTypeSummary(
                gender=v["gender"],
                age=v["age"],
                segment_count=v["segment_count"],
            )
            for v in w["summary"]
        ]
        results.append(
            WindowAnalysisResult(
                window_index=w["window_index"],
                segments=segments,
                summary=summary,
            )
        )
    return results


def load_cache(srt_content: str, cache_dir: Path | None = None) -> list[WindowAnalysisResult] | None:
    """Load cached analysis results for the given SRT content.

    Args:
        srt_content: The full original SRT file content.
        cache_dir: Optional custom cache directory.

    Returns:
        Cached results if found and valid, ``None`` otherwise.
    """
    srt_hash = _compute_srt_hash(srt_content)
    path = _cache_path(srt_hash, cache_dir)

    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        results = _dict_to_result(raw)
        logger.info("Loaded cached analysis from %s (%d windows)", path, len(results))
        return results
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Cache file %s is corrupt: %s — ignoring", path, e)
        path.unlink(missing_ok=True)
        return None


def save_cache(
    srt_content: str,
    results: list[WindowAnalysisResult],
    cache_dir: Path | None = None,
) -> Path:
    """Save analysis results to cache.

    Args:
        srt_content: The full original SRT file content.
        results: Analysis results to cache.
        cache_dir: Optional custom cache directory.

    Returns:
        Path to the cache file.
    """
    srt_hash = _compute_srt_hash(srt_content)
    path = _cache_path(srt_hash, cache_dir)

    data = _result_to_dict(results)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved analysis cache to %s (%d windows)", path, len(results))
    return path
