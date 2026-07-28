"""Dubbing CLI and core logic for OmniVoice.

Takes a source MP3, original SRT, and translated SRT (1:1 aligned),
clones each speaker's voice from original audio segments,
synthesizes translated text, and outputs a merged WAV file.

Usage:
    omnivoice-dub --mp3 video.mp3 --srt-original sub_en.srt \
        --srt-translated sub_vi.srt --language Vietnamese --output out.wav
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable

import numpy as np
import soundfile as sf
import srt
import torch
import torchaudio
from pydub import AudioSegment

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.audio import concatenate_audio_with_silence, remove_silence
from omnivoice.utils.common import get_best_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_SAMPLE_RATE = 24000
DEFAULT_PADDING_MS = 200
SHORT_SEGMENT_THRESHOLD_S = 1.5
MAX_SPEED = 3.0
MAX_RETRIES = 2

# Matches SRT lines that are entirely bracket/parenthesis wrapped (non-speech).
# Examples: "[music playing]", "(applause)", "[♪]"
_NON_SPEECH_RE = re.compile(
    r"^\s*[\[\(][^\]\)]*[\]\)]\s*$"
    r"|^\s*\[[^\]]*\][ ]*$"
    r"|^\s*\([^\)]*\)[ ]*$"
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SrtEntry:
    """One parsed SRT subtitle entry with float-second timestamps."""

    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class DubbingRequest:
    """All parameters for a dubbing run."""

    mp3_path: str | Path
    original_srt_path: str | Path
    translated_srt_path: str | Path
    language: str
    model: str = "k2-fsa/OmniVoice"
    device: str | None = None
    output: str | Path | None = None
    padding_ms: int = DEFAULT_PADDING_MS
    short_segment_threshold_s: float = SHORT_SEGMENT_THRESHOLD_S
    max_speed: float = MAX_SPEED
    max_retries: int = MAX_RETRIES
    num_step: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    postprocess_output: bool = True


@dataclass(frozen=True)
class DubbingResult:
    """Result of a completed dubbing run."""

    wav_path: str
    total_segments: int
    successful_segments: int
    failed_indices: list[int]
    metadata: dict[str, Any]


class DubbingError(ValueError):
    """Base error for dubbing pipeline failures."""


class SrtAlignmentError(DubbingError):
    """Raised when original and translated SRTs are not 1:1 aligned."""


class SegmentGenerationError(DubbingError):
    """Raised when a single segment fails after all retries."""

    def __init__(self, index: int, original_message: str):
        self.index = index
        super().__init__(f"Segment {index}: {original_message}")


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------


def parse_srt(file_path: str | Path) -> list[SrtEntry]:
    """Parse an SRT file into a list of :class:`SrtEntry`.

    Args:
        file_path: Path to a UTF-8 encoded ``.srt`` file.

    Returns:
        Parsed entries in subtitle order.

    Raises:
        FileNotFoundError: If the file does not exist.
        srt.SRTParseError: If the file is malformed.
    """
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    subtitles = list(srt.parse(content))

    entries: list[SrtEntry] = []
    for sub in subtitles:
        text = sub.content.replace("\n", " ").strip()
        entries.append(
            SrtEntry(
                index=sub.index,
                start_sec=sub.start.total_seconds(),
                end_sec=sub.end.total_seconds(),
                text=text,
            )
        )
    return entries


def validate_srt_alignment(
    original: list[SrtEntry],
    translated: list[SrtEntry],
) -> None:
    """Ensure original and translated SRTs are 1:1 aligned.

    Both SRTs must have the same number of entries, matching indices,
    and identical timestamp ranges (within 50 ms tolerance).

    Raises:
        SrtAlignmentError: If alignment validation fails.
    """
    if len(original) != len(translated):
        raise SrtAlignmentError(
            f"SRT entry count mismatch: original has {len(original)}, "
            f"translated has {len(translated)} entries"
        )

    for i, (orig, trans) in enumerate(zip(original, translated)):
        if orig.index != trans.index:
            raise SrtAlignmentError(
                f"Index mismatch at entry {i}: "
                f"original index {orig.index}, translated index {trans.index}"
            )
        start_diff = abs(orig.start_sec - trans.start_sec)
        end_diff = abs(orig.end_sec - trans.end_sec)
        tolerance = 0.05  # 50 ms
        if start_diff > tolerance or end_diff > tolerance:
            raise SrtAlignmentError(
                f"Timestamp mismatch at entry {i} (index {orig.index}): "
                f"original [{orig.start_sec:.3f}-{orig.end_sec:.3f}], "
                f"translated [{trans.start_sec:.3f}-{trans.end_sec:.3f}] "
                f"(diff: start={start_diff:.3f}s, end={end_diff:.3f}s)"
            )


# ---------------------------------------------------------------------------
# Non-speech detection
# ---------------------------------------------------------------------------


def is_non_speech(text: str) -> bool:
    """Check if an SRT text line is non-speech (sound effect, music cue, etc.).

    Detects lines wrapped entirely in ``[...]`` or ``(...)`` brackets,
    such as ``[music playing]``, ``(applause)``, ``[♪]``.

    Args:
        text: The subtitle text to check.

    Returns:
        ``True`` if the line looks like a non-speech cue.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return bool(_NON_SPEECH_RE.match(stripped))


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def _extract_audio_segment(
    audio: AudioSegment,
    entry: SrtEntry,
    padding_ms: int,
) -> AudioSegment:
    """Slice a segment from a pydub AudioSegment with padding.

    Args:
        audio: The full audio (MP3 loaded via pydub).
        entry: SRT entry with start/end timestamps.
        padding_ms: Extra padding on each side in milliseconds.

    Returns:
        A mono pydub AudioSegment.
    """
    start_ms = max(0, int(entry.start_sec * 1000) - padding_ms)
    end_ms = min(len(audio), int(entry.end_sec * 1000) + padding_ms)

    if end_ms <= start_ms:
        raise ValueError(
            f"Invalid segment bounds for entry {entry.index}: "
            f"start={start_ms}ms, end={end_ms}ms"
        )

    segment = audio[start_ms:end_ms]
    if segment.channels > 1:
        segment = segment.set_channels(1)
    return segment


def _audiosegment_to_numpy_mono(seg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a mono float32 numpy array of shape ``(T,)``."""
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    if seg.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples


def _resample_to_output_rate(audio_np: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample 1-D numpy audio to OUTPUT_SAMPLE_RATE."""
    if source_sr == OUTPUT_SAMPLE_RATE:
        return audio_np.copy()

    t = torch.from_numpy(audio_np).float()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    resampled = torchaudio.functional.resample(
        t, orig_freq=source_sr, new_freq=OUTPUT_SAMPLE_RATE
    )
    return resampled.squeeze(0).numpy()


def _trim_silence_numpy(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim leading/trailing silence and remove mid gaps from a 1-D numpy array."""
    if audio_np.size == 0:
        return audio_np
    # Convert to (C, T) for remove_silence
    audio_2d = audio_np[np.newaxis, :]
    trimmed = remove_silence(audio_2d, sample_rate, mid_sil=300, lead_sil=100, trail_sil=100)
    return trimmed[0]  # back to (T,)


# ---------------------------------------------------------------------------
# Fallback logic
# ---------------------------------------------------------------------------


def find_fallback_entry(
    entries: list[SrtEntry],
    current_index: int,
    min_duration_s: float,
) -> SrtEntry | None:
    """Find the nearest SRT entry with duration >= *min_duration_s*.

    Used when the segment at *current_index* is too short for
    reliable voice cloning. Searches outward from *current_index*,
    preferring earlier entries when distances are equal.

    Args:
        entries: All parsed SRT entries.
        current_index: Index of the short segment.
        min_duration_s: Minimum segment duration in seconds.

    Returns:
        The nearest qualifying entry, or ``None`` if none found.
    """
    n = len(entries)
    candidates: list[tuple[int, SrtEntry]] = []
    for i, entry in enumerate(entries):
        duration = entry.end_sec - entry.start_sec
        if duration >= min_duration_s:
            candidates.append((i, entry))

    if not candidates:
        return None

    # Find nearest by index distance, preferring earlier when tied
    candidates.sort(key=lambda x: (abs(x[0] - current_index), x[0]))
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Main dubbing pipeline
# ---------------------------------------------------------------------------


def _report_progress(
    callback: Callable[[float, str], None] | None,
    fraction: float,
    message: str,
) -> None:
    if callback is not None:
        callback(fraction, message)


def run_dubbing(
    request: DubbingRequest,
    progress_callback: Callable[[float, str], None] | None = None,
    model: OmniVoice | None = None,
) -> DubbingResult:
    """Execute a full dubbing pipeline.

    Steps:
    1. Parse and validate both SRT files (1:1 alignment).
    2. Load the source MP3.
    3. Load the OmniVoice model.
    4. For each SRT entry pair:
       - Non-speech entries: preserve original audio.
       - Short segments: fall back to nearest long segment for voice cloning.
       - Voice clone from source audio, synthesize translated text.
       - Apply speed control to fit original timing (speed ≥ 1.0).
       - Retry up to *max_retries* times on failure.
    5. Concatenate all segments into a single WAV.
    6. Save and return the result.

    Args:
        request: Full dubbing configuration.
        progress_callback: Optional ``(fraction, message)`` callback
            for progress reporting (0.0–1.0).

    Returns:
        A :class:`DubbingResult` with output path and statistics.

    Raises:
        FileNotFoundError: If any input file is missing.
        SrtAlignmentError: If SRTs are not 1:1 aligned.
    """
    _report_progress(progress_callback, 0.0, "Parsing SRT files…")

    # ── 1. Parse SRTs ──────────────────────────────────────────────
    original_entries = parse_srt(request.original_srt_path)
    translated_entries = parse_srt(request.translated_srt_path)
    validate_srt_alignment(original_entries, translated_entries)

    total = len(original_entries)
    if total == 0:
        raise DubbingError("Both SRT files are empty")

    _report_progress(progress_callback, 0.05, f"Parsed {total} SRT entries")

    # ── 2. Load MP3 ───────────────────────────────────────────────
    _report_progress(progress_callback, 0.08, "Loading source audio…")
    mp3_path = Path(request.mp3_path)
    full_audio = AudioSegment.from_mp3(str(mp3_path))
    mp3_sample_rate = full_audio.frame_rate

    _report_progress(progress_callback, 0.10, "Source audio loaded")

    # ── 3. Load model ─────────────────────────────────────────────
    _report_progress(progress_callback, 0.12, "Loading OmniVoice model…")
    if model is None:
        device = request.device or get_best_device()
        logger.info("Loading model from %s on %s …", request.model, device)
        model = OmniVoice.from_pretrained(
            request.model, device_map=device, dtype=torch.float16
        )
    _report_progress(progress_callback, 0.15, "Model loaded")

    # ── 4. Process segments ───────────────────────────────────────
    work_dir = Path(mkdtemp(prefix="omnivoice_dub_"))
    dubbed_audio_chunks: list[np.ndarray] = []
    failed_indices: list[int] = []
    segment_metadata: list[dict[str, Any]] = []

    gen_kwargs: dict[str, Any] = {
        "num_step": request.num_step,
        "guidance_scale": request.guidance_scale,
        "denoise": request.denoise,
        "postprocess_output": request.postprocess_output,
    }

    for i, (orig_entry, trans_entry) in enumerate(
        zip(original_entries, translated_entries)
    ):
        progress = 0.15 + 0.75 * (i / total)
        _report_progress(
            progress_callback, progress, f"Segment {i + 1}/{total}"
        )

        original_duration = orig_entry.end_sec - orig_entry.start_sec

        try:
            # ── Non-speech → keep original audio ──────────────────
            if is_non_speech(orig_entry.text):
                seg = _extract_audio_segment(full_audio, orig_entry, request.padding_ms)
                audio_np = _audiosegment_to_numpy_mono(seg)
                audio_np = _resample_to_output_rate(audio_np, seg.frame_rate)
                dubbed_audio_chunks.append(audio_np)
                segment_metadata.append(
                    {"index": i, "type": "non_speech", "original_text": orig_entry.text}
                )
                continue

            # ── Extract source audio for voice cloning ────────────
            source_segment = _extract_audio_segment(
                full_audio, orig_entry, request.padding_ms
            )
            source_data = _audiosegment_to_numpy_mono(source_segment)
            source_data = _trim_silence_numpy(source_data, source_segment.frame_rate)

            # Determine which entry to use for voice cloning
            clone_entry = orig_entry
            seg_duration = (
                orig_entry.end_sec - orig_entry.start_sec
            ) + (request.padding_ms * 2 / 1000.0)

            if seg_duration < request.short_segment_threshold_s:
                fallback = find_fallback_entry(
                    original_entries, i, request.short_segment_threshold_s
                )
                if fallback is not None:
                    clone_entry = fallback
                    fallback_seg = _extract_audio_segment(
                        full_audio, fallback, request.padding_ms
                    )
                    fallback_data = _audiosegment_to_numpy_mono(fallback_seg)
                    source_data = _trim_silence_numpy(
                        fallback_data, fallback_seg.frame_rate
                    )
                    logger.debug(
                        "Segment %d too short (%.1fs), using fallback entry %d",
                        i,
                        seg_duration,
                        fallback.index,
                    )
                else:
                    logger.warning(
                        "Segment %d too short (%.1fs) and no fallback found",
                        i,
                        seg_duration,
                    )

            # Save reference audio to temp file for the model
            ref_path = work_dir / f"ref_{i:04d}.wav"
            sf.write(str(ref_path), source_data, source_segment.frame_rate)

            # ── Voice clone + synthesize (with retry + speed control) ─
            dubbed_audio: np.ndarray | None = None
            current_speed: float = 1.0
            last_error: str = ""

            for attempt in range(request.max_retries + 1):
                try:
                    audios = model.generate(
                        text=trans_entry.text,
                        language=request.language,
                        ref_text=clone_entry.text,
                        ref_audio=str(ref_path),
                        speed=current_speed,
                        **gen_kwargs,
                    )
                    dubbed_audio = audios[0]
                    output_duration = len(dubbed_audio) / OUTPUT_SAMPLE_RATE

                    if output_duration > original_duration:
                        required_speed = output_duration / original_duration
                        if required_speed <= request.max_speed:
                            current_speed = required_speed
                            continue  # re-generate with adjusted speed
                        else:
                            logger.warning(
                                "Segment %d needs speed %.1fx > max %.1fx; "
                                "capping at max",
                                i,
                                required_speed,
                                request.max_speed,
                            )
                            current_speed = request.max_speed
                            audios = model.generate(
                                text=trans_entry.text,
                                language=request.language,
                                ref_text=clone_entry.text,
                                ref_audio=str(ref_path),
                                speed=current_speed,
                                **gen_kwargs,
                            )
                            dubbed_audio = audios[0]
                            output_duration = len(dubbed_audio) / OUTPUT_SAMPLE_RATE

                    break  # success

                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Segment %d attempt %d/%d failed: %s",
                        i,
                        attempt + 1,
                        request.max_retries + 1,
                        exc,
                    )
                    if attempt >= request.max_retries:
                        raise SegmentGenerationError(i, last_error) from exc

            if dubbed_audio is None:
                raise RuntimeError(
                    f"Unexpected: no audio generated for segment {i}"
                )

            # ── Speed control: pad silence if output too short ───
            output_duration = len(dubbed_audio) / OUTPUT_SAMPLE_RATE
            if output_duration < original_duration:
                pad_samples = int(
                    (original_duration - output_duration) * OUTPUT_SAMPLE_RATE
                )
                logger.debug(
                    "Segment %d output %.2fs < original %.2fs; "
                    "padding %.2fs silence",
                    i,
                    output_duration,
                    original_duration,
                    pad_samples / OUTPUT_SAMPLE_RATE,
                )
                dubbed_audio = np.concatenate(
                    [dubbed_audio, np.zeros(pad_samples, dtype=np.float32)]
                )

            dubbed_audio_chunks.append(dubbed_audio)
            segment_metadata.append(
                {
                    "index": i,
                    "type": "dubbed",
                    "speed": round(current_speed, 3),
                    "original_duration_s": round(original_duration, 3),
                    "output_duration_s": round(
                        len(dubbed_audio) / OUTPUT_SAMPLE_RATE, 3
                    ),
                }
            )

        except Exception as exc:
            logger.error("Segment %d failed: %s", i, exc)
            failed_indices.append(i)
            # Insert silence matching original segment duration
            sil_frames = max(1, int(original_duration * OUTPUT_SAMPLE_RATE))
            dubbed_audio_chunks.append(np.zeros(sil_frames, dtype=np.float32))
            segment_metadata.append(
                {"index": i, "type": "failed", "error": str(exc)}
            )

    # ── 5. Merge ──────────────────────────────────────────────────
    _report_progress(progress_callback, 0.92, "Merging audio…")
    if not dubbed_audio_chunks:
        raise DubbingError("No audio segments were generated")

    merged = concatenate_audio_with_silence(
        dubbed_audio_chunks, OUTPUT_SAMPLE_RATE, silence_duration=0.0
    )

    # ── 6. Write output ───────────────────────────────────────────
    output_path = Path(request.output) if request.output else Path.cwd() / "dubbed_output.wav"
    output_path = output_path.resolve()

    # Ensure directory exists
    output_dir = output_path.parent
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    sf.write(str(output_path), merged, OUTPUT_SAMPLE_RATE)
    logger.info("Dubbed audio saved to %s", output_path)

    # ── Cleanup ───────────────────────────────────────────────────
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    _report_progress(progress_callback, 1.0, "Done")

    successful = total - len(failed_indices)
    return DubbingResult(
        wav_path=str(output_path),
        total_segments=total,
        successful_segments=successful,
        failed_indices=failed_indices,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "language": request.language,
            "model": request.model,
            "padding_ms": request.padding_ms,
            "short_segment_threshold_s": request.short_segment_threshold_s,
            "max_speed": request.max_speed,
            "max_retries": request.max_retries,
            "segments": segment_metadata,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for the ``omnivoice-dub`` CLI."""
    parser = argparse.ArgumentParser(
        description="OmniVoice Dubbing — auto dubbing from SRT subtitles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required inputs
    parser.add_argument(
        "--mp3",
        type=str,
        required=True,
        help="Path to the source MP3 audio file.",
    )
    parser.add_argument(
        "--srt-original",
        type=str,
        required=True,
        dest="srt_original",
        help="Path to the original-language SRT file.",
    )
    parser.add_argument(
        "--srt-translated",
        type=str,
        required=True,
        dest="srt_translated",
        help="Path to the translated SRT file (1:1 aligned with original).",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Target language name (e.g. 'Vietnamese') or code (e.g. 'vi').",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output WAV file path (default: ./dubbed_output.wav).",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Auto-detected if not specified.",
    )

    # Pipeline tuning
    parser.add_argument(
        "--padding-ms",
        type=int,
        default=DEFAULT_PADDING_MS,
        help="Extra padding (ms) on each side when extracting audio segments.",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=SHORT_SEGMENT_THRESHOLD_S,
        dest="short_threshold",
        help="Minimum segment duration (seconds) for reliable voice cloning. "
        "Shorter segments use the nearest qualifying segment as fallback.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=MAX_SPEED,
        help="Maximum speed factor for time-compressing dubbed audio.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help="Maximum retry attempts per failed segment.",
    )

    # Generation parameters (passed through to OmniVoice)
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument(
        "--no-denoise",
        action="store_false",
        dest="denoise",
        help="Disable denoising.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_false",
        dest="postprocess_output",
        help="Disable output post-processing (silence removal, fade).",
    )

    return parser


def main() -> None:
    """CLI entry point for ``omnivoice-dub``."""
    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)

    args = get_parser().parse_args()

    request = DubbingRequest(
        mp3_path=args.mp3,
        original_srt_path=args.srt_original,
        translated_srt_path=args.srt_translated,
        language=args.language,
        model=args.model,
        device=args.device,
        output=args.output,
        padding_ms=args.padding_ms,
        short_segment_threshold_s=args.short_threshold,
        max_speed=args.max_speed,
        max_retries=args.max_retries,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        denoise=args.denoise,
        postprocess_output=args.postprocess_output,
    )

    logger.info("Starting dubbing pipeline…")
    result = run_dubbing(request)

    logger.info(
        "Dubbing complete: %d/%d segments successful. Output: %s",
        result.successful_segments,
        result.total_segments,
        result.wav_path,
    )

    if result.failed_indices:
        logger.warning(
            "Failed segments: %s",
            ", ".join(str(i) for i in result.failed_indices),
        )


if __name__ == "__main__":
    main()
