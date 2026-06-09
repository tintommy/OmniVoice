from __future__ import annotations

import csv
import importlib.util
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir, mkdtemp
from typing import Any, Iterable, Sequence

import soundfile as sf

_AUDIO_UTILS_PATH = Path(__file__).resolve().parents[1] / "utils" / "audio.py"
_AUDIO_UTILS_SPEC = importlib.util.spec_from_file_location(
    "omnivoice_audio_utils",
    _AUDIO_UTILS_PATH,
)
if _AUDIO_UTILS_SPEC is None or _AUDIO_UTILS_SPEC.loader is None:
    raise ImportError(f"Unable to load audio utilities from {_AUDIO_UTILS_PATH}")
_AUDIO_UTILS = importlib.util.module_from_spec(_AUDIO_UTILS_SPEC)
_AUDIO_UTILS_SPEC.loader.exec_module(_AUDIO_UTILS)
concatenate_audio_with_silence = _AUDIO_UTILS.concatenate_audio_with_silence

DEFAULT_OUTPUT_FORMAT = "WAV"
DEFAULT_DOWNLOAD_FORMAT = "ZIP"
MAX_QUEUE_ITEMS = 20
MAX_ITEM_CHARACTERS = 3000
MAX_TOTAL_CHARACTERS = 20000
CSV_COLUMNS = ("text",)


@dataclass(frozen=True)
class QueuedCloneItem:
    text: str


@dataclass(frozen=True)
class VoiceCloneQueueRequest:
    items: list[QueuedCloneItem]
    ref_audio: Any
    ref_text: str | None
    language: str | None
    instruct: str | None = None
    output_dir: Path | None = None
    speed: float = 1.0
    duration: float | None = None
    apply_pause_between_files: bool = True
    pause_between_files_ms: int = 300


@dataclass(frozen=True)
class VoiceCloneQueueValidationSummary:
    queue_items: int
    total_characters: int
    max_total_characters: int = MAX_TOTAL_CHARACTERS
    output_format: str = DEFAULT_OUTPUT_FORMAT
    download_all_format: str = DEFAULT_DOWNLOAD_FORMAT


@dataclass(frozen=True)
class VoiceCloneQueueResult:
    wav_paths: list[str]
    zip_path: str
    metadata: dict[str, Any]
    merged_wav_path: str


class VoiceCloneQueueError(ValueError):
    pass


class ValidationStateError(VoiceCloneQueueError):
    pass


class VoiceCloneQueueGenerationError(VoiceCloneQueueError):
    pass


def _report_progress(progress: Any, value: float, description: str) -> None:
    if progress is None:
        return
    progress(value, desc=description)


def _trim_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_empty_cell(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _coerce_rows(data: Any) -> list[Any]:
    if data is None:
        return []

    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict(orient="records")
        except TypeError:
            records = data.to_dict()
        if isinstance(records, list):
            return records

    if isinstance(data, list):
        return data

    if isinstance(data, tuple):
        return list(data)

    return []


def normalize_queue_rows(data: Any) -> list[dict[str, str]]:
    rows = _coerce_rows(data)
    normalized: list[dict[str, str]] = []

    for row in rows:
        if isinstance(row, dict):
            text = _trim_string(row.get("text"))
            extra_values: Iterable[Any] = row.values()
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            text = _trim_string(row[0] if len(row) > 0 else "")
            extra_values = row
        else:
            continue

        if all(_is_empty_cell(value) for value in extra_values):
            continue

        if not text:
            continue

        normalized.append({"text": text})

    return normalized


def queue_items_from_rows(data: Any) -> list[QueuedCloneItem]:
    return [QueuedCloneItem(**row) for row in normalize_queue_rows(data)]


def _csv_safe_cell(value: str) -> str:
    text = _trim_string(value)
    if text and text[0] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def export_sample_queue_csv() -> str:
    return "text\n"


def export_queue_csv(items: Sequence[QueuedCloneItem]) -> str:
    lines = ["text"]
    for item in items:
        text = _csv_safe_cell(item.text)
        if any(char in text for char in [",", '"', "\n", "\r"]):
            text = '"' + text.replace('"', '""') + '"'
        lines.append(text)
    return "\n".join(lines) + "\n"


def import_queue_csv(csv_text: str) -> list[QueuedCloneItem]:
    csv_text = csv_text.lstrip("\ufeff")
    reader = csv.DictReader(csv_text.splitlines())
    fieldnames = [name.lstrip("\ufeff") for name in (reader.fieldnames or [])]
    if fieldnames != list(CSV_COLUMNS):
        raise VoiceCloneQueueError("CSV must contain exactly one column: text")
    reader.fieldnames = fieldnames

    items: list[QueuedCloneItem] = []
    for row in reader:
        text = _trim_string(row.get("text"))
        if not text:
            continue
        items.append(QueuedCloneItem(text=text))
    return items


def validate_queue_request(request: VoiceCloneQueueRequest) -> VoiceCloneQueueValidationSummary:
    if request.ref_audio is None:
        raise VoiceCloneQueueError("Reference Audio is required.")
    if not request.items:
        raise VoiceCloneQueueError("At least one Queued Clone Item is required.")
    if len(request.items) > MAX_QUEUE_ITEMS:
        raise VoiceCloneQueueError(f"Queue must not exceed {MAX_QUEUE_ITEMS} items.")

    total_characters = 0
    for index, item in enumerate(request.items, start=1):
        text = _trim_string(item.text)
        if not text:
            raise VoiceCloneQueueError(f"Queued Clone Item {index} must provide text.")
        if len(text) > MAX_ITEM_CHARACTERS:
            raise VoiceCloneQueueError(
                f"Each Queued Clone Item must not exceed {MAX_ITEM_CHARACTERS} characters."
            )
        total_characters += len(text)

    if total_characters > MAX_TOTAL_CHARACTERS:
        raise VoiceCloneQueueError(
            f"Queue must not exceed {MAX_TOTAL_CHARACTERS} total characters."
        )

    if request.apply_pause_between_files:
        if not isinstance(request.pause_between_files_ms, int):
            raise VoiceCloneQueueError("pause_between_files_ms must be an integer.")
        if request.pause_between_files_ms < 0 or request.pause_between_files_ms > 5000:
            raise VoiceCloneQueueError("pause_between_files_ms must be between 0 and 5000.")

    return VoiceCloneQueueValidationSummary(
        queue_items=len(request.items),
        total_characters=total_characters,
    )


def validation_data_signature(request: VoiceCloneQueueRequest) -> dict[str, Any]:
    return {
        "items": [item.text for item in request.items],
        "ref_audio": request.ref_audio,
        "ref_text": _trim_string(request.ref_text),
        "language": request.language,
        "instruct": _trim_string(request.instruct),
        "speed": request.speed,
        "duration": request.duration,
    }


def create_validation_data(request: VoiceCloneQueueRequest) -> dict[str, Any]:
    validate_queue_request(request)
    return {"validated": True, "signature": validation_data_signature(request)}


def stale_ready_validation_data(validation_data: dict[str, Any] | None) -> dict[str, Any]:
    stale = dict(validation_data or {})
    stale["validated"] = False
    return stale


def is_validation_data_current(validation_data: dict[str, Any] | None, request: VoiceCloneQueueRequest) -> bool:
    if not validation_data or not validation_data.get("validated"):
        return False
    return validation_data.get("signature") == validation_data_signature(request)


def validate_generation_ready(validation_data: dict[str, Any] | None, request: VoiceCloneQueueRequest) -> None:
    if not is_validation_data_current(validation_data, request):
        raise ValidationStateError("Queue must be validated again before generation.")


def _resolve_work_dir(output_dir: Path | None) -> Path:
    parent = Path(output_dir) if output_dir is not None else Path(gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix="voice_clone_queue_", dir=str(parent)))


def _write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    sf.write(path, audio, sample_rate)



def generate_voice_clone_queue(
    *,
    model: Any,
    request: VoiceCloneQueueRequest,
    generation_config: Any,
    progress: Any = None,
) -> VoiceCloneQueueResult:
    summary = validate_queue_request(request)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = _resolve_work_dir(request.output_dir)
    wav_paths: list[Path] = []
    audio_arrays: list[np.ndarray] = []
    zip_path = work_dir / f"voice_clone_queue_{timestamp}.zip"
    merged_wav_path = work_dir / f"voice_clone_queue_{timestamp}_merged.wav"

    try:
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=request.ref_audio,
            ref_text=_trim_string(request.ref_text) or None,
        )

        for index, item in enumerate(request.items, start=1):
            _report_progress(
                progress,
                index / len(request.items),
                f"Generating item {index}/{len(request.items)}",
            )
            generate_kwargs = {
                "text": item.text,
                "language": request.language,
                "voice_clone_prompt": voice_clone_prompt,
                "speed": request.speed,
                "generation_config": generation_config,
            }
            if _trim_string(request.instruct):
                generate_kwargs["instruct"] = _trim_string(request.instruct)
            if request.duration is not None:
                generate_kwargs["duration"] = request.duration

            generated = model.generate(**generate_kwargs)
            if not isinstance(generated, Sequence) or not generated:
                raise VoiceCloneQueueGenerationError("Model did not return audio output.")
            audio = generated[0]
            wav_path = work_dir / f"voice_clone_queue_{timestamp}_{index:03d}.wav"
            _write_wav(wav_path, audio, int(model.sampling_rate))
            wav_paths.append(wav_path)
            audio_arrays.append(audio)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for wav_path in wav_paths:
                arcname = os.path.basename(wav_path.name)
                if arcname != wav_path.name or any(sep in arcname for sep in ("/", "\\")):
                    raise VoiceCloneQueueGenerationError("Unsafe queue output filename.")
                archive.write(wav_path, arcname=arcname)

        pause_ms = request.pause_between_files_ms if request.apply_pause_between_files else 0
        merged_audio = concatenate_audio_with_silence(
            audio_arrays,
            int(model.sampling_rate),
            pause_ms / 1000.0,
        )
        _write_wav(merged_wav_path, merged_audio, int(model.sampling_rate))

        metadata = {
            "queue_items": summary.queue_items,
            "total_characters": summary.total_characters,
            "sampling_rate": int(model.sampling_rate),
            "zip_members": [path.name for path in wav_paths],
            "merged_wav_filename": merged_wav_path.name,
            "pause_between_files_ms": pause_ms,
            "apply_pause_between_files": bool(request.apply_pause_between_files),
        }
        # Keep the work directory on success because Gradio serves returned file paths
        # after the handler returns. Failure paths remove the whole directory below.
        return VoiceCloneQueueResult(
            wav_paths=[str(path) for path in wav_paths],
            zip_path=str(zip_path),
            metadata=metadata,
            merged_wav_path=str(merged_wav_path),
        )
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        if isinstance(exc, VoiceCloneQueueError):
            raise
        raise VoiceCloneQueueGenerationError(f"Queue generation failed: {exc}") from exc
