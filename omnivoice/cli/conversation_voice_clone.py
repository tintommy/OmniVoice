from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir, mkstemp
from typing import Any, Iterable, Sequence

import numpy as np

DEFAULT_CONVERSATION_PAUSE_MS = 300
DEFAULT_CONVERSATION_SPEED = 1.0
MAX_DIALOGUE_LINES = 50
MAX_LINE_CHARACTERS = 3000
MAX_TOTAL_CHARACTERS = 20000
OUTPUT_SAMPLE_RATE = 24000
CSV_COLUMNS = ("speaker_name", "text")


@dataclass(frozen=True)
class VoiceProfile:
    speaker_name: str
    ref_audio: Any
    ref_text: str
    enabled: bool = True
    speaker_id: str | None = None


@dataclass(frozen=True)
class DialogueLine:
    speaker_name: str
    text: str


@dataclass(frozen=True)
class ConversationRequest:
    voice_profiles: list[VoiceProfile]
    dialogue_lines: list[DialogueLine]
    language: str
    pause_ms: int = DEFAULT_CONVERSATION_PAUSE_MS
    speed: float = DEFAULT_CONVERSATION_SPEED
    output_dir: Path | None = None


@dataclass(frozen=True)
class ConversationResult:
    audio_path: str
    download_path: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidationSummary:
    enabled_voice_profiles: int
    dialogue_lines: int
    total_characters: int
    pause_ms: int
    output_format: str = "WAV"


class ConversationVoiceCloneError(ValueError):
    pass


class ConversationAudioMergeError(ConversationVoiceCloneError):
    pass


class ConversationGenerationError(ConversationVoiceCloneError):
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


def normalize_dialogue_rows(data: Any) -> list[dict[str, str]]:
    rows = _coerce_rows(data)
    normalized: list[dict[str, str]] = []

    for row in rows:
        if isinstance(row, dict):
            speaker_name = _trim_string(row.get("speaker_name"))
            text = _trim_string(row.get("text"))
            extra_values: Iterable[Any] = row.values()
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            speaker_name = _trim_string(row[0] if len(row) > 0 else "")
            text = _trim_string(row[1] if len(row) > 1 else "")
            extra_values = row
        else:
            continue

        if all(_is_empty_cell(value) for value in extra_values):
            continue

        normalized.append({"speaker_name": speaker_name, "text": text})

    return normalized


def dialogue_lines_from_rows(data: Any) -> list[DialogueLine]:
    return [DialogueLine(**row) for row in normalize_dialogue_rows(data)]


def validate_voice_profiles(voice_profiles: Sequence[VoiceProfile]) -> list[VoiceProfile]:
    enabled_profiles = [profile for profile in voice_profiles if profile.enabled]
    if not enabled_profiles:
        raise ConversationVoiceCloneError("At least one enabled Voice Profile is required.")

    seen_names: set[str] = set()
    for profile in enabled_profiles:
        speaker_name = _trim_string(profile.speaker_name)
        ref_text = _trim_string(profile.ref_text)
        if not speaker_name:
            raise ConversationVoiceCloneError(
                "Every enabled Voice Profile must provide Speaker Name, Reference Voice, and Reference Text."
            )
        if profile.ref_audio is None:
            raise ConversationVoiceCloneError(
                "Every enabled Voice Profile must provide Speaker Name, Reference Voice, and Reference Text."
            )
        if not ref_text:
            raise ConversationVoiceCloneError(
                "Every enabled Voice Profile must provide Speaker Name, Reference Voice, and Reference Text."
            )
        if speaker_name in seen_names:
            raise ConversationVoiceCloneError(f"Duplicate Speaker Name: {speaker_name}")
        seen_names.add(speaker_name)

    return [
        VoiceProfile(
            speaker_name=_trim_string(profile.speaker_name),
            ref_audio=profile.ref_audio,
            ref_text=_trim_string(profile.ref_text),
            enabled=True,
            speaker_id=profile.speaker_id,
        )
        for profile in enabled_profiles
    ]


def validate_dialogue_lines(
    dialogue_lines: Sequence[DialogueLine],
    voice_profiles: Sequence[VoiceProfile],
) -> list[DialogueLine]:
    if not dialogue_lines:
        raise ConversationVoiceCloneError("At least one Dialogue Line is required.")
    if len(dialogue_lines) > MAX_DIALOGUE_LINES:
        raise ConversationVoiceCloneError(
            f"Maximum Dialogue Lines exceeded: {len(dialogue_lines)}/{MAX_DIALOGUE_LINES}."
        )

    available_speakers = [profile.speaker_name for profile in voice_profiles if profile.enabled]
    available_set = set(available_speakers)

    normalized_lines: list[DialogueLine] = []
    total_characters = 0
    for line in dialogue_lines:
        speaker_name = _trim_string(line.speaker_name)
        text = _trim_string(line.text)

        if not speaker_name:
            raise ConversationVoiceCloneError("Each Dialogue Line must have speaker_name.")
        if not text:
            raise ConversationVoiceCloneError("Each Dialogue Line must have text.")
        if speaker_name not in available_set:
            available_text = ", ".join(available_speakers)
            raise ConversationVoiceCloneError(
                f"Unknown speaker_name: {speaker_name}. Available speakers: {available_text}."
            )
        if len(text) > MAX_LINE_CHARACTERS:
            raise ConversationVoiceCloneError(
                f"Dialogue Line exceeds {MAX_LINE_CHARACTERS} characters for speaker {speaker_name}."
            )

        total_characters += len(text)
        normalized_lines.append(DialogueLine(speaker_name=speaker_name, text=text))

    if total_characters > MAX_TOTAL_CHARACTERS:
        raise ConversationVoiceCloneError(
            f"Maximum total text characters exceeded: {total_characters}/{MAX_TOTAL_CHARACTERS}."
        )

    return normalized_lines


def validate_conversation_request(request: ConversationRequest) -> ValidationSummary:
    voice_profiles = validate_voice_profiles(request.voice_profiles)
    dialogue_lines = validate_dialogue_lines(request.dialogue_lines, voice_profiles)

    language = _trim_string(request.language)
    if not language or language == "Auto":
        raise ConversationVoiceCloneError("Language is required and cannot be empty or Auto.")

    if request.pause_ms < 0 or request.pause_ms > 3000:
        raise ConversationVoiceCloneError("Pause between lines (ms) must be between 0 and 3000.")

    total_characters = sum(len(line.text) for line in dialogue_lines)
    return ValidationSummary(
        enabled_voice_profiles=len(voice_profiles),
        dialogue_lines=len(dialogue_lines),
        total_characters=total_characters,
        pause_ms=request.pause_ms,
    )


def _csv_safe_cell(value: Any) -> str:
    text = _trim_string(value)
    if text.startswith(("=", "+", "-", "@")):
        text = "'" + text
    return text


def export_dialogue_lines_csv(dialogue_lines: Sequence[DialogueLine]) -> str:
    lines = [",".join(CSV_COLUMNS)]
    for line in dialogue_lines:
        output = []
        for value in (line.speaker_name, line.text):
            text = _csv_safe_cell(value)
            if any(char in text for char in [",", '"', "\n", "\r"]):
                text = '"' + text.replace('"', '""') + '"'
            output.append(text)
        lines.append(",".join(output))
    return "\n".join(lines) + "\n"


def import_dialogue_lines_csv(csv_text: str) -> list[DialogueLine]:
    csv_text = csv_text.lstrip("\ufeff")
    reader = csv.DictReader(csv_text.splitlines())
    fieldnames = [name.lstrip("\ufeff") for name in (reader.fieldnames or [])]
    if fieldnames != list(CSV_COLUMNS):
        raise ConversationVoiceCloneError("CSV must contain exactly the columns: speaker_name,text")
    reader.fieldnames = fieldnames

    rows: list[DialogueLine] = []
    for row in reader:
        rows.append(
            DialogueLine(
                speaker_name=_trim_string(row.get("speaker_name")),
                text=_trim_string(row.get("text")),
            )
        )

    if len(rows) > MAX_DIALOGUE_LINES:
        raise ConversationVoiceCloneError(
            f"CSV must not exceed {MAX_DIALOGUE_LINES} rows."
        )

    return rows


def _ensure_mono_audio(audio: np.ndarray, index: int) -> np.ndarray:
    if not isinstance(audio, np.ndarray):
        raise ConversationAudioMergeError(f"Generated clip {index} is not a numpy.ndarray.")
    if audio.ndim != 1:
        raise ConversationAudioMergeError(
            f"Generated clip {index} must be mono audio at {OUTPUT_SAMPLE_RATE} Hz."
        )
    return audio.astype(np.float32, copy=False)


def merge_conversation_audio(
    clips: Sequence[np.ndarray],
    pause_ms: int,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> str:
    if not clips:
        raise ConversationAudioMergeError("At least one generated clip is required for merge.")
    if sample_rate != OUTPUT_SAMPLE_RATE:
        raise ConversationAudioMergeError(
            f"Expected mono audio at {OUTPUT_SAMPLE_RATE} Hz, got {sample_rate} Hz."
        )
    if pause_ms < 0 or pause_ms > 3000:
        raise ConversationAudioMergeError("Pause between lines (ms) must be between 0 and 3000.")

    normalized_clips = [_ensure_mono_audio(audio, index) for index, audio in enumerate(clips, start=1)]

    pause_samples = int(sample_rate * (pause_ms / 1000.0))
    silence = np.zeros(pause_samples, dtype=np.float32)
    merged_parts: list[np.ndarray] = []
    for index, clip in enumerate(normalized_clips):
        merged_parts.append(clip)
        if index < len(normalized_clips) - 1 and pause_samples > 0:
            merged_parts.append(silence)

    import soundfile as sf

    merged_audio = np.concatenate(merged_parts)
    when = timestamp or datetime.now()
    base_dir = Path(output_dir) if output_dir is not None else Path(gettempdir())
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"conversation_voice_clone_{when.strftime('%Y%m%d_%H%M%S')}_"
    fd, output_name = mkstemp(prefix=prefix, suffix=".wav", dir=str(base_dir))
    output_path = Path(output_name)
    import os

    os.close(fd)
    sf.write(output_path, merged_audio, sample_rate)
    return str(output_path)


def generate_conversation_audio(
    model: Any,
    request: ConversationRequest,
    progress: Any = None,
) -> ConversationResult:
    summary = validate_conversation_request(request)
    voice_profiles = validate_voice_profiles(request.voice_profiles)
    dialogue_lines = validate_dialogue_lines(request.dialogue_lines, voice_profiles)

    sampling_rate = int(getattr(model, "sampling_rate", OUTPUT_SAMPLE_RATE))
    if sampling_rate != OUTPUT_SAMPLE_RATE:
        raise ConversationGenerationError(
            f"Expected model sampling_rate to be {OUTPUT_SAMPLE_RATE}, got {sampling_rate}."
        )

    prompt_cache: dict[str, Any] = {}

    _report_progress(progress, 0.0, "Preparing Voice Profiles")
    for index, profile in enumerate(voice_profiles, start=1):
        _report_progress(
            progress,
            index / max(len(voice_profiles), 1) * 0.2,
            f"Preparing Voice Profiles ({index}/{len(voice_profiles)})",
        )
        prompt_cache[profile.speaker_name] = model.create_voice_clone_prompt(
            ref_audio=profile.ref_audio,
            ref_text=profile.ref_text,
        )

    clips: list[np.ndarray] = []
    total_lines = len(dialogue_lines)
    for index, line in enumerate(dialogue_lines, start=1):
        _report_progress(
            progress,
            0.2 + (index - 1) / max(total_lines, 1) * 0.7,
            f"Generating line {index}/{total_lines} — {line.speaker_name}",
        )
        prompt = prompt_cache[line.speaker_name]
        generated = model.generate(
            text=line.text,
            language=request.language,
            voice_clone_prompt=prompt,
            speed=request.speed,
        )
        if not generated:
            raise ConversationGenerationError(
                f"Model returned no audio for speaker {line.speaker_name}."
            )
        clips.append(_ensure_mono_audio(generated[0], len(clips) + 1))

    _report_progress(progress, 0.95, "Merging audio")
    output_path = merge_conversation_audio(
        clips=clips,
        pause_ms=request.pause_ms,
        sample_rate=sampling_rate,
        output_dir=request.output_dir,
    )
    _report_progress(progress, 1.0, "Done")

    return ConversationResult(
        audio_path=output_path,
        download_path=output_path,
        metadata={
            "dialogue_lines_generated": summary.dialogue_lines,
            "total_text_characters": summary.total_characters,
            "output_filename": Path(output_path).name,
        },
    )


def _profiles_from_ui_dicts(voice_profiles: Sequence[dict[str, Any]]) -> list[VoiceProfile]:
    profiles: list[VoiceProfile] = []
    for index, profile in enumerate(voice_profiles, start=1):
        slot_index = profile.get("slot_index", index)
        profiles.append(
            VoiceProfile(
                speaker_id=f"speaker_{slot_index}",
                speaker_name=_trim_string(profile.get("speaker_name")),
                ref_audio=profile.get("ref_audio"),
                ref_text=_trim_string(profile.get("ref_text")),
                enabled=bool(profile.get("enabled")),
            )
        )
    return profiles


def _request_from_ui_values(
    *,
    voice_profiles: Sequence[dict[str, Any]],
    dialogue_lines: Any,
    language: str,
    speed: float,
    pause_ms: int,
    output_dir: str | Path | None = None,
) -> ConversationRequest:
    return ConversationRequest(
        voice_profiles=_profiles_from_ui_dicts(voice_profiles),
        dialogue_lines=dialogue_lines_from_rows(dialogue_lines),
        language=_trim_string(language),
        pause_ms=int(pause_ms),
        speed=float(speed),
        output_dir=Path(output_dir) if output_dir is not None else None,
    )


def validate_conversation_voice_clone(
    *,
    voice_profiles: Sequence[dict[str, Any]],
    dialogue_lines: Any,
    language: str,
    speed: float,
    pause_ms: int,
) -> dict[str, Any]:
    request = _request_from_ui_values(
        voice_profiles=voice_profiles,
        dialogue_lines=dialogue_lines,
        language=language,
        speed=speed,
        pause_ms=pause_ms,
    )
    summary = validate_conversation_request(request)
    normalized_profiles = validate_voice_profiles(request.voice_profiles)
    normalized_lines = validate_dialogue_lines(request.dialogue_lines, normalized_profiles)
    return {
        "voice_profiles": [
            {
                "speaker_id": profile.speaker_id,
                "speaker_name": profile.speaker_name,
                "ref_audio": profile.ref_audio,
                "ref_text": profile.ref_text,
                "enabled": profile.enabled,
            }
            for profile in normalized_profiles
        ],
        "dialogue_lines": [
            {"speaker_name": line.speaker_name, "text": line.text}
            for line in normalized_lines
        ],
        "language": _trim_string(request.language),
        "speed": request.speed,
        "pause_ms": request.pause_ms,
        "enabled_voice_profiles_count": summary.enabled_voice_profiles,
        "dialogue_lines_count": summary.dialogue_lines,
        "total_characters": summary.total_characters,
        "output_format": summary.output_format,
    }


def generate_conversation_voice_clone(
    *,
    model: Any,
    validation_result: dict[str, Any],
    progress: Any = None,
) -> dict[str, Any]:
    request = ConversationRequest(
        voice_profiles=[
            VoiceProfile(
                speaker_id=_trim_string(profile.get("speaker_id")) or None,
                speaker_name=_trim_string(profile.get("speaker_name")),
                ref_audio=profile.get("ref_audio"),
                ref_text=_trim_string(profile.get("ref_text")),
                enabled=bool(profile.get("enabled", True)),
            )
            for profile in validation_result["voice_profiles"]
        ],
        dialogue_lines=[
            DialogueLine(
                speaker_name=_trim_string(line.get("speaker_name")),
                text=_trim_string(line.get("text")),
            )
            for line in validation_result["dialogue_lines"]
        ],
        language=_trim_string(validation_result["language"]),
        pause_ms=int(validation_result["pause_ms"]),
        speed=float(validation_result["speed"]),
    )
    result = generate_conversation_audio(model=model, request=request, progress=progress)
    return {
        "audio_path": result.audio_path,
        "download_path": result.download_path,
        "dialogue_lines_generated": result.metadata["dialogue_lines_generated"],
        "total_characters": result.metadata["total_text_characters"],
        "output_filename": result.metadata["output_filename"],
        "status": "Done.",
    }
