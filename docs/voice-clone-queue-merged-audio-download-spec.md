# Voice Clone Queue Merged Audio Download Spec

## 1. Feature Summary

Add a merged-audio download output to the **Queue Generation** section in the **Voice Clone** tab.

After a successful queue generation run, the app must create one WAV file by concatenating the audio files generated in that same run, in queue order. The merged file is offered as an additional download alongside the existing individual WAV outputs and ZIP download.

## 2. Scope

### In scope

- Add merged WAV generation for the latest successful Queue Generation run.
- Preserve queue order when concatenating generated audio.
- Insert optional silence between files.
- Default pause duration: `300ms`.
- Add UI controls for pause behavior.
- Add a dedicated merged WAV download output.
- Keep the existing ZIP download behavior unchanged.
- Add tests for merged audio behavior and pause validation.

### Out of scope

- MP3 export.
- User-defined output filename.
- Partial outputs when queue generation fails.
- Audio preview for the merged file.
- Cross-fade/fade-in/fade-out between files.
- Merging files from older queue runs.

## 3. UI Changes

In `omnivoice/cli/demo.py`, inside **Voice Clone** → **Queue Generation**, add:

1. Checkbox:
   - Label: `Apply pause between merged files`
   - Default: checked / `True`

2. Number input:
   - Label: `Pause duration (ms)`
   - Default: `300`
   - Minimum: `0`
   - Maximum: `5000`
   - Step: `50`

3. File output:
   - Label: `Download Merged Queue Audio (WAV)`

Recommended placement:

- Put checkbox and pause duration near `Generate Queue`.
- Put the merged WAV download output near the existing ZIP output.
- Keep existing outputs:
  - `Generated Queue WAV Files`
  - `Download All Queue WAVs (ZIP)`

## 4. Behavior

### Successful queue generation

When `Generate Queue` succeeds:

1. Generate each queue item as an individual WAV file, as currently implemented.
2. Create the ZIP file, as currently implemented.
3. Create a merged WAV file from only the WAV files generated in this same run.
4. Return the merged WAV path to the new Gradio file output.
5. Update metadata text to include merged audio information.

### Latest-run semantics

The merged WAV must include only files from the most recent successful queue generation run.

Implementation recommendation:

- Create the merged file inside the same temp work directory used by the current queue run.
- Build it from the in-memory/generated audio arrays or from `wav_paths` collected during that same function call.
- Do not read or reuse files from previous UI outputs.

### Queue order

The merged WAV must preserve queue order exactly:

```text
queue item 1 audio
+ optional silence
+ queue item 2 audio
+ optional silence
+ queue item 3 audio
...
```

### Single-item queue

If the queue has one item, still create a merged WAV file.

The merged file content should match the single generated WAV, but it should be written as a distinct merged output file.

### Failure behavior

Keep current failure behavior:

- If any item fails during generation, clean up the whole output directory.
- Do not return partial WAV files.
- Do not return a partial ZIP.
- Do not return a partial merged WAV.

## 5. Pause Behavior

### Default behavior

- `Apply pause between merged files`: `True`
- `Pause duration (ms)`: `300`
- Effective pause: `300ms`

### Checkbox off

If `Apply pause between merged files` is unchecked:

- Effective pause is `0ms`.
- `Pause duration (ms)` is ignored.
- Do not validate the pause duration value.

### Checkbox on

If `Apply pause between merged files` is checked:

- Validate `Pause duration (ms)` strictly.
- Allowed range: integer `0–5000`.
- Invalid values should raise a Gradio/user-facing error.

Suggested message:

```text
Pause duration must be between 0 and 5000 milliseconds.
```

## 6. Audio Format

Merged output format: **WAV only**.

Rationale:

- Existing Queue Generation outputs WAV files.
- `voice_clone_queue.py` already writes WAV through `soundfile`.
- WAV avoids adding FFmpeg/MP3 codec complexity.
- WAV is deterministic and easier to test.

## 7. Concatenation Semantics

Use silence-only concatenation.

Do not use cross-fade for this feature.

Rationale:

- The requested behavior is a pause/rest between files, not a transition effect.
- Silence-only concatenation does not alter the generated audio except for inserted gaps.
- It is easy to verify in tests:

```text
merged_length = sum(individual_lengths) + gap_samples * (item_count - 1)
```

## 8. File Naming

Merged file name:

```text
voice_clone_queue_YYYYMMDD_HHMMSS_merged.wav
```

This matches the existing queue output naming convention:

```text
voice_clone_queue_YYYYMMDD_HHMMSS_001.wav
voice_clone_queue_YYYYMMDD_HHMMSS_002.wav
voice_clone_queue_YYYYMMDD_HHMMSS.zip
```

## 9. Backend Design

### `omnivoice/utils/audio.py`

Add a generic helper for silence-only concatenation, for example:

```python
def concatenate_with_silence(
    chunks: list[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> np.ndarray:
    ...
```

Requirements:

- Input chunks are numpy audio arrays.
- Preserve channel count.
- Preserve queue order.
- If there is one chunk, return that chunk or a safe copy.
- Insert zero-valued silence between chunks.
- Do not apply fade/cross-fade.

### `omnivoice/cli/voice_clone_queue.py`

Extend queue generation orchestration:

1. Add pause settings to `VoiceCloneQueueRequest`:

```python
apply_pause_between_files: bool = True
pause_between_files_ms: int = 300
```

2. Do not include these fields in `validation_data_signature`.

Reason: changing merge pause settings should not require re-validating the queue, because it does not affect model generation.

3. Extend `VoiceCloneQueueResult`:

```python
@dataclass(frozen=True)
class VoiceCloneQueueResult:
    wav_paths: list[str]
    zip_path: str
    merged_wav_path: str
    metadata: dict[str, Any]
```

4. Generate `merged_wav_path` during the same successful run.
5. Write merged WAV with the model sampling rate.
6. Include merged metadata.

## 10. Metadata

Update Queue Metadata text to include merged output details.

Example:

```text
Queue items generated: 3
Total text characters: 1234
Output ZIP: voice_clone_queue_20260609_123456.zip
Merged audio: voice_clone_queue_20260609_123456_merged.wav
Pause between files: 300ms
```

If pause checkbox is off:

```text
Pause between files: No pause
```

## 11. Validation Rules

When pause is enabled:

- `pause_between_files_ms` must be an integer.
- Minimum: `0`.
- Maximum: `5000`.

When pause is disabled:

- Skip duration validation.
- Use effective pause `0ms`.

## 12. Test Plan

Update `tests/test_voice_clone_queue.py`.

Recommended tests:

1. Successful queue generation creates merged WAV.
2. Merged WAV sample rate matches model sampling rate.
3. Merged WAV length equals:

```text
sum(individual_audio_lengths) + gap_samples * (item_count - 1)
```

4. Single-item queue still creates a merged WAV.
5. Pause disabled creates merged audio with no inserted silence.
6. Invalid pause duration raises an error when pause is enabled.
7. Invalid pause duration is ignored when pause is disabled.
8. Failure during generation cleans up merged output along with WAV/ZIP outputs.

## 13. Implementation Files

Expected files to modify:

- `omnivoice/utils/audio.py`
- `omnivoice/cli/voice_clone_queue.py`
- `omnivoice/cli/demo.py`
- `tests/test_voice_clone_queue.py`

## 14. Final Decisions

- Generate merged audio during `Generate Queue`, not via a separate merge button.
- Use WAV only.
- Use silence-only gaps, no cross-fade.
- Add checkbox to enable/disable pause.
- Checkbox defaults to enabled.
- Pause duration defaults to `300ms`.
- Pause duration input range: `0–5000ms`, step `50ms`.
- If pause disabled, ignore duration and use `0ms`.
- Always create merged WAV after a successful run, even for one queue item.
- Keep current fail-cleanup behavior.
- Store merged output path as a first-class field on `VoiceCloneQueueResult`.
- Do not require queue re-validation when only pause settings change.
