# Conversation Voice Clone Implementation Spec

## 1. Feature Summary

Add a new Web UI tab named **Conversation Voice Clone**. The tab lets users create a final **Conversation Audio** file from multiple ordered **Dialogue Lines**, where each line is generated using a selected **Voice Profile**.

## 2. Domain Terms

The following product-domain terms are defined in `CONTEXT.md`:

- **Voice Profile**: a reusable cloned-speaker configuration consisting of Speaker Name, Reference Voice, and Reference Text.
- **Speaker ID**: a stable system-generated identifier for a Voice Profile.
- **Speaker Name**: the user-facing label for a Voice Profile shown in the interface.
- **Dialogue Line**: one ordered line in a generated conversation, consisting of a selected Voice Profile and text to speak.
- **Conversation Audio**: the final generated audio file formed from ordered Dialogue Lines joined with a shared pause duration.

## 3. UI Placement

- Add a new tab named **Conversation Voice Clone**.
- Place it immediately after the existing **Voice Clone** tab.
- Phase 1 supports **Voice Clone only**; do not add Voice Design support in this tab.

## 4. Voice Profiles UI

Use **five fixed Voice Profile slots/collapsibles**.

Each slot contains:

- `Enable this speaker` checkbox
- `Speaker Name`
- `Reference Voice`
- `Reference Text`

Rules:

- Disabled slots are ignored completely.
- Enabled slots must provide all required fields:
  - Speaker Name
  - Reference Voice
  - Reference Text
- Speaker Name must be unique across enabled Voice Profiles.
- Reference Text is required.
- Reference Text may be entered manually or loaded from an uploaded `.txt` file.
- Do not auto-transcribe Reference Text in Phase 1.
- Show helper text for Reference Voice:

```text
Recommended reference audio: 3–10 seconds. Longer clips may slow inference or reduce cloning quality.
```

## 5. Dialogue Lines UI

Use a **DataFrame editor** with two columns:

```csv
speaker_name,text
```

Rules:

- Row order is the conversation order.
- `speaker_name` must match an enabled Voice Profile's Speaker Name.
- If a Voice Profile's Speaker Name is changed, existing Dialogue Lines are not automatically updated.
- If a Dialogue Line references an unknown speaker, validation fails with a clear message, for example:

```text
Unknown speaker_name: Alice. Available speakers: Alicia, Bob.
```

## 6. CSV Import and Export

Phase 1 supports CSV import/export for Dialogue Lines only.

CSV format:

```csv
speaker_name,text
Alice,"Xin chào Bob"
Bob,"Chào Alice"
```

Rules:

- Required columns: `speaker_name`, `text`.
- Row order is preserved as the conversation order.
- Do not support full project import/export in Phase 1.
- Do not include Reference Voice files in CSV import/export.

## 7. Conversation Controls

### Pause

Add one shared pause setting for the full conversation:

- Label: `Pause between lines (ms)`
- Default: `300`
- Min: `0`
- Max: `3000`

### Language

- Add one required language dropdown for the full conversation.
- Do not add per-line language in Phase 1.

### Speed

- Add one shared speed setting for the full conversation.
- Reuse the existing Voice Clone tab's speed range/default.
- If the existing Voice Clone tab has no speed control, fall back to `1.0`.
- Do not add per-line speed in Phase 1.

## 8. Validation

Add a dedicated **Validate** button. Users must click **Validate** before **Generate** for every generation run. If inputs change after validation, Generate must fail and ask the user to validate again. After a successful Generate, validation state is cleared so the next Generate requires a fresh Validate click.

### Voice Profile validation

- At least one Voice Profile must be enabled.
- Every enabled slot must have Speaker Name, Reference Voice, and Reference Text.
- Speaker Name must be unique across enabled Voice Profiles.

### Dialogue Line validation

- At least one Dialogue Line is required.
- Maximum Dialogue Lines: `50`.
- Each Dialogue Line must have `speaker_name`.
- Each Dialogue Line must have `text`.
- `speaker_name` must exist in the enabled Voice Profiles.
- Maximum characters per Dialogue Line: `3000`.
- Maximum total text characters across the conversation: `20000`.

### CSV validation

- CSV must contain `speaker_name` and `text` columns.
- CSV must not exceed `50` rows.
- Dialogue Line validation still applies after import.

### Validation summary

After successful validation, show a summary:

- Enabled Voice Profiles count
- Dialogue Lines count
- Total characters / `20000`
- Pause duration
- Output format: WAV

Generate should require a current successful validation state and update this summary only from that validated state.

## 9. Generate Behavior

### Execution

- Default generation concurrency: `1`.
- Generate Dialogue Lines safely and sequentially in Phase 1.
- Final merge order must always follow Dialogue Line row order.

### Voice Profile cache

Within one Generate run:

- Process each Voice Profile's Reference Voice + Reference Text once.
- Cache the resulting voice-clone prompt/profile data for that Generate run.
- Reuse the cached prompt for all Dialogue Lines using the same Voice Profile.

Do not persist this cache across multiple Generate runs in Phase 1.

### Error handling

- If any Dialogue Line generation fails, fail the entire Conversation Audio generation.
- Do not skip failed lines.
- Do not create a final merged audio file after a line failure.
- Do not expose partial clips to the user in Phase 1.

## 10. Progress UI

Show line-level progress during generation, for example:

```text
Generating line 7/20 — Alice
```

Minimum expected progress phases:

- Preparing Voice Profiles
- Generating line `i/n` — `speaker_name`
- Merging audio
- Done

Detailed logs are not required in Phase 1.

## 11. Audio Merge

### Output format

Final output format: **WAV**.

### Merge strategy

Use:

- numpy concatenate
- silence arrays based on the shared pause duration
- soundfile write WAV

Do not use ffmpeg or pydub in Phase 1.

### Compatibility

Before merging:

- Validate all generated clips are compatible.
- Expected format: mono audio at 24 kHz.
- If clips are incompatible, fail with a clear error.
- Do not auto-resample or mix down in Phase 1.

## 12. Output

After successful generation, show:

- Audio player for direct playback.
- Separate file/download component or link for downloading the final WAV.

### File handling

- Create the final WAV as a temporary/session file.
- Do not use a persistent output folder in Phase 1.
- Use a timestamped filename:

```text
conversation_voice_clone_<timestamp>.wav
```

Example:

```text
conversation_voice_clone_20260609_153012.wav
```

### Metadata

Show basic metadata:

- Dialogue Lines generated
- Total text characters
- Output filename

Do not include final duration, sample rate, or elapsed time in Phase 1.

## 13. Reset Behavior

Add a **Reset Conversation** button.

Reset should clear:

- Dialogue Lines
- validation result
- summary
- final audio output
- download output

Reset should keep:

- Voice Profiles
- Reference Voice uploads
- Reference Text
- Speaker Names

## 14. Intermediate Clips

- Do not show intermediate clips.
- Clean up intermediate data after merge.
- Do not keep partial outputs for user preview in Phase 1.

## 15. Explicit Non-Goals for Phase 1

Do not implement these in Phase 1:

- Preview individual Dialogue Lines.
- Preview individual Voice Profiles.
- Auto-transcribe Reference Text.
- Persistent Voice Profile storage.
- Full project import/export.
- MP3 output.
- Per-line pause.
- Per-line speed.
- Per-line language.
- Voice Design mode in conversation.
- Persistent cache across Generate runs.
- Confirmation modal for large jobs.
- `Use example conversation` button.
- ADR.

## 16. Implementation Work Breakdown

### Step 1 — Locate existing Gradio structure

Find the current:

- Voice Clone tab/component
- tab ordering
- model generation handler
- output audio handling
- existing speed/language controls

### Step 2 — Add Conversation Voice Clone tab

Add the tab immediately after Voice Clone.

Create UI sections:

1. Voice Profiles
2. Dialogue Lines
3. Conversation Settings
4. Validation/Summary
5. Generate/Progress
6. Output player/download

### Step 3 — Implement Voice Profile collection

Create an internal representation similar to:

```python
VoiceProfile = {
    "speaker_id": "...",
    "speaker_name": "...",
    "ref_audio": "...",
    "ref_text": "...",
}
```

For Phase 1, slot index can be used for stable Speaker IDs:

```text
speaker_1
speaker_2
speaker_3
speaker_4
speaker_5
```

### Step 4 — Implement Dialogue Lines handling

Use DataFrame columns:

```python
["speaker_name", "text"]
```

Normalize rows by:

- trimming `speaker_name`
- trimming and validating `text`
- preserving row order

### Step 5 — Implement validation function

Create one validation function reused by:

- Validate button
- Generate button

It should return:

- success/failure
- user-facing message
- normalized Voice Profiles
- normalized Dialogue Lines
- summary

### Step 6 — Implement CSV import/export

- Import CSV into the Dialogue Lines DataFrame.
- Export current Dialogue Lines DataFrame to a CSV temp file/download.
- Validation happens on Validate/Generate, not necessarily immediately on import.

### Step 7 — Implement generation function

Pseudo-flow:

```python
validate inputs
prepare Voice Profile prompt cache
for each Dialogue Line:
    get Voice Profile by speaker_name
    generate line audio using cached profile prompt
    collect audio array in order
validate audio compatibility
insert silence arrays between clips
concatenate
write timestamped WAV temp file
return audio player path, download path, and metadata
```

### Step 8 — Add progress updates

During the generation loop, show:

```text
Preparing Voice Profiles...
Generating line i/n — speaker_name
Merging audio...
Done
```

### Step 9 — Verify manually

Manual verification cases:

- One speaker, one line.
- Two speakers, alternating lines.
- Duplicate Speaker Name fails.
- Missing Reference Text fails.
- Unknown `speaker_name` in DataFrame fails.
- More than 50 lines fails.
- Line above 3000 characters fails.
- Total text above 20000 characters fails.
- Pause `0ms` works.
- Pause `3000ms` works.
- Output WAV plays.
- Download file works.
- Reset clears script/output but keeps Voice Profiles.

## 17. Acceptance Criteria

The feature is complete when:

- The **Conversation Voice Clone** tab exists immediately after **Voice Clone**.
- Users can define up to five enabled Voice Profiles.
- Users can edit Dialogue Lines via DataFrame.
- Users can import/export Dialogue Lines as CSV.
- The Validate button catches invalid inputs before model generation.
- Generate validates again before invoking the model.
- Generate creates all line audios with the correct selected Voice Profiles.
- Voice Profile prompt/cache is reused within a single Generate run.
- Final WAV is merged in correct Dialogue Line order.
- Shared pause duration is inserted between lines.
- Output audio player works.
- Separate download file/link works.
- Reset Conversation clears Dialogue Lines/output but keeps Voice Profiles.
- All agreed limits are enforced.
