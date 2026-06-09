# Voice Clone Queue Generation Implementation Spec

## 1. Feature Summary

Add a **Queue Generation** section inside the existing **Voice Clone** Web UI tab. The section lets users create a **Voice Clone Queue**: multiple independent text items generated sequentially with the same selected Voice Profile and the same Voice Clone settings.

The existing single-item Voice Clone flow must remain unchanged.

## 2. Domain Terms

The following terms are defined in `CONTEXT.md`:

- **Voice Profile**: a reusable cloned-speaker configuration consisting of a speaker-facing name, a reference voice, and reference text.
- **Queued Clone Item**: one text segment queued for voice-clone generation using the same selected Voice Profile.
- **Voice Clone Queue**: an ordered set of Queued Clone Items generated as separate audio files using the same selected Voice Profile.

Do not use **Dialogue Line** for Queue Generation items. Dialogue Lines belong to conversation generation.

## 3. UI Placement

- Add Queue Generation inside the existing **Voice Clone** tab.
- Use a section or accordion named **Queue Generation**.
- Keep the existing single Voice Clone UI and behavior unchanged.
- Recommended placement: below the current single-generation Voice Clone controls/output in the same tab.

## 4. Shared Voice Clone Inputs

Queue Generation uses the existing Voice Clone inputs/settings:

- Reference Audio
- Reference Text
- Reference Text File `.txt`
- Language
- Instruct
- Generation Settings:
  - Speed
  - Duration
  - Inference Steps
  - Denoise
  - Guidance Scale
  - Preprocess Prompt
  - Postprocess Output

Do not duplicate Reference Audio/Text/settings inside Queue Generation.

## 5. Queue Input UI

Use a DataFrame editor with one column:

```csv
text
```

Rules:

- Each non-empty row is one **Queued Clone Item**.
- Row order is generation order.
- Empty rows are ignored.
- Users can add/delete rows through the DataFrame.

## 6. Queue CSV Import/Export

Queue Generation supports CSV import/export.

CSV format:

```csv
text
Đoạn text 1
Đoạn text 2
```

Required behavior:

- Import queue CSV with one required column: `text`.
- Export current queue to CSV.
- Export sample CSV with only the header row:

```csv
text
```

Encoding:

- Read CSV with UTF-8 BOM tolerance.
- Vietnamese text with accents must import correctly.
- Prefer `utf-8-sig` when reading/writing CSV files used by the Web UI.

## 7. Validation Flow

Add a dedicated **Validate Queue** button.

Generate Queue must require a current successful validation state:

- If the user clicks Generate Queue before Validate Queue, show an error.
- If queue inputs or shared Voice Clone settings change after validation, Generate Queue must fail and ask the user to validate again.
- After a successful Generate Queue run, clear validation state so the next Generate Queue requires a fresh Validate Queue click.

This mirrors the current Conversation Voice Clone validation behavior.

## 8. Validation Rules

Voice Clone shared inputs:

- Reference Audio is required.
- Reference Text follows the existing Voice Clone behavior:
  - It may be entered manually.
  - It may be loaded from a `.txt` file.
  - If existing Voice Clone allows empty Reference Text for auto-transcription, Queue Generation may reuse that behavior unless implementation decides to require Reference Text for performance predictability.

Queue items:

- At least one Queued Clone Item is required.
- Maximum queue items: `20`.
- Maximum characters per item: `3000`.
- Maximum total characters across queue: `20000`.

Validation summary should include:

- Queue item count
- Total characters / `20000`
- Output format: WAV
- Download all format: ZIP

## 9. Generate Behavior

Execution:

- Generate Queue processes items sequentially.
- Default concurrency: `1`.
- Generation order must follow DataFrame row order.
- Show progress per item, for example:

```text
Generating item 7/20
```

Voice prompt reuse:

- Create or reuse the same voice-clone prompt for all Queued Clone Items in one Generate Queue run.
- Do not recreate the voice prompt for every queue item.

Error handling:

- If any item fails, fail the entire queue.
- Do not skip failed items.
- Do not return partial outputs to the user in Phase 1.

## 10. Output Files

Each Queued Clone Item produces one WAV file.

Output filenames use timestamp + item order:

```text
voice_clone_queue_20260609_153012_001.wav
voice_clone_queue_20260609_153012_002.wav
voice_clone_queue_20260609_153012_003.wav
```

Download-all ZIP filename:

```text
voice_clone_queue_20260609_153012.zip
```

Use temp/session files. Do not create persistent project output folders in Phase 1.

## 11. Output UI

After successful Generate Queue:

- Show a `gr.File(file_count="multiple")` output listing all generated WAV files.
- Allow users to download individual WAV files from that list.
- Show a separate ZIP download output for downloading all files.

Do not add audio preview players for each queue item in Phase 1.

## 12. Explicit Non-Goals for Phase 1

Do not implement these in Phase 1:

- Separate Reference Audio/Text/settings for Queue Generation.
- MP3 output.
- Parallel queue generation.
- Partial output return after failure.
- Skip failed items and continue.
- Audio preview per queue item.
- Dynamic textbox components.
- Persistent queue storage.
- ADR.

## 13. Implementation Work Breakdown

### Step 1 — Add testable queue core helpers

Recommended module:

```text
omnivoice/cli/voice_clone_queue.py
```

Suggested responsibilities:

- Normalize queue DataFrame rows.
- Import/export queue CSV.
- Validate queue request.
- Generate queued WAV files sequentially.
- Create ZIP download file.
- Return output file paths and metadata.

### Step 2 — Add UI section in Voice Clone tab

Inside the existing Voice Clone tab:

- Add **Queue Generation** accordion/section.
- Add queue DataFrame with one `text` column.
- Add import/export/sample CSV controls.
- Add Validate Queue and Generate Queue buttons.
- Add status/summary outputs.
- Add multiple WAV file output.
- Add ZIP download output.

### Step 3 — Reuse current Voice Clone settings

Wire Queue Generation to existing Voice Clone components:

- Reference Audio
- Reference Text
- Language
- Instruct
- Generation settings

Do not duplicate those controls.

### Step 4 — Implement validation-state freshness

Store validated queue state in `gr.State`.

Generate Queue must:

1. Require non-empty validation state.
2. Compare current input fingerprint against validated fingerprint.
3. Fail if inputs changed.
4. Clear validation state after successful generation.

### Step 5 — Implement generation

Pseudo-flow:

```python
validate queue request
create voice_clone_prompt once
for each queued item in order:
    generate audio using shared voice_clone_prompt and shared settings
    write WAV file with timestamp + index
if all succeeded:
    create ZIP containing all WAV files
return list of WAV paths + ZIP path + metadata
```

### Step 6 — Verify

Manual verification cases:

- Single queue item generates one WAV.
- Multiple queue items generate ordered WAV files.
- ZIP contains all WAV files.
- Validate Queue is required before Generate Queue.
- Changing queue after validation forces revalidation.
- Generate Queue clears validation after success.
- Vietnamese CSV imports correctly.
- Sample CSV contains only `text` header.
- More than 20 items fails.
- Item over 3000 characters fails.
- Total over 20000 characters fails.
- Missing Reference Audio fails.

## 14. Acceptance Criteria

The feature is complete when:

- Queue Generation section exists inside Voice Clone tab.
- Existing single Voice Clone behavior is unchanged.
- Queue input uses a DataFrame with one `text` column.
- Queue CSV import/export/sample export works.
- Vietnamese CSV imports correctly.
- Validate Queue is required before Generate Queue.
- Generate Queue refuses stale validation after input changes.
- Generate Queue clears validation after success.
- Queue generates one WAV per item in order.
- All queue items reuse the same voice-clone prompt within one run.
- Any item failure fails the whole queue.
- Generated WAV files are shown as downloadable multiple files.
- ZIP download-all is available.
- Limits are enforced: 20 items, 3000 chars/item, 20000 chars total.
