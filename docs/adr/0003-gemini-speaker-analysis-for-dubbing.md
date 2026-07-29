# 0003 - Gemini Speaker Analysis for Dubbing

## Status

Proposed

## Context

The Dubbing Project feature currently clones voices per-segment from the source audio — each SRT line independently extracts audio from its timestamp and clones the voice within that slice. There is no concept of speaker identity across segments. Users must manually manage which voice samples to use, with the only option being `single_speaker=True` for all segments.

When a video has multiple distinct speakers, per-segment cloning produces inconsistent voice quality because:
1. Each segment clones from a potentially noisy or short audio slice
2. The same speaker across different segments may be cloned with different voice characteristics
3. Short segments (< 1.5s) fall back to neighboring segments, which may belong to a different speaker

A pre-analysis step that identifies distinct voice types (by gender and age) and allows the user to provide one clean voice sample per type would significantly improve output consistency and quality.

## Decision

**Use Gemini API (`gemini-3.1-flash-lite`) to analyze the original SRT and classify each segment by Voice Type (gender + age).**

### Key details:

- **Model**: `gemini-3.1-flash-lite` — lowest cost, adequate reasoning for dialogue context analysis
- **Input**: Raw SRT text grouped into Analysis Windows of approximately 60 seconds, bounded by natural SRT line boundaries (no mid-line cuts)
- **Output**: Structured JSON via Gemini's `response_schema` — per-segment `{index, gender, age}` assignments plus a summary of voice types found
- **Voice Type vocabulary**: 8 combinations of gender (`male`/`female`) × age (`child`/`young`/`middle-aged`/`old`)
- **Rate limits**: 15 RPM, 500 RPD — sufficient for ~8 hours of video per day (500 one-minute windows)
- **Caching**: Results cached to `~/.omnivoice/cache/<sha256(srt_content)>.json` to avoid redundant API calls
- **Retry policy**: 3 attempts with exponential backoff; on persistent failure, fall back to per-segment cloning for that window
- **No cross-window identity tracking**: Each Analysis Window is analyzed independently. Voice types may vary between windows for the same speaker — this is acceptable because the goal is within-scene differentiation, not character tracking

### Pipeline change:

```
Before (current):
  Source MP3 → Demucs (optional) → per-segment voice cloning → synthesize → merge

After (with speaker analysis):
  Original SRT → Gemini Speaker Analysis → Voice Types identified
  User provides Voice Samples (audio + transcript) per Voice Type
  Source MP3 → Demucs (optional) → per-segment synthesis with Voice Sample → merge
  (Fallback: per-segment voice cloning for windows where Gemini failed)
```

### Code organization:

New module `omnivoice/speaker/` containing:
- `analyzer.py` — Gemini API calls, retry logic, response parsing
- `cache.py` — JSON file cache with SHA256-based keys
- `schema.py` — Gemini response schema definition

`DubbingRequest` gains an optional `voice_map: dict[str, tuple[str, str]]` field mapping Voice Type labels to `(audio_path, transcript_text)` tuples. When empty, the pipeline falls back to the existing per-segment cloning behavior.

### Gradio UI flow:

Within the existing Dubbing tab, three sequential sections:
1. **Analyze**: Upload original SRT → click "Analyze Speakers" → see results
2. **Voice Samples**: For each Voice Type found, upload audio file + enter transcript text
3. **Generate**: Upload translated SRT + select language → generate dubbed audio

## Considered Options

- **Rule-based heuristic** (regex for speaker labels like `[John]:`, keyword matching): Only works when SRT contains explicit speaker annotations. Most real-world SRT files do not. Rejected.
- **Audio-based speaker diarization** (e.g., pyannote.audio): Requires GPU, adds significant processing time, and the source audio quality in dubbing use cases (background music, overlapping dialogue) makes diarization unreliable. Rejected.
- **Gemini 2.5 Pro**: Better reasoning but higher cost and latency. Overkill for gender/age classification. Rejected.
- **Merge voice types across windows** (deduplicate same speaker appearing in different windows): Adds complexity and another Gemini call. Not necessary since voice type assignment is scene-local and the same speaker may reasonably be classified slightly differently in different contexts. Rejected.

## Consequences

- Voice quality consistency improves significantly when user provides clean voice samples
- Gemini API dependency — feature requires internet access and a valid API key
- Rate limit of 500 RPD caps daily throughput at ~8 hours of video content
- Caching ensures repeated dubbing runs on the same original SRT do not consume additional API quota
- Gemini costs are negligible for typical use (500 requests/day of short SRT analysis ~ $0.50/day)
- Per-segment cloning remains as a fallback, preserving backward compatibility
