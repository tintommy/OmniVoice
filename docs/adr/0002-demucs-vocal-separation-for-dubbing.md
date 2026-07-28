# 0002 - Demucs Vocal Separation for Dubbing

## Status

Accepted

## Context

The Dubbing Project feature clones a speaker's voice from Source Segments extracted from the source MP3. When the source audio contains background music, sound effects, or environmental noise, the voice clone quality degrades because the model tries to replicate non-speech characteristics alongside the speaker's voice.

The existing codebase already has two denoising mechanisms:

1. **`<|denoise|>` token**: A learned token prepended to the style text during inference, telling the model to suppress noise during diffusion generation. This is applied *during* synthesis, not *before* voice cloning.

2. **Sidon speech enhancement** (`omnivoice/scripts/denoise_audio.py`): A batch preprocessing script for training data that removes background noise from raw audio. It is designed for speech enhancement (noise suppression), not music/sound-effect separation, and is not integrated into the dubbing pipeline.

Neither mechanism addresses the core problem for dubbing use cases involving video content with background music or soundtracks: the reference audio used for voice cloning contains non-speech content that pollutes the cloned voice.

## Decision

**Use Demucs (`htdemucs_ft` model) as a pre-processing step to extract the `vocals` stem from the source MP3 before the dubbing pipeline begins.**

Key details of the decision:

- **Model**: `htdemucs_ft` (Hybrid Demucs fine-tuned variant) — best quality-to-VRAM ratio for a T4 GPU (~3 GB VRAM).
- **Placement**: Run once on the entire source MP3 at the start of the dubbing pipeline, producing a Vocals-Separated Audio file. All subsequent Source Segment extraction operates on this clean vocals-only audio.
- **`denoise=False`**: When Demucs is active, the model's `denoise` flag is set to `False` to prevent double-processing. The `<|denoise|>` token is designed for noisy audio; applying it to already-pure vocals may degrade quality.
- **Non-Speech Segments**: Even segments marked as non-speech (e.g., `[music]`, `(applause)`) use the vocals stem. If such a segment contains no vocals, it results in silence — preferable to leaking music into the voice clone reference.
- **API**: `demucs.api.Separator` Python API, not CLI subprocess, to avoid unnecessary disk I/O.

### Pipeline Change

```
Before:
  Source MP3 → SRT slicing → extract segment → clone voice → synthesize

After:
  Source MP3 → Demucs (vocals only) → SRT slicing → extract segment → clone voice → synthesize
```

## Considered Options

- **Sidon speech enhancement**: Removes background noise but cannot separate music or sound effects from speech. Inadequate for video content with background music. Rejected because the use case requires source separation, not noise suppression.

- **Demucs per segment (post-slicing)**: Running Demucs on each individual Source Segment rather than the full MP3. Rejected because:
  - Demucs quality degrades on short audio clips (<10s)
  - N× computational cost (N = number of segments) vs 1× for the full file
  - More complex pipeline with per-segment error handling

- **No pre-processing (status quo)**: Relying solely on the `<|denoise|>` token. Rejected because the token operates on synthesized output, not the reference audio for cloning. The speaker's voice would still be cloned from a noisy reference.

- **Demucs CLI via subprocess**: Simpler to implement but adds file I/O overhead and makes error handling more brittle. Rejected in favor of the Python API.

## Consequences

- Dubbing quality improves significantly for source material with background music or soundtracks (films, videos, podcasts with intro/outro music).
- Adds ~30-60s of preprocessing time per source MP3 on T4 GPU (Demucs full-audio separation).
- Adds `demucs` as a new Python dependency with PyTorch overlap (already present).
- Vocals-Separated Audio is ephemeral (stored in temp directory, cleaned up after pipeline completes).
- If source audio has no background music/noise, Demucs adds compute overhead with minimal benefit. Future optimization: skip Demucs if silence detection shows no non-speech content.
