# 0001-dubbing-timing-strategy

## Status

Accepted

## Context

The Dubbing Project feature synthesizes translated audio that must align with the original video's timing. Each SRT segment has a fixed duration in the source video. When a translated Dubbed Segment is synthesized, its natural spoken duration may differ from the original segment's duration — either longer (more syllables) or shorter (fewer syllables).

## Decision

**Use speed control to fit each Dubbed Segment within its original SRT timestamp window, with a speed floor of 1.0.**

- If the synthesized audio is longer than the original segment → increase playback speed (speed > 1.0) to compress it into the available window.
- If the synthesized audio is shorter → read at natural speed (speed = 1.0) and pad the remaining duration with silence.

Speed is clamped to a maximum of 3.0 to prevent unintelligible output. Segments requiring speed > 3.0 are logged as warnings but still generated.

## Considered Options

- **Keep natural speech timing, insert dynamic pauses** — More natural-sounding but causes timing drift relative to the source video. Rejected because maintaining video-audio sync is the primary goal of this feature.
- **Stretch both shorter and longer segments (speed < 1.0 allowed)** — Flexible but produces robotic-quality audio at slow speeds. Rejected: slowing speech below 1.0 sounds unnatural.
- **Pad silence only, no speed adjustment** — Simplest but causes cumulative timing drift across the entire output. Rejected.

## Consequences

- Dubbed Audio always has the same total duration as the source video, enabling direct remuxing.
- Segments with significant text expansion (e.g., Vietnamese translation of English) may exhibit accelerated speech.
- Future optimization: segment-level duration prediction before synthesis could reduce the need for speed adjustment.
