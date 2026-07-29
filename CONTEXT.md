# OmniVoice

OmniVoice is a multilingual text-to-speech and voice cloning system. This context defines product-domain language used when discussing voice generation workflows.

## Language

**Voice Profile**:
A reusable cloned-speaker configuration consisting of a speaker-facing name, a reference voice, and reference text.
_Avoid_: Speaker ID when referring to the whole cloned voice configuration

**Speaker ID**:
A stable system-generated identifier for a Voice Profile.
_Avoid_: User-facing speaker name

**Speaker Name**:
The user-facing label for a Voice Profile shown in the interface.
_Avoid_: Speaker ID

**Dialogue Line**:
One ordered line in a generated conversation, consisting of a selected Voice Profile and text to speak.
_Avoid_: Row, sentence when referring to the conversation unit

**Conversation Audio**:
A final generated audio file formed from ordered Dialogue Lines joined with a shared pause duration.
_Avoid_: Merged output when referring to the user-facing result

**Queued Clone Item**:
One text segment queued for voice-clone generation using the same selected Voice Profile.
_Avoid_: Dialogue Line when the text is not part of a conversation

**Voice Clone Queue**:
An ordered set of Queued Clone Items generated as separate audio files using the same selected Voice Profile.
_Avoid_: Conversation when the items are independent outputs

**Dubbing Project**:
A complete dubbing pipeline that takes a source MP3, an original SRT, and a translated SRT, and produces a final dubbed WAV. Voice cloning can use either per-segment Source Segments from the original audio, or user-provided Voice Samples mapped to Voice Types (after Speaker Analysis). The source MP3 is optionally pre-processed through Demucs vocal separation.
_Avoid_: Auto-dub, video dub, SRT voice clone

**Vocals-Separated Audio**:
The output of the Demucs source separation model applied to the source MP3 before dubbing. Contains only the `vocals` stem — all music, sound effects, and background noise are removed. This is the audio source from which all Source Segments are extracted.
_Avoid_: Clean audio, denoised audio — those imply speech enhancement, not source separation

**Source Segment**:
One audio slice extracted from the Vocals-Separated Audio using the timestamp of a single SRT line, padded ±200ms and trimmed of silence. Used as the reference audio for cloning the speaker's voice in that segment.
_Avoid_: Reference audio when referring to timestamp-bound extraction

**Dubbed Segment**:
The synthesized audio output for one translated SRT line, spoken in the voice cloned from the corresponding Source Segment. Timing is adjusted via speed control to fit within the original segment's duration.
_Avoid_: Dialogue Line, Queued Clone Item — those belong to different workflows

**Dubbed Audio**:
The final merged WAV file formed by concatenating all Dubbed Segments and preserved non-speech segments in original SRT order, with timing aligned to the source video.
_Avoid_: Conversation Audio, Merged Queue Audio — those are different workflows

**Non-Speech Segment**:
An SRT line whose content is not spoken dialogue (e.g. `[music]`, `(applause)`). Detected by bracket/parenthesis patterns. The original audio is preserved as-is rather than voice-cloned.
_Avoid_: Source Segment when the content is non-speech

**Speaker Analysis**:
The Gemini-powered process of analyzing the original SRT to identify distinct Voice Types present in the dialogue. Runs per Analysis Window and outputs per-segment Voice Type assignments.
_Avoid_: Speaker detection, diarization — those imply audio-based identification

**Voice Type**:
A speaker classification label combining gender (`male`/`female`) and approximate age (`child`/`young`/`middle-aged`/`old`). Eight possible values (e.g. `male-young`, `female-middle-aged`). Used during dubbing to select the appropriate Voice Sample for each segment.
_Avoid_: Speaker label, character name — those imply identity tracking across scenes

**Analysis Window**:
A chunk of SRT lines grouped by natural subtitle boundaries to approximately 60 seconds of content. Each window is sent to Gemini independently for Speaker Analysis.
_Avoid_: Scene, chunk — those are generic terms

**Voice Sample**:
A user-provided audio file (WAV or MP3) paired with its transcript text, representing one Voice Type. Used as the reference audio for voice cloning all Dubbed Segments assigned to that Voice Type, replacing per-segment Source Segment cloning.
_Avoid_: Reference audio, speaker profile — those belong to different workflows
