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
