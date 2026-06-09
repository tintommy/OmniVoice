from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "omnivoice"
    / "cli"
    / "conversation_voice_clone.py"
)
SPEC = importlib.util.spec_from_file_location("conversation_voice_clone", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
conversation_voice_clone = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = conversation_voice_clone
SPEC.loader.exec_module(conversation_voice_clone)

ConversationAudioMergeError = conversation_voice_clone.ConversationAudioMergeError
ConversationRequest = conversation_voice_clone.ConversationRequest
ConversationVoiceCloneError = conversation_voice_clone.ConversationVoiceCloneError
DialogueLine = conversation_voice_clone.DialogueLine
VoiceProfile = conversation_voice_clone.VoiceProfile
export_dialogue_lines_csv = conversation_voice_clone.export_dialogue_lines_csv
generate_conversation_audio = conversation_voice_clone.generate_conversation_audio
import_dialogue_lines_csv = conversation_voice_clone.import_dialogue_lines_csv
merge_conversation_audio = conversation_voice_clone.merge_conversation_audio
normalize_dialogue_rows = conversation_voice_clone.normalize_dialogue_rows
validate_conversation_request = conversation_voice_clone.validate_conversation_request


class FakeDataFrame:
    def __init__(self, records):
        self._records = records

    def to_dict(self, orient="records"):
        if orient != "records":
            raise AssertionError("Unexpected orient")
        return self._records


class FakeModel:
    def __init__(self):
        self.sampling_rate = 24000
        self.prompt_calls = []
        self.generate_calls = []

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        prompt = {"ref_audio": ref_audio, "ref_text": ref_text}
        self.prompt_calls.append(prompt)
        return prompt

    def generate(self, *, text, language, voice_clone_prompt, speed):
        self.generate_calls.append(
            {
                "text": text,
                "language": language,
                "voice_clone_prompt": voice_clone_prompt,
                "speed": speed,
            }
        )
        length = len(self.generate_calls) + 1
        return [np.full(length, fill_value=float(length), dtype=np.float32)]


class ConversationVoiceCloneTests(unittest.TestCase):
    def test_normalize_dialogue_rows_accepts_multiple_shapes_and_drops_empty_rows(self):
        rows = normalize_dialogue_rows(
            FakeDataFrame(
                [
                    {"speaker_name": " Alice ", "text": " Hello "},
                    {"speaker_name": "  ", "text": "   "},
                ]
            )
        )
        self.assertEqual(rows, [{"speaker_name": "Alice", "text": "Hello"}])

        rows = normalize_dialogue_rows(
            [
                [" Bob ", " Hi "],
                [None, None],
                {"speaker_name": "Cara", "text": " Yo "},
            ]
        )
        self.assertEqual(
            rows,
            [
                {"speaker_name": "Bob", "text": "Hi"},
                {"speaker_name": "Cara", "text": "Yo"},
            ],
        )

    def test_validate_conversation_request_rejects_auto_language_and_unknown_speaker(self):
        request = ConversationRequest(
            voice_profiles=[
                VoiceProfile(speaker_name="Alice", ref_audio="a.wav", ref_text="ref")
            ],
            dialogue_lines=[DialogueLine(speaker_name="Bob", text="Hello")],
            language="Auto",
        )

        with self.assertRaisesRegex(
            ConversationVoiceCloneError, "Unknown speaker_name: Bob"
        ):
            validate_conversation_request(request)

        fixed_request = ConversationRequest(
            voice_profiles=request.voice_profiles,
            dialogue_lines=[DialogueLine(speaker_name="Alice", text="Hello")],
            language="Auto",
        )
        with self.assertRaisesRegex(
            ConversationVoiceCloneError,
            "Language is required and cannot be empty or Auto",
        ):
            validate_conversation_request(fixed_request)

    def test_csv_import_export_only_supports_expected_columns(self):
        dialogue_lines = [
            DialogueLine(speaker_name="Alice", text='Xin chào, "Bob"'),
            DialogueLine(speaker_name="Bob", text="Chào Alice"),
        ]

        exported = export_dialogue_lines_csv(dialogue_lines)
        self.assertEqual(exported.splitlines()[0], "speaker_name,text")

        imported = import_dialogue_lines_csv(exported)
        self.assertEqual(imported, dialogue_lines)

        bom_imported = import_dialogue_lines_csv(
            "\ufeffspeaker_name,text\nAlice,Tiếng Việt có dấu: xin chào thế giới\n"
        )
        self.assertEqual(
            bom_imported,
            [DialogueLine(speaker_name="Alice", text="Tiếng Việt có dấu: xin chào thế giới")],
        )

        formula_exported = export_dialogue_lines_csv(
            [DialogueLine(speaker_name="Alice", text="=IMPORTDATA('http://example.com')")]
        )
        self.assertIn("'=IMPORTDATA", formula_exported)

        with self.assertRaisesRegex(
            ConversationVoiceCloneError,
            "CSV must contain exactly the columns: speaker_name,text",
        ):
            import_dialogue_lines_csv("speaker,text\nAlice,Hello\n")

    def test_merge_conversation_audio_inserts_silence_and_writes_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = merge_conversation_audio(
                clips=[
                    np.array([0.1, 0.2], dtype=np.float32),
                    np.array([0.3], dtype=np.float32),
                ],
                pause_ms=100,
                sample_rate=24000,
                output_dir=temp_dir,
                timestamp=datetime(2026, 6, 9, 15, 30, 12),
            )

            output_name = Path(output_path).name
            self.assertTrue(output_name.startswith("conversation_voice_clone_20260609_153012_"))
            self.assertTrue(output_name.endswith(".wav"))
            audio, sample_rate = sf.read(output_path, dtype="float32")
            self.assertEqual(sample_rate, 24000)
            self.assertEqual(audio.shape, (2 + 2400 + 1,))
            self.assertTrue(
                np.allclose(audio[:2], np.array([0.1, 0.2], dtype=np.float32), atol=5e-5)
            )
            self.assertTrue(np.allclose(audio[2 : 2 + 2400], 0.0, atol=5e-5))
            self.assertTrue(
                np.allclose(audio[-1:], np.array([0.3], dtype=np.float32), atol=5e-5)
            )

    def test_merge_conversation_audio_rejects_non_mono_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                ConversationAudioMergeError,
                "Generated clip 1 must be mono audio at 24000 Hz",
            ):
                merge_conversation_audio(
                    clips=[np.zeros((2, 2), dtype=np.float32)],
                    pause_ms=0,
                    sample_rate=24000,
                    output_dir=temp_dir,
                )

    def test_generate_conversation_audio_reuses_prompt_cache_and_generates_sequentially(self):
        model = FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            request = ConversationRequest(
                voice_profiles=[
                    VoiceProfile(
                        speaker_name="Alice",
                        ref_audio="alice.wav",
                        ref_text="Alice ref",
                    ),
                    VoiceProfile(
                        speaker_name="Bob",
                        ref_audio="bob.wav",
                        ref_text="Bob ref",
                    ),
                ],
                dialogue_lines=[
                    DialogueLine(speaker_name="Alice", text="Line one"),
                    DialogueLine(speaker_name="Alice", text="Line two"),
                    DialogueLine(speaker_name="Bob", text="Line three"),
                ],
                language="English",
                pause_ms=0,
                speed=1.25,
                output_dir=Path(temp_dir),
            )

            result = generate_conversation_audio(model, request)
            self.assertTrue(Path(result.audio_path).exists())

        self.assertEqual(len(model.prompt_calls), 2)
        self.assertEqual(
            [call["text"] for call in model.generate_calls],
            ["Line one", "Line two", "Line three"],
        )
        self.assertIs(
            model.generate_calls[0]["voice_clone_prompt"],
            model.generate_calls[1]["voice_clone_prompt"],
        )
        self.assertIsNot(
            model.generate_calls[0]["voice_clone_prompt"],
            model.generate_calls[2]["voice_clone_prompt"],
        )
        self.assertEqual(result.audio_path, result.download_path)
        self.assertEqual(result.metadata["dialogue_lines_generated"], 3)
        self.assertEqual(
            result.metadata["total_text_characters"],
            len("Line oneLine twoLine three"),
        )
        self.assertEqual(result.metadata["output_filename"], Path(result.audio_path).name)


if __name__ == "__main__":
    unittest.main()
