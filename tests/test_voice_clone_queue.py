from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "omnivoice"
    / "cli"
    / "voice_clone_queue.py"
)
SPEC = importlib.util.spec_from_file_location("voice_clone_queue", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
voice_clone_queue = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = voice_clone_queue
SPEC.loader.exec_module(voice_clone_queue)

MAX_ITEM_CHARACTERS = voice_clone_queue.MAX_ITEM_CHARACTERS
MAX_QUEUE_ITEMS = voice_clone_queue.MAX_QUEUE_ITEMS
MAX_TOTAL_CHARACTERS = voice_clone_queue.MAX_TOTAL_CHARACTERS
DEFAULT_DOWNLOAD_FORMAT = voice_clone_queue.DEFAULT_DOWNLOAD_FORMAT
DEFAULT_OUTPUT_FORMAT = voice_clone_queue.DEFAULT_OUTPUT_FORMAT
QueuedCloneItem = voice_clone_queue.QueuedCloneItem
ValidationStateError = voice_clone_queue.ValidationStateError
VoiceCloneQueueError = voice_clone_queue.VoiceCloneQueueError
VoiceCloneQueueRequest = voice_clone_queue.VoiceCloneQueueRequest
create_validation_data = voice_clone_queue.create_validation_data
export_queue_csv = voice_clone_queue.export_queue_csv
export_sample_queue_csv = voice_clone_queue.export_sample_queue_csv
generate_voice_clone_queue = voice_clone_queue.generate_voice_clone_queue
import_queue_csv = voice_clone_queue.import_queue_csv
is_validation_data_current = voice_clone_queue.is_validation_data_current
normalize_queue_rows = voice_clone_queue.normalize_queue_rows
queue_items_from_rows = voice_clone_queue.queue_items_from_rows
stale_ready_validation_data = voice_clone_queue.stale_ready_validation_data
validate_generation_ready = voice_clone_queue.validate_generation_ready
validate_queue_request = voice_clone_queue.validate_queue_request
validation_data_signature = voice_clone_queue.validation_data_signature


class FakeDataFrame:
    def __init__(self, records):
        self._records = records

    def to_dict(self, orient="records"):
        if orient != "records":
            raise AssertionError("Unexpected orient")
        return self._records


class FakeProgress:
    def __init__(self):
        self.calls = []

    def __call__(self, value, desc=None):
        self.calls.append({"value": value, "desc": desc})


class FakeModel:
    def __init__(self, *, fail_on_text: str | None = None):
        self.sampling_rate = 22050
        self.fail_on_text = fail_on_text
        self.prompt_calls = []
        self.generate_calls = []

    def create_voice_clone_prompt(self, ref_audio, ref_text):
        prompt = {"ref_audio": ref_audio, "ref_text": ref_text}
        self.prompt_calls.append(prompt)
        return prompt

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.fail_on_text is not None and kwargs["text"] == self.fail_on_text:
            raise RuntimeError(f"boom: {kwargs['text']}")
        length = len(self.generate_calls) + 1
        return [np.full(length, fill_value=float(length), dtype=np.float32)]


class VoiceCloneQueueTests(unittest.TestCase):
    def test_normalize_queue_rows_accepts_dataframe_and_drops_empty_rows(self):
        df = FakeDataFrame([
            {"text": "First line"},
            {"text": ""},
            {"text": "Second line"},
            {"text": None},
        ])
        rows = normalize_queue_rows(df)
        self.assertEqual(rows, [{"text": "First line"}, {"text": "Second line"}])

    def test_normalize_queue_rows_accepts_list_of_dicts(self):
        rows = normalize_queue_rows([{"text": "Line 1"}, {"text": "Line 2"}])
        self.assertEqual(rows, [{"text": "Line 1"}, {"text": "Line 2"}])

    def test_normalize_queue_rows_accepts_list_of_lists(self):
        rows = normalize_queue_rows([["Line 1"], ["Line 2"], ["  "]])
        self.assertEqual(rows, [{"text": "Line 1"}, {"text": "Line 2"}])

    def test_queue_items_from_rows_preserves_order(self):
        items = queue_items_from_rows([["First"], ["Second"], ["Third"]])
        self.assertEqual([item.text for item in items], ["First", "Second", "Third"])

    def test_import_queue_csv_with_utf8_bom(self):
        items = import_queue_csv("\ufefftext\nĐoạn text 1\nĐoạn text 2\n")
        self.assertEqual([item.text for item in items], ["Đoạn text 1", "Đoạn text 2"])

    def test_export_queue_csv_formats_correctly(self):
        csv_text = export_queue_csv([
            QueuedCloneItem(text="First line"),
            QueuedCloneItem(text="Second line"),
        ])
        self.assertEqual(csv_text, "text\nFirst line\nSecond line\n")

    def test_export_queue_csv_neutralizes_formulas(self):
        csv_text = export_queue_csv([
            QueuedCloneItem(text="=SUM(A1:A10)"),
            QueuedCloneItem(text="+HYPERLINK"),
            QueuedCloneItem(text="-A1"),
            QueuedCloneItem(text="@cmd"),
        ])
        self.assertIn("'=SUM(A1:A10)", csv_text)
        self.assertIn("'+HYPERLINK", csv_text)
        self.assertIn("'-A1", csv_text)
        self.assertIn("'@cmd", csv_text)

    def test_export_sample_queue_csv_returns_header_only(self):
        self.assertEqual(export_sample_queue_csv(), "text\n")

    def test_validate_queue_request_requires_item(self):
        request = VoiceCloneQueueRequest(items=[], ref_audio=object(), ref_text="", language="auto")
        with self.assertRaisesRegex(VoiceCloneQueueError, "At least one"):
            validate_queue_request(request)

    def test_validate_queue_request_requires_reference_audio(self):
        request = VoiceCloneQueueRequest(
            items=[QueuedCloneItem(text="Hello")],
            ref_audio=None,
            ref_text="",
            language="auto",
        )
        with self.assertRaisesRegex(VoiceCloneQueueError, "Reference Audio"):
            validate_queue_request(request)

    def test_validate_queue_request_enforces_limits_and_returns_summary(self):
        items = [QueuedCloneItem(text="Hello"), QueuedCloneItem(text="World")]
        request = VoiceCloneQueueRequest(items=items, ref_audio=object(), ref_text="", language="auto")
        summary = validate_queue_request(request)
        self.assertEqual(summary.queue_items, 2)
        self.assertEqual(summary.total_characters, 10)
        self.assertEqual(summary.max_total_characters, MAX_TOTAL_CHARACTERS)
        self.assertEqual(summary.output_format, DEFAULT_OUTPUT_FORMAT)
        self.assertEqual(summary.download_all_format, DEFAULT_DOWNLOAD_FORMAT)

    def test_validate_queue_request_rejects_too_many_items(self):
        items = [QueuedCloneItem(text="x") for _ in range(MAX_QUEUE_ITEMS + 1)]
        request = VoiceCloneQueueRequest(items=items, ref_audio=object(), ref_text="", language="auto")
        with self.assertRaisesRegex(VoiceCloneQueueError, str(MAX_QUEUE_ITEMS)):
            validate_queue_request(request)

    def test_validate_queue_request_rejects_item_too_long(self):
        items = [QueuedCloneItem(text="x" * (MAX_ITEM_CHARACTERS + 1))]
        request = VoiceCloneQueueRequest(items=items, ref_audio=object(), ref_text="", language="auto")
        with self.assertRaisesRegex(VoiceCloneQueueError, str(MAX_ITEM_CHARACTERS)):
            validate_queue_request(request)

    def test_validate_queue_request_rejects_total_too_long(self):
        items = [QueuedCloneItem(text="x" * 3000) for _ in range(7)]
        request = VoiceCloneQueueRequest(items=items, ref_audio=object(), ref_text="", language="auto")
        with self.assertRaisesRegex(VoiceCloneQueueError, str(MAX_TOTAL_CHARACTERS)):
            validate_queue_request(request)

    def test_validation_data_helpers_detect_staleness(self):
        request = VoiceCloneQueueRequest(
            items=[QueuedCloneItem(text="Hello")],
            ref_audio="ref.wav",
            ref_text="",
            language="Auto",
            instruct="",
        )
        current = create_validation_data(request)
        self.assertTrue(is_validation_data_current(current, request))
        stale = stale_ready_validation_data(current)
        self.assertFalse(is_validation_data_current(stale, request))

    def test_validate_generation_ready_requires_fresh_validation(self):
        request = VoiceCloneQueueRequest(
            items=[QueuedCloneItem(text="Hello")],
            ref_audio="ref.wav",
            ref_text="",
            language="Auto",
            instruct="",
        )
        validation_data = create_validation_data(request)
        validate_generation_ready(validation_data, request)
        stale = stale_ready_validation_data(validation_data)
        with self.assertRaises(ValidationStateError):
            validate_generation_ready(stale, request)

    def test_generate_voice_clone_queue_reuses_prompt_and_writes_files_and_zip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel()
            progress = FakeProgress()
            request = VoiceCloneQueueRequest(
                items=[QueuedCloneItem(text="One"), QueuedCloneItem(text="Two")],
                ref_audio="ref.wav",
                ref_text="",
                language="Auto",
                instruct="be expressive",
                output_dir=Path(temp_dir),
                speed=1.25,
                duration=9.5,
            )
            result = generate_voice_clone_queue(
                model=model,
                request=request,
                generation_config={"steps": 10},
                progress=progress,
            )

            self.assertEqual(len(model.prompt_calls), 1)
            self.assertEqual(model.prompt_calls[0], {"ref_audio": "ref.wav", "ref_text": None})
            self.assertEqual([call["text"] for call in model.generate_calls], ["One", "Two"])
            self.assertEqual([call["language"] for call in model.generate_calls], ["Auto", "Auto"])
            self.assertEqual([call["voice_clone_prompt"] for call in model.generate_calls], [model.prompt_calls[0], model.prompt_calls[0]])
            self.assertEqual([call["instruct"] for call in model.generate_calls], ["be expressive", "be expressive"])
            self.assertEqual([call["duration"] for call in model.generate_calls], [9.5, 9.5])
            self.assertEqual([call["speed"] for call in model.generate_calls], [1.25, 1.25])
            self.assertEqual([call["generation_config"] for call in model.generate_calls], [{"steps": 10}, {"steps": 10}])
            self.assertEqual(len(result.wav_paths), 2)
            self.assertTrue(Path(result.zip_path).exists())
            self.assertEqual(result.metadata["queue_items"], 2)
            self.assertEqual(result.metadata["sampling_rate"], 22050)
            self.assertEqual(result.metadata["zip_members"], [Path(path).name for path in result.wav_paths])
            for wav_path in result.wav_paths:
                self.assertTrue(Path(wav_path).exists())
                audio, sample_rate = sf.read(wav_path)
                self.assertEqual(sample_rate, 22050)
                self.assertGreater(len(audio), 0)
            self.assertEqual([call["desc"] for call in progress.calls], ["Generating item 1/2", "Generating item 2/2"])

    def test_generate_voice_clone_queue_cleans_up_all_outputs_on_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = FakeModel(fail_on_text="Two")
            request = VoiceCloneQueueRequest(
                items=[QueuedCloneItem(text="One"), QueuedCloneItem(text="Two")],
                ref_audio="ref.wav",
                ref_text="ref text",
                language=None,
                output_dir=Path(temp_dir),
            )
            with self.assertRaisesRegex(VoiceCloneQueueError, "Two"):
                generate_voice_clone_queue(
                    model=model,
                    request=request,
                    generation_config={"steps": 10},
                )
            remaining_files = list(Path(temp_dir).rglob("*"))
            self.assertEqual(remaining_files, [])


if __name__ == "__main__":
    unittest.main()
