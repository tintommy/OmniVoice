#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

from omnivoice.cli.conversation_voice_clone import (
    ConversationVoiceCloneError,
    DialogueLine,
    export_dialogue_lines_csv,
    generate_conversation_voice_clone,
    import_dialogue_lines_csv,
    validate_conversation_voice_clone,
)

from omnivoice.cli.voice_clone_queue import (
    QueuedCloneItem,
    VoiceCloneQueueError,
    VoiceCloneQueueRequest,
    export_queue_csv,
    export_sample_queue_csv,
    generate_voice_clone_queue,
    import_queue_csv,
    normalize_queue_rows,
    validate_queue_request,
)

QUEUE_AVAILABLE = True


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    def _load_reference_text_file(file_obj: Any) -> str:
        if file_obj is None:
            return ""

        file_path = getattr(file_obj, "name", file_obj)
        if not isinstance(file_path, str):
            raise gr.Error("Unsupported reference text file input.")

        path = Path(file_path)
        if path.suffix.lower() != ".txt":
            raise gr.Error("Reference text file must be a .txt file.")

        try:
            return path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise gr.Error("Reference text file must be UTF-8 encoded.") from exc

    def _conversation_language_choices() -> list[str]:
        return [lang for lang in _ALL_LANGUAGES if lang != "Auto"]

    def _empty_dialogue_rows() -> list[list[str]]:
        return [["", ""] for _ in range(5)]

    def _normalize_dialogue_rows(rows: Any) -> list[dict[str, str]]:
        if rows is None:
            return []
        if hasattr(rows, "to_dict"):
            try:
                rows = rows.to_dict(orient="records")
            except TypeError:
                rows = rows.to_dict()
        normalized = []
        for row in rows or []:
            if isinstance(row, dict):
                speaker_name = str(row.get("speaker_name", "") or "").strip()
                text = str(row.get("text", "") or "").strip()
            else:
                values = list(row) if isinstance(row, (list, tuple)) else [row]
                speaker_name = str(values[0] if len(values) > 0 else "").strip()
                text = str(values[1] if len(values) > 1 else "").strip()
            if speaker_name or text:
                normalized.append({"speaker_name": speaker_name, "text": text})
        return normalized

    def _collect_voice_profiles(*slot_values: Any) -> list[dict[str, Any]]:
        profiles = []
        for index in range(0, len(slot_values), 4):
            slot_number = index // 4 + 1
            enabled, speaker_name, ref_audio, ref_text = slot_values[index : index + 4]
            file_path = getattr(ref_audio, "name", ref_audio) if ref_audio is not None else None
            profiles.append(
                {
                    "slot_index": slot_number,
                    "enabled": bool(enabled),
                    "speaker_name": str(speaker_name or "").strip(),
                    "ref_audio": file_path,
                    "ref_text": str(ref_text or "").strip(),
                }
            )
        return profiles

    def _conversation_validation_fingerprint(result: dict[str, Any]) -> str:
        payload = {
            "voice_profiles": result["voice_profiles"],
            "dialogue_lines": result["dialogue_lines"],
            "language": result["language"],
            "speed": result["speed"],
            "pause_ms": result["pause_ms"],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _validate_conversation_adapter(
        language: str,
        speed: float,
        pause_ms: float,
        dialogue_rows: Any,
        *slot_values: Any,
    ):
        voice_profiles = _collect_voice_profiles(*slot_values)
        dialogue_lines = _normalize_dialogue_rows(dialogue_rows)
        try:
            result = validate_conversation_voice_clone(
                voice_profiles=voice_profiles,
                dialogue_lines=dialogue_lines,
                language=language,
                speed=float(speed),
                pause_ms=int(pause_ms),
            )
        except ConversationVoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc
        result["validation_fingerprint"] = _conversation_validation_fingerprint(result)
        summary = (
            f"Enabled Voice Profiles: {result['enabled_voice_profiles_count']}\n"
            f"Dialogue Lines: {result['dialogue_lines_count']}\n"
            f"Total characters: {result['total_characters']} / 20000\n"
            f"Pause duration: {result['pause_ms']} ms\n"
            f"Output format: {result.get('output_format', 'WAV')}"
        )
        return result, summary, "Validation passed."

    def _generate_conversation_adapter(
        validation_state: dict[str, Any] | None,
        language: str,
        speed: float,
        pause_ms: float,
        dialogue_rows: Any,
        *slot_values: Any,
        progress=gr.Progress(),
    ):
        if validation_state is None:
            raise gr.Error("Please click Validate before Generate.")
        current_result, summary, _ = _validate_conversation_adapter(
            language,
            speed,
            pause_ms,
            dialogue_rows,
            *slot_values,
        )
        if validation_state.get("validation_fingerprint") != current_result.get(
            "validation_fingerprint"
        ):
            raise gr.Error("Inputs changed. Please click Validate again before Generate.")
        try:
            metadata = generate_conversation_voice_clone(
                model=model,
                validation_result=validation_state,
                progress=progress,
            )
        except ConversationVoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc
        metadata_text = (
            f"Dialogue Lines generated: {metadata['dialogue_lines_generated']}\n"
            f"Total text characters: {metadata['total_characters']}\n"
            f"Output filename: {metadata['output_filename']}"
        )
        return (
            None,
            summary,
            metadata.get("status", "Done. Please click Validate again before generating another conversation."),
            metadata.get("audio_path"),
            metadata.get("download_path"),
            metadata_text,
        )

    def _reset_conversation_outputs():
        return None, _empty_dialogue_rows(), "", "", None, None, ""

    def _import_dialogue_csv(file_obj: Any):
        if file_obj is None:
            return _empty_dialogue_rows(), "", ""

        file_path = getattr(file_obj, "name", file_obj)
        if not isinstance(file_path, str):
            raise gr.Error("Unsupported CSV input.")

        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise gr.Error("Dialogue CSV must be a .csv file.")

        try:
            imported_lines = import_dialogue_lines_csv(path.read_text(encoding="utf-8-sig"))
        except ConversationVoiceCloneError as exc:
            raise gr.Error(str(exc)) from exc

        rows = [[line.speaker_name, line.text] for line in imported_lines]
        return rows or _empty_dialogue_rows(), "", "Dialogue CSV loaded."

    def _export_dialogue_csv(dialogue_rows: Any):
        rows = _normalize_dialogue_rows(dialogue_rows)
        if not rows:
            raise gr.Error("Add at least one Dialogue Line before export.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(tempfile.gettempdir()) / f"conversation_voice_clone_dialogue_{timestamp}.csv"
        dialogue_lines = [
            DialogueLine(speaker_name=row["speaker_name"], text=row["text"])
            for row in rows
        ]
        output_path.write_text(
            export_dialogue_lines_csv(dialogue_lines),
            encoding="utf-8-sig",
        )
        return str(output_path), "Dialogue CSV exported."

    def _export_sample_dialogue_csv():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(tempfile.gettempdir()) / f"conversation_voice_clone_dialogue_sample_{timestamp}.csv"
        output_path.write_text("speaker_name,text\n", encoding="utf-8-sig")
        return str(output_path), "Sample Dialogue CSV exported."

    def _empty_queue_rows() -> list[list[str]]:
        return [[""] for _ in range(5)]

    def _normalize_queue_rows(rows: Any) -> list[dict[str, str]]:
        return normalize_queue_rows(rows)

    def _queue_validation_fingerprint(result: dict[str, Any]) -> str:
        payload = {
            "queue_items": result["queue_items"],
            "reference_audio": result["reference_audio"],
            "reference_text": result.get("reference_text"),
            "language": result["language"],
            "instruct": result.get("instruct"),
            "generation_config": result["generation_config"],
            "speed": result.get("speed"),
            "duration": result.get("duration"),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _validate_queue_adapter(
        ref_audio: str | None,
        ref_text: str,
        language: str,
        instruct: str,
        num_step: float,
        guidance_scale: float,
        denoise: bool,
        speed: float | None,
        duration: float | None,
        preprocess_prompt: bool,
        postprocess_output: bool,
        queue_rows: Any,
    ):
        queue_items = [QueuedCloneItem(text=row["text"]) for row in _normalize_queue_rows(queue_rows)]
        generation_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )
        lang = language if (language and language != "Auto") else None
        duration_value = None if duration is None or float(duration) <= 0 else float(duration)
        speed_value = None if speed is None else float(speed)
        request = VoiceCloneQueueRequest(
            items=queue_items,
            ref_audio=ref_audio,
            ref_text=(ref_text or None),
            language=lang,
            instruct=(instruct or None),
            speed=1.0 if speed_value is None else speed_value,
            duration=duration_value,
        )
        try:
            summary_obj = validate_queue_request(request)
        except VoiceCloneQueueError as exc:
            raise gr.Error(str(exc)) from exc
        result = {
            "queue_items": [{"text": item.text} for item in queue_items],
            "queue_items_count": summary_obj.queue_items,
            "total_characters": summary_obj.total_characters,
            "output_format": summary_obj.output_format,
            "download_all_format": summary_obj.download_all_format,
        }
        result["generation_config"] = {
            "num_step": int(num_step or 32),
            "guidance_scale": float(guidance_scale) if guidance_scale is not None else 2.0,
            "denoise": bool(denoise) if denoise is not None else True,
            "preprocess_prompt": bool(preprocess_prompt),
            "postprocess_output": bool(postprocess_output),
        }
        result["reference_audio"] = ref_audio
        result["reference_text"] = ref_text or None
        result["language"] = lang
        result["instruct"] = instruct or None
        result["speed"] = None if speed is None else float(speed)
        result["duration"] = None if duration is None or float(duration) <= 0 else float(duration)
        result["validation_fingerprint"] = _queue_validation_fingerprint(result)
        summary = (
            f"Queue items: {result['queue_items_count']}\n"
            f"Total characters: {result['total_characters']} / 20000\n"
            f"Output format: {result.get('output_format', 'WAV')}\n"
            f"Download all format: {result.get('download_all_format', 'ZIP')}"
        )
        return result, summary, "Validation passed."

    def _generate_queue_adapter(
        validation_state: dict[str, Any] | None,
        ref_audio: str | None,
        ref_text: str,
        language: str,
        instruct: str,
        num_step: float,
        guidance_scale: float,
        denoise: bool,
        speed: float | None,
        duration: float | None,
        preprocess_prompt: bool,
        postprocess_output: bool,
        queue_rows: Any,
        progress=gr.Progress(),
    ):
        if validation_state is None:
            raise gr.Error("Please click Validate Queue before Generate Queue.")
        current_result, summary, _ = _validate_queue_adapter(
            ref_audio,
            ref_text,
            language,
            instruct,
            num_step,
            guidance_scale,
            denoise,
            speed,
            duration,
            preprocess_prompt,
            postprocess_output,
            queue_rows,
        )
        if validation_state.get("validation_fingerprint") != current_result.get(
            "validation_fingerprint"
        ):
            raise gr.Error(
                "Queue inputs changed. Please click Validate Queue again before Generate Queue."
            )
        generation_config = OmniVoiceGenerationConfig(
            num_step=validation_state["generation_config"]["num_step"],
            guidance_scale=validation_state["generation_config"]["guidance_scale"],
            denoise=validation_state["generation_config"]["denoise"],
            preprocess_prompt=validation_state["generation_config"]["preprocess_prompt"],
            postprocess_output=validation_state["generation_config"]["postprocess_output"],
        )
        try:
            request = VoiceCloneQueueRequest(
                items=[
                    QueuedCloneItem(text=item["text"])
                    for item in validation_state["queue_items"]
                ],
                ref_audio=validation_state["reference_audio"],
                ref_text=validation_state.get("reference_text"),
                language=validation_state.get("language"),
                instruct=validation_state.get("instruct"),
                speed=1.0 if validation_state.get("speed") is None else float(validation_state["speed"]),
                duration=validation_state.get("duration"),
            )
            result = generate_voice_clone_queue(
                model=model,
                request=request,
                generation_config=generation_config,
                progress=progress,
            )
            metadata = {
                "wav_paths": result.wav_paths,
                "zip_path": result.zip_path,
                "queue_items_generated": result.metadata["queue_items"],
                "total_characters": result.metadata["total_characters"],
                "zip_filename": Path(result.zip_path).name,
                "status": "Done. Please click Validate Queue again before generating another queue.",
            }
        except VoiceCloneQueueError as exc:
            raise gr.Error(str(exc)) from exc
        metadata_text = (
            f"Queue items generated: {metadata['queue_items_generated']}\n"
            f"Total text characters: {metadata['total_characters']}\n"
            f"ZIP filename: {metadata['zip_filename']}"
        )
        return (
            None,
            summary,
            metadata.get(
                "status",
                "Done. Please click Validate Queue again before generating another queue.",
            ),
            metadata.get("wav_paths") or metadata.get("audio_paths"),
            metadata.get("zip_path") or metadata.get("download_path"),
            metadata_text,
        )

    def _import_queue_csv(file_obj: Any):
        if file_obj is None:
            return _empty_queue_rows(), "", ""
        file_path = getattr(file_obj, "name", file_obj)
        if not isinstance(file_path, str):
            raise gr.Error("Unsupported CSV input.")
        path = Path(file_path)
        if path.suffix.lower() != ".csv":
            raise gr.Error("Queue CSV must be a .csv file.")

        try:
            imported_items = import_queue_csv(path.read_text(encoding="utf-8-sig"))
        except VoiceCloneQueueError as exc:
            raise gr.Error(str(exc)) from exc

        rows = [[item.text] for item in imported_items]
        return rows or _empty_queue_rows(), "", "Queue CSV loaded."

    def _export_queue_csv(queue_rows: Any):
        rows = _normalize_queue_rows(queue_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(tempfile.gettempdir()) / f"voice_clone_queue_{timestamp}.csv"
        queue_items = [QueuedCloneItem(text=row["text"]) for row in rows]
        output_path.write_text(
            export_queue_csv(queue_items),
            encoding="utf-8-sig",
        )
        return str(output_path), "Queue CSV exported."

    def _export_sample_queue_csv():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(tempfile.gettempdir()) / f"voice_clone_queue_sample_{timestamp}.csv"
        output_path.write_text(export_sample_queue_csv(), encoding="utf-8-sig")
        return str(output_path), "Sample Queue CSV exported."

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    #queue-items-table .table-wrap {max-height: 420px !important; overflow-y: auto !important;}
    #queue-items-table textarea {max-height: 160px !important; overflow-y: auto !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional)" " / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_ref_text_file = gr.File(
                            label="Reference Text File (.txt)",
                            file_types=[".txt"],
                            type="filepath",
                        )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )

                vc_ref_text_file.change(
                    _load_reference_text_file,
                    inputs=vc_ref_text_file,
                    outputs=vc_ref_text,
                )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

                # ==============================================================
                # Queue Generation Section
                # ==============================================================
                if QUEUE_AVAILABLE:
                    with gr.Accordion("Queue Generation", open=False):
                        queue_validation_state = gr.State(value=None)
                        
                        gr.Markdown("### Queue Settings")
                        gr.Markdown(
                            "Queue Generation uses the same Reference Audio/Text and settings from above. "
                            "Each text item in the queue will be generated with those shared settings."
                        )
                        
                        with gr.Row():
                            queue_csv_import = gr.File(
                                label="Import Queue CSV",
                                file_types=[".csv"],
                                type="filepath",
                            )
                            queue_sample_btn = gr.Button("Export Sample CSV")
                            queue_sample_file = gr.File(label="Sample CSV")
                            queue_export_btn = gr.Button("Export Queue CSV")
                            queue_export_file = gr.File(label="Queue CSV")
                        
                        gr.Markdown("### Queue Items")
                        gr.Markdown(
                            "Add text items to generate. Each row will produce one WAV file. Maximum 20 items, 3000 chars per item, 20000 total chars."
                        )
                        queue_df = gr.Dataframe(
                            headers=["text"],
                            datatype=["str"],
                            value=_empty_queue_rows(),
                            row_count=(5, "dynamic"),
                            col_count=(1, "fixed"),
                            label="Queue Items",
                            wrap=True,
                            max_height=420,
                            column_widths=["100%"],
                            elem_id="queue-items-table",
                        )
                        
                        with gr.Row():
                            queue_validate_btn = gr.Button("Validate Queue")
                            queue_generate_btn = gr.Button("Generate Queue", variant="primary")
                        
                        queue_summary = gr.Textbox(label="Queue Summary", lines=4)
                        queue_status = gr.Textbox(label="Queue Status", lines=2)
                        queue_wav_files = gr.File(
                            label="Generated Queue WAV Files",
                            file_count="multiple",
                        )
                        queue_zip_file = gr.File(label="Download All Queue WAVs (ZIP)")
                        queue_metadata = gr.Textbox(label="Queue Metadata", lines=3)

                        queue_csv_import.change(
                            _import_queue_csv,
                            inputs=queue_csv_import,
                            outputs=[queue_df, queue_summary, queue_status],
                        )
                        queue_export_btn.click(
                            _export_queue_csv,
                            inputs=queue_df,
                            outputs=[queue_export_file, queue_status],
                        )
                        queue_sample_btn.click(
                            _export_sample_queue_csv,
                            outputs=[queue_sample_file, queue_status],
                        )
                        queue_validate_btn.click(
                            _validate_queue_adapter,
                            inputs=[
                                vc_ref_audio,
                                vc_ref_text,
                                vc_lang,
                                vc_instruct,
                                vc_ns,
                                vc_gs,
                                vc_dn,
                                vc_sp,
                                vc_du,
                                vc_pp,
                                vc_po,
                                queue_df,
                            ],
                            outputs=[queue_validation_state, queue_summary, queue_status],
                        )
                        queue_generate_btn.click(
                            _generate_queue_adapter,
                            inputs=[
                                queue_validation_state,
                                vc_ref_audio,
                                vc_ref_text,
                                vc_lang,
                                vc_instruct,
                                vc_ns,
                                vc_gs,
                                vc_dn,
                                vc_sp,
                                vc_du,
                                vc_pp,
                                vc_po,
                                queue_df,
                            ],
                            outputs=[
                                queue_validation_state,
                                queue_summary,
                                queue_status,
                                queue_wav_files,
                                queue_zip_file,
                                queue_metadata,
                            ],
                        )

            # ==============================================================
            # Conversation Voice Clone
            # ==============================================================
            with gr.TabItem("Conversation Voice Clone"):
                conversation_validation_state = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Voice Profiles")
                        cvc_slot_inputs = []
                        for slot_idx in range(1, 6):
                            with gr.Accordion(f"Voice Profile {slot_idx}", open=(slot_idx == 1)):
                                cvc_enabled = gr.Checkbox(
                                    label="Enable this speaker",
                                    value=(slot_idx == 1),
                                )
                                cvc_speaker_name = gr.Textbox(
                                    label="Speaker Name",
                                    placeholder=f"Speaker {slot_idx}",
                                )
                                cvc_ref_audio = gr.Audio(
                                    label="Reference Voice",
                                    type="filepath",
                                    elem_classes="compact-audio",
                                )
                                gr.Markdown(
                                    "<span style='font-size:0.85em;color:#888;'>"
                                    "Recommended reference audio: 3–10 seconds. Longer clips may slow inference or reduce cloning quality."
                                    "</span>"
                                )
                                cvc_ref_text = gr.Textbox(
                                    label="Reference Text",
                                    lines=2,
                                    placeholder="Enter the transcript for the reference voice, or upload a .txt file below.",
                                )
                                cvc_ref_text_file = gr.File(
                                    label="Reference Text File (.txt)",
                                    file_types=[".txt"],
                                    type="filepath",
                                )
                                cvc_ref_text_file.change(
                                    _load_reference_text_file,
                                    inputs=cvc_ref_text_file,
                                    outputs=cvc_ref_text,
                                )
                                cvc_slot_inputs.extend(
                                    [
                                        cvc_enabled,
                                        cvc_speaker_name,
                                        cvc_ref_audio,
                                        cvc_ref_text,
                                    ]
                                )

                    with gr.Column(scale=1):
                        gr.Markdown("### Conversation Settings")
                        cvc_lang = gr.Dropdown(
                            label="Language / 语种",
                            choices=_conversation_language_choices(),
                            value=_conversation_language_choices()[0],
                            allow_custom_value=False,
                            interactive=True,
                        )
                        cvc_sp = gr.Slider(
                            0.5,
                            1.5,
                            value=1.0,
                            step=0.05,
                            label="Speed",
                            info="1.0 = normal. >1 faster, <1 slower.",
                        )
                        cvc_pause = gr.Slider(
                            0,
                            3000,
                            value=300,
                            step=50,
                            label="Pause between lines (ms)",
                        )
                        with gr.Row():
                            cvc_validate_btn = gr.Button("Validate")
                            cvc_generate_btn = gr.Button("Generate", variant="primary")
                            cvc_reset_btn = gr.Button("Reset Conversation")

                        cvc_summary = gr.Textbox(label="Summary", lines=5)
                        cvc_status = gr.Textbox(label="Status / 状态", lines=2)
                        cvc_audio = gr.Audio(
                            label="Conversation Audio",
                            type="filepath",
                        )
                        cvc_download = gr.File(label="Download WAV")
                        cvc_metadata = gr.Textbox(label="Metadata", lines=3)

                gr.Markdown("### Dialogue Lines")
                gr.Markdown(
                    "Use `speaker_name,text` CSV format. The speaker column is intentionally narrow; put long Vietnamese dialogue in the text column."
                )
                with gr.Row():
                    cvc_dialogue_csv = gr.File(
                        label="Import Dialogue CSV",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    cvc_sample_btn = gr.Button("Export Sample CSV")
                    cvc_sample_file = gr.File(label="Sample CSV")
                cvc_dialogue_df = gr.Dataframe(
                    headers=["speaker_name", "text"],
                    datatype=["str", "str"],
                    row_count=(5, "dynamic"),
                    col_count=(2, "fixed"),
                    value=_empty_dialogue_rows(),
                    interactive=True,
                    wrap=True,
                    label="Dialogue Lines",
                    max_height=700,
                    column_widths=["30%", "70%"],
                )
                with gr.Row():
                    cvc_export_btn = gr.Button("Export Dialogue CSV")
                    cvc_export_file = gr.File(label="Dialogue CSV Export")

                cvc_validate_inputs = [
                    cvc_lang,
                    cvc_sp,
                    cvc_pause,
                    cvc_dialogue_df,
                    *cvc_slot_inputs,
                ]

                cvc_validate_btn.click(
                    _validate_conversation_adapter,
                    inputs=cvc_validate_inputs,
                    outputs=[
                        conversation_validation_state,
                        cvc_summary,
                        cvc_status,
                    ],
                )

                cvc_generate_btn.click(
                    _generate_conversation_adapter,
                    inputs=[conversation_validation_state, *cvc_validate_inputs],
                    outputs=[
                        conversation_validation_state,
                        cvc_summary,
                        cvc_status,
                        cvc_audio,
                        cvc_download,
                        cvc_metadata,
                    ],
                )

                cvc_reset_btn.click(
                    _reset_conversation_outputs,
                    outputs=[
                        conversation_validation_state,
                        cvc_dialogue_df,
                        cvc_summary,
                        cvc_status,
                        cvc_audio,
                        cvc_download,
                        cvc_metadata,
                    ],
                )

                cvc_dialogue_csv.change(
                    _import_dialogue_csv,
                    inputs=cvc_dialogue_csv,
                    outputs=[cvc_dialogue_df, cvc_summary, cvc_status],
                )

                cvc_sample_btn.click(
                    _export_sample_dialogue_csv,
                    outputs=[cvc_sample_file, cvc_status],
                )

                cvc_export_btn.click(
                    _export_dialogue_csv,
                    inputs=cvc_dialogue_df,
                    outputs=[cvc_export_file, cvc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
