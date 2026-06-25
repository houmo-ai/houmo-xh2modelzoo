# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni HMONNX streaming audio generation.

This path deliberately keeps HuggingFace ``model.generate()`` in charge of
Thinker/Talker sequencing and only taps the Talker residual codec IDs as they
are produced. The previous fully independent stage pipeline could produce a wav
file while feeding malformed residual-code frames into Code2Wav; this file is
the conservative validation path for audio quality.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os.path as osp
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, PARENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from qwen_omni_utils import process_mm_info
except ImportError:

    def process_mm_info(conversation, use_audio_in_video=False):
        audios, images, videos = [], [], []
        for turn in conversation:
            for item in turn.get("content", []):
                item_type = item.get("type")
                if item_type == "audio":
                    audios.append(item.get("audio"))
                elif item_type == "image":
                    images.append(item.get("image"))
                elif item_type == "video":
                    videos.append(item.get("video"))
        return audios, images, videos


class StreamingCode2WavDecoder:
    """Incrementally decode real Talker residual codec frames.

    ``add_residual_codes`` accepts one Talker step with shape
    ``[batch, num_quantizers]``. Internally codes are accumulated as
    ``[batch, num_quantizers, steps]`` and decoded with left-context overlap.
    Inputs with shape ``[1, 1]`` are rejected because they represent a single
    residual-code group, not a full codec frame.
    """

    def __init__(
        self,
        code2wav,
        chunk_size: int = 300,
        left_context_size: int = 25,
        num_quantizers: int = 16,
        on_audio_chunk: Optional[Callable[[torch.Tensor, int, int], None]] = None,
    ):
        self.code2wav = code2wav
        self.chunk_size = max(1, int(chunk_size))
        self.left_context_size = max(0, int(left_context_size))
        self.num_quantizers = int(num_quantizers)
        self.on_audio_chunk = on_audio_chunk
        self._codes: Optional[torch.Tensor] = None
        self._emitted_steps = 0
        self._chunk_index = 0
        self._audio_chunks: list[torch.Tensor] = []

    def add_residual_codes(self, residual_codes: torch.Tensor):
        residual_codes = self._normalize_residual_codes(residual_codes)
        self._codes = residual_codes if self._codes is None else torch.cat([self._codes, residual_codes], dim=-1)
        self._emit_ready_chunks(final=False)

    def finalize(self) -> torch.Tensor:
        self._emit_ready_chunks(final=True)
        if not self._audio_chunks:
            return torch.empty(1, 0, dtype=torch.float32)
        return torch.cat(self._audio_chunks, dim=-1).to(torch.float32)

    def _normalize_residual_codes(self, residual_codes: torch.Tensor) -> torch.Tensor:
        if not isinstance(residual_codes, torch.Tensor):
            residual_codes = torch.as_tensor(residual_codes)
        residual_codes = residual_codes.detach().to(torch.long)

        if residual_codes.dim() == 1:
            if residual_codes.numel() != self.num_quantizers:
                raise ValueError(
                    f"Expected {self.num_quantizers} residual code groups, got shape {tuple(residual_codes.shape)}"
                )
            residual_codes = residual_codes.view(1, self.num_quantizers, 1)
        elif residual_codes.dim() == 2:
            if int(residual_codes.shape[-1]) != self.num_quantizers:
                raise ValueError(
                    f"Expected residual code shape [B,{self.num_quantizers}], got {tuple(residual_codes.shape)}"
                )
            residual_codes = residual_codes.unsqueeze(-1)
        elif residual_codes.dim() == 3:
            if int(residual_codes.shape[1]) != self.num_quantizers:
                raise ValueError(
                    f"Expected residual code shape [B,{self.num_quantizers},T], got {tuple(residual_codes.shape)}"
                )
        else:
            raise ValueError(f"Expected residual codes with 1-3 dims, got shape {tuple(residual_codes.shape)}")

        return residual_codes

    def _emit_ready_chunks(self, final: bool):
        if self._codes is None:
            return
        total_steps = int(self._codes.shape[-1])
        while total_steps - self._emitted_steps >= self.chunk_size:
            self._emit_range(self._emitted_steps, self._emitted_steps + self.chunk_size)
        if final and total_steps > self._emitted_steps:
            self._emit_range(self._emitted_steps, total_steps)

    def _emit_range(self, start_step: int, end_step: int):
        assert self._codes is not None
        context_size = self.left_context_size if start_step - self.left_context_size > 0 else start_step
        codes_chunk = self._codes[..., start_step - context_size : end_step]
        wav_chunk = self._decode_codes(codes_chunk)
        if context_size:
            wav_chunk = wav_chunk[..., context_size * int(self.code2wav.total_upsample) :]
        wav_chunk = wav_chunk.reshape(1, -1).to(torch.float32)
        self._audio_chunks.append(wav_chunk)
        if self.on_audio_chunk is not None:
            self.on_audio_chunk(wav_chunk, self._chunk_index, end_step)
        self._chunk_index += 1
        self._emitted_steps = end_step

    def _decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        wav = self.code2wav.forward(codes)
        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav)
        return wav


def _extract_residual_codes(outputs) -> Optional[torch.Tensor]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if isinstance(hidden_states, (list, tuple)) and hidden_states:
        residual_codes = hidden_states[-1]
    elif isinstance(hidden_states, torch.Tensor):
        residual_codes = hidden_states
    else:
        residual_codes = None
    return residual_codes if isinstance(residual_codes, torch.Tensor) else None


@contextlib.contextmanager
def patch_talker_step_callback(talker, on_residual_codes: Callable[[torch.Tensor], None]):
    """Patch Talker generation to stream official residual codec IDs."""
    original = talker._update_model_kwargs_for_generation

    def patched(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        residual_codes = _extract_residual_codes(outputs)
        if residual_codes is not None:
            on_residual_codes(residual_codes)
        return original(outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens)

    talker._update_model_kwargs_for_generation = patched.__get__(talker, type(talker))
    try:
        yield
    finally:
        talker._update_model_kwargs_for_generation = original


def generate_stream(model, stream_decoder: Optional[StreamingCode2WavDecoder] = None, **generate_kwargs) -> Iterator[dict]:
    """Run Qwen3-Omni generation and yield audio chunks while generation runs."""
    if stream_decoder is None:
        stream_decoder = StreamingCode2WavDecoder(model.code2wav)

    event_queue: queue.Queue[dict] = queue.Queue()
    previous_callback = stream_decoder.on_audio_chunk

    def on_audio_chunk(audio_chunk: torch.Tensor, chunk_index: int, code_steps: int):
        event_queue.put(
            {
                "type": "audio_chunk",
                "audio": audio_chunk,
                "chunk_index": chunk_index,
                "code_steps": code_steps,
                "sample_rate": 24000,
            }
        )
        if previous_callback is not None:
            previous_callback(audio_chunk, chunk_index, code_steps)

    def generation_worker():
        stream_decoder.on_audio_chunk = on_audio_chunk
        try:
            with torch.no_grad(), patch_talker_step_callback(model.talker, stream_decoder.add_residual_codes):
                text_ids, final_audio = model.generate(**generate_kwargs)
            streamed_audio = stream_decoder.finalize()
            if streamed_audio.numel() == 0 and isinstance(final_audio, torch.Tensor):
                streamed_audio = final_audio.reshape(1, -1).to(torch.float32)
            event_queue.put(
                {"type": "complete", "text_ids": text_ids, "audio": streamed_audio, "sample_rate": 24000}
            )
        except BaseException as exc:
            event_queue.put({"type": "error", "error": exc})
        finally:
            stream_decoder.on_audio_chunk = previous_callback

    worker = threading.Thread(target=generation_worker, name="qwen3-omni-stream-generate", daemon=True)
    worker.start()

    while True:
        event = event_queue.get()
        if event.get("type") == "error":
            worker.join(timeout=1)
            raise event["error"]
        yield event
        if event.get("type") == "complete":
            worker.join(timeout=1)
            break


def build_cases():
    data_dir = PARENT_DIR / "data"
    return {
        "text": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of China? Answer in one sentence."}],
            },
        ],
        "vision": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(data_dir / "cars.jpg")},
                    {"type": "text", "text": "What do you see? Answer in one short sentence."},
                ],
            },
        ],
        "audio": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(data_dir / "cough.wav")},
                    {"type": "text", "text": "What do you hear? Answer in one short sentence."},
                ],
            },
        ],
        "multimodal": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(data_dir / "cars.jpg")},
                    {"type": "audio", "audio": str(data_dir / "cough.wav")},
                    {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
                ],
            },
        ],
    }


def _apply_requested_replacements(native_model, artifacts: dict, args, logger) -> set[str]:
    applied_module_names: set[str] = set()
    replacement_mode = getattr(args, "replacement_mode", "all")
    if not artifacts or replacement_mode == "none":
        return applied_module_names

    if replacement_mode in {"all", "full_hmonnx"}:
        from _hmonnx_pipeline import apply_artifact_replacements
        from qwen3_omni_validate_text_hmonnx_replacement import _build_text_hmonnx_generate_patch

        missing = [key for key in ("text", "talker", "talker_prediction", "code2wav") if key not in artifacts]
        if missing:
            raise RuntimeError(
                f"--replacement-mode {replacement_mode} requires discovered artifacts: "
                f"text, talker, talker_prediction, code2wav; missing {missing}"
            )

        apply_artifact_replacements(native_model, artifacts, logger)
        native_accept_hidden_layer = getattr(getattr(native_model.config, "talker_config", None), "accept_hidden_layer", None)
        native_model.thinker.generate = _build_text_hmonnx_generate_patch(
            native_model.thinker,
            artifacts["text"],
            logger,
            accept_hidden_layer=native_accept_hidden_layer,
        )
        applied_module_names.add("text")
        for key in ("audio", "vision", "talker", "talker_prediction", "code2wav"):
            if key in artifacts:
                applied_module_names.add(key)
        return applied_module_names

    if replacement_mode == "code2wav":
        from _hmonnx_pipeline import _replace_code2wav, _resolve_meta_path

        code2wav_meta = artifacts.get("code2wav")
        if code2wav_meta is None:
            raise RuntimeError("--replacement-mode code2wav requires a discovered code2wav artifact")
        _replace_code2wav(
            native_model,
            _resolve_meta_path(code2wav_meta, "code2wav_hmonnx"),
            int(code2wav_meta["static_code_len"]),
            logger,
        )
        return {"code2wav"}

    if replacement_mode == "talker_code2wav":
        from _hmonnx_pipeline import _patch_talker_prediction_shadow, _patch_talker_shadow, _replace_code2wav, _resolve_meta_path

        missing = [key for key in ("talker", "talker_prediction", "code2wav") if key not in artifacts]
        if missing:
            raise RuntimeError(
                "--replacement-mode talker_code2wav requires discovered artifacts: "
                f"talker, talker_prediction, code2wav; missing {missing}"
            )
        _patch_talker_shadow(native_model, artifacts["talker"], logger)
        _patch_talker_prediction_shadow(native_model, artifacts["talker_prediction"], logger)
        _replace_code2wav(
            native_model,
            _resolve_meta_path(artifacts["code2wav"], "code2wav_hmonnx"),
            int(artifacts["code2wav"]["static_code_len"]),
            logger,
        )
        return {"talker", "talker_prediction", "code2wav"}

    raise ValueError(f"Unknown replacement mode: {replacement_mode}")


def main(args):
    from _hmonnx_pipeline import (
        _build_safe_validation_max_memory,
        _ensure_hm_pixel_values,
        _patch_inputs_embeds_generation_device,
        _patch_runtime_device_property,
        _resolve_validation_device_map,
        discover_artifacts,
        save_json,
    )
    from transformers import Qwen3OmniMoeForConditionalGeneration
    from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
    from xhquant.api import get_root_logger, set_random_seed, xhquant_init

    hf_model_path = osp.normpath(osp.abspath(args.model))
    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    stream_dir = work_dir / "stream"
    stream_dir.mkdir(exist_ok=True, parents=True)

    xhquant_init(work_dir / "stream.log", debug=args.debug)
    logger = get_root_logger()
    set_random_seed(args.seed)

    resolved_device_map = _resolve_validation_device_map(args.device_map, logger)
    max_memory = _build_safe_validation_max_memory(logger) if resolved_device_map == "auto" else None
    logger.info(f"Loading HF model from {hf_model_path} for streaming generation")
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map=resolved_device_map,
        max_memory=max_memory,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()

    processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)
    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype

    artifacts = discover_artifacts(work_dir) if args.auto_discover else {}
    applied_module_names = _apply_requested_replacements(native_model, artifacts, args, logger)
    if args.code2wav_hmonnx:
        from _hmonnx_pipeline import _replace_code2wav

        _replace_code2wav(native_model, Path(args.code2wav_hmonnx), args.code2wav_static_code_len, logger)
        applied_module_names.add("code2wav")

    _patch_inputs_embeds_generation_device(native_model.talker, "talker", logger)
    _patch_inputs_embeds_generation_device(native_model.talker.code_predictor, "talker.code_predictor", logger)
    if hasattr(native_model, "code2wav"):
        _patch_runtime_device_property(native_model.code2wav, "code2wav", logger)

    cases = build_cases()
    selected_cases = args.cases.split(",") if args.cases else list(cases.keys())
    results = {}
    events_jsonl = stream_dir / "stream_events.jsonl"
    with open(events_jsonl, "w", encoding="utf-8") as event_file:
        for case_name in selected_cases:
            case_name = case_name.strip()
            if case_name not in cases:
                logger.warning(f"Unknown case: {case_name}, skipping")
                continue

            conversation = cases[case_name]
            use_audio_in_video = True
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                seconds_per_chunk=2.0,
                position_id_per_seconds=13,
                use_audio_in_video=use_audio_in_video,
            )
            inputs = _ensure_hm_pixel_values(inputs)
            inputs = inputs.to(device).to(dtype)
            inputs.pop("hm_pixel_values", None)
            inputs.pop("hm_pixel_values_videos", None)

            decoder = StreamingCode2WavDecoder(
                native_model.code2wav,
                chunk_size=args.stream_chunk_size,
                left_context_size=args.stream_left_context_size,
                num_quantizers=getattr(args, "num_quantizers", 16),
            )
            chunk_paths = []
            complete_event = None
            for event in generate_stream(
                native_model,
                stream_decoder=decoder,
                **inputs,
                speaker=args.speaker,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=args.max_new_tokens,
                talker_max_new_tokens=getattr(args, "talker_max_new_tokens", None),
            ):
                serializable = {
                    k: v
                    for k, v in event.items()
                    if k not in ("audio", "text_ids") and not isinstance(v, torch.Tensor)
                }
                serializable["case"] = case_name
                if event["type"] == "audio_chunk":
                    chunk_path = stream_dir / f"{case_name}_chunk_{event['chunk_index']:04d}.wav"
                    sf.write(str(chunk_path), event["audio"].reshape(-1).detach().cpu().numpy(), samplerate=24000)
                    serializable["audio_file"] = chunk_path.name
                    chunk_paths.append(chunk_path.name)
                    logger.info(
                        f"[{case_name}] audio_chunk index={event['chunk_index']} "
                        f"code_steps={event['code_steps']} samples={event['audio'].numel()}"
                    )
                elif event["type"] == "complete":
                    complete_event = event
                event_file.write(json.dumps(serializable, ensure_ascii=False) + "\n")

            if complete_event is None:
                raise RuntimeError(f"Streaming generation for case {case_name} did not produce a complete event")

            sequences = (
                complete_event["text_ids"].sequences
                if hasattr(complete_event["text_ids"], "sequences")
                else complete_event["text_ids"]
            )
            decoded = processor.batch_decode(
                sequences[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            full_audio = complete_event["audio"]
            full_wav_path = stream_dir / f"{case_name}_stream_output.wav"
            if isinstance(full_audio, torch.Tensor) and full_audio.numel() > 0:
                sf.write(str(full_wav_path), full_audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
            results[case_name] = {
                "text": decoded,
                "audio_file": full_wav_path.name if full_wav_path.exists() else None,
                "audio_chunks": chunk_paths,
            }
            logger.info(f"[{case_name}] Text: {decoded}")

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model_path": hf_model_path,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "talker_max_new_tokens": getattr(args, "talker_max_new_tokens", None),
        "stream_chunk_size": args.stream_chunk_size,
        "stream_left_context_size": args.stream_left_context_size,
        "hmonnx_modules": sorted(applied_module_names),
        "discovered_hmonnx_artifacts": sorted(artifacts.keys()),
        "events_jsonl": str(events_jsonl.relative_to(work_dir)),
        "results": results,
    }
    save_json(stream_dir / "stream_meta.json", meta)
    logger.info(f"Streaming generation complete. Meta saved to {stream_dir / 'stream_meta.json'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-Omni HMONNX streaming audio generation")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--talker-max-new-tokens", type=int, default=None)
    parser.add_argument("--cases", type=str, default=None, help="comma-separated: text,vision,audio,multimodal")
    parser.add_argument("--speaker", type=str, default="Ethan")
    parser.add_argument("--stream-chunk-size", type=int, default=50)
    parser.add_argument("--stream-left-context-size", type=int, default=25)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument("--code2wav-hmonnx", type=str, default=None, help="path to code2wav hmonnx file")
    parser.add_argument("--code2wav-static-code-len", type=int, default=126)
    parser.add_argument("--auto-discover", action="store_true", help="auto load exported qwen3omni artifacts under work-dir")
    parser.add_argument(
        "--replacement-mode",
        type=str,
        default="all",
        choices=["all", "full_hmonnx", "code2wav", "talker_code2wav", "none"],
        help="which discovered artifacts to apply before generation",
    )
    parser.add_argument("--device-map", type=str, default="cuda:0", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
