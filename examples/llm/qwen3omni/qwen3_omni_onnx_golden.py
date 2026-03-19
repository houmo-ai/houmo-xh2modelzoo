# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni HMONNX golden output generation.

Loads all per-module HMONNX artifacts, replaces native model components
with HMONNXInference sessions, and runs full multi-modal generate to
produce golden text + audio outputs for on-device regression testing.
"""

import argparse
import json
import os.path as osp
import time
import sys
import types
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import apply_artifact_replacements, discover_artifacts, save_json, validate_golden_outputs
from xhquant.api import get_root_logger, set_random_seed, xhquant_init  # isort:skip
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference  # isort:skip

try:
    from qwen_omni_utils import process_mm_info
except ImportError:

    def process_mm_info(conversation, use_audio_in_video=False):
        audios, images, videos = [], [], []
        for turn in conversation:
            for item in turn.get("content", []):
                tp = item.get("type")
                if tp == "audio":
                    audios.append(item.get("audio"))
                elif tp == "image":
                    images.append(item.get("image"))
                elif tp == "video":
                    videos.append(item.get("video"))
        return audios, images, videos

# ────────────────────── helpers ──────────────────────


def _replace_code2wav(native_model, code2wav_hmonnx_path, static_code_len, logger):
    """Monkey-patch code2wav with HMONNX session."""
    session = HMONNXInference(str(code2wav_hmonnx_path))
    native_model.code2wav.hmonnx = session
    native_model.code2wav.hmonnx_max_code_len = static_code_len

    def forward(self, codes):
        max_code_len = int(self.hmonnx_max_code_len)
        code_len = int(codes.shape[-1])
        if code_len > max_code_len:
            raise ValueError(f"code2wav hmonnx max code len is {max_code_len}, but got {code_len}")
        hmonnx_input = codes.to(torch.int32)
        if code_len < max_code_len:
            hmonnx_input = torch.nn.functional.pad(hmonnx_input, (0, max_code_len - code_len))
        wav = self.hmonnx.forward(hmonnx_input)
        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        if not isinstance(wav, torch.Tensor):
            wav = torch.as_tensor(wav, device=codes.device)
        expected_wav_len = code_len * self.total_upsample
        return wav[..., :expected_wav_len]

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        max_code_len = int(self.hmonnx_max_code_len)
        safe_chunk = min(chunk_size, max(1, max_code_len - left_context_size))
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + safe_chunk, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            chunk_token_len = end_index - start_index
            if chunk_token_len + context_size > max_code_len:
                context_size = max(0, max_code_len - chunk_token_len)
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.forward(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    native_model.code2wav.forward = types.MethodType(forward, native_model.code2wav)
    native_model.code2wav.chunked_decode = types.MethodType(chunked_decode, native_model.code2wav)
    logger.info(f"code2wav replaced with HMONNX: {code2wav_hmonnx_path}")


def _replace_projection(module, hmonnx_path, name, logger):
    """Monkey-patch a Linear projection with HMONNX session."""
    session = HMONNXInference(str(hmonnx_path))
    module._hmonnx_session = session

    def forward(self, x):
        out = self._hmonnx_session.forward(x.cpu().to(torch.float16))
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out)
        return out.to(x.device).to(x.dtype)

    module.forward = types.MethodType(forward, module)
    logger.info(f"{name} replaced with HMONNX: {hmonnx_path}")


# ────────────────────── main ──────────────────────


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "golden.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    set_random_seed(args.seed)

    golden_dir = work_dir / "golden"
    golden_dir.mkdir(exist_ok=True, parents=True)

    # ---- 1. Load HF model ----
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    logger.info(f"Loading HF model from {hf_model_path}")
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)
    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype

    # ---- 2. Replace modules with HMONNX if artifacts provided ----
    applied_artifacts = {}
    if args.auto_discover:
        applied_artifacts = discover_artifacts(work_dir)
        if applied_artifacts:
            apply_artifact_replacements(native_model, applied_artifacts, logger)

    if args.code2wav_hmonnx:
        _replace_code2wav(native_model, args.code2wav_hmonnx, args.code2wav_static_code_len, logger)
        applied_artifacts["code2wav"] = {"path": args.code2wav_hmonnx}

    if args.hidden_projection_hmonnx:
        _replace_projection(native_model.talker.hidden_projection, args.hidden_projection_hmonnx, "hidden_projection", logger)
        applied_artifacts.setdefault("projection", {})["hidden_projection_hmonnx"] = args.hidden_projection_hmonnx

    if args.text_projection_hmonnx:
        _replace_projection(native_model.talker.text_projection, args.text_projection_hmonnx, "text_projection", logger)
        applied_artifacts.setdefault("projection", {})["text_projection_hmonnx"] = args.text_projection_hmonnx

    # ---- 3. Prepare inputs ----
    cases = {
        "text": [
            {"role": "user", "content": [{"type": "text", "text": "What is the capital of China? Answer in one sentence."}]},
        ],
        "vision": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(SCRIPT_DIR / "data" / "cars.jpg")},
                    {"type": "text", "text": "What do you see? Answer in one short sentence."},
                ],
            },
        ],
        "audio": [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(SCRIPT_DIR / "data" / "cough.wav")},
                    {"type": "text", "text": "What do you hear? Answer in one short sentence."},
                ],
            },
        ],
        "multimodal": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(SCRIPT_DIR / "data" / "cars.jpg")},
                    {"type": "audio", "audio": str(SCRIPT_DIR / "data" / "cough.wav")},
                    {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
                ],
            },
        ],
    }

    selected_cases = args.cases.split(",") if args.cases else list(cases.keys())
    golden_results = {}

    for case_name in selected_cases:
        if case_name not in cases:
            logger.warning(f"Unknown case: {case_name}, skipping")
            continue
        conversation = cases[case_name]
        logger.info(f"=== Running golden case: {case_name} ===")

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        inputs = inputs.to(device).to(dtype)

        with torch.no_grad():
            text_ids, audio = native_model.generate(
                **inputs, speaker="Ethan",
                thinker_return_dict_in_generate=True, use_audio_in_video=True,
                max_new_tokens=args.max_new_tokens,
            )

        decoded = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        logger.info(f"[{case_name}] Text: {decoded}")

        result = {
            "text": decoded,
            "input_ids_shape": list(inputs["input_ids"].shape),
        }

        if audio is not None:
            wav_path = golden_dir / f"golden_{case_name}.wav"
            sf.write(str(wav_path), audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
            result["audio_file"] = str(wav_path.relative_to(work_dir))
            logger.info(f"[{case_name}] Audio saved to {wav_path}")

        # Save input_ids and output_ids for regression
        torch.save(
            {
                "input_ids": inputs["input_ids"].cpu(),
                "output_ids": text_ids.sequences.cpu(),
            },
            golden_dir / f"golden_{case_name}_ids.pt",
        )
        golden_results[case_name] = result

    # ---- 4. Save golden meta ----
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model_path": hf_model_path,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "hmonnx_modules": sorted(list(applied_artifacts.keys())),
        "results": golden_results,
    }
    meta_file = golden_dir / "golden_meta.json"
    save_json(meta_file, meta)
    logger.info(f"Golden generation complete. Meta saved to {meta_file}")

    validation = validate_golden_outputs(golden_dir, list(golden_results.keys()))
    validation_file = golden_dir / "golden_validation.json"
    save_json(validation_file, validation)
    logger.info(f"Golden output validation passed. Report saved to {validation_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-Omni HMONNX golden output generation")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cases", type=str, default=None, help="comma-separated: text,vision,audio,multimodal")

    # HMONNX artifact paths (optional — if provided, replace native module)
    parser.add_argument("--code2wav-hmonnx", type=str, default=None, help="path to code2wav hmonnx file")
    parser.add_argument("--code2wav-static-code-len", type=int, default=126)
    parser.add_argument("--hidden-projection-hmonnx", type=str, default=None, help="path to hidden_projection hmonnx")
    parser.add_argument("--text-projection-hmonnx", type=str, default=None, help="path to text_projection hmonnx")
    parser.add_argument("--auto-discover", action="store_true", help="auto load exported qwen3omni artifacts under work-dir")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
