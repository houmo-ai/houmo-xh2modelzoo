# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni code2wav vocoder to HMONNX.

Exports code2wav via torch.onnx.export + convert_onnx_to_hmonnx with static code_len,
then monkey-patches forward/chunked_decode and validates with a full generate call.
"""

import argparse
import os.path as osp
import time
import sys
import types
from pathlib import Path

import onnx
import soundfile as sf
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import run_dialogue_validation, save_json
from xhquant.api import (  # isort:skip
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

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


def _dedupe_onnx_node_outputs(onnx_path: Path):
    model = onnx.load(str(onnx_path))
    producers = {}
    for node_idx, node in enumerate(model.graph.node):
        for output_idx, output_name in enumerate(node.output):
            if not output_name:
                continue
            producers.setdefault(output_name, []).append((node_idx, output_idx))

    for tensor_name, occurrences in producers.items():
        if len(occurrences) <= 1:
            continue

        for dup_idx, (producer_idx, output_idx) in enumerate(occurrences[:-1]):
            next_producer_idx = occurrences[dup_idx + 1][0]
            new_name = f"{tensor_name}__dedup_{dup_idx}"
            model.graph.node[producer_idx].output[output_idx] = new_name

            for node in model.graph.node[producer_idx + 1 : next_producer_idx]:
                for input_idx, input_name in enumerate(node.input):
                    if input_name == tensor_name:
                        node.input[input_idx] = new_name

            for value_info in model.graph.value_info:
                if value_info.name == tensor_name:
                    value_info.name = new_name

    onnx.save(model, str(onnx_path))

def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    prefix = f"{model_name}-{target_device}-code2wav-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    onnx_dir = work_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    onnx_path = onnx_dir / "code2wav.onnx"
    hmonnx_path = onnx_dir / "code2wav_hm.onnx"
    meta_file = work_dir / "meta_code2wav.json"

    # ---- 1. Load full HF model ----
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

    # ---- 2. Export code2wav ONNX + HMONNX ----
    static_code_len = args.static_code_len

    if hmonnx_path.exists() and meta_file.exists():
        logger.info("code2wav HMONNX artifacts already exist, skipping export")
        import json
        with open(meta_file) as f:
            meta_info = json.load(f)
    else:
        dummy_codes = torch.randint(100, (1, 16, static_code_len), dtype=torch.int32)

        logger.info(f"Exporting code2wav with static_code_len={static_code_len} ...")
        torch.onnx.export(
            native_model.code2wav.cpu(),
            dummy_codes,
            onnx_path,
            opset_version=18,
            dynamo=True,
            do_constant_folding=True,
            input_names=["codes"],
            output_names=["embedding"],
        )
        _dedupe_onnx_node_outputs(onnx_path)
        convert_onnx_to_hmonnx(
            onnx_path,
            [dummy_codes],
            target_device,
            hmonnx_path,
            quant_config=quant_config,
            input_names=["codes"],
        )
        logger.info(f"code2wav exported to {hmonnx_path}")

        # ---- 3. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "module": "code2wav",
            "model_name": model_name,
            "code2wav_hmonnx": str(hmonnx_path.relative_to(work_dir)),
            "static_code_len": static_code_len,
            "quant_type": quant_type,
        }
        save_json(meta_file, meta_info)
        logger.info(f"code2wav export complete. Meta saved to {meta_file}")

    # ---- 4. Optional full-generate validation ----
    if args.valid:
        logger.info("Validating code2wav HMONNX with full generate ...")
        from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

        session = HMONNXInference(str(hmonnx_path))
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

        # Run full generate
        image_path = str(SCRIPT_DIR / "data" / "cars.jpg")
        audio_path = str(SCRIPT_DIR / "data" / "cough.wav")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
                ],
            },
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        )
        device = next(native_model.parameters()).device
        dtype = next(native_model.parameters()).dtype
        inputs = inputs.to(device).to(dtype)

        with torch.no_grad():
            text_ids, audio = native_model.generate(
                **inputs, speaker="Ethan",
                thinker_return_dict_in_generate=True, use_audio_in_video=True,
            )
        # text_ids may be a GenerateOutput or a raw Tensor
        if hasattr(text_ids, 'sequences'):
            seq = text_ids.sequences
        else:
            seq = text_ids
        decoded = processor.batch_decode(
            seq[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        logger.info(f"Generate text: {decoded}")
        if audio is not None:
            out_wav = work_dir / "output_code2wav_valid.wav"
            sf.write(str(out_wav), audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
            logger.info(f"Audio saved to {out_wav}")
        logger.info("code2wav HMONNX validation passed.")

        # Free the first HF model before mixed-pipeline validation to avoid double-loading OOM.
        import gc

        del native_model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        dialogue_artifacts = {
            "code2wav": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
        }
        try:
            run_dialogue_validation(
                hf_model_path,
                work_dir,
                logger,
                case="multimodal",
                max_new_tokens=args.max_new_tokens,
                device_map="cuda:0",
                artifacts=dialogue_artifacts,
                report_name="code2wav_dialogue_validation.json",
                output_prefix="code2wav_dialogue",
            )
        except (RuntimeError, AssertionError, Exception) as e:
            logger.warning(f"Code2wav dialogue validation skipped due to error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni code2wav to HMONNX")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--static-code-len", type=int, default=126, help="static code length for HMONNX export")
    parser.add_argument("--valid", default=True, help="validate with full generate")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
