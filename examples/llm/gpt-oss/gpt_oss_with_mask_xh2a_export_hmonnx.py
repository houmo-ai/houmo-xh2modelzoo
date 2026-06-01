# Copyright 2025 HOUMO AI
#
# File: gpt_oss_with_mask_xh2a_export_hmonnx.py
# Description:
#   Example script: llm/gpt-oss/gpt_oss_with_mask_xh2a_export_hmonnx.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os.path as osp
from pathlib import Path

import torch

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.gpt_oss_with_mask import GptOssWithMaskConvertConfig, GptOssWithMaskInference

from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip


def _run_hmonnx_greedy_dialogue(
    inference_engine: GptOssWithMaskInference,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    """Run dialogue generation with pure HMONNX sessions (no HF generate wrapper)."""
    eos_token_id = tokenizer.eos_token_id
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids

    generated_ids = []
    past_seq_length = int(prompt_ids.shape[-1])

    # Prefill
    inference_engine.set_phase_prefill(True)
    prefill_data = {
        "input_ids": prompt_ids,
        "past_seq_length": 0,
    }
    prefill_inputs = inference_engine.prepare_inputs(prefill_data, inference_engine.get_input_sequence_length())
    prefill_logits = inference_engine.forward(*prefill_inputs)
    next_token_id = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    generated_ids.append(next_token_id)

    # Decode
    inference_engine.set_phase_prefill(False)
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        decode_data = {
            "input_ids": torch.tensor([[next_token_id]], dtype=torch.long),
            "past_seq_length": past_seq_length,
        }
        decode_inputs = inference_engine.prepare_inputs(decode_data, 1)
        decode_logits = inference_engine.forward(*decode_inputs)

        next_token_id = int(torch.argmax(decode_logits[:, -1, :], dim=-1).item())
        generated_ids.append(next_token_id)
        past_seq_length += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")


def validate_hmonnx_export(work_dir: Path, hf_model_path: str, logger):
    """Validate exported hmonnx by running a pure-HMONNX conversation test."""
    logger.info("=" * 60)
    logger.info("Starting HMONNX export validation...")
    logger.info("=" * 60)
    
    try:
        meta_file = work_dir / "meta.json"
        if not meta_file.exists():
            logger.warning(f"Meta file not found: {meta_file}")
            return False
        
        # Load inference engine
        inference_engine = GptOssWithMaskInference(str(meta_file), fast_mode=False)
        device = inference_engine.device
        logger.info(f"Device: {device}")
        
        # Load tokenizer
        tokenizer = inference_engine.get_tokenizer(hf_model_path)
        
        prompt = "你是谁？"
        logger.info(f"Input prompt: {prompt}")
        logger.info("Generating response with pure HMONNX sessions (max 32 tokens, non-fast mode)...")
        output_text = _run_hmonnx_greedy_dialogue(
            inference_engine=inference_engine,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=32,
        )
        
        logger.info(f"Generated output: {output_text}")
        
        # Check if output is reasonable
        if output_text and len(output_text) > 5 and "analysisNeed to answer" not in output_text:
            logger.info("✓ Validation passed! HMONNX export is working correctly.")
            return True
        else:
            logger.warning("✗ Validation failed! Generated output is too short or empty.")
            return False
            
    except Exception as e:
        logger.error(f"✗ Validation failed with exception: {e}", exc_info=True)
        return False


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    # quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
    config = GptOssWithMaskConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        sliding_window=args.sliding_window,
        num_logits_to_keep=args.num_logits_to_keep,
        rope_max_length=args.rope_max_length,
    )

    prefix = f"{model_name}-{target_device}-{args.context_length//1024}k-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "GptOssForCausalLM", config, str(work_dir))
    
    # Validate the exported HMONNX
    if not args.skip_validation:
        logger.info("")
        validate_hmonnx_export(work_dir, hf_model_path, logger)
        logger.info("")
    else:
        logger.info("Skipping HMONNX validation as requested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", default="data/datasets/gpt-oss-20b", type=str, help="HuggingFace model path")
    parser.add_argument("--context-length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--quant-type", default="w8a8h0_sefp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=128,
        help="sliding window size for attention, if None, use global attention",
    )
    parser.add_argument("--num_logits_to_keep", type=int, default=1, help="not for test ppl")
    parser.add_argument(
        "--rope-max-length",
        type=int,
        default=None,
        help="max rope position length, e.g. 262144 for 256k. If None, auto-detect from HF config",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip HMONNX validation after export",
    )
    args = parser.parse_args()
    main(args)

