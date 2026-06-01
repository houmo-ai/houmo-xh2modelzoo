# Copyright 2025 HOUMO AI
#
# File: gpt_oss_with_mask_xh2a_hmonnx_golden.py
# Description:
#   Generate golden data by running GPT-OSS HMONNX prefill/decode sessions.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from pathlib import Path

import torch

from xhquant.api import HMONNXGoldenInference, get_root_logger, xhquant_init
from xhquant.xhonnxruntime import config as xhonnxruntime_config

from xh_model_zoo.xh_llm.models.gpt_oss_with_mask import GptOssWithMask_HFCompatible, GptOssWithMaskInference
from xh_model_zoo.xh_llm.utils import auto_offload


def build_model_inputs(tokenizer, prompt: str, device: torch.device):
    return tokenizer(prompt, return_tensors="pt").to(device)


def build_golden_sessions(inference_engine: GptOssWithMaskInference, golden_dir: Path):
    prefill_golden_dir = golden_dir / "prefill"
    decode_golden_dir = golden_dir / "decode"
    prefill_golden_dir.mkdir(exist_ok=True, parents=True)
    decode_golden_dir.mkdir(exist_ok=True, parents=True)

    prefill_session = HMONNXGoldenInference(str(inference_engine.prefill_onnx_file))
    decode_session = HMONNXGoldenInference(str(inference_engine.decode_onnx_file))

    prefill_session.save_golden = True
    decode_session.save_golden = True
    prefill_session.golden_dir = prefill_golden_dir
    decode_session.golden_dir = decode_golden_dir
    prefill_session.step = 0
    decode_session.step = 0

    if inference_engine.fast_mode:
        prefill_session.to_fast_mode()
        decode_session.to_fast_mode()

    prefill_session.exec_device = inference_engine.execution_device
    decode_session.exec_device = inference_engine.execution_device
    prefill_session.to(inference_engine.device)
    decode_session.to(inference_engine.device)

    inference_engine.prefill_session = prefill_session
    inference_engine.decode_session = decode_session


def main(args):
    model_config_file = Path(args.config)
    assert model_config_file.exists(), f"meta.json not found: {model_config_file}"

    default_golden_dir = model_config_file.parent / "golden"
    golden_dir = Path(args.golden_dir) if args.golden_dir is not None else default_golden_dir
    golden_dir.mkdir(exist_ok=True, parents=True)

    log_file = golden_dir / "golden.log"
    xhquant_init(str(log_file), args.debug)
    logger = get_root_logger()

    inference_engine = GptOssWithMaskInference(
        args.config,
        fast_mode=args.fast,
        device=args.device,
        execution_device=args.execution_device,
    )

    hf_model_path = inference_engine.meta_info.get("hf_model_path", None)
    if hf_model_path is None:
        hf_model_path = args.hf_model
    assert hf_model_path is not None, "hf model path is required"
    assert Path(hf_model_path).exists(), f"HF model path {hf_model_path} does not exist"

    tokenizer = inference_engine.get_tokenizer(hf_model_path)
    model_inputs = build_model_inputs(tokenizer, args.prompt, inference_engine.device)

    # Use golden-enabled sessions to dump prefill/decode golden inputs/outputs.
    build_golden_sessions(inference_engine, golden_dir)

    wrapped_hf_model = GptOssWithMask_HFCompatible.to_hf_compatible(hf_model_path, inference_engine)
    auto_offload(wrapped_hf_model, "XH2aQuantGptOssBlock")
    wrapped_hf_model.eval()  # type: ignore
    wrapped_hf_model.to(inference_engine.device)  # type: ignore

    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False

    with torch.no_grad():
        generated_ids = wrapped_hf_model.generate(  # type: ignore
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = int(model_inputs.input_ids.shape[1])
    output_ids = generated_ids[0][input_len:].detach().cpu()
    content = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True).strip("\n")

    ids_file = golden_dir / "golden_ids.pt"
    torch.save(
        {
            "input_ids": model_inputs.input_ids.detach().cpu(),
            "output_ids": generated_ids.detach().cpu(),
            "new_token_ids": output_ids,
        },
        ids_file,
    )

    summary = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "meta_config": str(model_config_file),
        "hf_model_path": str(hf_model_path),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "golden_dir": str(golden_dir),
        "prefill_golden_dir": str((golden_dir / "prefill")),
        "decode_golden_dir": str((golden_dir / "decode")),
        "ids_file": str(ids_file),
        "response_text": content,
    }
    summary_file = golden_dir / "golden_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Golden generation done: {golden_dir}")
    logger.info(f"Response: {content}")
    logger.info(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GPT-OSS golden from exported HMONNX")
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/gpt-oss-20b-XH2a-2k-w8a8h0_sefp/meta.json",
        help="path to exported meta.json",
    )
    parser.add_argument("--hf-model", type=str, default="data/datasets/gpt-oss-20b", help="fallback HF model path")
    parser.add_argument("--prompt", type=str, default="你是谁？", help="prompt used to generate golden")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="generation max new tokens")
    parser.add_argument("--golden-dir", type=str, default=None, help="output directory for golden files")
    parser.add_argument("--device", type=str, default="cuda", help="torch device")
    parser.add_argument("--execution-device", type=str, default="cuda", help="runtime execution device")
    parser.add_argument("--fast", action="store_true", help="enable fast mode")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()
    main(args)
