# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch
from transformers import TextStreamer
from xhquant.api import HMONNXGoldenInference, get_root_logger, xhquant_init
from xhquant.xhonnxruntime import config as xhonnxruntime_config

from xh_model_zoo.xh_llm.models.hy_mt2 import HyMT2HFCompatible, HyMT2Inference


PROMPT = "将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n今天天气真好。"


def _enable_golden(inference_engine: HyMT2Inference, golden_dir: str):
    golden_root = Path(golden_dir)
    golden_root.mkdir(exist_ok=True, parents=True)
    inference_engine.prefill_session = HMONNXGoldenInference(inference_engine.prefill_onnx_file)
    inference_engine.prefill_session.save_golden = True
    inference_engine.prefill_session.golden_dir = str(golden_root / "prefill")
    inference_engine.prefill_session.reset_step()
    inference_engine.prefill_session.exec_device = inference_engine.execution_device
    inference_engine.prefill_session.to(inference_engine.device)

    inference_engine.decode_session = HMONNXGoldenInference(inference_engine.decode_onnx_file)
    inference_engine.decode_session.save_golden = True
    inference_engine.decode_session.golden_dir = str(golden_root / "decode")
    inference_engine.decode_session.reset_step()
    inference_engine.decode_session.exec_device = inference_engine.execution_device
    inference_engine.decode_session.to(inference_engine.device)


def main(args):
    xhquant_init(None, args.debug)
    inference_engine = HyMT2Inference(
        args.config,
        fast_mode=args.fast,
        device=args.device,
        execution_device=args.execution_device,
    )
    if args.golden:
        _enable_golden(inference_engine, args.golden_dir)

    hf_model_path = inference_engine.meta_info.get("hf_model_path", None) or args.hf_model
    assert hf_model_path is not None and Path(hf_model_path).exists(), f"HF model path {hf_model_path} does not exist."

    logger = get_root_logger()
    device = inference_engine.device
    tokenizer = inference_engine.tokenizer
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    model_inputs = {"input_ids": inputs}

    wraped_hf_model = HyMT2HFCompatible.to_hf_compatible(hf_model_path, inference_engine)
    wraped_hf_model.eval()
    wraped_hf_model.to(device)
    streamer = TextStreamer(tokenizer) if args.stream else None
    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False
    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        streamer=streamer,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
    with torch.no_grad():
        generated_ids = wraped_hf_model.generate(
            **model_inputs,
            **generation_kwargs,
        )
    output_ids = generated_ids[0][inputs.shape[-1] :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    logger.info(f"response: {response}")
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="work_dirs/Hy-MT2-7B-XH2a-4k-w8a8/meta.json")
    parser.add_argument("--hf-model", type=str, default="/data01/datasets/Hy-MT2-7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execution_device", type=str, default="cuda:0", help="execution device")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--golden", action="store_true", help="export prefill/decode golden while generating")
    parser.add_argument("--golden-dir", type=str, default="work_dirs/Hy-MT2-7B-XH2a-4k-w8a8h1_sefp/golden")
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stream", action="store_true", help="stream generated text")
    parser.add_argument("--do-sample", action="store_true", help="enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    main(args)
