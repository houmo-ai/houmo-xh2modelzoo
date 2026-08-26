"""Native Transformers floating-point baseline for MiniCPM5."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native MiniCPM5 inference.")
    parser.add_argument("--model-dir", type=Path, required=True, help="MiniCPM5 HF model directory.")
    parser.add_argument("--prompt", default="请用中文简单介绍 MiniCPM5。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp16")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        dtype=_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    ).eval()
    model = model.to(args.device)
    assert type(model).__name__ == "MiniCPM5MoEForCausalLM", type(model).__name__

    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text(encoding="utf-8")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(args.device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    output_ids = generated_ids[0, model_inputs.input_ids.shape[1] :]
    content = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    print(content)


if __name__ == "__main__":
    main()
