#!/usr/bin/env python3
"""Eval script #1/3: fp16 non-LoRA-split (merged W_xf, no SVD).

Loads `weights/qwen3_xf` as a vanilla Qwen3ForCausalLM in fp16. The merged
xfinetune has LoRA folded into base weights, so the model is functionally
identical to the original W_xf — there is no separate LoRA tensor at
inference time. This is the "no LoRA split" reference.

Generation params: temperature=0.3, top_p=0.9, top_k=20,
repetition_penalty=1.1, max_new_tokens=32768.

Outputs each case as a .docx under work_dirs/customer_eval/<out-tag>/.
"""
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import time
from pathlib import Path

# shared
from _eval_common import (
    BaseBackend, GenResult, GEN_TEMPERATURE, GEN_TOP_P, GEN_TOP_K,
    GEN_REPETITION_PENALTY, GEN_MAX_NEW_TOKENS, GEN_DO_SAMPLE,
    DEFAULT_PROMPT_MD, DEFAULT_DOCX_DIR, DEFAULT_OUT_ROOT,
    load_system_prompt, extract_docx_text, build_messages,
    write_docx_output, discover_docx, write_summary,
)


class FP16NoLoRASplitBackend(BaseBackend):
    """Pure fp16 HF backend. No LoRA at all — model is the merged W_xf."""

    def __init__(self, model_dir: str, device: str = "cuda:0", dtype="float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
        )
        self.model.to(device)
        self.model.eval()

    def build_prompt_text(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def count_tokens(self, prompt_text):
        return len(self.tokenizer(prompt_text).input_ids)

    def generate(self, prompt_text, max_new_tokens):
        torch = self.torch
        mi = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **mi,
                max_new_tokens=max_new_tokens,
                do_sample=GEN_DO_SAMPLE,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                top_k=GEN_TOP_K,
                repetition_penalty=GEN_REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][len(mi.input_ids[0]):].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip("\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen3_xf",
                    help="HF dir of merged xf (LoRA folded in). Default: weights/qwen3_xf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--prompt-md", default=DEFAULT_PROMPT_MD)
    ap.add_argument("--docx-dir", default=DEFAULT_DOCX_DIR)
    ap.add_argument("--out-tag", required=True,
                    help="subdir under work_dirs/customer_eval for outputs")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--only", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    system_prompt = load_system_prompt(args.prompt_md)
    docx_files = discover_docx(args.docx_dir, only=args.only, max_docs=args.max_docs)
    out_dir = Path(DEFAULT_OUT_ROOT) / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval-fp16-nolora] model={args.model} dtype={args.dtype}")
    print(f"[eval-fp16-nolora] {len(docx_files)} docx -> {out_dir}")
    print(f"[eval-fp16-nolora] gen: T={GEN_TEMPERATURE} top_p={GEN_TOP_P} "
          f"top_k={GEN_TOP_K} rep_pen={GEN_REPETITION_PENALTY} "
          f"max_new={args.max_new_tokens} do_sample={GEN_DO_SAMPLE}")
    backend = FP16NoLoRASplitBackend(args.model, device=args.device, dtype=args.dtype)

    summary = []
    for f in docx_files:
        transcript = extract_docx_text(str(f))
        messages = build_messages(system_prompt, transcript)
        t0 = time.time()
        r: GenResult = backend.run(messages, args.max_new_tokens)
        meta = {
            "原始docx": f.name,
            "backend": "fp16_nolora_split",
            "model": args.model,
            "dtype": args.dtype,
            "in_tokens": r.in_tokens,
            "out_tokens": r.out_tokens,
            "seconds": r.seconds,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "top_k": GEN_TOP_K,
            "repetition_penalty": GEN_REPETITION_PENALTY,
            "max_new_tokens": args.max_new_tokens,
        }
        title = f"庭审笔录精简 / qwen3_xf / fp16-nolora-split / {f.stem}"
        write_docx_output(out_dir / f"{f.stem}.docx",
                          title=title, meta=meta, body_text=r.text)
        rec = {"docx": f.name, "in_tokens": r.in_tokens, "out_tokens": r.out_tokens,
               "seconds": r.seconds, "out_docx": f"{f.stem}.docx"}
        summary.append(rec)
        print(f"  [gen] {f.name}: in={r.in_tokens} out={r.out_tokens} {r.seconds:.1f}s")

    sp = write_summary(out_dir, summary)
    print(f"[eval-fp16-nolora] done -> {sp}")


if __name__ == "__main__":
    main()
