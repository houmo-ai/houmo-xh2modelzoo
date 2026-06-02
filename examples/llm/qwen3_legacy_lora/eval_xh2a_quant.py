#!/usr/bin/env python3
"""Eval script #3/3: xh2a quantized (HMONNX meta.json export).

Loads a meta.json produced by `qwen3_legacy_lora_xh2a_export_hmonnx.py`,
wraps the HMONNX engine into a HF-compatible module via
Qwen3LegacyLoRAHFCompatible, and runs inference through .generate().

Generation params: temperature=0.3, top_p=0.9, top_k=20,
repetition_penalty=1.1, max_new_tokens=32768.

Outputs each case as a .docx under work_dirs/customer_eval/<out-tag>/.

NOTE on speed: xh2a bit-accurate simulation runs ~2-3 tok/s on a single
A100 (see project memory). 5 docx × up to 32768 tokens is many hours; for
sanity use --max-new-tokens 256 to smoke-test the wiring first, then bump.
"""
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import argparse
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


class XH2AQuantBackend(BaseBackend):
    """Quantized HMONNX backend driven by a meta.json export.

    Mirrors examples/llm/qwen3_legacy_lora/qwen3_legacy_xh2a_hmonnx_test.py
    and the eval_customer_docx.XH2ABackend class.
    """

    def __init__(self, meta_json: str, device: str = "cuda:0", fast_mode: bool = True):
        import torch
        from xhquant.api import xhquant_init
        from xhquant.xhonnxruntime import config as xhonnxruntime_config
        from xh_model_zoo.xh_llm.models.qwen3_legacy_lora import (
            Qwen3LegacyLoRAHFCompatible, Qwen3LegacyLoRAInference,
        )

        self.torch = torch
        self.device = device
        xhquant_init(None, False)
        self.engine = Qwen3LegacyLoRAInference(
            meta_json, fast_mode=fast_mode,
            device=device, execution_device=device,
        )
        self.engine.enable_lora = True
        hf_model_path = self.engine.meta_info.get("hf_model_path", None)
        if hf_model_path is None or not Path(hf_model_path).exists():
            raise FileNotFoundError(
                f"hf_model_path from meta.json not found: {hf_model_path}. "
                "Pass a meta.json whose hf_config/hf_model_path resolve."
            )
        self.tokenizer = self.engine.tokenizer
        self.max_input_len = self.engine.prefill_input_sequence_length
        xhonnxruntime_config.disable_progress = True
        xhonnxruntime_config.verbose_progress = False
        self.wrapped = Qwen3LegacyLoRAHFCompatible.to_hf_compatible(
            hf_model_path, self.engine,
        )
        self.wrapped.eval()
        self.wrapped.to(device)

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
            out = self.wrapped.generate(
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
    ap.add_argument("--model", required=True,
                    help="path to meta.json (xh2a export)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-fast-mode", action="store_true",
                    help="disable xh2a fast_mode (slower but more debuggable)")
    ap.add_argument("--prompt-md", default=DEFAULT_PROMPT_MD)
    ap.add_argument("--docx-dir", default=DEFAULT_DOCX_DIR)
    ap.add_argument("--out-tag", required=True)
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

    print(f"[eval-xh2a] meta={args.model} fast_mode={not args.no_fast_mode}")
    print(f"[eval-xh2a] {len(docx_files)} docx -> {out_dir}")
    print(f"[eval-xh2a] gen: T={GEN_TEMPERATURE} top_p={GEN_TOP_P} "
          f"top_k={GEN_TOP_K} rep_pen={GEN_REPETITION_PENALTY} "
          f"max_new={args.max_new_tokens} do_sample={GEN_DO_SAMPLE}")
    backend = XH2AQuantBackend(
        args.model, device=args.device, fast_mode=not args.no_fast_mode,
    )

    summary = []
    for f in docx_files:
        transcript = extract_docx_text(str(f))
        messages = build_messages(system_prompt, transcript)
        t0 = time.time()
        r: GenResult = backend.run(messages, args.max_new_tokens)
        meta = {
            "原始docx": f.name,
            "backend": "xh2a_quant",
            "meta_json": args.model,
            "in_tokens": r.in_tokens,
            "out_tokens": r.out_tokens,
            "seconds": r.seconds,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "top_k": GEN_TOP_K,
            "repetition_penalty": GEN_REPETITION_PENALTY,
            "max_new_tokens": args.max_new_tokens,
        }
        title = f"庭审笔录精简 / qwen3_xf / xh2a-quant / {f.stem}"
        write_docx_output(out_dir / f"{f.stem}.docx",
                          title=title, meta=meta, body_text=r.text)
        rec = {"docx": f.name, "in_tokens": r.in_tokens, "out_tokens": r.out_tokens,
               "seconds": r.seconds, "out_docx": f"{f.stem}.docx"}
        summary.append(rec)
        print(f"  [gen] {f.name}: in={r.in_tokens} out={r.out_tokens} {r.seconds:.1f}s")

    sp = write_summary(out_dir, summary)
    print(f"[eval-xh2a] done -> {sp}")


if __name__ == "__main__":
    main()
