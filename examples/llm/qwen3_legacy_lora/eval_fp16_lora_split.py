#!/usr/bin/env python3
"""Eval script #2/3: fp16 LoRA-split (new_base + GGUF LoRA, weight-patched).

Loads `weights/qwen3_xf_base` (SVD-decomposed new_base) as Qwen3ForCausalLM
in fp16, then reads `weights/qwen3_xf_lora.gguf` and folds the LoRA deltas
back into the base linear weights: W' = W_new_base + B @ A (alpha/rank=1.0
under decompose_lora_svd.py conventions, so no scale multiply needed).

Mathematically equivalent to running with LoRA at runtime; uses the LoRA
SPLIT representation (separate base dir + GGUF) as the source of truth.
This is what gets exported/quantized in the xh2a path, so verifying the
fp16 LoRA-split form matches the merged form is the round-trip check.

Generation params: temperature=0.3, top_p=0.9, top_k=20,
repetition_penalty=1.1, max_new_tokens=32768.

Outputs each case as a .docx under work_dirs/customer_eval/<out-tag>/.
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


def _fold_lora_into_base(model, gguf_path: str) -> int:
    """Fold GGUF LoRA deltas into base linear weights.

    Math: for each target linear, compute `delta = scale * (B @ A)` where
        A: (r, in)     <- lora_a
        B: (out, r)    <- lora_b
        scale = adapter.lora.alpha / rank     (= 1.0 under decompose_lora_svd convention)
    and add `delta` to the corresponding `.weight` parameter.
    After folding, model is mathematically identical to the original merged
    `qwen3_xf` (new_base + scale*B@A == W_xf, exact in fp32).

    Returns the number of layers folded.
    """
    import torch
    from xh_model_zoo.xh_llm.utils import load_gguf_checkpoint_for_lora

    # load_gguf_checkpoint_for_lora returns tensors in compute-shape (r,in)/(out,r)
    # (transformers' gguf loader handles the (in,r)/(r,out) physical <-> (r,in)/(out,r)
    # logical transposition for us — see xh_model_zoo/xh_llm/models/lora_layer.py:174).
    parsed = load_gguf_checkpoint_for_lora(
        gguf_path, return_tensors=True, model_to_load=model,
    )
    alpha = float(parsed.get("adapter.lora.alpha", 1.0))
    tensors = parsed["tensors"]

    # infer rank from the first lora_a we find
    first_a = next((k for k in tensors if k.endswith("weight_lora_a")), None)
    if first_a is None:
        raise RuntimeError(f"no weight_lora_a tensors found in {gguf_path}")
    rank = int(tensors[first_a].shape[0])
    scale = alpha / rank
    print(f"[lora-split] GGUF: {len(tensors)} tensors, alpha={alpha}, rank={rank}, "
          f"scale={scale:.4f}")

    n_folded = 0
    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if p.ndim != 2:
            continue
        a_key = name + "_lora_a"
        b_key = name + "_lora_b"
        if a_key not in tensors or b_key not in tensors:
            continue
        # A/B live on CPU (load_gguf_checkpoint_for_lora is CPU-only). Move
        # them to the parameter's device BEFORE the matmul so the whole
        # `B @ A` graph runs on GPU — avoids per-layer CPU<->GPU churn.
        dev = p.device
        A = tensors[a_key].to(device=dev, dtype=torch.float32)  # (r, in)
        B = tensors[b_key].to(device=dev, dtype=torch.float32)  # (out, r)
        # delta shape must match (out, in) of p
        delta = (scale * (B @ A)).to(p.dtype)
        assert delta.shape == p.shape, (
            f"shape mismatch for {name}: delta {tuple(delta.shape)} vs weight {tuple(p.shape)}"
        )
        with torch.no_grad():
            p.add_(delta)
        n_folded += 1
    return n_folded


class FP16LoRASplitBackend(BaseBackend):
    """Loads new_base HF dir, folds in the GGUF LoRA, runs HF generate()."""

    def __init__(self, base_dir: str, gguf_path: str,
                 device: str = "cuda:0", dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_dir)
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_dir, torch_dtype=torch_dtype,
        )
        self.model.to(device)
        self.model.eval()
        n_folded = _fold_lora_into_base(self.model, gguf_path)
        print(f"[lora-split] folded {n_folded} LoRA deltas from {gguf_path} into base weights")

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
    ap.add_argument("--base", default="weights/qwen3_xf_base",
                    help="HF dir of SVD-decomposed new_base. Default: weights/qwen3_xf_base")
    ap.add_argument("--gguf", default="weights/qwen3_xf_lora.gguf",
                    help="LoRA GGUF. Default: weights/qwen3_xf_lora.gguf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
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

    print(f"[eval-fp16-lora-split] base={args.base} gguf={args.gguf} dtype={args.dtype}")
    print(f"[eval-fp16-lora-split] {len(docx_files)} docx -> {out_dir}")
    print(f"[eval-fp16-lora-split] gen: T={GEN_TEMPERATURE} top_p={GEN_TOP_P} "
          f"top_k={GEN_TOP_K} rep_pen={GEN_REPETITION_PENALTY} "
          f"max_new={args.max_new_tokens} do_sample={GEN_DO_SAMPLE}")
    backend = FP16LoRASplitBackend(args.base, args.gguf,
                                    device=args.device, dtype=args.dtype)

    summary = []
    for f in docx_files:
        transcript = extract_docx_text(str(f))
        messages = build_messages(system_prompt, transcript)
        t0 = time.time()
        r: GenResult = backend.run(messages, args.max_new_tokens)
        meta = {
            "原始docx": f.name,
            "backend": "fp16_lora_split",
            "base": args.base,
            "gguf": args.gguf,
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
        title = f"庭审笔录精简 / qwen3_xf / fp16-lora-split / {f.stem}"
        write_docx_output(out_dir / f"{f.stem}.docx",
                          title=title, meta=meta, body_text=r.text)
        rec = {"docx": f.name, "in_tokens": r.in_tokens, "out_tokens": r.out_tokens,
               "seconds": r.seconds, "out_docx": f"{f.stem}.docx"}
        summary.append(rec)
        print(f"  [gen] {f.name}: in={r.in_tokens} out={r.out_tokens} {r.seconds:.1f}s")

    sp = write_summary(out_dir, summary)
    print(f"[eval-fp16-lora-split] done -> {sp}")


if __name__ == "__main__":
    main()
