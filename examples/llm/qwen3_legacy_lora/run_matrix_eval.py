# Copyright 2025 HOUMO AI
#
# File: run_matrix_eval.py
# Description:
#   Discover every exported xh2a LoRA variant (work_dirs/qwen3_xf_base-XH2a-*-lora-*/
#   meta.json) and run the customer docx eval on each via eval_customer_docx.XH2ABackend.
#   Also (re)runs the fp16 baseline B = reconstructed split model (qwen3_xf_base + lora
#   folded) as a sanity reference. Writes per-variant outputs under
#   work_dirs/customer_eval/<variant_tag>/ and a combined index.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
from pathlib import Path

EVAL = "examples/llm/qwen3_legacy_lora/eval_customer_docx.py"
PY = sys.executable
EVAL_ROOT = Path("work_dirs/customer_eval")


def discover_variants():
    """Return list of (tag, meta_json_path) for every completed export."""
    variants = []
    for meta in sorted(Path("work_dirs").glob("qwen3_xf_base-XH2a-16k-*-lora-*/meta.json")):
        d = meta.parent
        # require both onnx present (complete export)
        if not list(d.glob("hmonnx/prefill/*.onnx")):
            continue
        if not list(d.glob("hmonnx/decode/*.onnx")):
            continue
        tag = d.name.replace("qwen3_xf_base-XH2a-", "").replace("-lora", "")
        variants.append((tag, str(meta)))
    return variants


def run_one(tag, model_arg, backend, gpu, max_new_tokens, docx_dir, prompt_md,
            max_docs=0, only=None):
    """Launch one eval as a detached process pinned to a physical GPU. Returns Popen."""
    out_tag = tag if backend == "xh2a" else f"baseline_{tag}"
    logf = open(f"work_dirs/logs/matrix/eval_{out_tag}.log", "w")
    env_prefix = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    import os
    env = dict(os.environ, **env_prefix)
    cmd = [
        PY, EVAL,
        "--model", model_arg,
        "--backend", backend,
        "--out-tag", out_tag,
        "--device", "cuda:0",
        "--max-new-tokens", str(max_new_tokens),
        "--docx-dir", docx_dir,
        "--prompt-md", prompt_md,
    ]
    if max_docs and max_docs > 0:
        cmd += ["--max-docs", str(max_docs)]
    if only:
        cmd += ["--only", only]
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    print(f"[eval] launched {out_tag} on GPU {gpu} pid {p.pid}")
    return p, out_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--docx-dir", default="/data01/home/yujy/work/xunfei/xf_data")
    ap.add_argument("--prompt-md", default="/data01/home/yujy/work/xunfei/xf_data/prompt.md")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="tier-1: only eval first N docx per variant")
    ap.add_argument("--only", type=str, default=None,
                    help="only docx whose filename contains this substring")
    args = ap.parse_args()

    variants = discover_variants()
    print(f"[eval] discovered {len(variants)} completed export variants:")
    for tag, meta in variants:
        print(f"    {tag}  ({meta})")
    if args.list_only:
        return

    gpus = [g.strip() for g in args.gpus.split(",")]
    Path("work_dirs/logs/matrix").mkdir(parents=True, exist_ok=True)

    # simple GPU-pooled scheduler: at most len(gpus) concurrent evals
    pending = list(variants)
    running = []  # (Popen, tag, gpu)
    free_gpus = list(gpus)
    results = {}
    while pending or running:
        while pending and free_gpus:
            tag, meta = pending.pop(0)
            gpu = free_gpus.pop(0)
            p, out_tag = run_one(tag, meta, "xh2a", gpu, args.max_new_tokens,
                                 args.docx_dir, args.prompt_md,
                                 max_docs=args.max_docs, only=args.only)
            running.append((p, out_tag, gpu))
        # poll
        import time
        time.sleep(10)
        still = []
        for p, out_tag, gpu in running:
            if p.poll() is None:
                still.append((p, out_tag, gpu))
            else:
                rc = p.returncode
                results[out_tag] = rc
                free_gpus.append(gpu)
                print(f"[eval] {out_tag} exited rc={rc}")
        running = still

    print(f"[eval] all done: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
