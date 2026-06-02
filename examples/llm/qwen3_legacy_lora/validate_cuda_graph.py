#!/usr/bin/env python3
"""A/B validate CUDA graph for xh2a decode: correctness (token IDs identical)
+ speedup (per-token time). Runs the SAME variant twice on one GPU."""
import argparse, os, time
from pathlib import Path
import torch


def run(meta_json, device, n_tokens, use_graph):
    os.environ["XH2A_CUDA_GRAPH"] = "1" if use_graph else "0"
    from xhquant.api import xhquant_init
    from xh_model_zoo.xh_llm.models.qwen3_legacy_lora import (
        Qwen3LegacyLoRAHFCompatible, Qwen3LegacyLoRAInference,
    )
    xhquant_init(None, False)
    eng = Qwen3LegacyLoRAInference(meta_json, fast_mode=True,
                                   device=device, execution_device=device)
    eng.enable_lora = True
    hf = eng.meta_info.get("hf_model_path")
    tok = eng.tokenizer
    wrapped = Qwen3LegacyLoRAHFCompatible.to_hf_compatible(hf, eng)
    wrapped.eval(); wrapped.to(device)
    msgs = [{"role": "user", "content": "请用一句话介绍中华人民共和国的首都。"}]
    text = tok.apply_chat_template([msgs], tokenize=False, add_generation_prompt=True)
    mi = tok(text, return_tensors="pt").to(device)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out = wrapped.generate(**mi, max_new_tokens=n_tokens, do_sample=False,
                               pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    ids = out[0][len(mi.input_ids[0]):].tolist()
    return ids, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to meta.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=64)
    args = ap.parse_args()

    # baseline first (graph OFF), then graph ON — separate processes would be
    # cleaner but same-process is fine since each builds its own engine/session.
    print(f"[validate] {args.n_tokens} tokens, device={args.device}")
    print("[validate] === run A: CUDA graph OFF (baseline) ===")
    ids_off, dt_off = run(args.model, args.device, args.n_tokens, use_graph=False)
    print(f"[validate]   OFF: {len(ids_off)} tok in {dt_off:.1f}s "
          f"({len(ids_off)/dt_off:.2f} tok/s)")

    print("[validate] === run B: CUDA graph ON ===")
    ids_on, dt_on = run(args.model, args.device, args.n_tokens, use_graph=True)
    print(f"[validate]   ON:  {len(ids_on)} tok in {dt_on:.1f}s "
          f"({len(ids_on)/dt_on:.2f} tok/s)")

    same = ids_off == ids_on
    print("\n[validate] ============ RESULT ============")
    print(f"[validate] token IDs identical: {same}")
    if not same:
        n = min(len(ids_off), len(ids_on))
        first_div = next((i for i in range(n) if ids_off[i] != ids_on[i]), n)
        print(f"[validate]   len OFF={len(ids_off)} ON={len(ids_on)} "
              f"first divergence @ idx {first_div}")
    if dt_on > 0:
        print(f"[validate] speedup: {dt_off/dt_on:.2f}x "
              f"({dt_off:.1f}s -> {dt_on:.1f}s)")
    print("[validate] PASS" if same else "[validate] FAIL: outputs differ")


if __name__ == "__main__":
    main()
