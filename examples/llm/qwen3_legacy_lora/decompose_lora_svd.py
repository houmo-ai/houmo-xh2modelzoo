# Copyright 2025 HOUMO AI
#
# File: decompose_lora_svd.py
# Description:
#   Reconstruct a separable LoRA adapter (rank-r) from a MERGED finetuned model.
#   Given original base (Qwen3-8B) and merged finetune (qwen3_xf) that share an
#   identical weight-key set, for every all-linear target:
#       delta   = W_xf - W_base                       (fp32)
#       delta  ~= U_r S_r Vh_r        (truncated SVD, rank r)
#       A       = sqrt(S_r) @ Vh_r    -> (r, in)
#       B       = U_r @ sqrt(S_r)     -> (out, r)
#       new_base= W_xf - (B @ A)      (residual absorbed -> exact in fp32)
#   alpha is fixed to r so the downstream scale (alpha/r) == 1.0 and
#       new_base + scale*(B@A) == W_xf   exactly (fp32).
#
#   GGUF on-disk convention (Pinned 2026-06-02 by P9 review):
#     - lora_a tensor: stored physically as (in, r), data layout is (r, in)
#     - lora_b tensor: stored physically as (r, out), data layout is (out, r)
#     - readers (e.g. transformers' load_gguf_checkpoint_for_lora) trans-
#       pose back to (r, in)/(out, r) for direct use; raw gguf.GGUFReader
#       exposes the (in, r)/(r, out) physical shape, which can confuse.
#     - downstream applications that take lora_a/lora_b straight from a
#       module's `weight_lora_a`/`weight_lora_b` buffers (e.g.
#       xh_model_zoo/xh_llm/models/lora_layer.py:174) auto-detect the
#       mismatch and transpose both, so the math is always
#       `W' = W + scale * (B @ A)` regardless of which path you take.
#
#   Stores new_base as a HF dir (fp16) + LoRA as GGUF read by
#   xh_model_zoo.xh_llm.utils.load_gguf_checkpoint_for_lora.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

TARGETS = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def gguf_base_name(hf_weight_key: str):
    """model.layers.N.self_attn.q_proj.weight -> blk.N.attn_q.weight (or None)."""
    if not hf_weight_key.endswith(".weight"):
        return None
    body = hf_weight_key[: -len(".weight")]
    parts = body.split(".")
    if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
        return None
    layer = parts[2]
    sub = ".".join(parts[3:])
    if sub not in TARGETS:
        return None
    return f"blk.{layer}.{TARGETS[sub]}.weight"


def load_full_state_dict(model_dir: str):
    """Load all safetensors shards of an HF dir into a single fp32 cpu dict."""
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [p.name for p in model_dir.glob("*.safetensors")]
    sd = {}
    for shard in shards:
        part = load_file(str(model_dir / shard))
        for k, v in part.items():
            sd[k] = v
        del part
    return sd


def decompose(args):
    rank = args.rank
    base_dir = Path(args.base)
    xf_dir = Path(args.merged)
    out_base = Path(args.out_base)
    out_gguf = Path(args.out_gguf)

    print(f"[load] base   = {base_dir}")
    base_sd = load_full_state_dict(str(base_dir))
    print(f"[load] merged = {xf_dir}")
    xf_sd = load_full_state_dict(str(xf_dir))

    assert set(base_sd) == set(xf_sd), "weight key sets differ between base and merged"

    # alpha == rank  =>  downstream scale (alpha/r) == 1.0  =>  exact reconstruction
    alpha = float(rank)
    lora_tensors = {}      # gguf_name(.lora_a/.lora_b) -> np.float16
    new_base_sd = {}       # hf_key -> fp16 tensor (new base)
    max_recon_residual = 0.0
    n_lora = 0

    for key in xf_sd:
        gname = gguf_base_name(key)
        w_xf = xf_sd[key].to(torch.float32)
        if gname is None:
            # not a LoRA target: absorb any delta directly into new base
            new_base_sd[key] = xf_sd[key].clone()
            continue

        w_base = base_sd[key].to(torch.float32)
        delta = w_xf - w_base  # (out, in)
        # truncated SVD
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(rank, S.shape[0])
        Us, Ss, Vhs = U[:, :r], S[:r], Vh[:r, :]
        sqrtS = torch.sqrt(Ss)
        A = (sqrtS.unsqueeze(1) * Vhs)          # (r, in)
        B = (Us * sqrtS.unsqueeze(0))           # (out, r)
        BA = B @ A                              # (out, in) ~= delta
        new_base = w_xf - BA                    # exact: new_base + 1.0*BA == w_xf

        residual = (new_base + BA - w_xf).abs().max().item()
        max_recon_residual = max(max_recon_residual, residual)

        new_base_sd[key] = new_base.to(torch.float16)
        lora_tensors[gname + ".lora_a"] = A.to(torch.float16).numpy()
        lora_tensors[gname + ".lora_b"] = B.to(torch.float16).numpy()
        n_lora += 1
        if n_lora <= 3 or args.smoke:
            print(f"  [svd] {key}  delta_rank<= {r}  A{tuple(A.shape)} B{tuple(B.shape)} "
                  f"BA_err(vs delta)={float((BA-delta).abs().max()):.3e}")

    print(f"[svd] decomposed {n_lora} linear layers @ rank {rank}; "
          f"fp32 reconstruction max|new_base+BA-W_xf| = {max_recon_residual:.3e}")
    return new_base_sd, lora_tensors, alpha, base_sd, xf_sd


def write_hf_base(new_base_sd, src_dir, out_base):
    """Save new_base as an HF dir: copy all non-weight files from src, write fp16 shards."""
    from safetensors.torch import save_file

    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir)
    # copy config / tokenizer / etc. (everything that is not a weight shard / index)
    for p in src_dir.iterdir():
        if p.suffix == ".safetensors" or p.name == "model.safetensors.index.json":
            continue
        if p.is_file():
            shutil.copy2(p, out_base / p.name)
    # single-shard save (8B fp16 ~16GB; fine on this box). keep contiguous + cpu.
    sd = {k: v.contiguous() for k, v in new_base_sd.items()}
    save_file(sd, str(out_base / "model.safetensors"), metadata={"format": "pt"})
    print(f"[base] wrote {len(sd)} tensors -> {out_base/'model.safetensors'}")


def write_gguf_lora(lora_tensors, alpha, out_gguf):
    from gguf import GGUFWriter

    out_gguf = Path(out_gguf)
    writer = GGUFWriter(str(out_gguf), arch="qwen3")
    writer.add_string("general.name", "qwen3_xf_lora_svd")
    # downstream reads adapter.lora.alpha as the scale buffer; with alpha==rank
    # the per-layer LoRALayer scale (alpha/r) == 1.0
    writer.add_float32("adapter.lora.alpha", float(alpha))
    for name, arr in lora_tensors.items():
        writer.add_tensor(name, np.ascontiguousarray(arr))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[gguf] wrote {len(lora_tensors)} lora tensors + alpha={alpha} -> {out_gguf}")


def verify_readback(out_base, out_gguf):
    """Confirm the GGUF reads back with correct *.weight_lora_a/_b keys."""
    import torch
    from transformers import AutoModelForCausalLM

    from xh_model_zoo.xh_llm.utils import load_gguf_checkpoint_for_lora

    print("[verify] loading new_base to drive the gguf->hf name map...")
    model = AutoModelForCausalLM.from_pretrained(out_base, torch_dtype=torch.float16)
    parsed = load_gguf_checkpoint_for_lora(str(out_gguf), return_tensors=True, model_to_load=model)
    alpha = parsed.get("adapter.lora.alpha", None)
    tensors = parsed["tensors"]
    a_keys = [k for k in tensors if k.endswith("_lora_a")]
    b_keys = [k for k in tensors if k.endswith("_lora_b")]
    print(f"[verify] adapter.lora.alpha = {alpha}")
    print(f"[verify] readback lora_a keys = {len(a_keys)}, lora_b keys = {len(b_keys)}")
    for k in sorted(a_keys)[:3]:
        print(f"    {k}  shape={tuple(tensors[k].shape)}")
    assert len(a_keys) == len(b_keys) > 0, "no lora keys read back"
    assert all(k.startswith("model.layers.") and "weight_lora_a" in k for k in a_keys[:1])
    return model, parsed


def verify_logits(out_base, out_gguf, merged_dir, device="cuda:0"):
    """End-to-end: (new_base + scale*B@A) logits vs original merged qwen3_xf logits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from xh_model_zoo.xh_llm.utils import load_gguf_checkpoint_for_lora

    tok = AutoTokenizer.from_pretrained(merged_dir)
    base = AutoModelForCausalLM.from_pretrained(out_base, torch_dtype=torch.float16)
    parsed = load_gguf_checkpoint_for_lora(str(out_gguf), return_tensors=True, model_to_load=base)
    scale = float(parsed["adapter.lora.alpha"]) / 64.0  # informational; alpha==rank=>1.0
    tensors = parsed["tensors"]

    # fold lora back into base in fp32 to reconstruct the merged model
    sd = base.state_dict()
    folded = 0
    for ak in [k for k in tensors if k.endswith("_lora_a")]:
        stem = ak[: -len("_lora_a")]            # model.layers.N....weight
        bk = stem + "_lora_b"
        A = tensors[ak].to(torch.float32)
        B = tensors[bk].to(torch.float32)
        w = sd[stem].to(torch.float32) + scale * (B @ A)
        sd[stem] = w.to(torch.float16)
        folded += 1
    base.load_state_dict(sd)
    print(f"[verify] folded {folded} lora pairs back into base")

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.float16)
    base.to(device).eval()
    merged.to(device).eval()

    text = "<|im_start|>user\n中国的首都是哪里？<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        lb = base(**ids).logits.float()
        lm = merged(**ids).logits.float()
    diff = (lb - lm).abs().max().item()
    print(f"[verify] end-to-end logits max abs diff (reconstructed vs qwen3_xf) = {diff:.3e}")
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="weights/Qwen3-8B")
    ap.add_argument("--merged", default="weights/qwen3_xf")
    ap.add_argument("--out-base", default="weights/qwen3_xf_base")
    ap.add_argument("--out-gguf", default="weights/qwen3_xf_lora.gguf")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="decompose only, skip writing full base; quick path sanity")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip decompose; just readback + logits on existing artifacts")
    args = ap.parse_args()

    if args.verify_only:
        verify_readback(args.out_base, args.out_gguf)
        verify_logits(args.out_base, args.out_gguf, args.merged, args.device)
        return

    new_base_sd, lora_tensors, alpha, _, _ = decompose(args)
    write_gguf_lora(lora_tensors, alpha, args.out_gguf)
    if args.smoke:
        print("[smoke] decompose + gguf write OK; base dir skipped.")
        return
    write_hf_base(new_base_sd, args.merged, args.out_base)
    verify_readback(args.out_base, args.out_gguf)
    verify_logits(args.out_base, args.out_gguf, args.merged, args.device)
    print("[done] LoRA decomposition complete.")


if __name__ == "__main__":
    main()
