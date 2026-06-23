"""Diagnose split HMONNX layer-by-layer: print key values to spot anomalies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from xhquant.core import CacheTensor
from xhquant.xhonnxruntime import HMONNXGrapInference, HMONNXInference


def _create_session(path: Path, device: torch.device, exec_device: torch.device):
    try:
        s = HMONNXGrapInference(str(path))
    except AttributeError as exc:
        if "'str' object has no attribute 'name'" not in str(exc):
            raise
        s = HMONNXInference(str(path))
    s.to(device)
    s.exec_device = exec_device
    return s


def _run(session, feed):
    outputs = session.run(feed)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    names = session.get_output_names()
    return {n: o for n, o in zip(names, outputs, strict=False)}


def _load_emb(path: Path) -> nn.Module:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj.eval()
    if isinstance(obj, dict) and "weight" in obj:
        emb = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1], dtype=obj["weight"].dtype)
        emb.load_state_dict(obj)
        return emb.eval()
    raise TypeError(f"Bad embedding: {type(obj)}")


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def diagnose(work_dir: str, token_id: int, device_id: int):
    work_dir = Path(work_dir).resolve()
    device = torch.device(f"cuda:{device_id}")
    exec_device = torch.device(f"cuda:{device_id}")

    meta = json.loads((work_dir / "split_moe_meta.json").read_text())
    num_layers = int(meta["num_hidden_layers"])
    num_experts = int(meta["num_experts"])
    top_k = int(meta["num_experts_per_tok"])
    layer_types = meta.get("layer_types", [])

    # Token embedding
    emb_path = _resolve(work_dir, meta["token_embedding_file"])
    emb = _load_emb(emb_path).to(device).eval()

    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
    hidden = emb(token).to(device)
    past_seq_len = 0

    print(f"Token: {token_id}, hidden dtype: {hidden.dtype}, device: {hidden.device}")
    print(f"Hidden stats: min={hidden.min().item():.4f}, max={hidden.max().item():.4f}, mean={hidden.mean().item():.4f}")
    print()

    layer_caches: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx in range(num_layers):
        premoe_path = work_dir / "hmonnx" / "premoe" / f"layer_{layer_idx:03d}_premoe.onnx"
        session = _create_session(premoe_path, device, exec_device)

        # Build feed
        pos = torch.tensor([[past_seq_len]], dtype=torch.int32, device=device)
        past = torch.tensor([past_seq_len], dtype=torch.int32, device=device)
        cur = torch.tensor([1], dtype=torch.int32, device=device)
        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            info = session.get_input(name)
            shape = tuple(int(d) for d in info.shape)
            if name == "hidden_in":
                feed[name] = hidden.to(dtype=info.dtype)
            elif name in ("time_position_ids", "hight_position_ids", "height_position_ids", "width_position_ids"):
                feed[name] = pos.to(dtype=info.dtype)
            elif name in ("valid_length", "past_seq_length"):
                feed[name] = past.to(dtype=info.dtype)
            elif name in ("current_length", "current_input_length"):
                feed[name] = cur.to(dtype=info.dtype)
            elif name == "linear_attn_mask":
                feed[name] = torch.ones(shape, dtype=info.dtype, device=device)
            elif name.startswith(("past_key_cache", "past_value_cache", "past_conv_cache", "past_recurrent_state")):
                lc = layer_caches.setdefault(layer_idx, {})
                if name not in lc:
                    lc[name] = CacheTensor(torch.zeros(shape, dtype=info.dtype, device=device))
                feed[name] = lc[name]
            else:
                feed[name] = torch.zeros(shape, dtype=info.dtype, device=device)

        outputs = _run(session, feed)

        # Update caches
        for name in list(layer_caches.get(layer_idx, {})):
            if name in outputs:
                layer_caches[layer_idx][name] = outputs[name]

        moe_input = outputs["moe_input"]
        topk_id_t = outputs["topk_id"].to(torch.long)
        topk_gate = outputs["topk_gate"]
        shared_out = outputs["shared_out"]
        residual1 = outputs["residual1"]

        expert_ids = [int(topk_id_t.reshape(-1)[i].item()) for i in range(top_k)]
        gate_vals = [float(topk_gate.reshape(-1)[i].item()) for i in range(top_k)]

        # Check for anomalies
        issues = []
        if moe_input.isnan().any() or moe_input.isinf().any():
            issues.append("moe_input has NaN/Inf")
        if shared_out.isnan().any() or shared_out.isinf().any():
            issues.append("shared_out has NaN/Inf")
        if residual1.isnan().any() or residual1.isinf().any():
            issues.append("residual1 has NaN/Inf")
        for eid in expert_ids:
            if eid < 0 or eid >= num_experts:
                issues.append(f"expert_id {eid} out of range [0, {num_experts})")

        lt = layer_types[layer_idx] if layer_idx < len(layer_types) else "?"
        print(
            f"Layer {layer_idx:02d} [{lt}]: "
            f"experts={expert_ids} gates={[f'{g:.4f}' for g in gate_vals]} "
            f"hidden_norm={hidden.norm().item():.2f} "
            f"moe_norm={moe_input.norm().item():.2f} "
            f"shared_norm={shared_out.norm().item():.2f} "
            f"residual1_norm={residual1.norm().item():.2f}"
        )
        if issues:
            print(f"  *** ISSUES: {issues}")

        # Run experts
        expert_outs = []
        for k_idx, eid in enumerate(expert_ids):
            epath = work_dir / "hmonnx" / "experts" / f"layer_{layer_idx:03d}" / f"expert_{eid:03d}.onnx"
            esess = _create_session(epath, device, exec_device)
            einfo = esess.get_input(esess.get_input_names()[0])
            eout = _run(esess, {"expert_input": moe_input.to(dtype=einfo.dtype)})
            eo = eout.get("expert_out", next(iter(eout.values())))
            expert_outs.append(eo)
            if eo.isnan().any() or eo.isinf().any():
                print(f"  *** Expert {eid} output has NaN/Inf")
            if layer_idx == 0:
                print(f"  Expert {eid}: out_norm={eo.norm().item():.2f}")

        # PostMoE
        ppath = work_dir / "hmonnx" / "postmoe" / "postmoe_decode_npu_agg.onnx"
        psess = _create_session(ppath, device, exec_device)
        pfeed: Dict[str, torch.Tensor] = {}
        for name in psess.get_input_names():
            pinfo = psess.get_input(name)
            if name.startswith("expert_out_"):
                idx = int(name.split("_")[-1])
                pfeed[name] = expert_outs[idx].to(dtype=pinfo.dtype)
            elif name == "topk_gate":
                pfeed[name] = topk_gate.to(dtype=pinfo.dtype)
            elif name == "shared_out":
                pfeed[name] = shared_out.to(dtype=pinfo.dtype)
            elif name == "residual1":
                pfeed[name] = residual1.to(dtype=pinfo.dtype)
        pout = _run(psess, pfeed)
        hidden = pout.get("hidden_out", next(iter(pout.values())))

        if layer_idx == 0:
            print(f"  PostMoE hidden_norm={hidden.norm().item():.2f}")

        past_seq_len += 1

    # Head
    hpath = work_dir / "hmonnx" / "head" / "head.onnx"
    hsess = _create_session(hpath, device, exec_device)
    hinfo = hsess.get_input(hsess.get_input_names()[0])
    hout = _run(hsess, {"hidden_in": hidden.to(dtype=hinfo.dtype)})
    logits = hout.get("logits", next(iter(hout.values())))

    top5_vals, top5_ids = torch.topk(logits[0, -1, :], 5)
    print(f"\nFinal logits top-5:")
    for i in range(5):
        tid = int(top5_ids[i].item())
        tval = float(top5_vals[i].item())
        print(f"  {i+1}. id={tid} val={tval:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--token-id", type=int, default=12675)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    diagnose(args.work_dir, args.token_id, args.device_id)


if __name__ == "__main__":
    main()
