# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Inspect largest HMONNX errors in VLA-JEPA context layer submodules."""

from __future__ import annotations

import sys
from pathlib import Path

_VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _VLA_JEPA_ROOT.parents[2]
for _path in (_VLA_JEPA_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from export.export_context_graph_wrapper import DEFAULT_MODEL, build_graph_inputs, build_wrapper
from debug.probe_context_layer_parts import DEFAULT_OUT_DIR, build_part_specs, run_part_hmonnx
from debug.probe_context_single_layer import DEFAULT_SAMPLE
from common.paths import set_default_libero_config_path


def unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    indices = []
    for dim in reversed(shape):
        indices.append(flat_index % dim)
        flat_index //= dim
    return tuple(reversed(indices))


def topdiff_summary(
    part_name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    topk: int,
) -> dict[str, Any]:
    ref = reference.detach().float().cpu()
    cand = candidate.detach().float().cpu()
    inp = input_tensor.detach().float().cpu()
    diff = (ref - cand).abs()
    flat_diff = diff.reshape(-1)
    values, flat_indices = torch.topk(flat_diff, k=min(topk, flat_diff.numel()))

    token_variance = (inp * inp).mean(dim=-1, keepdim=True)
    token_rms = torch.sqrt(token_variance)
    weight_cpu = None if weight is None else weight.detach().float().cpu()

    items = []
    for value, flat_index in zip(values.tolist(), flat_indices.tolist(), strict=True):
        index = unravel_index(flat_index, tuple(ref.shape))
        batch, token, channel = index
        item = {
            "index": {"batch": batch, "token": token, "channel": channel},
            "abs_diff": float(value),
            "reference": float(ref[index]),
            "candidate": float(cand[index]),
            "input": float(inp[index]),
            "token_rms": float(token_rms[batch, token, 0]),
            "token_variance": float(token_variance[batch, token, 0]),
            "token_input_min": float(inp[batch, token].min()),
            "token_input_max": float(inp[batch, token].max()),
        }
        if weight_cpu is not None:
            item["weight"] = float(weight_cpu[channel])
        items.append(item)

    return {
        "part": part_name,
        "shape": list(ref.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(torch.quantile(diff.reshape(-1), 0.95)),
        "topk": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--quant-type", default="w16a16_sefp")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--parts",
        nargs="*",
        default=["input_layernorm", "post_attention_layernorm"],
        choices=("input_layernorm", "post_attention_layernorm"),
    )
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(67)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    graph_inputs = build_graph_inputs(sample, args.device, dtype)
    wrapper = build_wrapper(args.model, args.device, dtype, int(graph_inputs[0].shape[1]))
    language_model = wrapper.wrapped_qwen.language_model
    specs = build_part_specs(language_model, args.layer_idx, graph_inputs)

    report: dict[str, Any] = {
        "model": args.model,
        "sample": args.sample,
        "layer_idx": args.layer_idx,
        "dtype": args.dtype,
        "quant_type": args.quant_type,
        "parts": {},
    }

    for part_name in args.parts:
        spec = specs[part_name]
        stem = f"context_layer{args.layer_idx}_{part_name}_{args.dtype}"
        onnx_path = out_dir / f"{stem}.onnx"
        hmonnx_path = out_dir / f"{stem}_{args.quant_type}.hmonnx.onnx"
        if not onnx_path.exists() or not hmonnx_path.exists():
            raise FileNotFoundError(f"Missing probe files for {part_name}: {onnx_path}, {hmonnx_path}")

        hmonnx_output, _onnx_input_names = run_part_hmonnx(
            hmonnx_path, onnx_path, spec["inputs"], spec["input_names"], args.hmonnx_device
        )
        module = getattr(spec["model"], part_name, None)
        weight = getattr(module, "weight", None)
        report["parts"][part_name] = topdiff_summary(
            part_name=part_name,
            reference=spec["reference"],
            candidate=hmonnx_output.to(args.device),
            input_tensor=spec["inputs"][0],
            weight=weight,
            topk=args.topk,
        )

    report_path = out_dir / f"context_layer{args.layer_idx}_rmsnorm_topdiff_{args.dtype}_{args.quant_type}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
