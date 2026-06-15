# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Probe submodules inside one VLA-JEPA Qwen context self-attention block."""

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

import onnx
import torch
from torch import nn

from compare.compare_context_graph_hmonnx import diff_summary
from export.export_action_head import tensor_summary
from export.export_context_graph_wrapper import DEFAULT_MODEL, NativeCausalMaskedSoftmax, build_graph_inputs, build_wrapper
from patches.patch_hmonnx_rmsnorm_casts import fuse_rmsnorm
from debug.probe_context_layer_parts import export_part_onnx, run_part_hmonnx, select_onnx_inputs
from debug.probe_context_single_layer import DEFAULT_SAMPLE, compute_position_embeddings, run_prefix_hidden
from common.paths import output_str, set_default_libero_config_path

DEFAULT_OUT_DIR = output_str("context_encoder", "attention_part_probe")


class OneInputModule(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class QueryRopePart(nn.Module):
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.attn = attn

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        query_states, _key_states = self.attn.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        return query_states


class KeyRopePart(nn.Module):
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.attn = attn

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        _query_states, key_states = self.attn.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        return key_states


class QueryScalePart(nn.Module):
    def __init__(self, kv_scale: float) -> None:
        super().__init__()
        self.kv_scale = float(kv_scale)

    def forward(self, query_states: torch.Tensor) -> torch.Tensor:
        return query_states * self.kv_scale


class KeyForScoresPart(nn.Module):
    def __init__(self, kv_repeat_indices: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("kv_repeat_indices", kv_repeat_indices.detach().clone(), persistent=False)

    def forward(self, key_states: torch.Tensor) -> torch.Tensor:
        key_states = key_states.transpose(2, 3)
        return key_states.index_select(1, self.kv_repeat_indices)


class ValueRepeatPart(nn.Module):
    def __init__(self, kv_repeat_indices: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("kv_repeat_indices", kv_repeat_indices.detach().clone(), persistent=False)

    def forward(self, value_states: torch.Tensor) -> torch.Tensor:
        return value_states.index_select(1, self.kv_repeat_indices)


class AttentionCorePart(nn.Module):
    def __init__(self, num_attention_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = int(head_dim)
        self.masked_softmax = NativeCausalMaskedSoftmax(dim=-1)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states_for_scores: torch.Tensor,
        value_states: torch.Tensor,
        past_seq_length: torch.Tensor,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query_states, key_states_for_scores)
        attn_weights = self.masked_softmax(attn_weights, past_seq_length)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        bsz = attn_output.shape[0]
        q_len = attn_output.shape[1]
        return attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)


class AttentionMatmulScoresPart(nn.Module):
    def forward(self, query_states: torch.Tensor, key_states_for_scores: torch.Tensor) -> torch.Tensor:
        return torch.matmul(query_states, key_states_for_scores)


class AttentionSoftmaxPart(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masked_softmax = NativeCausalMaskedSoftmax(dim=-1)

    def forward(self, attn_weights: torch.Tensor, past_seq_length: torch.Tensor) -> torch.Tensor:
        return self.masked_softmax(attn_weights, past_seq_length)


class AttentionValueMatmulPart(nn.Module):
    def __init__(self, num_attention_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = int(head_dim)

    def forward(self, attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        bsz = attn_output.shape[0]
        q_len = attn_output.shape[1]
        return attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)


def _ref_attention_intermediates(
    language_model: nn.Module,
    layer_idx: int,
    graph_inputs: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    with torch.no_grad():
        layer_input = run_prefix_hidden(language_model, graph_inputs, layer_idx)
        input_norm = layer.input_layernorm(layer_input)
        position_embeddings = compute_position_embeddings(
            language_model, graph_inputs[1], graph_inputs[2], graph_inputs[3]
        )
        cos, sin = position_embeddings
        bsz, q_len, _hidden = input_norm.size()
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim

        q_linear = attn.q_proj(input_norm)
        k_linear = attn.k_proj(input_norm)
        v_linear = attn.v_proj(input_norm)

        q_view = q_linear.view(bsz, q_len, num_heads, head_dim)
        k_view = k_linear.view(bsz, q_len, num_kv_heads, head_dim)
        v_view = v_linear.view(bsz, q_len, num_kv_heads, head_dim)

        q_norm = attn.q_norm(q_view)
        k_norm = attn.k_norm(k_view)
        q_states = q_norm.transpose(1, 2)
        k_states = k_norm.transpose(1, 2)
        v_states = v_view.transpose(1, 2)

        q_rope, k_rope = attn.apply_rotary_pos_emb(q_states, k_states, cos, sin, unsqueeze_dim=1)
        q_scaled = q_rope * attn.kv_scale
        k_for_scores = k_rope.transpose(2, 3).index_select(1, attn.kv_repeat_indices)
        attn_scores = torch.matmul(q_scaled, k_for_scores)
        attn_probs = attn.masked_softmax(attn_scores, graph_inputs[4])
        v_repeated = v_states.index_select(1, attn.kv_repeat_indices)
        pre_o = torch.matmul(attn_probs, v_repeated).transpose(1, 2).reshape(
            bsz, q_len, num_heads * head_dim
        )
        o_proj = attn.o_proj(pre_o)

    return {
        "input_norm": input_norm,
        "cos": cos,
        "sin": sin,
        "q_linear": q_linear,
        "k_linear": k_linear,
        "v_linear": v_linear,
        "q_view": q_view,
        "k_view": k_view,
        "v_view": v_view,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "q_states": q_states,
        "k_states": k_states,
        "v_states": v_states,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "q_scaled": q_scaled,
        "k_for_scores": k_for_scores,
        "attn_scores": attn_scores,
        "attn_probs": attn_probs,
        "v_repeated": v_repeated,
        "pre_o": pre_o,
        "o_proj": o_proj,
    }


def build_attention_part_specs(
    language_model: nn.Module,
    layer_idx: int,
    graph_inputs: tuple[torch.Tensor, ...],
) -> dict[str, dict[str, Any]]:
    layer = language_model.layers[layer_idx]
    attn = layer.self_attn
    ref = _ref_attention_intermediates(language_model, layer_idx, graph_inputs)

    return {
        "q_proj": {"model": OneInputModule(attn.q_proj), "inputs": (ref["input_norm"],), "input_names": ["hidden_states"], "reference": ref["q_linear"]},
        "k_proj": {"model": OneInputModule(attn.k_proj), "inputs": (ref["input_norm"],), "input_names": ["hidden_states"], "reference": ref["k_linear"]},
        "v_proj": {"model": OneInputModule(attn.v_proj), "inputs": (ref["input_norm"],), "input_names": ["hidden_states"], "reference": ref["v_linear"]},
        "q_norm": {"model": OneInputModule(attn.q_norm), "inputs": (ref["q_view"],), "input_names": ["query_states"], "reference": ref["q_norm"]},
        "k_norm": {"model": OneInputModule(attn.k_norm), "inputs": (ref["k_view"],), "input_names": ["key_states"], "reference": ref["k_norm"]},
        "q_rope": {"model": QueryRopePart(attn), "inputs": (ref["q_states"], ref["k_states"], ref["cos"], ref["sin"]), "input_names": ["query_states", "key_states", "cos", "sin"], "reference": ref["q_rope"]},
        "k_rope": {"model": KeyRopePart(attn), "inputs": (ref["q_states"], ref["k_states"], ref["cos"], ref["sin"]), "input_names": ["query_states", "key_states", "cos", "sin"], "reference": ref["k_rope"]},
        "q_scale": {"model": QueryScalePart(attn.kv_scale), "inputs": (ref["q_rope"],), "input_names": ["query_states"], "reference": ref["q_scaled"]},
        "k_for_scores": {"model": KeyForScoresPart(attn.kv_repeat_indices), "inputs": (ref["k_rope"],), "input_names": ["key_states"], "reference": ref["k_for_scores"]},
        "v_repeat": {"model": ValueRepeatPart(attn.kv_repeat_indices), "inputs": (ref["v_states"],), "input_names": ["value_states"], "reference": ref["v_repeated"]},
        "attn_scores": {"model": AttentionMatmulScoresPart(), "inputs": (ref["q_scaled"], ref["k_for_scores"]), "input_names": ["query_states", "key_states_for_scores"], "reference": ref["attn_scores"]},
        "attn_softmax": {"model": AttentionSoftmaxPart(), "inputs": (ref["attn_scores"], graph_inputs[4]), "input_names": ["attn_weights", "past_seq_length"], "reference": ref["attn_probs"]},
        "attn_value": {"model": AttentionValueMatmulPart(attn.config.num_attention_heads, attn.head_dim), "inputs": (ref["attn_probs"], ref["v_repeated"]), "input_names": ["attn_weights", "value_states"], "reference": ref["pre_o"]},
        "attention_core": {"model": AttentionCorePart(attn.config.num_attention_heads, attn.head_dim), "inputs": (ref["q_scaled"], ref["k_for_scores"], ref["v_repeated"], graph_inputs[4]), "input_names": ["query_states", "key_states_for_scores", "value_states", "past_seq_length"], "reference": ref["pre_o"]},
        "o_proj": {"model": OneInputModule(attn.o_proj), "inputs": (ref["pre_o"],), "input_names": ["hidden_states"], "reference": ref["o_proj"]},
    }


def maybe_fuse_rmsnorm(hmonnx_path: Path) -> int:
    model = onnx.load(str(hmonnx_path), load_external_data=False)
    count = fuse_rmsnorm(model)
    if count:
        onnx.save(model, str(hmonnx_path))
    return count


def convert_part_to_hmonnx_with_ops(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    quant_type: str,
    ops: dict[str, str],
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    example_inputs, onnx_input_names = select_onnx_inputs(onnx_path, inputs, input_names)
    quant_config = create_quant_config(
        QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, ops=ops)
    )
    convert_onnx_to_hmonnx(
        str(onnx_path),
        example_inputs,
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config,
        simplify=False,
    )
    return {
        "hmonnx_path": str(hmonnx_path),
        "size_mb": hmonnx_path.stat().st_size / 1024 / 1024,
        "onnx_input_names": onnx_input_names,
        "ops": ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--quant-type", default="w8a8_sefp")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--parts", nargs="*", default=None)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    parser.add_argument("--matmul-quant-type", default=None)
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(67)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    graph_inputs = build_graph_inputs(sample, args.device, dtype)
    wrapper = build_wrapper(args.model, args.device, dtype, int(graph_inputs[0].shape[1]))
    language_model = wrapper.wrapped_qwen.language_model
    specs = build_attention_part_specs(language_model, args.layer_idx, graph_inputs)
    selected_parts = args.parts or list(specs)

    report: dict[str, Any] = {
        "model": args.model,
        "sample": args.sample,
        "layer_idx": args.layer_idx,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "dtype": args.dtype,
        "quant_type": args.quant_type,
        "fuse_rmsnorm": bool(args.fuse_rmsnorm),
        "matmul_quant_type": args.matmul_quant_type,
        "parts": {},
    }

    for part_name in selected_parts:
        if part_name not in specs:
            raise ValueError(f"Unknown part {part_name}; valid parts: {sorted(specs)}")
        spec = specs[part_name]
        part_model = spec["model"].to(device=args.device, dtype=dtype).eval()
        part_inputs = spec["inputs"]
        input_names = spec["input_names"]
        reference = spec["reference"]

        stem = f"context_layer{args.layer_idx}_attn_{part_name}_{args.dtype}"
        if args.fuse_rmsnorm:
            stem = f"{stem}_fused_rmsnorm"
        if args.matmul_quant_type:
            stem = f"{stem}_matmul_{args.matmul_quant_type}"
        onnx_path = out_dir / f"{stem}.onnx"
        hmonnx_path = out_dir / f"{stem}_{args.quant_type}.hmonnx.onnx"

        export_part_onnx(part_model, part_inputs, input_names, onnx_path, args.opset)
        ops = {"MatMul": args.matmul_quant_type} if args.matmul_quant_type else {}
        hmonnx_export = convert_part_to_hmonnx_with_ops(
            onnx_path, hmonnx_path, part_inputs, input_names, args.quant_type, ops
        )
        fused_rmsnorm_patterns = maybe_fuse_rmsnorm(hmonnx_path) if args.fuse_rmsnorm else 0
        hmonnx_output, onnx_input_names = run_part_hmonnx(
            hmonnx_path, onnx_path, part_inputs, input_names, args.hmonnx_device
        )

        report["parts"][part_name] = {
            "onnx": str(onnx_path),
            "hmonnx": str(hmonnx_path),
            "onnx_size_mb": onnx_path.stat().st_size / 1024 / 1024,
            "hmonnx_export": hmonnx_export,
            "fused_rmsnorm_patterns": fused_rmsnorm_patterns,
            "onnx_input_names": onnx_input_names,
            "inputs": {str(idx): tensor_summary(item) for idx, item in enumerate(part_inputs)},
            "reference": tensor_summary(reference),
            "torch_vs_hmonnx": diff_summary(reference, hmonnx_output.to(args.device)),
        }

    suffix = "_fused_rmsnorm" if args.fuse_rmsnorm else ""
    if args.matmul_quant_type:
        suffix = f"{suffix}_matmul_{args.matmul_quant_type}"
    report_path = out_dir / f"context_layer{args.layer_idx}_attention_parts_{args.dtype}_{args.quant_type}{suffix}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
