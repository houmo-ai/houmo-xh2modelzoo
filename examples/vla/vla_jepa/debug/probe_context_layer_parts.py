# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Probe submodules inside one VLA-JEPA Qwen context decoder layer."""

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
from export.export_context_graph_wrapper import DEFAULT_MODEL, build_graph_inputs, build_wrapper, fuse_hmonnx_rmsnorm
from debug.probe_context_single_layer import DEFAULT_SAMPLE, compute_position_embeddings, run_prefix_hidden
from common.paths import output_str, set_default_libero_config_path

DEFAULT_OUT_DIR = output_str("context_encoder", "layer_part_probe")


class InputLayerNormPart(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.input_layernorm = layer.input_layernorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.input_layernorm(hidden_states)


class SelfAttentionPart(nn.Module):
    def __init__(self, language_model: nn.Module, layer: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.self_attn = layer.self_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = compute_position_embeddings(
            self.language_model, time_position_ids, height_position_ids, width_position_ids
        )
        output, _attn_weights, _present = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        return output


class PostAttentionLayerNormPart(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.post_attention_layernorm(hidden_states)


class MLPPart(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.mlp = layer.mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class AttentionResidualPart(nn.Module):
    def __init__(self, language_model: nn.Module, layer: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model
        self.layer = layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        position_embeddings = compute_position_embeddings(
            self.language_model, time_position_ids, height_position_ids, width_position_ids
        )
        hidden_states = self.layer.input_layernorm(hidden_states)
        hidden_states, _attn_weights, _present = self.layer.self_attn(
            hidden_states=hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_embeddings=position_embeddings,
        )
        return residual + hidden_states


class PostNormMLPPart(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.post_attention_layernorm(hidden_states))


class TailAfterAttentionPart(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def export_part_onnx(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    onnx_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["output"],
    )


def select_onnx_inputs(
    onnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
) -> tuple[list[torch.Tensor], list[str]]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx_input_names = [value.name for value in model.graph.input]
    input_by_name = dict(zip(input_names, inputs, strict=True))
    return [input_by_name[name].detach().cpu() for name in onnx_input_names], onnx_input_names


def convert_part_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    quant_type: str,
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    example_inputs, onnx_input_names = select_onnx_inputs(onnx_path, inputs, input_names)
    quant_config = create_quant_config(QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type))
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
    }


def run_part_hmonnx(
    hmonnx_path: Path,
    onnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    hmonnx_device: str,
) -> tuple[torch.Tensor, list[str]]:
    from xhquant.api import HMONNXInference

    _, onnx_input_names = select_onnx_inputs(onnx_path, inputs, input_names)
    input_by_name = dict(zip(input_names, inputs, strict=True))
    session = HMONNXInference(str(hmonnx_path))
    session.to(hmonnx_device)
    hmonnx_inputs = []
    for input_info in session.inputs:
        item = input_by_name[input_info.name].detach().cpu().to(dtype=input_info.dtype)
        hmonnx_inputs.append(item.to(hmonnx_device))
    with torch.no_grad():
        output = session(*hmonnx_inputs)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output, onnx_input_names


def build_part_specs(
    language_model: nn.Module,
    layer_idx: int,
    graph_inputs: tuple[torch.Tensor, ...],
) -> dict[str, dict[str, Any]]:
    layer = language_model.layers[layer_idx]
    with torch.no_grad():
        layer_input = run_prefix_hidden(language_model, graph_inputs, layer_idx)
        position_embeddings = compute_position_embeddings(
            language_model, graph_inputs[1], graph_inputs[2], graph_inputs[3]
        )
        input_norm = layer.input_layernorm(layer_input)
        attn_out, _attn_weights, _present = layer.self_attn(
            hidden_states=input_norm,
            past_seq_length=graph_inputs[4],
            current_input_length=graph_inputs[5],
            position_embeddings=position_embeddings,
        )
        attn_residual = layer_input + attn_out
        post_norm = layer.post_attention_layernorm(attn_residual)
        mlp_out = layer.mlp(post_norm)

    return {
        "input_layernorm": {
            "model": InputLayerNormPart(layer),
            "inputs": (layer_input,),
            "input_names": ["hidden_states"],
            "reference": input_norm,
        },
        "self_attn": {
            "model": SelfAttentionPart(language_model, layer),
            "inputs": (input_norm, graph_inputs[1], graph_inputs[2], graph_inputs[3]),
            "input_names": ["hidden_states", "time_position_ids", "height_position_ids", "width_position_ids"],
            "reference": attn_out,
        },
        "post_attention_layernorm": {
            "model": PostAttentionLayerNormPart(layer),
            "inputs": (attn_residual,),
            "input_names": ["hidden_states"],
            "reference": post_norm,
        },
        "mlp": {
            "model": MLPPart(layer),
            "inputs": (post_norm,),
            "input_names": ["hidden_states"],
            "reference": mlp_out,
        },
        "attn_residual": {
            "model": AttentionResidualPart(language_model, layer),
            "inputs": (
                layer_input,
                graph_inputs[1],
                graph_inputs[2],
                graph_inputs[3],
                graph_inputs[4],
                graph_inputs[5],
            ),
            "input_names": [
                "hidden_states",
                "time_position_ids",
                "height_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
            ],
            "reference": attn_residual,
        },
        "post_norm_mlp": {
            "model": PostNormMLPPart(layer),
            "inputs": (attn_residual,),
            "input_names": ["hidden_states"],
            "reference": mlp_out,
        },
        "tail_after_attn": {
            "model": TailAfterAttentionPart(layer),
            "inputs": (attn_residual,),
            "input_names": ["hidden_states"],
            "reference": attn_residual + mlp_out,
        },
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
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--parts", nargs="*", default=None)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(61)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    graph_inputs = build_graph_inputs(sample, args.device, dtype)
    wrapper = build_wrapper(args.model, args.device, dtype, int(graph_inputs[0].shape[1]))
    language_model = wrapper.wrapped_qwen.language_model
    specs = build_part_specs(language_model, args.layer_idx, graph_inputs)
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

        stem = f"context_layer{args.layer_idx}_{part_name}_{args.dtype}"
        onnx_path = out_dir / f"{stem}.onnx"
        hmonnx_path = out_dir / f"{stem}_{args.quant_type}.hmonnx.onnx"
        compare_hmonnx_path = hmonnx_path
        fused_hmonnx = None
        if args.fuse_rmsnorm:
            compare_hmonnx_path = out_dir / f"{stem}_{args.quant_type}.fused_rmsnorm.hmonnx.onnx"

        export_part_onnx(part_model, part_inputs, input_names, onnx_path, args.opset)
        hmonnx_export = convert_part_to_hmonnx(onnx_path, hmonnx_path, part_inputs, input_names, args.quant_type)
        if args.fuse_rmsnorm:
            fused_hmonnx = fuse_hmonnx_rmsnorm(hmonnx_path, compare_hmonnx_path)
        hmonnx_output, onnx_input_names = run_part_hmonnx(
            compare_hmonnx_path, onnx_path, part_inputs, input_names, args.hmonnx_device
        )

        report["parts"][part_name] = {
            "onnx": str(onnx_path),
            "hmonnx": str(hmonnx_path),
            "compare_hmonnx": str(compare_hmonnx_path),
            "onnx_size_mb": onnx_path.stat().st_size / 1024 / 1024,
            "hmonnx_export": hmonnx_export,
            "fused_hmonnx": fused_hmonnx,
            "onnx_input_names": onnx_input_names,
            "inputs": {str(idx): tensor_summary(item) for idx, item in enumerate(part_inputs)},
            "reference": tensor_summary(reference),
            "torch_vs_hmonnx": diff_summary(reference, hmonnx_output.to(args.device)),
        }

    report_suffix = args.quant_type if not args.fuse_rmsnorm else f"{args.quant_type}_fused_rmsnorm"
    report_path = out_dir / f"context_layer{args.layer_idx}_parts_{args.dtype}_{report_suffix}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
