# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export and compare a single VLA-JEPA Qwen context decoder layer."""

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
from torch import nn

from compare.compare_context_graph_hmonnx import diff_summary, run_hmonnx_context
from export.export_action_head import tensor_summary
from common.paths import output_str, set_default_libero_config_path
from export.export_context_graph_wrapper import (
    DEFAULT_MODEL,
    build_graph_inputs,
    build_wrapper,
    convert_to_hmonnx,
    export_onnx,
    fuse_hmonnx_rmsnorm,
)

DEFAULT_SAMPLE = output_str("context_encoder", "live_eval", "task0_seed1000_context_graph_sample.pt")
DEFAULT_OUT_DIR = output_str("context_encoder", "layer_probe")

LAYER_INPUT_NAMES = [
    "hidden_states",
    "time_position_ids",
    "height_position_ids",
    "width_position_ids",
    "past_seq_length",
    "current_input_length",
]


def compute_position_embeddings(
    language_model: nn.Module,
    time_position_ids: torch.Tensor,
    height_position_ids: torch.Tensor,
    width_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = language_model.rotary_emb.cos_cached
    sin = language_model.rotary_emb.sin_cached

    time_cos = cos[time_position_ids]
    time_sin = sin[time_position_ids]
    height_cos = cos[height_position_ids]
    height_sin = sin[height_position_ids]
    width_cos = cos[width_position_ids]
    width_sin = sin[width_position_ids]

    time_cos = time_cos * language_model.time_mask
    time_sin = time_sin * language_model.time_mask
    height_cos = height_cos * language_model.hight_mask
    height_sin = height_sin * language_model.hight_mask
    width_cos = width_cos * language_model.width_mask
    width_sin = width_sin * language_model.width_mask

    cos = time_cos + height_cos + width_cos
    sin = time_sin + height_sin + width_sin
    return cos.squeeze(1).unsqueeze(0).unsqueeze(0), sin.squeeze(1).unsqueeze(0).unsqueeze(0)


def run_prefix_hidden(
    language_model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    target_layer: int,
) -> torch.Tensor:
    (
        inputs_embeds,
        time_position_ids,
        height_position_ids,
        width_position_ids,
        past_seq_length,
        current_input_length,
        _embodied_positions,
        deepstack_visual_embed_0,
        deepstack_visual_embed_1,
        deepstack_visual_embed_2,
    ) = inputs
    position_embeddings = compute_position_embeddings(
        language_model, time_position_ids, height_position_ids, width_position_ids
    )
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers[:target_layer]):
        hidden_states = decoder_layer(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_embeddings=position_embeddings,
        )
        if layer_idx == 0:
            hidden_states = hidden_states + deepstack_visual_embed_0
        if layer_idx == 1:
            hidden_states = hidden_states + deepstack_visual_embed_1
        if layer_idx == 2:
            hidden_states = hidden_states + deepstack_visual_embed_2
    return hidden_states


class SingleDecoderLayerGraph(nn.Module):
    def __init__(self, language_model: nn.Module, layer_idx: int) -> None:
        super().__init__()
        self.language_model = language_model
        self.layer = language_model.layers[layer_idx]
        self.layer_idx = int(layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = compute_position_embeddings(
            self.language_model, time_position_ids, height_position_ids, width_position_ids
        )
        return self.layer(
            hidden_states,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            position_embeddings=position_embeddings,
        )


def export_layer_onnx(model: nn.Module, inputs: tuple[torch.Tensor, ...], onnx_path: Path, opset: int) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=LAYER_INPUT_NAMES,
        output_names=["hidden_states_out"],
    )


def select_layer_onnx_inputs(onnx_path: Path, inputs: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], list[str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx_input_names = [value.name for value in model.graph.input]
    input_by_name = dict(zip(LAYER_INPUT_NAMES, inputs, strict=True))
    return [input_by_name[name].detach().cpu() for name in onnx_input_names], onnx_input_names


def convert_layer_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    quant_type: str,
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    example_inputs, onnx_input_names = select_layer_onnx_inputs(onnx_path, inputs)
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


def run_layer_hmonnx(
    hmonnx_path: str,
    onnx_path: str,
    inputs: tuple[torch.Tensor, ...],
    hmonnx_device: str,
) -> tuple[torch.Tensor, list[str]]:
    from xhquant.api import HMONNXInference

    _, onnx_input_names = select_layer_onnx_inputs(Path(onnx_path), inputs)
    input_by_name = dict(zip(LAYER_INPUT_NAMES, inputs, strict=True))
    session = HMONNXInference(hmonnx_path)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--quant-type", default="w16a16_sefp")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(59)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"context_layer{args.layer_idx}_{args.dtype}"
    onnx_path = out_dir / f"{stem}.onnx"
    hmonnx_path = out_dir / f"{stem}_{args.quant_type}.hmonnx.onnx"
    compare_hmonnx_path = hmonnx_path
    report_suffix = args.quant_type
    if args.fuse_rmsnorm:
        compare_hmonnx_path = out_dir / f"{stem}_{args.quant_type}.fused_rmsnorm.hmonnx.onnx"
        report_suffix = f"{args.quant_type}_fused_rmsnorm"
    report_path = out_dir / f"{stem}_{report_suffix}_compare_report.json"

    sample = torch.load(args.sample, map_location="cpu", weights_only=False)
    graph_inputs = build_graph_inputs(sample, args.device, dtype)
    wrapper = build_wrapper(args.model, args.device, dtype, int(graph_inputs[0].shape[1]))
    language_model = wrapper.wrapped_qwen.language_model

    with torch.no_grad():
        layer_input_hidden = run_prefix_hidden(language_model, graph_inputs, args.layer_idx)

    layer_inputs = (
        layer_input_hidden,
        graph_inputs[1],
        graph_inputs[2],
        graph_inputs[3],
        graph_inputs[4],
        graph_inputs[5],
    )
    layer_graph = SingleDecoderLayerGraph(language_model, args.layer_idx).to(device=args.device, dtype=dtype).eval()
    with torch.no_grad():
        torch_output = layer_graph(*layer_inputs)

    export_layer_onnx(layer_graph, layer_inputs, onnx_path, args.opset)
    hmonnx_export = convert_layer_to_hmonnx(onnx_path, hmonnx_path, layer_inputs, args.quant_type)
    fused_hmonnx = None
    if args.fuse_rmsnorm:
        fused_hmonnx = fuse_hmonnx_rmsnorm(hmonnx_path, compare_hmonnx_path)
    hmonnx_output, onnx_input_names = run_layer_hmonnx(
        str(compare_hmonnx_path), str(onnx_path), layer_inputs, args.hmonnx_device
    )

    report = {
        "model": args.model,
        "sample": args.sample,
        "layer_idx": args.layer_idx,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "dtype": args.dtype,
        "quant_type": args.quant_type,
        "onnx": str(onnx_path),
        "hmonnx": str(hmonnx_path),
        "compare_hmonnx": str(compare_hmonnx_path),
        "fuse_rmsnorm": bool(args.fuse_rmsnorm),
        "onnx_size_mb": onnx_path.stat().st_size / 1024 / 1024,
        "hmonnx_export": hmonnx_export,
        "fused_hmonnx": fused_hmonnx,
        "onnx_input_names": onnx_input_names,
        "layer_input": tensor_summary(layer_input_hidden),
        "torch_output": tensor_summary(torch_output),
        "torch_vs_hmonnx": diff_summary(torch_output, hmonnx_output.to(args.device)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
