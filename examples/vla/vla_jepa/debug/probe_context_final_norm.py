# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export and compare final norm/gather after a VLA-JEPA Qwen context prefix."""

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
from debug.probe_context_single_layer import DEFAULT_SAMPLE, run_prefix_hidden
from common.paths import output_str, set_default_libero_config_path

DEFAULT_OUT_DIR = output_str("context_encoder", "final_norm_probe")
INPUT_NAMES = ["hidden_states", "embodied_positions"]


class FinalNormGatherGraph(nn.Module):
    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.norm = language_model.norm

    def forward(self, hidden_states: torch.Tensor, embodied_positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        gather_index = embodied_positions.reshape(32)
        return hidden_states.index_select(1, gather_index)


def export_onnx(model: nn.Module, inputs: tuple[torch.Tensor, ...], onnx_path: Path, opset: int) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=INPUT_NAMES,
        output_names=["conditioning_tokens"],
    )


def select_onnx_inputs(
    onnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[list[torch.Tensor], list[str]]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx_input_names = [value.name for value in model.graph.input]
    input_by_name = dict(zip(INPUT_NAMES, inputs, strict=True))
    return [input_by_name[name].detach().cpu() for name in onnx_input_names], onnx_input_names


def convert_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    quant_type: str,
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    example_inputs, onnx_input_names = select_onnx_inputs(onnx_path, inputs)
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


def run_hmonnx(
    hmonnx_path: Path,
    onnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    hmonnx_device: str,
) -> tuple[torch.Tensor, list[str]]:
    from xhquant.api import HMONNXInference

    _, onnx_input_names = select_onnx_inputs(onnx_path, inputs)
    input_by_name = dict(zip(INPUT_NAMES, inputs, strict=True))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmonnx-device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--quant-type", default="w8a16_sefp")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(67)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"context_final_norm_after_layers{args.num_layers}_{args.dtype}"
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
        hidden_states = run_prefix_hidden(language_model, graph_inputs, args.num_layers)

    probe_inputs = (hidden_states, graph_inputs[6])
    final_norm_graph = FinalNormGatherGraph(language_model).to(device=args.device, dtype=dtype).eval()
    with torch.no_grad():
        torch_output = final_norm_graph(*probe_inputs)

    export_onnx(final_norm_graph, probe_inputs, onnx_path, args.opset)
    hmonnx_export = convert_to_hmonnx(onnx_path, hmonnx_path, probe_inputs, args.quant_type)
    fused_hmonnx = None
    if args.fuse_rmsnorm:
        fused_hmonnx = fuse_hmonnx_rmsnorm(hmonnx_path, compare_hmonnx_path)
    hmonnx_output, onnx_input_names = run_hmonnx(compare_hmonnx_path, onnx_path, probe_inputs, args.hmonnx_device)

    report = {
        "model": args.model,
        "sample": args.sample,
        "num_layers": args.num_layers,
        "device": args.device,
        "hmonnx_device": args.hmonnx_device,
        "dtype": args.dtype,
        "quant_type": args.quant_type,
        "fuse_rmsnorm": bool(args.fuse_rmsnorm),
        "onnx": str(onnx_path),
        "hmonnx": str(hmonnx_path),
        "compare_hmonnx": str(compare_hmonnx_path),
        "onnx_size_mb": onnx_path.stat().st_size / 1024 / 1024,
        "hmonnx_export": hmonnx_export,
        "fused_hmonnx": fused_hmonnx,
        "onnx_input_names": onnx_input_names,
        "inputs": {str(idx): tensor_summary(item) for idx, item in enumerate(probe_inputs)},
        "torch_output": tensor_summary(torch_output),
        "torch_vs_hmonnx": diff_summary(torch_output, hmonnx_output.to(args.device)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
