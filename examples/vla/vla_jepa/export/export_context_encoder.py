# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

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

import numpy as np
import torch
from torch import nn

from export.export_action_head import tensor_summary
from common.inspect_policy import DEFAULT_MODEL, load_policy
from common.paths import output_str, set_default_libero_config_path

DEFAULT_SAMPLE = output_str("context_encoder", "dummy_sample.pt")
DEFAULT_OUT_DIR = output_str("context_encoder")


class ContextEncoderFromQwenInputs(nn.Module):
    """Qwen input tensors -> embodied action conditioning tokens.

    This mirrors VLAJEPAModel.predict_action after qwen.build_inputs.
    Tokenization and image processing stay outside this ONNX boundary.
    """

    def __init__(self, vla_model: nn.Module) -> None:
        super().__init__()
        self.vla_model = vla_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        qwen_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        embodied_mask = input_ids == self.vla_model.embodied_action_token_id
        embodied_indices = embodied_mask.nonzero(as_tuple=True)
        last_hidden = self.vla_model._qwen_last_decoder_hidden(qwen_inputs)
        batch_size, _, hidden_size = last_hidden.shape
        return last_hidden[embodied_indices[0], embodied_indices[1], :].view(
            batch_size, -1, hidden_size
        )



def force_eager_attention(module: nn.Module) -> None:
    for submodule in module.modules():
        config = getattr(submodule, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"


def load_sample(sample_path: str, device: str) -> dict[str, Any]:
    sample = torch.load(sample_path, map_location="cpu")
    qwen_inputs = sample["qwen_inputs"]
    for key, value in qwen_inputs.items():
        if isinstance(value, torch.Tensor):
            qwen_inputs[key] = value.to(device)
    if isinstance(sample.get("conditioning_tokens"), torch.Tensor):
        sample["conditioning_tokens"] = sample["conditioning_tokens"].to(device)
    return sample


def sample_inputs(sample: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    qwen_inputs = sample["qwen_inputs"]
    return (
        qwen_inputs["input_ids"],
        qwen_inputs["attention_mask"],
        qwen_inputs["mm_token_type_ids"],
        qwen_inputs["pixel_values"],
        qwen_inputs["image_grid_thw"],
    )


def diff_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().cpu().float().numpy()
    cand = candidate.detach().cpu().float().numpy()
    diff = np.abs(ref - cand)
    return {
        "reference": tensor_summary(reference),
        "candidate": tensor_summary(candidate),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_rtol_1e-4_atol_1e-4": bool(np.allclose(ref, cand, rtol=1e-4, atol=1e-4)),
    }


def export_onnx(model: nn.Module, inputs: tuple[torch.Tensor, ...], onnx_path: Path, opset: int) -> None:
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        ],
        output_names=["conditioning_tokens"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--force-eager-attention", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(37)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "export_context_encoder_report.json"
    onnx_path = out_dir / "vla_jepa_context_encoder.onnx"

    sample = load_sample(args.sample, args.device)
    inputs = sample_inputs(sample)

    policy = load_policy(args.model, args.device, disable_world_model=True)
    if args.force_eager_attention:
        force_eager_attention(policy.model)
    policy.eval()
    wrapper = ContextEncoderFromQwenInputs(policy.model).eval().to(args.device)

    with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
        output = wrapper(*inputs)

    reference = sample["conditioning_tokens"]
    report: dict[str, Any] = {
        "model": args.model,
        "sample": args.sample,
        "device": args.device,
        "force_eager_attention": args.force_eager_attention,
        "inputs": {
            "input_ids": tensor_summary(inputs[0]),
            "attention_mask": tensor_summary(inputs[1]),
            "mm_token_type_ids": tensor_summary(inputs[2]),
            "pixel_values": tensor_summary(inputs[3]),
            "image_grid_thw": tensor_summary(inputs[4]),
        },
        "pytorch_compare": diff_summary(reference, output),
    }

    if args.export_onnx:
        try:
            export_onnx(wrapper, inputs, onnx_path, args.opset)
        except Exception as exc:  # noqa: BLE001 - export errors are part of route discovery.
            report["onnx_export"] = "failed"
            report["onnx_export_error_type"] = type(exc).__name__
            report["onnx_export_error"] = str(exc)
        else:
            report["onnx_export"] = "ok"
            report["onnx_path"] = str(onnx_path)
            report["onnx_size_mb"] = onnx_path.stat().st_size / 1024 / 1024
    else:
        report["onnx_export"] = "skipped; pass --export-onnx to attempt export"

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
