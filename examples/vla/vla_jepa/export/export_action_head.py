# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export, validate, and optionally quantize the VLA-JEPA action head step.

The training ``VLAJEPAActionHead.forward`` samples random noise and returns a
loss, so it is not the right ONNX boundary for inference. This script exports
the deterministic step used inside ``predict_action``:

    conditioning_tokens, actions, state, timesteps
        -> _build_inputs
        -> DiT
        -> action_decoder
        -> pred_velocity

That boundary is the first practical unit for ONNX export and quantization.
"""

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
import onnx
import onnxruntime as ort
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

from common.paths import DEFAULT_MODEL, output_str, set_default_libero_config_path


DEFAULT_OUT_DIR = output_str("action_head")


class ActionHeadStep(nn.Module):
    """One inference denoise step from VLAJEPAActionHead.predict_action."""

    def __init__(self, action_head: nn.Module) -> None:
        super().__init__()
        self.action_head = action_head

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.action_head._build_inputs(conditioning_tokens, actions, state, timesteps)
        pred = self.action_head.model(
            hidden_states=hidden_states,
            encoder_hidden_states=conditioning_tokens,
            timestep=timesteps,
        )
        return self.action_head.action_decoder(pred[:, -actions.shape[1] :])


def load_action_head(model_path: str, device: str) -> nn.Module:
    config = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
    config.device = device
    if hasattr(config, "enable_world_model"):
        config.enable_world_model = False
    policy = VLAJEPAPolicy.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        strict=False,
    )
    action_head = policy.model.action_model
    action_head.eval()
    action_head.float()
    return action_head


def make_dummy_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(11)
    conditioning_tokens = torch.randn(1, 32, 2048, generator=generator, dtype=torch.float32).to(device)
    actions = torch.randn(1, 7, 7, generator=generator, dtype=torch.float32).to(device)
    state = torch.zeros(1, 1, 8, dtype=torch.float32, device=device)
    timesteps = torch.tensor([0], dtype=torch.long, device=device)
    return conditioning_tokens, actions, state, timesteps


def tensor_summary(tensor: torch.Tensor | np.ndarray) -> dict[str, Any]:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
    }


def validate_onnx(onnx_path: Path, inputs: tuple[torch.Tensor, ...], torch_output: torch.Tensor) -> dict[str, Any]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    feed = {name: value.detach().cpu().numpy() for name, value in zip(input_names, inputs, strict=True)}
    ort_output = session.run(None, feed)[0]

    torch_np = torch_output.detach().cpu().float().numpy()
    diff = np.abs(torch_np - ort_output.astype(np.float32))
    return {
        "input_names": input_names,
        "output_names": [item.name for item in session.get_outputs()],
        "torch_output": tensor_summary(torch_output),
        "ort_output": tensor_summary(ort_output),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_rtol_1e-4_atol_1e-4": bool(np.allclose(torch_np, ort_output, rtol=1e-4, atol=1e-4)),
    }


def convert_to_hmonnx(
    onnx_path: Path,
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    quant_type: str,
    simplify: bool,
) -> dict[str, Any]:
    from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config

    quant_config = create_quant_config(
        QuantScheme(
            target_device=DeviceType.XH2a,
            quant_type=quant_type,
        )
    )
    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(onnx_path),
        [item.detach().cpu() for item in inputs],
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config,
        simplify=simplify,
    )
    return {
        "hmonnx_path": str(hmonnx_path),
        "quant_type": quant_type,
        "simplify": simplify,
        "size_mb": hmonnx_path.stat().st_size / 1024 / 1024,
    }


def run_hmonnx_golden(
    hmonnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
    golden_dir: Path,
    device: str,
) -> dict[str, Any]:
    from xhquant.api import HMONNXGoldenInference

    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_path))
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.step = 0
    session.to(device)
    run_inputs = [inputs[0].half(), inputs[1].half(), inputs[2].half(), inputs[3].int()]
    run_inputs = [item.to(device) for item in run_inputs]
    with torch.no_grad():
        output = session(*run_inputs)
    return {
        "golden_dir": str(golden_dir),
        "device": device,
        "output": tensor_summary(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-onnx-export", action="store_true")
    parser.add_argument("--skip-ort", action="store_true")
    parser.add_argument("--convert-hmonnx", action="store_true")
    parser.add_argument("--quant-type", default="w8a8_sefp")
    parser.add_argument("--hmonnx-file", default=None)
    parser.add_argument("--no-simplify-hmonnx", action="store_true")
    parser.add_argument("--run-hmonnx-golden", action="store_true")
    parser.add_argument("--hmonnx-golden-device", default="cpu")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(11)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "vla_jepa_action_head_step.onnx"
    hmonnx_path = Path(args.hmonnx_file) if args.hmonnx_file else out_dir / f"vla_jepa_action_head_step_{args.quant_type}.hmonnx.onnx"
    report_path = out_dir / "export_action_head_report.json"

    inputs = make_dummy_inputs(args.device)
    torch_output = None

    if not args.skip_onnx_export:
        action_head = load_action_head(args.model, args.device)
        model = ActionHeadStep(action_head).eval().to(args.device)
        with torch.no_grad():
            torch_output = model(*inputs)

        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["conditioning_tokens", "actions", "state", "timesteps"],
            output_names=["pred_velocity"],
        )
    elif not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(onnx_model)

    report: dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "opset": args.opset,
        "onnx_path": str(onnx_path),
        "inputs": {
            "conditioning_tokens": tensor_summary(inputs[0]),
            "actions": tensor_summary(inputs[1]),
            "state": tensor_summary(inputs[2]),
            "timesteps": tensor_summary(inputs[3]),
        },
    }
    if torch_output is not None:
        report["torch_output"] = tensor_summary(torch_output)
    if not args.skip_ort and torch_output is not None:
        report["onnxruntime"] = validate_onnx(onnx_path, inputs, torch_output)
    elif not args.skip_ort:
        report["onnxruntime"] = "skipped because --skip-onnx-export did not produce a PyTorch reference output"
    if args.convert_hmonnx:
        report["hmonnx"] = convert_to_hmonnx(
            onnx_path,
            hmonnx_path,
            inputs,
            args.quant_type,
            simplify=not args.no_simplify_hmonnx,
        )
        if args.run_hmonnx_golden:
            report["hmonnx_golden"] = run_hmonnx_golden(
                hmonnx_path,
                inputs,
                out_dir / "golden" / hmonnx_path.stem,
                args.hmonnx_golden_device,
            )

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
