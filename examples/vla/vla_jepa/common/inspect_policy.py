# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Inspect VLA-JEPA inference boundaries and tensor shapes.

This script loads a local LeRobot VLA-JEPA checkpoint, runs one action
prediction on a synthetic batch matching the checkpoint config, and records the
shape/dtype/device at the main inference boundaries:

- LeRobot batch -> native VLA-JEPA examples
- Qwen3-VL input packing
- Qwen3-VL last decoder hidden state
- embodied action tokens
- action head diffusion step inputs/outputs
- final action chunk and selected action

It intentionally disables the training-only world model by default because
`predict_action` does not use it.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import PreTrainedConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

from common.paths import DEFAULT_MODEL, docs_str, set_default_libero_config_path


DEFAULT_REPORT = docs_str("vla_jepa_inspect_report.md")


class ShapeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def add(self, name: str, value: Any) -> None:
        self.events.append((name, describe(value)))

    def add_text(self, name: str, value: str) -> None:
        self.events.append((name, value))

    def print(self) -> None:
        for name, value in self.events:
            print(f"\n## {name}")
            print(format_desc(value))

    def to_markdown(self) -> str:
        lines = ["# VLA-JEPA Inspect Report", ""]
        for name, value in self.events:
            lines.append(f"## {name}")
            lines.append("")
            lines.append("```text")
            lines.append(format_desc(value))
            lines.append("```")
            lines.append("")
        return "\n".join(lines)


def describe(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return type(value).__name__
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Image.Image):
        return {"type": "PIL.Image", "size": list(value.size), "mode": value.mode}
    if isinstance(value, Mapping):
        return {str(k): describe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, tuple):
        return {f"tuple[{i}]": describe(v, depth + 1) for i, v in enumerate(value)}
    if isinstance(value, list):
        if len(value) == 0:
            return []
        if len(value) <= 4:
            return [describe(v, depth + 1) for v in value]
        return {
            "type": "list",
            "len": len(value),
            "first": describe(value[0], depth + 1),
            "last": describe(value[-1], depth + 1),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def format_desc(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def make_dummy_batch(config: Any, device: str) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    generator = torch.Generator(device="cpu").manual_seed(7)

    for key, feature in config.input_features.items():
        shape = tuple(feature.shape)
        if key.startswith("observation.images"):
            # VLA-JEPA expects image tensors in [0, 1], C,H,W.
            batch[key] = torch.rand((1, *shape), generator=generator, dtype=torch.float32).to(device)
        elif key == "observation.state":
            batch[key] = torch.zeros((1, *shape), dtype=torch.float32, device=device)

    batch["task"] = ["Pick up the object and place it into the target area."]
    return batch


def patch_methods(policy: VLAJEPAPolicy, recorder: ShapeRecorder) -> None:
    original_prepare = policy._prepare_model_inputs

    def prepare_wrapper(self: VLAJEPAPolicy, batch: dict[str, torch.Tensor]) -> list[dict]:
        recorder.add("LeRobot batch input", batch)
        examples = original_prepare(batch)
        compact = []
        for ex in examples:
            compact.append({k: v for k, v in ex.items() if k in {"image", "video", "lang", "state", "action"}})
        recorder.add("Native examples after _prepare_model_inputs", compact)
        return examples

    policy._prepare_model_inputs = MethodType(prepare_wrapper, policy)

    original_build_inputs = policy.model.qwen.build_inputs

    def build_inputs_wrapper(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        recorder.add("Qwen3VLInterface.build_inputs args", {"args": args[1:], "kwargs": kwargs})
        out = original_build_inputs(*args, **kwargs)
        recorder.add("Qwen3VLInterface.build_inputs output", out)
        return out

    policy.model.qwen.build_inputs = build_inputs_wrapper

    original_last_hidden = policy.model._qwen_last_decoder_hidden

    def last_hidden_wrapper(qwen_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = original_last_hidden(qwen_inputs)
        recorder.add("Qwen last decoder hidden", out)
        return out

    policy.model._qwen_last_decoder_hidden = last_hidden_wrapper

    action_head = policy.model.action_model
    original_build_action_inputs = action_head._build_inputs

    def action_build_inputs_wrapper(conditioning_tokens, actions, state, timesteps):
        recorder.add(
            "ActionHead._build_inputs input",
            {
                "conditioning_tokens": conditioning_tokens,
                "actions": actions,
                "state": state,
                "timesteps": timesteps,
            },
        )
        out = original_build_action_inputs(conditioning_tokens, actions, state, timesteps)
        recorder.add("ActionHead._build_inputs output hidden_states", out)
        return out

    action_head._build_inputs = action_build_inputs_wrapper

    original_predict_action = action_head.predict_action

    def predict_action_wrapper(conditioning_tokens, state=None):
        recorder.add("ActionHead.predict_action input", {"conditioning_tokens": conditioning_tokens, "state": state})
        out = original_predict_action(conditioning_tokens, state)
        recorder.add("ActionHead.predict_action output action chunk", out)
        return out

    action_head.predict_action = predict_action_wrapper


def add_module_hooks(policy: VLAJEPAPolicy, recorder: ShapeRecorder) -> list[Any]:
    names = {
        "model.qwen.model.model.language_model.layers.-1": policy.model.qwen.model.model.language_model.layers[-1],
        "model.action_model.model": policy.model.action_model.model,
        "model.action_model.action_encoder": policy.model.action_model.action_encoder,
        "model.action_model.state_encoder": policy.model.action_model.state_encoder,
        "model.action_model.action_decoder": policy.model.action_model.action_decoder,
    }
    handles = []
    for name, module in names.items():
        if module is None:
            continue

        def hook(_module, inputs, output, *, _name=name):
            recorder.add(f"module hook: {_name}", {"inputs": inputs, "output": output})

        handles.append(module.register_forward_hook(hook))
    return handles


def load_policy(model_path: str, device: str, disable_world_model: bool) -> VLAJEPAPolicy:
    config = PreTrainedConfig.from_pretrained(model_path, local_files_only=True)
    config.device = device
    if disable_world_model and hasattr(config, "enable_world_model"):
        config.enable_world_model = False
    return VLAJEPAPolicy.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        strict=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--keep-world-model", action="store_true")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")

    recorder = ShapeRecorder()
    recorder.add_text("Inspect config", f"model={args.model}\ndevice={args.device}\nworld_model_enabled={args.keep_world_model}")

    torch.manual_seed(7)
    policy = load_policy(args.model, args.device, disable_world_model=not args.keep_world_model)
    policy.eval()
    policy.reset()

    recorder.add_text("Policy class", policy.__class__.__name__)
    recorder.add(
        "Policy config summary",
        {
            "input_features": policy.config.input_features,
            "output_features": policy.config.output_features,
            "chunk_size": policy.config.chunk_size,
            "n_action_steps": policy.config.n_action_steps,
            "action_dim": policy.config.action_dim,
            "state_dim": policy.config.state_dim,
            "num_inference_timesteps": policy.config.num_inference_timesteps,
            "num_embodied_action_tokens_per_instruction": policy.config.num_embodied_action_tokens_per_instruction,
            "action_model_type": policy.config.action_model_type,
            "enable_world_model": policy.config.enable_world_model,
        },
    )
    recorder.add_text(
        "Key module classes",
        "\n".join(
            [
                f"policy.model.qwen: {policy.model.qwen.__class__.__name__}",
                f"policy.model.qwen.model: {policy.model.qwen.model.__class__.__name__}",
                f"policy.model.action_model: {policy.model.action_model.__class__.__name__}",
                f"policy.model.action_model.model: {policy.model.action_model.model.__class__.__name__}",
            ]
        ),
    )

    patch_methods(policy, recorder)
    handles = add_module_hooks(policy, recorder)

    batch = make_dummy_batch(policy.config, args.device)
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(batch)
        recorder.add("policy.predict_action_chunk output", action_chunk)
        policy.reset()
        selected_action = policy.select_action(batch)
        recorder.add("policy.select_action output", selected_action)

    for handle in handles:
        handle.remove()

    recorder.print()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(recorder.to_markdown())
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
