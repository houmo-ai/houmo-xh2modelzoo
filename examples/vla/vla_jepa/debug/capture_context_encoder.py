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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export.export_action_head import tensor_summary
from common.inspect_policy import DEFAULT_MODEL, load_policy, make_dummy_batch
from common.paths import output_str, set_default_libero_config_path

DEFAULT_OUT_DIR = output_str("context_encoder")


def detach_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


def summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
        }
    if isinstance(value, Mapping):
        return {key: summarize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) <= 4:
            return [summarize_value(item) for item in value]
        return {"type": "list", "len": len(value), "first": summarize_value(value[0])}
    if isinstance(value, tuple):
        return {f"tuple[{idx}]": summarize_value(item) for idx, item in enumerate(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def capture_context(policy: Any, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    examples = policy._prepare_model_inputs(batch)
    batch_images = [ex["image"] for ex in examples]
    instructions = [ex["lang"] for ex in examples]
    state_np = None
    if "state" in examples[0] and examples[0]["state"] is not None:
        state_np = np.stack([ex["state"] for ex in examples])

    qwen_inputs = policy.model.qwen.build_inputs(
        images=batch_images,
        instructions=instructions,
        action_prompt=policy.model.replace_prompt,
        embodied_prompt=policy.model.embodied_replace_prompt,
    )
    embodied_mask = qwen_inputs["input_ids"] == policy.model.embodied_action_token_id
    embodied_indices = embodied_mask.nonzero(as_tuple=True)

    device_type = next(policy.model.parameters()).device.type
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        last_hidden = policy.model._qwen_last_decoder_hidden(qwen_inputs)
        batch_size, _, hidden_size = last_hidden.shape
        conditioning_tokens = last_hidden[embodied_indices[0], embodied_indices[1], :].view(
            batch_size, -1, hidden_size
        )

    state_tensor = None
    if state_np is not None:
        state_tensor = torch.from_numpy(np.array(state_np)).to(
            device=last_hidden.device, dtype=last_hidden.dtype
        )

    return {
        "prepared_examples_summary": summarize_value(
            [{key: val for key, val in ex.items() if key in {"image", "lang", "state"}} for ex in examples]
        ),
        "qwen_inputs": qwen_inputs,
        "embodied_indices": [item.detach().cpu() for item in embodied_indices],
        "last_hidden": last_hidden,
        "conditioning_tokens": conditioning_tokens,
        "state": state_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-name", default="dummy_sample")
    args = parser.parse_args()

    set_default_libero_config_path()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(31)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / f"{args.sample_name}.pt"
    report_path = out_dir / f"{args.sample_name}_report.json"

    policy = load_policy(args.model, args.device, disable_world_model=True)
    policy.eval()
    policy.reset()

    batch = make_dummy_batch(policy.config, args.device)
    captured = capture_context(policy, batch)
    serializable = detach_to_cpu(captured)
    torch.save(serializable, sample_path)

    report = {
        "model": args.model,
        "device": args.device,
        "sample_path": str(sample_path),
        "batch": summarize_value(batch),
        "prepared_examples": captured["prepared_examples_summary"],
        "qwen_inputs": summarize_value(captured["qwen_inputs"]),
        "embodied_indices": summarize_value(captured["embodied_indices"]),
        "last_hidden": summarize_value(captured["last_hidden"]),
        "conditioning_tokens": summarize_value(captured["conditioning_tokens"]),
        "state": summarize_value(captured["state"]),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
