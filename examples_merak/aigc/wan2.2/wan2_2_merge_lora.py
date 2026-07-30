#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Merge Wan2.2 PEFT LoRA adapters into standalone diffusion safetensors.

This helper writes merged checkpoints under ``<model-dir>/merged`` so the
Merak Wan2.2 exporter can pick them up through ``use_resolved_float_loader``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_other_model.models.wan2_2.wan2_2_converter import (  # noqa: E402
    Wan22ConvertConfig,
    Wan22Converter,
)

from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402


LORA_WEIGHT_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")
PREFIXES = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Wan2.2 checkpoint directory.")
    parser.add_argument(
        "--lora-dir",
        action="append",
        required=True,
        help="PEFT LoRA adapter directory. Can be passed multiple times; applied in order.",
    )
    parser.add_argument(
        "--noise-model",
        choices=["low_noise_model", "high_noise_model"],
        action="append",
        default=None,
        help="Which diffusion model(s) to merge. Defaults to high_noise_model.",
    )
    parser.add_argument("--task", default="i2v-A14B")
    parser.add_argument("--output-dir", default=None, help="Defaults to <model-dir>/merged.")
    parser.add_argument("--output-dtype", choices=["keep", "float16", "bfloat16", "float32"], default="keep")
    parser.add_argument("--device", default="cuda:0", help="Used only when the base checkpoint is a directory.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def _load_adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    candidates = [adapter_dir / "adapter_model.safetensors", adapter_dir / "pytorch_lora_weights.safetensors"]
    for candidate in candidates:
        if candidate.is_file():
            return load_file(str(candidate), device="cpu")
    shards = sorted(adapter_dir.glob("*.safetensors"))
    if len(shards) == 1:
        return load_file(str(shards[0]), device="cpu")
    raise FileNotFoundError(f"Cannot find a unique LoRA safetensors file in {adapter_dir}")


def _strip_prefixes(module_path: str) -> str:
    module_path = module_path.replace(".base_layer", "")
    for prefix in PREFIXES:
        if module_path.startswith(prefix):
            return module_path[len(prefix) :]
    return module_path


def _find_target_key(base_state: dict[str, torch.Tensor], module_path: str) -> str:
    stripped = _strip_prefixes(module_path)
    candidates = [f"{stripped}.weight"]
    if stripped.startswith("module."):
        candidates.append(f"{stripped[len('module.'): ]}.weight")
    for key in candidates:
        if key in base_state:
            return key
    suffix = f".{stripped}.weight"
    matches = [key for key in base_state if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    preview = ", ".join(matches[:5]) if matches else "no suffix matches"
    raise KeyError(f"Cannot resolve LoRA target module {module_path!r}; candidates: {preview}")


def _extract_lora_pairs(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = LORA_WEIGHT_RE.match(key)
        if match is None:
            continue
        pairs.setdefault(match.group("module"), {})[match.group("side")] = tensor
    if not pairs:
        raise ValueError("No LoRA A/B tensors found in adapter")
    incomplete = [module for module, sides in pairs.items() if set(sides) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"Incomplete LoRA tensor pairs: {incomplete[:5]}")
    return pairs


def _orient_lora(a: torch.Tensor, b: torch.Tensor, base_shape: tuple[int, ...], module_path: str):
    if len(base_shape) != 2:
        raise ValueError(f"Only 2-D Linear weights are supported for {module_path}, got {base_shape}")
    out_dim, in_dim = base_shape
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"LoRA tensors must be 2-D for {module_path}, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] == in_dim and b.shape[0] == out_dim and a.shape[0] == b.shape[1]:
        return a, b
    if a.shape[0] == in_dim and b.shape[1] == out_dim and a.shape[1] == b.shape[0]:
        return a.T.contiguous(), b.T.contiguous()
    raise ValueError(
        f"LoRA tensor shapes do not match base weight for {module_path}: "
        f"base={base_shape}, A={tuple(a.shape)}, B={tuple(b.shape)}"
    )


def _adapter_scale(adapter_dir: Path, rank: int) -> float:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {config_path}")
    config = _read_json(config_path)
    alpha = float(config.get("lora_alpha", rank))
    return alpha / float(rank)


def _apply_adapter(base_state: dict[str, torch.Tensor], adapter_dir: Path) -> int:
    tensors = _load_adapter_tensors(adapter_dir)
    pairs = _extract_lora_pairs(tensors)
    merged = 0
    for module_path, sides in sorted(pairs.items()):
        target_key = _find_target_key(base_state, module_path)
        base_weight = base_state[target_key]
        a, b = _orient_lora(sides["A"], sides["B"], tuple(base_weight.shape), module_path)
        rank = int(a.shape[0])
        scale = _adapter_scale(adapter_dir, rank)
        delta = torch.matmul(b.float(), a.float()).mul_(scale)
        base_state[target_key] = (base_weight.float() + delta).to(base_weight.dtype)
        merged += 1
    return merged


def _convert_output_dtype(state: dict[str, torch.Tensor], dtype_name: str) -> dict[str, torch.Tensor]:
    if dtype_name == "keep":
        return state
    dtype = getattr(torch, dtype_name)
    return {key: value.to(dtype) if torch.is_floating_point(value) else value for key, value in state.items()}


def _resolve_unmerged_noise_model_path(model_dir: Path, task: str, noise_model: str) -> Path:
    cfg = WAN_CONFIGS[task]
    if noise_model == "low_noise_model":
        subfolder = cfg.low_noise_checkpoint
        role = "low_noise"
    else:
        subfolder = cfg.high_noise_checkpoint
        role = "high_noise"

    split_dir = model_dir / "split_files" / "diffusion_models"
    if split_dir.is_dir():
        split_candidates = sorted(
            path for path in split_dir.iterdir() if path.suffix == ".safetensors" and role in path.name
        )
        if split_candidates:
            return split_candidates[0]
    return model_dir / subfolder


def _load_base_state(model_dir: Path, task: str, device: torch.device, noise_model: str):
    source_path = _resolve_unmerged_noise_model_path(model_dir, task, noise_model)
    if source_path.is_file():
        return load_file(str(source_path), device="cpu"), source_path
    if not source_path.is_dir():
        raise FileNotFoundError(f"Cannot find base {noise_model} checkpoint: {source_path}")
    model = WanModel.from_pretrained(str(model_dir), subfolder=source_path.name)
    model.eval().requires_grad_(False).to(torch.float16).to(device)
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}, source_path


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_dir / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_models = args.noise_model or ["high_noise_model"]
    lora_dirs = [Path(path).expanduser().resolve() for path in args.lora_dir]

    Wan22Converter(str(model_dir), Wan22ConvertConfig(task=args.task, use_resolved_float_loader=True))
    device = torch.device(args.device)
    for noise_model in noise_models:
        base_state, source_path = _load_base_state(model_dir, args.task, device, noise_model)
        total_pairs = 0
        for adapter_dir in lora_dirs:
            total_pairs += _apply_adapter(base_state, adapter_dir)
        base_state = _convert_output_dtype(base_state, args.output_dtype)

        adapter_tag = "_".join(path.name for path in lora_dirs)
        output_file = output_dir / f"wan2_2_{noise_model}_{adapter_tag}_merged.safetensors"
        if output_file.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite to replace: {output_file}")
        save_file(base_state, str(output_file), metadata={"format": "pt", "source": str(source_path)})
        print(f"Merged {total_pairs} LoRA target(s) for {noise_model}: {output_file}")


if __name__ == "__main__":
    main()
