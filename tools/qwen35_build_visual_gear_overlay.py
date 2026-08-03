#!/usr/bin/env python3
"""Build a Qwen3.5 Merak comparison artifact without copying text weights."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path


INPUT_NAMES = [
    "pixel_values",
    "position_ids",
    "position_weights",
    "rotary_position_ids",
    "attention_mask",
]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def build_overlay(fixed_model_dir: Path, visual_gear_dir: Path, output_dir: Path) -> Path:
    fixed_model_dir = fixed_model_dir.resolve(strict=True)
    visual_gear_dir = visual_gear_dir.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")

    fixed_meta_path = fixed_model_dir / "golden_meta_info.json"
    fixed_merak_path = fixed_model_dir / "merak_config.json"
    manifest_path = visual_gear_dir / "visual_gears.json"
    meta = _read_json(fixed_meta_path)
    merak_config = _read_json(fixed_merak_path)
    manifest = _read_json(manifest_path)
    gears = manifest.get("gears")
    if not isinstance(gears, list) or not gears:
        raise ValueError(f"visual gear manifest has no gears: {manifest_path}")

    hf_config = _read_json(fixed_model_dir / "hf_config" / "config.json")
    vision_config = hf_config.get("vision_config")
    if not isinstance(vision_config, dict):
        raise ValueError("fixed model HF config has no vision_config")

    resolved_gears = []
    for gear in gears:
        resolved = dict(gear)
        graph_path = visual_gear_dir / str(resolved["hmonnx"])
        if not graph_path.is_file():
            raise FileNotFoundError(f"visual gear graph does not exist: {graph_path}")
        resolved["hmonnx"] = str(graph_path.resolve())
        resolved_gears.append(resolved)

    largest = resolved_gears[-1]
    fixed_visual = meta.get("visual_config")
    if not isinstance(fixed_visual, dict):
        raise ValueError("fixed golden metadata has no visual_config")
    model_config = meta.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("fixed golden metadata has no model_config")
    nested_fixed_visual = model_config.get("visual_config")
    if not isinstance(nested_fixed_visual, dict):
        raise ValueError("fixed model_config has no visual_config")
    dynamic_visual = {
        **nested_fixed_visual,
        **fixed_visual,
        "max_size_w": int(fixed_visual.get("image_size_w", 448)),
        "max_size_h": int(fixed_visual.get("image_size_h", 448)),
        "hmonnx": largest["hmonnx"],
        "onnx": None,
        "visual_input_mode": manifest["visual_input_mode"],
        "image_token_capacity": int(largest["image_token_capacity"]),
        "patch_token_capacity": int(largest["patch_token_capacity"]),
        "image_token_gears": [int(gear["image_token_capacity"]) for gear in resolved_gears],
        "gears": resolved_gears,
        "gear_manifest": str(manifest_path.resolve()),
        "routing_policy": manifest["routing_policy"],
        "shared_weight_loader": manifest["shared_weight_loader"],
        "attention_mask_format": manifest["attention_mask_format"],
        "attention_mask_shape": manifest["attention_mask_shape"],
        "attention_mask_operator": manifest["attention_mask_operator"],
        "rotary_position_format": manifest["rotary_position_format"],
        "visual_rope_cache_length": int(manifest["visual_rope_cache_length"]),
        "input_names": INPUT_NAMES,
        "hidden_size": int(vision_config["hidden_size"]),
        "num_heads": int(vision_config["num_heads"]),
        "in_channels": int(vision_config["in_channels"]),
        "num_position_embeddings": int(vision_config["num_position_embeddings"]),
    }
    meta["visual_config"] = dynamic_visual
    model_config["visual_config"] = deepcopy(dynamic_visual)

    output_dir.mkdir(parents=True)
    for source in fixed_model_dir.iterdir():
        if source.name in {"golden_meta_info.json", "merak_config.json"}:
            continue
        os.symlink(source, output_dir / source.name, target_is_directory=source.is_dir())
    (output_dir / "golden_meta_info.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "merak_config.json").write_text(
        json.dumps(merak_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-model-dir", type=Path, required=True)
    parser.add_argument("--visual-gear-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_overlay(args.fixed_model_dir, args.visual_gear_dir, args.output_dir))


if __name__ == "__main__":
    main()
