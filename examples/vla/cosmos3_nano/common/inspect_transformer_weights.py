# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Inspect Cosmos3-Nano transformer safetensors keys and branch shapes.

This script only reads safetensors metadata/header entries. It does not load
full tensor payloads into memory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open


try:
    from common.paths import default_model_root
except ModuleNotFoundError:
    from paths import default_model_root


DEFAULT_MODEL = default_model_root()
_COSMOS3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = _COSMOS3_ROOT / "data" / "transformer_weight_map.json"
DEFAULT_MD = _COSMOS3_ROOT / "data" / "transformer_weight_map.md"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_key(key: str) -> str:
    if key.startswith("layers."):
        if ".self_attn." in key:
            if ".add_" in key or ".to_add_out" in key:
                return "layers.generator_added_attention"
            if ".norm_" in key:
                return "layers.attention_qk_norm"
            return "layers.self_attention"
        if ".mlp_moe_gen." in key:
            return "layers.generator_moe_mlp"
        if ".mlp." in key:
            return "layers.text_mlp"
        if ".input_layernorm" in key or ".post_attention_layernorm" in key:
            return "layers.norm"
        return "layers.other"
    if key.startswith("patch_embedding") or key.startswith("time_embedder") or key.startswith("extra_pos_embed"):
        return "generator.latent_time_position"
    if key.startswith("proj_in") or key.startswith("proj_out"):
        return "generator.latent_projection"
    if key.startswith("context_embedder") or key.startswith("norm_out") or key.startswith("scale_shift_table"):
        return "generator.output_conditioning"
    if key.startswith("action_"):
        return "action"
    if key.startswith("audio_"):
        return "audio"
    if key.startswith("embed_tokens") or key.startswith("lm_head") or key.startswith("norm"):
        return "reasoner.text_head"
    if key.startswith("visual_") or key.startswith("vision_"):
        return "reasoner.visual_bridge"
    return "other"


def tensor_shapes(transformer_dir: Path, weight_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_shard[shard].append(key)

    result: dict[str, dict[str, Any]] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(transformer_dir / shard, framework="pt", device="cpu") as f:
            for key in keys:
                result[key] = {
                    "shape": list(f.get_slice(key).get_shape()),
                    "dtype": str(f.get_slice(key).get_dtype()),
                    "shard": shard,
                    "branch": classify_key(key),
                }
    return result


def summarize_layers(shapes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_layer: dict[int, Counter[str]] = defaultdict(Counter)
    per_layer_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key, info in shapes.items():
        if not key.startswith("layers."):
            continue
        parts = key.split(".")
        layer_idx = int(parts[1])
        branch = info["branch"]
        per_layer[layer_idx][branch] += 1
        if layer_idx in (0, 1, 35) and len(per_layer_sample[layer_idx]) < 48:
            per_layer_sample[layer_idx].append({"key": key, "shape": info["shape"], "dtype": info["dtype"], "branch": branch})
    return {
        "layer_ids": sorted(per_layer),
        "per_layer_branch_counts": {str(k): dict(v) for k, v in sorted(per_layer.items())},
        "sample_layers": {str(k): v for k, v in sorted(per_layer_sample.items())},
    }


def build_report(model_dir: Path) -> dict[str, Any]:
    transformer_dir = model_dir / "transformer"
    config = load_json(transformer_dir / "config.json")
    index = load_json(transformer_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = index["weight_map"]
    shapes = tensor_shapes(transformer_dir, weight_map)

    branch_counts = Counter(info["branch"] for info in shapes.values())
    branch_params: dict[str, int] = defaultdict(int)
    for info in shapes.values():
        n = 1
        for dim in info["shape"]:
            n *= dim
        branch_params[info["branch"]] += n

    top_level = Counter(key.split(".")[0] for key in shapes)
    non_layer_keys = [
        {"key": key, "shape": info["shape"], "dtype": info["dtype"], "branch": info["branch"]}
        for key, info in sorted(shapes.items())
        if not key.startswith("layers.")
    ]

    shape_by_key_sample = [
        {"key": key, "shape": info["shape"], "dtype": info["dtype"], "branch": info["branch"]}
        for key, info in sorted(shapes.items())[:120]
    ]

    return {
        "model_dir": str(model_dir),
        "transformer_dir": str(transformer_dir),
        "config_summary": {
            "class": config.get("_class_name"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "latent_channel": config.get("latent_channel"),
            "patch_latent_dim": config.get("patch_latent_dim"),
            "action_dim": config.get("action_dim"),
            "max_action_dim": config.get("max_action_dim"),
            "sound_dim": config.get("sound_dim"),
            "use_moe": config.get("use_moe"),
            "action_gen": config.get("action_gen"),
            "sound_gen": config.get("sound_gen"),
        },
        "index_metadata": index.get("metadata", {}),
        "num_tensors": len(shapes),
        "num_shards": len(set(weight_map.values())),
        "top_level_counts": dict(top_level.most_common()),
        "branch_counts": dict(branch_counts.most_common()),
        "branch_params": dict(sorted(branch_params.items())),
        "layer_summary": summarize_layers(shapes),
        "non_layer_keys": non_layer_keys,
        "shape_by_key_sample": shape_by_key_sample,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Cosmos3-Nano Transformer Weight Map", ""]
    lines += ["## Summary", "", "```json", json.dumps({k: report[k] for k in ("config_summary", "index_metadata", "num_tensors", "num_shards", "top_level_counts", "branch_counts", "branch_params")}, indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Layer Summary", "", "```json", json.dumps(report["layer_summary"], indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Non Layer Keys", "", "```json", json.dumps(report["non_layer_keys"], indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Shape Sample", "", "```json", json.dumps(report["shape_by_key_sample"], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.model.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_output.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps(report if args.print_json else {"json_output": str(args.json_output), "md_output": str(args.md_output), "num_tensors": report["num_tensors"], "branch_counts": report["branch_counts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
