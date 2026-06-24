# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Inspect Cosmos3-Nano model package and planned export boundaries.

This script is intentionally static: it reads configs and safetensors index
files without loading the 30GB transformer weights. The output is meant to
drive Stage 0 export work by fixing component dimensions, weight-key layout,
and the first graph boundaries to implement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


try:
    from common.paths import default_model_root
except ModuleNotFoundError:
    from paths import default_model_root


DEFAULT_MODEL = default_model_root()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def summarize_config(model_dir: Path) -> dict[str, Any]:
    top_config = load_json(model_dir / "config.json")
    transformer_config = load_json(model_dir / "transformer" / "config.json")
    vision_config = load_json(model_dir / "vision_encoder" / "config.json")
    vae_config = load_json(model_dir / "vae" / "config.json")
    sound_config = load_json(model_dir / "sound_tokenizer" / "config.json")
    model_index = load_json(model_dir / "model_index.json")

    text_config = top_config.get("text_config", {})

    return {
        "model_dir": str(model_dir),
        "model_type": top_config.get("model_type"),
        "architectures": top_config.get("architectures"),
        "pipeline": model_index,
        "text": {
            "model_type": text_config.get("model_type"),
            "num_hidden_layers": text_config.get("num_hidden_layers"),
            "hidden_size": text_config.get("hidden_size"),
            "intermediate_size": text_config.get("intermediate_size"),
            "num_attention_heads": text_config.get("num_attention_heads"),
            "num_key_value_heads": text_config.get("num_key_value_heads"),
            "head_dim": text_config.get("head_dim"),
            "vocab_size": text_config.get("vocab_size"),
            "max_position_embeddings": text_config.get("max_position_embeddings"),
            "rope_theta": text_config.get("rope_theta"),
            "rope_scaling": text_config.get("rope_scaling"),
        },
        "transformer": {
            "class": transformer_config.get("_class_name"),
            "num_hidden_layers": transformer_config.get("num_hidden_layers"),
            "hidden_size": transformer_config.get("hidden_size"),
            "intermediate_size": transformer_config.get("intermediate_size"),
            "num_attention_heads": transformer_config.get("num_attention_heads"),
            "num_key_value_heads": transformer_config.get("num_key_value_heads"),
            "head_dim": transformer_config.get("head_dim"),
            "vocab_size": transformer_config.get("vocab_size"),
            "use_moe": transformer_config.get("use_moe"),
            "action_gen": transformer_config.get("action_gen"),
            "sound_gen": transformer_config.get("sound_gen"),
            "latent_channel": transformer_config.get("latent_channel"),
            "patch_latent_dim": transformer_config.get("patch_latent_dim"),
            "action_dim": transformer_config.get("action_dim"),
            "max_action_dim": transformer_config.get("max_action_dim"),
            "sound_dim": transformer_config.get("sound_dim"),
            "position_embedding_type": transformer_config.get("position_embedding_type"),
            "rope_scaling": transformer_config.get("rope_scaling"),
        },
        "vision_encoder": {
            "architectures": vision_config.get("architectures"),
            "model_type": vision_config.get("model_type"),
            "depth": vision_config.get("depth"),
            "hidden_size": vision_config.get("hidden_size"),
            "intermediate_size": vision_config.get("intermediate_size"),
            "num_heads": vision_config.get("num_heads"),
            "out_hidden_size": vision_config.get("out_hidden_size"),
            "patch_size": vision_config.get("patch_size"),
            "temporal_patch_size": vision_config.get("temporal_patch_size"),
            "spatial_merge_size": vision_config.get("spatial_merge_size"),
            "deepstack_visual_indexes": vision_config.get("deepstack_visual_indexes"),
        },
        "vae": {
            "class": vae_config.get("_class_name"),
            "in_channels": vae_config.get("in_channels"),
            "out_channels": vae_config.get("out_channels"),
            "base_dim": vae_config.get("base_dim"),
            "decoder_base_dim": vae_config.get("decoder_base_dim"),
            "dim_mult": vae_config.get("dim_mult"),
            "patch_size": vae_config.get("patch_size"),
            "latents_mean_len": len(vae_config.get("latents_mean") or []),
            "latents_std_len": len(vae_config.get("latents_std") or []),
        },
        "sound_tokenizer": {
            "model_type": sound_config.get("model_type"),
            "sampling_rate": sound_config.get("sampling_rate"),
            "stereo": sound_config.get("stereo"),
            "hop_size": sound_config.get("hop_size"),
            "enc_dim": sound_config.get("enc_dim"),
            "enc_latent_dim": sound_config.get("enc_latent_dim"),
            "dec_dim": sound_config.get("dec_dim"),
            "dec_out_channels": sound_config.get("dec_out_channels"),
            "vocoder_input_dim": sound_config.get("vocoder_input_dim"),
        },
    }


def summarize_weight_index(index_file: Path, model_dir: Path) -> dict[str, Any]:
    if not index_file.exists():
        return {"index_file": rel(index_file, model_dir), "exists": False}

    index = load_json(index_file)
    weight_map = index.get("weight_map", {})
    top_level = Counter(k.split(".")[0] for k in weight_map)
    layer_ids = sorted({int(k.split(".")[1]) for k in weight_map if k.startswith("layers.")})

    per_layer_counts: dict[int, int] = defaultdict(int)
    layer0_parts = Counter()
    attn_parts = Counter()
    for key in weight_map:
        parts = key.split(".")
        if key.startswith("layers."):
            layer_idx = int(parts[1])
            per_layer_counts[layer_idx] += 1
            if layer_idx == 0 and len(parts) >= 4:
                layer0_parts[".".join(parts[2:4])] += 1
            if ".self_attn." in key and len(parts) >= 4:
                attn_parts[parts[3]] += 1

    shard_sizes = Counter(weight_map.values())
    branch_counts = {
        "text_mlp": sum(1 for k in weight_map if ".mlp." in k),
        "moe_gen_mlp": sum(1 for k in weight_map if ".mlp_moe_gen." in k),
        "self_attn": sum(1 for k in weight_map if ".self_attn." in k),
        "generator_added_attn": sum(1 for k in weight_map if ".self_attn.add_" in k or ".self_attn.to_add_out" in k),
        "qk_norm": sum(1 for k in weight_map if ".self_attn.norm_" in k),
        "action": sum(1 for k in weight_map if k.startswith("action_") or ".action_" in k),
        "audio": sum(1 for k in weight_map if k.startswith("audio_") or ".audio_" in k),
        "vae_or_latent_proj": sum(1 for k in weight_map if k.startswith("proj_")),
        "time_embedder": sum(1 for k in weight_map if k.startswith("time_embedder.")),
    }

    return {
        "index_file": rel(index_file, model_dir),
        "exists": True,
        "metadata": index.get("metadata", {}),
        "num_weights": len(weight_map),
        "num_shards": len(shard_sizes),
        "top_level_counts": dict(top_level.most_common()),
        "layer_ids": layer_ids,
        "per_layer_counts": dict(sorted(per_layer_counts.items())),
        "layer0_parts": dict(layer0_parts.most_common()),
        "attention_parts": dict(attn_parts.most_common()),
        "branch_counts": branch_counts,
        "sample_keys": list(weight_map)[:40],
    }


def planned_boundaries() -> dict[str, Any]:
    return {
        "reasoner": {
            "graphs": ["reasoner_prefill_multimodal", "reasoner_decode"],
            "inputs": [
                "input_ids or inputs_embeds",
                "vision embeddings / deepstack features",
                "mRoPE position ids",
                "KV cache",
            ],
            "outputs": ["logits", "updated KV cache"],
            "notes": [
                "Requires a Cosmos3 text/reasoner wrapper because transformer key names differ from standard Qwen3-VL.",
                "Confirm whether pure reasoner path uses only mlp/to_* or also mlp_moe_gen/add_* branches.",
            ],
        },
        "generator": {
            "graphs": ["generator_condition_encoder", "generator_denoiser_step", "generator_vae_decoder"],
            "inputs": ["condition states", "noisy latent z_t", "timestep", "position/modality metadata"],
            "outputs": ["noise/v/x0 prediction", "decoded image/video frames"],
            "notes": [
                "Do not export scheduler loop; export one denoise step.",
                "Keep timestep embedding, final latent head, norm/softmax in higher precision for first quant pass.",
            ],
        },
        "forward_dynamics": {
            "graphs": ["fd_observation_encoder", "fd_action_encoder", "fd_denoiser_step", "fd_observation_decoder"],
            "inputs": ["current observation", "action sequence", "task/text condition", "timestep"],
            "outputs": ["future latent / future observation"],
            "notes": ["Action conditioning should stay FP16 in first pass."],
        },
        "policy": {
            "graphs": ["policy_observation_encoder", "policy_action_denoiser_step", "policy_action_head"],
            "inputs": ["observation", "task condition", "state/proprio", "optional noisy action"],
            "outputs": ["action sequence"],
            "notes": ["Do not quantize action head in first pass; evaluate in closed loop."],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Cosmos3-Nano Inspect Report", ""]
    for section, value in report.items():
        lines.append(f"## {section}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Cosmos3-Nano model directory")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown report path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    report = {
        "config_summary": summarize_config(model_dir),
        "transformer_weight_index": summarize_weight_index(
            model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
            model_dir,
        ),
        "top_level_weight_index": summarize_weight_index(model_dir / "model.safetensors.index.json", model_dir),
        "planned_export_boundaries": planned_boundaries(),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False) if args.json else render_markdown(report)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
