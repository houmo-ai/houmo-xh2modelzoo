"""Create a small layer-prefix checkpoint without changing tensor values."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _keep(name: str, max_layers: int, include_mtp: bool) -> bool:
    match = _LAYER_RE.match(name)
    if match:
        return int(match.group(1)) < max_layers
    if name.startswith("mtp."):
        return include_mtp
    return name in {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }


def reduce_checkpoint(source: Path, output: Path, max_layers: int, include_mtp: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    source_map = source_index["weight_map"]
    selected = {name: shard for name, shard in source_map.items() if _keep(name, max_layers, include_mtp)}
    by_shard: dict[str, list[str]] = {}
    for name, shard in selected.items():
        by_shard.setdefault(shard, []).append(name)

    target_map = {}
    total_size = 0
    target_shard_idx = 0
    for source_shard, names in sorted(by_shard.items()):
        if all(name.startswith("mtp.") for name in names):
            target_name = "missing_tensors_passthrough.safetensors"
            target_path = output / target_name
            if target_path.exists():
                target_path.unlink()
            os.link(source / source_shard, target_path)
            with safe_open(str(target_path), framework="pt") as handle:
                for name in names:
                    total_size += handle.get_slice(name).get_shape()[0] * 0  # size is optional metadata
            target_map.update({name: target_name for name in names})
            continue

        target_shard_idx += 1
        target_name = f"model-reduced-{target_shard_idx:05d}.safetensors"
        with safe_open(str(source / source_shard), framework="pt") as handle:
            tensors = {name: handle.get_tensor(name) for name in sorted(names)}
        save_file(tensors, output / target_name, metadata={"format": "pt"})
        total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        target_map.update({name: target_name for name in names})
        del tensors

    for path in source.iterdir():
        if path.suffix == ".safetensors" or path.name.endswith(".safetensors.index.json"):
            continue
        if path.is_file():
            shutil.copy2(path, output / path.name)

    config = json.loads((output / "config.json").read_text())
    config["num_hidden_layers"] = max_layers
    config["layer_types"] = config.get("layer_types", [])[:max_layers]
    dynamic = config.get("quantization_config", {}).get("dynamic", {})
    config.get("quantization_config", {})["dynamic"] = {
        pattern: value
        for pattern, value in dynamic.items()
        if not pattern.startswith("-:")
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    target_index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(target_map.items()))}
    (output / "model.safetensors.index.json").write_text(json.dumps(target_index, indent=2) + "\n")
    print(f"wrote {len(target_map)} tensors across {target_shard_idx} target shards to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-layers", type=int, required=True)
    parser.add_argument("--include-mtp", action="store_true")
    args = parser.parse_args()
    reduce_checkpoint(args.source, args.output, args.max_layers, args.include_mtp)


if __name__ == "__main__":
    main()
