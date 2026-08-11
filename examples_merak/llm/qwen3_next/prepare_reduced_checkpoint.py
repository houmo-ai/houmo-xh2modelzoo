"""Create a small layer-prefix checkpoint without changing tensor values."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _reduce_config_layers(config: dict, max_layers: int) -> None:
    config["num_hidden_layers"] = max_layers
    if "layer_types" in config:
        config["layer_types"] = config["layer_types"][:max_layers]


def _passthrough_shard_name(source_shard: str) -> str:
    return f"mtp-{Path(source_shard).stem}.safetensors"


def _safetensors_data_size(path: Path, names: list[str]) -> int:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return sum(header[name]["data_offsets"][1] - header[name]["data_offsets"][0] for name in names)


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
        source_path = source / source_shard
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            source_names = set(handle.keys())

        if source_names == set(names) and all(name.startswith("mtp.") for name in names):
            target_name = _passthrough_shard_name(source_shard)
            target_path = output / target_name
            if target_path.exists():
                target_path.unlink()
            _link_or_copy(source_path, target_path)
            total_size += _safetensors_data_size(target_path, names)
            target_map.update({name: target_name for name in names})
            continue

        target_shard_idx += 1
        target_name = f"model-reduced-{target_shard_idx:05d}.safetensors"
        with safe_open(str(source_path), framework="pt") as handle:
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
    _reduce_config_layers(config, max_layers)
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
