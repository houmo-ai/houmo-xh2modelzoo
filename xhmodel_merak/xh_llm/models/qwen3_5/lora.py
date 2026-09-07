"""Keep-LoRA support shared by Qwen3.5/Qwen3.6 dense and MoE exports."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from xhmodel_merak.configuration_utils import BaseConfig

from ...types import ExportData, VLLMModelMeta


_LORA_WEIGHT_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")
_VISUAL_PATH_PARTS = {
    "visual",
    "vision",
    "vision_encoder",
    "vision_model",
    "vision_tower",
}


class XHQwen3_5LoRAConfig(BaseConfig):  # noqa: N801
    """Configuration for exporting one static HMONNX pair per PEFT adapter."""

    def __init__(
        self,
        *,
        path: list[str] | tuple[str, ...],
        w_schema: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if "mode" in kwargs:
            raise ValueError(
                "Qwen3.5 LoRA no longer accepts 'mode'; only keep-LoRA export is supported. "
                "Merge an adapter into the HF model externally when merged export is needed."
            )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported Qwen3.5 LoRA config fields: {names}")
        if not isinstance(path, (list, tuple)) or not path:
            raise ValueError("Qwen3.5 lora.path must be a non-empty list of adapter directories")
        if any(not isinstance(adapter_path, str) or not adapter_path.strip() for adapter_path in path):
            raise ValueError("Every Qwen3.5 lora.path entry must be a non-empty string")

        adapter_names = [Path(adapter_path.rstrip("/\\")).name for adapter_path in path]
        if any(name in {"", ".", ".."} for name in adapter_names):
            raise ValueError("Every Qwen3.5 LoRA path must have a safe final directory name")
        duplicate_names = sorted({name for name in adapter_names if adapter_names.count(name) > 1})
        if duplicate_names:
            raise ValueError(
                "Qwen3.5 LoRA output directory names must be unique; duplicate final path "
                f"components: {duplicate_names}"
            )

        if w_schema is not None and not isinstance(w_schema, Mapping):
            raise TypeError("Qwen3.5 lora.w_schema must be a mapping when provided")
        normalized_w_schema = copy.deepcopy(dict(w_schema)) if w_schema else None
        if normalized_w_schema and "act_schema" in normalized_w_schema:
            raise ValueError(
                "Qwen3.5 lora.w_schema only controls LoRA A/B weights; activation precision "
                "is inherited from export.model.quant_scheme"
            )

        self.path = list(path)
        # None (including YAML ``w_schema: {}``) deliberately means no node
        # override.  Passing an empty schema to xhquant would otherwise select
        # xhquant's own W8 default instead of inheriting the model scheme.
        self.w_schema = normalized_w_schema


@dataclass(frozen=True)
class LoRAWeightPair:
    module_path: str
    a_key: str
    b_key: str
    a_shape: tuple[int, ...]
    b_shape: tuple[int, ...]


@dataclass(frozen=True)
class LoRAAdapterSpec:
    name: str
    source_path: str
    config_path: str
    adapter_config: dict[str, Any]
    weight_map: dict[str, str]
    pairs: tuple[LoRAWeightPair, ...]
    rank: int
    alpha: float

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


def coerce_lora_config(value: Mapping[str, Any] | XHQwen3_5LoRAConfig | None) -> XHQwen3_5LoRAConfig | None:
    if value is None:
        return None
    if isinstance(value, XHQwen3_5LoRAConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Qwen3.5 lora must be a mapping when provided")
    return XHQwen3_5LoRAConfig(**dict(value))


def inspect_lora_adapters(
    config: Mapping[str, Any] | XHQwen3_5LoRAConfig | None,
) -> list[LoRAAdapterSpec]:
    lora_config = coerce_lora_config(config)
    if lora_config is None:
        return []
    return [inspect_lora_adapter(path) for path in lora_config.path]


def inspect_lora_adapter(adapter_path: str) -> LoRAAdapterSpec:
    source_path = Path(adapter_path).expanduser()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Qwen3.5 LoRA adapter directory does not exist: {adapter_path}")
    source_path = source_path.resolve()
    config_path = source_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter_config.json: {config_path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_config, dict):
        raise TypeError(f"PEFT adapter config must be a JSON object: {config_path}")
    _validate_peft_config(adapter_config, config_path)

    rank = int(adapter_config.get("r", 0))
    alpha = float(adapter_config.get("lora_alpha", 0.0))
    if rank <= 0:
        raise ValueError(f"PEFT adapter r must be positive in {config_path}, got {rank}")
    if alpha <= 0:
        raise ValueError(f"PEFT adapter lora_alpha must be positive in {config_path}, got {alpha}")

    weight_map, tensor_shapes = _inspect_adapter_weights(source_path)
    pair_items: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = defaultdict(dict)
    unsupported_keys: list[str] = []
    for key, shape in tensor_shapes.items():
        match = _LORA_WEIGHT_RE.match(key)
        if match is None:
            unsupported_keys.append(key)
            continue
        module_path = match.group("module")
        if _is_visual_module_path(module_path):
            raise ValueError(
                "Qwen3.5 keep-LoRA does not support ViT/visual adapters; "
                f"found visual tensor {key!r} in {source_path}"
            )
        side = match.group("side")
        if side in pair_items[module_path]:
            raise ValueError(f"Duplicate LoRA {side} tensor for module {module_path!r} in {source_path}")
        pair_items[module_path][side] = (key, shape)

    if unsupported_keys:
        preview = ", ".join(unsupported_keys[:3])
        raise ValueError(f"Unsupported non-LoRA tensors in PEFT adapter {source_path}: {preview}")
    if not pair_items:
        raise ValueError(f"No LoRA A/B tensors found in adapter {source_path}")

    pairs: list[LoRAWeightPair] = []
    for module_path, sides in sorted(pair_items.items()):
        if set(sides) != {"A", "B"}:
            raise ValueError(
                f"Incomplete LoRA tensor pair for module {module_path!r} in {source_path}; "
                f"found sides {sorted(sides)}"
            )
        a_key, a_shape = sides["A"]
        b_key, b_shape = sides["B"]
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"LoRA A/B tensors must be 2-D for module {module_path!r}, got {a_shape} and {b_shape}")
        has_standard_rank = a_shape[0] == rank and b_shape[1] == rank
        has_transposed_rank = a_shape[1] == rank and b_shape[0] == rank
        if not (has_standard_rank or has_transposed_rank):
            raise ValueError(
                f"LoRA tensor rank does not match adapter r={rank} for module {module_path!r}: "
                f"A={a_shape}, B={b_shape}"
            )
        pairs.append(LoRAWeightPair(module_path, a_key, b_key, a_shape, b_shape))

    return LoRAAdapterSpec(
        name=Path(adapter_path.rstrip("/\\")).name,
        source_path=str(source_path),
        config_path=str(config_path),
        adapter_config=adapter_config,
        weight_map=weight_map,
        pairs=tuple(pairs),
        rank=rank,
        alpha=alpha,
    )


def attach_lora_buffers(model: nn.Module, adapter: LoRAAdapterSpec) -> int:
    """Attach a validated adapter to the corresponding original Linear modules."""

    tensor_by_key = _load_adapter_tensors(adapter)
    linear_modules = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    registered = 0
    resolved_names: set[str] = set()
    for pair in adapter.pairs:
        linear_name, linear = _resolve_linear(pair.module_path, pair, linear_modules)
        if linear_name in resolved_names:
            raise ValueError(
                f"Multiple LoRA tensor pairs resolve to the same Linear {linear_name!r} "
                f"for adapter {adapter.source_path}"
            )
        resolved_names.add(linear_name)
        if hasattr(linear, "weight_lora_a") or hasattr(linear, "weight_lora_b"):
            raise ValueError(f"Linear {linear_name!r} already has LoRA buffers")

        lora_a, lora_b = _orient_lora_weights(
            tensor_by_key[pair.a_key],
            tensor_by_key[pair.b_key],
            tuple(linear.weight.shape),
            linear_name,
        )
        # These are frontend-only transport buffers.  The graph rewrite turns
        # them into LoRALinear parameters, so they should not remain as extra
        # state on the base Linear in exported checkpoints.
        linear.register_buffer("weight_lora_a", lora_a, persistent=False)
        linear.register_buffer("weight_lora_b", lora_b, persistent=False)
        registered += 1

    if registered != len(adapter.pairs):
        raise RuntimeError(f"Attached {registered} LoRA pairs but adapter contains {len(adapter.pairs)} pairs")
    return registered


def apply_lora_to_frontend(
    frontend_model: Any,
    adapter: LoRAAdapterSpec,
    w_schema: Mapping[str, Any] | None,
) -> None:
    """Rewrite both target graphs to ``src + B(A(X)) * (alpha / rank)``."""

    from ...lora_layer import LoRALinear, apply_lora_to_linear

    expected_lora_linears = len(adapter.pairs) * 2
    for mode in ("prefill", "decode"):
        graph_module = getattr(frontend_model, mode)
        apply_lora_to_linear(
            graph_module,
            [],
            lora_scale=adapter.alpha,
            runtime_mask=False,
        )
        lora_nodes = []
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            module = graph_module.get_submodule(str(node.target))
            if not isinstance(module, LoRALinear):
                continue
            lora_nodes.append(node)
            # Always create a node entry.  On externally quantized models,
            # xhquant supplies a W8A8 fallback for modules without
            # ``quant_weight``; an empty explicit entry suppresses that
            # fallback and lets both W/A inherit the main quant scheme.
            quant_config = copy.deepcopy(node.meta.get("quant_config", {}))
            if w_schema is not None:
                quant_config["w_schema"] = copy.deepcopy(dict(w_schema))
            node.meta["quant_config"] = quant_config
        if len(lora_nodes) != expected_lora_linears:
            raise RuntimeError(
                f"Qwen3.5 {mode} graph contains {len(lora_nodes)} LoRA A/B Linear nodes; "
                f"expected {expected_lora_linears} for adapter {adapter.name!r}"
            )

    # Qwen3.5 normally shares base Linear modules between prefill and decode.
    # Preserve that property for the newly inserted A/B modules as well.
    prefill_lora_modules = {
        name: module for name, module in frontend_model.prefill.named_modules() if isinstance(module, LoRALinear)
    }
    for name, module in list(frontend_model.decode.named_modules()):
        if isinstance(module, LoRALinear) and name in prefill_lora_modules:
            frontend_model.decode.set_submodule(name, prefill_lora_modules[name])


def finalize_lora_metadata(
    meta_info: VLLMModelMeta,
    exported_info: ExportData,
    exported_adapters: list[tuple[LoRAAdapterSpec, ExportData]],
    lora_config: XHQwen3_5LoRAConfig,
) -> None:
    """Write one complete, loadable model view for every exported adapter."""

    root_dir = Path(exported_info.exported_dir)
    root_entries = []
    for adapter, adapter_export in exported_adapters:
        graph_meta = adapter_export.meta
        child_meta = copy.deepcopy(meta_info)
        for field in (
            "prefill_hmonnx",
            "prefill_hmonnx_md5",
            "decode_hmonnx",
            "decode_hmonnx_md5",
        ):
            setattr(child_meta, field, getattr(graph_meta, field))

        adapter_dir = Path(adapter_export.exported_dir)
        _materialize_lora_shared_artifacts(child_meta, root_dir, adapter_dir)
        child_meta.prefill_onnx = child_meta.prefill_hmonnx
        child_meta.decode_onnx = child_meta.decode_hmonnx
        child_meta.token_embedding_file = child_meta.quant_embedding
        child_meta.active_lora = {
            "name": adapter.name,
            "source_path": adapter.source_path,
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "scale": adapter.scale,
            "target_linear_count": len(adapter.pairs),
            "w_schema": copy.deepcopy(lora_config.w_schema),
        }
        child_model_config = getattr(child_meta, "model_config", None)
        child_lora_config = getattr(child_model_config, "lora", None)
        if child_lora_config is not None:
            child_lora_config.path = [adapter.source_path]

        with (adapter_dir / "adapter_config.json").open("w", encoding="utf-8") as fout:
            json.dump(adapter.adapter_config, fout, ensure_ascii=False, indent=2)
        with (adapter_dir / "golden_meta_info.json").open("w", encoding="utf-8") as fout:
            json.dump(child_meta.to_dict(), fout, ensure_ascii=False, indent=4)

        relative_dir = adapter_dir.relative_to(root_dir)
        root_entries.append(
            {
                "name": adapter.name,
                "source_path": adapter.source_path,
                "meta_file": str((relative_dir / "golden_meta_info.json").as_posix()),
                "prefill_hmonnx": str((relative_dir / graph_meta.prefill_hmonnx).as_posix()),
                "decode_hmonnx": str((relative_dir / graph_meta.decode_hmonnx).as_posix()),
                "rank": adapter.rank,
                "alpha": adapter.alpha,
                "scale": adapter.scale,
                "target_linear_count": len(adapter.pairs),
                "w_schema": copy.deepcopy(lora_config.w_schema),
            }
        )

    meta_info.lora_adapters = root_entries


def _materialize_lora_shared_artifacts(
    meta_info: VLLMModelMeta,
    root_dir: Path,
    adapter_dir: Path,
) -> None:
    """Mirror shared artifacts under an adapter without duplicating model weights."""

    hf_config = _relative_export_path(meta_info.hf_config, "hf_config")
    _copy_artifact_tree(root_dir / hf_config, adapter_dir / hf_config)

    quant_embedding = _relative_export_path(meta_info.quant_embedding, "quant_embedding")
    _symlink_artifact_file(root_dir / quant_embedding, adapter_dir / quant_embedding)

    component_dirs = _shared_hmonnx_component_dirs(meta_info)
    for component_dir in component_dirs:
        _mirror_artifact_tree(root_dir / component_dir, adapter_dir / component_dir)

    # Flat visual_m* stages share a root-level manifest outside their trees.
    # Only link it separately when it was not already mirrored with a component.
    gear_manifest = getattr(getattr(meta_info, "visual_config", None), "gear_manifest", None)
    if gear_manifest:
        manifest = _relative_export_path(gear_manifest, "gear_manifest")
        if not any(manifest.is_relative_to(component_dir) for component_dir in component_dirs):
            _symlink_artifact_file(root_dir / manifest, adapter_dir / manifest)


def _shared_hmonnx_component_dirs(meta_info: VLLMModelMeta) -> list[Path]:
    target_keys = {"prefill_hmonnx", "decode_hmonnx", "prefill_onnx", "decode_onnx"}
    component_dirs: set[Path] = set()

    def _visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if (
                    isinstance(child, str)
                    and key not in target_keys
                    and (key in {"hmonnx", "onnx"} or key.endswith("_hmonnx") or key.endswith("_onnx"))
                ):
                    artifact = _relative_export_path(child, key)
                    if len(artifact.parts) < 2:
                        raise ValueError(f"Shared HMONNX artifact must be inside a component directory: {child}")
                    component_dir = Path(artifact.parts[0])
                    if component_dir.name not in {"prefill", "decode"}:
                        component_dirs.add(component_dir)
                else:
                    _visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                _visit(child)

    _visit(meta_info.to_dict())
    return sorted(component_dirs, key=lambda path: path.as_posix())


def _relative_export_path(value: str | None, field_name: str) -> Path:
    if not value:
        raise ValueError(f"LoRA child metadata requires a non-empty {field_name} path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"LoRA child artifact path must be root-relative: {field_name}={value!r}")
    return path


def _copy_artifact_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Shared artifact directory does not exist: {source}")
    if os.path.lexists(destination):
        raise FileExistsError(f"LoRA child artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False)


def _mirror_artifact_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Shared model component directory does not exist: {source}")
    if os.path.lexists(destination):
        raise FileExistsError(f"LoRA child component already exists: {destination}")
    destination.mkdir(parents=True)

    def _mirror_directory(source_dir: Path, destination_dir: Path) -> None:
        for source_path in sorted(source_dir.iterdir(), key=lambda path: path.name):
            if source_path.is_dir():
                if source_path.name.startswith("step_"):
                    continue
                child_dir = destination_dir / source_path.name
                child_dir.mkdir()
                _mirror_directory(source_path, child_dir)
            else:
                _symlink_artifact_file(source_path, destination_dir / source_path.name)

    _mirror_directory(source, destination)


def _symlink_artifact_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Shared artifact file does not exist: {source}")
    if os.path.lexists(destination):
        raise FileExistsError(f"LoRA child artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(Path(os.path.relpath(source, destination.parent)))


def _validate_peft_config(config: Mapping[str, Any], config_path: Path) -> None:
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Only PEFT LoRA adapters are supported: {config_path}")
    unsupported_truthy = (
        "alpha_pattern",
        "rank_pattern",
        "modules_to_save",
        "target_parameters",
        "layer_replication",
        "use_dora",
        "use_qalora",
        "use_rslora",
        "lora_bias",
        "fan_in_fan_out",
    )
    enabled = [name for name in unsupported_truthy if config.get(name)]
    if enabled:
        raise ValueError(f"Unsupported PEFT LoRA options in {config_path}: {', '.join(enabled)}")
    if str(config.get("bias", "none")).lower() != "none":
        raise ValueError(f"PEFT LoRA bias must be 'none' for keep-LoRA export: {config_path}")


def _inspect_adapter_weights(adapter_path: Path) -> tuple[dict[str, str], dict[str, tuple[int, ...]]]:
    single_file = adapter_path / "adapter_model.safetensors"
    index_file = adapter_path / "adapter_model.safetensors.index.json"
    bin_file = adapter_path / "adapter_model.bin"

    if single_file.is_file():
        file_paths = [single_file]
        indexed_weight_map = None
    elif index_file.is_file():
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
        indexed_weight_map = index_data.get("weight_map")
        if not isinstance(indexed_weight_map, dict) or not indexed_weight_map:
            raise ValueError(f"Invalid adapter safetensors index: {index_file}")
        file_paths = sorted({adapter_path / value for value in indexed_weight_map.values()})
    elif bin_file.is_file():
        state_dict = torch.load(str(bin_file), map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"PEFT adapter_model.bin must contain a state dict: {bin_file}")
        shapes = {key: tuple(value.shape) for key, value in state_dict.items() if torch.is_tensor(value)}
        return {key: str(bin_file) for key in shapes}, shapes
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors, sharded index, or adapter_model.bin found in {adapter_path}"
        )

    from safetensors import safe_open

    weight_map: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for file_path in file_paths:
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing LoRA safetensors shard: {file_path}")
        with safe_open(str(file_path), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in shapes:
                    raise ValueError(f"Duplicate tensor {key!r} across LoRA shards in {adapter_path}")
                shapes[key] = tuple(reader.get_slice(key).get_shape())
                weight_map[key] = str(file_path)
    if indexed_weight_map is not None:
        if set(indexed_weight_map) != set(weight_map):
            raise ValueError(f"LoRA safetensors index does not match shard contents: {index_file}")
    return weight_map, shapes


def _load_adapter_tensors(adapter: LoRAAdapterSpec) -> dict[str, torch.Tensor]:
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, file_path in adapter.weight_map.items():
        keys_by_file[file_path].append(key)

    tensors: dict[str, torch.Tensor] = {}
    for file_path, keys in keys_by_file.items():
        if file_path.endswith(".bin"):
            state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
            tensors.update({key: state_dict[key] for key in keys})
            continue
        from safetensors import safe_open

        with safe_open(file_path, framework="pt", device="cpu") as reader:
            tensors.update({key: reader.get_tensor(key) for key in keys})
    return tensors


def _module_path_candidates(module_path: str) -> list[str]:
    candidates = [module_path]
    prefixes = (
        "base_model.model.",
        "base_model.",
    )
    changed = True
    while changed:
        changed = False
        for value in list(candidates):
            for prefix in prefixes:
                if value.startswith(prefix):
                    stripped = value[len(prefix) :]
                    if stripped not in candidates:
                        candidates.append(stripped)
                        changed = True
    for value in list(candidates):
        stripped = value
        for prefix in ("model.", "language_model."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                if stripped not in candidates:
                    candidates.append(stripped)
    return candidates


def _resolve_linear(
    module_path: str,
    pair: LoRAWeightPair,
    linear_modules: Mapping[str, nn.Linear],
) -> tuple[str, nn.Linear]:
    candidates = _module_path_candidates(module_path)
    exact = [(name, linear_modules[name]) for name in candidates if name in linear_modules]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        names = [name for name, _ in exact]
        raise ValueError(f"Ambiguous exact Linear matches for LoRA module {module_path!r}: {names}")

    suffix_matches: dict[str, nn.Linear] = {}
    for candidate in candidates:
        if candidate.count(".") < 2:
            continue
        for name, module in linear_modules.items():
            if name.endswith(f".{candidate}") or candidate.endswith(f".{name}"):
                suffix_matches[name] = module
    if len(suffix_matches) == 1:
        return next(iter(suffix_matches.items()))
    if not suffix_matches:
        raise ValueError(
            f"LoRA target module {module_path!r} was not found in the Qwen3.5 language model "
            f"(A={pair.a_shape}, B={pair.b_shape})"
        )
    raise ValueError(
        f"LoRA target module {module_path!r} matches multiple Qwen3.5 Linear modules: " f"{sorted(suffix_matches)}"
    )


def _orient_lora_weights(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    linear_shape: tuple[int, ...],
    linear_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (lora_b.shape[0], lora_a.shape[1]) == linear_shape and lora_a.shape[0] == lora_b.shape[1]:
        return lora_a, lora_b
    if (lora_a.shape[0], lora_b.shape[1]) == linear_shape and lora_a.shape[1] == lora_b.shape[0]:
        return lora_a.T.contiguous(), lora_b.T.contiguous()
    raise ValueError(
        f"LoRA shapes for {linear_name!r} do not match base Linear {linear_shape}: "
        f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
    )


def _is_visual_module_path(module_path: str) -> bool:
    return any(part.lower() in _VISUAL_PATH_PARTS for part in module_path.split("."))
