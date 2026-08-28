"""Low-memory, one-MoE-at-a-time Laguna HMONNX export support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch.nn as nn

from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...big_hf_model_helper import BigHFModelExportHelper
from .float_checkpoint_compat import (
    LagunaSplitExperts,
    fuse_split_experts,
    split_expert_checkpoint_loader,
)


def checkpoint_router_bias_layout(model_dir: str | Path) -> str:
    """Return the router-bias parent used by the concrete safetensors files."""

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_names = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        from safetensors import safe_open

        weight_names = set()
        for safetensor_path in model_dir.glob("*.safetensors"):
            with safe_open(str(safetensor_path), framework="pt") as checkpoint:
                weight_names.update(checkpoint.keys())

    expert_layout = any(name.endswith(".experts.e_score_correction_bias") for name in weight_names)
    gate_layout = any(name.endswith(".gate.e_score_correction_bias") for name in weight_names)
    if expert_layout == gate_layout:
        raise RuntimeError(
            "Laguna checkpoint must use exactly one router-bias layout; "
            f"experts={expert_layout}, gate={gate_layout}"
        )
    return "experts" if expert_layout else "gate"


def move_router_bias_to_checkpoint_layout(model: nn.Module) -> int:
    """Move router correction bias to the official checkpoint key location."""

    moved = 0
    for sparse_block in model.modules():
        experts = getattr(sparse_block, "experts", None)
        gate = getattr(sparse_block, "gate", None)
        if not isinstance(experts, LagunaSplitExperts) or gate is None:
            continue
        bias = gate._parameters.pop("e_score_correction_bias", None)
        if bias is None:
            continue
        if "e_score_correction_bias" in experts._parameters:
            raise RuntimeError("Laguna split experts already own e_score_correction_bias")
        experts.register_parameter("e_score_correction_bias", bias)
        moved += 1
    return moved


def restore_router_bias_from_checkpoint_layout(model: nn.Module) -> int:
    """Restore the runtime gate path after loading one checkpoint MoE block."""

    restored = 0
    for sparse_block in model.modules():
        experts = getattr(sparse_block, "experts", None)
        gate = getattr(sparse_block, "gate", None)
        if not isinstance(experts, LagunaSplitExperts) or gate is None:
            continue
        bias = experts._parameters.pop("e_score_correction_bias", None)
        if bias is None:
            continue
        if "e_score_correction_bias" in gate._parameters:
            raise RuntimeError("Laguna router gate already owns e_score_correction_bias")
        gate.register_parameter("e_score_correction_bias", bias)
        restored += 1
    return restored


def _remove_registered_members(module: nn.Module) -> None:
    module._modules.clear()
    module._parameters.clear()
    module._buffers.clear()
    if any((list(module.children()), list(module.parameters()), list(module.buffers()))):
        raise RuntimeError("Laguna streaming placeholder retained registered tensors")


class _LagunaSparseMoeBlockPlaceHolder(DynamicModule):
    PLACEHOLDER_TYPE_NAME = "LagunaSparseMoeBlock"

    def forward(self, hidden_states):
        return hidden_states

    def _setup(self, *args, **kwargs):
        del args, kwargs
        _remove_registered_members(self)
        return self


def _runtime_aliases(hf_model: Optional[nn.Module], type_name: str) -> dict[type[nn.Module], str]:
    if hf_model is None:
        return {}
    return {
        type(module): type_name
        for module in hf_model.modules()
        if type(module).__name__ == type_name
    }


class LagunaBigHFModel(BigHFModelExportHelper):
    """Stream Laguna's sparse expert blocks while retaining its main graph."""

    PLACEHOLDER_TYPES = ["LagunaSparseMoeBlock"]

    @classmethod
    def _refresh_registrations(cls, hf_model: nn.Module) -> None:
        from ._model import register_wrap_modules

        register_wrap_modules(hf_model)

    @classmethod
    def initialize_process_worker_after_model_load(cls, hf_model: nn.Module) -> None:
        cls._refresh_registrations(hf_model)

    @classmethod
    def initialize_process_worker_after_quantized_preprocess(cls, hf_model: nn.Module) -> None:
        cls._refresh_registrations(hf_model)

    @classmethod
    def register_placeholder(
        cls,
        registry: _DMRegistryCls,
        hf_model: Optional[nn.Module] = None,
    ) -> None:
        mapping = _runtime_aliases(hf_model, "LagunaSparseMoeBlock")
        missing = {module_cls: name for module_cls, name in mapping.items() if module_cls not in registry}
        if missing:
            registry.register_module(missing, dm_class=_LagunaSparseMoeBlockPlaceHolder)

    def prepare_loaded_placeholder_module(self, module: nn.Module, wrap_cfg) -> nn.Module:
        del wrap_cfg
        restored = restore_router_bias_from_checkpoint_layout(module)
        gates_with_bias = sum(
            "e_score_correction_bias" in gate._parameters
            for sparse_block in module.modules()
            if (gate := getattr(sparse_block, "gate", None)) is not None
        )
        if restored not in {0, 1} or gates_with_bias != 1:
            raise RuntimeError(
                "Laguna streamed MoE router-bias restoration was incomplete: "
                f"restored={restored}, runtime_gates={gates_with_bias}"
            )
        with split_expert_checkpoint_loader(str(self._hf_model_dir)) as native_experts_cls:
            converted = fuse_split_experts(module, native_experts_cls)
        if converted != 1:
            raise RuntimeError(
                "Laguna streamed expert fusion was incomplete: "
                f"expected=1, converted={converted}"
            )
        return module


__all__ = [
    "LagunaBigHFModel",
    "checkpoint_router_bias_layout",
    "move_router_bias_to_checkpoint_layout",
    "restore_router_bias_from_checkpoint_layout",
]
