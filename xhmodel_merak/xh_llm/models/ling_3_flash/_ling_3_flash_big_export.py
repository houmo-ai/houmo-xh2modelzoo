from __future__ import annotations

from typing import Optional

import torch

from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...big_hf_model_helper import BigHFModelExportHelper
from ._compat import patch_ling_remote_code_compatibility


def _remove_registered_members(module: torch.nn.Module) -> dict[str, int]:
    module._modules.clear()
    module._parameters.clear()
    module._buffers.clear()
    counts = {
        "modules": len(list(module.children())),
        "parameters": len(list(module.parameters())),
        "buffers": len(list(module.buffers())),
    }
    if any(counts.values()):
        raise RuntimeError(f"Ling placeholder still owns registered tensors: {counts}")
    return counts


class _LingSparseMoeBlockPlaceHolder(DynamicModule):
    PLACEHOLDER_TYPE_NAME = "LinearBailingMoeV3SparseMoeBlock"

    def forward(self, hidden_states):
        return hidden_states

    def _setup(self, *args, **kwargs):
        del args, kwargs
        self._placeholder_registered_member_counts = _remove_registered_members(self)


def _runtime_aliases(hf_model: Optional[torch.nn.Module], type_name: str) -> dict[type, str]:
    if hf_model is None:
        return {}
    return {
        type(module): type_name
        for module in hf_model.modules()
        if type(module).__name__ == type_name
    }


def _register_if_missing(registry: _DMRegistryCls, mapping: dict[type, str], wrapper) -> None:
    missing = {cls: name for cls, name in mapping.items() if cls not in registry}
    if missing:
        registry.register_module(missing, dm_class=wrapper)


class Ling3FlashBigHFModel(BigHFModelExportHelper):
    """Stream Ling's 512-expert blocks one layer at a time during export."""

    RUNTIME_PLACEHOLDER_REPLACEMENTS = {
        "LinearBailingMoeV3SparseMoeBlock": _LingSparseMoeBlockPlaceHolder,
    }
    PLACEHOLDER_TYPES = [
        "BailingMoeV3SparseMoeBlock",
        # AutoRound can fuse a sparse block during quantized preprocessing.
        "LinearBailingMoeV3SparseMoeBlock",
    ]

    @classmethod
    def initialize_process_worker(cls) -> None:
        patch_ling_remote_code_compatibility()
        from ._model import register_wrap_modules

        register_wrap_modules()

    @classmethod
    def _refresh_process_worker_registrations(
        cls,
        hf_model: torch.nn.Module,
    ) -> None:
        """Register the exact trusted-code classes created in this worker.

        Spawn workers import checkpoint remote code independently, so two
        classes with the same name are not necessarily the same Python class
        object.  DynamicModule's registry is keyed by class identity; scanning
        registrations made before model construction can therefore miss the
        worker-local sparse-MoE class and let FX enter its data-dependent
        Python loop.  Refresh from the concrete model after both construction
        and quantizer preprocessing.
        """

        patch_ling_remote_code_compatibility()
        from ._model import register_wrap_modules

        register_wrap_modules(hf_model)

    @classmethod
    def initialize_process_worker_after_model_load(
        cls,
        hf_model: torch.nn.Module,
    ) -> None:
        cls._refresh_process_worker_registrations(hf_model)

    @classmethod
    def initialize_process_worker_after_quantized_preprocess(
        cls,
        hf_model: torch.nn.Module,
    ) -> None:
        cls._refresh_process_worker_registrations(hf_model)

    @classmethod
    def register_placeholder(
        cls,
        registry: _DMRegistryCls,
        hf_model: Optional[torch.nn.Module] = None,
    ):
        mapping = {}
        for type_name in cls.PLACEHOLDER_TYPES:
            mapping.update(_runtime_aliases(hf_model, type_name))
        _register_if_missing(registry, mapping, _LingSparseMoeBlockPlaceHolder)


__all__ = ["Ling3FlashBigHFModel"]
