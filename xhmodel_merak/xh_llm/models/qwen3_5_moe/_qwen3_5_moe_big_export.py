import importlib
from typing import TYPE_CHECKING, Container, Optional, Tuple

import torch
from torch import Tensor
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeSparseMoeBlock,
)

from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...big_hf_model_helper import BigHFModelExportHelper


def _registered_member_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "modules": len(list(module.children())),
        "parameters": len(list(module.parameters())),
        "buffers": len(list(module.buffers())),
    }


def _remove_registered_members(module: torch.nn.Module) -> dict[str, int]:
    module._modules.clear()
    module._parameters.clear()
    module._buffers.clear()
    counts = _registered_member_counts(module)
    if any(counts.values()):
        raise RuntimeError(f"Placeholder module still has registered members after cleanup: {counts}")
    return counts


if TYPE_CHECKING:

    class _Qwen3_5MoeGatedDeltaNetBase(DynamicModule, Qwen3_5MoeGatedDeltaNet):  # type: ignore[misc]  # noqa: N801
        ...

else:
    _Qwen3_5MoeGatedDeltaNetBase = DynamicModule


class _Qwen3_5MoeGatedDeltaNet_PlaceHolder(_Qwen3_5MoeGatedDeltaNetBase):  # noqa: N801
    """Qwen3.5 GatedDeltaNet linear attention wrapper.

    Key difference from Qwen3Next: Qwen3.5 uses **separate** projections
    (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) instead of Qwen3Next's
    merged projections (in_proj_qkvz, in_proj_ba).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_cache: Optional[list[Tensor]] = None,
        recurrent_state: Optional[Tensor] = None,
        linear_attn_mask: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
    ):
        if conv_cache is None:
            conv_cache = []
        return hidden_states, *conv_cache, recurrent_state

    def _setup(self, *args, **kwargs):
        self._placeholder_registered_member_counts = _remove_registered_members(self)


# ============================================================================
# Full Attention (with gating + partial rotary)
# ============================================================================


if TYPE_CHECKING:

    class _Qwen3_5MoeAttentionBase(DynamicModule, Qwen3_5MoeAttention):  # type: ignore[misc]  # noqa: N801
        ...

else:
    _Qwen3_5MoeAttentionBase = DynamicModule


class _Qwen3_5MoeAttention_PlaceHolder(_Qwen3_5MoeAttentionBase):  # noqa: N801
    """Qwen3.5 full attention with gating mechanism.

    q_proj outputs ``query + gate`` (2× heads), attention result is
    multiplied by ``sigmoid(gate)``.  Uses partial rotary (only first
    ``rotary_dim`` of ``head_dim`` get RoPE).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_seq_length: Optional[Tensor] = None,
        current_input_length: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_k_cache: Optional[Tensor] = None,
        past_v_cache: Optional[Tensor] = None,
    ) -> torch.FloatTensor:
        return hidden_states

    def _setup(self, *args, **kwargs):
        self._placeholder_registered_member_counts = _remove_registered_members(self)


if TYPE_CHECKING:

    class _Qwen3_5MoeSparseMoeBlockBase(DynamicModule, Qwen3_5MoeSparseMoeBlock):  # type: ignore[misc]  # noqa: N801
        ...

else:
    _Qwen3_5MoeSparseMoeBlockBase = DynamicModule


class _Qwen3_5MoeSparseMoeBlock_PlaceHolder(_Qwen3_5MoeSparseMoeBlockBase):  # noqa: N801
    PLACEHOLDER_TYPE_NAME = "LinearQwen3_5MoeSparseMoeBlock"

    def forward(self, x):
        return x

    def _setup(self, *args, **kwargs):
        self._placeholder_registered_member_counts = _remove_registered_members(self)


_HF_QWEN35_MOE_MODELING = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"


def _resolve_hf_qwen35_class(class_name: str):
    try:
        module = importlib.import_module(_HF_QWEN35_MOE_MODELING)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        if not (_HF_QWEN35_MOE_MODELING == missing_module or _HF_QWEN35_MOE_MODELING.startswith(missing_module + ".")):
            raise
        return None
    return getattr(module, class_name, None)


def _with_hf_alias(local_cls, registry_name: str):
    mapping = {local_cls: registry_name}
    hf_cls = _resolve_hf_qwen35_class(local_cls.__name__)
    if hf_cls is not None and hf_cls is not local_cls:
        mapping[hf_cls] = registry_name
    return mapping


def _register_module_if_missing(registry: _DMRegistryCls, cls_to_key: dict, dm_class) -> None:
    cls_to_key = {cls: key for cls, key in cls_to_key.items() if cls not in registry}
    if cls_to_key:
        registry.register_module(cls_to_key, dm_class=dm_class)


def _filter_by_placeholder_types(cls_to_key: dict, placeholder_types: Container[str]) -> dict:
    """仅保留 key 命中 placeholder 配置的类型映射。"""
    return {cls: key for cls, key in cls_to_key.items() if key in placeholder_types}


def _runtime_aliases(hf_model: Optional[torch.nn.Module], type_name: str) -> dict[type, str]:
    if hf_model is None:
        return {}
    return {type(module): type_name for module in hf_model.modules() if type(module).__name__ == type_name}


class Qwen3_5_MOE_BigHFModel(BigHFModelExportHelper):  # noqa: N801
    RUNTIME_PLACEHOLDER_REPLACEMENTS = {
        "LinearQwen3_5MoeSparseMoeBlock": _Qwen3_5MoeSparseMoeBlock_PlaceHolder,
    }
    PLACEHOLDER_TYPES = [
        # "Qwen3_5MoeAttention",
        "Qwen3_5MoeSparseMoeBlock",
        # AutoRound replaces the HF sparse block with this fused runtime class
        # during ``preprocess_model``.  It must remain the same placeholder
        # boundary or tracing enters its data-dependent Python expert loop.
        "LinearQwen3_5MoeSparseMoeBlock",  # auto round量化的模型会将Qwen3_5MoeSparseMoeBlock转为LinearQwen3_5MoeSparseMoeBlock
    ]

    @classmethod
    def initialize_process_worker(cls) -> None:
        from ._moe_model import register_wrap_modules

        register_wrap_modules()

    @classmethod
    def register_placeholder(cls, registry: _DMRegistryCls, hf_model: Optional[torch.nn.Module] = None):
        placeholder_types = cls.PLACEHOLDER_TYPES
        _register_module_if_missing(
            registry,
            _filter_by_placeholder_types(
                {
                    **_with_hf_alias(Qwen3_5MoeAttention, "Qwen3_5MoeAttention"),
                    **_runtime_aliases(hf_model, "Qwen3_5MoeAttention"),
                },
                placeholder_types,
            ),
            _Qwen3_5MoeAttention_PlaceHolder,
        )
        _register_module_if_missing(
            registry,
            _filter_by_placeholder_types(
                {
                    **_with_hf_alias(Qwen3_5MoeGatedDeltaNet, "Qwen3_5MoeGatedDeltaNet"),
                    **_runtime_aliases(hf_model, "Qwen3_5MoeGatedDeltaNet"),
                },
                placeholder_types,
            ),
            _Qwen3_5MoeGatedDeltaNet_PlaceHolder,
        )
        _register_module_if_missing(
            registry,
            _filter_by_placeholder_types(
                {
                    **_with_hf_alias(Qwen3_5MoeSparseMoeBlock, "Qwen3_5MoeSparseMoeBlock"),
                    **_runtime_aliases(hf_model, "Qwen3_5MoeSparseMoeBlock"),
                },
                placeholder_types,
            ),
            _Qwen3_5MoeSparseMoeBlock_PlaceHolder,
        )
