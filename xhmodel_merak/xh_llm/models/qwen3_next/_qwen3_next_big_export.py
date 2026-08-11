import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from safetensors import safe_open
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

from xhquant.api import get_xhquant_logger
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_model import XHBaseModel
from ...big_hf_model_helper import BigHFModelExportHelper
from ...register import XHLLM_TRACEABLE_MODULES


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

    class _Qwen3NextSparseMoeBlockBase(DynamicModule, Qwen3NextSparseMoeBlock):  # type: ignore[misc]  # noqa: N801
        ...

else:
    _Qwen3NextSparseMoeBlockBase = DynamicModule


class _Qwen3NextSparseMoeBlock_PlaceHolder(_Qwen3NextSparseMoeBlockBase):  # noqa: N801
    PLACEHOLDER_TYPE_NAME = "LinearQwen3NextSparseMoeBlock"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def _setup(self, *args, **kwargs):
        self._placeholder_registered_member_counts = _remove_registered_members(self)


def _register_module_if_missing(registry: _DMRegistryCls, cls_to_key: dict, dm_class) -> None:
    cls_to_key = {cls: key for cls, key in cls_to_key.items() if cls not in registry}
    if cls_to_key:
        registry.register_module(cls_to_key, dm_class=dm_class)


def _runtime_aliases(hf_model: Optional[torch.nn.Module], type_name: str) -> dict[type, str]:
    if hf_model is None:
        return {}
    return {type(module): type_name for module in hf_model.modules() if type(module).__name__ == type_name}


def _resolve_defused_moe_class():
    try:
        from defuser.modeling.unfused_moe.qwen3_next import LinearQwen3NextSparseMoeBlock
    except ImportError:
        return None
    return LinearQwen3NextSparseMoeBlock


def _checkpoint_uses_fused_expert_tensors(model_dir: str | Path) -> bool:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        tensor_names = json.loads(index_path.read_text()).get("weight_map", {})
        return any(name.endswith(".mlp.experts.gate_up_proj") for name in tensor_names)

    for checkpoint_path in model_dir.glob("*.safetensors"):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as reader:
            if any(name.endswith(".mlp.experts.gate_up_proj") for name in reader.keys()):
                return True
    return False


def register_runtime_wrap_modules(registry: _DMRegistryCls = XHLLM_TRACEABLE_MODULES) -> None:
    linear_sparse_moe_cls = _resolve_defused_moe_class()
    if linear_sparse_moe_cls is None:
        return

    from ._model import _Qwen3NextSparseMoeBlock

    _register_module_if_missing(
        registry,
        {linear_sparse_moe_cls: "LinearQwen3NextSparseMoeBlock"},
        _Qwen3NextSparseMoeBlock,
    )


class Qwen3NextBigHFModel(BigHFModelExportHelper):
    RUNTIME_PLACEHOLDER_REPLACEMENTS = {
        "LinearQwen3NextSparseMoeBlock": _Qwen3NextSparseMoeBlock_PlaceHolder,
    }
    PLACEHOLDER_TYPES = [
        "Qwen3NextSparseMoeBlock",
        "LinearQwen3NextSparseMoeBlock",
    ]

    @staticmethod
    def _preprocess_quantized_hf_model(
        hf_model,
        hf_model_dir: str | Path | None = None,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> None:
        config = hf_model.config
        quant_config = getattr(config, "quantization_config", None)
        quant_method = XHBaseModel._get_quantization_method(quant_config) if quant_config is not None else None
        if quant_method != "gptq" or hf_model_dir is None or not _checkpoint_uses_fused_expert_tensors(hf_model_dir):
            BigHFModelExportHelper._preprocess_quantized_hf_model(
                hf_model,
                hf_model_dir,
                skip_module_prefixes=skip_module_prefixes,
            )
            return

        from ... import big_hf_model_helper

        preprocessor = big_hf_model_helper.GPTQModelQuantizedModelPreprocessor(hf_model_dir)
        replaced = preprocessor.preprocess(
            hf_model,
            skip_module_prefixes=skip_module_prefixes,
            convert_model_structure=False,
        )
        get_xhquant_logger().info(
            "Preserved fused Qwen3-Next expert tensors and replaced %d checkpoint-backed Linear modules.",
            len(replaced),
        )
        XHBaseModel._trim_cpu_allocator()

    @classmethod
    def initialize_process_worker(cls) -> None:
        from ._model import register_wrap_modules

        register_wrap_modules()
        register_runtime_wrap_modules()

    @classmethod
    def register_placeholder(cls, registry: _DMRegistryCls, hf_model: Optional[torch.nn.Module] = None):
        linear_sparse_moe_cls = _resolve_defused_moe_class()
        module_types = {
            Qwen3NextSparseMoeBlock: "Qwen3NextSparseMoeBlock",
            **_runtime_aliases(hf_model, "Qwen3NextSparseMoeBlock"),
            **_runtime_aliases(hf_model, "LinearQwen3NextSparseMoeBlock"),
        }
        if linear_sparse_moe_cls is not None:
            module_types[linear_sparse_moe_cls] = "LinearQwen3NextSparseMoeBlock"
        _register_module_if_missing(
            registry,
            module_types,
            _Qwen3NextSparseMoeBlock_PlaceHolder,
        )
