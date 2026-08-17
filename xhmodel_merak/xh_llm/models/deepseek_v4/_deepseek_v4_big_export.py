"""Low-memory, one-MoE-at-a-time DeepSeek-V4 HMONNX export."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Optional

from torch import nn

from xhquant.api import get_xhquant_logger
from xhquant.utils.registry import _DMRegistryCls

from ..._dequant_converter import replace_gptqmodel_quant_linears_with_packed
from ...base_model import XHBaseModel
from ...big_hf_model_helper import (
    BigHFModelExportHelper,
    GPTQModelQuantizedModelPreprocessor,
    _model_floating_dtype,
)
from .moe import DeepSeekV4MoE, DeepSeekV4MoEPlaceHolder


_STREAMING_PLACEHOLDER_MARKER = "_xh_v4_streaming_placeholder"
_STATIC_MOE_TARGET = re.compile(r"^blocks\.layers\.(\d+)\.mlp$")


class DeepSeekV4BigHFModel(BigHFModelExportHelper):
    """Stream complete V4 MoE blocks while retaining attention in the main graph.

    The checkpoint path is ``model.layers.N.mlp`` while the static Merak graph
    path is ``blocks.layers.N.mlp``. This helper owns that explicit mapping and
    retains GPTQModel/AutoRound checkpoint words until each XH2 QModule is
    quantized; no dense BF16 expert copy is created.
    """

    PLACEHOLDER_TYPES = ["DeepseekV4SparseMoeBlock"]
    _GPTQMODEL_METHODS = {"gptq", "auto-round", "auto_round", "autoround"}

    @classmethod
    def register_placeholder(
        cls,
        registry: _DMRegistryCls,
        hf_model: Optional[nn.Module] = None,
    ) -> None:
        # V4 builds a dedicated static model before FX tracing instead of
        # converting the HF SparseMoeBlock through the DynamicModule registry.
        del cls, registry, hf_model

    @classmethod
    def _preprocess_quantized_hf_model(
        cls,
        hf_model,
        hf_model_dir: str | Path | None = None,
        include_module_prefixes: Optional[list[str]] = None,
        skip_module_prefixes: Optional[list[str]] = None,
    ) -> None:
        config = hf_model.config
        quant_config = getattr(config, "quantization_config", None)
        if quant_config is None:
            return
        quant_method = XHBaseModel._get_quantization_method(quant_config)
        if quant_method not in cls._GPTQMODEL_METHODS:
            raise NotImplementedError(
                "DeepSeek-V4 low-memory export currently requires a GPTQModel "
                f"GPTQ/AutoRound checkpoint, got quant_method={quant_method!r}."
            )
        model_dir = hf_model_dir or getattr(config, "_name_or_path", None)
        if not model_dir:
            raise ValueError("hf_model_dir is required for DeepSeek-V4 packed export")
        preprocessor = GPTQModelQuantizedModelPreprocessor(
            model_dir,
            dtype=_model_floating_dtype(hf_model),
        )
        replaced = preprocessor.preprocess(
            hf_model,
            include_module_prefixes=include_module_prefixes,
            skip_module_prefixes=skip_module_prefixes,
        )
        get_xhquant_logger().info(
            "Prepared %d DeepSeek-V4 GPTQModel packed Linear shells on meta.",
            len(replaced),
        )
        XHBaseModel._trim_cpu_allocator()

    @classmethod
    def mark_streaming_placeholders(
        cls,
        hf_model: nn.Module,
        module_prefixes: Optional[list[str]] = None,
    ) -> int:
        """Mark selected checkpoint MoEs so conversion emits weightless leaves.

        A full checkpoint may back a shorter diagnostic export. In that case
        only those exact layer prefixes may become streaming placeholders;
        marking all 43 checkpoint MoEs makes the static placeholder inventory
        disagree with ``max_layers``.
        """

        marked = 0
        configured = set(cls.PLACEHOLDER_TYPES)
        selected = None if module_prefixes is None else set(module_prefixes)
        for name, module in hf_model.named_modules():
            if selected is not None and name not in selected:
                continue
            if cls._placeholder_type_name_candidates(module) & configured:
                setattr(module, _STREAMING_PLACEHOLDER_MARKER, True)
                marked += 1
        if marked == 0:
            raise RuntimeError("DeepSeek-V4 low-memory export found no SparseMoeBlock")
        return marked

    def register_static_placeholder_modules(self, *models: nn.Module) -> int:
        """Teach the frontend tracer about the post-conversion MoE boundary."""

        found = 0
        for model in models:
            for module in model.modules():
                if isinstance(module, DeepSeekV4MoEPlaceHolder):
                    self._placeholder_types.add(type(module))
                    found += 1
        if found == 0:
            raise RuntimeError("Static DeepSeek-V4 graph contains no MoE placeholders")
        for model in models:
            self.register_layer_as_placeholder(model)
        return found

    @staticmethod
    def _convert_loaded_packed_linears(module: nn.Module) -> int:
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear

        packed = [
            (name, submodule)
            for name, submodule in module.named_modules()
            if isinstance(submodule, PackableQuantLinear)
        ]
        for name, submodule in packed:
            qweight = getattr(submodule, "qweight", None)
            if qweight is None or getattr(qweight, "is_meta", False):
                raise RuntimeError(f"DeepSeek-V4 packed Linear {name!r} was not materialized")
        converted = replace_gptqmodel_quant_linears_with_packed(module)
        if len(converted) != len(packed):
            raise RuntimeError(
                "DeepSeek-V4 did not transfer every materialized GPTQ Linear: "
                f"expected={len(packed)}, converted={len(converted)}"
            )
        return len(converted)

    def prepare_materialized_non_placeholder_model(
        self,
        hf_model,
        quantization_config: Any,
    ) -> None:
        quant_method = XHBaseModel._get_quantization_method(quantization_config)
        if quant_method not in self._GPTQMODEL_METHODS:
            return super().prepare_materialized_non_placeholder_model(
                hf_model,
                quantization_config,
            )
        converted = self._convert_loaded_packed_linears(hf_model)
        get_xhquant_logger().info(
            "Retained %d non-MoE GPTQ Linears without dense dequantization.",
            converted,
        )

    @staticmethod
    def checkpoint_moe_target(static_target: str) -> str:
        match = _STATIC_MOE_TARGET.fullmatch(static_target)
        if match is None:
            raise ValueError(f"Unexpected DeepSeek-V4 static MoE target: {static_target!r}")
        return f"model.layers.{int(match.group(1))}.mlp"

    def _load_place_holder_module_once(
        self,
        empty_hf_model_for_placeholder,
        module_target: str,
    ) -> DeepSeekV4MoE:
        checkpoint_target = self.checkpoint_moe_target(module_target)
        source_moe = copy.deepcopy(empty_hf_model_for_placeholder.get_submodule(checkpoint_target))
        self._load_module_from_safetensor(
            source_moe,
            checkpoint_target,
        )
        config = empty_hf_model_for_placeholder.config
        quantization_config = getattr(config, "quantization_config", None)
        converted = 0
        if quantization_config is not None:
            self._prepare_loaded_gptqmodel_modules(
                source_moe,
                quantization_config,
                model_config=config,
            )
            converted = self._convert_loaded_packed_linears(source_moe)
        static_moe = DeepSeekV4MoE.from_hf(
            source_moe,
            config,
            fast_mode=True,
        )
        get_xhquant_logger().debug(
            "Materialized streamed DeepSeek-V4 MoE %s from %s (%d packed linears)",
            module_target,
            checkpoint_target,
            converted,
        )
        return static_moe


__all__ = ["DeepSeekV4BigHFModel"]
