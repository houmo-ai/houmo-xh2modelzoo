from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from xhquant.api import get_xhquant_logger

from ...builder import register_llm_model
from ...types import LLMModelState
from ..qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from ..qwen3_5.workflow import Qwen35Workflow
from ..qwen3_5_moe.qwen3_5_moe_hmonnx_inference import XHQwen3_5MoeHMONNXModel
from ..qwen3_5_moe.qwen3_5_moe_model import (
    Qwen3_5Moe_ModelMeta,
    XHQwen3_5MoeModel,
    build_qwen3_5_moe_hf_compatible_model,
)
from ._dynamic_prune_moe import register_dynamic_prune_block
from .xh_qwen3_5_moe_dynamic_prune_config import XHQwen3_5MoeDynamicPruneModelConfig


class Qwen35DynamicPruneWorkflow(Qwen35Workflow):
    """Reuse Qwen35Workflow; translate only dynamic-prune model overrides."""

    expected_model_cls_names = {"XHQwen3_5MoeDynamicPruneModel"}
    expected_model_config_cls_names = {"XHQwen3_5MoeDynamicPruneModelConfig"}
    family_label = "Qwen3.5-MoE dynamic prune"

    def export(
        self,
        quant_result,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ):
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_cfg = workflow_config.export["model"]
        prune_cfg = model_cfg.get("dynamic_prune", {})
        scalar_path = prune_cfg.get("s_scalar_path")
        if scalar_path is None and bool(prune_cfg.get("auto_s_scalar", False)):
            scalar_path = _resolve_method1_s_scalar(
                model_dir=Path(self.model_dir),
                output_dir=Path(output_dir),
                model_cfg=model_cfg,
                prune_cfg=prune_cfg,
            )
        merged_overrides = dict(config_overrides or {})
        merged_overrides.update(
            {
                "export.model.dynamic_prune_threshold": float(prune_cfg.get("threshold", 0.0)),
                "export.model.dynamic_prune_s_scalar_path": scalar_path,
            }
        )
        return super().export(quant_result, output_dir, device, merged_overrides)


def _first_hybrid_block_layer_count(model_dir: Path) -> int:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    layer_types = text_config.get("layer_types")
    if not layer_types:
        raise ValueError(f"Cannot resolve Qwen3.5/Qwen3.6 layer_types from {model_dir / 'config.json'}")
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            return layer_idx + 1
    raise ValueError("Cannot resolve first hybrid block: no full_attention layer found")


def _resolve_method1_s_scalar(
    *,
    model_dir: Path,
    output_dir: Path,
    model_cfg: Mapping[str, Any],
    prune_cfg: Mapping[str, Any],
) -> str:
    script_path = Path(__file__).parents[4] / "examples/llm/qwen3_5_moe_prune/qwen3_5_moe_prune_xh2a_export_hmonnx.py"
    spec = importlib.util.spec_from_file_location("qwen3_5_moe_prune_export", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Method1 implementation from {script_path}")
    prune_export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prune_export)

    scalar_path = Path(prune_cfg.get("s_scalar_output") or output_dir.parent / f"{output_dir.name}_method1_s_scalar.pt")
    if scalar_path.exists() and not bool(prune_cfg.get("overwrite_s_scalar", False)):
        return str(scalar_path)

    max_layers = None
    if bool(model_cfg.get("only_first_block", False)):
        max_layers = _first_hybrid_block_layer_count(model_dir)
    elif model_cfg.get("max_layers") is not None:
        max_layers = int(model_cfg["max_layers"])

    logger = logging.getLogger(__name__)
    compute_device = prune_export._resolve_s_scalar_device(prune_cfg.get("s_scalar_compute_device", "auto"))
    eps = float(prune_cfg.get("s_scalar_eps", 1e-8))
    weight_map = prune_export._load_weight_map(model_dir)
    layer_keys = prune_export._find_fused_moe_layer_keys(weight_map)
    if max_layers is not None:
        layer_keys = [item for item in layer_keys if item[0] < max_layers]

    scalars = {}
    for layer_idx, gamma_key, gate_up_key, down_key in layer_keys:
        logger.info(f"Computing method1 s_scalar for layer {layer_idx}: {gate_up_key}")
        gamma = prune_export._load_checkpoint_tensor(model_dir, weight_map, gamma_key)
        gate_up_proj = prune_export._load_checkpoint_tensor(model_dir, weight_map, gate_up_key)
        down_proj = prune_export._load_checkpoint_tensor(model_dir, weight_map, down_key)
        scalars[str(layer_idx)] = prune_export._build_s_scalar_for_layer(
            gamma,
            gate_up_proj,
            down_proj,
            compute_device,
            eps,
        )
        del gamma, gate_up_proj, down_proj
        if compute_device.startswith("cuda"):
            torch.cuda.empty_cache()

    scalar_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scalars, scalar_path)
    logger.info(f"Saved method1 s_scalar for {len(scalars)} layers to {scalar_path}")
    return str(scalar_path)


def _load_s_scalars(path: str | None):
    if not path:
        return None
    scalar_path = Path(path)
    if scalar_path.suffix.lower() == ".json":
        payload = json.loads(scalar_path.read_text(encoding="utf-8"))
        return {str(key): torch.as_tensor(value, dtype=torch.float32) for key, value in payload.items()}
    return torch.load(scalar_path, map_location="cpu", weights_only=True)


@register_llm_model("Qwen3_5MoeForConditionalGeneration_dynamic_prune")
class XHQwen3_5MoeDynamicPruneModel(XHQwen3_5MoeModel):  # noqa: N801
    HF_MODEL_CLS = Qwen3_5MoeForConditionalGeneration
    META_CLS = Qwen3_5Moe_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5MoeHMONNXModel
    CONFIG_CLS = XHQwen3_5MoeDynamicPruneModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_moe_hf_compatible_model)
    WORKFLOW_CLS = (
        "xhmodel_merak.xh_llm.models.qwen3_5_moe_dynamic_prune.qwen3_5_moe_dynamic_prune_model:"
        "Qwen35DynamicPruneWorkflow"
    )

    def __init__(self, config: XHQwen3_5MoeDynamicPruneModelConfig):
        super().__init__(config)
        self.visual = None
        self._models.pop("visual", None)
        self.wrap_cfg["dynamic_prune_threshold"] = config.dynamic_prune_threshold

    def get_export_info(self, output_dir):
        # Dynamic-prune delivery is LLM-only, matching the original prune
        # exporter. Bypass the Qwen3.5 VLM layout that requires a visual model.
        return super(XHQwen3_5Model, self).get_export_info(output_dir)

    def export_hmonnx(self, output_dir):
        # The Qwen3.5 text graph is a prefill/decode ModelSwitcher.  The
        # generic LLM exporter assumes one quant graph and calls fixed() on
        # the switcher itself, so retain the generic LLM-only directory/meta
        # layout while fixing each Qwen3.5 graph explicitly.
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()

        exported_info = self.get_export_info(output_dir)
        self._export_hmonnx(exported_info)
        meta_info = exported_info.meta.to_dict()
        with open(Path(exported_info.exported_dir) / "golden_meta_info.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=4)
        get_xhquant_logger().info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info

    def init_wrap_model(self, hf_model):
        register_dynamic_prune_block()
        return super().init_wrap_model(hf_model)

    def _wraped_post(self, hf_model):
        super()._wraped_post(hf_model)
        scalars = _load_s_scalars(self.config.dynamic_prune_s_scalar_path)
        if scalars is None:
            return
        for ordinal, (name, module) in enumerate(self._wrap_model.named_modules()):
            moeblock = getattr(module, "moeblock", None)
            if moeblock is None or moeblock.s_scalar is None:
                continue
            layer_idx = int(name.split("layers.", 1)[1].split(".", 1)[0]) if "layers." in name else ordinal
            selected = None
            if isinstance(scalars, dict):
                for key in (name, str(layer_idx), f"layer_{layer_idx}", f"layers.{layer_idx}"):
                    if key in scalars:
                        selected = scalars[key]
                        break
            elif isinstance(scalars, torch.Tensor):
                selected = scalars if scalars.ndim == 1 else scalars[layer_idx]
            if selected is None:
                continue
            selected = torch.as_tensor(selected, dtype=moeblock.s_scalar.dtype, device=moeblock.s_scalar.device)
            if selected.numel() != moeblock.s_scalar.numel():
                raise ValueError(
                    f"s_scalar for {name} has {selected.numel()} entries; expected {moeblock.s_scalar.numel()}"
                )
            moeblock.s_scalar = selected.reshape_as(moeblock.s_scalar)
