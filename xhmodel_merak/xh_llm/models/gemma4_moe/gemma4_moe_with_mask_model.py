from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MethodType
from typing import Any, cast

import torch
import torch.nn as nn
from safetensors import safe_open
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoTokenizer, Gemma4ForConditionalGeneration

from xhmodel_merak.xh_llm.llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from xhquant.api import ConfigDict, get_xhquant_logger

from ...base_llm_model import XHLLMModelProcessor
from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheMixin
from ...vision_llm_model import VisionLLMModel
from .gemma4_moe_hf_compatible import build_gemma4_moe_with_mask_hf_compatible_model
from .gemma4_moe_hmonnx_inference import XHGemma4MoeWithMaskHMONNXModel, _gen_mask_v2, aligned
from .gemma4_moe_visual_model import XHGemma4MoeVisualModel
from .xh_gemma4_moe_config import XHGemma4MoeWithMaskConfig


def _convert_float_backed_gptq_linear(
    module: nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    for attr_name in (
        "qweight",
        "qzeros",
        "scales",
        "g_idx",
        "wf",
        "wf_unsqueeze_zero",
        "wf_unsqueeze_neg_one",
        "quant_weight",
    ):
        if hasattr(module, attr_name):
            delattr(module, attr_name)

    weight = weight.detach().to("cpu")
    if bias is not None:
        bias = bias.detach().to("cpu")

    module.__class__ = nn.Linear
    module.in_features = int(weight.shape[1])
    module.out_features = int(weight.shape[0])
    module.forward = MethodType(nn.Linear.forward, module)

    module._parameters.pop("weight", None)
    module._parameters["weight"] = nn.Parameter(weight, requires_grad=False)
    if bias is None:
        module._parameters.pop("bias", None)
        module.bias = None
    else:
        module._parameters.pop("bias", None)
        module._parameters["bias"] = nn.Parameter(bias, requires_grad=False)
    module._buffers.pop("quant_weight", None)
    module._xhquant_weight_origin = "float"


class Gemma4MoeKVCacheMixin(KVCacheMixin):
    def __init__(self, kv_cache_config):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes: list[list[int]] = []

    def set_layer_kv_shapes(self, layer_kv_shapes: list[list[int]]):
        self.layer_kv_shapes = layer_kv_shapes
        self.kvcache_config.num_layers = len(layer_kv_shapes)
        if layer_kv_shapes:
            self.kvcache_config.kv_cache_shape = layer_kv_shapes[0]

    def prepare_kv_cache(self, dtype=torch.float16):
        if not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for shape in self.layer_kv_shapes:
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))


class Gemma4MoeWithMaskInputProcessor(BaseLLMInputProcessor):
    def __init__(self, config: BaseInputProcessorConfig, sliding_window_cfg: dict[str, Any]):
        super().__init__(config)
        self.sliding_window_cfg = sliding_window_cfg

    def prepare_casual_mask(self, x: Tensor, valid_length: int | Tensor, attention_max_length: int) -> Tensor:
        mask = _gen_mask_v2(x, valid_length, attention_max_length)
        attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)
        return attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)

    def forward(self, data: dict | tuple | list) -> list[torch.Tensor]:
        inputs_embeds, past_seq_length, seq_length, past_key_caches, past_value_caches = super().forward(data)
        bz, nq = inputs_embeds.shape[:2]
        local_attention_mask = None
        global_attention_mask = None
        if self.sliding_window_cfg.get("has_global_attention", False):
            global_window = self.sliding_window_cfg.get("global_attention_window_size", 2048)
            x = torch.empty((bz, nq, global_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            global_attention_mask = self.prepare_casual_mask(x, past_seq_length, -1)
        if self.sliding_window_cfg.get("has_local_attention", False):
            local_window = self.sliding_window_cfg.get("local_attention_window_size", 1024) + nq - 1
            local_window = aligned(local_window, 16)
            x = torch.empty((bz, nq, local_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            local_attention_mask = self.prepare_casual_mask(
                x,
                past_seq_length,
                self.sliding_window_cfg.get("sliding_window", 1024),
            )
        # Keep cache lists *grouped* (not flattened) so that the wrapped
        # `_Gemma4ForCausalLM._forward(... past_key_caches, past_value_caches)`
        # signature still matches at frontend trace time. Downstream stages
        # (`_to_quanted` / `_export_hmonnx`) call `unfold_args` themselves to
        # flatten the cache lists for ptq calibration / export tracing,
        # mirroring xhquant_llm's source semantics.
        outputs: list = [inputs_embeds, past_seq_length, seq_length]
        if local_attention_mask is not None:
            outputs.append(local_attention_mask)
        if global_attention_mask is not None:
            outputs.append(global_attention_mask)
        outputs.append(past_key_caches)
        outputs.append(past_value_caches)
        return outputs


@register_llm_model("Gemma4ForConditionalGeneration_with_mask")
class XHGemma4MoeWithMaskModel(VisionLLMModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HMONNXINFERENCE_CLS = XHGemma4MoeWithMaskHMONNXModel
    CONFIG_CLS = XHGemma4MoeWithMaskConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_gemma4_moe_with_mask_hf_compatible_model)

    def __init__(self, config: XHGemma4MoeWithMaskConfig):
        super().__init__(config)
        self.config = cast(XHGemma4MoeWithMaskConfig, self.config)
        self._kvcache_mixin = Gemma4MoeKVCacheMixin(self.kvcache_config)
        self.fallback_hf_model_dir = config.fallback_hf_model
        if self.config.visual_config is not None:
            self.visual = XHGemma4MoeVisualModel(self.config.visual_config)
            self.visual.config.model_name = f"{self.config.model_name}_visual"
        self.sliding_window_cfg = {
            "sliding_window": self.config.sliding_window,
            "local_attention_window_size": self.config.local_attention_window_size,
            "global_attention_window_size": self.config.global_attention_window_size,
            "has_local_attention": self.config.has_local_attention,
            "has_global_attention": self.config.has_global_attention,
        }

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        if hasattr(self, "visual"):
            self.visual.work_dir = str(Path(work_dir) / "visual")

    def get_tokenizer(self, **kwargs):
        assert self.hf_model_dir is not None
        kwargs.setdefault("trust_remote_code", True)
        return AutoTokenizer.from_pretrained(self.hf_model_dir, **kwargs)

    def get_tf_processor(self):
        return XHLLMModelProcessor(self.get_tokenizer(trust_remote_code=True))

    @staticmethod
    def _build_checkpoint_weight_map(model_dir: str | Path) -> dict[str, str]:
        model_dir = Path(model_dir)
        weight_map: dict[str, str] = {}
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, encoding="utf-8") as f:
                data = json.load(f)
            for key, rel_path in data.get("weight_map", {}).items():
                weight_map[key] = str(model_dir / rel_path)
            return weight_map

        safetensors_file = model_dir / "model.safetensors"
        if safetensors_file.exists():
            with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = str(safetensors_file)
        return weight_map

    def _get_hf_checkpoint_weight_map(self) -> dict[str, str]:
        cached = getattr(self, "_hf_checkpoint_weight_map", None)
        if cached is not None:
            return cached

        weight_map = self._build_checkpoint_weight_map(self.hf_model_dir)
        fallback_hf_model_dir = self.fallback_hf_model_dir
        if fallback_hf_model_dir is not None:
            fallback_path = Path(fallback_hf_model_dir)
            primary_path = Path(self.hf_model_dir)
            if fallback_path != primary_path:
                for key, file_path in self._build_checkpoint_weight_map(fallback_path).items():
                    weight_map.setdefault(key, file_path)

        self._hf_checkpoint_weight_map = weight_map
        return weight_map

    def _load_checkpoint_tensor_from_weight_map(
        self,
        tensor_name: str,
        weight_map: dict[str, str],
    ) -> torch.Tensor | None:
        file_path = weight_map.get(tensor_name)
        if file_path is None:
            return None
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_name)

    def _load_hf_checkpoint_tensor(self, tensor_name: str) -> torch.Tensor | None:
        return self._load_checkpoint_tensor_from_weight_map(tensor_name, self._get_hf_checkpoint_weight_map())

    def _checkpoint_uses_gptqmodel(self) -> bool:
        config_path = Path(self.hf_model_dir) / "config.json"
        if not config_path.exists():
            return False
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        quantization_config = config_data.get("quantization_config", {})
        quant_method = str(quantization_config.get("quant_method", "")).lower()
        meta = str(quantization_config.get("meta", "")).lower()
        return quant_method == "gptq" and "gptqmodel" in meta

    def _infer_fallback_missing_keys(self, hf_model: Gemma4ForConditionalGeneration) -> list[str]:
        fallback_hf_model_dir = self.fallback_hf_model_dir
        if fallback_hf_model_dir is None:
            return []

        primary_weight_map = self._build_checkpoint_weight_map(self.hf_model_dir)
        fallback_weight_map = self._build_checkpoint_weight_map(fallback_hf_model_dir)
        if len(fallback_weight_map) == 0:
            return []

        expected_quant_suffixes = {"qweight", "qzeros", "scales", "g_idx"}
        named_tensors: dict[str, torch.Tensor] = dict(hf_model.named_parameters())
        named_tensors.update(dict(hf_model.named_buffers()))

        missing_keys = []
        for key in named_tensors:
            if key not in fallback_weight_map or key in primary_weight_map:
                continue

            if "." in key:
                prefix, suffix = key.rsplit(".", 1)
                if suffix in expected_quant_suffixes and (
                    f"{prefix}.weight" in primary_weight_map or f"{prefix}.qweight" in primary_weight_map
                ):
                    continue

            missing_keys.append(key)

        return missing_keys

    def _load_missing_tensors_from_fallback(
        self,
        hf_model: Gemma4ForConditionalGeneration,
        missing_keys: list[str],
    ) -> tuple[int, list[str]]:
        fallback_hf_model_dir = self.fallback_hf_model_dir
        if fallback_hf_model_dir is None or len(missing_keys) == 0:
            return 0, missing_keys

        fallback_weight_map = self._build_checkpoint_weight_map(fallback_hf_model_dir)
        if len(fallback_weight_map) == 0:
            return 0, missing_keys

        named_tensors: dict[str, torch.Tensor] = dict(hf_model.named_parameters())
        named_tensors.update(dict(hf_model.named_buffers()))

        fallback_state_dict = {}
        unresolved_missing = []
        for key in missing_keys:
            target_tensor = named_tensors.get(key)
            fallback_tensor = self._load_checkpoint_tensor_from_weight_map(key, fallback_weight_map)
            if target_tensor is None or fallback_tensor is None:
                unresolved_missing.append(key)
                continue
            if tuple(target_tensor.shape) != tuple(fallback_tensor.shape):
                unresolved_missing.append(key)
                continue
            fallback_state_dict[key] = fallback_tensor

        if len(fallback_state_dict) == 0:
            return 0, unresolved_missing

        hf_model.load_state_dict(fallback_state_dict, strict=False)
        unresolved_missing = [key for key in unresolved_missing if key not in fallback_state_dict]
        return len(fallback_state_dict), unresolved_missing

    def _filter_expected_missing_quant_keys(self, missing_keys: list[str]) -> list[str]:
        checkpoint_weight_map = self._get_hf_checkpoint_weight_map()
        expected_suffixes = {"qweight", "qzeros", "scales", "g_idx"}
        filtered_missing = []
        for key in missing_keys:
            if "." not in key:
                filtered_missing.append(key)
                continue
            prefix, suffix = key.rsplit(".", 1)
            if suffix in expected_suffixes and f"{prefix}.weight" in checkpoint_weight_map:
                continue
            filtered_missing.append(key)
        return filtered_missing

    def _restore_runtime_only_buffers(self, hf_model: Gemma4ForConditionalGeneration) -> list[str]:
        restored: list[str] = []
        text_model = hf_model.model.language_model

        rotary_emb = getattr(text_model, "rotary_emb", None)
        if rotary_emb is not None:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

            fresh_rotary = Gemma4TextRotaryEmbedding(text_model.config, device="cpu")
            rotary_emb.max_seq_len_cached = fresh_rotary.max_seq_len_cached
            rotary_emb.original_max_seq_len = fresh_rotary.original_max_seq_len

            for attr_name in (
                "full_attention_attention_scaling",
                "sliding_attention_attention_scaling",
            ):
                if hasattr(fresh_rotary, attr_name):
                    setattr(rotary_emb, attr_name, getattr(fresh_rotary, attr_name))

            for buffer_name in (
                "full_attention_inv_freq",
                "full_attention_original_inv_freq",
                "sliding_attention_inv_freq",
                "sliding_attention_original_inv_freq",
            ):
                current = getattr(rotary_emb, buffer_name, None)
                expected = getattr(fresh_rotary, buffer_name, None)
                if current is None or expected is None:
                    continue

                needs_restore = tuple(current.shape) != tuple(expected.shape)
                if not needs_restore:
                    needs_restore = not torch.allclose(
                        current.detach().float().cpu(),
                        expected.detach().float().cpu(),
                        atol=1e-6,
                        rtol=1e-6,
                    )
                if needs_restore:
                    rotary_emb._buffers[buffer_name] = expected.to(device=current.device, dtype=current.dtype)
                    restored.append(buffer_name)

        embed_tokens = getattr(text_model, "embed_tokens", None)
        embed_scale = getattr(embed_tokens, "embed_scale", None)
        if embed_tokens is not None and embed_scale is not None:
            expected_embed_scale = torch.tensor(
                text_model.config.hidden_size**0.5,
                device=embed_scale.device,
                dtype=embed_scale.dtype,
            )
            if not torch.allclose(embed_scale.detach().float().cpu(), expected_embed_scale.detach().float().cpu(), atol=1e-3, rtol=1e-3):
                embed_tokens._buffers["embed_scale"] = expected_embed_scale
                restored.append("embed_scale")
        return restored

    def _build_torch_backend_gptq_config(self):
        config_path = Path(self.hf_model_dir) / "config.json"
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)

        quantization_config = dict(config_data.get("quantization_config", {}))
        quantization_config["backend"] = "torch"

        try:
            from transformers.utils.quantization_config import GPTQConfig

            return GPTQConfig.from_dict(quantization_config)
        except Exception:
            return quantization_config

    def _dequantize_gptqmodel(self, hf_model: Gemma4ForConditionalGeneration) -> Gemma4ForConditionalGeneration:
        from gptqmodel.nn_modules.qlinear import PackableQuantLinear

        from xhmodel_merak.xh_llm._dequant_converter import gptqmodel_torch_qlinear_converter

        checkpoint_weight_map = self._get_hf_checkpoint_weight_map()
        dequant_linears = [
            (name, module)
            for name, module in hf_model.named_modules()
            if isinstance(module, PackableQuantLinear)
            and (name.startswith("model.language_model.") or name == "lm_head")
        ]
        get_xhquant_logger().info("Dequantizing %d Gemma4 GPTQModel Linear modules", len(dequant_linears))

        converted_gptq = 0
        converted_float = 0
        for name, module in dequant_linears:
            float_weight_name = f"{name}.weight"
            quant_weight_name = f"{name}.qweight"
            if float_weight_name in checkpoint_weight_map and quant_weight_name not in checkpoint_weight_map:
                float_weight = self._load_hf_checkpoint_tensor(float_weight_name)
                bias = self._load_hf_checkpoint_tensor(f"{name}.bias")
                if float_weight is None:
                    continue
                _convert_float_backed_gptq_linear(module, float_weight, bias)
                converted_float += 1
                continue

            gptqmodel_torch_qlinear_converter(module)
            module.in_features = int(getattr(module, "in_features", getattr(module, "infeatures", module.weight.shape[1])))
            module.out_features = int(getattr(module, "out_features", getattr(module, "outfeatures", module.weight.shape[0])))
            if not hasattr(module, "bias"):
                module.bias = None
            module.forward = MethodType(nn.Linear.forward, module)
            module._xhquant_weight_origin = "gptq"
            converted_gptq += 1

        get_xhquant_logger().info(
            "Converted Gemma4 GPTQModel Linear modules: %d gptq, %d float fallback",
            converted_gptq,
            converted_float,
        )

        hf_model.quantization_method = None  # type: ignore[attr-defined]
        hf_model._is_hf_initialized = False  # type: ignore[attr-defined]
        if hasattr(hf_model.config, "quantization_config"):
            hf_model.config.quantization_config = None
        return hf_model

    def _truncate_hf_model_for_export(self, hf_model: Gemma4ForConditionalGeneration) -> None:
        get_max_decode_layers = getattr(self.config, "get_max_decode_layers", None)
        if get_max_decode_layers is None:
            return

        max_layers = int(get_max_decode_layers())
        if max_layers <= 0:
            return

        text_model = hf_model.model.language_model
        layers = getattr(text_model, "layers", None)
        if layers is None or len(layers) <= max_layers:
            return

        text_model.layers = nn.ModuleList(list(layers[:max_layers]))
        text_model.config.num_hidden_layers = max_layers
        text_config = getattr(hf_model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            text_config.num_hidden_layers = max_layers
        get_xhquant_logger().info(
            "Truncated Gemma4 language model layers to %d for current export config",
            max_layers,
        )

    def _load_hf_model(self, device_map="cpu", **kwargs) -> Gemma4ForConditionalGeneration:
        assert self.hf_model_dir is not None
        logger = get_xhquant_logger()
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
            attn_implementation="eager",
            output_loading_info=True,
            **kwargs,
        )

        self._uses_gptqmodel_checkpoint = self._checkpoint_uses_gptqmodel()
        if self._uses_gptqmodel_checkpoint:
            logger.info("[GPTQModel] Detected Gemma4 GPTQModel checkpoint, loading with torch backend")
            load_kwargs["quantization_config"] = self._build_torch_backend_gptq_config()

        with torch.no_grad():
            loaded = Gemma4ForConditionalGeneration.from_pretrained(self.hf_model_dir, **load_kwargs)

        hf_model = loaded
        loading_info: dict[str, Any] = {}
        if isinstance(loaded, tuple):
            hf_model, loading_info = loaded

        self._truncate_hf_model_for_export(hf_model)

        missing_keys = list(loading_info.get("missing_keys", []))
        missing_keys.extend(self._infer_fallback_missing_keys(hf_model))
        missing_keys = list(dict.fromkeys(missing_keys))
        loaded_from_fallback, unresolved_missing = self._load_missing_tensors_from_fallback(hf_model, missing_keys)
        if loaded_from_fallback > 0:
            logger.info(
                "Loaded %d missing tensors from fallback checkpoint: %s",
                loaded_from_fallback,
                self.fallback_hf_model_dir,
            )

        restored_runtime_buffers = self._restore_runtime_only_buffers(hf_model)
        if len(restored_runtime_buffers) > 0:
            logger.info(
                "Restored %d runtime-only buffers from Gemma4 config: %s",
                len(restored_runtime_buffers),
                restored_runtime_buffers,
            )

        if self._uses_gptqmodel_checkpoint:
            hf_model = self._dequantize_gptqmodel(hf_model)

        unresolved_missing = self._filter_expected_missing_quant_keys(unresolved_missing)
        if len(unresolved_missing) > 0:
            logger.warning(
                "Gemma4 fallback checkpoint still misses %d tensors; first 10: %s",
                len(unresolved_missing),
                unresolved_missing[:10],
            )

        if getattr(hf_model.config, "tie_word_embeddings", False):
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False
        return hf_model.eval()

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        from transformers import Gemma4ForConditionalGeneration

        kwargs.setdefault("torch_dtype", torch.bfloat16)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("device_map", "cpu")
        kwargs.setdefault("attn_implementation", "eager")
        hf_model = Gemma4ForConditionalGeneration.from_pretrained(hf_model_dir, **kwargs).eval()
        if getattr(hf_model.config, "tie_word_embeddings", False):
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False
        return hf_model

    def get_native_model(self):
        if self.config.quant_weight is not None and len(self.config.quant_weight) > 0:
            raise RuntimeError("Gemma4 MoE with-mask does not support external quant_weight with GPTQModel loading.")
        return self._load_hf_model(device_map="cpu")

    def _apply_fallback_linear_w_schema(self, frontend_model: nn.Module) -> None:
        quant_cfg = self.get_quant_cfg()
        fallback_w_schema = quant_cfg.get("fallback_w_schema", None)
        if fallback_w_schema is None and getattr(self, "_uses_gptqmodel_checkpoint", False):
            fallback_w_schema = ConfigDict(dict(fp_mode="sefp", hidden_bit=True, bits=8))

        if fallback_w_schema is None:
            return

        nodes_cfg = quant_cfg.get("nodes_cfg", {})
        has_gptq_linear = False
        fallback_nodes = []
        for node in frontend_model.graph.nodes:
            if node.op != "call_module":
                continue
            module = frontend_model.get_submodule(node.target)
            if not isinstance(module, nn.Linear):
                continue
            if getattr(module, "_xhquant_weight_origin", None) == "gptq":
                has_gptq_linear = True
                continue
            if getattr(module, "keep16bit", False):
                continue
            fallback_nodes.append(node)

        if not has_gptq_linear:
            return

        applied = 0
        for node in fallback_nodes:
            if node.target in nodes_cfg or node.name in nodes_cfg:
                continue
            node_quant_cfg = copy.deepcopy(node.meta.get("quant_config", {}))
            if "w_schema" in node_quant_cfg:
                continue
            node_quant_cfg["w_schema"] = copy.deepcopy(fallback_w_schema)
            node.meta["quant_config"] = node_quant_cfg
            applied += 1

        if applied > 0:
            get_xhquant_logger().info(
                "Applied fallback sefp-8 w_schema to %d float Linear nodes in mixed GPTQModel Gemma4 graph",
                applied,
            )

    def _to_quanted(self, frontend_model, state):
        self._apply_fallback_linear_w_schema(frontend_model)
        return super()._to_quanted(frontend_model, state)

    def _get_language_model(self, hf_model: Any) -> Any:
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
            return hf_model.model.language_model
        return super()._get_language_model(hf_model)

    def _patch_no_scale_rmsnorm(self, text_model: Any) -> None:
        from xhquant.nn import RMSNorm as _RMSNorm

        hidden_size = text_model.config.hidden_size
        for layer in self._wrap_model.model.layers:
            attn = layer.self_attn
            vnorm = getattr(attn, "v_norm", None)
            if vnorm is not None and hasattr(vnorm, "norm"):
                inner = vnorm.norm
                if not isinstance(inner, _RMSNorm):
                    eps = getattr(inner, "eps", 1e-6)
                    new_norm = _RMSNorm(attn.head_dim, eps)
                    new_norm.weight.requires_grad_(False)
                    vnorm.norm = new_norm
            router_norm = getattr(layer, "moe_router_norm", None)
            if router_norm is not None and hasattr(router_norm, "norm"):
                inner = router_norm.norm
                if not isinstance(inner, _RMSNorm):
                    eps = getattr(inner, "eps", 1e-6)
                    new_norm = _RMSNorm(hidden_size, eps)
                    new_norm.weight.requires_grad_(False)
                    router_norm.norm = new_norm

    def init_wrap_model(self, hf_model: Any) -> Any:
        if hf_model is None:
            hf_model = self.get_native_model()

        from transformers import Gemma4ForCausalLM

        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls

        llm_register_wrap_cls(hf_model)
        text_model = hf_model.model.language_model

        def _remap_tied_weight_keys(tied_weight_keys: Any) -> Any:
            if isinstance(tied_weight_keys, dict):
                return {
                    key.replace("model.language_model.", "model."): value.replace(
                        "model.language_model.", "model."
                    )
                    for key, value in tied_weight_keys.items()
                }
            if isinstance(tied_weight_keys, (list, tuple, set)):
                return type(tied_weight_keys)(
                    key.replace("model.language_model.", "model.") for key in tied_weight_keys
                )
            return tied_weight_keys

        causal_lm = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
        torch.nn.Module.__init__(causal_lm)
        causal_lm.model = text_model
        causal_lm.lm_head = hf_model.lm_head
        causal_lm.config = text_model.config
        causal_lm.generation_config = getattr(hf_model, "generation_config", None)
        causal_lm._tied_weights_keys = _remap_tied_weight_keys(getattr(hf_model, "_tied_weights_keys", []))
        causal_lm.all_tied_weights_keys = _remap_tied_weight_keys(
            getattr(hf_model, "all_tied_weights_keys", causal_lm._tied_weights_keys)
        )

        wraped_model = super().init_wrap_model(causal_lm)
        self._patch_no_scale_rmsnorm(text_model)
        self.generation_config = getattr(hf_model, "generation_config", None)
        self.num_hidden_layers = text_model.config.num_hidden_layers
        return wraped_model

    def _wraped_post(self, hf_model: Any):
        super()._wraped_post(hf_model)
        language_model = self._get_language_model(self._wrap_model)

        # Gemma4TextScaledWordEmbedding applies embed_scale at runtime.
        # Bake that scale into the exported embedding weights so quant_embedding.pt
        # matches the HMONNX text graph's expected inputs.
        orig_embed = language_model.get_input_embeddings()
        orig_device = orig_embed.weight.device
        embed_copy = copy.deepcopy(orig_embed.cpu())
        orig_embed.to(orig_device)
        if hasattr(embed_copy, "embed_scale"):
            embed_copy.weight.data = (embed_copy.weight.float() * embed_copy.embed_scale).to(embed_copy.weight.dtype)
        self.embed_tokens = nn.Embedding(
            embed_copy.num_embeddings,
            embed_copy.embedding_dim,
            _weight=embed_copy.weight,
        ).to(orig_device)

        layer_kv_shapes: list[list[int]] = []
        if self.use_cache:
            for layer in language_model.layers:
                attn = layer.self_attn
                layer_kv_shapes.append(
                    [
                        1,
                        attn.k_proj.out_features // attn.head_dim,
                        self.config.context_max_length,
                        attn.head_dim,
                    ]
                )
        self._kvcache_mixin.set_layer_kv_shapes(layer_kv_shapes)

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        return Gemma4MoeWithMaskInputProcessor(
            BaseInputProcessorConfig(
                embed_tokens=self.embed_tokens,
                input_sequence_length=self.wrap_cfg.input_sequence_length,
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                pad_token_id=self.pad_token_id,
            ),
            self.sliding_window_cfg,
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = super().get_export_cfg()
        insert_at = 3
        if self.sliding_window_cfg.get("has_local_attention", False):
            export_cfg["input_names"].insert(insert_at, "local_attention_mask")
            insert_at += 1
        if self.sliding_window_cfg.get("has_global_attention", False):
            export_cfg["input_names"].insert(insert_at, "global_attention_mask")
        if bool(self.wrap_cfg.get("output_hidden_states_for_export", False)):
            export_cfg["output_names"].append("last_hidden_state")
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info):
        meta_info.sliding_window_cfg = self.sliding_window_cfg
        meta_info.kv_cache_shapes_per_layer = list(self.get_kvcache_mixin().layer_kv_shapes)
        return meta_info
