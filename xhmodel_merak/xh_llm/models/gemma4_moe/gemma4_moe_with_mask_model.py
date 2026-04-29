from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForImageTextToText, AutoTokenizer, Gemma4ForConditionalGeneration

from xhmodel_merak.xh_llm.llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor

from ...base_llm_model import XHLLMModelProcessor
from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheMixin
from ...vision_llm_model import VisionLLMModel
from .gemma4_moe_hf_compatible import build_gemma4_moe_with_mask_hf_compatible_model
from .gemma4_moe_hmonnx_inference import XHGemma4MoeWithMaskHMONNXModel, _gen_mask_v2, aligned
from .gemma4_moe_visual_model import XHGemma4MoeVisualModel
from .xh_gemma4_moe_config import XHGemma4MoeWithMaskConfig


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
        return self.get_hf_model(self.hf_model_dir, quant_weight=self.config.quant_weight)

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
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info):
        meta_info.sliding_window_cfg = self.sliding_window_cfg
        meta_info.kv_cache_shapes_per_layer = list(self.get_kvcache_mixin().layer_kv_shapes)
        return meta_info
