import copy
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM

from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheMixin
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from ...types import KVCacheConfig
from .spark import IPTForCausalLM
from .spark_moe_hmonnx_inference import XHSparkMoeHMONNXModel


class _MLAKVCacheMixin(KVCacheMixin):
    """KVCacheMixin variant that supports separate k/v cache shapes for absorbed MLA."""

    def __init__(self, kv_cache_config: KVCacheConfig, v_cache_shape: list[int]):
        super().__init__(kv_cache_config)
        self._v_cache_shape = v_cache_shape

    def prepare_kv_cache(self):
        if not self.use_cache:
            return
        if self.kvcache_config.num_layers <= 0:
            return
        k_shape = self.kvcache_config.kv_cache_shape
        v_shape = self._v_cache_shape
        cache_dtype = self.kvcache_config.cache_torch_dtype
        for _i in range(self.kvcache_config.num_layers):
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(k_shape, dtype=cache_dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(v_shape, dtype=cache_dtype)))


class XHSparkModeModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("IPTForCausalLM")
class XHSparkModeModel(TextLLMModel):
    transformers_min_version = "4.51.0"
    transformers_max_version = "4.57.6"
    HF_MODEL_CLS = IPTForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHSparkMoeHMONNXModel
    CONFIG_CLS = XHSparkModeModelConfig

    def __init__(self, config: XHSparkModeModelConfig):
        super().__init__(config)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as sparkmoe_register_wrap_modules

        sparkmoe_register_wrap_modules()
        super().init_wrap_model(hf_model)

    def _wraped_post(self, hf_model: Any):
        """Override to handle IPT model structure (transformer.layers, MLA cache shapes)."""
        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)

        self.embed_tokens = copy.deepcopy(llm_model.get_input_embeddings())
        self.pad_token_id = llm_model.config.eos_token_id

        if self.use_cache:
            num_decoder_layers = llm_model.config.num_hidden_layers
            max_layers = self.config.get_max_decode_layers()
            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers
            self.kvcache_config.num_layers = num_decoder_layers

            layer0_attn = llm_model.transformer.layers[0].attention
            if llm_model.config.apply_mla:
                # Absorbed MLA: k cache stores rope vectors, v cache stores latent vectors
                k_cache_shape = [
                    1,
                    1,
                    self.config.context_max_length,
                    layer0_attn.qk_rope_head_dim,
                ]
                v_cache_shape = [
                    1,
                    1,
                    self.config.context_max_length,
                    layer0_attn.kv_lora_rank,
                ]
                self.kvcache_config.kv_cache_shape = [k_cache_shape, v_cache_shape]
                # Replace the default mixin with MLA-aware variant
                # self._kvcache_mixin = _MLAKVCacheMixin(self.kvcache_config, v_cache_shape)
            else:
                self.kvcache_config.kv_cache_shape = [
                    1,
                    llm_model.config.num_key_value_heads,
                    self.config.context_max_length,
                    layer0_attn.head_dim,
                ]

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        hf_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        hf_model = cast(IPTForCausalLM, hf_model)
        hf_model.unfuse_experts(use_padding_expert=False)
        hf_model.unfuse_mlp()
        return hf_model
