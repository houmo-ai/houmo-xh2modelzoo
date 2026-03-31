import copy
from typing import Any

from transformers import AutoModelForCausalLM

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .spark import IPTForCausalLM
from .spark_moe_hmonnx_inference import XHSparkMoeHMONNXModel


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
                # MLA: K cache uses qk_head_dim, V cache uses v_head_dim
                # Use qk_head_dim for the base cache shape
                self.kvcache_config.kv_cache_shape = [
                    1,
                    llm_model.config.num_attention_heads,
                    self.config.context_max_length,
                    layer0_attn.qk_head_dim,
                ]
                self._mla_v_head_dim = layer0_attn.v_head_dim
            else:
                self.kvcache_config.kv_cache_shape = [
                    1,
                    llm_model.config.num_key_value_heads,
                    self.config.context_max_length,
                    layer0_attn.head_dim,
                ]
