import copy
from typing import Any

from transformers import AutoModelForCausalLM
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM

from xhmodel_merak.configuration_utils import BaseAttrDict

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from ...types import LLMModelMeta
from .glm_4_moe_lite_hmonnx_inference import XHGlm4MoeLiteHMONNXModel


class XHGlm4MoeLiteModelConfig(TextLLMModelConfig):
    pass


class XHGlm4MoeLiteModelMeta(LLMModelMeta):
    pass


@register_llm_model("Glm4MoeLiteForCausalLM")
class XHGlm4MoeLiteModel(TextLLMModel):
    transformers_min_version = "5.2.0"
    HF_MODEL_CLS = Glm4MoeLiteForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHGlm4MoeLiteHMONNXModel
    CONFIG_CLS = XHGlm4MoeLiteModelConfig
    META_CLS = XHGlm4MoeLiteModelMeta

    def __init__(self, config: XHGlm4MoeLiteModelConfig):
        super().__init__(config)
        self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as glm_register_wrap_modules

        glm_register_wrap_modules()
        return super().init_wrap_model(hf_model)

    @classmethod
    def _postprocess_gptqmodel_structure(cls, native_hf_model: Any, **kwargs) -> Any:
        from .gptqmodel_compat import convert_gptqmodel_moe_structure

        convert_gptqmodel_moe_structure(native_hf_model)
        return native_hf_model

    def _wraped_post(self, hf_model: Glm4MoeLiteForCausalLM):
        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)
        self.embed_tokens = copy.deepcopy(llm_model.get_input_embeddings())

        eos_token_id = llm_model.config.eos_token_id
        self.pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id

        if self.use_cache:
            num_decoder_layers = llm_model.config.num_hidden_layers
            max_layers = self.config.get_max_decode_layers()
            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers

            first_attn = llm_model.layers[0].self_attn
            qk_rope_head_dim = int(getattr(llm_model.config, "qk_rope_head_dim", first_attn.qk_rope_head_dim))
            kv_lora_rank = int(getattr(llm_model.config, "kv_lora_rank", first_attn.kv_lora_rank))
            compress_kv_dim = kv_lora_rank + qk_rope_head_dim
            batch_size = self.config.batch_size

            self.kvcache_config.num_layers = num_decoder_layers
            key_cache_shape = [
                batch_size,
                1,
                self.config.context_max_length,
                compress_kv_dim,
            ]
            value_cache_shape = [
                batch_size,
                1,
                self.config.context_max_length,
                kv_lora_rank,
            ]
            self.kvcache_config.kv_cache_shape = [key_cache_shape, value_cache_shape]
            self.kvcache_config.batch_size = batch_size
            self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

        hf_model = None
