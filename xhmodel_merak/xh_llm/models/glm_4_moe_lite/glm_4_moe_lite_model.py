import copy
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM

from xhmodel_merak.configuration_utils import BaseAttrDict

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from ...types import LLMModelMeta
from .cache import Glm4MoeLiteKVCacheConfig, Glm4MoeLiteKVCacheMixin
from .glm_4_moe_lite_hmonnx_inference import XHGlm4MoeLiteHMONNXModel


class XHGlm4MoeLiteModelConfig(TextLLMModelConfig):
    pass


class XHGlm4MoeLiteModelMeta(LLMModelMeta):
    KVCACHE_CONFOG_CLS = Glm4MoeLiteKVCacheConfig


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
        self._kvcache_config = Glm4MoeLiteKVCacheConfig()
        self._kvcache_mixin = Glm4MoeLiteKVCacheMixin(self.kvcache_config)
        self.use_cache = config.use_cache
        self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules as glm_register_wrap_modules

        glm_register_wrap_modules()
        return super().init_wrap_model(hf_model)

    @classmethod
    def _should_use_local_gptqmodel_path(cls, config: Any, hf_model_dir: str) -> bool:
        from .gptqmodel_compat import detect_gptqmodel_moe_format

        quantization_config = getattr(config, "quantization_config", None)
        quant_method = getattr(quantization_config, "quant_method", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method", quant_method)
        return str(quant_method).lower() == "gptq" and detect_gptqmodel_moe_format(hf_model_dir)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        # 只对 GLM split-MoE GPTQModel checkpoint 走本地化导出路径；
        # 其他路径全部回退给框架原生实现，避免把模型私有逻辑扩散到公共层。
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        if not cls._should_use_local_gptqmodel_path(config, hf_model_dir):
            return super().get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)

        assert quant_weight is None or len(quant_weight) == 0, (
            "Model is already quantized, quant_weight should be None or empty when loading quantized model."
        )

        from .gptqmodel_compat import convert_gptqmodel_moe_structure, glm_gptqmodel_load_context
        from .gptqmodel_dequant import dequantize_gptqmodel_linears

        with glm_gptqmodel_load_context(hf_model_dir, **kwargs):
            hf_model = cls._load_gptqmodel(hf_model_dir, **kwargs)
        hf_model = dequantize_gptqmodel_linears(hf_model, hf_model_dir, keep_quant_weight=False)
        convert_gptqmodel_moe_structure(hf_model)
        return hf_model

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
            batch_size = self.config.batch_size

            self.kvcache_config.num_layers = num_decoder_layers
            self.kvcache_config.key_cache_shape = [
                batch_size,
                1,
                self.config.context_max_length,
                qk_rope_head_dim,
            ]
            self.kvcache_config.value_cache_shape = [
                batch_size,
                1,
                self.config.context_max_length,
                kv_lora_rank,
            ]
            self.kvcache_config.kv_cache_shape = self.kvcache_config.value_cache_shape
            self.kvcache_config.batch_size = batch_size
            self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

        hf_model = None
