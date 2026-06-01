import copy
from typing import Any

from .base_llm_model import BaseLLMModel
from .text_llm_hf_compatible import TextLLMInferenceMixin, build_text_llm_hf_compatible
from .types import TextLLMModelConfig


class TextLLMModel(BaseLLMModel, TextLLMInferenceMixin):
    """
    文本模型的输入参数:
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: list,
        past_value_caches: list,
    """

    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_text_llm_hf_compatible)

    def __init__(
        self,
        config: TextLLMModelConfig,
    ):
        super().__init__(config)

    def _get_language_model(self, hf_model: Any) -> Any:
        if not hasattr(hf_model, "model"):
            llm_model = hf_model
        else:
            llm_model = hf_model.model
        return llm_model

    def _wraped_post(self, hf_model: Any):
        super()._wraped_post(hf_model)
        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)

        # self.embed_tokens.weight 和 lm_head.weight 可能是相同对象
        self.embed_tokens = copy.deepcopy(llm_model.get_input_embeddings())

        self.pad_token_id = llm_model.config.eos_token_id
        if self.use_cache:
            num_decoder_layers = llm_model.config.num_hidden_layers
            head_dim = llm_model.layers[0].self_attn.head_dim
            max_layers = self.config.get_max_decode_layers()
            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers
            self.kvcache_config.num_layers = num_decoder_layers
            self.kvcache_config.kv_cache_shape = [
                1,
                llm_model.config.num_key_value_heads,
                self.config.context_max_length,
                head_dim,
            ]

        hf_model = None
