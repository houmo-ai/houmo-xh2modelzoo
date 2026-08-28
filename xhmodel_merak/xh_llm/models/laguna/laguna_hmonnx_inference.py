from ...hmonnx import TextLLMHMONNXModel
from ...llm_data_processor import BaseInputProcessorConfig
from .data_preprocess import LagunaDataPreprocess, LagunaSlidingMaskForwardMixin
from .kv_cache import LagunaKVCacheMixinHMONNX


class XHLagunaHMONNXModel(LagunaSlidingMaskForwardMixin, TextLLMHMONNXModel):
    def __init__(self, meta, **kwargs):
        super().__init__(meta, **kwargs)
        self.uses_explicit_sliding_attention_mask = bool(
            getattr(meta, "uses_explicit_sliding_attention_mask", False)
        )
        self.sliding_window = int(getattr(meta, "sliding_window", 0) or 0)
        layer_kv_shapes = [list(shape) for shape in (getattr(meta, "layer_kv_shapes", []) or [])]
        if layer_kv_shapes:
            self._kvcache_mixin = LagunaKVCacheMixinHMONNX(self.kvcache_config)
            self._kvcache_mixin.set_layer_kv_shapes(layer_kv_shapes)
            self._sync_page_attention_mode_to_kvcache()

    def _get_data_preprocessor(self):
        if not self.uses_explicit_sliding_attention_mask:
            return super()._get_data_preprocessor()
        return LagunaDataPreprocess(
            BaseInputProcessorConfig(
                embed_tokens=self.get_input_embeddings(),
                input_sequence_length=self.get_input_sequence_length(),
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                enable_page_attention=self.enable_page_attention,
                pad_token_id=self.pad_token_id,
            ),
            sliding_window=self.sliding_window,
        )

    def get_tokenizer(self, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("fix_mistral_regex", True)
        return super().get_tokenizer(**kwargs)
