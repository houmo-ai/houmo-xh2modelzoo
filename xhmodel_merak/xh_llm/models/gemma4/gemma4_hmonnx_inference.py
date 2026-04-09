from __future__ import annotations

import torch

from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheMixin
from xhmodel_merak.xh_llm.utils import unfold_args

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import KVCacheConfig, LLMModelMeta, VLLMModelMeta
from .data_preprocess import Gemma4DataPreprocess
from .gemma4_processor import XHGemma4Processor


class Gemma4VisualHMONNXModel(HMONNXModel):
    def forward(self, *args):
        return super().forward(*args)


class Gemma4KVCacheMixinHMONNX(KVCacheMixin):
    """KV-cache mixin that supports heterogeneous layer shapes (Gemma4 has
    sliding_attention + full_attention with different num_kv_heads / head_dim)."""

    def __init__(self, kv_cache_config: KVCacheConfig, layer_kv_shapes: list[list[int]]):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes = layer_kv_shapes

    def prepare_kv_cache(self, dtype=torch.float16):
        if not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for shape in self.layer_kv_shapes:
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))


class XHGemma4HMONNXModel(VisonLLMHMONNXModel):
    def __init__(self, meta_info: VLLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        self.visual = Gemma4VisualHMONNXModel(self.visual_meta.hmonnx)

        layer_kv_shapes = getattr(meta_info, "layer_kv_shapes", [])
        self._kvcache_mixin = Gemma4KVCacheMixinHMONNX(self.kvcache_config, layer_kv_shapes)
        self.layer_types = getattr(meta_info, "layer_types", [])
        self.sliding_window = getattr(meta_info, "sliding_window", 1024)

    def _set_device(self, device):
        super()._set_device(device)
        self.visual.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def get_tf_processor(self):
        return XHGemma4Processor.from_pretrained(self.hf_model_dir)

    def forward(self, *args):
        args = unfold_args(args)
        args = [arg.to(torch.int32) if arg.dtype == torch.int64 else arg for arg in args]
        outs = super().forward(*args)

        if isinstance(outs, (tuple, list)):
            logits = outs[0]
        else:
            logits = outs
        return logits

    def _get_data_preprocessor(self) -> Gemma4DataPreprocess:
        return Gemma4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            context_length=self.meta_info.model_config.context_max_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=self.pad_token_id,
            image_token_id=getattr(self.meta_info.model_config, "image_token_id", -1) or -1,
            sliding_window=self.sliding_window,
        )

    def _set_enable_golden(self, enable: bool) -> None:
        super()._set_enable_golden(enable)
        self.visual.enable_golden = enable
