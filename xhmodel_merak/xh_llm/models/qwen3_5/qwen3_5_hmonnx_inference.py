import torch

from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheWithLinearMixin
from xhquant.core import CacheTensor

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import LLMModelMeta
from .data_preprocess import Qwen3_5_DataPreprocess
from .hybrid_cache_runtime import (
    commit_hybrid_cache_outputs,
    get_spec_decode_verify_steps,
    model_config_prefill_recurrent_state_uses_cache,
    normalize_hybrid_hmonnx_args,
)
from .qwen3_5_processor import XHQwen3_5Processor
from .split_conv_cache_utils import (
    _flatten_split_conv_cache_outputs,
    _regroup_flat_split_conv_cache,  # noqa: F401 - compatibility re-export
)


def _model_config_prefill_recurrent_state_uses_cache(model_config) -> bool:
    """Compatibility alias for callers that imported the former local helper."""

    return model_config_prefill_recurrent_state_uses_cache(model_config)


class VisualHMONNXModel(HMONNXModel):
    def forward(self, *args):
        out = super().forward(*args)
        return out


class Qwen3_5HMONNXKVCacheMixin(KVCacheWithLinearMixin):  # noqa: N801
    """HMONNX cache mixin that mirrors the exported split q/k/v conv-cache signature."""

    def __init__(self, kv_cache_config) -> None:
        super().__init__(kv_cache_config)
        self.split_conv_cache: bool = False

    def prepare_other_cache(self):
        if not self.split_conv_cache:
            return super().prepare_other_cache()

        linear_cfg = self.kvcache_config.linear_kv_cache_config
        value_dim = linear_cfg.num_v_heads * linear_cfg.head_v_dim
        key_dim = (linear_cfg.conv_dim - value_dim) // 2
        if key_dim <= 0 or linear_cfg.conv_dim != key_dim * 2 + value_dim:
            raise RuntimeError(
                "Invalid linear kv cache config for split conv cache: "
                f"conv_dim={linear_cfg.conv_dim}, value_dim={value_dim}"
            )

        cache_dtype = linear_cfg.cache_torch_dtype
        batch_size = linear_cfg.batch_size
        kernel_size = linear_cfg.conv_kernel_size
        for _i in range(linear_cfg.num_layers):
            conv_cache_q = CacheTensor(
                torch.zeros(
                    batch_size,
                    key_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            conv_cache_k = CacheTensor(
                torch.zeros(
                    batch_size,
                    key_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            conv_cache_v = CacheTensor(
                torch.zeros(
                    batch_size,
                    value_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            self.past_conv_caches.append((conv_cache_q, conv_cache_k, conv_cache_v))

            recurrent_cache_shape = [
                batch_size,
                linear_cfg.num_v_heads,
                linear_cfg.head_k_dim,
                linear_cfg.head_v_dim,
            ]
            self.past_recurrent_states.append(
                CacheTensor(torch.zeros(recurrent_cache_shape, dtype=cache_dtype, device=self._device))
            )

    def _set_device(self, device):
        self._device = device
        for cache_tensor in _flatten_split_conv_cache_outputs(self.past_conv_caches):
            cache_tensor.to(device)
        for cache_tensor in self.past_recurrent_states:
            cache_tensor.to(device)


class XHQwen3_5_HMONNXModel(VisonLLMHMONNXModel):  # noqa: N801
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        enable_golden = kwargs.get("enable_golden", False)
        self.visual = VisualHMONNXModel(
            self.visual_meta.hmonnx, device_map=[self.prefill_model.device], enable_golden=enable_golden
        )
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(getattr(meta_info.model_config, "split_conv_cache", False))
        self._sync_page_attention_mode_to_kvcache()

    @property
    def past_conv_caches(self):
        return self._kvcache_mixin.past_conv_caches

    @property
    def past_recurrent_states(self):
        return self._kvcache_mixin.past_recurrent_states

    def _set_device(self, device):
        super()._set_device(device)
        self.visual.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def _get_spec_decode_verify_steps(self) -> int:
        return get_spec_decode_verify_steps(self.meta_info)

    def _prefill_recurrent_state_uses_cache(self) -> bool:
        return _model_config_prefill_recurrent_state_uses_cache(self.meta_info.model_config)

    def get_input_sequence_length(self) -> int:
        if getattr(self, "_llm_prefill", True):
            return self.meta_info.model_config.prefill_chunk_length
        return self._get_spec_decode_verify_steps()

    def get_tf_processor(self):
        processor = XHQwen3_5Processor.from_pretrained(self.hf_model_dir)
        meta_info = self.meta_info.model_config
        processor.config.patch_size = meta_info.visual_config.patch_size
        processor.config.max_size_h = meta_info.visual_config.max_size_h
        processor.config.max_size_w = meta_info.visual_config.max_size_w
        return processor

    def to_fast(self):
        """转换为快速推理模式，返回一个新的模型实例。"""
        # 默认实现直接返回自己，子类可以重写此方法以支持快速推理模式
        if self.fast_mode:
            return self
        self.fast_mode = True
        self.prefill_model.to_fast()
        # self.decode_model.to_fast()
        # self.visual.to_fast()
        return self

    def forward(self, *args):
        args = normalize_hybrid_hmonnx_args(args)
        outs = super().forward(*args)

        return commit_hybrid_cache_outputs(self, outs, model_label="Qwen3.5")

    def _get_data_preprocessor(self) -> Qwen3_5_DataPreprocess:
        input_sequence_length = self.get_input_sequence_length()

        data_preprocess = Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=input_sequence_length,
            image_size_w=self.visual_meta.max_size_w,
            image_size_h=self.visual_meta.max_size_h,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=self.visual_meta.patch_size,
            # temporal_patch_size=self.visual_meta.temporal_patch_size,
            image_token_id=self.meta_info.model_config.image_token_id,
            video_token_id=self.meta_info.model_config.video_token_id,
            vision_start_token_id=self.meta_info.model_config.vision_start_token_id,
            vision_end_token_id=self.meta_info.model_config.vision_end_token_id,
            spatial_merge_size=self.meta_info.model_config.spatial_merge_size,
            enable_page_attention=self.enable_page_attention,
        )
        return data_preprocess

    def _set_enable_golden(self, enable: bool) -> None:
        super()._set_enable_golden(enable)
        self.visual.enable_golden = enable
