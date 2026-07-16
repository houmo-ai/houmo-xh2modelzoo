from __future__ import annotations

import torch

from xhquant.core.hmfp_kv_cache import allocate_hmfp_paged_kv_cache

from ...hmonnx import TextLLMHMONNXModel
from ...types import LLMModelMeta
from ..qwen3_5.hybrid_cache_runtime import (
    commit_hybrid_cache_outputs,
    get_spec_decode_verify_steps,
    model_config_prefill_recurrent_state_uses_cache,
    normalize_hybrid_hmonnx_args,
)
from ..qwen3_5.qwen3_5_hmonnx_inference import Qwen3_5HMONNXKVCacheMixin
from .data_preprocess import Qwen3NextDataPreprocess


class XHQwen3NextHMONNXModel(TextLLMHMONNXModel):
    """Hybrid full-attention/GDN HMONNX runtime for text-only Qwen3-Next."""

    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(getattr(meta_info.model_config, "split_conv_cache", True))
        self._sync_page_attention_mode_to_kvcache()
        self._paged_kv_caches = None
        self._page_attention_block_size = 64

    def prepare_page_attention_context(self, past_seq_length: int, current_input_length: int) -> None:
        """Bind identity-mapped paged caches for standalone generation.

        Production schedulers provide their own physical block allocation and
        slot mapping through ``set_page_attention_context``.  The example
        runner owns one request, so an identity mapping is sufficient and
        exercises the exact same converted PageAttention graph.
        """
        if not self.enable_page_attention:
            return
        cache_shape = self.kvcache_config.kv_cache_shape
        cache_axis = self.kvcache_config.cache_axis
        context_length = int(cache_shape[cache_axis])
        past_seq_length = int(past_seq_length)
        current_input_length = int(current_input_length)
        graph_length = int(self.get_input_sequence_length())
        if past_seq_length < 0:
            raise ValueError(f"past_seq_length must be non-negative, got {past_seq_length}")
        if not 0 <= current_input_length <= graph_length:
            raise ValueError(
                "current_input_length must fit the fixed page-attention graph: "
                f"got {current_input_length}, graph length {graph_length}"
            )
        end_position = past_seq_length + current_input_length
        if end_position > context_length:
            raise ValueError(f"Page-attention request position {end_position} exceeds context length {context_length}")

        device = torch.device(self.device)
        block_size = self._page_attention_block_size
        num_blocks = (context_length + block_size - 1) // block_size
        caches = self._paged_kv_caches
        if caches is None or any(torch.device(cache.device) != device for cache in caches):
            caches = [
                allocate_hmfp_paged_kv_cache(
                    num_blocks=num_blocks,
                    block_size=block_size,
                    num_kv_heads=int(cache_shape[1]),
                    head_size=int(cache_shape[-1]),
                    device=device,
                )
                for _ in range(self.kvcache_config.num_layers)
            ]
            self._paged_kv_caches = caches

        block_ids = torch.arange(num_blocks, dtype=torch.int64, device=device)
        # PageAttention consumes graph-shaped K/V tensors, including padded MTP
        # verify positions.  Keep the metadata shape/address capture-stable and
        # mark padded positions as invalid so its slot-guarded cache writers do
        # not store them.  For example decode T=5/current=1 maps to
        # ``[past, -1, -1, -1, -1]`` rather than a one-element buffer.
        slot_mapping = torch.full((graph_length,), -1, dtype=torch.int64, device=device)
        if current_input_length:
            slot_mapping[:current_input_length] = torch.arange(
                past_seq_length,
                end_position,
                dtype=torch.int64,
                device=device,
            )
        self.set_page_attention_context(caches, block_ids, slot_mapping, block_size)

    @property
    def past_conv_caches(self):
        return self._kvcache_mixin.past_conv_caches

    @property
    def past_recurrent_states(self):
        return self._kvcache_mixin.past_recurrent_states

    def _prefill_recurrent_state_uses_cache(self) -> bool:
        return model_config_prefill_recurrent_state_uses_cache(self.meta_info.model_config)

    def _get_spec_decode_verify_steps(self) -> int:
        return get_spec_decode_verify_steps(self.meta_info)

    def get_input_sequence_length(self) -> int:
        if self.is_prefill():
            return self.meta_info.model_config.prefill_chunk_length
        return self._get_spec_decode_verify_steps()

    def forward(self, *args):
        args = normalize_hybrid_hmonnx_args(args)
        outputs = super().forward(*args)
        return commit_hybrid_cache_outputs(self, outputs, model_label="Qwen3-Next")

    def _get_data_preprocessor(self) -> Qwen3NextDataPreprocess:
        return Qwen3NextDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            enable_page_attention=self.enable_page_attention,
            pad_token_id=self.pad_token_id,
        )
