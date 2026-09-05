import torch

from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import LLMModelMeta
from ..qwen3_5.hybrid_cache_runtime import (
    commit_hybrid_cache_outputs,
    get_spec_decode_verify_steps,
    model_config_prefill_recurrent_state_uses_cache,
    normalize_hybrid_hmonnx_args,
)
from ..qwen3_5.qwen3_5_hmonnx_inference import (
    Qwen3_5HMONNXKVCacheMixin,
    VisualTokenGearHMONNXModel,
)
from ..qwen3_5.qwen3_5_processor import XHQwen3_5Processor
from ..qwen3_5.visual_token_gears import VISUAL_INPUT_PATCHES
from .data_preprocess import Qwen3_5_DataPreprocess


def _model_config_prefill_recurrent_state_uses_cache(model_config) -> bool:
    """Compatibility alias for the shared hybrid-cache contract helper."""

    return model_config_prefill_recurrent_state_uses_cache(model_config)


class XHQwen3_5MoeHMONNXModel(VisonLLMHMONNXModel):  # noqa: N801
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        enable_golden = kwargs.get("enable_golden", False)
        if not getattr(self.visual_meta, "gears", None):
            raise ValueError("Qwen3.5 MoE visual metadata must contain patch-token gears")
        self.visual = VisualTokenGearHMONNXModel(
            self.visual_meta,
            device=self.prefill_model.device,
            enable_golden=enable_golden,
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
        processor.config.patch_size = self.visual_meta.patch_size
        processor.config.visual_input_mode = VISUAL_INPUT_PATCHES
        max_patch_capacity = max(int(gear.patch_token_capacity) for gear in self.visual_meta.gears)
        processor.config.max_pixels = max_patch_capacity * int(self.visual_meta.patch_size) ** 2
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

        return commit_hybrid_cache_outputs(self, outs, model_label="Qwen3.5-MoE")

    def _get_data_preprocessor(self) -> Qwen3_5_DataPreprocess:
        input_sequence_length = self.get_input_sequence_length()

        data_preprocess = Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=input_sequence_length,
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

    def _make_position_ids(self, seq_length: int, past_seq_length: int, device: torch.device):
        pos = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long, device=device)
        return pos, pos, pos

    def prepare_text_only_inputs(self, input_ids: torch.Tensor, input_sequence_length: int, past_seq_length: int = 0):
        if input_ids.shape[0] != 1:
            raise ValueError("Batch size must be 1 in HMONNX inference mode.")

        device = self.device
        seq_length = input_ids.shape[1]
        if seq_length > input_sequence_length:
            raise ValueError(
                f"Input sequence length ({seq_length}) exceeds max input sequence length ({input_sequence_length})"
            )

        input_ids = input_ids.to(device)
        if input_sequence_length > seq_length:
            padding = torch.full(
                (1, input_sequence_length - seq_length),
                self.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            input_ids = torch.cat([input_ids, padding], dim=-1)

        inputs_embeds = self.embed_tokens.to(device)(input_ids)
        time_pos, hight_pos, width_pos = self._make_position_ids(seq_length, past_seq_length, device)
        if input_sequence_length > seq_length:
            pad_len = input_sequence_length - seq_length
            last_pos = time_pos[-1:] if seq_length > 0 else torch.zeros(1, dtype=torch.long, device=device)
            pos_pad = last_pos.expand(pad_len)
            time_pos = torch.cat([time_pos, pos_pad])
            hight_pos = torch.cat([hight_pos, pos_pad])
            width_pos = torch.cat([width_pos, pos_pad])

        return (
            inputs_embeds,
            time_pos.to(torch.int32),
            hight_pos.to(torch.int32),
            width_pos.to(torch.int32),
            torch.tensor([past_seq_length], dtype=torch.int32, device=device),
            torch.tensor([seq_length], dtype=torch.int32, device=device),
            self.past_key_caches,
            self.past_value_caches,
            self.past_conv_caches,
            self.past_recurrent_states,
        )

    @torch.no_grad()
    def prefill_only(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        seq_len = input_ids.shape[1]
        if seq_len == 0:
            return None

        isl = self.meta_info.model_config.prefill_chunk_length
        self.set_prefill()
        self.set_input_sequence_length(isl)

        saved_conv = [cache.data.clone() for cache in self.past_conv_caches]
        saved_rec = [cache.data.clone() for cache in self.past_recurrent_states]
        all_logits: list[torch.Tensor] = []

        pos = 0
        while pos < seq_len:
            chunk_end = min(pos + isl, seq_len)
            chunk_len = chunk_end - pos
            chunk = input_ids[:, pos:chunk_end]

            for idx, cache in enumerate(self.past_conv_caches):
                cache.data = saved_conv[idx].clone()
            for idx, cache in enumerate(self.past_recurrent_states):
                cache.data = saved_rec[idx].clone()

            (
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_length_t,
                seq_length_t,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            ) = self.prepare_text_only_inputs(chunk, isl, past_seq_length=0)
            linear_mask = torch.ones(1, isl, dtype=inputs_embeds.dtype, device=self.device)

            out = self.prefill_model(
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_length_t,
                seq_length_t,
                linear_mask,
                *past_key_caches,
                *past_value_caches,
                *past_conv_caches,
                *past_recurrent_states,
            )
            logits = out[0] if isinstance(out, (tuple, list)) else out
            all_logits.append(logits[:, :chunk_len, :])
            pos = chunk_end

        for idx, cache in enumerate(self.past_conv_caches):
            cache.data = saved_conv[idx]
        for idx, cache in enumerate(self.past_recurrent_states):
            cache.data = saved_rec[idx]

        return torch.cat(all_logits, dim=1) if all_logits else None
