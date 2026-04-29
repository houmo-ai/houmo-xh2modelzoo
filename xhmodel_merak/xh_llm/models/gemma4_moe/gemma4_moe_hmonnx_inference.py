from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from xhquant.api import ConfigDict, get_xhquant_logger

from ...hmonnx.text_llm_hmonnx_model import TextLLMHMONNXModel
from ...kv_cache_mixin import KVCacheMixin
from ..gemma4.data_preprocess import Gemma4DataPreprocess


class Gemma4MoeKVCacheMixinHMONNX(KVCacheMixin):
    def __init__(self, kv_cache_config, layer_kv_shapes: list[list[int]]):
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


def aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


def _prepare_window_attention_mask(inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    nq, nk = inputs_tensor.size(-2), inputs_tensor.size(-1)
    attention_mask = torch.ones([1, nq, nk], device=inputs_tensor.device, dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return attention_mask


def _gen_mask_v2(x: Tensor, valid_length: int | Tensor, attention_max_length: int = -1) -> Tensor:
    if isinstance(valid_length, int):
        valid_length = torch.tensor(valid_length, device=x.device)
    valid_length = valid_length.reshape(-1)
    if x.shape[0] != valid_length.numel() or (valid_length[0].item() == 0 and valid_length.shape[0] == 2):
        return _prepare_window_attention_mask(x, valid_length)

    bsz, nq, nk = x.size(0), x.size(-2), x.size(-1)
    masks = []
    for i in range(bsz):
        b_valid_length = int(valid_length[i].item())
        if attention_max_length > 0:
            b_valid_length = min(b_valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(nq, nk, dtype=torch.bool, device=x.device), diagonal=b_valid_length
        ).logical_not()
        if attention_max_length > 0:
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=b_valid_length - attention_max_length
            )
            attention_mask = torch.where(sliding_window_mask, True, attention_mask)
        masks.append(attention_mask.unsqueeze(0).unsqueeze(0))
    return torch.cat(masks, dim=0)


class XHGemma4MoeWithMaskHMONNXModel(TextLLMHMONNXModel):
    def __init__(self, meta):
        super().__init__(meta)
        self._maybe_load_baked_token_embedding()
        sliding_window_cfg = getattr(meta, "sliding_window_cfg", None)
        layer_kv_shapes = getattr(meta, "kv_cache_shapes_per_layer", [])
        if layer_kv_shapes:
            self._kvcache_mixin = Gemma4MoeKVCacheMixinHMONNX(self.kvcache_config, layer_kv_shapes)
        if sliding_window_cfg is None:
            model_config = getattr(meta, "model_config", None)
            sliding_window_cfg = getattr(model_config, "sliding_window_cfg", None)
        if sliding_window_cfg is None:
            sliding_window_cfg = {}
        self.sliding_window_cfg = ConfigDict(sliding_window_cfg)

    def _maybe_load_baked_token_embedding(self) -> None:
        prefill_path = Path(self.meta_info.prefill_hmonnx).resolve()
        export_root = prefill_path.parents[2]
        token_embedding_path = export_root / "token_embedding.pt"
        if not token_embedding_path.exists():
            return

        try:
            state_dict = torch.load(str(token_embedding_path), map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(str(token_embedding_path), map_location="cpu", weights_only=False)
        if isinstance(state_dict, torch.nn.Embedding):
            state_dict = state_dict.state_dict()

        self.embed_tokens.load_state_dict(state_dict)
        logger = get_xhquant_logger()
        logger.info(f"Loaded baked token embedding from {token_embedding_path}")

    def forward(self, *args):
        args = [arg.to(torch.int32) if getattr(arg, "dtype", None) == torch.int64 else arg for arg in args]
        out = super().forward(*args)
        if isinstance(out, (tuple, list)) and len(out) == 1:
            return out[0]
        return out

    def _get_data_preprocessor(self) -> Gemma4DataPreprocess:
        model_config = self.meta_info.model_config
        return Gemma4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            context_length=model_config.context_max_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=self.pad_token_id,
            image_token_id=getattr(model_config, "image_token_id", -1) or -1,
            sliding_window=self.sliding_window_cfg.get("sliding_window", 1024),
        )

    def prepare_casual_mask(self, x: Tensor, valid_length: int | Tensor, attention_max_length: int) -> Tensor:
        mask = _gen_mask_v2(x, valid_length, attention_max_length)
        attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)
        return attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)

    def prepare_attention_masks(
        self,
        inputs_embeds: Tensor,
        past_seq_length: int | Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        bz, nq = inputs_embeds.shape[:2]
        local_attention_mask = None
        global_attention_mask = None
        if self.sliding_window_cfg.get("has_global_attention", False):
            global_window = self.sliding_window_cfg.get(
                "global_attention_window_size",
                self.meta_info.model_config.context_max_length,
            )
            x = torch.empty((bz, nq, global_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            global_attention_mask = self.prepare_casual_mask(x, past_seq_length, -1)
        if self.sliding_window_cfg.get("has_local_attention", False):
            local_window = self.sliding_window_cfg.get("local_attention_window_size", 1024) + nq - 1
            local_window = aligned(local_window, 16)
            x = torch.empty((bz, nq, local_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            local_attention_mask = self.prepare_casual_mask(
                x,
                past_seq_length,
                self.sliding_window_cfg.get("sliding_window", 1024),
            )
        return local_attention_mask, global_attention_mask

    def prepare_inputs_with_masks(self, data: dict[str, Any]) -> tuple:
        data_processor = self.get_data_preprocessor()
        (
            inputs_embeds,
            past_seq_length,
            seq_length,
            full_attention_mask,
            sliding_attention_mask,
            past_key_caches,
            past_value_caches,
        ) = data_processor(data)
        outputs = [inputs_embeds, past_seq_length, seq_length]
        if sliding_attention_mask is not None:
            outputs.append(sliding_attention_mask)
        if full_attention_mask is not None:
            outputs.append(full_attention_mask)
        outputs.extend(past_key_caches)
        outputs.extend(past_value_caches)
        return tuple(outputs)


Gemma4MoeWithMaskHMONNXModel = XHGemma4MoeWithMaskHMONNXModel
