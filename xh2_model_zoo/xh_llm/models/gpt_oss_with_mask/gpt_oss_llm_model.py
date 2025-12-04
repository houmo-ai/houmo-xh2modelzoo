from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Union, cast

import torch
from torch import Tensor
from transformers.models.gpt_oss import GptOssForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from xhquant.api import QuantGraph
from xhquant.core import HybridCacheTensor,CacheTensor

from ..builder import MODELS
from ..llm_base_model import LLMBaseModel


def _prepare_window_attention_mask(inputs_tensor: Tensor, cu_seqlens: Tensor) -> Tensor:
    nq, nk = inputs_tensor.size(-2), inputs_tensor.size(-1)
    attention_mask = torch.ones(
        (1, nq, nk),
        device=inputs_tensor.device,
        dtype=torch.bool,
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return attention_mask


def _gen_mask_v2(x: Tensor, valid_length: Union[int, Tensor], attention_max_length: int = -1):
    if isinstance(valid_length, int):
        valid_length = torch.tensor(valid_length).to(x.device)
    valid_length = valid_length.reshape(-1)
    if x.shape[0] != valid_length.numel() or (valid_length[0].item() == 0 and valid_length.shape[0] == 2):
        return _prepare_window_attention_mask(x, valid_length)

    bsz, nq, nk = x.size(0), x.size(-2), x.size(-1)
    masks: List[Tensor] = []
    for i in range(bsz):
        b_valid_length = int(valid_length[i].item())
        if attention_max_length > 0:
            b_valid_length = min(b_valid_length, attention_max_length - 1)
        attention_mask = torch.tril(
            torch.ones(nq, nk, dtype=torch.bool, device=x.device),
            diagonal=b_valid_length,
        ).logical_not()
        if attention_max_length > 0:
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool),
                diagonal=b_valid_length - attention_max_length,
            )
            attention_mask = torch.where(sliding_window_mask, True, attention_mask)
        masks.append(attention_mask.unsqueeze(0).unsqueeze(0))
    return torch.cat(masks, dim=0)


def aligned(size: int, align: int) -> int:
    return ((size + align - 1) // align) * align


@dataclass
class SlidingWindowConfig:
    sliding_window: int = -1
    local_attention_window_size: int = -1
    global_attention_window_size: int = -1
    has_global_attention: bool = False
    has_local_attention: bool = False


@MODELS.register_module()
class XHGptOssWithMaskModel(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant: bool = True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        self.sliding_window_cfg = SlidingWindowConfig()
        if hasattr(wrap_cfg, "get"):
            sliding_window = wrap_cfg.get("sliding_window", -1)
        else:
            sliding_window = getattr(wrap_cfg, "sliding_window", -1)
        if sliding_window is None:
            sliding_window = -1
        self.sliding_window_cfg.sliding_window = int(sliding_window)

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def prepare_casual_mask(self, x: Tensor, valid_length: Union[int, Tensor], attention_max_length: int):
        mask = _gen_mask_v2(x, valid_length, attention_max_length)
        attention_mask = torch.zeros_like(mask, dtype=x.dtype, device=x.device)
        return attention_mask.masked_fill(mask, torch.finfo(x.dtype).min)

    def prepare_kv_cache(self, num_decoder_layers, kv_cache_shape):
        self.past_key_caches = []
        self.past_value_caches = []
        if not self.use_cache:
            return

        has_local_attention = False
        has_global_attention = False

        for i in range(num_decoder_layers):
            layer = self._wrap_model.model.layers[i].self_attn
            # if getattr(layer, "sliding_window", None) is not None:
            #     layer_kv_cache_shape = list(kv_cache_shape)
            #     layer_kv_cache_shape[2] = layer.sliding_window + self.get_input_sequence_length()
            # else:
            layer_kv_cache_shape = kv_cache_shape

            self.past_key_caches.append(CacheTensor(torch.zeros(layer_kv_cache_shape, dtype=torch.float16)))
            self.past_value_caches.append(CacheTensor(torch.zeros(layer_kv_cache_shape, dtype=torch.float16)))

            if getattr(layer, "sliding_window", None) is not None:
                self.sliding_window_cfg.local_attention_window_size = layer.sliding_window
                has_local_attention = True
            else:
                self.sliding_window_cfg.global_attention_window_size = layer_kv_cache_shape[2]
                has_global_attention = True

        self.sliding_window_cfg.has_local_attention = has_local_attention
        self.sliding_window_cfg.has_global_attention = has_global_attention
        if not self.sliding_window_cfg.has_local_attention:
            self.sliding_window_cfg.has_global_attention = False

        if self.sliding_window_cfg.has_local_attention and "local_attention_mask" not in self.export_cfg.input_names:
            self.export_cfg.input_names.append("local_attention_mask")
        if self.sliding_window_cfg.has_global_attention and "global_attention_mask" not in self.export_cfg.input_names:
            self.export_cfg.input_names.append("global_attention_mask")

        for layer_idx in range(num_decoder_layers):
            name = f"past_key_cache_{layer_idx}"
            if name not in self.export_cfg.input_names:
                self.export_cfg.input_names.append(name)
        for layer_idx in range(num_decoder_layers):
            name = f"past_value_cache_{layer_idx}"
            if name not in self.export_cfg.input_names:
                self.export_cfg.input_names.append(name)

    def init_wrap_model(self, hf_model=None):
        from ._model import register_wrap_modules as register_with_mask_modules

        register_with_mask_modules()

        super().init_wrap_model(hf_model)
        hf_model = self.wrap_model
        if isinstance(hf_model, GptOssForCausalLM):
            llm_model = hf_model.model
        else:
            raise ValueError(f"{type(hf_model)} is not supported")

        self.token_embedding = deepcopy(llm_model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = llm_model.config.num_hidden_layers
        head_dim = llm_model.layers[0].self_attn.head_dim
        self.pad_token_id = hf_model.config.eos_token_id
        self.head_dim = head_dim

        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [1, llm_model.config.num_key_value_heads, self.cache_length, head_dim],
            )

        hf_model = None

    def get_hf_model(self, device_map="cpu", **kwargs) -> GptOssForCausalLM:
        hf_model = super().get_hf_model(device_map, **kwargs)
        sliding_window = self.sliding_window_cfg.sliding_window
        # 总是设置 sliding_window，确保 LLMCache 的 attention_max_length 正确计算
        hf_model.model.config.sliding_window = sliding_window
        if sliding_window > 0:
            hf_model.model.config.use_sliding_window = True
        else:
            hf_model.model.config.use_sliding_window = False
        # for layer in hf_model.model.layers:
        #     layer.self_attn.sliding_window = sliding_window
        
        # 重新设置滑动窗口配置，确保和实际模型一致
        for i in range(hf_model.model.config.num_hidden_layers):
            self_attn = hf_model.model.layers[i].self_attn
            if getattr(self_attn, "sliding_window", None) is not None and self_attn.sliding_window > 0:
                self.sliding_window_cfg.local_attention_window_size = self_attn.sliding_window
                self.sliding_window_cfg.has_local_attention = True
            else:
                # Check if past_key_caches exists and has enough elements
                if hasattr(self, "past_key_caches") and self.past_key_caches and i < len(self.past_key_caches):
                    self.sliding_window_cfg.global_attention_window_size = self.past_key_caches[i].shape[2]
                else:
                    # Fallback to cache_length if past_key_caches is not initialized yet
                    self.sliding_window_cfg.global_attention_window_size = self.cache_length
                self.sliding_window_cfg.has_global_attention = True
        if not self.sliding_window_cfg.has_local_attention:
            self.sliding_window_cfg.has_global_attention = False
        return cast(GptOssForCausalLM, hf_model)

    def convert_to_quant_graph(self, target_device: str) -> Optional[QuantGraph]:
        return super().convert_to_quant_graph(target_device)

    def _forward(
        self,
        *args,
        **kwargs,
    ):
        logits = self.forward(*args)
        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        inputs = super().prepare_inputs(data)
        (
            inputs_embeds,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        ) = inputs

        bz, nq = inputs_embeds.shape[:2]
        local_attention_mask = None
        global_attention_mask = None

        if self.sliding_window_cfg.has_global_attention:
            width = self.sliding_window_cfg.global_attention_window_size
            x = torch.empty((bz, nq, width), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            global_attention_mask = self.prepare_casual_mask(x, past_seq_length, -1).to(self.execution_device)

        if self.sliding_window_cfg.has_local_attention:
            local_window = self.sliding_window_cfg.local_attention_window_size + nq - 1
            local_window = aligned(local_window, 16)
            x = torch.empty((bz, nq, local_window), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            local_attention_mask = self.prepare_casual_mask(
                x,
                past_seq_length,
                self.sliding_window_cfg.sliding_window,
            ).to(self.execution_device)

        outputs: List[Any] = [
            inputs_embeds,
            past_seq_length,
            seg_length,
            local_attention_mask,
            global_attention_mask,
            past_key_caches,
            past_value_caches,
        ]
        return tuple(outputs)

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        return self.prepare_inputs(data)
