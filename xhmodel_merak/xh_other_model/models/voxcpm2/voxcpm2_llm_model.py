"""VoxCPM2 LM 侧的 xh2modelzoo 包装类。

VoxCPM2 里有两个独立的 MiniCPMModel 实例需要导出:
- base_lm:     带 embed_tokens、带 rope、带 causal mask、带 KV cache
- residual_lm: 无 embed_tokens(vocab=0)、无 rope(no_rope=True)、带 causal mask、带 KV cache

本文件提供两个 LLMBaseModel 子类,分别处理这两个模型的 wrap / PTQ /
prepare_inputs / KV cache 管理。

设计要点:
1. **不覆盖 `init_wrap_model` 中的全局遍历路径**。为了精确控制 wrap 作用范围
   (不波及 locenc/locdit 里的 MiniCPMModel),本类直接接受外部注入好的
   hf_module(即 base_lm 或 residual_lm),内部只对这一个 module 调
   `wrap_llm_model`,wrap 遍历天然只走到这个 module 的子树。
2. residual_lm **没有** embed_tokens,prepare_inputs 只接受 `inputs_embeds`。
3. base_lm 的 embed_tokens 有 `scale_emb` 缩放,scale 在 host 侧做(参考
   VoxCPM2Model._inference 的 `text_embed = embed_tokens(text) * scale_emb`),
   导出图本身不包含这个乘法,和 qwen3_asr 的做法一致。
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from voxcpm.modules.minicpm4 import MiniCPMModel

from ...base_llm_model import LLMBaseModel
from ...builder import register_other_model, wrap_llm_model
from .voxcpm2_llm_model_impl import register_wrap_cls


# ---------------------------------------------------------------------------
# 共同基类
# ---------------------------------------------------------------------------

class _XHVoxCPM2LMBase(LLMBaseModel):
    """base_lm / residual_lm 的共同逻辑。

    子类只需要在 `get_hf_model()` / `init_wrap_model()` 里指定哪一个
    MiniCPMModel 实例,其余接口(prepare_inputs, _forward, KV cache
    管理)都在本基类内完成。
    """

    # 子类覆盖:导出时 wrap 需要知道输入是 embeds 还是 ids。
    _supports_input_ids: bool = True

    def __init__(
        self,
        hf_module: MiniCPMModel,   # 已经从 voxcpm2 里拎出来的 MiniCPMModel 实例
        wrap_cfg,
        quant_config,
        frontend_type: str = "TorchFX",
        allow_quant: bool = True,
        export_cfg=None,
    ):
        # LLMBaseModel 的 hf_model 参数既可以是路径也可以是 nn.Module
        # 参考 qwen3_asr 里的用法,我们直接传 module
        super().__init__(
            hf_model=hf_module,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

        # 这些属性在 init_wrap_model 之后才能填
        self.num_key_value_heads: Optional[int] = None
        self.num_attention_heads: Optional[int] = None
        self.hidden_size: Optional[int] = None
        self.head_dim: Optional[int] = None

    # -------- HF model loading (VoxCPM2 特例:直接用已实例化的 module) --------

    def get_hf_model(self, device_map: str = "cpu", **kwargs):
        """LLMBaseModel 原本会从 path 加载;VoxCPM2 里我们直接用已实例化的 module。"""
        if isinstance(self.hf_model_dir, nn.Module):
            return self.hf_model_dir
        raise RuntimeError(
            "XHVoxCPM2LM 必须在构造时传入 MiniCPMModel 实例,不支持从 path 加载。"
        )

    def get_tokenizer(self):
        # VoxCPM2 的 tokenizer 在 host pipeline 里单独持有,不属于单个 LM 图
        return None

    def get_processor(self):
        return None

    # -------- wrap --------

    def init_wrap_model(self, hf_model: Optional[MiniCPMModel] = None):
        """对 MiniCPMModel 实例做 wrap。

        hf_model 可以显式传入,也可以用构造时传入的 self.hf_model_dir。
        wrap 的作用范围天然局限在这个实例的子树,不会波及 voxcpm2 的其他部分。
        """
        # 确保注册表已加载
        register_wrap_cls(None)

        if hf_model is None:
            hf_model = self.hf_model_dir
        assert isinstance(hf_model, MiniCPMModel), (
            f"init_wrap_model expects a MiniCPMModel instance, got {type(hf_model)}"
        )

        # 调用全局 wrap 逻辑,但只作用于这个子树
        self._wrap_model = wrap_llm_model(hf_model, self.wrap_cfg)

        # 记录 shape 信息
        config = hf_model.config
        self.config = config
        self.generation_config = None
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = (
            config.kv_channels
            if config.kv_channels is not None
            else config.hidden_size // config.num_attention_heads
        )

        # token embedding 处理
        if config.vocab_size > 0:
            self.token_embedding = hf_model.embed_tokens
        else:
            # residual_lm 没有 embed,置空即可
            self.token_embedding = None

        # 准备 KV cache buffer
        batch_size = 1
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, self.num_key_value_heads, self.cache_length, self.head_dim],
            )

    # -------- 输入 / 前向 --------

    @staticmethod
    def _normalize_past_seq_length(past_seq_length) -> torch.Tensor:
        past_seq_length = torch.as_tensor(past_seq_length, dtype=torch.int32)
        if past_seq_length.ndim == 0:
            past_seq_length = past_seq_length.unsqueeze(0)
        if past_seq_length.numel() != 1:
            raise ValueError(
                f"Only batch size 1 is supported, but got past_seq_length={past_seq_length.tolist()}"
            )
        return past_seq_length.reshape(1)

    def prepare_inputs(self, data: Union[dict, tuple, list], out_padding: bool = True):
        """构造 (inputs_embeds, past_seq_length, current_input_length,
           past_key_caches, past_value_caches) 五元组。
        """
        device = self.execution_device

        # --- 1. 解析 inputs_embeds(或 input_ids) ---
        if "input_ids" in data and self._supports_input_ids:
            assert self.token_embedding is not None, (
                "input_ids path requires token_embedding, but this LM has no embed_tokens."
            )
            raw_input_ids = torch.as_tensor(data["input_ids"], dtype=torch.long)
            if raw_input_ids.ndim == 1:
                raw_input_ids = raw_input_ids.unsqueeze(0)
            if raw_input_ids.shape[0] != 1:
                raise ValueError(
                    f"Only batch size 1 is supported, but got input_ids shape {tuple(raw_input_ids.shape)}"
                )
            seq_length = raw_input_ids.shape[1]
            if seq_length > self.input_sequence_length:
                raise ValueError(
                    f"Input sequence length {seq_length} exceeds max {self.input_sequence_length}"
                )
            input_ids = raw_input_ids.to(device)
            if self.input_sequence_length > seq_length and out_padding:
                padding_input_ids = torch.full(
                    (1, self.input_sequence_length - seq_length),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
            self.token_embedding.to(device)
            inputs_embeds = self.token_embedding(input_ids)
            current_length = torch.tensor([seq_length], dtype=torch.int32, device=device)
        elif "input_embeds" in data or "inputs_embeds" in data:
            inputs_embeds = data.get("input_embeds")
            if inputs_embeds is None:
                inputs_embeds = data.get("inputs_embeds")
            inputs_embeds = torch.as_tensor(inputs_embeds)
            if inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            if inputs_embeds.shape[0] != 1:
                raise ValueError(
                    f"Only batch size 1 is supported, but got input_embeds shape {tuple(inputs_embeds.shape)}"
                )
            seq_length = inputs_embeds.shape[1]
            if seq_length > self.input_sequence_length:
                raise ValueError(
                    f"Input sequence length {seq_length} exceeds max {self.input_sequence_length}"
                )
            inputs_embeds = inputs_embeds.to(device)
            if self.input_sequence_length > seq_length and out_padding:
                # residual_lm 没 token_embedding,pad 用 0 embedding
                pad_shape = (1, self.input_sequence_length - seq_length, inputs_embeds.shape[-1])
                padding = torch.zeros(pad_shape, dtype=inputs_embeds.dtype, device=device)
                inputs_embeds = torch.cat([inputs_embeds, padding], dim=1)
            current_length = torch.tensor([seq_length], dtype=torch.int32, device=device)
        else:
            raise KeyError(
                "prepare_inputs requires either 'input_ids' or 'input_embeds'/'inputs_embeds'."
            )

        # --- 2. past_seq_length ---
        past_seq_length = self._normalize_past_seq_length(data["past_seq_length"]).to(device)
        assert torch.all(past_seq_length >= 0)

        # --- 3. KV cache ---
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches

        return (
            inputs_embeds.to(device),
            past_seq_length,
            current_length.to(device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        return self.prepare_inputs(data)

    # -------- forward --------

    def _forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: list,
        past_value_caches: list,
    ):
        hidden = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        # 复用 BaseModelOutputWithPast.last_hidden_state 槽位承载输出
        return BaseModelOutputWithPast(last_hidden_state=hidden)


# ---------------------------------------------------------------------------
# base_lm wrap
# ---------------------------------------------------------------------------

@register_other_model("XHVoxCPM2BaseLMModel", master=False)
class XHVoxCPM2BaseLMModel(_XHVoxCPM2LMBase):
    """VoxCPM2 base_lm 的 xh2modelzoo wrap。

    - vocab_size=73448,带 embed_tokens
    - use_mup=False,scale_depth 不生效
    - no_rope=False,带 LongRoPE
    - 负责处理 text/audio 混合 embeddings 输入(host 侧已经混好,传入是
      最终的 inputs_embeds [1, N, H])
    """
    _supports_input_ids = True


# ---------------------------------------------------------------------------
# residual_lm wrap
# ---------------------------------------------------------------------------

@register_other_model("XHVoxCPM2ResidualLMModel", master=False)
class XHVoxCPM2ResidualLMModel(_XHVoxCPM2LMBase):
    """VoxCPM2 residual_lm 的 xh2modelzoo wrap。

    - vocab_size=0,embed_tokens = nn.Identity(),不支持 input_ids
    - no_rope=True,不使用 LongRoPE
    - 输入是 [base_lm 输出 经过 fusion_concat_proj] 后的 residual_inputs
    """
    _supports_input_ids = False
