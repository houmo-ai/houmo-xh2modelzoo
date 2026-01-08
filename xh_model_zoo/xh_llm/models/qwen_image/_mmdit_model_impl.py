import math
from pickle import NONE
from re import X
import sys
from turtle import forward
import types
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from regex import T
import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.nn import LLMCacheV2, MaskedSoftmax, RMSNorm
from xhquant.nn.modules import none_parameter_module
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

@XHLLM_TRACEABLE_MODULES.register_module({QwenImageTransformer2DModel: "QwenImageTransformer2DModel"})
class _QwenImageTransformer2DModel(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ):
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = ( self.time_text_embed(timestep, hidden_states) )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        # if True:
        #     self.transformer_blocks = self.transformer_blocks.to(torch.bfloat16)
        #     hidden_states = hidden_states.to(torch.bfloat16)
        #     encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
        #     encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bfloat16)
        #     temb = temb.to(torch.bfloat16)
        #     image_rotary_emb =( image_rotary_emb[0].to(torch.bfloat16), image_rotary_emb[1].to(torch.bfloat16))


        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
                modulate_index=modulate_index,
            )
        
        # if True:
        #     hidden_states =hidden_states.to(torch.float16)
        #     temb = temb.to(torch.float16)
        if True:
            self.norm_out = self.norm_out.to(torch.bfloat16)

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)

        if True:
            hidden_states = hidden_states.to(torch.float16)

        output = self.proj_out(hidden_states)
        return output

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg


@XHLLM_TRACEABLE_MODULES.register_module({QwenImageTransformerBlock: "QwenImageTransformerBlock"})
class _QwenImageTransformerBlock(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output # [1, 1, 3072]


        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output


        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states    

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

def register_wrap_cls(hf_model):
    pass