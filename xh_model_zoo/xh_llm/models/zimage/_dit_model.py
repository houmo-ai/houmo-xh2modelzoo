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
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
from diffusers.models.normalization import RMSNorm as zimage_rms
from diffusers.models.transformers.transformer_z_image import Attention

import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@XHLLM_TRACEABLE_MODULES.register_module({Attention: "Attention"})
class _Attention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis_r: Optional[torch.Tensor] = None,
        freqs_cis_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (30, -1))
        key = key.unflatten(-1, (30, -1))
        value = value.unflatten(-1, (30, -1))

        # Apply Norms
        query = self.norm_q(query)
        key = self.norm_k(key)


        query = self.apply_rotary_emb(query, freqs_cis_r, freqs_cis_l)
        key = self.apply_rotary_emb(key, freqs_cis_r, freqs_cis_l)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        value  = value.to(dtype)
        # # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        # if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]

        query = query * self.kv_scale
        key = key.transpose(2, 3)
        attn_weights = torch.matmul(query, key) 
        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, value) # [1, 4096, 30, 128]

        # hidden_states = out.permute(0, 2, 1, 3) # [1, 30, 4096, 128]
        # Compute joint attention
        # hidden_states = dispatch_attention_fn(
        #     query,
        #     key,
        #     value,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     backend=self._attention_backend,
        #     parallel_config=self._parallel_config,
        # )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype) # torch.Size([1, 30, 524288])

        output = self.to_out[0](hidden_states)
        output = self.to_out[1](output)
        return output
    

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        self.device = "cuda"

        _kv_scale = 1 / math.sqrt(128)
        self.kv_scale = _kv_scale

        self.masked_softmax = MaskedSoftmax(dim=-1)
    
    def apply_rotary_emb(self, x_in,  roper, ropei):
        x_reshaped = x_in.reshape(1, 4096, 30, -1, 2)
        x_real = x_reshaped[..., 0]  # 实部，形状：[... , n]
        x_imag = x_reshaped[..., 1]  # 虚部，形状：[... , n]    

        out_real = x_real * roper - x_imag * ropei  # 结果实部
        out_imag = x_real * ropei + x_imag * roper  # 结果虚部

        x_rotated = torch.stack([out_real, out_imag], dim=-1)  # 形状：[... , n, 2]
        # 2. 展平最后两维（和原代码flatten(3)一致）
        x_out = x_rotated.flatten(3)  
        return x_out     

@XHLLM_TRACEABLE_MODULES.register_module({zimage_rms: "zimage_rms"})
class _RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = RMSNorm(hidden_size, self.eps)
        self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        return self



@XHLLM_TRACEABLE_MODULES.register_module({ZImageTransformer2DModel: "ZImageTransformer2DModel"})
class _ZImageTransformer2DModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        # self.device = "cuda"
    def forward(
        self,
        x: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        t,
        cap_feats: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        return_dict: bool = True,
        controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
        siglip_feats: Optional[List[List[torch.Tensor]]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):

        device = x[0].device

        # Single embedding for all tokens
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0]) # 1000 * 0
        t_noisy = t_clean = None

        (
            x, # [4096, 64]
            cap_feats, # [128, 2560]
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_pad_mask,
            cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size) # [16, 1, 128, 128]   [101, 2560]   2 1 
        x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # X embed & refine
        x_seqlens = [len(xi) for xi in x]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x, dim=0))  # embed
        x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
        )

        for layer in self.noise_refiner:
            x = (
                layer(x, x_mask, x_freqs, adaln_input, x_noise_tensor, t_noisy, t_clean)
            )

        # Cap embed & refine
        cap_seqlens = [len(ci) for ci in cap_feats]
        cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
        cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
            list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        )

        for layer in self.context_refiner:
            cap_feats = (
                layer(cap_feats, cap_mask, cap_freqs)
            )

        # Siglip embed & refine
        siglip_seqlens = siglip_freqs = None

        # Unified sequence
        unified, unified_freqs, unified_mask, unified_noise_tensor = self._build_unified_sequence(
            x,
            x_freqs,
            x_seqlens,
            x_noise_mask,
            cap_feats,
            cap_freqs,
            cap_seqlens,
            cap_noise_mask,
            siglip_feats,
            siglip_freqs,
            siglip_seqlens,
            siglip_noise_mask,
            False,
            device,
        )

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = (
                layer(unified, unified_mask, unified_freqs, adaln_input, unified_noise_tensor, t_noisy, t_clean)
            )

        unified = (
            self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        )

        # Unpatchify
        x = self.unpatchify(list(unified.unbind(dim=0)), x_size, patch_size, f_patch_size, x_pos_offsets)
        return x


@XHLLM_TRACEABLE_MODULES.register_module({ZImageTransformerBlock: "ZImageTransformerBlock"})
class ZImageTransformerBlock(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
        self.device = "cuda"
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cisr: torch.Tensor,
        freqs_cisl: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,

        # noise_mask: Optional[torch.Tensor] = None,
        # adaln_noisy: Optional[torch.Tensor] = None,
        # adaln_clean: Optional[torch.Tensor] = None,

        # scale_msa: Optional[torch.Tensor] = None,
        # gate_msa: Optional[torch.Tensor] = None,
        # scale_mlp: Optional[torch.Tensor] = None,
        # gate_mlp: Optional[torch.Tensor] = None,
    ):

        # seq_len = x.shape[1]

        mod = self.adaLN_modulation(adaln_input)
        # scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
        
        mod = mod.unsqueeze(1)
        scale_msa = mod[:,:, :3840]
        gate_msa = mod[:,:, 3840: 7680]
        scale_mlp = mod[:,:, 7680:  7680 + 3840]
        gate_mlp = mod[:,:, 7680 + 3840:]

        # gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        gate_msa, gate_mlp = torch.tanh(gate_msa), torch.tanh(gate_mlp)
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # Attention block

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis_r=freqs_cisr, freqs_cis_l=freqs_cisl
        ) #  [1, 4096, 3840]    [1,1,3840]  [4096]  [1, 4096, 64]
        x = x + gate_msa * self.attention_norm2(attn_out)

        # FFN block
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))

        return x
    
def register_wrap_cls(hf_model):
    pass