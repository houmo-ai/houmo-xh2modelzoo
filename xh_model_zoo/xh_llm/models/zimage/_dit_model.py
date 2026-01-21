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
from diffusers.models.transformers.transformer_z_image import Attention, FeedForward

import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

def get_correct_rearrange_matrix(feature_dim=128):
    """
    生成和原函数完全等价的置换矩阵
    关键修正：调整矩阵的行/列顺序，匹配原函数的重排逻辑
    """
    half_dim = feature_dim // 2
    # 1. 构造重排后的目标索引（和原函数完全一致）
    # 目标顺序：[0,2,4,...,126, 1,3,...,127]
    odd_indices = torch.arange(0, feature_dim, 2)  # [0,2,...,126] (64个)
    even_indices = torch.arange(1, feature_dim, 2) # [1,3,...,127] (64个)
    target_idx = torch.cat([odd_indices, even_indices], dim=0)  # shape [128]
    
    # 2. 生成正确的置换矩阵（关键修正：按行索引重排，而非列）
    # 置换矩阵P的定义：P[i,j] = 1 当且仅当 输出的第i维 = 输入的第j维
    perm_matrix = torch.zeros((feature_dim, feature_dim))
    for i in range(feature_dim):
        # 输出的第i维 对应 输入的第target_idx[i]维
        perm_matrix[i, target_idx[i]] = 1.0
    
    return perm_matrix

def get_restore_matrix(feature_dim=128):
    """
    生成和restore_x_order完全等价的还原置换矩阵
    Returns:
        restore_matrix: 形状为[feature_dim, feature_dim]的置换矩阵
    """
    half_dim = feature_dim // 2
    # 1. 构造还原的目标索引（逆操作）
    # 输入重排后的维度：[0,1,2,...,63, 64,65,...,127] → 对应原始的[0,2,4,...,126, 1,3,...,127]
    # 还原目标：输出维度i → 若i为偶数，取输入的i//2；若i为奇数，取输入的half_dim + (i//2)
    restore_idx = torch.zeros(feature_dim, dtype=torch.long)
    for i in range(feature_dim):
        if i % 2 == 0:
            # 偶数位（0,2,4...）→ 取重排后前64维的第i//2位
            restore_idx[i] = i // 2
        else:
            # 奇数位（1,3,5...）→ 取重排后后64维的第i//2位
            restore_idx[i] = half_dim + (i // 2)
    
    # 2. 生成还原置换矩阵（显式定义：输出i对应输入restore_idx[i]）
    restore_matrix = torch.zeros((feature_dim, feature_dim))
    for i in range(feature_dim):
        restore_matrix[i, restore_idx[i]] = 1.0
    
    return restore_matrix


@XHLLM_TRACEABLE_MODULES.register_module({FeedForward: "FeedForward"})
class _FeedForward(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg
    def forward(self, x):
        out = self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)).clip(-65504, 65504))
        return out


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
        query = self.to_q(hidden_states) # [1, 4096, 3840] = > [1, 4096, 3840]  # linear 3840,3840
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # query = query.unflatten(-1, (30, -1))  # [1, 4096, 3840] =>  [1, 4096, 30, 128]
        # key = key.unflatten(-1, (30, -1))
        # value = value.unflatten(-1, (30, -1))

        query = query.reshape(1, -1, 30, 128)  # [1, 4096, 3840] =>  [1, 4096, 30, 128]
        key = key.reshape(1, -1, 30, 128)
        value = value.reshape(1, -1, 30, 128)

        # Apply Norms
        query = self.norm_q(query)
        key = self.norm_k(key)

        query = self.apply_rotary_emb(query, freqs_cis_r, freqs_cis_l) # [1, 4096, 30, 128])
        key = self.apply_rotary_emb(key, freqs_cis_r, freqs_cis_l) # [1, 4096, 30, 128])

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # Cast to correct dtype

        # dtype = query.dtype
        # query, key = query.to(dtype), key.to(dtype)
        # value  = value.to(dtype)

        # # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        # if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]

        query = query * self.kv_scale
        key = key.transpose(2, 3)
        attn_weights = torch.matmul(query, key) 
        attn_weights = self.maskedadd(attn_weights, attention_mask)
        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, value) # [1, 4096, 30, 128]
        hidden_states = hidden_states.permute(0, 2, 1, 3) # [1, 30, 4096, 128]
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
        # hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.reshape(1, -1, 3840)

        # hidden_states = hidden_states.to(dtype) 

        output = self.to_out[0](hidden_states).clip(-65504, 65504)
        output = self.to_out[1](output)
        return output
    

    def _setup(self, cfg: ConfigDict):
        self.cfg = cfg

        _kv_scale = 1 / math.sqrt(128)
        self.kv_scale = _kv_scale

        self.maskedadd = xhnn.MaskedAdd()
        # self.masked_softmax = MaskedSoftmax(dim=-1)

        self.enable_rope = cfg.get("enable_rope", True)
        if self.enable_rope:
            self.rope = xhnn.Rope()
        
        self.restore_m = get_restore_matrix(128).half().to(self.device)
        self.correct_m = get_correct_rearrange_matrix(128).half().to(self.device)
    
    def apply_rotary_emb(self, x_in,  roper, ropei):
        if True:
            x_rearranged = torch.matmul(self.correct_m, x_in.unsqueeze(-1)).squeeze(-1)
            # cos2 = roper.repeat([1,1,1,2])
            # sin2 = ropei.repeat([1,1,1,2])
            cos2 = roper
            sin2 = ropei
            x_out = self.rope(x_rearranged, cos2, sin2)
            x_out = torch.matmul(self.restore_m, x_out.unsqueeze(-1)).squeeze(-1)
        else:
            x_reshaped = x_in.reshape(1, self.token_length, 30, -1, 2) # [1, 4096, 30, 64, 2]
            x_real = x_reshaped[..., 0]  # 实部，形状：[... , n]  # 1/2
            x_imag = x_reshaped[..., 1]  # 虚部，形状：[... , n]  # 

            out_real = x_real * roper - x_imag * ropei  # 结果实部 # x [x_real,x_imag] 
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
        self.context_refiner[0].attention.token_length = self.cfg['token_len']
        self.context_refiner[1].attention.token_length = self.cfg['token_len']
        self.noise_refiner[0].attention.token_length = 4096
        self.noise_refiner[1].attention.token_length = 4096

        self.freqs_cis_r = cfg["f_real"]
        self.freqs_cis_l = cfg["f_imag"]
        self.cap_freqs_r = cfg["c_f_real"]
        self.cap_freqs_l = cfg["c_f_imag"]


        for layer in self.layers:
            layer.attention.token_length = self.cfg['token_len'] + 4096
        # self.device = "cuda"

        self.inp_linear = self.all_x_embedder[f"{2}-{1}"]
        self.out_linear = self.all_final_layer[f"{2}-{1}"]
    def forward(
        self,
        x: Union[List[torch.Tensor], List[List[torch.Tensor]]],
        x_mask=None, adaln_input=None, # freqs_cis_r=None, freqs_cis_l=None, 
        # t=None, 
        cap_feats: Union[List[torch.Tensor], List[List[torch.Tensor]]]=None, cap_mask= None, 
        
        cap_pad_mask = None, n_cap_pad_mask = None,
        
        # cap_freqs_r = None, cap_freqs_l = None,
        # return_dict: bool = True,
        # controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
        # siglip_feats: Optional[List[List[torch.Tensor]]] = None,
        # image_noise_mask: Optional[List[List[int]]] = None,
        # patch_size: int = 2,
        # f_patch_size: int = 1,
    ):
        # device = "cuda"

        # Single embedding for all tokens
        # adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0]) # 1000 * 0
        # t_noisy = t_clean = None

        # (
        #     x, # [4096, 64]
        #     cap_feats, # [128, 2560]
        #     x_size,
        #     x_pos_ids,
        #     cap_pos_ids,
        #     x_pad_mask,
        #     cap_pad_mask,
        # ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size) # [16, 1, 128, 128]   [101, 2560]   2 1 
        # x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None

        # X embed & refine
        # x_seqlens = [4096] # [len(xi) for xi in x]

        # x = self.all_x_embedder[f"{2}-{1}"](x).unsqueeze(0)  # embed
        x = self.inp_linear(x).unsqueeze(0)  # embed

        # x, x_freqs, x_mask, _, x_noise_tensor = self._prepare_sequence(
        #     list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
        # )

        for layer in self.noise_refiner:
            x = (
                layer(x, x_mask, self.freqs_cis_r, self.freqs_cis_l, adaln_input, ) # , None, t_noisy, t_clean
            )
        
        # x_ori = torch.load("/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/zimage/dit/x.pt")

        # Cap embed & refine
        # cap_seqlens = [len(ci) for ci in cap_feats]
        # cap_feats = self.cap_embedder(torch.cat(cap_feats, dim=0))  # embed
        # cap_feats, cap_freqs, cap_mask, _, _ = self._prepare_sequence(
        #     list(cap_feats.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
        # )
        cap_feats = self.cap_embedder(cap_feats)
        # cap_feats[cap_pad_mask] = self.cap_pad_token

        cap_feats = cap_feats * n_cap_pad_mask + cap_pad_mask * self.cap_pad_token

        cap_feats = cap_feats.unsqueeze(0)

        for layer in self.context_refiner:
            cap_feats = (
                layer(cap_feats, cap_mask, self.cap_freqs_r, self.cap_freqs_l)
            )

        # Siglip embed & refine
        # siglip_seqlens = siglip_freqs = None

        # cap_feats_ori = torch.load("/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/zimage/dit/cap_feats.pt")


        unified = torch.concat([x, cap_feats], dim=1)
        unified_freqs_r = torch.concat([self.freqs_cis_r, self.cap_freqs_r], dim=1)
        unified_freqs_l = torch.concat([self.freqs_cis_l, self.cap_freqs_l], dim=1)
        unified_mask = torch.concat([x_mask, cap_mask], dim=1)

        # Main transformer layers
        for layer_idx, layer in enumerate(self.layers):
            unified = (
                layer(unified, unified_mask, unified_freqs_r, unified_freqs_l, adaln_input)
            )

        unified = (
            self.out_linear(unified, c=adaln_input)
        )

        return unified


@XHLLM_TRACEABLE_MODULES.register_module({ZImageTransformerBlock: "ZImageTransformerBlock"})
class _ZImageTransformerBlock(DynamicModule):
    def _setup(self, cfg: ConfigDict, device="cuda"):
        self.cfg = cfg
        self.device = device


        self.slice_1 = xhnn.Slice([0], [3840], [2], [1])
        self.slice_2 = xhnn.Slice([3840], [7680], [2], [1])
        self.slice_3 = xhnn.Slice([7680], [7680 + 3840], [2], [1])
        self.slice_4 = xhnn.Slice([7680 + 3840], [sys.maxsize], [2], [1])
    
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
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input) # [1, 15360]
            # scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            
            mod = mod.unsqueeze(1)
            # scale_msa = mod[:,:, :3840]
            # gate_msa = mod[:,:, 3840: 7680]
            # scale_mlp = mod[:,:, 7680:  7680 + 3840]
            # gate_mlp = mod[:,:, 7680 + 3840:]

            scale_msa = self.slice_1(mod)
            gate_msa = self.slice_2(mod)
            scale_mlp = self.slice_3(mod)
            gate_mlp = self.slice_4(mod)

            # gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            gate_msa, gate_mlp = torch.tanh(gate_msa), torch.tanh(gate_mlp)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis_r=freqs_cisr, freqs_cis_l=freqs_cisl
            ) #  [1, 4096, 3840]    [1,1,3840]  [4096]  [1, 4096, 64]
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp).clip(-65504, 65504) )
        else:
            # Attention block
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis_r=freqs_cisr, freqs_cis_l=freqs_cisl)
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))            
        return x
    
def register_wrap_cls(hf_model):
    pass