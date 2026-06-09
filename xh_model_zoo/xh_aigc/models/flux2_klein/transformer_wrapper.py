import sys
from typing import Any, Dict, Optional, Tuple, Union

import accelerate
import torch
import torch.nn as nn
# from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2TransformerBlock,
    Flux2SwiGLU,
    AdaLayerNormContinuous,
)
from xhquant.utils.registry.dynamic_module import DynamicModule
from xhquant import nn as xhnn
from torch.nn import RMSNorm



def _reshape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch_size, seq_len, inner_dim = x.shape
    head_dim = inner_dim // heads
    return x.reshape(batch_size, seq_len, heads, head_dim)


def register_transformer_wrap_modules(transformer: nn.Module) -> nn.Module:
    transformer = accelerate.hooks.remove_hook_from_module(transformer, recurse=True)
    wrapped_classes = {
        Flux2Transformer2DModel: _Flux2Transformer2DModel,
        Flux2TransformerBlock: _Flux2TransformerBlock,
        Flux2SingleTransformerBlock: _Flux2SingleTransformerBlock,
        Flux2Attention: _Flux2Attention,
        Flux2ParallelSelfAttention: _Flux2ParallelSelfAttention,
        Flux2FeedForward: _Flux2FeedForward,
        Flux2Modulation:_Flux2Modulation,
        Flux2SwiGLU: _Flux2SwiGLU,
        AdaLayerNormContinuous: _AdaLayerNormContinuous,
        RMSNorm:_RMSNorm,  # for text encoder
    }
    for _, module in list(transformer.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            dynamic_cls.convert(module)
    return transformer


class _RMSNorm(DynamicModule):
    def forward(self, hidden_states):
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[Dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = xhnn.RMSNorm(hidden_size, self.eps)
        # self.norm.weight.data = self.weight.data.clone()
        self.norm.weight = self.weight
        return self

class _AdaLayerNormContinuous(DynamicModule):
    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding))
        # scale, shift = torch.chunk(emb, 2, dim=1)

        scale = self.slice_1(emb)
        shift = self.slice_2(emb)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

    def _setup(self):
        self.slice_1 = xhnn.Slice([0], [3072], [1], [1])
        self.slice_2 = xhnn.Slice([3072], [6144], [1], [1])  
        return self

class _Flux2SwiGLU(DynamicModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x1, x2 = x.chunk(2, dim=-1)
        x1 = self.slice_1(x)
        x2 = self.slice_2(x)
        x = self.gate_fn(x1) * x2
        return x

    def _setup(self):
        self.slice_1 = xhnn.Slice([0], [9216], [2], [1])
        self.slice_2 = xhnn.Slice([9216], [18432], [2], [1])          
        return self

class _Flux2Modulation(DynamicModule):
    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        # mod = mod.unsqueeze(1)
        return mod

    @staticmethod
    # split inside the transformer blocks, to avoid passing tuples into checkpoints https://github.com/huggingface/diffusers/issues/12776
    def split(mod: torch.Tensor, mod_param_sets: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        # if mod.ndim == 2:
        mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        # Return tuple of 3-tuples of modulation params shift/scale/gate
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))

    def _setup(self):
        return self


class _Flux2FeedForward(DynamicModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x

    def _setup(self):
        return self


class _Flux2TransformerBlock(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        # (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = _Flux2Modulation.split(temb_mod_img, 2)
        # (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = _Flux2Modulation.split(
        #     temb_mod_txt, 2
        # )
        temb_mod_img = temb_mod_img.unsqueeze(1)
        temb_mod_txt = temb_mod_txt.unsqueeze(1)

        shift_msa = self.slice_1(temb_mod_img)
        scale_msa = self.slice_2(temb_mod_img)
        gate_msa = self.slice_3(temb_mod_img)
        shift_mlp = self.slice_4(temb_mod_img)
        scale_mlp = self.slice_5(temb_mod_img)
        gate_mlp = self.slice_6(temb_mod_img)

        c_shift_msa = self.slice_1(temb_mod_txt)
        c_scale_msa = self.slice_2(temb_mod_txt)
        c_gate_msa = self.slice_3(temb_mod_txt)
        c_shift_mlp = self.slice_4(temb_mod_txt)
        c_scale_mlp = self.slice_5(temb_mod_txt)
        c_gate_mlp = self.slice_6(temb_mod_txt)


        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)
        # if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def _setup(self):
        self.slice_1 = xhnn.Slice([0], [3072], [2], [1])
        self.slice_2 = xhnn.Slice([3072], [3072*2], [2], [1])
        self.slice_3 = xhnn.Slice([3072*2], [3072*3], [2], [1])
        self.slice_4 = xhnn.Slice([3072*3], [3072*4], [2], [1])
        self.slice_5 = xhnn.Slice([3072*4], [3072*5], [2], [1])
        self.slice_6 = xhnn.Slice([3072*5], [3072*6], [2], [1])            
        return self


class _Flux2SingleTransformerBlock(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # mod_shift, mod_scale, mod_gate = _Flux2Modulation.split(temb_mod, 1)[0]
        temb_mod = temb_mod.unsqueeze(1)

        mod_shift = self.slice_1(temb_mod)
        mod_scale = self.slice_2(temb_mod)
        mod_gate  = self.slice_3(temb_mod)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        # if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        return hidden_states

    def _setup(self):
        self.slice_1 = xhnn.Slice([0], [3072], [2], [1])
        self.slice_2 = xhnn.Slice([3072], [3072*2], [2], [1])
        self.slice_3 = xhnn.Slice([3072*2], [3072*3], [2], [1])
        self.slice_4 = xhnn.Slice([3072*3], [3072*4], [2], [1])
        self.slice_5 = xhnn.Slice([3072*4], [3072*5], [2], [1])
        self.slice_6 = xhnn.Slice([3072*5], [3072*6], [2], [1])            
        return self


class _Flux2Transformer2DModel(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.Tensor = None,
        concat_rotary_cos: torch.Tensor = None,
        concat_rotary_sin: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        num_txt_tokens = encoder_hidden_states.shape[1]

        # temb = self._get_fixed_time_guidance_temb(timestep, guidance).to(hidden_states.dtype)
        temb = self.flux2_fixed_temb[timestep].to(hidden_states.dtype)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        concat_rotary_emb = (concat_rotary_cos, concat_rotary_sin)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output

    def _setup(self):
        self.time_guidance_embed = self.time_guidance_embed.to(torch.float32)
        embed_param = next(self.time_guidance_embed.parameters())

        fixed_temds = []
        
        for timpesteps in [1000.0, 968.0, 908.0, 768.0]:
            fixed_timesteps = torch.tensor([timpesteps], dtype=torch.float32)
            with torch.no_grad():
                fixed_temb = self.time_guidance_embed(
                    fixed_timesteps.to(device=embed_param.device),
                    None,
                ).detach()
            
            fixed_temds.append(fixed_temb)

        self.register_buffer("flux2_fixed_timestep_values", fixed_timesteps.to(device=fixed_temb.device), persistent=False)
        self.register_buffer("flux2_fixed_temb", torch.cat(fixed_temds, dim=0), persistent=False)
        return self


class _Flux2Attention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # query = _reshape_heads(query, self.heads)
        # key = _reshape_heads(key, self.heads)
        # value = _reshape_heads(value, self.heads)

        query = query.reshape(1, query.shape[1], self.heads, 128)
        key = key.reshape(1, key.shape[1], self.heads, 128)
        value = value.reshape(1, value.shape[1], self.heads, 128)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            # encoder_query = _reshape_heads(self.add_q_proj(encoder_hidden_states), self.heads)
            # encoder_key = _reshape_heads(self.add_k_proj(encoder_hidden_states), self.heads)
            # encoder_value = _reshape_heads(self.add_v_proj(encoder_hidden_states), self.heads)

            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.reshape(1, encoder_query.shape[1], self.heads, 128)
            encoder_key = encoder_key.reshape(1, encoder_key.shape[1], self.heads, 128)                             
            encoder_value = encoder_value.reshape(1, encoder_value.shape[1], self.heads, 128)

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = self.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)
        attn_weights = torch.matmul(query, key) * self.kv_scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, self.heads * self.head_dim)

        if encoder_hidden_states is not None:
            # encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            #     [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            # )
            encoder_hidden_states = self.slice_1(hidden_states)
            hidden_states = self.slice_2(hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states

    def apply_rotary_emb(self, x_in,  freqs_cis, sequence_dim=1):
        cos, sin = freqs_cis
        x_real = x_in[..., ::2]
        x_imag = x_in[..., 1::2]                  
        # x_real = self.slice_3(x_real)   
        # x_imag = self.slice_4(x_imag)
        # x_rotated = torch.concat([-x_imag, x_real], dim=-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        x_out = x_in * cos + x_rotated * sin
        return x_out

    def _setup(self):
        self.slice_1 = xhnn.Slice([0], [512], [1], [1])
        self.slice_2 = xhnn.Slice([512], [sys.maxsize], [1], [1])

        # self.slice_3 = xhnn.Slice([0], [64], [3], [1])   
        # self.slice_4 = xhnn.Slice([64], [128], [3], [1])   

        self.kv_scale = self.head_dim ** -0.5
        return self


class _Flux2ParallelSelfAttention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        mlp_hidden_states = self.mlp_proj(hidden_states)

        # query = _reshape_heads(query, self.heads)
        # key = _reshape_heads(key, self.heads)
        # value = _reshape_heads(value, self.heads)
        
        query = query.reshape(1,  query.shape[1], self.heads, 128)
        key = key.reshape(1, key.shape[1], self.heads, 128)
        value = value.reshape(1, value.shape[1], self.heads, 128)


        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = self.apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2).transpose(2, 3)
        value = value.transpose(1, 2)
        attn_weights = torch.matmul(query, key) * self.kv_scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(attn_output.shape[0], -1, self.heads * self.head_dim)

        mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        hidden_states = self.to_out(hidden_states)
        return hidden_states

    def apply_rotary_emb(self, x_in,  freqs_cis, sequence_dim=1):
        cos, sin = freqs_cis
        x_real = x_in[..., ::2]
        x_imag = x_in[..., 1::2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        x_out = (x_in * cos + x_rotated * sin).to(x_in.dtype)
        return x_out

    def _setup(self):

        weight = self.to_qkv_mlp_proj.weight.data.clone()
        bias = self.to_qkv_mlp_proj.bias.data.clone() if self.to_qkv_mlp_proj.bias is not None else None

        q_start = 0
        k_start = self.inner_dim
        v_start = self.inner_dim * 2
        mlp_start = self.inner_dim * 3
        mlp_dim = self.mlp_hidden_dim * self.mlp_mult_factor
        has_bias = bias is not None

        self.q_proj = nn.Linear(self.to_qkv_mlp_proj.in_features, self.inner_dim, bias=has_bias)
        self.k_proj = nn.Linear(self.to_qkv_mlp_proj.in_features, self.inner_dim, bias=has_bias)
        self.v_proj = nn.Linear(self.to_qkv_mlp_proj.in_features, self.inner_dim, bias=has_bias)
        self.mlp_proj = nn.Linear(self.to_qkv_mlp_proj.in_features, mlp_dim, bias=has_bias)

        self.q_proj.weight.data = weight[q_start:k_start]
        self.k_proj.weight.data = weight[k_start:v_start]
        self.v_proj.weight.data = weight[v_start:mlp_start]
        self.mlp_proj.weight.data = weight[mlp_start:]
        if bias is not None:
            self.q_proj.bias.data = bias[q_start:k_start]
            self.k_proj.bias.data = bias[k_start:v_start]
            self.v_proj.bias.data = bias[v_start:mlp_start]
            self.mlp_proj.bias.data = bias[mlp_start:]

        self.kv_scale = self.head_dim ** -0.5
        return self


class Flux2TransformerExportWrapper(nn.Module):
    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        concat_rotary_cos: torch.Tensor,
        concat_rotary_sin: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        

        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            concat_rotary_cos=concat_rotary_cos,
            concat_rotary_sin=concat_rotary_sin,
            guidance=guidance,
            return_dict=False,
        )[0]

    @staticmethod
    def build_rotary_inputs(transformer: nn.Module, img_ids: torch.Tensor, txt_ids: torch.Tensor):
        """Graph 外预处理：由 ids 生成 RoPE cos/sin，避免 trace 图内出现 Python 循环和频率生成。"""
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_cos, image_rotary_sin = transformer.pos_embed(img_ids)
        text_rotary_cos, text_rotary_sin = transformer.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_cos, image_rotary_cos], dim=0)[None, :, None, :].half(),
            torch.cat([text_rotary_sin, image_rotary_sin], dim=0)[None, :, None, :].half(),
        )
        
        return concat_rotary_emb
