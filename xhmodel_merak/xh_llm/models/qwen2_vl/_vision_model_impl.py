import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    Qwen2VLVisionBlock,
    Qwen2VisionTransformerPretrainedModel,
    VisionAttention,
    rotate_half,
)
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    # xhquant may fuse vision RoPE as a 4D operator. Keep both the fused 4D
    # path and the eager 3D path valid so wrap and quanted_aligned share code.
    if tensor.dim() == 4:
        cos = cos.reshape(1, tensor.shape[1], 1, tensor.shape[-1]).float()
        sin = sin.reshape(1, tensor.shape[1], 1, tensor.shape[-1]).float()
    else:
        cos = cos.reshape(tensor.shape[0], 1, tensor.shape[-1]).float()
        sin = sin.reshape(tensor.shape[0], 1, tensor.shape[-1]).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.reshape_as(tensor).to(orig_dtype)


@XHLLM_TRACEABLE_MODULES.register_module({VisionAttention: "VisionAttention"})
class _VisionAttention(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.patch_size = cfg.patch_size

        head_dim = self.qkv.out_features // 3 // self.num_heads
        self.kv_scale = 1 / math.sqrt(head_dim)

        # Split HF's packed qkv projection into explicit q/k/v projections.
        # This produces a simpler graph for ONNX export and xhquant alignment.
        weight = self.qkv.weight.data.clone()
        bias = self.qkv.bias.data.clone()
        dim = weight.shape[0] // 3

        weight = weight.permute(1, 0).reshape(dim, 3, dim).permute(1, 2, 0)
        bias = bias.reshape(3, dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        self.q_proj.weight.data = weight[0]
        self.q_proj.bias.data = bias[0]
        self.k_proj.weight.data = weight[1]
        self.k_proj.bias.data = bias[1]
        self.v_proj.weight.data = weight[2]
        self.v_proj.bias.data = bias[2]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        # Preserve a synthetic batch dimension throughout attention. The fused
        # RoPE frontend emits 4D tensors, so dropping back to 3D before matmul
        # causes shape propagation failures in quanted_aligned mode.
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1).unsqueeze(0)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1).unsqueeze(0)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1).unsqueeze(0)

        cos, sin = position_embeddings
        q = apply_rotary_pos_emb_vision(q, cos, sin)
        k = apply_rotary_pos_emb_vision(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(-2, -1)
        q = q * self.kv_scale
        dtype = q.dtype
        attn_weights = torch.matmul(q, k).to(dtype)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


@XHLLM_TRACEABLE_MODULES.register_module({PatchEmbed: "PatchEmbed"})
class _PatchEmbed(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.patch_size = cfg.patch_size

        # Qwen2-VL processor emits flattened patches that already contain both
        # temporal frames. Flatten Conv3d weights into a Linear layer to preserve
        # HF semantics while keeping the ONNX graph simple for xhquant.
        patch_dim = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.proj_linear = nn.Linear(patch_dim, self.embed_dim, bias=False)
        weight = self.proj.weight
        self.proj_linear.weight.data = weight.reshape(self.embed_dim, patch_dim).contiguous()
        del self.proj
        assert self.temporal_patch_size == 2, "temporal_patch_size must be 2"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return self.proj_linear(hidden_states)


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VLVisionBlock: "Qwen2VLVisionBlock"})
class _Qwen2VLVisionBlock(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module({Qwen2VisionTransformerPretrainedModel: "Qwen2VisionTransformerPretrainedModel"})
class _Qwen2VisionTransformerPretrainedModel(DynamicModule):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False
        self.max_size_w = cfg.max_size_w
        self.max_size_h = cfg.max_size_h
        self.max_size_t = cfg.max_size_t
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size

        assert self.max_size_w % self.patch_size == 0, "max_size_w must be divisible by patch_size"
        assert self.max_size_h % self.patch_size == 0, "max_size_h must be divisible by patch_size"
        grid_size_w = self.max_size_w // self.patch_size
        grid_size_h = self.max_size_h // self.patch_size
        grid_size_t = self.max_size_t // self.temporal_patch_size
        seq_length = grid_size_t * grid_size_h * grid_size_w
        cu_seqlens = torch.tensor([0, seq_length])
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

        # The Merak visual graph is exported for one fixed image size. Precompute
        # the matching RoPE tables so the exported graph has no dynamic grid ops.
        grid_thw = torch.tensor([[grid_size_t, grid_size_h, grid_size_w]])
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = (self.cos.to(hidden_states.device), self.sin.to(hidden_states.device))
        cu_seqlens = self.cu_seqlens.to(hidden_states.device)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return self.merger(hidden_states)


def register_wrap_cls(hf_model):
    pass
