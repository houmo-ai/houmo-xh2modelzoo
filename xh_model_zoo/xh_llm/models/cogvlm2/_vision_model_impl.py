import sys

import torch
import torch.nn.functional as F
from xhquant import nn as xhnn
from xhquant.api import ConfigDict

from ..builder import XHLLM_TRACEABLE_MODULES, DynamicRegister

DType = torch.dtype


def memory_efficient_attention_pytorch(query, key, value, attn_bias=None, p=0.0, scale=None):
    # query     [batch, seq_len, n_head, head_dim]
    # key       [batch, seq_len, n_head, head_dim]
    # value     [batch, seq_len, n_head, head_dim]
    # attn_bias [batch, n_head, seq_len, seq_len]

    if scale is None:
        scale = 1 / query.shape[-1] ** 0.5

    # BLHC -> BHLC
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    query = query * scale
    # BHLC @ BHCL -> BHLL
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = F.dropout(attn, p)
    # BHLL @ BHLC -> BHLC
    out = attn @ value
    # BHLC -> BLHC
    out = out.transpose(1, 2)
    return out


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.visual.Transformer": "cogvlm2.vision.Transformer",
    }
)
class _Transformer(DynamicRegister):
    def _setup(self, cfg: ConfigDict):
        self.only_first_block = False

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
            if self.only_first_block:
                break

        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.visual.EVA2CLIPModel": "cogvlm2.vision.EVA2CLIPModel",
    }
)
class _EVA2CLIPModel(DynamicRegister):
    def _setup(self, *args, **kwargs):
        # self.softmax = xhnn.SoftmaxPlus(-1, dtype=torch.float32)
        self.grid_size = int(9216**0.5)
        self.slice = xhnn.Slice([1], [sys.maxsize], [1], [1])

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)
        x = self.transformer(x)
        # x = x[:, 1:]
        x = self.slice(x)

        # b, s, h = x.shape  #[B, 9216, 1792]
        # grid_size = int(s**0.5)
        b = 1
        s = 9216
        h = 1792
        grid_size = self.grid_size
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)  # x is [1, 2304, 4096]
        # boi = self.boi.expand(x.shape[0], -1, -1)
        # eoi = self.eoi.expand(x.shape[0], -1, -1)

        boi = self.boi.expand(1, -1, -1)
        eoi = self.eoi.expand(1, -1, -1)
        # boi = self.boi
        # eoi = self.eoi
        # print(boi.shape)
        x = torch.cat((boi, x, eoi), dim=1)
        return x


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.visual.PatchEmbedding": "cogvlm2.visual.PatchEmbedding",
    }
)
class _PatchEmbedding(DynamicRegister):
    def _setup(self, *args, **kwargs):
        pass

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)  # x: [1, 9216, 1792]
        # cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        cls_token = self.cls_embedding.expand(1, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        "transformers_modules.cogvlm2-llama3-chat-19B.visual.Attention": "cogvlm2.visual.Attention",
    }
)
class _Attention(DynamicRegister):
    def _setup(self, *args, **kwargs):
        pass

    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        # B, L, _ = x.shape   # [1, 9217, 1792]
        B = 1
        L = 9217
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)  # 3, B, L, H, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        # out = xops.memory_efficient_attention(
        #     q,
        #     k,
        #     v,
        #     scale=self.scale,
        # )
        # print(out.shape)
        out = memory_efficient_attention_pytorch(q, k, v, scale=self.scale)
        output = self.dense(out.reshape(B, L, -1))
        output = self.output_dropout(output)
        return output


def register_wrap_cls(hf_model):
    # vision = hf_model.model.vision
    # print(type(vision))
    # print(type(vision.transformer.layers[0].attention))
    # print(type(vision.patch_embedding))
    # assert False
    # _EVA2CLIPModel.register(type(vision))
    # _Attention.register(type(vision.transformer.layers[0].attention))
    # _PatchEmbedding.register(type(vision.patch_embedding))
    pass
