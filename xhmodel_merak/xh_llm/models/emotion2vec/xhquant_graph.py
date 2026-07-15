from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from xhquant.nn import (
    Add,
    Div,
    Gelu,
    Less,
    MatMul,
    Softmax,
    Sub,
    XHConv1d,
    XHLinear,
)


def _copy_layer_norm(source: nn.LayerNorm) -> nn.LayerNorm:
    target = nn.LayerNorm(
        source.normalized_shape,
        eps=source.eps,
        elementwise_affine=source.elementwise_affine,
    )
    if source.elementwise_affine:
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
    return target


def _copy_linear(source: nn.Linear) -> XHLinear:
    target = XHLinear(source.in_features, source.out_features, bias=source.bias is not None)
    target.weight.data.copy_(source.weight.data)
    if source.bias is not None:
        target.bias.data.copy_(source.bias.data)
    return target


class XHEmotion2vecFrameMask(nn.Module):
    FEATURE_ENCODER_SPEC = ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2))

    def __init__(self, frame_count: int = 799, feature_encoder_spec: Sequence[tuple[int, int]] | None = None):
        super().__init__()
        self.frame_count = int(frame_count)
        self.less = Less()
        feature_encoder_spec = feature_encoder_spec or self.FEATURE_ENCODER_SPEC
        self.length_stages = nn.ModuleList(
            [_XHConvOutputLength(kernel_size, stride) for kernel_size, stride in feature_encoder_spec]
        )
        self.register_buffer("frame_positions", torch.arange(self.frame_count, dtype=torch.int64).unsqueeze(0))
        self.register_buffer(
            "frame_positions_one_based", torch.arange(1, self.frame_count + 1, dtype=torch.int64).unsqueeze(0)
        )

    def output_lengths(self, valid_samples: torch.Tensor) -> torch.Tensor:
        lengths = valid_samples.to(torch.int32)
        for stage in self.length_stages:
            lengths = stage(lengths)
        return lengths

    def forward(self, valid_samples: torch.Tensor) -> torch.Tensor:
        return self.less(self.output_lengths(valid_samples).unsqueeze(1), self.frame_positions_one_based)


class _XHConvOutputLength(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.sub = Sub()
        self.div = Div()
        self.add = Add()
        self.register_buffer("kernel_size", torch.tensor(kernel_size, dtype=torch.int32))
        self.register_buffer("stride", torch.tensor(stride, dtype=torch.int32))
        self.register_buffer("one", torch.tensor(1, dtype=torch.int32))

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.add(self.div(self.sub(lengths, self.kernel_size), self.stride), self.one)


class _XHFeatureStage(nn.Module):
    def __init__(self, conv: nn.Conv1d, norm: nn.LayerNorm | None = None):
        super().__init__()
        self.conv = XHConv1d(conv)
        self.norm = norm if norm is not None else nn.LayerNorm(conv.out_channels)
        self.gelu = Gelu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.gelu(x)


class XHEmotion2vecFeatureEncoder(nn.Module):
    def __init__(self, stages: Sequence[_XHFeatureStage]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    @classmethod
    def from_spec(cls, spec: Sequence[tuple[int, int, int]], input_channels: int = 1):
        stages: list[_XHFeatureStage] = []
        in_channels = int(input_channels)
        for out_channels, kernel_size, stride in spec:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
            stages.append(_XHFeatureStage(conv))
            in_channels = out_channels
        return cls(stages)

    @classmethod
    def from_funasr(cls, local_encoder: nn.Module):
        stages = []
        for source_stage in local_encoder.conv_layers:
            conv = source_stage[0]
            norm = source_stage[2][1]
            stages.append(_XHFeatureStage(conv, _copy_layer_norm(norm)))
        return cls(stages)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        for stage in self.stages:
            x = stage(x)
        return x


class _XHPositionStage(nn.Module):
    def __init__(self, source_stage: nn.Sequential):
        super().__init__()
        self.conv = XHConv1d(source_stage[0])
        self.norm = _copy_layer_norm(source_stage[3])
        self.gelu = Gelu()
        self.remove = int(source_stage[1].remove)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.remove:
            x = x[:, :, : -self.remove]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.gelu(x)


class XHEmotion2vecPositionEncoder(nn.Module):
    def __init__(self, stages: Sequence[_XHPositionStage]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    @classmethod
    def from_funasr(cls, relative_positional_encoder: nn.Sequential):
        return cls([_XHPositionStage(stage) for stage in relative_positional_encoder[1:-1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for stage in self.stages:
            x = stage(x)
        return x.transpose(1, 2)


class XHEmotion2vecSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = XHLinear(self.embed_dim, 3 * self.embed_dim)
        self.proj = XHLinear(self.embed_dim, self.embed_dim)
        self.qk_matmul = MatMul()
        self.av_matmul = MatMul()
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor, additive_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, frame_count, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, frame_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
        if additive_padding_mask is not None:
            attention = attention + additive_padding_mask
        attention = self.softmax(attention)
        output = self.av_matmul(attention, v).transpose(1, 2).reshape(batch_size, frame_count, self.embed_dim)
        return self.proj(output)

    @classmethod
    def from_funasr(cls, source: nn.Module):
        target = cls(source.qkv.in_features, source.num_heads)
        target.scale = float(source.scale)
        target.qkv = _copy_linear(source.qkv)
        target.proj = _copy_linear(source.proj)
        return target


class XHEmotion2vecMLP(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.fc1 = _copy_linear(source.fc1)
        self.gelu = Gelu()
        self.fc2 = _copy_linear(source.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gelu(self.fc1(x)))


class XHEmotion2vecTransformerBlock(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        if source.layer_norm_first:
            raise ValueError("emotion2vec graph expects post-norm blocks")
        self.attn = XHEmotion2vecSelfAttention.from_funasr(source.attn)
        self.norm1 = _copy_layer_norm(source.norm1)
        self.mlp = XHEmotion2vecMLP(source.mlp)
        self.norm2 = _copy_layer_norm(source.norm2)

    def forward(
        self,
        x: torch.Tensor,
        additive_padding_mask: torch.Tensor,
        alibi_bias: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, additive_padding_mask + alibi_bias))
        return self.norm2(x + self.mlp(x))


def _alibi_slopes(heads: int) -> torch.Tensor:
    def power_of_two(count: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(count) - 3)))
        return [start * start**index for index in range(count)]

    if math.log2(heads).is_integer():
        values = power_of_two(heads)
    else:
        closest = 2 ** math.floor(math.log2(heads))
        values = power_of_two(closest) + _alibi_slopes(2 * closest)[0::2][: heads - closest].tolist()
    return torch.tensor(values, dtype=torch.float32)


def build_alibi_bias(
    *,
    frame_count: int,
    num_heads: int,
    num_extra_tokens: int,
    alibi_scale: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(frame_count, dtype=torch.float32)
    distance = -(positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    alibi = _alibi_slopes(num_heads).view(1, num_heads, 1, 1) * distance.view(1, 1, frame_count, frame_count)
    alibi = torch.nn.functional.pad(alibi, (num_extra_tokens, 0, num_extra_tokens, 0))
    scale = alibi_scale.detach().float().clamp_min(0).reshape(-1)
    if scale.numel() == 1:
        return alibi * scale
    if scale.numel() != num_heads:
        raise ValueError(f"ALiBi scale has {scale.numel()} heads, expected {num_heads}")
    return alibi * scale.view(1, num_heads, 1, 1)


class XHEmotion2vecGraphModel(nn.Module):
    def __init__(self, native_model: nn.Module, window_samples: int = 256000):
        super().__init__()
        audio = native_model.modality_encoders["AUDIO"]
        self.window_samples = int(window_samples)
        feature_encoder_spec = [
            (int(stage[0].kernel_size[0]), int(stage[0].stride[0])) for stage in audio.local_encoder.conv_layers
        ]
        frame_count = self.window_samples
        for kernel_size, stride in feature_encoder_spec:
            frame_count = (frame_count - kernel_size) // stride + 1
        self.frame_mask = XHEmotion2vecFrameMask(
            frame_count=frame_count,
            feature_encoder_spec=feature_encoder_spec,
        )
        self.feature_encoder = XHEmotion2vecFeatureEncoder.from_funasr(audio.local_encoder)
        self.project_norm = _copy_layer_norm(audio.project_features[1])
        self.project = _copy_linear(audio.project_features[2])
        self.position_encoder = XHEmotion2vecPositionEncoder.from_funasr(audio.relative_positional_encoder)
        self.extra_tokens = nn.Parameter(audio.extra_tokens.detach().clone())
        self.prenet_norm = _copy_layer_norm(audio.context_encoder.norm)
        source_blocks = list(audio.context_encoder.blocks) + list(native_model.blocks)
        self.blocks = nn.ModuleList([XHEmotion2vecTransformerBlock(block) for block in source_blocks])
        self.num_extra_tokens = int(audio.modality_cfg.num_extra_tokens)
        num_heads = int(source_blocks[0].attn.num_heads)
        alibi = build_alibi_bias(
            frame_count=frame_count,
            num_heads=num_heads,
            num_extra_tokens=self.num_extra_tokens,
            alibi_scale=audio.alibi_scale,
        )
        self.register_buffer("alibi_bias", alibi)
        self.register_buffer("negative_mask_value", torch.tensor(-10000.0, dtype=torch.float32))

    @classmethod
    def from_funasr(cls, native_model: nn.Module, window_samples: int = 256000):
        return cls(native_model, window_samples=window_samples)

    def forward(self, waveform: torch.Tensor, valid_samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_padding_mask = self.frame_mask(valid_samples)
        x = self.feature_encoder(waveform)
        x = self.project(self.project_norm(x.transpose(1, 2)))
        x = x + self.position_encoder(x)
        x = torch.cat([self.extra_tokens.expand(x.shape[0], -1, -1), x], dim=1)
        extra_mask = torch.zeros(
            frame_padding_mask.shape[0], self.num_extra_tokens, dtype=torch.bool, device=frame_padding_mask.device
        )
        full_padding_mask = torch.cat([extra_mask, frame_padding_mask], dim=1)
        additive_padding_mask = full_padding_mask[:, None, None, :].to(x.dtype) * self.negative_mask_value
        x = self.prenet_norm(x)
        for block in self.blocks:
            x = block(x, additive_padding_mask, self.alibi_bias)
        x = x[:, self.num_extra_tokens :]
        return x, frame_padding_mask
