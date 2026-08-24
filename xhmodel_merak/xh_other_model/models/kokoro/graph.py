from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper, shape_inference
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils import remove_weight_norm as remove_legacy_weight_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from xhquant import nn as xhnn

from .assets import KokoroAssets, sha256
from .host import (
    StaticSample,
    build_sine_wavs,
    duration_from_logits,
    duration_to_alignment,
    harmonic_spectrogram,
    istft_waveform,
    make_frame_masks,
    make_reverse_idx,
    make_rmsnorm_scales,
    make_text_mask,
    prepare_lstm_inputs,
    prepare_shared_lstm_inputs,
    restore_bidirectional_outputs,
)


GRAPH_ROLES = (
    "front_base",
    "duration_block0",
    "duration_block1",
    "duration_block2",
    "duration_predictor",
    "text_lstm",
    "align_core",
    "shared_fwd_lstm",
    "shared_bwd_lstm",
    "f0_branch",
    "noise_branch",
    "decoder",
    "source_merge",
    "generator",
)

FORBIDDEN_NPU_OPS = frozenset(
    {
        "Atan",
        "If",
        "Loop",
        "NonZero",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
        "SequenceAt",
        "SequenceEmpty",
        "SequenceInsert",
        "Sign",
        "STFT",
    }
)


@dataclass(frozen=True)
class GraphArtifact:
    role: str
    path: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_contracts: tuple[dict[str, Any], ...]
    output_contracts: tuple[dict[str, Any], ...]
    node_count: int
    op_counts: dict[str, int]
    onnx_sha256: str
    pytorch_vs_onnx: tuple[dict[str, float], ...]
    structural_rewrites: dict[str, int]
    reference_path: Path | None = None


@dataclass
class StaticExportBundle:
    graphs: dict[str, GraphArtifact]
    feeds: dict[str, dict[str, Tensor]]
    outputs: dict[str, dict[str, Tensor]]
    sample: StaticSample
    valid_frames: Tensor
    waveform: Tensor
    duration: Tensor
    strict_validation: dict[str, Any]
    rewrites: dict[str, int]


def load_official_model(assets: KokoroAssets) -> nn.Module:
    root = str(assets.source_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import kokoro
        from kokoro.model import KModel
    except ImportError as error:
        raise ImportError(
            "Kokoro export requires the pinned upstream package and its dependencies. "
            "Install it with `pip install <model-dir>/source/kokoro`."
        ) from error
    version = getattr(kokoro, "__version__", None)
    if version != "0.9.4":
        raise RuntimeError(f"Kokoro 0.9.4 is required, found {version!r}")
    imported_from = Path(kokoro.__file__).resolve()
    if not imported_from.is_relative_to(assets.source_root.resolve()):
        raise RuntimeError(
            f"Kokoro was imported from {imported_from}, not the pinned source "
            f"checkout {assets.source_root}; restart Python after installing the pinned checkout"
        )
    model = KModel(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        config=str(assets.config),
        model=str(assets.checkpoint),
    )
    return model.eval()


def materialize_weight_norm(root: nn.Module) -> int:
    """Freeze both modern parametrization and legacy weight_norm wrappers."""

    count = 0
    for module in list(root.modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and "weight" in parametrizations:
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            count += 1
            continue
        if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            remove_legacy_weight_norm(module, name="weight")
            count += 1
    leftovers = [
        name
        for name, module in root.named_modules()
        if (getattr(module, "parametrizations", None) is not None and "weight" in module.parametrizations)
        or hasattr(module, "weight_g")
        or hasattr(module, "weight_v")
    ]
    if leftovers:
        raise RuntimeError(f"weight_norm materialization incomplete: {leftovers}")
    return count


class DepthwiseDeconvK3S2AsElementwise(nn.Module):
    def __init__(self, source: nn.ConvTranspose1d) -> None:
        super().__init__()
        if not (
            source.in_channels == source.out_channels == source.groups
            and source.kernel_size == (3,)
            and source.stride == (2,)
            and source.padding == (1,)
            and source.output_padding == (1,)
            and source.dilation == (1,)
        ):
            raise ValueError("unsupported depthwise ConvTranspose1d contract")
        channels = source.in_channels
        weight = source.weight.detach()
        self.channels = channels
        self.register_buffer("left", weight[:, 0, 0].reshape(1, channels, 1).clone())
        self.register_buffer("center", weight[:, 0, 1].reshape(1, channels, 1).clone())
        self.register_buffer("right", weight[:, 0, 2].reshape(1, channels, 1).clone())
        self.bias = source.bias.detach().reshape(1, channels, 1).clone() if source.bias is not None else None

    def forward(self, values: Tensor) -> Tensor:
        even = values * self.center
        shifted = torch.cat([values[:, :, 1:], values[:, :, -1:] * 0.0], dim=-1)
        odd = values * self.right + shifted * self.left
        if self.bias is not None:
            even = even + self.bias
            odd = odd + self.bias
        return torch.stack([even, odd], dim=-1).reshape(values.shape[0], self.channels, values.shape[-1] * 2)


class SingleChannelConvK3S2AsElementwise(nn.Module):
    def __init__(self, source: nn.Conv1d) -> None:
        super().__init__()
        if not (
            source.in_channels == source.out_channels == source.groups == 1
            and source.kernel_size == (3,)
            and source.stride == (2,)
            and source.padding == (1,)
            and source.dilation == (1,)
        ):
            raise ValueError("unsupported single-channel Conv1d contract")
        weight = source.weight.detach().reshape(3)
        self.register_buffer("left", weight[0].reshape(1, 1, 1).clone())
        self.register_buffer("center", weight[1].reshape(1, 1, 1).clone())
        self.register_buffer("right", weight[2].reshape(1, 1, 1).clone())
        self.bias = source.bias.detach().reshape(1, 1, 1).clone() if source.bias is not None else None

    def forward(self, values: Tensor) -> Tensor:
        padded = F.pad(values, (1, 1), mode="constant", value=0.0)
        result = padded[:, :, 0:-2:2] * self.left + padded[:, :, 1:-1:2] * self.center + padded[:, :, 2::2] * self.right
        return result + self.bias if self.bias is not None else result


def rewrite_depthwise_deconvolution(root: nn.Module) -> int:
    replacements: list[tuple[nn.Module, str, nn.ConvTranspose1d]] = []
    for parent in root.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.ConvTranspose1d) and (
                child.in_channels == child.out_channels == child.groups
                and child.kernel_size == (3,)
                and child.stride == (2,)
                and child.padding == (1,)
                and child.output_padding == (1,)
                and child.dilation == (1,)
            ):
                replacements.append((parent, name, child))
    for parent, name, source in replacements:
        replacement = DepthwiseDeconvK3S2AsElementwise(source).to(
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            sample = torch.randn(1, source.in_channels, 8, dtype=source.weight.dtype)
            error = float((source(sample) - replacement(sample)).abs().max().item())
        if error > 1e-5:
            raise RuntimeError(f"depthwise ConvTranspose rewrite mismatch: {error}")
        setattr(parent, name, replacement)
    return len(replacements)


def rewrite_single_channel_stride2_convolution(root: nn.Module) -> int:
    replacements: list[tuple[nn.Module, str, nn.Conv1d]] = []
    for parent in root.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv1d) and (
                child.in_channels == child.out_channels == child.groups == 1
                and child.kernel_size == (3,)
                and child.stride == (2,)
                and child.padding == (1,)
                and child.dilation == (1,)
            ):
                replacements.append((parent, name, child))
    for parent, name, source in replacements:
        replacement = SingleChannelConvK3S2AsElementwise(source).to(
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        with torch.no_grad():
            for length in (7, 8):
                sample = torch.randn(1, 1, length, dtype=source.weight.dtype)
                error = float((source(sample) - replacement(sample)).abs().max().item())
                if error > 1e-6:
                    raise RuntimeError(f"single-channel Conv rewrite mismatch: {error}")
        setattr(parent, name, replacement)
    return len(replacements)


def _copy_lstm_weights(target: nn.LSTM, source: nn.LSTM, direction: str) -> None:
    suffix = "" if direction == "forward" else "_reverse"
    with torch.no_grad():
        target.weight_ih_l0.copy_(getattr(source, f"weight_ih_l0{suffix}"))
        target.weight_hh_l0.copy_(getattr(source, f"weight_hh_l0{suffix}"))
        if source.bias:
            target.bias_ih_l0.copy_(getattr(source, f"bias_ih_l0{suffix}"))
            target.bias_hh_l0.copy_(getattr(source, f"bias_hh_l0{suffix}"))


def _dual_forward_lstm(source: nn.LSTM) -> tuple[nn.LSTM, nn.LSTM]:
    common = {
        "input_size": source.input_size,
        "hidden_size": source.hidden_size,
        "num_layers": 1,
        "batch_first": True,
        "bidirectional": False,
        "bias": source.bias,
    }
    forward = nn.LSTM(**common)
    backward = nn.LSTM(**common)
    _copy_lstm_weights(forward, source, "forward")
    _copy_lstm_weights(backward, source, "backward")
    return forward, backward


class KokoroFrontBaseStatic(nn.Module):
    def __init__(self, model: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.embedding = model.text_encoder.embedding
        self.cnn = model.text_encoder.cnn
        self.register_buffer(
            "position",
            torch.arange(text_max_length, dtype=torch.int32).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        valid_len: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ids = input_ids.long()
        padding = self.position >= valid_len.reshape(1, 1)
        # A four-dimensional additive mask bypasses Transformers' internal
        # dtype-min mask construction.  Host code supplies only 0 and -65504,
        # so conversion to FP16 never turns a float32 minimum into -inf.
        bert_hidden = self.bert(ids, attention_mask=attention_mask)
        duration_base = self.bert_encoder(bert_hidden).transpose(-1, -2)
        values = self.embedding(ids)
        values = values.transpose(1, 2).masked_fill(padding.unsqueeze(1), 0.0)
        for layer in self.cnn:
            values = layer(values).masked_fill(padding.unsqueeze(1), 0.0)
        return duration_base, values


class DurationLSTMBlockStatic(nn.Module):
    def __init__(self, source_lstm: nn.LSTM, adaln: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.forward_lstm, self.backward_lstm = _dual_forward_lstm(source_lstm)
        self.adaln = adaln
        self.text_max_length = text_max_length

    def forward(
        self,
        forward_input: Tensor,
        backward_input: Tensor,
        style: Tensor,
        reverse_idx: Tensor,
        mask: Tensor,
    ) -> Tensor:
        forward, _ = self.forward_lstm(forward_input)
        backward_reversed, _ = self.backward_lstm(backward_input)
        backward = torch.index_select(backward_reversed, 1, reverse_idx.long())
        values = torch.cat([forward, backward], dim=-1) * mask
        values = self.adaln(values, style) * mask
        expanded_style = style.unsqueeze(1).expand(-1, self.text_max_length, -1)
        return torch.cat([values, expanded_style], dim=-1) * mask


class DurationPredictorStatic(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.forward_lstm, self.backward_lstm = _dual_forward_lstm(model.predictor.lstm)
        self.projection = model.predictor.duration_proj

    def forward(
        self,
        forward_input: Tensor,
        backward_input: Tensor,
        reverse_idx: Tensor,
        mask: Tensor,
    ) -> Tensor:
        forward, _ = self.forward_lstm(forward_input)
        backward_reversed, _ = self.backward_lstm(backward_input)
        backward = torch.index_select(backward_reversed, 1, reverse_idx.long())
        return self.projection(torch.cat([forward, backward], dim=-1) * mask)


class DualForwardLSTMStatic(nn.Module):
    def __init__(self, source_lstm: nn.LSTM) -> None:
        super().__init__()
        self.forward_lstm, self.backward_lstm = _dual_forward_lstm(source_lstm)

    def forward(self, forward_input: Tensor, backward_input: Tensor) -> tuple[Tensor, Tensor]:
        forward, _ = self.forward_lstm(forward_input)
        backward, _ = self.backward_lstm(backward_input)
        return forward, backward


class PackedBidirectionalLSTMStatic(nn.Module):
    """Keep one standard bidirectional ONNX LSTM with sequence lengths.

    ``pack_padded_sequence`` is used only to express the valid-prefix contract
    to the legacy ONNX exporter.  For batch 1 and ``enforce_sorted=True`` it
    exports one ``direction=bidirectional`` LSTM node with ``sequence_lens``;
    no forward/backward weight copying or Host-side reversal is involved.
    """

    def __init__(self, source_lstm: nn.LSTM, total_length: int) -> None:
        super().__init__()
        if not source_lstm.bidirectional:
            raise ValueError("Kokoro deployment requires a bidirectional LSTM")
        if source_lstm.num_layers != 1:
            raise ValueError("Kokoro deployment supports one-layer LSTM modules")
        if not source_lstm.batch_first:
            raise ValueError("Kokoro deployment expects batch_first LSTM modules")
        self.lstm = source_lstm
        self.total_length = int(total_length)

    def forward(self, values: Tensor, valid_lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(
            values,
            valid_lengths.to(device="cpu"),
            batch_first=True,
            enforce_sorted=True,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=self.total_length,
        )
        return output


class DurationBidirectionalLSTMBlockStatic(nn.Module):
    def __init__(self, source_lstm: nn.LSTM, adaln: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.lstm = PackedBidirectionalLSTMStatic(source_lstm, text_max_length)
        self.adaln = adaln
        self.text_max_length = int(text_max_length)

    def forward(
        self,
        values: Tensor,
        style: Tensor,
        valid_lengths: Tensor,
        mask: Tensor,
    ) -> Tensor:
        values = self.lstm(values, valid_lengths) * mask
        values = self.adaln(values, style) * mask
        expanded_style = style.unsqueeze(1).expand(-1, self.text_max_length, -1)
        return torch.cat([values, expanded_style], dim=-1) * mask


class DurationPredictorBidirectionalStatic(nn.Module):
    def __init__(self, model: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.lstm = PackedBidirectionalLSTMStatic(model.predictor.lstm, text_max_length)
        self.projection = model.predictor.duration_proj

    def forward(self, values: Tensor, valid_lengths: Tensor, mask: Tensor) -> Tensor:
        return self.projection(self.lstm(values, valid_lengths) * mask)


class SharedChunkLSTMStatic(nn.Module):
    def __init__(self, source: nn.LSTM, reverse_weights: bool) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            source.input_size,
            source.hidden_size,
            num_layers=1,
            bias=source.bias,
            batch_first=True,
            bidirectional=False,
        )
        _copy_lstm_weights(
            self.lstm,
            source,
            "backward" if reverse_weights else "forward",
        )

    def forward(self, values: Tensor, h0: Tensor, c0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output, (hidden, cell) = self.lstm(values, (h0, c0))
        return output, hidden, cell


class AlignCoreStatic(nn.Module):
    def forward(
        self,
        duration_features: Tensor,
        text_features: Tensor,
        alignment: Tensor,
        style: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = torch.matmul(duration_features.transpose(-1, -2), alignment)
        asr = torch.matmul(text_features, alignment)
        return encoded, asr, style[:, :128], style[:, 128:]


def _eps(module: nn.Module) -> float:
    return float(getattr(getattr(module, "norm", module), "eps", 1e-5))


def _masked_adain(module: nn.Module, values: Tensor, style: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=-1, keepdim=True) / count
    variance = ((values - mean).square() * mask).sum(dim=-1, keepdim=True) / count
    normalized = (values - mean) / torch.sqrt(variance + _eps(module))
    gamma, beta = module.fc(style).unsqueeze(-1).chunk(2, dim=1)
    return ((1.0 + gamma) * normalized + beta) * mask


class _MaskedTemporalRMSNorm(nn.Module):
    """Use native XH2 RMSNorm with Host-provided valid-region scales."""

    def __init__(self, temporal_length: int, eps: float) -> None:
        super().__init__()
        if temporal_length <= 0:
            raise ValueError("temporal_length must be positive")
        self.norm = xhnn.RMSNorm(int(temporal_length), eps=eps, reduce_dim=-1)
        self.norm.weight.requires_grad_(False)

    def forward(self, centered: Tensor, scale: Tensor, inverse_scale: Tensor) -> Tensor:
        # centered is zero in the padded region. RMSNorm divides its squared
        # sum by static T, while masked AdaIN divides by valid L. Host supplies
        # sqrt(T/L) and sqrt(L/T) independently so each gets the nearest input
        # representation instead of deriving one from an already-rounded one.
        return self.norm(centered * scale) * inverse_scale


def _masked_adain_rmsnorm(
    module: nn.Module,
    rmsnorm: _MaskedTemporalRMSNorm,
    values: Tensor,
    style: Tensor,
    mask: Tensor,
    scale: Tensor,
    inverse_scale: Tensor,
) -> Tensor:
    mask = mask.to(dtype=values.dtype)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=-1, keepdim=True) / count
    centered = (values - mean) * mask
    normalized = rmsnorm(centered, scale, inverse_scale)
    gamma, beta = module.fc(style).unsqueeze(-1).chunk(2, dim=1)
    return ((1.0 + gamma) * normalized + beta) * mask


def _masked_adain_block(
    block: nn.Module,
    values: Tensor,
    style: Tensor,
    input_mask: Tensor,
    output_mask: Tensor,
) -> Tensor:
    values = values * input_mask
    shortcut = block.upsample(values)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    shortcut = shortcut * output_mask
    residual = _masked_adain(block.norm1, values, style, input_mask)
    residual = block.actv(residual)
    residual = block.pool(residual) * output_mask
    residual = block.conv1(block.dropout(residual)) * output_mask
    residual = _masked_adain(block.norm2, residual, style, output_mask)
    residual = block.actv(residual)
    residual = block.conv2(block.dropout(residual)) * output_mask
    return (residual + shortcut) * output_mask / math.sqrt(2.0)


def _masked_adain_rmsnorm_block(
    block: nn.Module,
    norm1: _MaskedTemporalRMSNorm,
    norm2: _MaskedTemporalRMSNorm,
    values: Tensor,
    style: Tensor,
    input_mask: Tensor,
    output_mask: Tensor,
    scale: Tensor,
    inverse_scale: Tensor,
) -> Tensor:
    values = values * input_mask
    shortcut = block.upsample(values)
    if block.learned_sc:
        shortcut = block.conv1x1(shortcut)
    shortcut = shortcut * output_mask
    residual = _masked_adain_rmsnorm(
        block.norm1,
        norm1,
        values,
        style,
        input_mask,
        scale,
        inverse_scale,
    )
    residual = block.actv(residual)
    residual = block.pool(residual) * output_mask
    residual = block.conv1(block.dropout(residual)) * output_mask
    residual = _masked_adain_rmsnorm(
        block.norm2,
        norm2,
        residual,
        style,
        output_mask,
        scale,
        inverse_scale,
    )
    residual = block.actv(residual)
    residual = block.conv2(block.dropout(residual)) * output_mask
    return (residual + shortcut) * output_mask / math.sqrt(2.0)


def _masked_generator_block(
    block: nn.Module,
    values: Tensor,
    style: Tensor,
    mask: Tensor,
) -> Tensor:
    values = values * mask
    for conv1, conv2, norm1, norm2, alpha1, alpha2 in zip(
        block.convs1,
        block.convs2,
        block.adain1,
        block.adain2,
        block.alpha1,
        block.alpha2,
        strict=True,
    ):
        residual = _masked_adain(norm1, values, style, mask)
        residual = (residual + torch.sin(alpha1 * residual).square() / alpha1) * mask
        residual = conv1(residual) * mask
        residual = _masked_adain(norm2, residual, style, mask)
        residual = (residual + torch.sin(alpha2 * residual).square() / alpha2) * mask
        residual = conv2(residual) * mask
        values = (values + residual) * mask
    return values


class F0BranchStatic(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        frame_max_length: int,
        *,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = model.predictor.F0
        self.projection = model.predictor.F0_proj
        self.frame_max_length = int(frame_max_length)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.register_buffer(
            "frame_bucket",
            torch.tensor(float(frame_max_length), dtype=torch.float32),
            persistent=False,
        )
        self.rmsnorms = nn.ModuleList()
        if not self.use_rmsnorm:
            return

        current_length = int(frame_max_length)
        if current_length <= 0:
            raise ValueError("frame_max_length must be positive")
        upsampled = False
        for block in self.blocks:
            output_length = current_length
            if block.upsample_type != "none":
                if upsampled:
                    raise ValueError("F0 branch supports exactly one temporal upsample")
                output_length *= 2
                upsampled = True
            self.rmsnorms.append(
                nn.ModuleList(
                    [
                        _MaskedTemporalRMSNorm(current_length, _eps(block.norm1)),
                        _MaskedTemporalRMSNorm(output_length, _eps(block.norm2)),
                    ]
                )
            )
            current_length = output_length
        if current_length != 2 * frame_max_length:
            raise ValueError(f"F0 branch must upsample F={frame_max_length} to 2F, got {current_length}")

    def forward(
        self,
        shared: Tensor,
        style: Tensor,
        mask_f: Tensor,
        mask_2f: Tensor,
        norm_scales: Tensor | None = None,
    ) -> Tensor:
        values = shared * mask_f
        current_mask = mask_f
        scale = inverse_scale = None
        if self.use_rmsnorm:
            if norm_scales is None:
                # Single-graph compatibility path. Standalone/modular F0 graphs
                # supply these values from Host and do not export this path.
                valid_count = mask_f.to(dtype=shared.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
                scale = torch.sqrt(self.frame_bucket.to(dtype=shared.dtype) / valid_count)
                inverse_scale = torch.reciprocal(scale)
            else:
                if tuple(norm_scales.shape) != (1, 2, 1):
                    raise ValueError(f"norm_scales must be [1,2,1], got {tuple(norm_scales.shape)}")
                scale = norm_scales[:, 0:1, :]
                inverse_scale = norm_scales[:, 1:2, :]
        for index, block in enumerate(self.blocks):
            output_mask = mask_2f if block.upsample_type != "none" else current_mask
            if self.use_rmsnorm:
                values = _masked_adain_rmsnorm_block(
                    block,
                    self.rmsnorms[index][0],
                    self.rmsnorms[index][1],
                    values,
                    style,
                    current_mask,
                    output_mask,
                    scale,
                    inverse_scale,
                )
            else:
                values = _masked_adain_block(block, values, style, current_mask, output_mask)
            current_mask = output_mask
        return (self.projection(values) * mask_2f).reshape(1, 2 * self.frame_max_length)


class NoiseBranchStatic(nn.Module):
    def __init__(self, model: nn.Module, frame_max_length: int) -> None:
        super().__init__()
        self.blocks = model.predictor.N
        self.projection = model.predictor.N_proj
        self.frame_max_length = int(frame_max_length)

    def forward(self, shared: Tensor, style: Tensor, mask_f: Tensor, mask_2f: Tensor) -> Tensor:
        values = shared * mask_f
        current_mask = mask_f
        for block in self.blocks:
            output_mask = mask_2f if block.upsample_type != "none" else current_mask
            values = _masked_adain_block(block, values, style, current_mask, output_mask)
            current_mask = output_mask
        return (self.projection(values) * mask_2f).reshape(1, 2 * self.frame_max_length)


class DecoderStatic(nn.Module):
    def __init__(self, model: nn.Module, frame_max_length: int) -> None:
        super().__init__()
        decoder = model.decoder
        self.f0_conv = decoder.F0_conv
        self.noise_conv = decoder.N_conv
        self.encode = decoder.encode
        self.asr_residual = decoder.asr_res
        self.decode = decoder.decode
        self.frame_max_length = frame_max_length

    def forward(
        self,
        asr: Tensor,
        f0: Tensor,
        noise: Tensor,
        style: Tensor,
        mask_f: Tensor,
        mask_2f: Tensor,
    ) -> Tensor:
        mask_2f_flat = mask_2f.reshape(1, 2 * self.frame_max_length)
        f0 = f0 * mask_2f_flat
        noise = noise * mask_2f_flat
        asr = asr * mask_f
        f0_down = self.f0_conv(f0.unsqueeze(1)).reshape(1, 1, self.frame_max_length)
        noise_down = self.noise_conv(noise.unsqueeze(1)).reshape(1, 1, self.frame_max_length)
        f0_down = f0_down * mask_f
        noise_down = noise_down * mask_f
        values = torch.cat([asr, f0_down, noise_down], dim=1)
        values = _masked_adain_block(self.encode, values, style, mask_f, mask_f)
        asr_residual = self.asr_residual(asr) * mask_f
        current_mask = mask_f
        append_condition = True
        for block in self.decode:
            if append_condition:
                values = torch.cat([values, asr_residual, f0_down, noise_down], dim=1)
                values = values * current_mask
            output_mask = mask_2f if block.upsample_type != "none" else current_mask
            values = _masked_adain_block(block, values, style, current_mask, output_mask)
            current_mask = output_mask
            if block.upsample_type != "none":
                append_condition = False
        return values * mask_2f


class SourceMergeStatic(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        source = model.decoder.generator.m_source
        self.linear = source.l_linear
        self.tanh = source.l_tanh

    def forward(self, sine_wavs: Tensor) -> Tensor:
        return self.tanh(self.linear(sine_wavs))


class GeneratorStatic(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.generator = model.decoder.generator

    def forward(
        self,
        generator_feature: Tensor,
        style: Tensor,
        harmonic: Tensor,
        mask_2f: Tensor,
        mask_20f: Tensor,
        mask_wave: Tensor,
    ) -> Tensor:
        generator = self.generator
        values = generator_feature * mask_2f
        harmonic = harmonic * mask_wave
        stage_masks = (mask_20f, mask_wave)
        for stage in range(generator.num_upsamples):
            stage_mask = stage_masks[stage]
            values = F.leaky_relu(values, 0.1)
            source = generator.noise_convs[stage](harmonic) * stage_mask
            source = _masked_generator_block(generator.noise_res[stage], source, style, stage_mask)
            values = generator.ups[stage](values)
            if stage == generator.num_upsamples - 1:
                values = torch.cat([values[:, :, 1:2], values], dim=-1)
            values = (values + source) * stage_mask
            branches = [
                _masked_generator_block(
                    generator.resblocks[stage * generator.num_kernels + branch],
                    values,
                    style,
                    stage_mask,
                )
                for branch in range(generator.num_kernels)
            ]
            values = torch.stack(branches).sum(dim=0) / generator.num_kernels
            values = values * stage_mask
        values = F.leaky_relu(values) * mask_wave
        values = generator.conv_post(values)
        channels = generator.post_n_fft // 2 + 1
        magnitude = torch.exp(values[:, :channels]) * mask_wave
        phase = torch.sin(values[:, channels:]) * mask_wave
        return torch.cat([magnitude, phase], dim=1)


def export_static_graphs(
    *,
    model: nn.Module,
    sample: StaticSample,
    output_dir: str | Path,
    text_max_length: int,
    frame_max_length: int,
    lstm_chunk_length: int,
    opset: int,
    seed: int,
    simplify: bool,
    validate_onnx: bool,
    f0_norm_mode: str = "adain",
) -> StaticExportBundle:
    if frame_max_length % lstm_chunk_length:
        raise ValueError("frame_max_length must be divisible by lstm_chunk_length")
    output = Path(output_dir).expanduser().resolve()
    if f0_norm_mode not in {"adain", "rmsnorm"}:
        raise ValueError("f0_norm_mode must be adain or rmsnorm")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    dynamic_ids = sample.input_ids[:, : int(sample.valid_len.item())].long()
    torch.manual_seed(seed)
    with torch.no_grad():
        dynamic_waveform, dynamic_duration = model.forward_with_tokens(
            dynamic_ids,
            sample.style,
            float(sample.speed.item()),
        )

    rewrites = {"weight_norm_materialized": materialize_weight_norm(model)}
    feeds: dict[str, dict[str, Tensor]] = {}
    outputs: dict[str, dict[str, Tensor]] = {}
    artifacts: dict[str, GraphArtifact] = {}

    def emit(
        role: str,
        module: nn.Module,
        feed: dict[str, Tensor],
        output_names: tuple[str, ...],
        filename: str,
    ) -> dict[str, Tensor]:
        artifact, result = _export_graph(
            role=role,
            module=module.eval(),
            feed=feed,
            output_names=output_names,
            path=output / filename,
            opset=opset,
            simplify=simplify,
            validate_onnx=validate_onnx,
        )
        artifacts[role] = artifact
        feeds[role] = {name: value.detach().cpu() for name, value in feed.items()}
        outputs[role] = {name: value.detach().cpu() for name, value in result.items()}
        return result

    front = KokoroFrontBaseStatic(model, text_max_length)
    result = emit(
        "front_base",
        front,
        {"input_ids": sample.input_ids, "valid_len": sample.valid_len},
        ("duration_base", "text_features"),
        f"kokoro_front_base_b1_t{text_max_length}.onnx",
    )
    duration_base = result["duration_base"]
    text_features = result["text_features"]
    style = sample.style[:, 128:]
    values = duration_base.permute(2, 0, 1)
    expanded_style = style.unsqueeze(0).expand(values.shape[0], -1, -1)
    values = torch.cat([values, expanded_style], dim=-1).transpose(0, 1)
    reverse_idx = make_reverse_idx(text_max_length, sample.valid_len)
    text_mask = make_text_mask(text_max_length, sample.valid_len)
    duration_encoder = model.predictor.text_encoder
    for block_index in range(3):
        forward, backward = prepare_lstm_inputs(values, sample.valid_len)
        result = emit(
            f"duration_block{block_index}",
            DurationLSTMBlockStatic(
                duration_encoder.lstms[2 * block_index],
                duration_encoder.lstms[2 * block_index + 1],
                text_max_length,
            ),
            {
                "forward_input": forward,
                "backward_input": backward,
                "style": style,
                "reverse_idx": reverse_idx,
                "mask": text_mask,
            },
            ("duration_features",),
            f"kokoro_duration_block{block_index}_b1_t{text_max_length}.onnx",
        )
        values = result["duration_features"]
    duration_features = values

    forward, backward = prepare_lstm_inputs(duration_features, sample.valid_len)
    result = emit(
        "duration_predictor",
        DurationPredictorStatic(model),
        {
            "forward_input": forward,
            "backward_input": backward,
            "reverse_idx": reverse_idx,
            "mask": text_mask,
        },
        ("duration_logits",),
        f"kokoro_duration_predictor_b1_t{text_max_length}.onnx",
    )
    duration = duration_from_logits(result["duration_logits"], sample.speed, sample.valid_len)

    text_values = text_features.transpose(1, 2)
    forward, backward = prepare_lstm_inputs(text_values, sample.valid_len)
    result = emit(
        "text_lstm",
        DualForwardLSTMStatic(model.text_encoder.lstm),
        {"forward_input": forward, "backward_input": backward},
        ("forward_output", "backward_output"),
        f"kokoro_text_lstm_b1_t{text_max_length}.onnx",
    )
    text_encoded = restore_bidirectional_outputs(
        result["forward_output"],
        result["backward_output"],
        sample.valid_len,
    ).transpose(1, 2)

    alignment, valid_frames = duration_to_alignment(
        duration,
        frame_max_length,
        sample.valid_len,
    )
    mask_f, mask_2f, mask_20f, mask_wave = make_frame_masks(
        valid_frames,
        frame_max_length,
    )
    result = emit(
        "align_core",
        AlignCoreStatic(),
        {
            "duration_features": duration_features,
            "text_features": text_encoded,
            "alignment": alignment,
            "style": sample.style,
        },
        ("encoded", "asr", "decoder_style", "prosody_style"),
        f"kokoro_align_core_b1_t{text_max_length}_f{frame_max_length}.onnx",
    )
    encoded = result["encoded"]
    asr = result["asr"]
    decoder_style = result["decoder_style"]
    prosody_style = result["prosody_style"]
    shared_forward_input, shared_backward_input = prepare_shared_lstm_inputs(
        encoded,
        valid_frames,
    )

    shared_results = []
    for reverse, role in ((False, "shared_fwd_lstm"), (True, "shared_bwd_lstm")):
        module = SharedChunkLSTMStatic(model.predictor.shared, reverse_weights=reverse).eval()
        full_input = shared_backward_input if reverse else shared_forward_input
        initial_h = torch.zeros(1, 1, 256)
        initial_c = torch.zeros(1, 1, 256)
        first_chunk = full_input[:, :lstm_chunk_length]
        emit(
            role,
            module,
            {"values": first_chunk, "h0": initial_h, "c0": initial_c},
            ("output", "hidden", "cell"),
            f"kokoro_{'bwd' if reverse else 'fwd'}_lstm_chunk{lstm_chunk_length}.onnx",
        )
        chunks = []
        hidden, cell = initial_h, initial_c
        with torch.no_grad():
            for start in range(0, frame_max_length, lstm_chunk_length):
                chunk, hidden, cell = module(
                    full_input[:, start : start + lstm_chunk_length],
                    hidden,
                    cell,
                )
                chunks.append(chunk)
        shared_results.append(torch.cat(chunks, dim=1))
    shared = restore_bidirectional_outputs(
        shared_results[0],
        shared_results[1],
        valid_frames,
    ).transpose(1, 2)

    f0_module = F0BranchStatic(
        model,
        frame_max_length,
        use_rmsnorm=f0_norm_mode == "rmsnorm",
    )
    rewrites["f0_depthwise_deconv"] = rewrite_depthwise_deconvolution(f0_module)
    f0_feed = {
        "shared": shared,
        "style": prosody_style,
        "mask_f": mask_f,
        "mask_2f": mask_2f,
    }
    if f0_norm_mode == "rmsnorm":
        f0_feed["norm_scales"] = make_rmsnorm_scales(valid_frames, frame_max_length)
    result = emit(
        "f0_branch",
        f0_module,
        f0_feed,
        ("f0",),
        f"kokoro_f0_branch_b1_f{frame_max_length}.onnx",
    )
    f0 = result["f0"]

    noise_module = NoiseBranchStatic(model, frame_max_length)
    rewrites["noise_depthwise_deconv"] = rewrite_depthwise_deconvolution(noise_module)
    result = emit(
        "noise_branch",
        noise_module,
        {
            "shared": shared,
            "style": prosody_style,
            "mask_f": mask_f,
            "mask_2f": mask_2f,
        },
        ("noise",),
        f"kokoro_noise_branch_b1_f{frame_max_length}.onnx",
    )
    noise = result["noise"]

    decoder = DecoderStatic(model, frame_max_length)
    rewrites["decoder_depthwise_deconv"] = rewrite_depthwise_deconvolution(decoder)
    rewrites["decoder_stride2_conv"] = rewrite_single_channel_stride2_convolution(decoder)
    result = emit(
        "decoder",
        decoder,
        {
            "asr": asr,
            "f0": f0,
            "noise": noise,
            "style": decoder_style,
            "mask_f": mask_f,
            "mask_2f": mask_2f,
        },
        ("generator_feature",),
        f"kokoro_decoder_b1_f{frame_max_length}.onnx",
    )
    generator_feature = result["generator_feature"]

    sine_wavs = build_sine_wavs(
        f0,
        valid_frames,
        frame_max_length,
        seed=seed,
    )
    result = emit(
        "source_merge",
        SourceMergeStatic(model),
        {"sine_wavs": sine_wavs},
        ("harmonic_source",),
        f"kokoro_source_merge_b1_f{frame_max_length}.onnx",
    )
    harmonic = harmonic_spectrogram(
        result["harmonic_source"],
        valid_frames,
        frame_max_length,
    )
    result = emit(
        "generator",
        GeneratorStatic(model),
        {
            "generator_feature": generator_feature,
            "style": decoder_style,
            "harmonic": harmonic,
            "mask_2f": mask_2f,
            "mask_20f": mask_20f,
            "mask_wave": mask_wave,
        },
        ("spec_phase",),
        f"kokoro_generator_b1_f{frame_max_length}.onnx",
    )
    waveform = istft_waveform(result["spec_phase"], valid_frames)

    strict_validation = _validate_static_pipeline(
        dynamic_waveform=dynamic_waveform,
        dynamic_duration=dynamic_duration,
        static_waveform=waveform,
        static_duration=duration[:, : int(sample.valid_len.item())],
    )
    return StaticExportBundle(
        graphs=artifacts,
        feeds=feeds,
        outputs=outputs,
        sample=sample,
        valid_frames=valid_frames,
        waveform=waveform,
        duration=duration,
        strict_validation=strict_validation,
        rewrites=rewrites,
    )


def _export_graph(
    *,
    role: str,
    module: nn.Module,
    feed: dict[str, Tensor],
    output_names: tuple[str, ...],
    path: Path,
    opset: int,
    simplify: bool,
    validate_onnx: bool,
) -> tuple[GraphArtifact, dict[str, Tensor]]:
    input_names = tuple(feed)
    inputs = tuple(feed.values())
    with torch.no_grad():
        raw_outputs = module(*inputs)
    if not isinstance(raw_outputs, tuple):
        raw_outputs = (raw_outputs,)
    if len(raw_outputs) != len(output_names):
        raise RuntimeError(f"{role} returned {len(raw_outputs)} outputs, expected {len(output_names)}")
    reference = {name: value.detach().cpu() for name, value in zip(output_names, raw_outputs, strict=True)}
    torch.onnx.export(
        module,
        inputs,
        path,
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    if simplify:
        import onnxsim

        model = onnx.load(path)
        model, checked = onnxsim.simplify(
            model,
            # Randomly generated valid_len may be zero and makes the BERT
            # mask produce NaNs.  Equivalence is checked below with the real,
            # valid calibration feed instead.
            check_n=0,
            perform_optimization=True,
            skip_fuse_bn=True,
        )
        if not checked:
            raise RuntimeError(f"onnxsim equivalence check failed for {role}")
        onnx.save(model, path)
    structural_rewrites = _canonicalize_static_graph(path)
    structural_rewrites["negative_transpose_permutations"] = _fix_negative_transpose_permutations(path)
    structural_rewrites["lstm_input_shapes"] = _inject_lstm_input_shapes(path)
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model)
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    forbidden = sorted(FORBIDDEN_NPU_OPS.intersection(op_counts))
    if forbidden:
        raise RuntimeError(f"{role} contains Host-only/NPU-forbidden ops: {forbidden}")
    bidirectional = [
        node.name
        for node in model.graph.node
        if node.op_type == "LSTM"
        and any(
            attribute.name == "direction" and onnx.helper.get_attribute_value(attribute) not in {b"", b"forward"}
            for attribute in node.attribute
        )
    ]
    if bidirectional:
        raise RuntimeError(f"{role} still contains bidirectional LSTM nodes: {bidirectional}")
    metrics = _validate_onnx_outputs(path, feed, reference, output_names) if validate_onnx else tuple()
    return (
        GraphArtifact(
            role=role,
            path=path,
            input_names=input_names,
            output_names=output_names,
            input_contracts=tuple(_value_contract(value) for value in model.graph.input),
            output_contracts=tuple(_value_contract(value) for value in model.graph.output),
            node_count=len(model.graph.node),
            op_counts=dict(sorted(op_counts.items())),
            onnx_sha256=sha256(path),
            pytorch_vs_onnx=metrics,
            structural_rewrites=structural_rewrites,
        ),
        reference,
    )


def _validate_onnx_outputs(
    path: Path,
    feed: dict[str, Tensor],
    reference: dict[str, Tensor],
    output_names: tuple[str, ...],
) -> tuple[dict[str, float], ...]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        list(output_names),
        {name: value.detach().cpu().numpy() for name, value in feed.items()},
    )
    metrics = []
    for name, value in zip(output_names, actual, strict=True):
        expected = reference[name].numpy()
        value = np.asarray(value)
        if value.shape != expected.shape:
            raise RuntimeError(f"{path.name}:{name} shape mismatch {value.shape} != {expected.shape}")
        difference = np.abs(value.astype(np.float64) - expected.astype(np.float64))
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean()) if difference.size else 0.0
        if max_abs > 2e-4:
            raise RuntimeError(f"{path.name}:{name} PyTorch/ONNX max_abs={max_abs} exceeds 2e-4")
        metrics.append({"name": name, "max_abs": max_abs, "mean_abs": mean_abs})
    return tuple(metrics)


def _validate_static_pipeline(
    *,
    dynamic_waveform: Tensor,
    dynamic_duration: Tensor,
    static_waveform: Tensor,
    static_duration: Tensor,
) -> dict[str, Any]:
    expected_duration = dynamic_duration.reshape(-1).cpu()
    actual_duration = static_duration.reshape(-1).cpu()
    duration_exact = torch.equal(expected_duration, actual_duration)
    if not duration_exact:
        raise RuntimeError(
            f"static duration mismatch: expected {expected_duration.tolist()}, got {actual_duration.tolist()}"
        )
    expected_wave = dynamic_waveform.reshape(-1).float().cpu()
    actual_wave = static_waveform.reshape(-1).float().cpu()
    if expected_wave.shape != actual_wave.shape:
        raise RuntimeError(
            f"static waveform shape mismatch: {tuple(expected_wave.shape)} != {tuple(actual_wave.shape)}"
        )
    difference = (expected_wave - actual_wave).abs()
    denominator = torch.linalg.vector_norm(expected_wave) * torch.linalg.vector_norm(actual_wave)
    cosine = float(torch.dot(expected_wave, actual_wave) / denominator) if denominator else 1.0
    result = {
        "duration_exact": True,
        "waveform_samples": int(expected_wave.numel()),
        "waveform_max_abs": float(difference.max().item()),
        "waveform_mean_abs": float(difference.mean().item()),
        "waveform_cosine": cosine,
        "acceptance": {
            "waveform_max_abs": 3e-2,
            "waveform_mean_abs": 5e-4,
            "waveform_cosine": 0.9995,
        },
    }
    if (
        result["waveform_max_abs"] > result["acceptance"]["waveform_max_abs"]
        or result["waveform_mean_abs"] > result["acceptance"]["waveform_mean_abs"]
        or cosine < result["acceptance"]["waveform_cosine"]
    ):
        raise RuntimeError(f"static pipeline failed strict PyTorch validation: {result}")
    return result


def _fix_negative_transpose_permutations(path: Path) -> int:
    model = onnx.load(path)
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        for attribute in node.attribute:
            if attribute.name != "perm":
                continue
            normalized = [value if value >= 0 else value + len(attribute.ints) for value in attribute.ints]
            if normalized != list(attribute.ints):
                attribute.ClearField("ints")
                attribute.ints.extend(normalized)
                fixed += 1
    if fixed:
        onnx.save(model, path)
    return fixed


def _canonicalize_static_graph(path: Path) -> dict[str, int]:
    """Materialize static structural tensors for the xhquant ONNX parser."""

    model = onnx.load(path)
    model = shape_inference.infer_shapes(model)
    value_info = {
        value.name: value
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    constant_of_shape = 0
    replacement_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type != "ConstantOfShape":
            replacement_nodes.append(node)
            continue
        output = value_info.get(node.output[0])
        shape = _static_value_shape(output) if output is not None else None
        if shape is None:
            replacement_nodes.append(node)
            continue
        fill = np.asarray(0, dtype=np.float32)
        for attribute in node.attribute:
            if attribute.name == "value":
                fill = numpy_helper.to_array(attribute.t)
                break
        tensor = numpy_helper.from_array(np.full(shape, fill.reshape(-1)[0], dtype=fill.dtype))
        replacement_nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=list(node.output),
                name=node.name,
                value=tensor,
            )
        )
        constant_of_shape += 1
    if constant_of_shape:
        model.graph.ClearField("node")
        model.graph.node.extend(replacement_nodes)

    initializer_names = {value.name for value in model.graph.initializer}
    materialized = 0
    retained_nodes = []
    for node in model.graph.node:
        attribute = (
            next((item for item in node.attribute if item.name == "value"), None)
            if node.op_type == "Constant" and len(node.output) == 1
            else None
        )
        if attribute is None or attribute.type != onnx.AttributeProto.TENSOR or node.output[0] in initializer_names:
            retained_nodes.append(node)
            continue
        tensor = onnx.TensorProto()
        tensor.CopyFrom(attribute.t)
        tensor.name = node.output[0]
        model.graph.initializer.append(tensor)
        initializer_names.add(tensor.name)
        materialized += 1
    if materialized:
        model.graph.ClearField("node")
        model.graph.node.extend(retained_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {
        "constant_of_shape": constant_of_shape,
        "tensor_constants": materialized,
    }


def _static_value_shape(value: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    dimensions = value.type.tensor_type.shape.dim
    if any(not dimension.HasField("dim_value") for dimension in dimensions):
        return None
    return tuple(int(dimension.dim_value) for dimension in dimensions)


def _inject_lstm_input_shapes(path: Path) -> int:
    model = onnx.load(path)
    initializers = {value.name: value for value in model.graph.initializer}
    values = {value.name: value for value in model.graph.value_info}
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "LSTM" or len(node.input) < 2:
            continue
        weights = initializers.get(node.input[1])
        if weights is None or len(weights.dims) != 3:
            continue
        sequence = _find_static_sequence_length(model, node.input[0])
        if sequence is None:
            continue
        values[node.input[0]] = helper.make_tensor_value_info(
            node.input[0],
            TensorProto.FLOAT,
            [sequence, 1, int(weights.dims[-1])],
        )
        fixed += 1
    if fixed:
        model.graph.ClearField("value_info")
        model.graph.value_info.extend(values.values())
        onnx.save(model, path)
    return fixed


def _find_static_sequence_length(model: onnx.ModelProto, name: str) -> int | None:
    for value in (*model.graph.input, *model.graph.value_info):
        if value.name != name:
            continue
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].HasField("dim_value"):
            return int(dimensions[0].dim_value)
    return None


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor.elem_type),
        "shape": dimensions,
    }


def artifact_metadata(artifact: GraphArtifact, root: Path) -> dict[str, Any]:
    metadata = {
        "role": artifact.role,
        "onnx_file": str(artifact.path.relative_to(root)),
        "onnx_sha256": artifact.onnx_sha256,
        "inputs": list(artifact.input_contracts),
        "outputs": list(artifact.output_contracts),
        "node_count": artifact.node_count,
        "op_counts": artifact.op_counts,
        "pytorch_vs_onnx": list(artifact.pytorch_vs_onnx),
        "structural_rewrites": artifact.structural_rewrites,
    }
    if artifact.reference_path is not None:
        metadata.update(
            {
                "reference_onnx_file": str(artifact.reference_path.relative_to(root)),
                "reference_onnx_sha256": sha256(artifact.reference_path),
            }
        )
    return metadata


def write_feed_manifest(bundle: StaticExportBundle, path: str | Path) -> str:
    """Debug-only shape manifest; tensor values remain in memory and are not golden data."""

    target = Path(path)
    target.write_text(
        json.dumps(
            {
                role: {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in feed.items()}
                for role, feed in bundle.feeds.items()
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(target)


__all__ = [
    "FORBIDDEN_NPU_OPS",
    "GRAPH_ROLES",
    "GraphArtifact",
    "StaticExportBundle",
    "artifact_metadata",
    "export_static_graphs",
    "load_official_model",
    "materialize_weight_norm",
    "rewrite_depthwise_deconvolution",
    "rewrite_single_channel_stride2_convolution",
    "write_feed_manifest",
]
