from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def load_official_model(
    melo_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate MeloTTS without importing its text/BERT frontend."""

    root = str(Path(melo_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from melo.models import SynthesizerTrn

    with Path(config_path).open("r", encoding="utf-8") as source:
        config: dict[str, Any] = json.load(source)
    data = config["data"]
    train = config["train"]
    model = SynthesizerTrn(
        len(config["symbols"]),
        data["filter_length"] // 2 + 1,
        train["segment_size"] // data["hop_length"],
        n_speakers=data["n_speakers"],
        num_tones=config["num_tones"],
        num_languages=config["num_languages"],
        **config["model"],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


class EncoderDurationStatic(nn.Module):
    """Fixed-L official text encoder plus deterministic duration predictor."""

    def __init__(
        self,
        model: nn.Module,
        text_max_length: int,
        lang_id: int = 3,
        bert_channels: int = 1024,
        ja_bert_channels: int = 768,
    ) -> None:
        super().__init__()
        if text_max_length <= 0:
            raise ValueError("text_max_length must be positive")
        self.enc_p = model.enc_p
        self.dp = model.dp
        self.emb_g = model.emb_g
        position = torch.arange(text_max_length).view(1, -1)
        self.register_buffer(
            "language",
            (position.remainder(2) * int(lang_id)).long(),
            persistent=False,
        )
        self.register_buffer(
            "bert",
            torch.zeros(1, bert_channels, text_max_length),
            persistent=False,
        )
        self.register_buffer(
            "ja_bert",
            torch.zeros(1, ja_bert_channels, text_max_length),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        x_lengths: Tensor,
        tones: Tensor,
        sid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = x.long()
        x_lengths = x_lengths.long()
        tones = tones.long()
        sid = sid.long()
        g = self.emb_g(sid).unsqueeze(-1)
        hidden, m_p, logs_p, x_mask = self.enc_p(
            x,
            x_lengths,
            tones,
            self.language,
            self.bert,
            self.ja_bert,
            g=g,
        )
        logw = self.dp(hidden, x_mask, g=g)
        return m_p, logs_p, logw, x_mask, g


class MaskAwareGenerator(nn.Module):
    """Official generator with a valid-length mask at every resolution."""

    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.generator = generator
        self.upsample_rates = tuple(int(layer.stride[0]) for layer in generator.ups)

    def forward(self, z: Tensor, g: Tensor, y_mask: Tensor) -> Tensor:
        decoder = self.generator
        mask = y_mask.to(dtype=z.dtype)
        x = decoder.conv_pre(z * mask)
        x = (x + decoder.cond(g)) * mask
        for stage, rate in enumerate(self.upsample_rates):
            x = F.leaky_relu(x, 0.1)
            x = decoder.ups[stage](x)
            mask = torch.repeat_interleave(mask, repeats=rate, dim=2)
            x = x * mask
            offset = stage * decoder.num_kernels
            paths = [decoder.resblocks[offset + branch](x, x_mask=mask) for branch in range(decoder.num_kernels)]
            x = torch.stack(paths).sum(dim=0)
            x = (x / decoder.num_kernels) * mask
        x = F.leaky_relu(x, 0.01) * mask
        x = decoder.conv_post(x) * mask
        return torch.tanh(x) * mask


class FlowMaskedDecoderStatic(nn.Module):
    """Fixed-T inverse flow plus mask-aware waveform generator."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.flow = model.flow
        self.decoder = MaskAwareGenerator(model.dec)

    def forward(self, z_p: Tensor, y_mask: Tensor, g: Tensor) -> Tensor:
        mask = y_mask.to(dtype=z_p.dtype)
        z = self.flow(z_p * mask, mask, g=g, reverse=True)
        return self.decoder(z * mask, g, mask)


@dataclass(frozen=True)
class PreparedDecoderInputs:
    z_p: Tensor
    y_mask: Tensor
    g: Tensor
    y_length: int
    durations: Tensor


def prepare_decoder_inputs(
    m_p: Tensor,
    logs_p: Tensor,
    logw: Tensor,
    x_mask: Tensor,
    g: Tensor,
    acoustic_max_length: int,
    *,
    length_scale: float = 1.0,
    noise_scale: float = 0.0,
    noise: Tensor | None = None,
) -> PreparedDecoderInputs:
    """CPU bridge matching MeloTTS ceil-duration path expansion."""

    if acoustic_max_length <= 0:
        raise ValueError("acoustic_max_length must be positive")
    if m_p.ndim != 3 or m_p.shape[0] != 1:
        raise ValueError("only batch=1 [1,C,L] is supported")
    if logs_p.shape != m_p.shape:
        raise ValueError("logs_p must have the same shape as m_p")
    if logw.shape != x_mask.shape or logw.shape[:2] != (1, 1):
        raise ValueError("logw/x_mask must be [1,1,L]")
    duration_f = torch.exp(logw.float()) * x_mask.float() * float(length_scale)
    durations = torch.ceil(duration_f).long()[0, 0]
    raw_length = int(durations.sum().item())
    y_length = max(raw_length, 1)
    if y_length > acoustic_max_length:
        raise ValueError(f"predicted acoustic length {y_length} exceeds Tmax={acoustic_max_length}")
    channels = int(m_p.shape[1])
    if raw_length:
        token_index = torch.repeat_interleave(
            torch.arange(durations.numel(), device=durations.device),
            durations,
        )
        expanded_m = m_p.float().index_select(2, token_index)
        expanded_logs = logs_p.float().index_select(2, token_index)
    else:
        expanded_m = torch.zeros(1, channels, 1, dtype=torch.float32, device=m_p.device)
        expanded_logs = torch.zeros_like(expanded_m)
    if noise is None:
        epsilon = torch.zeros_like(expanded_m) if noise_scale == 0 else torch.randn_like(expanded_m)
    else:
        if tuple(noise.shape) != tuple(expanded_m.shape):
            raise ValueError("noise shape does not match expanded latent")
        epsilon = noise.to(device=m_p.device, dtype=torch.float32)
    z_valid = expanded_m + epsilon * torch.exp(expanded_logs) * float(noise_scale)
    z_p = torch.zeros(
        1,
        channels,
        acoustic_max_length,
        dtype=torch.float32,
        device=m_p.device,
    )
    y_mask = torch.zeros(
        1,
        1,
        acoustic_max_length,
        dtype=torch.float32,
        device=m_p.device,
    )
    z_p[:, :, :y_length] = z_valid
    y_mask[:, :, :y_length] = 1
    return PreparedDecoderInputs(
        z_p=z_p,
        y_mask=y_mask,
        g=g.float(),
        y_length=y_length,
        durations=durations,
    )


__all__ = [
    "EncoderDurationStatic",
    "FlowMaskedDecoderStatic",
    "MaskAwareGenerator",
    "PreparedDecoderInputs",
    "load_official_model",
    "prepare_decoder_inputs",
]
