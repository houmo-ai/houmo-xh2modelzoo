from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .minicpmo_token2wav_modules import HiftForwardWrapper, deterministic_hift_source


FLOW_CACHE_NAMES = (
    "conformer_cnn_cache",
    "conformer_att_cache",
    "estimator_cnn_cache",
    "estimator_att_cache",
)
FLOW_ATT_CACHE_NAMES = ("conformer_att_cache", "estimator_att_cache")
FLOW_CACHE_AXES = {"conformer_att_cache": 3, "estimator_att_cache": 4}


class FlowStreamWrapper(torch.nn.Module):
    def __init__(
        self,
        flow_module: torch.nn.Module,
        n_timesteps: int,
        cache_capacity: int,
        prompt_cache_length: int = 50,
        tail_capacity: int = 100,
    ) -> None:
        super().__init__()
        self.flow = flow_module
        self.n_timesteps = int(n_timesteps)
        self.cache_capacity = int(cache_capacity)
        self.prompt_cache_length = int(prompt_cache_length)
        self.tail_capacity = int(tail_capacity)

    def forward(
        self,
        tokens: torch.Tensor,
        embedding: torch.Tensor,
        conformer_cnn_cache: torch.Tensor,
        conformer_att_cache: torch.Tensor,
        estimator_cnn_cache: torch.Tensor,
        estimator_att_cache: torch.Tensor,
        conformer_cache_valid_length: torch.Tensor,
        estimator_cache_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }
        feat, present = self.flow.inference_chunk(
            tokens,
            embedding,
            cache,
            last_chunk=False,
            n_timesteps=self.n_timesteps,
        )
        return _pack_flow_outputs(
            feat,
            present,
            self.cache_capacity,
            self.prompt_cache_length,
            self.tail_capacity,
            {
                "conformer_att_cache": tuple(conformer_att_cache.shape),
                "estimator_att_cache": tuple(estimator_att_cache.shape),
            },
            FLOW_CACHE_AXES,
            {
                "conformer_att_cache": conformer_cache_valid_length,
                "estimator_att_cache": estimator_cache_valid_length,
            },
        )


class FlowStreamFinalWrapper(FlowStreamWrapper):
    def forward(
        self,
        tokens: torch.Tensor,
        embedding: torch.Tensor,
        conformer_cnn_cache: torch.Tensor,
        conformer_att_cache: torch.Tensor,
        estimator_cnn_cache: torch.Tensor,
        estimator_att_cache: torch.Tensor,
        conformer_cache_valid_length: torch.Tensor,
        estimator_cache_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache = {
            "conformer_cnn_cache": conformer_cnn_cache,
            "conformer_att_cache": conformer_att_cache,
            "estimator_cnn_cache": estimator_cnn_cache,
            "estimator_att_cache": estimator_att_cache,
        }
        feat, present = self.flow.inference_chunk(
            tokens,
            embedding,
            cache,
            last_chunk=True,
            n_timesteps=self.n_timesteps,
        )
        return _pack_flow_outputs(
            feat,
            present,
            self.cache_capacity,
            self.prompt_cache_length,
            self.tail_capacity,
            {
                "conformer_att_cache": tuple(conformer_att_cache.shape),
                "estimator_att_cache": tuple(estimator_att_cache.shape),
            },
            FLOW_CACHE_AXES,
            {
                "conformer_att_cache": conformer_cache_valid_length,
                "estimator_att_cache": estimator_cache_valid_length,
            },
        )


def _pack_flow_outputs(
    feat: torch.Tensor,
    present: Mapping[str, torch.Tensor],
    cache_capacity: int,
    prompt_cache_length: int,
    tail_capacity: int,
    configured_shapes: Mapping[str, tuple[int, ...]],
    cache_axes: Mapping[str, int],
    input_valid_lengths: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    bounded: dict[str, torch.Tensor] = {}
    valid_lengths: dict[str, torch.Tensor] = {}
    for name in FLOW_ATT_CACHE_NAMES:
        value = present[name]
        expected_shape = configured_shapes[name]
        expected_prefix = tuple(int(dimension) for dimension in expected_shape)
        axis = cache_axes[name]
        target_shape = expected_prefix[:axis] + (cache_capacity,) + expected_prefix[axis + 1 :]
        actual_shape = tuple(int(dimension) for dimension in value.shape)
        if actual_shape[:axis] != target_shape[:axis] or actual_shape[axis + 1 :] != target_shape[axis + 1 :]:
            raise RuntimeError(f"Unexpected {name} shape: {actual_shape} != {expected_prefix}")
        prefix_length = min(prompt_cache_length, cache_capacity)
        tail_length = min(tail_capacity, max(value.shape[axis] - prefix_length, 0))
        prefix = value.narrow(axis, 0, prefix_length)
        tail = value.narrow(axis, value.shape[axis] - tail_length, tail_length)
        bounded_value = torch.cat((prefix, tail), dim=axis)
        capacity_padding = cache_capacity - bounded_value.shape[axis]
        padding = [0, 0] * value.dim()
        padding[2 * (value.dim() - axis - 1) + 1] = capacity_padding
        bounded[name] = F.pad(bounded_value, tuple(padding)).reshape(target_shape)
        valid_lengths[name] = torch.minimum(
            input_valid_lengths[name].reshape(()).to(torch.int64) + feat.shape[2],
            torch.tensor(cache_capacity, dtype=torch.int64, device=feat.device),
        )
    for name in FLOW_ATT_CACHE_NAMES:
        configured = tuple(configured_shapes[name])
        axis = cache_axes[name]
        target = configured[:axis] + (cache_capacity,) + configured[axis + 1 :]
        if tuple(bounded[name].shape) != target:
            raise RuntimeError(f"Packed {name} cache does not match configured shape")
    return (
        feat,
        present["conformer_cnn_cache"],
        bounded["conformer_att_cache"],
        present["estimator_cnn_cache"],
        bounded["estimator_att_cache"],
        valid_lengths["conformer_att_cache"].reshape(1).to(torch.int32),
        valid_lengths["estimator_att_cache"].reshape(1).to(torch.int32),
    )


class HiftStreamWrapper(HiftForwardWrapper):
    def __init__(
        self,
        hift_module: torch.nn.Module,
        source_cache_length: int = 3840,
        mel_cache_length: int = 8,
        waveform_length: int | None = None,
    ) -> None:
        super().__init__(hift_module, waveform_length=waveform_length)
        self.source_cache_length = int(source_cache_length)
        self.mel_cache_length = int(mel_cache_length)

    def _forward_full(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        current_valid_frames = speech_feat_valid_length.reshape(()).to(torch.int64)
        past_valid_frames = past_mel_valid_length.reshape(()).to(torch.int64)
        current_mask = torch.arange(speech_feat.shape[2], device=speech_feat.device) < current_valid_frames
        speech_feat = speech_feat * current_mask.reshape(1, 1, -1)
        past_mask = torch.arange(self.mel_cache_length, device=speech_feat.device) < past_valid_frames
        past_mel = past_mel[:, :, -self.mel_cache_length :] * past_mask.reshape(1, 1, -1)
        total_capacity = self.mel_cache_length + speech_feat.shape[2]
        combined = torch.zeros(
            speech_feat.shape[0],
            speech_feat.shape[1],
            total_capacity,
            dtype=speech_feat.dtype,
            device=speech_feat.device,
        )
        for valid_length in range(self.mel_cache_length + 1):
            candidate = torch.cat((past_mel[:, :, :valid_length], speech_feat), dim=2)
            candidate = F.pad(candidate, (0, self.mel_cache_length - valid_length))
            selected = (past_valid_frames == valid_length).to(speech_feat.dtype)
            combined = combined + candidate * selected
        total_valid_frames = past_valid_frames + current_valid_frames
        total_mask = torch.arange(total_capacity, device=speech_feat.device) < total_valid_frames
        speech_feat = combined * total_mask.reshape(1, 1, -1)
        f0 = self.hift.f0_predictor(speech_feat)
        f0 = self.hift.f0_upsamp(f0[:, None]).transpose(1, 2)
        sine_generator = self.hift.m_source.l_sin_gen
        sine_waves, _ = deterministic_hift_source(
            f0,
            phase_noise,
            source_noise,
            sampling_rate=int(sine_generator.sampling_rate),
            upsample_scale=int(sine_generator.upsample_scale),
            sine_amp=float(sine_generator.sine_amp),
            noise_std=float(sine_generator.noise_std),
            voiced_threshold=float(sine_generator.voiced_threshold),
        )
        source = self.hift.m_source.l_tanh(self.hift.m_source.l_linear(sine_waves)).transpose(1, 2)
        padded_past_source = F.pad(
            past_source[:, :, -self.source_cache_length :],
            (0, source.shape[2] - self.source_cache_length),
        )
        source_cache_mask = (
            torch.arange(source.shape[2], device=source.device) < past_source_valid_length.reshape(()).to(torch.int64)
        ).reshape(1, 1, -1)
        source = torch.where(source_cache_mask, padded_past_source.to(source), source)
        original_stft = getattr(self.hift, "_stft", None)
        original_istft = getattr(self.hift, "_istft", None)
        self.hift._stft = self._exportable_stft
        self.hift._istft = self._exportable_istft
        try:
            waveform = self.hift.decode(x=speech_feat, s=source)
        finally:
            if original_stft is not None:
                self.hift._stft = original_stft
            if original_istft is not None:
                self.hift._istft = original_istft
        source_length = speech_feat.shape[2] * int(sine_generator.upsample_scale)
        return waveform, source.reshape(1, 1, source_length)

    def forward(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._forward_full(
            speech_feat,
            speech_feat_valid_length,
            past_mel,
            past_mel_valid_length,
            past_source,
            past_source_valid_length,
            phase_noise,
            source_noise,
        )
        return outputs


class HiftStreamFinalWrapper(HiftStreamWrapper):
    def forward(
        self,
        speech_feat: torch.Tensor,
        speech_feat_valid_length: torch.Tensor,
        past_mel: torch.Tensor,
        past_mel_valid_length: torch.Tensor,
        past_source: torch.Tensor,
        past_source_valid_length: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._forward_full(
            speech_feat,
            speech_feat_valid_length,
            past_mel,
            past_mel_valid_length,
            past_source,
            past_source_valid_length,
            phase_noise,
            source_noise,
        )
        return outputs


__all__ = [
    "FLOW_CACHE_NAMES",
    "FlowStreamFinalWrapper",
    "FlowStreamWrapper",
    "HiftStreamWrapper",
    "HiftStreamFinalWrapper",
]
