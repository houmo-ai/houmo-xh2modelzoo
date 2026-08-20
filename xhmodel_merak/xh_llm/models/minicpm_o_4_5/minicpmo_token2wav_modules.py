from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _normalize_embedding(
    embedding: torch.Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    if p != 2.0:
        raise ValueError("Token2Wav embedding normalization only supports p=2")
    squared_norm = torch.sum(embedding * embedding, dim=dim, keepdim=True)
    return embedding * torch.rsqrt(squared_norm.clamp_min(eps))


def deterministic_hift_source(
    f0: torch.Tensor,
    phase_noise: torch.Tensor,
    gaussian_noise: torch.Tensor,
    *,
    sampling_rate: int,
    upsample_scale: int,
    sine_amp: float,
    noise_std: float,
    voiced_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    harmonic_count = phase_noise.shape[1]
    harmonics = torch.arange(1, harmonic_count + 1, device=f0.device, dtype=f0.dtype).view(1, 1, -1)
    rad_values = (f0 * harmonics / sampling_rate) % 1
    rad_values = torch.cat(
        (rad_values[:, :1, :] + phase_noise.to(rad_values).unsqueeze(1), rad_values[:, 1:, :]),
        dim=1,
    )
    reduced = F.interpolate(
        rad_values.transpose(1, 2),
        scale_factor=1 / upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    phase = torch.cumsum(reduced, dim=1) * (2 * math.pi)
    phase = F.interpolate(
        phase.transpose(1, 2) * upsample_scale,
        scale_factor=upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    target_length = f0.shape[1]
    active_length = min(phase.shape[1], gaussian_noise.shape[1])
    phase = phase[:, :active_length]
    active_f0 = f0[:, :active_length]
    gaussian_noise = gaussian_noise[:, :active_length]
    uv = (active_f0 > voiced_threshold).to(torch.float32)
    noise_amplitude = uv * noise_std + (1 - uv) * sine_amp / 3
    sine_waves = torch.sin(phase) * sine_amp
    sine_waves = sine_waves * uv + noise_amplitude * gaussian_noise.to(sine_waves)
    padding = target_length - active_length
    sine_waves = F.pad(sine_waves, (0, 0, 0, padding))
    uv = F.pad(uv, (0, 0, 0, padding))
    return sine_waves, uv


class FlowFrontendWrapper(torch.nn.Module):
    def __init__(self, flow_module: torch.nn.Module, mel_capacity: int):
        super().__init__()
        self.flow = flow_module
        self.mel_capacity = int(mel_capacity)

    def forward(
        self,
        tokens: torch.Tensor,
        token_length: torch.Tensor,
        prompt_feat: torch.Tensor,
        prompt_feat_length: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = _normalize_embedding(embedding)
        embedding = self.flow.spk_embed_affine_layer(embedding)
        token_mask = (
            torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < token_length.unsqueeze(1)
        ).unsqueeze(-1)
        hidden = self.flow.input_embedding(tokens) * token_mask
        hidden, _ = self.flow.encoder.forward(hidden, token_length)
        hidden = self.flow.encoder_proj(hidden)
        hidden = hidden[:, : self.mel_capacity]
        mel_mask = torch.arange(self.mel_capacity, device=hidden.device).unsqueeze(0) < (
            token_length * self.flow.up_rate
        ).unsqueeze(1)
        prompt_mask = torch.arange(self.mel_capacity, device=hidden.device).unsqueeze(0) < prompt_feat_length.unsqueeze(
            1
        )
        condition = prompt_feat[:, : self.mel_capacity] * prompt_mask.unsqueeze(-1)
        return (
            hidden.transpose(1, 2).contiguous(),
            mel_mask.unsqueeze(1).to(hidden.dtype),
            embedding,
            condition.transpose(1, 2).contiguous(),
        )


class FlowStreamingFrontendWrapper(torch.nn.Module):
    """Export one official streaming Conformer boundary without the DiT solver."""

    def __init__(self, flow_module: torch.nn.Module, *, last_chunk: bool) -> None:
        super().__init__()
        self.flow = flow_module
        self.last_chunk = bool(last_chunk)

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid_length: torch.Tensor,
        embedding: torch.Tensor,
        past_conformer_cnn_cache: torch.Tensor,
        past_conformer_att_cache: torch.Tensor,
        conformer_cache_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        spks = self.flow.spk_embed_affine_layer(_normalize_embedding(embedding))
        token_mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < token_valid_length.reshape(1, 1)
        hidden = self.flow.input_embedding(tokens) * token_mask.unsqueeze(-1)
        encoder = getattr(self.flow, "encoder", None)
        fixed_cache_encoder = all(
            hasattr(encoder, name)
            for name in ("embed", "up_embed", "pre_lookahead_layer", "up_layer", "encoders", "up_encoders")
        )
        if fixed_cache_encoder:
            hidden, present_cnn, present_att, current_mel_valid_length = self._forward_fixed_cache_encoder(
                encoder,
                hidden,
                token_valid_length,
                past_conformer_cnn_cache,
                past_conformer_att_cache,
                conformer_cache_valid_length,
            )
        else:
            hidden, present_cnn, present_att = encoder.forward_chunk(
                xs=hidden,
                last_chunk=self.last_chunk,
                cnn_cache=past_conformer_cnn_cache,
                att_cache=past_conformer_att_cache,
            )
            current_mel_valid_length = torch.tensor(hidden.shape[1], dtype=torch.int32, device=hidden.device).reshape(1)
        mu = self.flow.encoder_proj(hidden).transpose(1, 2).contiguous()
        return (
            mu,
            spks,
            present_cnn,
            present_att,
            (
                conformer_cache_valid_length.reshape(()).to(torch.int32)
                + current_mel_valid_length.reshape(()).to(torch.int32)
            ).reshape(1),
        )

    @staticmethod
    def _attention_mask(
        *,
        cache_capacity: int,
        cache_valid_length: torch.Tensor,
        current_capacity: int,
        current_valid_length: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        cache_positions = torch.arange(cache_capacity, device=device)
        current_positions = torch.arange(current_capacity, device=device)
        cache_valid = cache_positions >= cache_capacity - cache_valid_length.reshape(()).to(torch.int64)
        current_valid = current_positions < current_valid_length.reshape(()).to(torch.int64)
        return torch.cat((cache_valid, current_valid)).reshape(1, 1, cache_capacity + current_capacity)

    def _forward_fixed_cache_encoder(
        self,
        encoder: torch.nn.Module,
        hidden: torch.Tensor,
        token_valid_length: torch.Tensor,
        past_cnn: torch.Tensor,
        past_att: torch.Tensor,
        cache_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_capacity = past_att.shape[3]
        if cache_capacity % 2 != 0:
            raise RuntimeError(f"Conformer attention-cache capacity must be even, got {cache_capacity}")
        cache_half_capacity = cache_capacity // 2
        base_layer_count = len(encoder.encoders)
        base_att = past_att[:base_layer_count, :, :, :cache_half_capacity]
        up_att = past_att[base_layer_count:]
        base_cnn = past_cnn[:, :, :2]
        up_cnn = past_cnn[:, :, 2:]

        hidden, _, _ = encoder.embed(hidden, None)
        hidden = hidden * (
            torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0) < token_valid_length.reshape(1, 1)
        ).unsqueeze(-1)
        pre_lookahead = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        if self.last_chunk:
            hidden = F.pad(hidden, (0, 0, 0, pre_lookahead))
        hidden, present_base_cnn = encoder.pre_lookahead_layer.forward_chunk(hidden, cache=base_cnn)

        base_current_capacity = hidden.shape[1]
        token_valid = token_valid_length.reshape(()).to(torch.int32)
        base_current_valid = token_valid if self.last_chunk else token_valid - pre_lookahead
        base_cache_valid = cache_valid_length.reshape(()).to(torch.int32) // int(encoder.up_layer.stride)
        base_mask = self._attention_mask(
            cache_capacity=cache_half_capacity,
            cache_valid_length=base_cache_valid,
            current_capacity=base_current_capacity,
            current_valid_length=base_current_valid,
            device=hidden.device,
        )
        base_pos_emb = encoder.embed.position_encoding(
            offset=None,
            size=cache_half_capacity + base_current_capacity,
        )
        present_base_att: list[torch.Tensor] = []
        for index, layer in enumerate(encoder.encoders):
            hidden, _, present, _ = layer(hidden, base_mask, base_pos_emb, att_cache=base_att[index])
            present_base_att.append(present)
        present_base_att_tensor = torch.stack(present_base_att, dim=0)

        hidden = hidden.transpose(1, 2).contiguous()
        hidden, _, present_up_cnn = encoder.up_layer.forward_chunk(hidden, None, cache=up_cnn)
        hidden = hidden.transpose(1, 2).contiguous()
        hidden, _, _ = encoder.up_embed(hidden, None)

        up_current_capacity = hidden.shape[1]
        up_current_valid = base_current_valid * int(encoder.up_layer.stride)
        up_mask = self._attention_mask(
            cache_capacity=cache_capacity,
            cache_valid_length=cache_valid_length,
            current_capacity=up_current_capacity,
            current_valid_length=up_current_valid,
            device=hidden.device,
        )
        up_pos_emb = encoder.embed.position_encoding(
            offset=None,
            size=cache_capacity + up_current_capacity,
        )
        present_up_att: list[torch.Tensor] = []
        for index, layer in enumerate(encoder.up_encoders):
            hidden, _, present, _ = layer(hidden, up_mask, up_pos_emb, att_cache=up_att[index])
            present_up_att.append(present)
        present_up_att_tensor = torch.stack(present_up_att, dim=0)
        if encoder.normalize_before:
            hidden = encoder.after_norm(hidden)

        present_att = torch.cat(
            (present_base_att_tensor.repeat(1, 1, 1, 2, 1), present_up_att_tensor),
            dim=0,
        )
        present_cnn = torch.cat((present_base_cnn, present_up_cnn), dim=2)
        return hidden, present_cnn, present_att, up_current_valid.reshape(1)


class FlowDecoderWrapper(torch.nn.Module):
    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        return self.estimator(x, mask, mu, t, spks, cond)


class FlowEstimatorStepWrapper(torch.nn.Module):
    """Export one cached official DiT estimator step with explicit cache tensors."""

    def __init__(self, estimator: torch.nn.Module) -> None:
        super().__init__()
        self.estimator = estimator

    def _functional_blocks_forward_chunk(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        past_cache_valid_length: torch.Tensor,
        current_frame_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = x.transpose(1, 2)
        hidden = self.estimator.in_proj(hidden)
        current_capacity = hidden.shape[1]
        cache_capacity = att_cache.shape[3]
        current_positions = torch.arange(current_capacity, device=hidden.device)
        cache_positions = torch.arange(cache_capacity, device=hidden.device)
        current_valid = current_positions < current_frame_valid_length.reshape(()).to(torch.int64)
        cache_valid = cache_positions >= cache_capacity - past_cache_valid_length.reshape(()).to(torch.int64)
        key_mask = torch.cat((current_valid, cache_valid))
        attention_mask = key_mask.reshape(1, 1, -1).expand(hidden.shape[0], current_capacity, -1)
        new_cnn: list[torch.Tensor] = []
        new_att: list[torch.Tensor] = []
        for index, block in enumerate(self.estimator.blocks):
            if all(
                hasattr(block, name) for name in ("adaLN_modulation", "norm1", "attn", "norm2", "mlp", "norm3", "conv")
            ):
                hidden, cnn_value, att_value = self._functional_block_forward_chunk(
                    block,
                    hidden,
                    t_embed,
                    cnn_cache[index],
                    att_cache[index],
                    attention_mask,
                    current_frame_valid_length,
                )
            else:
                hidden, cnn_value, att_value = block.forward_chunk(
                    hidden,
                    t_embed,
                    cnn_cache[index],
                    att_cache[index],
                    attention_mask,
                )
            new_cnn.append(cnn_value)
            new_att.append(att_value)
        output = self.estimator.final_layer(hidden, t_embed).transpose(1, 2)
        return output, torch.stack(new_cnn, dim=0), torch.stack(new_att, dim=0)

    @staticmethod
    def _functional_causal_conv_block(
        conv: torch.nn.Module,
        x: torch.Tensor,
        cnn_cache: torch.Tensor,
        current_frame_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache1, cache2 = cnn_cache.split((conv.in_channels, conv.out_channels), dim=1)
        indices = (
            torch.arange(2, device=x.device, dtype=torch.int64)
            + current_frame_valid_length.reshape(()).to(torch.int64)
            - 2
        )
        first_input = conv.block[0](x)
        first_output, _ = conv.block[1].forward_chunk(first_input, cache1)
        present_cache1 = torch.index_select(first_input, 2, indices)
        second_input = conv.block[2:6](first_output)
        second_output, _ = conv.block[6].forward_chunk(second_input, cache2)
        present_cache2 = torch.index_select(second_input, 2, indices)
        return conv.block[7](second_output), torch.cat((present_cache1, present_cache2), dim=1)

    def _functional_block_forward_chunk(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        current_frame_valid_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = block.adaLN_modulation(t_embed).chunk(9, dim=-1)

        def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return value * (1 + scale) + shift

        attention, present_att = block.attn.forward_chunk(
            modulate(block.norm1(x), shift_msa, scale_msa),
            att_cache,
            attention_mask,
        )
        x = x + gate_msa * attention
        convolution, present_cnn = self._functional_causal_conv_block(
            block.conv,
            modulate(block.norm3(x), shift_conv, scale_conv),
            cnn_cache,
            current_frame_valid_length,
        )
        x = x + gate_conv * convolution
        x = x + gate_mlp * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))
        return x, present_cnn, present_att

    def forward(
        self,
        x_cfg: torch.Tensor,
        mu_cfg: torch.Tensor,
        t_cfg: torch.Tensor,
        spks_cfg: torch.Tensor,
        cond_cfg: torch.Tensor,
        past_estimator_cnn_cache: torch.Tensor,
        past_estimator_att_cache: torch.Tensor,
        past_cache_valid_length: torch.Tensor | None = None,
        current_frame_valid_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if past_cache_valid_length is None:
            past_cache_valid_length = torch.tensor(
                [past_estimator_att_cache.shape[3]], dtype=torch.int32, device=past_estimator_att_cache.device
            )
        if current_frame_valid_length is None:
            current_frame_valid_length = torch.tensor([x_cfg.shape[2]], dtype=torch.int32, device=x_cfg.device)
        t_embed = self.estimator.t_embedder(t_cfg).unsqueeze(1)
        features = torch.cat(
            (
                x_cfg,
                mu_cfg,
                spks_cfg.unsqueeze(-1).expand(-1, -1, x_cfg.shape[-1]),
                cond_cfg,
            ),
            dim=1,
        )
        derivative, present_cnn, present_att = self._functional_blocks_forward_chunk(
            features,
            t_embed,
            past_estimator_cnn_cache,
            past_estimator_att_cache,
            past_cache_valid_length,
            current_frame_valid_length,
        )
        return derivative, present_cnn, present_att


class HiftForwardWrapper(torch.nn.Module):
    """Export wrapper for the official HiFT vocoder."""

    def __init__(self, hift_module: torch.nn.Module, waveform_length: int | None = None):
        super().__init__()
        self.hift = hift_module
        self.n_fft = int(self.hift.istft_params["n_fft"])
        self.hop_len = int(self.hift.istft_params["hop_len"])
        self.freq_bins = self.n_fft // 2 + 1

        k = torch.arange(self.freq_bins, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(self.n_fft, dtype=torch.float32).unsqueeze(0)
        angle = (2.0 * math.pi / float(self.n_fft)) * (k @ n)

        scale = torch.ones(self.freq_bins, dtype=torch.float32)
        if self.freq_bins > 2:
            scale[1:-1] = 2.0

        overlap_kernel = torch.eye(self.n_fft, dtype=torch.float32).unsqueeze(1)

        self.register_buffer("istft_cos_basis", torch.cos(angle))
        self.register_buffer("istft_sin_basis", torch.sin(angle))
        self.register_buffer("istft_scale", scale)
        self.register_buffer("istft_window", self.hift.stft_window.detach().float())
        self.register_buffer("istft_window_sq", self.hift.stft_window.detach().float().pow(2))
        self.register_buffer("overlap_add_kernel", overlap_kernel)
        if waveform_length is None:
            self.istft_envelope = None
        else:
            frame_count = waveform_length // self.hop_len + 1
            envelope_frames = self.istft_window_sq.view(1, self.n_fft, 1).expand(1, self.n_fft, frame_count)
            envelope = F.conv_transpose1d(envelope_frames, overlap_kernel, stride=self.hop_len)
            pad = self.n_fft // 2
            self.register_buffer("istft_envelope", envelope[:, :, pad:-pad].clamp_min(1e-8))

    def _exportable_stft(self, x: torch.Tensor):
        spec = torch.stft(
            x,
            self.hift.istft_params["n_fft"],
            self.hift.istft_params["hop_len"],
            self.hift.istft_params["n_fft"],
            window=self.hift.stft_window.to(x.device),
            return_complex=False,
        )
        return spec[..., 0], spec[..., 1]

    def _exportable_istft(self, magnitude: torch.Tensor, phase: torch.Tensor):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        cos_basis = self.istft_cos_basis.to(device=real.device, dtype=real.dtype)
        sin_basis = self.istft_sin_basis.to(device=real.device, dtype=real.dtype)
        scale = self.istft_scale.to(device=real.device, dtype=real.dtype)
        window = self.istft_window.to(device=real.device, dtype=real.dtype)
        overlap_kernel = self.overlap_add_kernel.to(device=real.device, dtype=real.dtype)

        frames = real.unsqueeze(-1) * cos_basis.view(1, self.freq_bins, 1, self.n_fft) - imag.unsqueeze(
            -1
        ) * sin_basis.view(1, self.freq_bins, 1, self.n_fft)
        frames = (frames * scale.view(1, self.freq_bins, 1, 1)).sum(dim=1) / float(self.n_fft)
        frames = frames.permute(0, 2, 1).contiguous()
        frames = frames * window.view(1, self.n_fft, 1)

        waveform = F.conv_transpose1d(frames, overlap_kernel, stride=self.hop_len)

        pad = self.n_fft // 2
        if pad > 0 and waveform.shape[-1] > 2 * pad:
            waveform = waveform[:, :, pad:-pad]
        if self.istft_envelope is None:
            window_sq = self.istft_window_sq.to(device=real.device, dtype=real.dtype)
            envelope_frames = window_sq.view(1, self.n_fft, 1).expand(frames.shape[0], self.n_fft, frames.shape[-1])
            envelope = F.conv_transpose1d(envelope_frames, overlap_kernel, stride=self.hop_len)
            if pad > 0 and envelope.shape[-1] > 2 * pad:
                envelope = envelope[:, :, pad:-pad]
            waveform = waveform / envelope.clamp_min(1e-8)
        else:
            waveform = waveform / self.istft_envelope.to(device=real.device, dtype=real.dtype)
        return waveform.squeeze(1)

    def forward(self, speech_feat: torch.Tensor, cache_source: torch.Tensor) -> torch.Tensor:
        original_stft = getattr(self.hift, "_stft", None)
        original_istft = getattr(self.hift, "_istft", None)
        self.hift._stft = self._exportable_stft
        self.hift._istft = self._exportable_istft
        try:
            wav, _ = self.hift.forward(speech_feat=speech_feat, cache_source=cache_source)
        finally:
            if original_stft is not None:
                self.hift._stft = original_stft
            if original_istft is not None:
                self.hift._istft = original_istft
        return wav


class DeterministicHiftForwardWrapper(HiftForwardWrapper):
    def forward(
        self,
        speech_feat: torch.Tensor,
        phase_noise: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> torch.Tensor:
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
        original_stft = getattr(self.hift, "_stft", None)
        original_istft = getattr(self.hift, "_istft", None)
        self.hift._stft = self._exportable_stft
        self.hift._istft = self._exportable_istft
        try:
            return self.hift.decode(x=speech_feat, s=source)
        finally:
            if original_stft is not None:
                self.hift._stft = original_stft
            if original_istft is not None:
                self.hift._istft = original_istft
