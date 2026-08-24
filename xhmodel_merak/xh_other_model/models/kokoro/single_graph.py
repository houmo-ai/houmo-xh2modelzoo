from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .assets import sha256
from .graph import (
    FORBIDDEN_NPU_OPS,
    DecoderStatic,
    DurationBidirectionalLSTMBlockStatic,
    DurationPredictorBidirectionalStatic,
    F0BranchStatic,
    GeneratorStatic,
    GraphArtifact,
    KokoroFrontBaseStatic,
    NoiseBranchStatic,
    PackedBidirectionalLSTMStatic,
    SourceMergeStatic,
    _canonicalize_static_graph,
    _fix_negative_transpose_permutations,
    _inject_lstm_input_shapes,
    _value_contract,
    materialize_weight_norm,
    rewrite_depthwise_deconvolution,
)
from .host import StaticSample, make_attention_mask
from .static_dsp import StaticISTFT20, StaticSTFT20


SINGLE_GRAPH_ROLE = "end_to_end"


def _prefix_mask(length: Tensor, maximum: int, *, channel_axis: bool) -> Tensor:
    positions = torch.arange(maximum, device=length.device, dtype=torch.int64)
    mask = positions < length.reshape(-1)[0].to(dtype=torch.int64)
    if channel_axis:
        return mask.to(dtype=torch.float32).reshape(1, 1, maximum)
    return mask.to(dtype=torch.float32).reshape(1, maximum, 1)


def _reverse_prefix_index(length: Tensor, maximum: int) -> Tensor:
    positions = torch.arange(maximum, device=length.device, dtype=torch.int64)
    # Keep the graph executable when the selected F bucket overflows.  The
    # un-clamped length is still returned to Host so it can retry a larger
    # bucket; clamping here only prevents an out-of-range Gather meanwhile.
    valid = length.reshape(-1)[0].to(dtype=torch.int64)
    valid = torch.where(valid < 0, torch.zeros_like(valid), valid)
    valid = torch.where(valid > maximum, torch.full_like(valid, maximum), valid)
    return torch.where(positions < valid, valid - 1 - positions, positions)


class StaticAlignment(nn.Module):
    """CumSum + Compare + MatMul monotonic length regulator for fixed T/F."""

    def __init__(self, frame_max_length: int) -> None:
        super().__init__()
        if frame_max_length <= 0:
            raise ValueError("frame_max_length must be positive")
        self.frame_max_length = int(frame_max_length)
        self.register_buffer(
            "frame_positions",
            torch.arange(frame_max_length, dtype=torch.float32).reshape(1, 1, -1),
            persistent=False,
        )

    def forward(
        self,
        duration_features: Tensor,
        text_features: Tensor,
        duration: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        duration_float = duration.to(dtype=torch.float32)
        ends = torch.cumsum(duration_float, dim=1)
        starts = ends - duration_float
        alignment = ((self.frame_positions >= starts.unsqueeze(-1)) & (self.frame_positions < ends.unsqueeze(-1))).to(
            dtype=duration_features.dtype
        )
        encoded = torch.matmul(duration_features.transpose(1, 2), alignment)
        asr = torch.matmul(text_features, alignment)
        valid_frames = ends[:, -1].to(dtype=torch.int32)
        return encoded, asr, alignment, valid_frames


class StaticSineSource(nn.Module):
    """Fixed-seed SineGen and SourceMerge without ONNX random operators."""

    SAMPLE_RATE = 24_000
    HARMONICS = 9
    F0_UPSAMPLE = 300

    def __init__(self, source_merge: SourceMergeStatic, frame_max_length: int, seed: int) -> None:
        super().__init__()
        self.source_merge = source_merge
        self.frame_max_length = int(frame_max_length)
        self.waveform_length = 600 * self.frame_max_length
        self.register_buffer(
            "harmonic_numbers",
            torch.arange(1, self.HARMONICS + 1, dtype=torch.float32).reshape(1, 1, -1),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            initial_phase = torch.rand((1, self.HARMONICS), dtype=torch.float32)
            initial_phase[:, 0] = 0.0
            noise_layout = torch.empty(
                1,
                self.HARMONICS,
                self.waveform_length,
                dtype=torch.float32,
            ).transpose(1, 2)
            noise = torch.randn_like(noise_layout)
        self.register_buffer("initial_phase", initial_phase)
        self.register_buffer("noise", noise.contiguous())

    def upsample_f0(self, f0_prediction: Tensor) -> Tensor:
        return F.interpolate(
            f0_prediction.float().unsqueeze(1),
            scale_factor=self.F0_UPSAMPLE,
        ).transpose(1, 2)

    def prepare_phase_increments(self, f0_up: Tensor) -> Tensor:
        radians = torch.remainder(
            f0_up * self.harmonic_numbers / self.SAMPLE_RATE,
            1.0,
        )
        radians = torch.cat(
            [
                radians[:, :1] + self.initial_phase.unsqueeze(1),
                radians[:, 1:],
            ],
            dim=1,
        )
        return F.interpolate(
            radians.transpose(1, 2),
            scale_factor=1 / self.F0_UPSAMPLE,
            mode="linear",
        ).transpose(1, 2)

    def phase_increments_to_sine(self, phase_increments: Tensor) -> Tensor:
        phase = torch.cumsum(phase_increments.float(), dim=1) * (2 * torch.pi)
        phase = F.interpolate(
            phase.transpose(1, 2) * self.F0_UPSAMPLE,
            scale_factor=self.F0_UPSAMPLE,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase) * 0.1

    def merge_source(
        self,
        sine: Tensor,
        f0_up: Tensor,
        waveform_mask: Tensor,
    ) -> Tensor:
        voiced = (f0_up > 10).to(dtype=sine.dtype)
        noise_amplitude = voiced * 0.003 + (1.0 - voiced) * (0.1 / 3.0)
        sine = sine * voiced + noise_amplitude * self.noise
        source = self.source_merge(sine)
        return source * waveform_mask.transpose(1, 2)

    def forward(self, f0_prediction: Tensor, waveform_mask: Tensor) -> Tensor:
        f0_up = self.upsample_f0(f0_prediction)
        phase_increments = self.prepare_phase_increments(f0_up)
        sine = self.phase_increments_to_sine(phase_increments)
        return self.merge_source(sine, f0_up, waveform_mask)


class KokoroEndToEndStatic(nn.Module):
    """Batch-1 fixed T/F Kokoro graph with only G2P/voice lookup on Host."""

    def __init__(
        self,
        model: nn.Module,
        *,
        text_max_length: int,
        frame_max_length: int,
        seed: int,
        stft_pad_mode: str = "length_aware_reflect",
        stft_phase_mode: str = "cordic",
        f0_norm_mode: str = "adain",
    ) -> None:
        super().__init__()
        if text_max_length <= 0 or frame_max_length <= 0:
            raise ValueError("text_max_length and frame_max_length must be positive")
        if f0_norm_mode not in {"adain", "rmsnorm"}:
            raise ValueError("f0_norm_mode must be adain or rmsnorm")
        self.text_max_length = int(text_max_length)
        self.frame_max_length = int(frame_max_length)
        self.waveform_length = 600 * self.frame_max_length
        self.front = KokoroFrontBaseStatic(model, self.text_max_length)
        duration_encoder = model.predictor.text_encoder
        self.duration_blocks = nn.ModuleList(
            [
                DurationBidirectionalLSTMBlockStatic(
                    duration_encoder.lstms[2 * index],
                    duration_encoder.lstms[2 * index + 1],
                    self.text_max_length,
                )
                for index in range(3)
            ]
        )
        self.duration_predictor = DurationPredictorBidirectionalStatic(
            model,
            self.text_max_length,
        )
        self.text_lstm = PackedBidirectionalLSTMStatic(
            model.text_encoder.lstm,
            self.text_max_length,
        )
        self.shared_lstm = PackedBidirectionalLSTMStatic(
            model.predictor.shared,
            self.frame_max_length,
        )
        self.alignment = StaticAlignment(self.frame_max_length)
        self.f0_branch = F0BranchStatic(
            model,
            self.frame_max_length,
            use_rmsnorm=f0_norm_mode == "rmsnorm",
        )
        self.noise_branch = NoiseBranchStatic(model, self.frame_max_length)
        self.decoder = DecoderStatic(model, self.frame_max_length)
        self.sine_source = StaticSineSource(
            SourceMergeStatic(model),
            self.frame_max_length,
            seed,
        )
        self.harmonic_stft = StaticSTFT20(
            pad_mode=stft_pad_mode,
            waveform_length=(self.waveform_length if stft_pad_mode == "length_aware_reflect" else None),
            phase_mode=stft_phase_mode,
        )
        self.generator = GeneratorStatic(model)
        self.istft = StaticISTFT20(self.frame_max_length)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        style: Tensor,
        speed: Tensor,
        valid_len: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        shared_input, generator_feature, duration, valid_frames, f0 = self.forward_acoustic(
            input_ids,
            attention_mask,
            style,
            speed,
            valid_len,
        )
        del shared_input
        harmonic = self.forward_harmonic_from_f0(f0, valid_frames)
        waveform = self.forward_generator_istft(
            generator_feature,
            harmonic,
            style,
            valid_frames,
        )
        return waveform, duration, valid_frames

    def forward_acoustic(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        style: Tensor,
        speed: Tensor,
        valid_len: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run the acoustic graph and expose its precision-split boundaries."""

        duration_base, text_features = self.front(input_ids, attention_mask, valid_len)
        text_mask = _prefix_mask(valid_len, self.text_max_length, channel_axis=False)
        prosody_style = style[:, 128:]
        values = duration_base.transpose(1, 2)
        values = (
            torch.cat(
                [
                    values,
                    prosody_style.unsqueeze(1).expand(-1, self.text_max_length, -1),
                ],
                dim=-1,
            )
            * text_mask
        )
        for block in self.duration_blocks:
            values = block(
                values,
                prosody_style,
                valid_len,
                text_mask,
            )
        duration_features = values
        duration_logits = self.duration_predictor(
            duration_features,
            valid_len,
            text_mask,
        )
        duration = torch.round(torch.sigmoid(duration_logits.float()).sum(dim=-1) / speed.reshape(1, 1).float()).clamp(
            min=1.0
        )
        duration = duration * text_mask.squeeze(-1)

        text_values = text_features.transpose(1, 2) * text_mask
        text_encoded = self.text_lstm(
            text_values,
            valid_len,
        )
        text_encoded = text_encoded * text_mask
        text_encoded = text_encoded.transpose(1, 2)

        encoded, asr, _alignment, valid_frames = self.alignment(
            duration_features,
            text_encoded,
            duration,
        )
        mask_f = _prefix_mask(valid_frames, self.frame_max_length, channel_axis=True)
        mask_2f = _prefix_mask(valid_frames * 2, 2 * self.frame_max_length, channel_axis=True)
        shared_values = encoded.transpose(1, 2) * mask_f.transpose(1, 2)
        shared = self._forward_shared(shared_values, valid_frames, mask_f)

        f0 = self.f0_branch(shared, prosody_style, mask_f, mask_2f)
        noise = self.noise_branch(shared, prosody_style, mask_f, mask_2f)
        decoder_style = style[:, :128]
        generator_feature = self.decoder(
            asr,
            f0,
            noise,
            decoder_style,
            mask_f,
            mask_2f,
        )
        return (
            shared_values,
            generator_feature,
            duration.to(dtype=torch.int64),
            valid_frames,
            f0,
        )

    def _forward_shared(
        self,
        shared_input: Tensor,
        valid_frames: Tensor,
        mask_f: Tensor,
    ) -> Tensor:
        shared = self.shared_lstm(
            shared_input,
            valid_frames,
        )
        return shared.transpose(1, 2) * mask_f

    def forward_harmonic_bridge(
        self,
        shared_input: Tensor,
        style: Tensor,
        valid_frames: Tensor,
    ) -> Tensor:
        """Recompute the phase-sensitive harmonic source in FP32 on Host."""

        mask_f = _prefix_mask(valid_frames, self.frame_max_length, channel_axis=True)
        mask_2f = _prefix_mask(valid_frames * 2, 2 * self.frame_max_length, channel_axis=True)
        shared = self._forward_shared(shared_input, valid_frames, mask_f)
        f0 = self.f0_branch(shared, style[:, 128:], mask_f, mask_2f)
        return self.forward_harmonic_from_f0(f0, valid_frames)

    def forward_harmonic_from_f0(
        self,
        f0: Tensor,
        valid_frames: Tensor,
    ) -> Tensor:
        mask_spec = _prefix_mask(
            valid_frames * 120 + 1,
            120 * self.frame_max_length + 1,
            channel_axis=True,
        )
        waveform_mask = _prefix_mask(
            valid_frames * 600,
            self.waveform_length,
            channel_axis=True,
        )
        harmonic_source = self.sine_source(f0, waveform_mask)
        valid_samples = valid_frames * 600
        return (
            self.harmonic_stft(
                harmonic_source,
                valid_samples if self.harmonic_stft.pad_mode == "length_aware_reflect" else None,
            )
            * mask_spec
        )

    def forward_generator_from_sine(
        self,
        generator_feature: Tensor,
        sine: Tensor,
        f0: Tensor,
        style: Tensor,
        valid_frames: Tensor,
    ) -> Tensor:
        mask_spec = _prefix_mask(
            valid_frames * 120 + 1,
            120 * self.frame_max_length + 1,
            channel_axis=True,
        )
        waveform_mask = _prefix_mask(
            valid_frames * 600,
            self.waveform_length,
            channel_axis=True,
        )
        f0_up = self.sine_source.upsample_f0(f0)
        harmonic_source = self.sine_source.merge_source(
            sine,
            f0_up,
            waveform_mask,
        )
        valid_samples = valid_frames * 600
        harmonic = (
            self.harmonic_stft(
                harmonic_source,
                valid_samples if self.harmonic_stft.pad_mode == "length_aware_reflect" else None,
            )
            * mask_spec
        )
        return self.forward_generator_istft(
            generator_feature,
            harmonic,
            style,
            valid_frames,
        )

    def forward_generator_istft(
        self,
        generator_feature: Tensor,
        harmonic: Tensor,
        style: Tensor,
        valid_frames: Tensor,
    ) -> Tensor:
        mask_2f = _prefix_mask(valid_frames * 2, 2 * self.frame_max_length, channel_axis=True)
        mask_20f = _prefix_mask(
            valid_frames * 20,
            20 * self.frame_max_length,
            channel_axis=True,
        )
        mask_spec = _prefix_mask(
            valid_frames * 120 + 1,
            120 * self.frame_max_length + 1,
            channel_axis=True,
        )
        waveform_mask = _prefix_mask(
            valid_frames * 600,
            self.waveform_length,
            channel_axis=True,
        )
        decoder_style = style[:, :128]
        spec_phase = self.generator(
            generator_feature,
            decoder_style,
            harmonic,
            mask_2f,
            mask_20f,
            mask_spec,
        )
        return self.istft(spec_phase, waveform_mask)


@dataclass(frozen=True)
class SingleGraphExport:
    artifact: GraphArtifact
    feed: dict[str, Tensor]
    outputs: dict[str, Tensor]
    validation: dict[str, Any]
    rewrites: dict[str, int]


def build_end_to_end_static(
    model: nn.Module,
    *,
    text_max_length: int,
    frame_max_length: int,
    seed: int,
    stft_pad_mode: str,
    stft_phase_mode: str = "cordic",
    f0_norm_mode: str = "adain",
) -> tuple[KokoroEndToEndStatic, dict[str, int]]:
    rewrites = {
        "weight_norm_materialized": materialize_weight_norm(model),
        # xhquant currently loses ConvTranspose1d.output_padding while it
        # promotes the operator to ConvTranspose2d.  Lower these three exact
        # depthwise K3/S2/P1/OP1 layers in the static wrapper so every F bucket
        # keeps the required 2F extent.  This is a source-level compatibility
        # lowering, not an ONNX post-export optimization pass.
        "depthwise_deconv_output_padding_lowered": rewrite_depthwise_deconvolution(model),
    }
    wrapper = KokoroEndToEndStatic(
        model,
        text_max_length=text_max_length,
        frame_max_length=frame_max_length,
        seed=seed,
        stft_pad_mode=stft_pad_mode,
        stft_phase_mode=stft_phase_mode,
        f0_norm_mode=f0_norm_mode,
    ).eval()
    return wrapper, rewrites


def export_end_to_end_static(
    *,
    model: nn.Module,
    sample: StaticSample,
    dynamic_waveform: Tensor,
    dynamic_duration: Tensor,
    output_path: str | Path,
    text_max_length: int,
    frame_max_length: int,
    seed: int,
    stft_pad_mode: str,
    stft_phase_mode: str,
    opset: int,
    simplify: bool,
    validate_onnx: bool,
    f0_norm_mode: str = "adain",
) -> SingleGraphExport:
    wrapper, rewrites = build_end_to_end_static(
        model,
        text_max_length=text_max_length,
        frame_max_length=frame_max_length,
        seed=seed,
        stft_pad_mode=stft_pad_mode,
        stft_phase_mode=stft_phase_mode,
        f0_norm_mode=f0_norm_mode,
    )
    feed = {
        "input_ids": sample.input_ids,
        "attention_mask": make_attention_mask(text_max_length, sample.valid_len),
        "style": sample.style,
        "speed": sample.speed,
        "valid_len": sample.valid_len,
    }
    inputs = tuple(feed.values())
    with torch.no_grad():
        waveform, duration, valid_frames = wrapper(*inputs)
    valid_samples = int(valid_frames.item()) * 600
    reference_waveform = dynamic_waveform.reshape(-1).float().cpu()
    candidate_waveform = waveform.reshape(-1)[:valid_samples].float().cpu()
    reference_duration = dynamic_duration.reshape(-1).cpu()
    candidate_duration = duration.reshape(-1)[: int(sample.valid_len.item())].cpu()
    validation = _metrics(
        reference_waveform,
        candidate_waveform,
        reference_duration,
        candidate_duration,
    )

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        path,
        input_names=list(feed),
        output_names=["waveform", "duration", "valid_frames"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    if simplify:
        import onnxsim

        graph = onnx.load(path)
        graph, checked = onnxsim.simplify(
            graph,
            check_n=0,
            perform_optimization=True,
            skip_fuse_bn=True,
        )
        if not checked:
            raise RuntimeError("onnxsim failed to simplify the Kokoro single graph")
        onnx.save(graph, path)
    structural = _canonicalize_static_graph(path)
    structural["negative_transpose_permutations"] = _fix_negative_transpose_permutations(path)
    structural["lstm_input_shapes"] = _inject_lstm_input_shapes(path)
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph)
    op_counts: dict[str, int] = {}
    for node in graph.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    lstm_nodes = [node for node in graph.graph.node if node.op_type == "LSTM"]
    if len(lstm_nodes) != 6:
        raise RuntimeError(f"single graph must preserve 6 high-level bidirectional LSTM nodes, got {len(lstm_nodes)}")
    for node in lstm_nodes:
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if attributes.get("direction") != b"bidirectional":
            raise RuntimeError(f"single graph LSTM {node.name!r} is not bidirectional")
    forbidden = sorted(FORBIDDEN_NPU_OPS.intersection(op_counts))
    if forbidden:
        raise RuntimeError(f"single graph contains forbidden NPU operators: {forbidden}")
    if op_counts.get("CumSum", 0) < 2:
        raise RuntimeError("single graph must keep Alignment and SineGen CumSum operations")
    onnx_metrics: tuple[dict[str, float], ...] = tuple()
    if validate_onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        actual = session.run(
            None,
            {name: value.detach().cpu().numpy() for name, value in feed.items()},
        )
        onnx_metrics = tuple(
            _tensor_metric(name, expected.detach().cpu().numpy(), value)
            for name, expected, value in zip(
                ("waveform", "duration", "valid_frames"),
                (waveform, duration, valid_frames),
                actual,
                strict=True,
            )
        )
    artifact = GraphArtifact(
        role=SINGLE_GRAPH_ROLE,
        path=path,
        input_names=tuple(feed),
        output_names=("waveform", "duration", "valid_frames"),
        input_contracts=tuple(_value_contract(value) for value in graph.graph.input),
        output_contracts=tuple(_value_contract(value) for value in graph.graph.output),
        node_count=len(graph.graph.node),
        op_counts=dict(sorted(op_counts.items())),
        onnx_sha256=sha256(path),
        pytorch_vs_onnx=onnx_metrics,
        structural_rewrites=structural,
    )
    return SingleGraphExport(
        artifact=artifact,
        feed={name: value.detach().cpu() for name, value in feed.items()},
        outputs={
            "waveform": waveform.detach().cpu(),
            "duration": duration.detach().cpu(),
            "valid_frames": valid_frames.detach().cpu(),
        },
        validation=validation,
        rewrites=rewrites,
    )


def _tensor_metric(name: str, expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if expected.shape != actual.shape:
        raise RuntimeError(f"{name} shape mismatch: {expected.shape} != {actual.shape}")
    if not np.isfinite(actual).all():
        non_finite = int(actual.size - np.isfinite(actual).sum())
        raise RuntimeError(f"{name} contains {non_finite} non-finite ONNX values")
    expected64 = expected.astype(np.float64).reshape(-1)
    actual64 = actual.astype(np.float64).reshape(-1)
    difference = np.abs(expected64 - actual64)
    denominator = float(np.linalg.norm(expected64) * np.linalg.norm(actual64))
    result = {
        "name": name,
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }
    if denominator:
        result["cosine"] = float(np.dot(expected64, actual64) / denominator)
    if name == "waveform":
        result.update(_spectral_metrics(expected64, actual64))
    return result


def _metrics(
    reference_waveform: Tensor,
    candidate_waveform: Tensor,
    reference_duration: Tensor,
    candidate_duration: Tensor,
) -> dict[str, Any]:
    if reference_waveform.shape != candidate_waveform.shape:
        raise RuntimeError(f"waveform shape mismatch: {reference_waveform.shape} != {candidate_waveform.shape}")
    difference = (reference_waveform - candidate_waveform).abs()
    denominator = torch.linalg.vector_norm(reference_waveform) * torch.linalg.vector_norm(candidate_waveform)
    result = {
        "duration_exact": bool(torch.equal(reference_duration, candidate_duration)),
        "valid_samples": int(reference_waveform.numel()),
        "waveform_max_abs": float(difference.max().item()),
        "waveform_mean_abs": float(difference.mean().item()),
        "waveform_cosine": float(torch.dot(reference_waveform, candidate_waveform) / denominator)
        if denominator
        else 1.0,
    }
    result.update(
        _spectral_metrics(
            reference_waveform.detach().cpu().numpy(),
            candidate_waveform.detach().cpu().numpy(),
        )
    )
    return result


def _spectral_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32).reshape(1, -1)
    candidate_tensor = torch.as_tensor(candidate, dtype=torch.float32).reshape(1, -1)
    window = torch.hann_window(1024, periodic=True)
    reference_magnitude = torch.stft(
        reference_tensor,
        n_fft=1024,
        hop_length=256,
        window=window,
        return_complex=True,
    ).abs()
    candidate_magnitude = torch.stft(
        candidate_tensor,
        n_fft=1024,
        hop_length=256,
        window=window,
        return_complex=True,
    ).abs()

    def cosine(left: Tensor, right: Tensor) -> float:
        left = left.reshape(-1).double()
        right = right.reshape(-1).double()
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        return float(torch.dot(left, right) / denominator) if denominator else 1.0

    return {
        "stft_magnitude_cosine": cosine(reference_magnitude, candidate_magnitude),
        "log_stft_magnitude_cosine": cosine(
            torch.log1p(reference_magnitude),
            torch.log1p(candidate_magnitude),
        ),
    }


__all__ = [
    "KokoroEndToEndStatic",
    "SINGLE_GRAPH_ROLE",
    "SingleGraphExport",
    "StaticAlignment",
    "StaticSineSource",
    "build_end_to_end_static",
    "export_end_to_end_static",
]
