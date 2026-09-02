from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .buckets import frame_bucket_key
from .graph import (
    DecoderStatic,
    DurationBidirectionalLSTMBlockStatic,
    DurationPredictorBidirectionalStatic,
    F0BranchStatic,
    GeneratorStatic,
    GraphArtifact,
    KokoroFrontBaseStatic,
    NoiseBranchStatic,
    PrefixReversedBidirectionalLSTMStatic,
    SourceMergeStatic,
    materialize_weight_norm,
    rewrite_albert_attention_double_mask_add,
    rewrite_depthwise_deconvolution,
)
from .host import (
    WAVEFORM_SAMPLES_PER_FRAME,
    duration_from_logits,
    duration_to_frame_indices,
    make_attention_mask,
    make_generator_rmsnorm_scales,
    make_reverse_idx,
)
from .precision_split import _export_partition
from .single_graph import StaticSineSource, _prefix_mask
from .static_dsp import StaticISTFT20, StaticSTFT20


TEXT_DURATION_ROLE = "text_duration"
FRAME_SYNTHESIS_ROLE = "frame_synthesis"
FRAME_ACOUSTIC_ROLE = "frame_acoustic"
PHASE_CORE_ROLE = "phase_core"
GENERATOR_ISTFT_ROLE = "generator_istft"


class TextDurationStatic(nn.Module):
    """T-only Front/Duration/Text graph.

    Duration reduction, F-bucket selection and indexed feature expansion are
    Host operations.  Consequently this graph has no F-dependent tensor or module.
    """

    def __init__(self, model: nn.Module, text_max_length: int) -> None:
        super().__init__()
        self.text_max_length = int(text_max_length)
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
        self.duration_predictor = DurationPredictorBidirectionalStatic(model, self.text_max_length)
        self.text_lstm = PrefixReversedBidirectionalLSTMStatic(
            model.text_encoder.lstm,
            self.text_max_length,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        style: Tensor,
        valid_len: Tensor,
        reverse_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
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
            values = block(values, prosody_style, reverse_indices, text_mask)
        duration_features = values
        duration_logits = self.duration_predictor(
            duration_features,
            reverse_indices,
            text_mask,
        )

        text_values = text_features.transpose(1, 2) * text_mask
        text_encoded = self.text_lstm(text_values, reverse_indices) * text_mask
        return duration_features, text_encoded.transpose(1, 2), duration_logits


class FrameCoreStatic(nn.Module):
    """F-only shared-LSTM, prosody, decoder, source and waveform backend."""

    def __init__(
        self,
        model: nn.Module,
        *,
        frame_max_length: int,
        seed: int,
        stft_pad_mode: str,
        stft_phase_mode: str,
        f0_norm_mode: str,
    ) -> None:
        super().__init__()
        if f0_norm_mode not in {"adain", "rmsnorm"}:
            raise ValueError("f0_norm_mode must be adain or rmsnorm")
        self.frame_max_length = int(frame_max_length)
        self.waveform_length = WAVEFORM_SAMPLES_PER_FRAME * self.frame_max_length
        self.shared_lstm = PrefixReversedBidirectionalLSTMStatic(
            model.predictor.shared,
            self.frame_max_length,
        )
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
        self.generator = GeneratorStatic(model, self.frame_max_length)
        self.istft = StaticISTFT20(self.frame_max_length)

    def forward_acoustic(
        self,
        encoded: Tensor,
        asr: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        reverse_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mask_f = _prefix_mask(valid_frames, self.frame_max_length, channel_axis=True)
        mask_2f = _prefix_mask(valid_frames * 2, 2 * self.frame_max_length, channel_axis=True)
        shared_input = encoded.transpose(1, 2) * mask_f.transpose(1, 2)
        shared = self.shared_lstm(shared_input, reverse_indices).transpose(1, 2) * mask_f
        prosody_style = style[:, 128:]
        f0 = self.f0_branch(shared, prosody_style, mask_f, mask_2f)
        noise = self.noise_branch(shared, prosody_style, mask_f, mask_2f)
        decoder_feature = self.decoder(
            asr,
            f0,
            noise,
            style[:, :128],
            mask_f,
            mask_2f,
        )
        return decoder_feature, f0

    def forward_generator_from_sine(
        self,
        decoder_feature: Tensor,
        sine: Tensor,
        f0: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        generator_norm_scales: Tensor,
    ) -> Tensor:
        waveform_mask = _prefix_mask(valid_frames * 600, self.waveform_length, channel_axis=True)
        f0_up = self.sine_source.upsample_f0(f0)
        harmonic_source = self.sine_source.merge_source(sine, f0_up, waveform_mask)
        return self._generator_from_source(
            decoder_feature,
            harmonic_source,
            style,
            valid_frames,
            generator_norm_scales,
        )

    def forward_generator_from_f0(
        self,
        decoder_feature: Tensor,
        f0: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        generator_norm_scales: Tensor,
    ) -> Tensor:
        waveform_mask = _prefix_mask(valid_frames * 600, self.waveform_length, channel_axis=True)
        harmonic_source = self.sine_source(f0, waveform_mask)
        return self._generator_from_source(
            decoder_feature,
            harmonic_source,
            style,
            valid_frames,
            generator_norm_scales,
        )

    def _generator_from_source(
        self,
        decoder_feature: Tensor,
        harmonic_source: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        generator_norm_scales: Tensor,
    ) -> Tensor:
        mask_2f = _prefix_mask(valid_frames * 2, 2 * self.frame_max_length, channel_axis=True)
        mask_20f = _prefix_mask(valid_frames * 20, 20 * self.frame_max_length, channel_axis=True)
        mask_spec = _prefix_mask(
            valid_frames * 120 + 1,
            120 * self.frame_max_length + 1,
            channel_axis=True,
        )
        waveform_mask = _prefix_mask(valid_frames * 600, self.waveform_length, channel_axis=True)
        valid_samples = valid_frames * 600
        harmonic = (
            self.harmonic_stft(
                harmonic_source,
                valid_samples if self.harmonic_stft.pad_mode == "length_aware_reflect" else None,
            )
            * mask_spec
        )
        spec_phase = self.generator(
            decoder_feature,
            style[:, :128],
            harmonic,
            mask_2f,
            mask_20f,
            mask_spec,
            generator_norm_scales,
        )
        return self.istft(spec_phase, waveform_mask)


class FrameSynthesisStatic(nn.Module):
    """Two-graph deployment continuation, including CumSum/phase/Sin on NPU."""

    def __init__(self, core: FrameCoreStatic) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        encoded: Tensor,
        asr: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        reverse_indices: Tensor,
        generator_norm_scales: Tensor,
    ) -> Tensor:
        decoder_feature, f0 = self.core.forward_acoustic(
            encoded,
            asr,
            style,
            valid_frames,
            reverse_indices,
        )
        return self.core.forward_generator_from_f0(
            decoder_feature,
            f0,
            style,
            valid_frames,
            generator_norm_scales,
        )


class FrameAcousticStatic(nn.Module):
    """Three-graph fallback first F graph, ending before phase accumulation."""

    def __init__(self, core: FrameCoreStatic) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        encoded: Tensor,
        asr: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        reverse_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        decoder_feature, f0 = self.core.forward_acoustic(
            encoded,
            asr,
            style,
            valid_frames,
            reverse_indices,
        )
        f0_up = self.core.sine_source.upsample_f0(f0)
        phase_increments = self.core.sine_source.prepare_phase_increments(f0_up)
        return decoder_feature, f0, phase_increments


class PhaseCoreStatic(nn.Module):
    def __init__(self, core: FrameCoreStatic) -> None:
        super().__init__()
        self.core = core

    def forward(self, phase_increments: Tensor) -> Tensor:
        return self.core.sine_source.phase_increments_to_sine(phase_increments)


class GeneratorISTFTStatic(nn.Module):
    def __init__(self, core: FrameCoreStatic) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        decoder_feature: Tensor,
        sine: Tensor,
        f0: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        generator_norm_scales: Tensor,
    ) -> Tensor:
        return self.core.forward_generator_from_sine(
            decoder_feature,
            sine,
            f0,
            style,
            valid_frames,
            generator_norm_scales,
        )


@dataclass(frozen=True)
class TextBucketExport:
    artifact: GraphArtifact
    feed: dict[str, Tensor]
    outputs: dict[str, Tensor]


@dataclass(frozen=True)
class FrameBucketExport:
    graphs: dict[str, GraphArtifact]
    feeds: dict[str, dict[str, Tensor]]
    outputs: dict[str, dict[str, Tensor]]
    waveform: Tensor


def prepare_independent_model(model: nn.Module) -> dict[str, int]:
    return {
        "albert_attention_double_mask_add": rewrite_albert_attention_double_mask_add(
            model.bert
        ),
        "weight_norm_materialized": materialize_weight_norm(model),
        "depthwise_deconv_output_padding_lowered": rewrite_depthwise_deconvolution(model),
    }


def run_duration_alignment_host(
    text_outputs: dict[str, Tensor],
    *,
    speed: Tensor,
    valid_len: Tensor,
    frame_max_length: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Expand token features on Host with a frame-to-token Gather index."""

    duration = duration_from_logits(text_outputs["duration_logits"], speed, valid_len)
    frame_indices, valid_frames = duration_to_frame_indices(
        duration,
        frame_max_length,
        valid_len,
    )
    duration_features = text_outputs["duration_features"]
    text_encoded = text_outputs["text_encoded"]
    if duration_features.ndim != 3 or duration_features.shape[:2] != duration.shape:
        raise ValueError("duration_features must be [1,T,C]")
    if text_encoded.ndim != 3 or text_encoded.shape[0] != 1 or text_encoded.shape[2] != duration.shape[1]:
        raise ValueError("text_encoded must be [1,C,T]")
    if duration_features.device != text_encoded.device:
        raise ValueError("duration_features and text_encoded must be on the same Host device")

    frame_indices = frame_indices.to(device=duration_features.device)
    frame_count = int(valid_frames.reshape(-1)[0].item())
    encoded = duration_features.new_zeros(1, duration_features.shape[2], frame_max_length)
    asr = text_encoded.new_zeros(1, text_encoded.shape[1], frame_max_length)
    encoded[:, :, :frame_count] = duration_features.index_select(1, frame_indices).transpose(1, 2)
    asr[:, :, :frame_count] = text_encoded.index_select(2, frame_indices)
    return {
        "encoded": encoded.contiguous(),
        "asr": asr.contiguous(),
        "valid_frames": valid_frames.contiguous(),
    }, duration


def export_text_bucket(
    *,
    model: nn.Module,
    sample: Any,
    output_path: str | Path,
    text_max_length: int,
    opset: int,
    validate_onnx: bool,
) -> TextBucketExport:
    module = TextDurationStatic(model, text_max_length).eval().cpu()
    feed = {
        "input_ids": sample.input_ids.detach().cpu(),
        "attention_mask": make_attention_mask(text_max_length, sample.valid_len),
        "style": sample.style.detach().cpu(),
        "valid_len": sample.valid_len.detach().cpu(),
        "reverse_indices": make_reverse_idx(text_max_length, sample.valid_len).to(
            dtype=torch.int64
        ),
    }
    with torch.no_grad():
        values = module(*tuple(feed.values()))
    outputs = dict(zip(("duration_features", "text_encoded", "duration_logits"), values, strict=True))
    artifact, _ = _export_partition(
        role=TEXT_DURATION_ROLE,
        module=module,
        feed=feed,
        expected=outputs,
        output_path=Path(output_path).expanduser().resolve(),
        opset=opset,
        simplify=False,
        validate_onnx=validate_onnx,
        expected_lstm_nodes=10,
        npu_graph=True,
    )
    return TextBucketExport(
        artifact=artifact,
        feed={name: value.detach().cpu().contiguous() for name, value in feed.items()},
        outputs={name: value.detach().cpu().contiguous() for name, value in outputs.items()},
    )


def export_frame_bucket(
    *,
    model: nn.Module,
    frame_inputs: dict[str, Tensor],
    style: Tensor,
    output_dir: str | Path,
    frame_max_length: int,
    seed: int,
    stft_pad_mode: str,
    stft_phase_mode: str,
    f0_norm_mode: str,
    opset: int,
    validate_onnx: bool,
    phase_on_npu: bool,
) -> FrameBucketExport:
    core = (
        FrameCoreStatic(
            model,
            frame_max_length=frame_max_length,
            seed=seed,
            stft_pad_mode=stft_pad_mode,
            stft_phase_mode=stft_phase_mode,
            f0_norm_mode=f0_norm_mode,
        )
        .eval()
        .cpu()
    )
    root = Path(output_dir).expanduser().resolve()
    bucket_key = frame_bucket_key(frame_max_length)
    feed = {
        "encoded": frame_inputs["encoded"].detach().cpu(),
        "asr": frame_inputs["asr"].detach().cpu(),
        "style": style.detach().cpu(),
        "valid_frames": frame_inputs["valid_frames"].detach().cpu(),
        "reverse_indices": make_reverse_idx(
            frame_max_length,
            frame_inputs["valid_frames"],
        ).to(dtype=torch.int64),
    }
    generator_norm_scales = make_generator_rmsnorm_scales(
        feed["valid_frames"],
        frame_max_length,
    )
    graphs: dict[str, GraphArtifact] = {}
    feeds: dict[str, dict[str, Tensor]] = {}
    outputs: dict[str, dict[str, Tensor]] = {}

    if phase_on_npu:
        module = FrameSynthesisStatic(core).eval()
        synthesis_feed = {**feed, "generator_norm_scales": generator_norm_scales}
        with torch.no_grad():
            waveform = module(*tuple(synthesis_feed.values()))
        expected = {"waveform": waveform}
        artifact, _ = _export_partition(
            role=FRAME_SYNTHESIS_ROLE,
            module=module,
            feed=synthesis_feed,
            expected=expected,
            output_path=(
                root / FRAME_SYNTHESIS_ROLE / bucket_key / f"kokoro_frame_synthesis_b1_f{frame_max_length}.onnx"
            ),
            opset=opset,
            simplify=False,
            validate_onnx=validate_onnx,
            expected_lstm_nodes=2,
            npu_graph=True,
        )
        graphs[FRAME_SYNTHESIS_ROLE] = artifact
        feeds[FRAME_SYNTHESIS_ROLE] = {
            name: value.detach().cpu().contiguous() for name, value in synthesis_feed.items()
        }
        outputs[FRAME_SYNTHESIS_ROLE] = {"waveform": waveform.detach().cpu().contiguous()}
        return FrameBucketExport(graphs, feeds, outputs, waveform.detach().cpu())

    acoustic_module = FrameAcousticStatic(core).eval()
    with torch.no_grad():
        acoustic_values = acoustic_module(*tuple(feed.values()))
    acoustic_outputs = dict(zip(("decoder_feature", "f0", "phase_increments"), acoustic_values, strict=True))
    acoustic_artifact, _ = _export_partition(
        role=FRAME_ACOUSTIC_ROLE,
        module=acoustic_module,
        feed=feed,
        expected=acoustic_outputs,
        output_path=(root / FRAME_ACOUSTIC_ROLE / bucket_key / f"kokoro_frame_acoustic_b1_f{frame_max_length}.onnx"),
        opset=opset,
        simplify=False,
        validate_onnx=validate_onnx,
        expected_lstm_nodes=2,
        npu_graph=True,
    )
    graphs[FRAME_ACOUSTIC_ROLE] = acoustic_artifact
    feeds[FRAME_ACOUSTIC_ROLE] = {name: value.detach().cpu().contiguous() for name, value in feed.items()}
    outputs[FRAME_ACOUSTIC_ROLE] = {name: value.detach().cpu().contiguous() for name, value in acoustic_outputs.items()}

    phase_module = PhaseCoreStatic(core).eval()
    phase_feed = {"phase_increments": acoustic_outputs["phase_increments"]}
    with torch.no_grad():
        sine = phase_module(*tuple(phase_feed.values()))
    phase_outputs = {"sine": sine}
    phase_artifact, _ = _export_partition(
        role=PHASE_CORE_ROLE,
        module=phase_module,
        feed=phase_feed,
        expected=phase_outputs,
        output_path=(root / PHASE_CORE_ROLE / bucket_key / f"kokoro_phase_core_b1_f{frame_max_length}.onnx"),
        opset=opset,
        simplify=False,
        validate_onnx=validate_onnx,
        expected_lstm_nodes=0,
        npu_graph=False,
    )
    graphs[PHASE_CORE_ROLE] = phase_artifact
    feeds[PHASE_CORE_ROLE] = {name: value.detach().cpu().contiguous() for name, value in phase_feed.items()}
    outputs[PHASE_CORE_ROLE] = {"sine": sine.detach().cpu().contiguous()}

    generator_module = GeneratorISTFTStatic(core).eval()
    generator_feed = {
        "decoder_feature": acoustic_outputs["decoder_feature"],
        "sine": sine,
        "f0": acoustic_outputs["f0"],
        "style": style.detach().cpu(),
        "valid_frames": frame_inputs["valid_frames"].detach().cpu(),
        "generator_norm_scales": generator_norm_scales,
    }
    with torch.no_grad():
        waveform = generator_module(*tuple(generator_feed.values()))
    generator_outputs = {"waveform": waveform}
    generator_artifact, _ = _export_partition(
        role=GENERATOR_ISTFT_ROLE,
        module=generator_module,
        feed=generator_feed,
        expected=generator_outputs,
        output_path=(root / GENERATOR_ISTFT_ROLE / bucket_key / f"kokoro_generator_istft_b1_f{frame_max_length}.onnx"),
        opset=opset,
        simplify=False,
        validate_onnx=validate_onnx,
        expected_lstm_nodes=0,
        npu_graph=True,
    )
    graphs[GENERATOR_ISTFT_ROLE] = generator_artifact
    feeds[GENERATOR_ISTFT_ROLE] = {name: value.detach().cpu().contiguous() for name, value in generator_feed.items()}
    outputs[GENERATOR_ISTFT_ROLE] = {"waveform": waveform.detach().cpu().contiguous()}
    return FrameBucketExport(graphs, feeds, outputs, waveform.detach().cpu())


__all__ = [
    "FRAME_ACOUSTIC_ROLE",
    "FRAME_SYNTHESIS_ROLE",
    "GENERATOR_ISTFT_ROLE",
    "PHASE_CORE_ROLE",
    "TEXT_DURATION_ROLE",
    "FrameAcousticStatic",
    "FrameBucketExport",
    "FrameCoreStatic",
    "FrameSynthesisStatic",
    "GeneratorISTFTStatic",
    "PhaseCoreStatic",
    "TextBucketExport",
    "TextDurationStatic",
    "export_frame_bucket",
    "export_text_bucket",
    "prepare_independent_model",
    "run_duration_alignment_host",
]
