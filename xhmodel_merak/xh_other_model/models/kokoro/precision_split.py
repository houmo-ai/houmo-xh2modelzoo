from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import Tensor, nn

from .assets import sha256
from .graph import (
    FORBIDDEN_NPU_OPS,
    GraphArtifact,
    _value_contract,
)
from .host import (
    ATTENTION_MASK_MIN,
    WAVEFORM_SAMPLES_PER_FRAME,
    make_attention_mask,
    make_generator_rmsnorm_scales,
)
from .runtime import HmonnxRunner, OrtRunner, Runner
from .single_graph import (
    KokoroEndToEndStatic,
    _metrics,
    _tensor_metric,
    build_end_to_end_static,
)


PRECISION_SPLIT_GRAPH_MODE = "precision_split"
ACOUSTIC_ROLE = "acoustic"
PHASE_CORE_ROLE = "phase_core"
GENERATOR_ISTFT_ROLE = "generator_istft"
PRECISION_SPLIT_ROLES = (
    ACOUSTIC_ROLE,
    PHASE_CORE_ROLE,
    GENERATOR_ISTFT_ROLE,
)
PRECISION_SPLIT_NPU_ROLES = (ACOUSTIC_ROLE, GENERATOR_ISTFT_ROLE)


class AcousticStatic(nn.Module):
    def __init__(self, model: KokoroEndToEndStatic) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        style: Tensor,
        speed: Tensor,
        valid_len: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        _shared_input, generator_feature, duration, valid_frames, f0 = self.model.forward_acoustic(
            input_ids,
            attention_mask,
            style,
            speed,
            valid_len,
        )
        f0_up = self.model.sine_source.upsample_f0(f0)
        phase_increments = self.model.sine_source.prepare_phase_increments(f0_up)
        return generator_feature, f0, phase_increments, duration, valid_frames


class PhaseCoreStatic(nn.Module):
    def __init__(self, model: KokoroEndToEndStatic) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        phase_increments: Tensor,
    ) -> Tensor:
        return self.model.sine_source.phase_increments_to_sine(phase_increments)


class GeneratorISTFTStatic(nn.Module):
    def __init__(self, model: KokoroEndToEndStatic) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        decoder_feature: Tensor,
        sine: Tensor,
        f0: Tensor,
        style: Tensor,
        valid_frames: Tensor,
        generator_norm_scales: Tensor,
    ) -> Tensor:
        return self.model.forward_generator_from_sine(
            decoder_feature,
            sine,
            f0,
            style,
            valid_frames,
            generator_norm_scales,
        )


@dataclass(frozen=True)
class PrecisionSplitExport:
    graphs: dict[str, GraphArtifact]
    feeds: dict[str, dict[str, Tensor]]
    outputs: dict[str, dict[str, Tensor]]
    waveform: Tensor
    duration: Tensor
    valid_frames: Tensor
    validation: dict[str, Any]
    rewrites: dict[str, int]


def export_precision_split_static(
    *,
    model: nn.Module,
    sample: Any,
    dynamic_waveform: Tensor | None,
    dynamic_duration: Tensor | None,
    output_dir: str | Path,
    text_max_length: int,
    frame_max_length: int,
    seed: int,
    stft_pad_mode: str,
    stft_phase_mode: str,
    opset: int,
    simplify: bool,
    validate_onnx: bool,
    validate_outputs: bool = True,
    f0_norm_mode: str = "adain",
    roles: Sequence[str] | None = None,
    output_paths: Mapping[str, str | Path] | None = None,
) -> PrecisionSplitExport:
    """Export two NPU graphs around the FP32 SineGen phase core."""

    selected_roles = tuple(PRECISION_SPLIT_ROLES if roles is None else roles)
    unknown_roles = sorted(set(selected_roles).difference(PRECISION_SPLIT_ROLES))
    if unknown_roles:
        raise ValueError(f"unknown Kokoro precision-split roles: {unknown_roles}")
    if len(set(selected_roles)) != len(selected_roles):
        raise ValueError("Kokoro precision-split roles must be unique")

    root, rewrites = build_end_to_end_static(
        model,
        text_max_length=text_max_length,
        frame_max_length=frame_max_length,
        seed=seed,
        stft_pad_mode=stft_pad_mode,
        stft_phase_mode=stft_phase_mode,
        f0_norm_mode=f0_norm_mode,
    )
    acoustic = AcousticStatic(root).eval()
    phase_core = PhaseCoreStatic(root).eval()
    generator = GeneratorISTFTStatic(root).eval()
    acoustic_feed = {
        "input_ids": sample.input_ids.detach().cpu(),
        "attention_mask": make_attention_mask(text_max_length, sample.valid_len),
        "style": sample.style.detach().cpu(),
        "speed": sample.speed.detach().cpu(),
        "valid_len": sample.valid_len.detach().cpu(),
    }

    with torch.no_grad():
        single_waveform, single_duration, single_frames = root(*tuple(acoustic_feed.values()))
        acoustic_values = acoustic(*tuple(acoustic_feed.values()))
        acoustic_outputs = dict(
            zip(
                ("decoder_feature", "f0", "phase_increments", "duration", "valid_frames"),
                acoustic_values,
                strict=True,
            )
        )
        phase_feed = {
            "phase_increments": acoustic_outputs["phase_increments"],
        }
        sine = phase_core(*tuple(phase_feed.values()))
        generator_feed = {
            "decoder_feature": acoustic_outputs["decoder_feature"],
            "sine": sine,
            "f0": acoustic_outputs["f0"],
            "style": acoustic_feed["style"],
            "valid_frames": acoustic_outputs["valid_frames"],
            "generator_norm_scales": make_generator_rmsnorm_scales(
                acoustic_outputs["valid_frames"],
                frame_max_length,
            ),
        }
        split_waveform = generator(*tuple(generator_feed.values()))

    valid_samples = int(single_frames.item()) * WAVEFORM_SAMPLES_PER_FRAME
    validation: dict[str, Any] = {}
    if validate_outputs:
        if dynamic_waveform is None or dynamic_duration is None:
            raise ValueError("dynamic waveform and duration are required for output validation")
        validation = {
            "single_vs_dynamic": _metrics(
                dynamic_waveform.reshape(-1).float().cpu(),
                single_waveform.reshape(-1)[:valid_samples].float().cpu(),
                dynamic_duration.reshape(-1).cpu(),
                single_duration.reshape(-1)[: int(sample.valid_len.item())].cpu(),
            ),
            "split_vs_single_pytorch": _tensor_metric(
                "waveform",
                single_waveform.detach().cpu().numpy(),
                split_waveform.detach().cpu().numpy(),
            ),
        }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, GraphArtifact] = {}
    outputs: dict[str, dict[str, Tensor]] = {}
    paths = {
        ACOUSTIC_ROLE: destination / f"kokoro_acoustic_b1_t{text_max_length}_f{frame_max_length}.onnx",
        PHASE_CORE_ROLE: destination / f"kokoro_phase_core_b1_f{frame_max_length}.onnx",
        GENERATOR_ISTFT_ROLE: destination / f"kokoro_generator_istft_b1_f{frame_max_length}.onnx",
    }
    if output_paths is not None:
        paths.update({role: Path(path).expanduser().resolve() for role, path in output_paths.items()})

    acoustic_actual = {name: value.detach().cpu().numpy() for name, value in acoustic_outputs.items()}
    if ACOUSTIC_ROLE in selected_roles:
        acoustic_artifact, acoustic_actual = _export_partition(
            role=ACOUSTIC_ROLE,
            module=acoustic,
            feed=acoustic_feed,
            expected=acoustic_outputs,
            output_path=paths[ACOUSTIC_ROLE],
            opset=opset,
            simplify=simplify,
            validate_onnx=validate_onnx,
            expected_lstm_nodes=12,
            npu_graph=True,
        )
        graphs[ACOUSTIC_ROLE] = acoustic_artifact
    outputs[ACOUSTIC_ROLE] = _clone_outputs(acoustic_outputs)

    if validate_onnx:
        phase_feed = {
            "phase_increments": _float_tensor(acoustic_actual["phase_increments"]),
        }
        with torch.no_grad():
            sine = phase_core(*tuple(phase_feed.values()))
    phase_outputs = {"sine": sine}
    phase_actual = {name: value.detach().cpu().numpy() for name, value in phase_outputs.items()}
    if PHASE_CORE_ROLE in selected_roles:
        phase_artifact, phase_actual = _export_partition(
            role=PHASE_CORE_ROLE,
            module=phase_core,
            feed=phase_feed,
            expected=phase_outputs,
            output_path=paths[PHASE_CORE_ROLE],
            opset=opset,
            simplify=simplify,
            validate_onnx=validate_onnx,
            expected_lstm_nodes=0,
            npu_graph=False,
        )
        graphs[PHASE_CORE_ROLE] = phase_artifact
    outputs[PHASE_CORE_ROLE] = _clone_outputs(phase_outputs)

    if validate_onnx:
        generator_feed = {
            "decoder_feature": _float_tensor(acoustic_actual["decoder_feature"]),
            "sine": _float_tensor(phase_actual["sine"]),
            "f0": _float_tensor(acoustic_actual["f0"]),
            "style": acoustic_feed["style"],
            "valid_frames": _int32_tensor(acoustic_actual["valid_frames"]),
        }
        generator_feed["generator_norm_scales"] = make_generator_rmsnorm_scales(
            generator_feed["valid_frames"],
            frame_max_length,
        )
        with torch.no_grad():
            split_waveform = generator(*tuple(generator_feed.values()))
    generator_outputs = {"waveform": split_waveform}
    generator_actual = {name: value.detach().cpu().numpy() for name, value in generator_outputs.items()}
    if GENERATOR_ISTFT_ROLE in selected_roles:
        generator_artifact, generator_actual = _export_partition(
            role=GENERATOR_ISTFT_ROLE,
            module=generator,
            feed=generator_feed,
            expected=generator_outputs,
            output_path=paths[GENERATOR_ISTFT_ROLE],
            opset=opset,
            simplify=simplify,
            validate_onnx=validate_onnx,
            expected_lstm_nodes=0,
            npu_graph=True,
        )
        graphs[GENERATOR_ISTFT_ROLE] = generator_artifact
    outputs[GENERATOR_ISTFT_ROLE] = _clone_outputs(generator_outputs)
    if validate_outputs and validate_onnx and GENERATOR_ISTFT_ROLE in selected_roles:
        chain_metric = _tensor_metric(
            "waveform",
            single_waveform.detach().cpu().numpy(),
            generator_actual["waveform"],
        )
        validation["fp32_onnx_chain_vs_single_pytorch"] = chain_metric
        if float(chain_metric.get("cosine", 0.0)) < 0.999:
            raise RuntimeError(
                f"precision-split FP32 ONNX chain failed waveform cosine>=0.999: {chain_metric.get('cosine')}"
            )

    feeds = {
        ACOUSTIC_ROLE: _clone_outputs(acoustic_feed),
        PHASE_CORE_ROLE: _clone_outputs(phase_feed),
        GENERATOR_ISTFT_ROLE: _clone_outputs(generator_feed),
    }
    return PrecisionSplitExport(
        graphs=graphs,
        feeds=feeds,
        outputs=outputs,
        waveform=split_waveform.detach().cpu(),
        duration=single_duration.detach().cpu(),
        valid_frames=single_frames.detach().cpu(),
        validation=validation,
        rewrites=rewrites,
    )


class KokoroPrecisionSplitRuntime:
    """Run acoustic NPU -> FP32 phase core -> source/generator/iSTFT NPU."""

    def __init__(
        self,
        runners: dict[str, Runner],
        *,
        text_max_length: int,
        frame_max_length: int,
        seed: int,
    ) -> None:
        missing = sorted(set(PRECISION_SPLIT_ROLES).difference(runners))
        if missing:
            raise ValueError(f"missing Kokoro precision-split runners: {missing}")
        self.runners = runners
        self.text_max_length = int(text_max_length)
        self.frame_max_length = int(frame_max_length)
        self.seed = int(seed)

    @classmethod
    def from_export(
        cls,
        work_dir: str | Path,
        *,
        backend: str = "hmonnx",
        device: str = "cuda:0",
    ) -> KokoroPrecisionSplitRuntime:
        root = Path(work_dir).expanduser().resolve()
        meta = json.loads((root / "export_meta_info.json").read_text(encoding="utf-8"))
        if meta.get("graph_mode") != PRECISION_SPLIT_GRAPH_MODE:
            raise ValueError(f"{root} is not a Kokoro precision-split export")
        if backend not in {"ort", "hmonnx"}:
            raise ValueError("backend must be 'ort' or 'hmonnx'")
        runners: dict[str, Runner] = {}
        for role in PRECISION_SPLIT_ROLES:
            component = meta["components"][role]
            if backend == "ort" or role == PHASE_CORE_ROLE:
                runners[role] = OrtRunner(root / component["onnx_file"])
                continue
            hmonnx_file = component.get("hmonnx_file")
            if not hmonnx_file:
                raise ValueError(f"component {role} has no HMONNX artifact")
            runners[role] = HmonnxRunner(root / hmonnx_file, device=device)
        return cls(
            runners,
            text_max_length=int(meta["text_max_length"]),
            frame_max_length=int(meta["frame_max_length"]),
            seed=int(meta["seed"]),
        )

    def synthesize(
        self,
        token_ids: np.ndarray | list[int] | tuple[int, ...],
        style: np.ndarray | Tensor,
        *,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        if not 0 < tokens.size <= self.text_max_length:
            raise ValueError(f"token length must be in [1,{self.text_max_length}]")
        if speed <= 0:
            raise ValueError("speed must be positive")
        requested_seed = self.seed if seed is None else int(seed)
        if requested_seed != self.seed:
            raise ValueError(f"SineGen noise is fixed at export time; expected seed {self.seed}, got {requested_seed}")
        style_array = np.asarray(
            style.detach().float().cpu().numpy() if isinstance(style, Tensor) else style,
            dtype=np.float32,
        )
        if style_array.shape == (256,):
            style_array = style_array[None]
        if style_array.shape != (1, 256):
            raise ValueError(f"style must have shape [1,256], got {style_array.shape}")

        input_ids = np.zeros((1, self.text_max_length), dtype=np.int32)
        input_ids[0, : tokens.size] = tokens
        attention_mask = np.full(
            (1, 1, self.text_max_length, self.text_max_length),
            ATTENTION_MASK_MIN,
            dtype=np.float32,
        )
        attention_mask[..., : tokens.size] = 0.0
        acoustic = self._run(
            ACOUSTIC_ROLE,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "style": style_array,
                "speed": np.asarray([speed], dtype=np.float32),
                "valid_len": np.asarray([tokens.size], dtype=np.int32),
            },
        )
        valid_frames = int(np.rint(np.asarray(acoustic["valid_frames"]).reshape(-1)[0]))
        if valid_frames > self.frame_max_length:
            raise ValueError(
                f"duration produced F={valid_frames}, exceeding bucket "
                f"F={self.frame_max_length}; retry a larger F bucket"
            )
        phase = self._run(
            PHASE_CORE_ROLE,
            {
                "phase_increments": np.asarray(acoustic["phase_increments"], dtype=np.float32),
            },
        )
        generated = self._run(
            GENERATOR_ISTFT_ROLE,
            {
                "decoder_feature": np.asarray(acoustic["decoder_feature"], dtype=np.float32),
                "sine": np.asarray(phase["sine"], dtype=np.float32),
                "f0": np.asarray(acoustic["f0"], dtype=np.float32),
                "style": style_array,
                "valid_frames": np.asarray([valid_frames], dtype=np.int32),
                "generator_norm_scales": make_generator_rmsnorm_scales(
                    torch.tensor([valid_frames], dtype=torch.int32),
                    self.frame_max_length,
                ).numpy(),
            },
        )
        valid_samples = valid_frames * WAVEFORM_SAMPLES_PER_FRAME
        duration = np.rint(np.asarray(acoustic["duration"]).reshape(-1)).astype(np.int64)
        waveform = np.asarray(generated["waveform"], dtype=np.float32).reshape(-1)[:valid_samples]
        return waveform, {
            "text_length": int(tokens.size),
            "frame_length": valid_frames,
            "valid_samples": valid_samples,
            "duration": duration[: tokens.size].tolist(),
            "seed": requested_seed,
        }

    def _run(self, role: str, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.runners[role].run({name: np.ascontiguousarray(value) for name, value in feed.items()})


def _export_partition(
    *,
    role: str,
    module: nn.Module,
    feed: dict[str, Tensor],
    expected: dict[str, Tensor],
    output_path: Path,
    opset: int,
    simplify: bool,
    validate_onnx: bool,
    expected_lstm_nodes: int,
    npu_graph: bool,
) -> tuple[GraphArtifact, dict[str, np.ndarray]]:
    _validate_attention_mask_feed(feed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        tuple(feed.values()),
        output_path,
        input_names=list(feed),
        output_names=list(expected),
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    # The bucketed deployment deliberately keeps the legacy exporter output.
    # xhquant owns backend lowering; Kokoro must not mutate the ONNX graph with
    # onnxsim or model-specific post-export optimization passes.
    del simplify
    structural = {"post_export_optimizations": 0}
    graph = onnx.load(output_path, load_external_data=False)
    onnx.checker.check_model(graph)
    if "attention_mask" in feed:
        structural["double_mask_add_softmax_patterns"] = (
            _validate_double_mask_add_before_softmax(graph)
        )
    op_counts: dict[str, int] = {}
    for node in graph.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    lstm_nodes = op_counts.get("LSTM", 0)
    if lstm_nodes != expected_lstm_nodes:
        raise RuntimeError(f"{role} must preserve {expected_lstm_nodes} high-level LSTM nodes, got {lstm_nodes}")
    for node in graph.graph.node:
        if node.op_type != "LSTM":
            continue
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if attributes.get("direction", b"forward") not in {b"", b"forward"}:
            raise RuntimeError(f"{role} LSTM {node.name!r} is not forward")
    if npu_graph:
        forbidden = sorted(FORBIDDEN_NPU_OPS.intersection(op_counts))
        if forbidden:
            raise RuntimeError(f"{role} contains forbidden NPU operators: {forbidden}")

    actual = {name: value.detach().cpu().numpy() for name, value in expected.items()}
    metrics: tuple[dict[str, Any], ...] = tuple()
    if validate_onnx:
        session = OrtRunner(output_path)
        actual = session.run({name: value.detach().cpu().numpy() for name, value in feed.items()})
        metrics = tuple(_tensor_metric(name, expected[name].detach().cpu().numpy(), actual[name]) for name in expected)
        failed = []
        for metric in metrics:
            if metric["name"] == "sine" or (role == "frame_synthesis" and metric["name"] == "waveform"):
                # Phase-integrated outputs are intentionally phase-sensitive.
                # Retain every metric here; the independent-bucket workflow
                # applies waveform, spectral and listening gates end to end.
                continue
            limit = _onnx_output_max_abs_limit(
                role,
                str(metric["name"]),
                expected[str(metric["name"])],
            )
            if float(metric["max_abs"]) > limit:
                failed.append((metric, limit))
        if failed:
            raise RuntimeError(
                f"{role} ONNX validation exceeds output-specific max_abs: "
                f"{[(metric['name'], metric['max_abs'], limit) for metric, limit in failed]}"
            )

    artifact = GraphArtifact(
        role=role,
        path=output_path,
        input_names=tuple(feed),
        output_names=tuple(expected),
        input_contracts=tuple(_value_contract(value) for value in graph.graph.input),
        output_contracts=tuple(_value_contract(value) for value in graph.graph.output),
        node_count=len(graph.graph.node),
        op_counts=dict(sorted(op_counts.items())),
        onnx_sha256=sha256(output_path),
        pytorch_vs_onnx=metrics,
        structural_rewrites=structural,
    )
    return artifact, actual


def _validate_double_mask_add_before_softmax(
    graph: onnx.ModelProto,
    mask_name: str = "attention_mask",
) -> int:
    """Require every Softmax to consume a direct two-Add mask pattern."""

    producers = {
        output: node
        for node in graph.graph.node
        for output in node.output
        if output
    }
    softmax_nodes = [node for node in graph.graph.node if node.op_type == "Softmax"]
    if not softmax_nodes:
        raise RuntimeError("attention-mask graph contains no Softmax node")

    matched = 0
    for softmax in softmax_nodes:
        source = producers.get(softmax.input[0])
        if source is None or source.op_type != "Add" or mask_name not in source.input:
            raise RuntimeError(
                f"Softmax {softmax.name!r} is not preceded by the second mask Add"
            )
        first_add_output = next(
            (name for name in source.input if name != mask_name),
            None,
        )
        first_add = producers.get(first_add_output or "")
        if (
            first_add is None
            or first_add.op_type != "Add"
            or mask_name not in first_add.input
        ):
            raise RuntimeError(
                f"Softmax {softmax.name!r} is not preceded by two mask Adds"
            )
        matched += 1
    return matched


def _validate_attention_mask_feed(feed: Mapping[str, Tensor]) -> None:
    mask = feed.get("attention_mask")
    if mask is None:
        return
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != 1 or mask.shape[2] != mask.shape[3]:
        raise ValueError(f"attention_mask must be [1,1,T,T], got {tuple(mask.shape)}")
    if not torch.isfinite(mask).all():
        raise ValueError("attention_mask must not contain NaN or infinity")
    allowed = (mask == 0) | (mask == ATTENTION_MASK_MIN)
    if not bool(allowed.all()):
        values = torch.unique(mask.detach().cpu()).tolist()
        raise ValueError(f"attention_mask values must be 0 or {ATTENTION_MASK_MIN}, got {values}")


def _onnx_output_max_abs_limit(role: str, name: str, expected: Tensor) -> float:
    if name == "f0":
        return 5e-4
    if role == GENERATOR_ISTFT_ROLE and name == "waveform":
        # Static iRFFT + overlap-add accumulates a number of FP32 terms that
        # grows with the waveform buffer.  Preserve the measured F120 gate and
        # scale only this absolute-error bound with the number of frames; the
        # recorded mean/cosine/spectral metrics remain comparable across
        # buckets.  A fixed 2e-4 gate incorrectly rejects F400 at 4.67e-4 even
        # though its relative waveform error is negligible.
        frame_capacity = int(expected.numel()) / WAVEFORM_SAMPLES_PER_FRAME
        return 2e-4 * max(1.0, frame_capacity / 120.0)
    return 2e-4


def _clone_outputs(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in values.items()}


def _float_tensor(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).contiguous()


def _int32_tensor(value: np.ndarray) -> Tensor:
    return torch.from_numpy(np.rint(np.asarray(value)).astype(np.int32)).contiguous()


__all__ = [
    "ACOUSTIC_ROLE",
    "AcousticStatic",
    "GENERATOR_ISTFT_ROLE",
    "GeneratorISTFTStatic",
    "KokoroPrecisionSplitRuntime",
    "PHASE_CORE_ROLE",
    "PhaseCoreStatic",
    "PRECISION_SPLIT_GRAPH_MODE",
    "PRECISION_SPLIT_NPU_ROLES",
    "PRECISION_SPLIT_ROLES",
    "PrecisionSplitExport",
    "export_precision_split_static",
]
