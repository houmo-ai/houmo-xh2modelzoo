from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .assets import resolve_model_assets, sha256, verify_release_assets
from .bucketed_runtime import BUCKETED_PRECISION_SPLIT_GRAPH_MODE, KokoroBucketedRuntime
from .buckets import (
    FRAME_BUCKETS,
    LSTM_VARIANTS,
    TOKEN_BUCKETS,
    audio_seconds_for_frame,
    frame_bucket_key,
    normalize_bucket_routes,
    token_bucket_key,
)
from .graph import (
    GRAPH_ROLES,
    GraphArtifact,
    artifact_metadata,
    export_static_graphs,
    load_official_model,
)
from .host import (
    ATTENTION_MASK_MIN,
    DEFAULT_INPUT_IDS,
    SAMPLE_RATE,
    default_static_sample,
    load_voice_style,
    save_voice_pack_numpy,
)
from .independent_split import (
    FRAME_ACOUSTIC_ROLE,
    FRAME_SYNTHESIS_ROLE,
    TEXT_DURATION_ROLE,
    export_frame_bucket,
    export_text_bucket,
    prepare_independent_model,
    run_duration_alignment_host,
)
from .independent_split import (
    GENERATOR_ISTFT_ROLE as INDEPENDENT_GENERATOR_ISTFT_ROLE,
)
from .independent_split import (
    PHASE_CORE_ROLE as INDEPENDENT_PHASE_CORE_ROLE,
)
from .precision_split import (
    PHASE_CORE_ROLE,
    PRECISION_SPLIT_GRAPH_MODE,
    PRECISION_SPLIT_NPU_ROLES,
    PRECISION_SPLIT_ROLES,
    KokoroPrecisionSplitRuntime,
    export_precision_split_static,
)
from .runtime import KokoroStaticRuntime, OrtRunner, Runner
from .single_graph import SINGLE_GRAPH_ROLE, export_end_to_end_static


XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV = "XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE"


class KokoroWorkflow(BaseOtherModelWorkflow):
    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise NotImplementedError("Kokoro performs ONNX PTQ during HMONNX export; quant must be null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        import torch

        config = self.workflow_config.with_overrides(config_overrides)
        model_dir = Path(self._resolve_export_model_dir(quant_result))
        export_cfg = config.build_export_dict()
        _validate_export_config(export_cfg)
        assets = resolve_model_assets(model_dir)
        asset_identity = verify_release_assets(assets)

        target = str(export_cfg["target_device"])
        graph_mode = str(export_cfg.get("graph_mode", "multi_graph"))
        text_max_length = int(export_cfg.get("text_max_length", TOKEN_BUCKETS[0]))
        frame_max_length = int(export_cfg.get("frame_max_length", FRAME_BUCKETS[0]))
        lstm_chunk_length = int(export_cfg.get("lstm_chunk_length", frame_max_length))
        opset = int(export_cfg.get("opset", 17))
        seed = int(export_cfg.get("seed", self.seed))
        simplify = bool(export_cfg.get("simplify", True))
        convert_hmonnx = bool(export_cfg.get("convert_hmonnx", True))
        torch_onnx_internal_optimize = bool(
            export_cfg.get("torch_onnx_internal_optimize", True)
        )
        decompose_lstm = bool(export_cfg.get("decompose_lstm", graph_mode == "multi_graph"))
        f0_norm_mode = str(export_cfg.get("f0_norm_mode", "adain"))

        work_dir = Path(output_dir).expanduser().resolve()
        onnx_dir = work_dir / "onnx"
        hmonnx_dir = work_dir / "hmonnx"
        work_dir.mkdir(parents=True, exist_ok=True)
        onnx_dir.mkdir(parents=True, exist_ok=True)
        if convert_hmonnx:
            hmonnx_dir.mkdir(parents=True, exist_ok=True)
        config_file = config.dump(str(work_dir / f"{config.name}.yaml"))
        runtime_assets = _export_runtime_assets(assets.voice, work_dir)

        model = load_official_model(assets)
        if graph_mode == BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
            return self._export_paired_bucketed_precision_split(
                model=model,
                assets=assets,
                asset_identity=asset_identity,
                runtime_assets=runtime_assets,
                export_cfg=export_cfg,
                work_dir=work_dir,
                onnx_dir=onnx_dir,
                hmonnx_dir=hmonnx_dir,
                config_file=config_file,
                target=target,
                opset=opset,
                seed=seed,
                simplify=simplify,
                convert_hmonnx=convert_hmonnx,
                torch_onnx_internal_optimize=torch_onnx_internal_optimize,
            )
        sample = default_static_sample(assets.voice, text_max_length)
        if graph_mode == PRECISION_SPLIT_GRAPH_MODE:
            return self._export_precision_split(
                model=model,
                sample=sample,
                assets=assets,
                asset_identity=asset_identity,
                runtime_assets=runtime_assets,
                export_cfg=export_cfg,
                work_dir=work_dir,
                onnx_dir=onnx_dir,
                hmonnx_dir=hmonnx_dir,
                config_file=config_file,
                target=target,
                text_max_length=text_max_length,
                frame_max_length=frame_max_length,
                opset=opset,
                seed=seed,
                simplify=simplify,
                convert_hmonnx=convert_hmonnx,
                decompose_lstm=decompose_lstm,
                torch_onnx_internal_optimize=torch_onnx_internal_optimize,
            )
        if graph_mode == "single_graph":
            return self._export_single_graph(
                model=model,
                sample=sample,
                assets=assets,
                asset_identity=asset_identity,
                runtime_assets=runtime_assets,
                export_cfg=export_cfg,
                work_dir=work_dir,
                onnx_dir=onnx_dir,
                hmonnx_dir=hmonnx_dir,
                config_file=config_file,
                target=target,
                text_max_length=text_max_length,
                frame_max_length=frame_max_length,
                opset=opset,
                seed=seed,
                simplify=simplify,
                convert_hmonnx=convert_hmonnx,
                decompose_lstm=decompose_lstm,
                torch_onnx_internal_optimize=torch_onnx_internal_optimize,
            )
        bundle = export_static_graphs(
            model=model,
            sample=sample,
            output_dir=onnx_dir,
            text_max_length=text_max_length,
            frame_max_length=frame_max_length,
            lstm_chunk_length=lstm_chunk_length,
            opset=opset,
            seed=seed,
            simplify=simplify,
            validate_onnx=False,
            validate_outputs=False,
            f0_norm_mode=f0_norm_mode,
        )

        components: dict[str, dict[str, Any]] = {}
        component_cfg = export_cfg["components"]
        for role in GRAPH_ROLES:
            artifact = bundle.graphs[role]
            quant_type = str(component_cfg[role]["quant_type"])
            component = artifact_metadata(artifact, work_dir)
            component["quant_type"] = quant_type
            if convert_hmonnx:
                hmonnx_path = hmonnx_dir / (f"{artifact.path.stem}_{target}_{quant_type}.onnx")
                _convert_hmonnx(
                    artifact,
                    bundle.feeds[role],
                    hmonnx_path,
                    target=target,
                    quant_type=quant_type,
                    debug=self.debug,
                    decompose_lstm=decompose_lstm,
                    torch_onnx_internal_optimize=torch_onnx_internal_optimize,
                )
                component["hmonnx_file"] = str(hmonnx_path.relative_to(work_dir))
                component["hmonnx_sha256"] = sha256(hmonnx_path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            components[role] = component

        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHKokoroModel",
            "jira": ["HMSW-4679", "CMS-706"],
            "source_project": "https://github.com/hexgrad/kokoro",
            "source_model": "hexgrad/Kokoro-82M-v1.1-zh",
            "asset_identity": asset_identity,
            "source_paths": {
                "source_root": str(assets.source_root),
                "config": str(assets.config),
                "checkpoint": str(assets.checkpoint),
                "voice": str(assets.voice),
            },
            "runtime_assets": runtime_assets,
            "torch_onnx_internal_optimize": torch_onnx_internal_optimize,
            "target_device": target,
            "workflow_config": str(Path(config_file).relative_to(work_dir)),
            "sample_rate": SAMPLE_RATE,
            "samples_per_frame": 600,
            "text_max_length": text_max_length,
            "frame_max_length": frame_max_length,
            "lstm_chunk_length": lstm_chunk_length,
            "seed": seed,
            "f0_norm_mode": f0_norm_mode,
            "components": components,
            "component_order": list(GRAPH_ROLES),
            "host_stages": [
                "G2P/tokenization and static bucket padding",
                "valid-prefix reversal and masks",
                "duration reduction and alignment expansion",
                "shared LSTM chunk/state scheduling",
                "deterministic SineGen randomness",
                "STFT and iSTFT",
            ],
            "graph_rewrites": bundle.rewrites,
            "reference_sample": {
                "text": sample.text,
                "phonemes": sample.phonemes,
                "voice": sample.voice,
                "input_ids": list(DEFAULT_INPUT_IDS),
                "duration": bundle.duration[0, : int(sample.valid_len.item())].tolist(),
                "frames": int(bundle.valid_frames.item()),
                "samples": int(bundle.waveform.numel()),
            },
            "hmonnx_converted": convert_hmonnx,
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def _export_paired_bucketed_precision_split(
        self,
        *,
        model: Any,
        assets: Any,
        asset_identity: Mapping[str, Any],
        runtime_assets: Mapping[str, Any],
        export_cfg: Mapping[str, Any],
        work_dir: Path,
        onnx_dir: Path,
        hmonnx_dir: Path,
        config_file: str,
        target: str,
        opset: int,
        seed: int,
        simplify: bool,
        convert_hmonnx: bool,
        torch_onnx_internal_optimize: bool,
    ) -> ExportResult:
        """Export the four approved paired T/F capacity gears.

        Duration reduction and alignment expansion intentionally stay between
        the two families on Host.  No exported graph carries both a T and an F
        capacity, so each graph is still emitted once per unique capacity.
        """

        import torch

        del simplify
        routes = normalize_bucket_routes(
            export_cfg.get("token_buckets"),
            export_cfg.get("audio_seconds_buckets"),
        )
        token_buckets = tuple(route.token_max_length for route in routes)
        audio_seconds_buckets = tuple(route.audio_seconds for route in routes)
        frame_buckets = tuple(route.frame_max_length for route in routes)
        variants = tuple(str(value) for value in export_cfg.get("lstm_variants", LSTM_VARIANTS))
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("export.lstm_variants must contain unique values")
        unsupported_variants = sorted(set(variants).difference(LSTM_VARIANTS))
        if unsupported_variants:
            raise ValueError(f"unsupported LSTM variants: {unsupported_variants}")

        phase_on_npu = bool(export_cfg.get("phase_on_npu", False))
        stft_pad_mode = str(export_cfg.get("stft_pad_mode", "length_aware_reflect"))
        stft_phase_mode = str(export_cfg.get("stft_phase_mode", "cordic"))
        f0_norm_mode = str(export_cfg.get("f0_norm_mode", "adain"))
        allow_decomposed_failure = bool(export_cfg.get("allow_decomposed_failure", True))
        component_cfg = export_cfg["components"]
        frame_lstm_role = FRAME_SYNTHESIS_ROLE if phase_on_npu else FRAME_ACOUSTIC_ROLE
        component_order = (
            (TEXT_DURATION_ROLE, FRAME_SYNTHESIS_ROLE)
            if phase_on_npu
            else (
                TEXT_DURATION_ROLE,
                FRAME_ACOUSTIC_ROLE,
                INDEPENDENT_PHASE_CORE_ROLE,
                INDEPENDENT_GENERATOR_ISTFT_ROLE,
            )
        )
        npu_component_order = tuple(
            role for role in component_order if role != INDEPENDENT_PHASE_CORE_ROLE
        )

        source_rewrites = prepare_independent_model(model)

        components: dict[str, dict[str, Any]] = {}
        for role in component_order:
            if role == INDEPENDENT_PHASE_CORE_ROLE:
                components[role] = {
                    "execution": "host_onnxruntime_fp32",
                    "precision": "fp32",
                    "reason": "CumSum, phase interpolation, and Sin precision fallback",
                    "buckets": {},
                }
            else:
                components[role] = {
                    "execution": "npu",
                    "quant_type": str(component_cfg[role]["quant_type"]),
                    "buckets": {},
                }

        decomposed_failures: list[dict[str, Any]] = []
        text_reference: Any | None = None
        for token_max_length in token_buckets:
            key = token_bucket_key(token_max_length)
            sample = default_static_sample(assets.voice, token_max_length)
            exported = export_text_bucket(
                model=model,
                sample=sample,
                output_path=(
                    onnx_dir
                    / TEXT_DURATION_ROLE
                    / key
                    / f"kokoro_text_duration_b1_t{token_max_length}.onnx"
                ),
                text_max_length=token_max_length,
                opset=opset,
                validate_onnx=False,
            )
            entry = artifact_metadata(exported.artifact, work_dir)
            entry.update(
                {
                    "token_max_length": token_max_length,
                    "hmonnx_variants": _convert_lstm_bucket_variants(
                        artifact=exported.artifact,
                        feed=exported.feed,
                        hmonnx_dir=hmonnx_dir,
                        role=TEXT_DURATION_ROLE,
                        bucket_key=key,
                        target=target,
                        quant_type=str(component_cfg[TEXT_DURATION_ROLE]["quant_type"]),
                        variants=variants,
                        expected_native_lstm_nodes=10,
                        convert_hmonnx=convert_hmonnx,
                        allow_decomposed_failure=allow_decomposed_failure,
                        torch_onnx_internal_optimize=torch_onnx_internal_optimize,
                        debug=self.debug,
                        work_dir=work_dir,
                        failures=decomposed_failures,
                    ),
                }
            )
            components[TEXT_DURATION_ROLE]["buckets"][key] = entry
            if text_reference is None:
                text_reference = (sample, exported)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if text_reference is None:
            raise RuntimeError("Kokoro text bucket export produced no graph")
        reference_sample, reference_text = text_reference
        reference_duration: Any | None = None
        reference_valid_frames: int | None = None
        reference_waveform: Any | None = None

        for frame_max_length in frame_buckets:
            key = frame_bucket_key(frame_max_length)
            seconds = audio_seconds_for_frame(frame_max_length)
            frame_inputs, duration = run_duration_alignment_host(
                reference_text.outputs,
                speed=reference_sample.speed,
                valid_len=reference_sample.valid_len,
                frame_max_length=frame_max_length,
            )
            valid_frames = int(frame_inputs["valid_frames"].item())
            if valid_frames > frame_max_length:
                raise RuntimeError(
                    f"reference sample requires F={valid_frames}, exceeding export bucket F={frame_max_length}"
                )
            exported = export_frame_bucket(
                model=model,
                frame_inputs=frame_inputs,
                style=reference_sample.style,
                output_dir=onnx_dir,
                frame_max_length=frame_max_length,
                seed=seed,
                stft_pad_mode=stft_pad_mode,
                stft_phase_mode=stft_phase_mode,
                f0_norm_mode=f0_norm_mode,
                opset=opset,
                validate_onnx=False,
                phase_on_npu=phase_on_npu,
            )
            for role, artifact in exported.graphs.items():
                entry = artifact_metadata(artifact, work_dir)
                entry.update(
                    {
                        "frame_max_length": frame_max_length,
                        "audio_seconds": seconds,
                    }
                )
                if role == frame_lstm_role:
                    entry["hmonnx_variants"] = _convert_lstm_bucket_variants(
                        artifact=artifact,
                        feed=exported.feeds[role],
                        hmonnx_dir=hmonnx_dir,
                        role=role,
                        bucket_key=key,
                        target=target,
                        quant_type=str(component_cfg[role]["quant_type"]),
                        variants=variants,
                        expected_native_lstm_nodes=2,
                        convert_hmonnx=convert_hmonnx,
                        allow_decomposed_failure=allow_decomposed_failure,
                        torch_onnx_internal_optimize=torch_onnx_internal_optimize,
                        debug=self.debug,
                        work_dir=work_dir,
                        failures=decomposed_failures,
                    )
                elif role != INDEPENDENT_PHASE_CORE_ROLE and convert_hmonnx:
                    quant_type = str(component_cfg[role]["quant_type"])
                    hmonnx_path = _bucketed_non_lstm_hmonnx_path(
                        hmonnx_dir,
                        role=role,
                        bucket_key=key,
                        target=target,
                        quant_type=quant_type,
                    )
                    _convert_hmonnx(
                        artifact,
                        exported.feeds[role],
                        hmonnx_path,
                        target=target,
                        quant_type=quant_type,
                        debug=self.debug,
                        decompose_lstm=False,
                        torch_onnx_internal_optimize=torch_onnx_internal_optimize,
                    )
                    entry.update(_hmonnx_artifact_metadata(hmonnx_path, work_dir))
                components[role]["buckets"][key] = entry
            if reference_duration is None:
                reference_duration = duration
                reference_valid_frames = valid_frames
                reference_waveform = exported.waveform
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if reference_duration is None or reference_valid_frames is None or reference_waveform is None:
            raise RuntimeError("Kokoro frame bucket export produced no graph")
        reference_samples = reference_valid_frames * 600
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHKokoroModel",
            "jira": ["HMSW-4679", "CMS-706"],
            "source_project": "https://github.com/hexgrad/kokoro",
            "source_model": "hexgrad/Kokoro-82M-v1.1-zh",
            "asset_identity": asset_identity,
            "source_paths": {
                "source_root": str(assets.source_root),
                "config": str(assets.config),
                "checkpoint": str(assets.checkpoint),
                "voice": str(assets.voice),
            },
            "runtime_assets": dict(runtime_assets),
            "torch_onnx_internal_optimize": torch_onnx_internal_optimize,
            "target_device": target,
            "workflow_config": str(Path(config_file).relative_to(work_dir)),
            "graph_mode": BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
            "primary_deployment": True,
            "sample_rate": SAMPLE_RATE,
            "samples_per_frame": 600,
            "seed": seed,
            "stft_pad_mode": stft_pad_mode,
            "stft_phase_mode": stft_phase_mode,
            "f0_norm_mode": f0_norm_mode,
            "bucket_presets": {
                "policy": "duration_driven_paired_t_f",
                "routes": [route.as_dict() for route in routes],
                "token": list(token_buckets),
                "audio_seconds": list(audio_seconds_buckets),
                "frame": list(frame_buckets),
                "frame_to_audio_seconds": {
                    str(frame): audio_seconds_for_frame(frame) for frame in frame_buckets
                },
                "export_count_per_lstm_variant": len(token_buckets) + len(frame_buckets),
                "cartesian_product_exported": False,
            },
            "components": components,
            "component_order": list(component_order),
            "npu_component_order": list(npu_component_order),
            "phase_boundary": {
                "mode": "npu" if phase_on_npu else "host_fp32",
                "preferred_mode": "host_fp32",
                "npu_graph_count": 2 if phase_on_npu else 3,
                "host_phase_stage": None if phase_on_npu else "CumSum -> phase interpolation -> Sin",
            },
            "mask_contract": {
                "input": "attention_mask",
                "shape": "[1,1,T,T]",
                "valid_key_value": 0.0,
                "invalid_key_value": ATTENTION_MASK_MIN,
                "contains_infinity": False,
                "padded_text_and_frame_features": 0.0,
            },
            "host_stages": [
                "ZHG2P/tokenization and voice-pack style lookup",
                "smallest token-fitting paired T/F route selection",
                "duration sigmoid/sum/speed/round and valid frame calculation",
                "retry the next paired route when duration exceeds its F capacity",
                "frame-to-token index construction and duration_features/asr Gather",
                "valid waveform prefix crop",
            ],
            "lstm_contract": {
                "onnx_direction": "forward",
                "sequence_lens": "omitted; all T steps are valid",
                "padded_bidirectional_lowering": (
                    "forward LSTM + Gather(prefix reversal) + forward LSTM + Gather(time restore)"
                ),
                "onnx_nodes": {TEXT_DURATION_ROLE: 10, frame_lstm_role: 2},
                "native_hmonnx": {
                    "operator": "ai.houmo.xh2a::LSTM",
                    "decompose_lstm": False,
                },
                "decomposed_hmonnx": {
                    "operator": "xhquant QLinear/Add/Mul/Sigmoid/Tanh/Split/Gather sequence",
                    "decompose_lstm": True,
                    "lowering_stage": "torch.export",
                    "weights": "quantized qweight + scale_or_exp",
                    "allow_per_bucket_failure": allow_decomposed_failure,
                },
                "selection_mechanism": "decompose_lstm API or XHQUANT_LSTM_EXPORT_MODE at torch.export",
            },
            "source_level_compatibility_lowerings": source_rewrites,
            "onnx_post_export_optimizations": 0,
            "xhquant_frontend_simplify": True,
            "decomposed_failures": decomposed_failures,
            "reference_sample": {
                "text": reference_sample.text,
                "phonemes": reference_sample.phonemes,
                "voice": reference_sample.voice,
                "input_ids": list(DEFAULT_INPUT_IDS),
                "duration": reference_duration[0, : int(reference_sample.valid_len.item())].tolist(),
                "frames": reference_valid_frames,
                "samples": reference_samples,
            },
            "hmonnx_converted": convert_hmonnx,
            "numerical_validation": "run examples_merak/audio/kokoro/compare_backends.py",
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def _export_precision_split(
        self,
        *,
        model: Any,
        sample: Any,
        assets: Any,
        asset_identity: Mapping[str, Any],
        runtime_assets: Mapping[str, Any],
        export_cfg: Mapping[str, Any],
        work_dir: Path,
        onnx_dir: Path,
        hmonnx_dir: Path,
        config_file: str,
        target: str,
        text_max_length: int,
        frame_max_length: int,
        opset: int,
        seed: int,
        simplify: bool,
        convert_hmonnx: bool,
        decompose_lstm: bool,
        torch_onnx_internal_optimize: bool,
    ) -> ExportResult:
        import torch

        stft_pad_mode = str(export_cfg.get("stft_pad_mode", "length_aware_reflect"))
        stft_phase_mode = str(export_cfg.get("stft_phase_mode", "cordic"))
        f0_norm_mode = str(export_cfg.get("f0_norm_mode", "adain"))
        bundle = export_precision_split_static(
            model=model,
            sample=sample,
            dynamic_waveform=None,
            dynamic_duration=None,
            output_dir=onnx_dir,
            text_max_length=text_max_length,
            frame_max_length=frame_max_length,
            seed=seed,
            stft_pad_mode=stft_pad_mode,
            stft_phase_mode=stft_phase_mode,
            opset=opset,
            simplify=simplify,
            validate_onnx=False,
            validate_outputs=False,
            f0_norm_mode=f0_norm_mode,
        )

        components: dict[str, dict[str, Any]] = {}
        component_cfg = export_cfg["components"]
        for role in PRECISION_SPLIT_ROLES:
            artifact = bundle.graphs[role]
            component = artifact_metadata(artifact, work_dir)
            if role == PHASE_CORE_ROLE:
                component.update(
                    {
                        "execution": "host_onnxruntime_fp32",
                        "precision": "fp32",
                        "reason": "Only SineGen CumSum, phase interpolation, and Sin must remain FP32",
                    }
                )
            else:
                quant_type = str(component_cfg[role]["quant_type"])
                component.update({"execution": "npu", "quant_type": quant_type})
                if convert_hmonnx:
                    hmonnx_path = hmonnx_dir / f"{artifact.path.stem}_{target}_{quant_type}.onnx"
                    _convert_hmonnx(
                        artifact,
                        bundle.feeds[role],
                        hmonnx_path,
                        target=target,
                        quant_type=quant_type,
                        debug=self.debug,
                        decompose_lstm=decompose_lstm,
                        torch_onnx_internal_optimize=torch_onnx_internal_optimize,
                    )
                    component["hmonnx_file"] = str(hmonnx_path.relative_to(work_dir))
                    component["hmonnx_sha256"] = sha256(hmonnx_path)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            components[role] = component

        valid_frames = int(bundle.valid_frames.item())
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHKokoroModel",
            "jira": ["HMSW-4679", "CMS-706"],
            "source_project": "https://github.com/hexgrad/kokoro",
            "source_model": "hexgrad/Kokoro-82M-v1.1-zh",
            "asset_identity": asset_identity,
            "source_paths": {
                "source_root": str(assets.source_root),
                "config": str(assets.config),
                "checkpoint": str(assets.checkpoint),
                "voice": str(assets.voice),
            },
            "runtime_assets": dict(runtime_assets),
            "torch_onnx_internal_optimize": torch_onnx_internal_optimize,
            "target_device": target,
            "workflow_config": str(Path(config_file).relative_to(work_dir)),
            "graph_mode": PRECISION_SPLIT_GRAPH_MODE,
            "primary_deployment": True,
            "sample_rate": SAMPLE_RATE,
            "samples_per_frame": 600,
            "text_max_length": text_max_length,
            "frame_max_length": frame_max_length,
            "seed": seed,
            "stft_pad_mode": stft_pad_mode,
            "stft_phase_mode": stft_phase_mode,
            "f0_norm_mode": f0_norm_mode,
            "components": components,
            "component_order": list(PRECISION_SPLIT_ROLES),
            "npu_component_order": list(PRECISION_SPLIT_NPU_ROLES),
            "host_stages": [
                "G2P/tokenization and T/F bucket selection",
                "voice-pack style lookup",
                "FP32 SineGen phase core: CumSum, phase interpolation, and Sin",
                "F-bucket overflow retry and output-prefix crop",
            ],
            "precision_boundary": {
                "acoustic_npu_to_host": ["phase_increments", "valid_frames"],
                "acoustic_npu_to_generator_npu": ["decoder_feature", "f0"],
                "host_to_generator_npu": ["sine"],
                "f0_execution": "npu_w16a16",
                "phase_increment_execution": "acoustic_npu_w16a16",
                "phase_core_execution": "host_fp32",
                "source_merge_execution": "generator_npu_w16a16",
                "harmonic_stft_execution": "generator_npu_w16a16",
            },
            "graph_rewrites": bundle.rewrites,
            "lstm_contract": {
                "acoustic_native_lstm_nodes": 12,
                "direction": "forward",
                "sequence_lens": "omitted; all T/F bucket steps are valid",
                "phase_core_cpu_lstm_nodes": 0,
                "generator_istft_lstm_nodes": 0,
                "time_step_unrolled_in_hmonnx": False,
            },
            "single_graph_status": {
                "production": False,
                "reason": "W16A16 phase accumulation in SineGen fails the waveform precision gate",
            },
            "reference_sample": {
                "text": sample.text,
                "phonemes": sample.phonemes,
                "voice": sample.voice,
                "input_ids": list(DEFAULT_INPUT_IDS),
                "duration": bundle.duration[0, : int(sample.valid_len.item())].tolist(),
                "frames": valid_frames,
                "samples": valid_frames * 600,
            },
            "hmonnx_converted": convert_hmonnx,
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def _export_single_graph(
        self,
        *,
        model: Any,
        sample: Any,
        assets: Any,
        asset_identity: Mapping[str, Any],
        runtime_assets: Mapping[str, Any],
        export_cfg: Mapping[str, Any],
        work_dir: Path,
        onnx_dir: Path,
        hmonnx_dir: Path,
        config_file: str,
        target: str,
        text_max_length: int,
        frame_max_length: int,
        opset: int,
        seed: int,
        simplify: bool,
        convert_hmonnx: bool,
        decompose_lstm: bool,
        torch_onnx_internal_optimize: bool,
    ) -> ExportResult:
        stft_pad_mode = str(export_cfg.get("stft_pad_mode", "length_aware_reflect"))
        stft_phase_mode = str(export_cfg.get("stft_phase_mode", "cordic"))
        f0_norm_mode = str(export_cfg.get("f0_norm_mode", "adain"))
        result = export_end_to_end_static(
            model=model,
            sample=sample,
            dynamic_waveform=None,
            dynamic_duration=None,
            output_path=onnx_dir / f"kokoro_end_to_end_b1_t{text_max_length}_f{frame_max_length}.onnx",
            text_max_length=text_max_length,
            frame_max_length=frame_max_length,
            seed=seed,
            stft_pad_mode=stft_pad_mode,
            stft_phase_mode=stft_phase_mode,
            opset=opset,
            simplify=simplify,
            validate_onnx=False,
            validate_outputs=False,
            f0_norm_mode=f0_norm_mode,
        )
        quant_type = str(export_cfg["components"][SINGLE_GRAPH_ROLE]["quant_type"])
        component = artifact_metadata(result.artifact, work_dir)
        component["quant_type"] = quant_type
        if convert_hmonnx:
            hmonnx_path = hmonnx_dir / (f"{result.artifact.path.stem}_{target}_{quant_type}.onnx")
            _convert_hmonnx(
                result.artifact,
                result.feed,
                hmonnx_path,
                target=target,
                quant_type=quant_type,
                debug=self.debug,
                decompose_lstm=decompose_lstm,
                torch_onnx_internal_optimize=torch_onnx_internal_optimize,
            )
            component["hmonnx_file"] = str(hmonnx_path.relative_to(work_dir))
            component["hmonnx_sha256"] = sha256(hmonnx_path)

        valid_frames = int(result.outputs["valid_frames"].item())
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "XHKokoroModel",
            "jira": ["HMSW-4679", "CMS-706"],
            "source_project": "https://github.com/hexgrad/kokoro",
            "source_model": "hexgrad/Kokoro-82M-v1.1-zh",
            "asset_identity": asset_identity,
            "source_paths": {
                "source_root": str(assets.source_root),
                "config": str(assets.config),
                "checkpoint": str(assets.checkpoint),
                "voice": str(assets.voice),
            },
            "runtime_assets": dict(runtime_assets),
            "torch_onnx_internal_optimize": torch_onnx_internal_optimize,
            "target_device": target,
            "workflow_config": str(Path(config_file).relative_to(work_dir)),
            "graph_mode": "single_graph",
            "sample_rate": SAMPLE_RATE,
            "samples_per_frame": 600,
            "text_max_length": text_max_length,
            "frame_max_length": frame_max_length,
            "seed": seed,
            "stft_pad_mode": stft_pad_mode,
            "stft_phase_mode": stft_phase_mode,
            "f0_norm_mode": f0_norm_mode,
            "components": {SINGLE_GRAPH_ROLE: component},
            "component_order": [SINGLE_GRAPH_ROLE],
            "host_stages": [
                "G2P/tokenization and T/F bucket selection",
                "voice-pack style lookup",
                "F-bucket overflow retry and output-prefix crop",
            ],
            "graph_rewrites": result.rewrites,
            "lstm_contract": {
                "logical_bidirectional_lstm": 6,
                "onnx_forward_lstm_nodes": 12,
                "valid_prefix": "explicit Gather prefix reversal outside standard forward LSTM",
                "time_step_unrolled_in_onnx": False,
                "native_hmonnx_lstm": convert_hmonnx and not decompose_lstm,
                "sequence_lens": "omitted; padding is handled by explicit Gather indices",
            },
            "reference_sample": {
                "text": sample.text,
                "phonemes": sample.phonemes,
                "voice": sample.voice,
                "input_ids": list(DEFAULT_INPUT_IDS),
                "duration": result.outputs["duration"][0, : int(sample.valid_len.item())].tolist(),
                "frames": valid_frames,
                "samples": valid_frames * 600,
            },
            "hmonnx_converted": convert_hmonnx,
            "hmonnx_is_unrolled_diagnostic": convert_hmonnx and decompose_lstm,
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        import torch

        work_dir = Path(export_result.work_dir).expanduser().resolve()
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"missing Kokoro export metadata: {meta_file}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if not meta.get("hmonnx_converted"):
            raise RuntimeError("dump_golden requires export.convert_hmonnx=true")

        tokens, style, speed, seed = self._golden_input(
            input_messages,
            default_seed=int(meta["seed"]),
        )
        if meta.get("graph_mode") == "single_graph":
            return self._dump_single_graph_golden(
                work_dir=work_dir,
                meta=meta,
                tokens=tokens,
                style=style,
                speed=speed,
                seed=seed,
                device=device,
            )
        if meta.get("graph_mode") == BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
            return self._dump_bucketed_precision_split_golden(
                work_dir=work_dir,
                meta=meta,
                tokens=tokens,
                style=style,
                speed=speed,
                seed=seed,
                device=device,
            )
        if meta.get("graph_mode") == PRECISION_SPLIT_GRAPH_MODE:
            return self._dump_precision_split_golden(
                work_dir=work_dir,
                meta=meta,
                tokens=tokens,
                style=style,
                speed=speed,
                seed=seed,
                device=device,
            )
        recordings: dict[str, list[dict[str, np.ndarray]]] = {role: [] for role in GRAPH_ROLES}
        runners: dict[str, Runner] = {}
        for role in GRAPH_ROLES:
            component = meta["components"][role]
            runners[role] = _RecordingRunner(
                OrtRunner(work_dir / component["onnx_file"]),
                recordings[role],
            )
        runtime = KokoroStaticRuntime(
            runners,
            text_max_length=int(meta["text_max_length"]),
            frame_max_length=int(meta["frame_max_length"]),
            lstm_chunk_length=int(meta["lstm_chunk_length"]),
            seed=int(meta["seed"]),
        )
        waveform, synthesis = runtime.synthesize(
            tokens,
            style,
            speed=speed,
            seed=seed,
        )

        golden_root = work_dir / "golden"
        if golden_root.exists():
            shutil.rmtree(golden_root)
        torch_device = str(device) if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        graph_manifest: dict[str, Any] = {}
        for role in GRAPH_ROLES:
            component = meta["components"][role]
            feeds = recordings[role]
            if not feeds:
                raise RuntimeError(f"runtime did not invoke graph {role}")
            _run_golden_component(
                work_dir / component["hmonnx_file"],
                feeds,
                golden_root / role,
                torch_device,
            )
            graph_manifest[role] = {
                "hmonnx_file": component["hmonnx_file"],
                "golden_dir": role,
                "steps": len(feeds),
            }
        manifest = golden_root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "input_ids": tokens.tolist(),
                    "speed": speed,
                    "seed": seed,
                    "synthesis": synthesis,
                    "waveform_summary": {
                        "samples": int(waveform.size),
                        "minimum": float(waveform.min()),
                        "maximum": float(waveform.max()),
                        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    },
                    "graphs": graph_manifest,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(manifest)

    def _dump_bucketed_precision_split_golden(
        self,
        *,
        work_dir: Path,
        meta: Mapping[str, Any],
        tokens: np.ndarray,
        style: np.ndarray,
        speed: float,
        seed: int,
        device: str,
    ) -> str:
        import torch

        if seed != int(meta["seed"]):
            raise ValueError(f"expected fixed SineGen seed {meta['seed']}, got {seed}")
        style_array = np.asarray(style, dtype=np.float32).reshape(1, 256)
        golden_root = work_dir / "golden"
        if golden_root.exists():
            shutil.rmtree(golden_root)
        torch_device = str(device) if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"

        runtime = KokoroBucketedRuntime(
            work_dir,
            meta,
            backend="ort",
            lstm_variant="native",
            device=device,
        )
        recordings: dict[tuple[str, str], list[dict[str, np.ndarray]]] = {}
        for role in meta["component_order"]:
            for key, bucket in meta["components"][role]["buckets"].items():
                feeds: list[dict[str, np.ndarray]] = []
                onnx_file = bucket.get("reference_onnx_file", bucket["onnx_file"])
                runtime._runners[(role, str(key))] = _RecordingRunner(
                    OrtRunner(work_dir / onnx_file),
                    feeds,
                )
                recordings[(role, str(key))] = feeds

        waveform, synthesis = runtime.synthesize(
            tokens,
            style_array,
            speed=speed,
            seed=seed,
        )
        token_key = token_bucket_key(int(synthesis["token_bucket"]))
        frame_key = frame_bucket_key(int(synthesis["frame_bucket"]))
        if runtime.phase_on_npu:
            selected = (
                (TEXT_DURATION_ROLE, token_key),
                (FRAME_SYNTHESIS_ROLE, frame_key),
            )
        else:
            selected = (
                (TEXT_DURATION_ROLE, token_key),
                (FRAME_ACOUSTIC_ROLE, frame_key),
                (INDEPENDENT_PHASE_CORE_ROLE, frame_key),
                (INDEPENDENT_GENERATOR_ISTFT_ROLE, frame_key),
            )

        graph_manifest: dict[str, Any] = {}
        for role, key in selected:
            feeds = recordings[(role, key)]
            if len(feeds) != 1:
                raise RuntimeError(
                    f"bucketed runtime invoked {role}/{key} {len(feeds)} times, expected once"
                )
            component = meta["components"][role]["buckets"][key]
            if role == INDEPENDENT_PHASE_CORE_ROLE:
                relative_dir = Path(role) / key
                destination = golden_root / relative_dir
                destination.mkdir(parents=True, exist_ok=True)
                outputs = OrtRunner(work_dir / component["onnx_file"]).run(feeds[0])
                np.savez(destination / "inputs.npz", **feeds[0])
                np.savez(destination / "outputs.npz", **outputs)
                graph_manifest[role] = {
                    "bucket": key,
                    "execution": "host_onnxruntime_fp32",
                    "onnx_file": component["onnx_file"],
                    "golden_dir": str(relative_dir),
                }
                continue

            if "hmonnx_variants" in component:
                variants: dict[str, Any] = {}
                for variant, entry in component["hmonnx_variants"].items():
                    if entry["status"] != "ok":
                        variants[variant] = {
                            "status": entry["status"],
                            "error": entry.get("error"),
                        }
                        continue
                    relative_dir = Path(role) / variant / key
                    _run_golden_component(
                        work_dir / entry["hmonnx_file"],
                        feeds,
                        golden_root / relative_dir,
                        torch_device,
                    )
                    variants[variant] = {
                        "status": "ok",
                        "hmonnx_file": entry["hmonnx_file"],
                        "golden_dir": str(relative_dir),
                        "steps": 1,
                    }
                graph_manifest[role] = {
                    "bucket": key,
                    "variants": variants,
                }
                continue

            relative_dir = Path(role) / key
            _run_golden_component(
                work_dir / component["hmonnx_file"],
                feeds,
                golden_root / relative_dir,
                torch_device,
            )
            graph_manifest[role] = {
                "bucket": key,
                "hmonnx_file": component["hmonnx_file"],
                "golden_dir": str(relative_dir),
                "steps": 1,
            }

        manifest = golden_root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "input_ids": tokens.tolist(),
                    "speed": speed,
                    "seed": seed,
                    "synthesis": synthesis,
                    "waveform_summary": {
                        "samples": int(waveform.size),
                        "minimum": float(waveform.min()),
                        "maximum": float(waveform.max()),
                        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    },
                    "graphs": graph_manifest,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(manifest)

    def _dump_precision_split_golden(
        self,
        *,
        work_dir: Path,
        meta: Mapping[str, Any],
        tokens: np.ndarray,
        style: np.ndarray,
        speed: float,
        seed: int,
        device: str,
    ) -> str:
        import torch

        recordings: dict[str, list[dict[str, np.ndarray]]] = {role: [] for role in PRECISION_SPLIT_NPU_ROLES}
        runners: dict[str, Runner] = {}
        for role in PRECISION_SPLIT_ROLES:
            component = meta["components"][role]
            runner: Runner = OrtRunner(work_dir / component["onnx_file"])
            if role in recordings:
                runner = _RecordingRunner(runner, recordings[role])
            runners[role] = runner
        runtime = KokoroPrecisionSplitRuntime(
            runners,
            text_max_length=int(meta["text_max_length"]),
            frame_max_length=int(meta["frame_max_length"]),
            seed=int(meta["seed"]),
        )
        waveform, synthesis = runtime.synthesize(
            tokens,
            style,
            speed=speed,
            seed=seed,
        )

        golden_root = work_dir / "golden"
        if golden_root.exists():
            shutil.rmtree(golden_root)
        torch_device = str(device) if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        graph_manifest: dict[str, Any] = {}
        for role in PRECISION_SPLIT_NPU_ROLES:
            component = meta["components"][role]
            feeds = recordings[role]
            if len(feeds) != 1:
                raise RuntimeError(f"precision-split runtime invoked {role} {len(feeds)} times, expected once")
            _run_golden_component(
                work_dir / component["hmonnx_file"],
                feeds,
                golden_root / role,
                torch_device,
            )
            graph_manifest[role] = {
                "hmonnx_file": component["hmonnx_file"],
                "golden_dir": role,
                "steps": 1,
            }
        graph_manifest[PHASE_CORE_ROLE] = {
            "onnx_file": meta["components"][PHASE_CORE_ROLE]["onnx_file"],
            "execution": "host_onnxruntime_fp32",
        }
        manifest = golden_root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "input_ids": tokens.tolist(),
                    "speed": speed,
                    "seed": seed,
                    "synthesis": synthesis,
                    "waveform_summary": {
                        "samples": int(waveform.size),
                        "minimum": float(waveform.min()),
                        "maximum": float(waveform.max()),
                        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    },
                    "graphs": graph_manifest,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(manifest)

    def _dump_single_graph_golden(
        self,
        *,
        work_dir: Path,
        meta: Mapping[str, Any],
        tokens: np.ndarray,
        style: np.ndarray,
        speed: float,
        seed: int,
        device: str,
    ) -> str:
        import torch

        if speed <= 0:
            raise ValueError("Kokoro speed must be positive")
        if seed != int(meta["seed"]):
            raise ValueError(
                f"single-graph SineGen randomness is fixed at export time; expected seed {meta['seed']}, got {seed}"
            )
        text_max_length = int(meta["text_max_length"])
        frame_max_length = int(meta["frame_max_length"])
        if tokens.size > text_max_length:
            raise ValueError(f"token length {tokens.size} exceeds T bucket {text_max_length}")
        input_ids = np.zeros((1, text_max_length), dtype=np.int32)
        input_ids[0, : tokens.size] = tokens
        feed = {
            "input_ids": input_ids,
            "style": np.asarray(style, dtype=np.float32).reshape(1, 256),
            "speed": np.asarray([speed], dtype=np.float32),
            "valid_len": np.asarray([tokens.size], dtype=np.int32),
        }
        component = meta["components"][SINGLE_GRAPH_ROLE]
        output = OrtRunner(work_dir / component["onnx_file"]).run(feed)
        valid_frames = int(np.asarray(output["valid_frames"]).reshape(-1)[0])
        if valid_frames > frame_max_length:
            raise ValueError(
                f"duration produced F={valid_frames}, exceeding bucket F={frame_max_length}; retry a larger F bucket"
            )
        valid_samples = valid_frames * int(meta["samples_per_frame"])
        waveform = np.asarray(output["waveform"], dtype=np.float32).reshape(-1)[:valid_samples]
        golden_root = work_dir / "golden"
        if golden_root.exists():
            shutil.rmtree(golden_root)
        torch_device = str(device) if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        _run_golden_component(
            work_dir / component["hmonnx_file"],
            [feed],
            golden_root / SINGLE_GRAPH_ROLE,
            torch_device,
        )
        manifest = golden_root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "input_ids": tokens.tolist(),
                    "speed": speed,
                    "seed": seed,
                    "synthesis": {
                        "frame_length": valid_frames,
                        "waveform_samples": valid_samples,
                    },
                    "waveform_summary": {
                        "samples": int(waveform.size),
                        "minimum": float(waveform.min()),
                        "maximum": float(waveform.max()),
                        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    },
                    "graphs": {
                        SINGLE_GRAPH_ROLE: {
                            "hmonnx_file": component["hmonnx_file"],
                            "golden_dir": SINGLE_GRAPH_ROLE,
                            "steps": 1,
                        }
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(manifest)

    def _golden_input(
        self,
        input_messages: Any,
        *,
        default_seed: int,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        assets = resolve_model_assets(self.model_dir)
        if input_messages is None:
            tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
            style = load_voice_style(assets.voice, phoneme_count=tokens.size - 2).numpy()
            return tokens, style, 1.0, int(default_seed)
        if not isinstance(input_messages, Mapping):
            raise TypeError("Kokoro golden input must be a mapping or None")
        tokens = np.asarray(input_messages["input_ids"], dtype=np.int32).reshape(-1)
        if tokens.size == 0:
            raise ValueError("input_ids must not be empty")
        if "style" in input_messages:
            style = np.asarray(input_messages["style"], dtype=np.float32)
        else:
            style = load_voice_style(
                assets.voice,
                phoneme_count=max(int(tokens.size) - 2, 1),
            ).numpy()
        return (
            tokens,
            style,
            float(input_messages.get("speed", 1.0)),
            int(input_messages.get("seed", default_seed)),
        )


class _RecordingRunner:
    def __init__(
        self,
        delegate: Runner,
        recordings: list[dict[str, np.ndarray]],
    ) -> None:
        self.delegate = delegate
        self.recordings = recordings

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.recordings.append({name: np.array(value, copy=True) for name, value in feed.items()})
        return self.delegate.run(feed)


def _validate_export_config(export_cfg: Mapping[str, Any]) -> None:
    numerical_validation_keys = sorted(
        {"validate_onnx", "validate_hmonnx"}.intersection(export_cfg)
    )
    if numerical_validation_keys:
        raise ValueError(
            "Kokoro workflow only quantizes and exports; run numerical validation "
            "from examples_merak/audio/kokoro/compare_backends.py instead of setting "
            f"export.{numerical_validation_keys[0]}"
        )
    if not isinstance(export_cfg.get("torch_onnx_internal_optimize", True), bool):
        raise TypeError("export.torch_onnx_internal_optimize must be a boolean")
    if export_cfg.get("model", {}).get("type") != "XHKokoroModel":
        raise ValueError("Kokoro requires export.model.type='XHKokoroModel'")
    if export_cfg.get("target_device") != "XH2a":
        raise ValueError("Kokoro workflow currently supports XH2a only")
    graph_mode = str(export_cfg.get("graph_mode", "multi_graph"))
    if graph_mode not in {
        "multi_graph",
        "single_graph",
        PRECISION_SPLIT_GRAPH_MODE,
        BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
    }:
        raise ValueError(
            "export.graph_mode must be multi_graph, single_graph, precision_split, or bucketed_precision_split"
        )
    if str(export_cfg.get("f0_norm_mode", "adain")) not in {"adain", "rmsnorm"}:
        raise ValueError("export.f0_norm_mode must be adain or rmsnorm")
    if graph_mode == BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
        if "bucket_routes" in export_cfg:
            raise ValueError(
                "export.bucket_routes is not supported; use matching export.token_buckets "
                "and export.audio_seconds_buckets"
            )
        normalize_bucket_routes(
            export_cfg.get("token_buckets"),
            export_cfg.get("audio_seconds_buckets"),
        )
        variants = tuple(str(value) for value in export_cfg.get("lstm_variants", LSTM_VARIANTS))
        if not variants or len(set(variants)) != len(variants):
            raise ValueError("export.lstm_variants must contain unique values")
        unsupported = sorted(set(variants).difference(LSTM_VARIANTS))
        if unsupported:
            raise ValueError(f"unsupported export.lstm_variants: {unsupported}")
    else:
        for name in ("text_max_length", "frame_max_length"):
            if int(export_cfg.get(name, 0)) <= 0:
                raise ValueError(f"export.{name} must be positive")
    components = export_cfg.get("components")
    if not isinstance(components, Mapping):
        raise TypeError("export.components must be a mapping")
    if graph_mode == "multi_graph":
        roles = GRAPH_ROLES
    elif graph_mode == "single_graph":
        roles = (SINGLE_GRAPH_ROLE,)
    elif graph_mode == BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
        roles = (
            (TEXT_DURATION_ROLE, FRAME_SYNTHESIS_ROLE)
            if bool(export_cfg.get("phase_on_npu", False))
            else (TEXT_DURATION_ROLE, FRAME_ACOUSTIC_ROLE, INDEPENDENT_GENERATOR_ISTFT_ROLE)
        )
    else:
        roles = PRECISION_SPLIT_NPU_ROLES
    missing = sorted(set(roles).difference(components))
    if missing:
        raise ValueError(f"export.components is missing Kokoro graphs: {missing}")
    for role in roles:
        value = components[role]
        if not isinstance(value, Mapping) or not value.get("quant_type"):
            raise ValueError(f"export.components.{role}.quant_type must be set")
    if graph_mode == "multi_graph":
        chunk = int(export_cfg.get("lstm_chunk_length", 0))
        if chunk <= 0:
            raise ValueError("export.lstm_chunk_length must be positive")
        if int(export_cfg["frame_max_length"]) % chunk:
            raise ValueError("export.frame_max_length must be divisible by lstm_chunk_length")
        return

    pad_mode = str(export_cfg.get("stft_pad_mode", "length_aware_reflect"))
    if pad_mode not in {"constant", "reflect", "replicate", "length_aware_reflect"}:
        raise ValueError("unsupported export.stft_pad_mode")
    phase_mode = str(export_cfg.get("stft_phase_mode", "cordic"))
    if phase_mode not in {"atan2", "cordic"}:
        raise ValueError("export.stft_phase_mode must be atan2 or cordic")
    if bool(export_cfg.get("convert_hmonnx", True)):
        if graph_mode == "single_graph" and phase_mode != "cordic":
            raise ValueError("single-graph HMONNX conversion requires stft_phase_mode=cordic")
        if graph_mode != BUCKETED_PRECISION_SPLIT_GRAPH_MODE and bool(export_cfg.get("decompose_lstm", False)):
            raise ValueError(f"{graph_mode} Kokoro HMONNX must keep native LSTM nodes; set export.decompose_lstm=false")


def _convert_hmonnx(
    artifact: GraphArtifact,
    feed: Mapping[str, Any],
    output_path: Path,
    *,
    target: str,
    quant_type: str,
    debug: bool,
    decompose_lstm: bool,
    torch_onnx_internal_optimize: bool = True,
) -> None:
    import torch

    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        xhquant_init,
    )

    try:
        device_type = getattr(DeviceType, target)
    except AttributeError as error:
        raise ValueError(f"unsupported xhquant target device: {target}") from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(output_path.parent / f"{output_path.stem}.log"), debug)
    quant_config = create_quant_config(QuantScheme(target_device=device_type, quant_type=quant_type))
    inputs = []
    for name in artifact.input_names:
        value = feed[name]
        if torch.is_tensor(value):
            inputs.append(value.detach().cpu())
        else:
            inputs.append(torch.from_numpy(np.asarray(value)))
    with _xhquant_torch_onnx_optimize_context(torch_onnx_internal_optimize):
        convert_onnx_to_hmonnx(
            str(artifact.path),
            inputs,
            device_type,
            str(output_path),
            quant_config=quant_config,
            input_names=list(artifact.input_names),
            output_names=list(artifact.output_names),
            # Use xhquant's normal frontend constant folding so static Slice and
            # Reshape parameters become graph constants.  No Kokoro-specific ONNX
            # optimization is applied before this conversion.
            simplify=True,
            decompose_lstm=decompose_lstm,
        )


@contextmanager
def _xhquant_torch_onnx_optimize_context(enabled: bool):
    """Scope exporter normalization without leaking environment state."""

    previous = os.environ.get(XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV)
    os.environ[XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV, None)
        else:
            os.environ[XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV] = previous


def _hmonnx_artifact_metadata(path: Path, work_dir: Path) -> dict[str, Any]:
    import onnx

    proto = onnx.load(str(path), load_external_data=False)
    op_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    lstm_attributes: list[dict[str, Any]] = []
    for node in proto.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        domain = node.domain or "ai.onnx"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if node.op_type == "LSTM":
            attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
            lstm_attributes.append(
                {
                    "name": node.name,
                    "domain": domain,
                    "direction": _decode_onnx_attribute(attributes.get("direction")),
                    "has_sequence_lens": len(node.input) > 8 and bool(node.input[8]),
                    "hidden_size": int(attributes.get("hidden_size", 0)),
                }
            )
    return {
        "hmonnx_file": str(path.relative_to(work_dir)),
        "hmonnx_sha256": sha256(path),
        "hmonnx_size_bytes": path.stat().st_size,
        "hmonnx_external_data": _external_data_metadata(proto, path, work_dir),
        "hmonnx_node_count": len(proto.graph.node),
        "hmonnx_op_counts": dict(sorted(op_counts.items())),
        "hmonnx_domain_counts": dict(sorted(domain_counts.items())),
        "hmonnx_lstm_nodes": len(lstm_attributes),
        "hmonnx_lstm_attributes": lstm_attributes,
    }


def _external_data_metadata(
    proto: Any,
    model_path: Path,
    work_dir: Path,
) -> list[dict[str, Any]]:
    import onnx

    locations: set[str] = set()
    for initializer in proto.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {item.key: item.value for item in initializer.external_data}
        location = fields.get("location")
        if location:
            locations.add(location)
    result: list[dict[str, Any]] = []
    for location in sorted(locations):
        external_path = model_path.parent / location
        if not external_path.is_file():
            raise FileNotFoundError(f"HMONNX external data referenced by {model_path} is missing: {external_path}")
        result.append(
            {
                "file": str(external_path.relative_to(work_dir)),
                "sha256": sha256(external_path),
                "size_bytes": external_path.stat().st_size,
            }
        )
    return result


def _export_runtime_assets(voice_path: str | Path, work_dir: Path) -> dict[str, Any]:
    """Export Host-only constants required by the self-contained demo."""

    source = Path(voice_path).expanduser().resolve()
    destination = work_dir / "assets" / "voices" / f"{source.stem}.npy"
    save_voice_pack_numpy(source, destination)
    voice = np.load(destination, mmap_mode="r", allow_pickle=False)
    return {
        "voice_pack": {
            "name": source.stem,
            "file": str(destination.relative_to(work_dir)),
            "sha256": sha256(destination),
            "dtype": str(voice.dtype),
            "shape": list(voice.shape),
            "size_bytes": destination.stat().st_size,
            "format": "numpy_npy_allow_pickle_false",
        }
    }


def _bucketed_lstm_hmonnx_path(
    hmonnx_dir: Path,
    *,
    role: str,
    variant: str,
    bucket_key: str,
    target: str,
    quant_type: str,
) -> Path:
    if variant not in LSTM_VARIANTS:
        raise ValueError(f"unsupported LSTM HMONNX variant: {variant}")
    return (
        hmonnx_dir
        / role
        / variant
        / bucket_key
        / f"kokoro_{role}_b1_{bucket_key}_{target}_{quant_type}_lstm_{variant}.onnx"
    )


def _bucketed_non_lstm_hmonnx_path(
    hmonnx_dir: Path,
    *,
    role: str,
    bucket_key: str,
    target: str,
    quant_type: str,
) -> Path:
    return (
        hmonnx_dir
        / role
        / bucket_key
        / f"kokoro_{role}_b1_{bucket_key}_{target}_{quant_type}.onnx"
    )


def _convert_lstm_bucket_variants(
    *,
    artifact: GraphArtifact,
    feed: Mapping[str, Any],
    hmonnx_dir: Path,
    role: str,
    bucket_key: str,
    target: str,
    quant_type: str,
    variants: tuple[str, ...],
    expected_native_lstm_nodes: int,
    convert_hmonnx: bool,
    allow_decomposed_failure: bool,
    torch_onnx_internal_optimize: bool,
    debug: bool,
    work_dir: Path,
    failures: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results = {
        variant: {
            "status": "not_requested",
            "decompose_lstm": variant == "decomposed",
            "lowering_stage": "torch.export",
        }
        for variant in LSTM_VARIANTS
    }
    if not convert_hmonnx:
        return results
    for variant in variants:
        output_path = _bucketed_lstm_hmonnx_path(
            hmonnx_dir,
            role=role,
            variant=variant,
            bucket_key=bucket_key,
            target=target,
            quant_type=quant_type,
        )
        started = time.perf_counter()
        try:
            _convert_hmonnx(
                artifact,
                feed,
                output_path,
                target=target,
                quant_type=quant_type,
                debug=debug,
                decompose_lstm=variant == "decomposed",
                torch_onnx_internal_optimize=torch_onnx_internal_optimize,
            )
            entry = _hmonnx_artifact_metadata(output_path, work_dir)
            entry.update(
                {
                    "status": "ok",
                    "decompose_lstm": variant == "decomposed",
                    "lowering_stage": "torch.export",
                    "conversion_seconds": time.perf_counter() - started,
                }
            )
            _validate_lstm_variant(
                entry,
                variant,
                expected_native_lstm_nodes=expected_native_lstm_nodes,
            )
        except Exception as error:
            if variant != "decomposed" or not allow_decomposed_failure:
                raise
            entry = {
                "status": "failed",
                "decompose_lstm": True,
                "lowering_stage": "torch.export",
                "error_type": type(error).__name__,
                "error": str(error),
                "conversion_seconds": time.perf_counter() - started,
            }
            failures.append({"role": role, "bucket": bucket_key, **entry})
        results[variant] = entry
    return results


def _decode_onnx_attribute(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _validate_lstm_variant(
    metadata: Mapping[str, Any],
    variant: str,
    *,
    expected_native_lstm_nodes: int,
) -> None:
    count = int(metadata["hmonnx_lstm_nodes"])
    if variant == "decomposed":
        if count:
            raise RuntimeError(f"decomposed HMONNX must not contain LSTM nodes, got {count}")
        return
    if count != expected_native_lstm_nodes:
        raise RuntimeError(f"native HMONNX must contain {expected_native_lstm_nodes} LSTM nodes, got {count}")
    for node in metadata["hmonnx_lstm_attributes"]:
        if node["domain"] != "ai.houmo.xh2a" or node["direction"] != "forward" or node["has_sequence_lens"]:
            raise RuntimeError(f"native HMONNX does not follow Kokoro's standard forward LSTM contract: {node}")


def _run_golden_component(
    hmonnx_path: Path,
    feeds: Sequence[Mapping[str, np.ndarray]],
    golden_dir: Path,
    device: str,
) -> None:
    import onnx
    import torch

    from xhquant.api import HMONNXGoldenInference

    proto = onnx.load(str(hmonnx_path), load_external_data=False)
    input_names = [value.name for value in proto.graph.input]
    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_path))
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    for step, feed in enumerate(feeds):
        tensors = []
        for name in input_names:
            array = np.asarray(feed[name])
            if np.issubdtype(array.dtype, np.integer):
                tensor = torch.from_numpy(array.astype(np.int32, copy=False))
            else:
                tensor = torch.from_numpy(array.astype(np.float16, copy=False))
            tensors.append(tensor.to(device))
        session.step = step
        session(*tensors)


__all__ = ["KokoroWorkflow"]
