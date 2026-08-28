from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .buckets import (
    LSTM_VARIANTS,
    audio_seconds_for_frame,
    frame_bucket_key,
    token_bucket_key,
)
from .host import (
    ATTENTION_MASK_MIN,
    WAVEFORM_SAMPLES_PER_FRAME,
    duration_from_logits,
    make_generator_rmsnorm_scales,
    make_reverse_idx,
)
from .independent_split import (
    FRAME_ACOUSTIC_ROLE,
    FRAME_SYNTHESIS_ROLE,
    GENERATOR_ISTFT_ROLE,
    PHASE_CORE_ROLE,
    TEXT_DURATION_ROLE,
    run_duration_alignment_host,
)
from .runtime import HmonnxRunner, OrtRunner, Runner


BUCKETED_PRECISION_SPLIT_GRAPH_MODE = "bucketed_precision_split"


class KokoroBucketedRuntime:
    """Route one utterance through the four approved paired T/F gears."""

    def __init__(
        self,
        work_dir: str | Path,
        meta: Mapping[str, Any],
        *,
        backend: str,
        lstm_variant: str,
        device: str,
    ) -> None:
        if backend not in {"ort", "hmonnx"}:
            raise ValueError("backend must be 'ort' or 'hmonnx'")
        if lstm_variant not in LSTM_VARIANTS:
            raise ValueError(f"lstm_variant must be one of {LSTM_VARIANTS}")
        if meta.get("graph_mode") != BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
            raise ValueError("metadata is not a Kokoro bucketed precision-split export")
        policy = meta.get("bucket_presets", {}).get("policy")
        if policy != "duration_driven_paired_t_f":
            raise ValueError(f"metadata uses unsupported Kokoro bucket policy: {policy!r}")

        self.work_dir = Path(work_dir).expanduser().resolve()
        self.meta = dict(meta)
        self.backend = backend
        self.lstm_variant = lstm_variant
        self.device = device
        self.seed = int(meta["seed"])
        self.phase_on_npu = meta.get("phase_boundary", {}).get("mode") == "npu"
        self.frame_lstm_role = FRAME_SYNTHESIS_ROLE if self.phase_on_npu else FRAME_ACOUSTIC_ROLE
        self.token_buckets = self._sorted_buckets(TEXT_DURATION_ROLE, "token_max_length")
        self.frame_buckets = self._sorted_buckets(self.frame_lstm_role, "frame_max_length")
        self.routes = tuple(
            sorted(
                (dict(route) for route in meta["bucket_presets"]["routes"]),
                key=lambda route: (
                    int(route["token_max_length"]),
                    int(route["frame_max_length"]),
                ),
            )
        )
        route_keys = [str(route["key"]) for route in self.routes]
        if not self.routes or len(route_keys) != len(set(route_keys)):
            raise ValueError("metadata must contain unique paired Kokoro routes")
        text_keys = {str(value["key"]) for value in self.token_buckets}
        frame_keys = {str(value["key"]) for value in self.frame_buckets}
        for route in self.routes:
            text_key = token_bucket_key(int(route["token_max_length"]))
            frame_key = frame_bucket_key(int(route["frame_max_length"]))
            if text_key not in text_keys or frame_key not in frame_keys:
                raise ValueError(f"paired route {route['key']} references a missing component bucket")
        self._runners: dict[tuple[str, str], Runner] = {}

    @classmethod
    def from_export(
        cls,
        work_dir: str | Path,
        *,
        backend: str = "hmonnx",
        lstm_variant: str = "native",
        device: str = "cuda:0",
    ) -> KokoroBucketedRuntime:
        root = Path(work_dir).expanduser().resolve()
        meta_file = root / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"missing Kokoro export metadata: {meta_file}")
        return cls(
            root,
            json.loads(meta_file.read_text(encoding="utf-8")),
            backend=backend,
            lstm_variant=lstm_variant,
            device=device,
        )

    def available_buckets(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            "token": tuple(dict(value) for value in self.token_buckets),
            "frame": tuple(dict(value) for value in self.frame_buckets),
        }

    def available_routes(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(route) for route in self.routes)

    def clear_runner_cache(self) -> None:
        """Release bucket-specific runners after bounded-memory validation runs."""

        self._runners.clear()
        gc.collect()
        device = torch.device(
            self.device
            if self.backend == "hmonnx" and str(self.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def synthesize(
        self,
        token_ids: np.ndarray | list[int] | tuple[int, ...],
        style: np.ndarray | Tensor,
        *,
        speed: float = 1.0,
        seed: int | None = None,
        token_bucket: int | None = None,
        frame_bucket: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        if tokens.size == 0:
            raise ValueError("token_ids must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")
        requested_seed = self.seed if seed is None else int(seed)
        if requested_seed != self.seed:
            raise ValueError(f"SineGen seed is fixed at export time; expected {self.seed}, got {requested_seed}")
        style_array = np.asarray(
            style.detach().float().cpu().numpy() if isinstance(style, Tensor) else style,
            dtype=np.float32,
        )
        if style_array.shape == (256,):
            style_array = style_array[None]
        if style_array.shape != (1, 256):
            raise ValueError(f"style must have shape [1,256], got {style_array.shape}")

        valid_len = torch.tensor([tokens.size], dtype=torch.int32)
        speed_tensor = torch.tensor([speed], dtype=torch.float32)
        candidate_routes = self._candidate_routes(
            int(tokens.size),
            token_bucket=token_bucket,
            frame_bucket=frame_bucket,
        )
        attempts: list[dict[str, int | str]] = []
        selected_route: dict[str, Any] | None = None
        text_output: dict[str, np.ndarray] | None = None
        duration: Tensor | None = None
        valid_frames = -1
        for route in candidate_routes:
            text_capacity = int(route["token_max_length"])
            frame_capacity = int(route["frame_max_length"])
            text_output = self._runner(
                TEXT_DURATION_ROLE,
                token_bucket_key(text_capacity),
            ).run(self._text_feed(tokens, style_array, text_capacity))
            duration = duration_from_logits(
                torch.from_numpy(np.asarray(text_output["duration_logits"], dtype=np.float32)),
                speed_tensor,
                valid_len,
            )
            valid_frames = int(duration.sum().item())
            attempts.append(
                {
                    "route": str(route["key"]),
                    "token_bucket": text_capacity,
                    "frame_bucket": frame_capacity,
                    "predicted_frames": valid_frames,
                }
            )
            if valid_frames <= frame_capacity:
                selected_route = route
                break
        if selected_route is None or text_output is None or duration is None:
            predicted = int(attempts[-1]["predicted_frames"]) if attempts else -1
            largest = candidate_routes[-1]
            raise ValueError(
                f"duration produced F={predicted}, exceeding the largest selected paired route "
                f"T={largest['token_max_length']}/F={largest['frame_max_length']}"
            )

        text_capacity = int(selected_route["token_max_length"])
        frame_capacity = int(selected_route["frame_max_length"])
        frame_key = frame_bucket_key(frame_capacity)
        frame_inputs, checked_duration = run_duration_alignment_host(
            {
                "duration_features": torch.from_numpy(np.asarray(text_output["duration_features"], dtype=np.float32)),
                "text_encoded": torch.from_numpy(np.asarray(text_output["text_encoded"], dtype=np.float32)),
                "duration_logits": torch.from_numpy(np.asarray(text_output["duration_logits"], dtype=np.float32)),
            },
            speed=speed_tensor,
            valid_len=valid_len,
            frame_max_length=frame_capacity,
        )
        if not torch.equal(duration, checked_duration):
            raise RuntimeError("Host duration changed between F routing and frame expansion")
        frame_feed = {
            "encoded": frame_inputs["encoded"].numpy(),
            "asr": frame_inputs["asr"].numpy(),
            "style": style_array,
            "valid_frames": np.asarray([valid_frames], dtype=np.int32),
            "reverse_indices": make_reverse_idx(
                frame_capacity,
                torch.tensor([valid_frames], dtype=torch.int32),
            )
            .numpy()
            .astype(np.int64, copy=False),
        }
        generator_norm_scales = make_generator_rmsnorm_scales(
            torch.tensor([valid_frames], dtype=torch.int32),
            frame_capacity,
        ).numpy()

        if self.phase_on_npu:
            generated = self._runner(FRAME_SYNTHESIS_ROLE, frame_key).run(
                {
                    **frame_feed,
                    "generator_norm_scales": generator_norm_scales,
                }
            )
        else:
            acoustic = self._runner(FRAME_ACOUSTIC_ROLE, frame_key).run(frame_feed)
            phase = self._runner(PHASE_CORE_ROLE, frame_key).run(
                {"phase_increments": np.asarray(acoustic["phase_increments"], dtype=np.float32)}
            )
            generated = self._runner(GENERATOR_ISTFT_ROLE, frame_key).run(
                {
                    "decoder_feature": np.asarray(acoustic["decoder_feature"], dtype=np.float32),
                    "sine": np.asarray(phase["sine"], dtype=np.float32),
                    "f0": np.asarray(acoustic["f0"], dtype=np.float32),
                    "style": style_array,
                    "valid_frames": np.asarray([valid_frames], dtype=np.int32),
                    "generator_norm_scales": generator_norm_scales,
                }
            )

        valid_samples = valid_frames * WAVEFORM_SAMPLES_PER_FRAME
        waveform = np.asarray(generated["waveform"], dtype=np.float32).reshape(-1)[:valid_samples]
        return waveform, {
            "text_length": int(tokens.size),
            "route": str(selected_route["key"]),
            "token_bucket": text_capacity,
            "frame_length": valid_frames,
            "frame_bucket": frame_capacity,
            "audio_bucket_seconds": audio_seconds_for_frame(frame_capacity),
            "valid_samples": valid_samples,
            "duration": duration[0, : tokens.size].tolist(),
            "backend": self.backend,
            "lstm_variant": self.lstm_variant,
            "phase_boundary": "npu" if self.phase_on_npu else "host_fp32",
            "npu_graph_count": 2 if self.phase_on_npu else 3,
            "attempts": attempts,
            "seed": requested_seed,
        }

    def _sorted_buckets(self, role: str, capacity_field: str) -> tuple[dict[str, Any], ...]:
        try:
            buckets = self.meta["components"][role]["buckets"]
        except KeyError as error:
            raise ValueError(f"metadata is missing Kokoro component {role}") from error
        values = tuple(
            sorted(
                ({"key": str(key), **dict(value)} for key, value in buckets.items()),
                key=lambda value: int(value[capacity_field]),
            )
        )
        capacities = [int(value[capacity_field]) for value in values]
        if not values or len(capacities) != len(set(capacities)):
            raise ValueError(f"metadata must contain unique {role} bucket capacities")
        return values

    def _candidate_routes(
        self,
        token_length: int,
        *,
        token_bucket: int | None,
        frame_bucket: int | None,
    ) -> tuple[dict[str, Any], ...]:
        candidates = tuple(
            route
            for route in self.routes
            if token_length <= int(route["token_max_length"])
            and (token_bucket is None or int(route["token_max_length"]) == int(token_bucket))
            and (frame_bucket is None or int(route["frame_max_length"]) == int(frame_bucket))
        )
        if candidates:
            return candidates
        raise ValueError(
            f"no exported paired route fits token_length={token_length}, "
            f"token_bucket={token_bucket}, frame_bucket={frame_bucket}"
        )

    @staticmethod
    def _text_feed(
        tokens: np.ndarray,
        style: np.ndarray,
        text_max_length: int,
    ) -> dict[str, np.ndarray]:
        input_ids = np.zeros((1, text_max_length), dtype=np.int32)
        input_ids[0, : tokens.size] = tokens
        attention_mask = np.full(
            (1, 1, text_max_length, text_max_length),
            ATTENTION_MASK_MIN,
            dtype=np.float32,
        )
        attention_mask[..., : tokens.size] = 0.0
        return {
            "input_ids": np.ascontiguousarray(input_ids),
            "attention_mask": np.ascontiguousarray(attention_mask),
            "style": np.ascontiguousarray(style),
            "valid_len": np.asarray([tokens.size], dtype=np.int32),
            "reverse_indices": make_reverse_idx(
                text_max_length,
                torch.tensor([tokens.size], dtype=torch.int32),
            )
            .numpy()
            .astype(np.int64, copy=False),
        }

    def _runner(self, role: str, key: str) -> Runner:
        cache_key = (role, key)
        runner = self._runners.get(cache_key)
        if runner is not None:
            return runner
        bucket = self.meta["components"][role]["buckets"][key]
        if self.backend == "ort" or role == PHASE_CORE_ROLE:
            path = bucket.get("reference_onnx_file", bucket["onnx_file"])
            runner = OrtRunner(self.work_dir / path)
        elif "hmonnx_variants" in bucket:
            variant = bucket["hmonnx_variants"][self.lstm_variant]
            if variant["status"] != "ok":
                raise RuntimeError(
                    f"{self.lstm_variant} HMONNX is unavailable for {role}/{key}: "
                    f"{variant.get('error', 'unknown error')}"
                )
            runner = HmonnxRunner(self.work_dir / variant["hmonnx_file"], device=self.device)
        else:
            path = bucket.get("hmonnx_file")
            if not path:
                raise RuntimeError(f"HMONNX is unavailable for {role}/{key}")
            runner = HmonnxRunner(self.work_dir / path, device=self.device)
        self._runners[cache_key] = runner
        return runner


__all__ = ["BUCKETED_PRECISION_SPLIT_GRAPH_MODE", "KokoroBucketedRuntime"]
