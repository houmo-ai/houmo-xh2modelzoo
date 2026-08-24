from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .graph import GRAPH_ROLES
from .host import (
    WAVEFORM_SAMPLES_PER_FRAME,
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


class Runner(Protocol):
    input_names: list[str]

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


class OrtRunner:
    def __init__(self, model: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [value.name for value in self.session.get_inputs()]
        self.output_names = [value.name for value in self.session.get_outputs()]

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {name: feed[name] for name in self.input_names})
        return dict(zip(self.output_names, outputs, strict=True))


class HmonnxRunner:
    def __init__(self, model: str | Path, device: str = "cuda:0") -> None:
        import onnx

        from xhquant.api import HMONNXInference

        proto = onnx.load(str(model), load_external_data=False)
        self.input_names = [value.name for value in proto.graph.input]
        self.output_names = [value.name for value in proto.graph.output]
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.session = HMONNXInference(str(model))
        self.session.to(self.device)

    @torch.no_grad()
    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs = [_to_hmonnx_tensor(feed[name]).to(self.device) for name in self.input_names]
        outputs = self.session(*inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return {
            name: value.detach().float().cpu().numpy() for name, value in zip(self.output_names, outputs, strict=True)
        }


class KokoroStaticRuntime:
    """Batch-1 scheduler for the 14 static Kokoro accelerator graphs.

    Tokenization/G2P, sequence reversal, duration expansion, recurrent state
    scheduling, source randomness, STFT and iSTFT intentionally stay on Host.
    """

    def __init__(
        self,
        runners: dict[str, Runner],
        *,
        text_max_length: int,
        frame_max_length: int,
        lstm_chunk_length: int,
        seed: int = 1234,
    ) -> None:
        missing = sorted(set(GRAPH_ROLES).difference(runners))
        if missing:
            raise ValueError(f"missing Kokoro graph runners: {missing}")
        if frame_max_length <= 0 or lstm_chunk_length <= 0:
            raise ValueError("frame_max_length and lstm_chunk_length must be positive")
        if frame_max_length % lstm_chunk_length:
            raise ValueError("frame_max_length must be divisible by lstm_chunk_length")
        self.runners = runners
        self.text_max_length = int(text_max_length)
        self.frame_max_length = int(frame_max_length)
        self.lstm_chunk_length = int(lstm_chunk_length)
        self.seed = int(seed)

    @classmethod
    def from_export(
        cls,
        work_dir: str | Path,
        *,
        backend: str = "ort",
        device: str = "cuda:0",
        ort_fallback_roles: Iterable[str] = (),
    ) -> "KokoroStaticRuntime":
        root = Path(work_dir).expanduser().resolve()
        meta = json.loads((root / "export_meta_info.json").read_text(encoding="utf-8"))
        if backend not in {"ort", "hmonnx"}:
            raise ValueError("backend must be 'ort' or 'hmonnx'")
        if isinstance(ort_fallback_roles, str):
            raise TypeError("ort_fallback_roles must be an iterable of graph role names")
        ort_fallback = set(ort_fallback_roles)
        unknown = sorted(ort_fallback.difference(GRAPH_ROLES))
        if unknown:
            raise ValueError(f"unknown Kokoro ORT fallback roles: {unknown}")
        runners: dict[str, Runner] = {}
        for role in GRAPH_ROLES:
            component = meta["components"][role]
            if backend == "ort" or role in ort_fallback:
                runners[role] = OrtRunner(root / component["onnx_file"])
            else:
                hmonnx_file = component.get("hmonnx_file")
                if not hmonnx_file:
                    raise ValueError(f"component {role} has no HMONNX artifact")
                runners[role] = HmonnxRunner(root / hmonnx_file, device=device)
        return cls(
            runners,
            text_max_length=int(meta["text_max_length"]),
            frame_max_length=int(meta["frame_max_length"]),
            lstm_chunk_length=int(meta["lstm_chunk_length"]),
            seed=int(meta["seed"]),
        )

    def synthesize(
        self,
        token_ids: np.ndarray | list[int] | tuple[int, ...],
        style: np.ndarray | torch.Tensor,
        *,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        if not 0 < tokens.size <= self.text_max_length:
            raise ValueError(f"token length must be in [1,{self.text_max_length}]")
        style_array = np.asarray(
            style.detach().cpu().numpy() if isinstance(style, torch.Tensor) else style,
            dtype=np.float32,
        )
        if style_array.shape == (256,):
            style_array = style_array[None]
        if style_array.shape != (1, 256):
            raise ValueError(f"style must have shape [1,256], got {style_array.shape}")
        if speed <= 0:
            raise ValueError("speed must be positive")

        padded = np.zeros((1, self.text_max_length), dtype=np.int32)
        padded[0, : tokens.size] = tokens
        valid_len = torch.tensor([tokens.size], dtype=torch.int32)
        front = self._run(
            "front_base",
            {"input_ids": padded, "valid_len": valid_len.numpy()},
        )

        duration_base = _torch(front["duration_base"])
        text_features = _torch(front["text_features"])
        prosody_style = _torch(style_array[:, 128:])
        values = duration_base.permute(2, 0, 1)
        expanded_style = prosody_style.unsqueeze(0).expand(values.shape[0], -1, -1)
        values = torch.cat([values, expanded_style], dim=-1).transpose(0, 1)
        reverse_idx = make_reverse_idx(self.text_max_length, valid_len)
        text_mask = make_text_mask(self.text_max_length, valid_len)

        for index in range(3):
            forward, backward = prepare_lstm_inputs(values, valid_len)
            result = self._run(
                f"duration_block{index}",
                {
                    "forward_input": _numpy(forward),
                    "backward_input": _numpy(backward),
                    "style": _numpy(prosody_style),
                    "reverse_idx": _numpy(reverse_idx),
                    "mask": _numpy(text_mask),
                },
            )
            values = _torch(result["duration_features"])
        duration_features = values

        forward, backward = prepare_lstm_inputs(duration_features, valid_len)
        duration_result = self._run(
            "duration_predictor",
            {
                "forward_input": _numpy(forward),
                "backward_input": _numpy(backward),
                "reverse_idx": _numpy(reverse_idx),
                "mask": _numpy(text_mask),
            },
        )
        duration = duration_from_logits(
            _torch(duration_result["duration_logits"]),
            torch.tensor([speed], dtype=torch.float32),
            valid_len,
        )

        forward, backward = prepare_lstm_inputs(text_features.transpose(1, 2), valid_len)
        text_result = self._run(
            "text_lstm",
            {
                "forward_input": _numpy(forward),
                "backward_input": _numpy(backward),
            },
        )
        text_encoded = restore_bidirectional_outputs(
            _torch(text_result["forward_output"]),
            _torch(text_result["backward_output"]),
            valid_len,
        ).transpose(1, 2)

        alignment, valid_frames = duration_to_alignment(
            duration,
            self.frame_max_length,
            valid_len,
        )
        mask_f, mask_2f, mask_20f, mask_wave = make_frame_masks(
            valid_frames,
            self.frame_max_length,
        )
        aligned = self._run(
            "align_core",
            {
                "duration_features": _numpy(duration_features),
                "text_features": _numpy(text_encoded),
                "alignment": _numpy(alignment),
                "style": style_array,
            },
        )
        encoded = _torch(aligned["encoded"])
        asr = _torch(aligned["asr"])
        decoder_style = _torch(aligned["decoder_style"])
        prosody_style = _torch(aligned["prosody_style"])
        shared_forward, shared_backward = prepare_shared_lstm_inputs(encoded, valid_frames)
        shared_parts = [
            self._run_chunked_lstm("shared_fwd_lstm", shared_forward),
            self._run_chunked_lstm("shared_bwd_lstm", shared_backward),
        ]
        shared = restore_bidirectional_outputs(
            shared_parts[0],
            shared_parts[1],
            valid_frames,
        ).transpose(1, 2)

        branch_feed = {
            "shared": _numpy(shared),
            "style": _numpy(prosody_style),
            "mask_f": _numpy(mask_f),
            "mask_2f": _numpy(mask_2f),
        }
        f0_feed = {
            **branch_feed,
            "norm_scales": _numpy(make_rmsnorm_scales(valid_frames, self.frame_max_length)),
        }
        f0 = _torch(self._run("f0_branch", f0_feed)["f0"])
        noise = _torch(self._run("noise_branch", branch_feed)["noise"])
        decoded = self._run(
            "decoder",
            {
                "asr": _numpy(asr),
                "f0": _numpy(f0),
                "noise": _numpy(noise),
                "style": _numpy(decoder_style),
                "mask_f": _numpy(mask_f),
                "mask_2f": _numpy(mask_2f),
            },
        )
        sine_wavs = build_sine_wavs(
            f0,
            valid_frames,
            self.frame_max_length,
            seed=self.seed if seed is None else int(seed),
        )
        source = self._run("source_merge", {"sine_wavs": _numpy(sine_wavs)})
        harmonic = harmonic_spectrogram(
            _torch(source["harmonic_source"]),
            valid_frames,
            self.frame_max_length,
        )
        generated = self._run(
            "generator",
            {
                "generator_feature": decoded["generator_feature"],
                "style": _numpy(decoder_style),
                "harmonic": _numpy(harmonic),
                "mask_2f": _numpy(mask_2f),
                "mask_20f": _numpy(mask_20f),
                "mask_wave": _numpy(mask_wave),
            },
        )
        waveform = istft_waveform(_torch(generated["spec_phase"]), valid_frames)
        valid_frame_count = int(valid_frames.item())
        samples = valid_frame_count * WAVEFORM_SAMPLES_PER_FRAME
        return (
            waveform.reshape(-1)[:samples].numpy().astype(np.float32, copy=False),
            {
                "text_length": int(tokens.size),
                "frame_length": valid_frame_count,
                "valid_samples": samples,
                "duration": duration[0, : tokens.size].tolist(),
                "seed": self.seed if seed is None else int(seed),
            },
        )

    def _run_chunked_lstm(self, role: str, values: torch.Tensor) -> torch.Tensor:
        hidden = np.zeros((1, 1, 256), dtype=np.float32)
        cell = np.zeros_like(hidden)
        chunks = []
        for start in range(0, self.frame_max_length, self.lstm_chunk_length):
            result = self._run(
                role,
                {
                    "values": _numpy(values[:, start : start + self.lstm_chunk_length]),
                    "h0": hidden,
                    "c0": cell,
                },
            )
            chunks.append(_torch(result["output"]))
            hidden = np.asarray(result["hidden"], dtype=np.float32)
            cell = np.asarray(result["cell"], dtype=np.float32)
        return torch.cat(chunks, dim=1)

    def _run(self, role: str, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.runners[role].run({name: np.ascontiguousarray(value) for name, value in feed.items()})


def _torch(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().contiguous()
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).contiguous()


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_hmonnx_tensor(value: np.ndarray) -> torch.Tensor:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return torch.from_numpy(array.astype(np.int32, copy=False))
    return torch.from_numpy(array.astype(np.float16, copy=False))


__all__ = [
    "HmonnxRunner",
    "KokoroStaticRuntime",
    "OrtRunner",
    "Runner",
]
