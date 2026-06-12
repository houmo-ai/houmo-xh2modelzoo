from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..builder import MODELS
from ..common.hmonnx_model import HMONNXModel


@dataclass
class Qwen3TTSDecoderState:
    """Runtime state for the HMONNX-friendly Qwen3-TTS stateful decoder."""

    pre_conv_history: Tensor
    latent_buffer: Tensor
    conv_history: Tensor
    kv_cache: List[Tensor] = field(default_factory=list)
    kv_valid_len: int = 0
    skip_samples: int = 0
    latent_audio: Optional[Tensor] = None


@MODELS.register_module()
class Qwen3TTSStatefulDecoderInference(HMONNXModel):
    """Static-shape HMONNX wrapper for Qwen3-TTS stateful decoder.

    HMONNX uses fixed input shapes for all state buffers. The runtime keeps the
    logical cache length in ``kv_valid_len`` and pads codec chunks to
    ``chunk_size`` before invoking the graph.
    """

    SAMPLES_PER_FRAME = 1920

    def __init__(
        self,
        model_cfg: Optional[str] = None,
        hmonnx_file: Optional[str] = None,
        num_layers: int = 8,
        num_heads: int = 16,
        head_dim: int = 64,
        kv_cache_window: int = 72,
        chunk_size: int = 12,
        samples_per_frame: int = SAMPLES_PER_FRAME,
        initial_output_skip_frames: int = 4,
    ) -> None:
        if model_cfg is not None:
            hmonnx_file, meta = self._load_from_meta(model_cfg, hmonnx_file)
            num_layers = int(meta.get("stateful_num_layers", meta.get("num_layers", num_layers)))
            num_heads = int(meta.get("stateful_num_heads", meta.get("num_heads", num_heads)))
            head_dim = int(meta.get("stateful_head_dim", meta.get("head_dim", head_dim)))
            kv_cache_window = int(meta.get("stateful_kv_cache_window", meta.get("kv_cache_window", kv_cache_window)))
            chunk_size = int(meta.get("stateful_chunk_size", meta.get("chunk_size", chunk_size)))
            samples_per_frame = int(meta.get("stateful_samples_per_frame", meta.get("samples_per_frame", samples_per_frame)))
            initial_output_skip_frames = int(meta.get("stateful_initial_output_skip_frames", initial_output_skip_frames))
        if hmonnx_file is None:
            raise ValueError("either model_cfg with stateful_hmonnx or hmonnx_file is required")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_cache_window = kv_cache_window
        self.chunk_size = max(1, int(chunk_size))
        self.samples_per_frame = samples_per_frame
        self.initial_output_skip_frames = max(0, int(initial_output_skip_frames))
        super().__init__(str(hmonnx_file))

    @staticmethod
    def _load_from_meta(model_cfg: str, hmonnx_file: Optional[str]) -> Tuple[str, dict]:
        meta_path = Path(model_cfg)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if hmonnx_file is not None:
            return hmonnx_file, meta
        model_dir = meta_path.parent
        for key in ("stateful_hmonnx", "hmonnx_stateful", "stateful_decoder_hmonnx"):
            if key in meta:
                return str(model_dir / meta[key]), meta
        raise KeyError(
            "model_cfg does not contain a stateful decoder path; expected one of "
            "stateful_hmonnx, hmonnx_stateful, stateful_decoder_hmonnx"
        )

    def initialize(self) -> None:
        if self.session is not None and hasattr(self.session, "initialize"):
            self.session.initialize()

    def create_state(
        self,
        history_num: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float16,
    ) -> Qwen3TTSDecoderState:
        if device is None:
            device = getattr(self, "device", torch.device("cpu"))
        kv_valid_len = max(0, min(int(history_num), self.kv_cache_window))
        pre_conv_history = torch.zeros(1, 512, 2, device=device, dtype=dtype)
        latent_buffer = torch.zeros(1, 1024, 4, device=device, dtype=dtype)
        conv_history = torch.zeros(1, 1024, 4, device=device, dtype=dtype)
        kv_cache: List[Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(torch.zeros(1, self.num_heads, self.kv_cache_window, self.head_dim, device=device, dtype=dtype))
            kv_cache.append(torch.zeros(1, self.num_heads, self.kv_cache_window, self.head_dim, device=device, dtype=dtype))
        return Qwen3TTSDecoderState(
            pre_conv_history=pre_conv_history,
            latent_buffer=latent_buffer,
            conv_history=conv_history,
            kv_cache=kv_cache,
            kv_valid_len=kv_valid_len,
        )

    def decode(
        self,
        audio_codes: Union[Tensor, list, tuple],
        state: Optional[Qwen3TTSDecoderState] = None,
        is_final: bool = False,
    ) -> Tuple[Tensor, Qwen3TTSDecoderState]:
        audio_codes = self._normalize_audio_codes(audio_codes)
        n_frames = int(audio_codes.shape[1])
        if n_frames <= self.chunk_size:
            return self._decode(audio_codes, state=state, is_final=is_final)

        pieces: List[Tensor] = []
        next_state = state
        for start in range(0, n_frames, self.chunk_size):
            end = min(start + self.chunk_size, n_frames)
            chunk_is_final = is_final and end >= n_frames
            piece, next_state = self._decode(audio_codes[:, start:end, :], state=next_state, is_final=chunk_is_final)
            if piece.numel() > 0:
                pieces.append(piece)
        if not pieces:
            device = audio_codes.device
            return torch.empty(0, device=device, dtype=torch.float32), next_state or self.create_state(device=device)
        return torch.cat(pieces, dim=0), next_state

    def _decode(
        self,
        audio_codes: Tensor,
        state: Optional[Qwen3TTSDecoderState] = None,
        is_final: bool = False,
    ) -> Tuple[Tensor, Qwen3TTSDecoderState]:
        if state is None:
            state = self.create_state(device=audio_codes.device)
        skip_counter = int(state.skip_samples)

        valid_frames = int(audio_codes.shape[1])
        if valid_frames == 0:
            if is_final and state.latent_audio is not None:
                audio = state.latent_audio.to(torch.float32)
                state.latent_audio = None
                return audio, state
            return torch.empty(0, device=audio_codes.device, dtype=torch.float32), state
        if valid_frames > self.chunk_size:
            raise ValueError(f"expected at most {self.chunk_size} codec frames, got {valid_frames}")
        if not is_final and valid_frames < self.chunk_size:
            raise ValueError(
                f"non-final chunks must contain exactly {self.chunk_size} codec frames, got {valid_frames}"
            )

        padded_codes = self._pad_to_chunk(audio_codes, self.chunk_size)
        device = state.pre_conv_history.device
        is_last = torch.tensor([1.0 if is_final else 0.0], device=device, dtype=torch.float16)
        kv_valid_len = torch.tensor([state.kv_valid_len], device=device, dtype=torch.int32)
        valid_frames_tensor = torch.tensor([valid_frames], device=device, dtype=torch.int32)
        keys = [state.kv_cache[2 * i] for i in range(self.num_layers)]
        values = [state.kv_cache[2 * i + 1] for i in range(self.num_layers)]
        outputs = super().forward(
            padded_codes.to(device, dtype=torch.int32),
            state.pre_conv_history,
            state.latent_buffer,
            state.conv_history,
            is_last,
            kv_valid_len,
            valid_frames_tensor,
            *keys,
            *values,
        )
        if not isinstance(outputs, (tuple, list)):
            raise RuntimeError("stateful decoder HMONNX must return multiple outputs")

        final_wav = outputs[0]
        valid_samples = int(outputs[1].reshape(-1)[0].item())
        next_state = self._build_state_from_outputs(outputs, state.kv_valid_len, valid_frames)

        initial_skip = 0
        if int(state.kv_valid_len) == 0:
            initial_skip = self.initial_output_skip_frames * self.samples_per_frame
        audio_start = int(initial_skip)
        audio_end = audio_start + int(valid_samples)

        if is_final:
            audio = final_wav[0, audio_start:audio_end] if valid_samples > 0 else torch.empty(0, device=final_wav.device, dtype=final_wav.dtype)
            next_state.latent_audio = None
        elif valid_samples > 0:
            audio = final_wav[0, audio_start:audio_end]
            next_state.latent_audio = final_wav[0, audio_end:]
        else:
            audio = torch.empty(0, device=final_wav.device, dtype=final_wav.dtype)
            next_state.latent_audio = final_wav[0, audio_start:]

        if skip_counter > 0 and audio.numel() > 0:
            if audio.numel() <= skip_counter:
                skip_counter -= int(audio.numel())
                audio = torch.empty(0, device=audio.device, dtype=audio.dtype)
            else:
                audio = audio[skip_counter:]
                skip_counter = 0
        next_state.skip_samples = 4 * self.samples_per_frame if is_final else skip_counter
        return audio.to(torch.float32), next_state

    def _build_state_from_outputs(self, outputs, prev_kv_valid_len: int, valid_frames: int) -> Qwen3TTSDecoderState:
        key_start = 5
        value_start = key_start + self.num_layers
        kv_cache: List[Tensor] = []
        for i in range(self.num_layers):
            kv_cache.append(outputs[key_start + i])
            kv_cache.append(outputs[value_start + i])
        return Qwen3TTSDecoderState(
            pre_conv_history=outputs[2],
            latent_buffer=outputs[3],
            conv_history=outputs[4],
            kv_cache=kv_cache,
            kv_valid_len=min(self.kv_cache_window, int(prev_kv_valid_len) + int(valid_frames)),
        )

    @staticmethod
    def _pad_to_chunk(audio_codes: Tensor, chunk_size: int) -> Tensor:
        pad_frames = chunk_size - int(audio_codes.shape[1])
        if pad_frames <= 0:
            return audio_codes
        pad = torch.zeros(audio_codes.shape[0], pad_frames, audio_codes.shape[2], device=audio_codes.device, dtype=audio_codes.dtype)
        return torch.cat([audio_codes, pad], dim=1)

    def _normalize_audio_codes(self, audio_codes: Union[Tensor, list, tuple]) -> Tensor:
        if torch.is_tensor(audio_codes):
            codes = audio_codes.detach()
        else:
            codes = torch.as_tensor(audio_codes)
        if codes.dim() == 1:
            if codes.numel() != 16:
                raise ValueError(f"expected one codec frame with width 16, got shape {tuple(codes.shape)}")
            codes = codes.reshape(1, 1, 16)
        elif codes.dim() == 2:
            if codes.shape[-1] != 16:
                raise ValueError(f"expected codec frames [N, 16], got shape {tuple(codes.shape)}")
            codes = codes.unsqueeze(0)
        elif codes.dim() == 3:
            if codes.shape[-1] == 16:
                pass
            elif codes.shape[1] == 16:
                codes = codes.transpose(1, 2)
            else:
                raise ValueError(f"expected [B, N, 16] or [B, 16, N], got shape {tuple(codes.shape)}")
        else:
            raise ValueError(f"unsupported codec code rank: {codes.dim()}")
        return codes.to(dtype=torch.int32)


__all__ = ["Qwen3TTSDecoderState", "Qwen3TTSStatefulDecoderInference"]
