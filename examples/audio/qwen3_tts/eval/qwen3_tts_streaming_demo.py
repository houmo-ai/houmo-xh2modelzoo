#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS GGUF-aligned streaming inference example (HMONNX / XH2a)

True GGUF alignment in this demo means:
    talker/predictor stream complete [16] codec frames -> stateful decoder session -> AUDIO/FINISH responses.

The live path intentionally mirrors HaujetZhao/Qwen3-TTS-GGUF:
    DecoderState(pre_conv_history, latent_buffer, conv_history, kv_cache, skip_samples, latent_audio)
    DecoderResponse(task_id, msg_type, index, audio, compute_time, state, recv_time)
    DecodeResult(responses)
    Timing(chunk_gen_times, decoder_compute_times, ...)

Supported Qwen3-TTS variants:
    customvoice  -> 0.6B CustomVoice, predefined speaker
    base         -> 0.6B Base, voice clone with reference audio/text
    voicedesign  -> 1.7B VoiceDesign, natural-language voice instruction

Important:
    --mode live is strict. It requires a stateful decoder meta.json/HMONNX/ONNX exported with GGUF-compatible inputs/outputs.
    The existing HMONNX speech_tokenizer export in this repo is stateless (codes -> wav), so it cannot provide
    GGUF-equivalent streaming without a stateful decoder export.

Usage:
    PYTHONPATH=/path/to/xh2modelzoo \
    python eval/qwen3_tts_streaming_demo.py \
        --config ./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py \
        --variant customvoice \
        --stateful-decoder work_dirs/qwen3_tts_stateful_decoder_xh2a/meta.json \
        --text "Hello, this is a streaming TTS test." \
        --speaker vivian --mode live --chunk-size 12
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

QWEN3_TTS_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(QWEN3_TTS_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN3_TTS_EXAMPLE_ROOT))

from xhquant.api import Config, set_random_seed
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_tts import Qwen3TTSHMONNXInference
from xh_model_zoo.xh_llm.models.qwen3_tts.qwen3_tts_stateful_decoder import (
    Qwen3TTSDecoderState as HMONNXDecoderState,
    Qwen3TTSStatefulDecoderInference,
)
from qwen3_tts_demo import DEFAULT_FRONTEND_HMONNX_DIR, VoiceCloneFrontendHMONNX


CODE_RATE_HZ = 12
SAMPLE_RATE = 24000
DEFAULT_VARIANT = "0_6B_customvoice"
VARIANT_ALIASES = {
    "customvoice": "0_6B_customvoice",
    "custom_voice": "0_6B_customvoice",
    "custom-voice": "0_6B_customvoice",
    "cv": "0_6B_customvoice",
    "0_6b_customvoice": "0_6B_customvoice",
    "0_6B_customvoice": "0_6B_customvoice",
    "base": "0_6B_base",
    "voiceclone": "0_6B_base",
    "voice_clone": "0_6B_base",
    "voice-clone": "0_6B_base",
    "0_6b_base": "0_6B_base",
    "0_6B_base": "0_6B_base",
    "voicedesign": "1_7B_voicedesign",
    "voice_design": "1_7B_voicedesign",
    "voice-design": "1_7B_voicedesign",
    "vd": "1_7B_voicedesign",
    "1_7b_voicedesign": "1_7B_voicedesign",
    "1_7B_voicedesign": "1_7B_voicedesign",
}


def _normalize_variant(variant: str) -> str:
    key = str(variant or DEFAULT_VARIANT).strip()
    return VARIANT_ALIASES.get(key, VARIANT_ALIASES.get(key.lower(), key))


def _normalize_tts_mode(mode: str) -> str:
    aliases = {
        "customvoice": "custom_voice",
        "custom-voice": "custom_voice",
        "custom_voice": "custom_voice",
        "voiceclone": "voice_clone",
        "voice-clone": "voice_clone",
        "voice_clone": "voice_clone",
        "base": "voice_clone",
        "voicedesign": "voice_design",
        "voice-design": "voice_design",
        "voice_design": "voice_design",
    }
    key = str(mode).strip().lower()
    if key not in aliases:
        raise ValueError(f"unsupported tts mode: {mode}")
    return aliases[key]


@dataclass
class TTSRequest:
    tts_mode: str
    text: str
    language: str
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None

    def as_kwargs(self):
        return dict(
            tts_mode=self.tts_mode,
            text=self.text,
            language=self.language,
            speaker=self.speaker,
            instruct=self.instruct,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
        )


def _resolve_tts_request(args: argparse.Namespace, cfg: Config) -> TTSRequest:
    tts_mode = _normalize_tts_mode(getattr(cfg, "tts_mode", "custom_voice"))
    text = args.text or getattr(cfg, "tts_text", "")
    language = args.language or "Chinese"
    speaker = args.speaker if args.speaker is not None else getattr(cfg, "tts_speaker", None)
    instruct = args.instruct if args.instruct is not None else getattr(cfg, "tts_instruct", None)
    ref_audio = args.ref_audio if args.ref_audio is not None else getattr(cfg, "ref_audio", None)
    ref_text = args.ref_text if args.ref_text is not None else getattr(cfg, "ref_text", None)

    if tts_mode == "custom_voice" and not speaker:
        speaker = "vivian"
    if tts_mode == "voice_design" and not instruct:
        raise ValueError("--instruct is required for voicedesign when the variant config does not provide one")
    if tts_mode == "voice_clone":
        if not ref_audio:
            raise ValueError("--ref-audio is required for base/voice-clone when the variant config does not provide one")
        if not Path(ref_audio).exists():
            raise FileNotFoundError(f"reference audio not found for base/voice-clone mode: {ref_audio}")
        if ref_text is None:
            ref_text = ""

    return TTSRequest(
        tts_mode=tts_mode,
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )


@dataclass
class DecoderState:
    """GGUF-compatible decoder state."""

    pre_conv_history: Optional[np.ndarray] = None
    latent_buffer: Optional[np.ndarray] = None
    conv_history: Optional[np.ndarray] = None
    kv_cache: List[np.ndarray] = field(default_factory=list)
    kv_valid_len: int = 0
    skip_samples: int = 0
    latent_audio: Optional[np.ndarray] = None


@dataclass
class DecoderSession:
    """GGUF-compatible per-task decoder session."""

    state: Optional[object] = None
    index: int = 0


@dataclass
class DecoderResponse:
    """GGUF-compatible decoder response packet."""

    task_id: str
    msg_type: str = "AUDIO"  # AUDIO, FINISH, READY, ERROR
    index: int = 0
    audio: Optional[np.ndarray] = None
    compute_time: float = 0.0
    state: Optional[object] = None
    recv_time: float = 0.0


@dataclass
class DecodeResult:
    """GGUF-compatible wrapper for decoder responses."""

    responses: List[DecoderResponse] = field(default_factory=list)

    @property
    def audio(self) -> Optional[np.ndarray]:
        pcms = [r.audio for r in self.responses if r.msg_type == "AUDIO" and r.audio is not None]
        return np.concatenate(pcms) if pcms else None

    @property
    def chunk_compute_times(self) -> List[float]:
        return [r.compute_time for r in self.responses if r.msg_type == "AUDIO"]

    @property
    def total_compute_time(self) -> float:
        return sum(self.chunk_compute_times)

    @property
    def first_response_time(self) -> float:
        valid = [r for r in self.responses if r.msg_type == "AUDIO"]
        return valid[0].recv_time if valid else 0.0

    @property
    def final_state(self) -> Optional[object]:
        for r in reversed(self.responses):
            if r.state is not None:
                return r.state
        return None


@dataclass
class Timing:
    """GGUF-compatible timing fields."""

    prompt_time: float = 0.0
    prefill_time: float = 0.0
    talker_loop_times: List[float] = field(default_factory=list)
    predictor_loop_times: List[float] = field(default_factory=list)
    chunk_gen_times: List[float] = field(default_factory=list)
    decoder_compute_times: List[float] = field(default_factory=list)
    total_steps: int = 0

    @property
    def first_decode_latency(self) -> float:
        return self.decoder_compute_times[0] if self.decoder_compute_times else 0.0

    @property
    def first_chunk_latency(self) -> float:
        return self.chunk_gen_times[0] if self.chunk_gen_times else 0.0

    @property
    def first_audio_latency(self) -> float:
        if self.chunk_gen_times and self.decoder_compute_times:
            return self.first_chunk_latency + self.first_decode_latency
        return 0.0

    @property
    def total_talker_time(self) -> float:
        return sum(self.talker_loop_times)

    @property
    def total_predictor_time(self) -> float:
        return sum(self.predictor_loop_times)

    @property
    def total_decoder_time(self) -> float:
        return sum(self.decoder_compute_times)

    @property
    def total_inference_time(self) -> float:
        return (
            self.prompt_time
            + self.prefill_time
            + self.total_talker_time
            + self.total_predictor_time
            + self.total_decoder_time
        )

    @property
    def inference_only_time(self) -> float:
        return self.prompt_time + self.prefill_time + self.total_talker_time + self.total_predictor_time


@dataclass
class LoopOutput:
    """GGUF-compatible loop output shell for callers that want metadata."""

    all_codes: List[np.ndarray]
    summed_embeds: List[np.ndarray]
    timing: Timing
    decode_result: Optional[DecodeResult] = None


class GGUFStatefulDecoder:
    """ONNX Runtime stateful decoder using GGUF's decoder protocol.

    Static HMONNX-friendly exports use fixed cache buffers plus ``kv_valid_len``
    and ``valid_frames``. Older dynamic ONNX exports are still accepted as a
    fallback, but the live HMONNX path should use the static-buffer layout.
    """

    NUM_LAYERS = 8
    NUM_HEADS = 16
    HEAD_DIM = 64
    KV_CACHE_WINDOW = 72
    SAMPLES_PER_FRAME = 1920
    INITIAL_OUTPUT_SKIP_FRAMES = 4

    def __init__(self, onnx_path: str, onnx_provider: str = "CUDA", chunk_size: int = 12):
        import onnxruntime as ort

        self.chunk_size = int(chunk_size)
        self.onnx_provider = onnx_provider.upper()
        self.onnx_path = str(onnx_path)
        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"stateful decoder ONNX not found: {self.onnx_path}")

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if self.onnx_provider in ("TRT", "TENSORRT") and "TensorrtExecutionProvider" in available:
            providers.insert(
                0,
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(Path(self.onnx_path).parent / "trt_cache"),
                    },
                ),
            )
        elif self.onnx_provider == "DML" and "DmlExecutionProvider" in available:
            providers.insert(0, "DmlExecutionProvider")
        elif self.onnx_provider == "CUDA" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(self.onnx_path, sess_options=sess_opts, providers=providers)
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.input_types = {i.name: i.type for i in self.sess.get_inputs()}
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.active_provider = self.sess.get_providers()[0]
        self.static_buffers = {"kv_valid_len", "valid_frames"}.issubset(self.input_names)

        float_input = next((i for i in self.sess.get_inputs() if "float" in i.type), None)
        self.dtype = np.float16 if float_input is not None and "float16" in float_input.type else np.float32
        self.code_dtype = np.int64 if "int64" in self.input_types.get("audio_codes", "") else np.int32
        self._validate_io()

        self.decode(
            np.zeros((self.chunk_size, 16), dtype=self.code_dtype),
            state=self.create_state(self.KV_CACHE_WINDOW),
            is_final=True,
        )

    def _validate_io(self) -> None:
        required = {"audio_codes", "is_last", "pre_conv_history", "latent_buffer", "conv_history"}
        for i in range(self.NUM_LAYERS):
            required.add(f"past_key_{i}")
            required.add(f"past_value_{i}")
        missing = sorted(required - self.input_names)
        if missing:
            raise RuntimeError(
                "stateful decoder is not GGUF-compatible; missing inputs: "
                + ", ".join(missing)
                + ". Export a stateful decoder first."
            )
        if len(self.output_names) < 5 + 2 * self.NUM_LAYERS:
            raise RuntimeError(
                f"stateful decoder is not GGUF-compatible; expected at least {5 + 2 * self.NUM_LAYERS} outputs, "
                f"got {len(self.output_names)}"
            )

    def create_state(self, history_num: int = 0) -> DecoderState:
        kv_valid_len = max(0, min(int(history_num), self.KV_CACHE_WINDOW))
        if self.static_buffers:
            pre_conv_history = np.zeros((1, 512, 2), dtype=self.dtype)
            latent_buffer = np.zeros((1, 1024, 4), dtype=self.dtype)
            conv_history = np.zeros((1, 1024, 4), dtype=self.dtype)
            kv_len = self.KV_CACHE_WINDOW
        else:
            if history_num != 0:
                pre_conv_history = np.zeros((1, 512, 2), dtype=self.dtype)
                latent_buffer = np.zeros((1, 1024, 4), dtype=self.dtype)
                conv_history = np.zeros((1, 1024, 4), dtype=self.dtype)
            else:
                pre_conv_history = np.zeros((1, 512, 0), dtype=self.dtype)
                latent_buffer = np.zeros((1, 1024, 0), dtype=self.dtype)
                conv_history = np.zeros((1, 1024, 0), dtype=self.dtype)
            kv_len = max(0, int(history_num))

        kv_cache = []
        for _ in range(self.NUM_LAYERS):
            kv_cache.append(np.zeros((1, self.NUM_HEADS, kv_len, self.HEAD_DIM), dtype=self.dtype))
            kv_cache.append(np.zeros((1, self.NUM_HEADS, kv_len, self.HEAD_DIM), dtype=self.dtype))
        return DecoderState(
            pre_conv_history=pre_conv_history,
            latent_buffer=latent_buffer,
            conv_history=conv_history,
            kv_cache=kv_cache,
            kv_valid_len=kv_valid_len,
            skip_samples=0,
            latent_audio=None,
        )

    def decode(
        self, audio_codes: np.ndarray, state: Optional[DecoderState] = None, is_final: bool = False
    ) -> Tuple[np.ndarray, DecoderState]:
        audio_codes = np.asarray(audio_codes)
        if audio_codes.ndim == 1:
            audio_codes = audio_codes.reshape(-1, 16)
        n_frames = audio_codes.shape[0] if audio_codes.ndim == 2 else audio_codes.shape[1]
        if n_frames <= self.chunk_size:
            return self._decode(audio_codes, state=state, is_final=is_final)

        full_audio = []
        curr_state = state
        for i in range(0, n_frames, self.chunk_size):
            chunk = audio_codes[i : i + self.chunk_size]
            is_last_chunk = i + self.chunk_size >= n_frames
            chunk_is_final = is_final if is_last_chunk else False
            chunk_audio, curr_state = self._decode(chunk, state=curr_state, is_final=chunk_is_final)
            full_audio.append(chunk_audio)
        return (np.concatenate(full_audio) if full_audio else np.array([], dtype=np.float32)), curr_state

    def _decode(
        self, audio_codes: np.ndarray, state: Optional[DecoderState] = None, is_final: bool = False
    ) -> Tuple[np.ndarray, DecoderState]:
        if state is None:
            state = self.create_state()
        skip_counter = state.skip_samples

        audio_codes = np.asarray(audio_codes)
        if audio_codes.ndim == 1:
            audio_codes = audio_codes.reshape(-1, 16)
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[np.newaxis, ...]
        n_frames = int(audio_codes.shape[1])

        if n_frames == 0:
            if is_final and state.latent_audio is not None:
                audio = state.latent_audio.astype(np.float32)
                state.latent_audio = None
                return audio, state
            return np.array([], dtype=np.float32), state

        valid_frames = n_frames
        if self.static_buffers:
            if valid_frames > self.chunk_size:
                raise ValueError(f"expected at most {self.chunk_size} codec frames, got {valid_frames}")
            if not is_final and valid_frames < self.chunk_size:
                raise ValueError(
                    f"non-final chunks must contain exactly {self.chunk_size} codec frames, got {valid_frames}"
                )
            audio_codes = self._pad_to_chunk(audio_codes, self.chunk_size)

        feed = {
            "audio_codes": audio_codes.astype(self.code_dtype, copy=False),
            "is_last": np.array([1.0 if is_final else 0.0], dtype=self.dtype),
            "pre_conv_history": state.pre_conv_history,
            "latent_buffer": state.latent_buffer,
            "conv_history": state.conv_history,
        }
        if self.static_buffers:
            feed["kv_valid_len"] = np.array([state.kv_valid_len], dtype=np.int32)
            feed["valid_frames"] = np.array([valid_frames], dtype=np.int32)
        for i in range(self.NUM_LAYERS):
            feed[f"past_key_{i}"] = state.kv_cache[2 * i]
            feed[f"past_value_{i}"] = state.kv_cache[2 * i + 1]

        outputs = self.sess.run(self.output_names, feed)
        final_wav = outputs[0]
        valid_samples = int(np.asarray(outputs[1]).reshape(-1)[0])
        new_state = self._build_state_from_outputs(outputs, state.kv_valid_len, valid_frames)

        initial_skip = 0
        if self.static_buffers and int(state.kv_valid_len) == 0:
            initial_skip = self.INITIAL_OUTPUT_SKIP_FRAMES * self.SAMPLES_PER_FRAME
        audio_start = int(initial_skip)
        audio_end = audio_start + int(valid_samples)

        if is_final:
            audio = final_wav[0, audio_start:audio_end] if valid_samples > 0 else np.array([], dtype=np.float32)
            new_state.latent_audio = None
        else:
            audio = final_wav[0, audio_start:audio_end] if valid_samples > 0 else np.array([], dtype=np.float32)
            new_state.latent_audio = final_wav[0, audio_end:]

        if skip_counter > 0 and len(audio) > 0:
            if len(audio) <= skip_counter:
                skip_counter -= len(audio)
                audio = np.array([], dtype=np.float32)
            else:
                audio = audio[skip_counter:]
                skip_counter = 0
        new_state.skip_samples = 4 * self.SAMPLES_PER_FRAME if is_final else skip_counter
        return audio.astype(np.float32), new_state

    def _build_state_from_outputs(self, outputs, prev_kv_valid_len: int, valid_frames: int) -> DecoderState:
        state = DecoderState(
            pre_conv_history=outputs[2],
            latent_buffer=outputs[3],
            conv_history=outputs[4],
            kv_cache=[],
            kv_valid_len=min(self.KV_CACHE_WINDOW, int(prev_kv_valid_len) + int(valid_frames)),
            skip_samples=0,
            latent_audio=None,
        )
        base_idx = 5
        for i in range(self.NUM_LAYERS):
            state.kv_cache.append(outputs[base_idx + i])
            state.kv_cache.append(outputs[base_idx + self.NUM_LAYERS + i])
        return state

    @staticmethod
    def _pad_to_chunk(audio_codes: np.ndarray, chunk_size: int) -> np.ndarray:
        pad_frames = int(chunk_size) - int(audio_codes.shape[1])
        if pad_frames <= 0:
            return audio_codes
        pad = np.zeros((audio_codes.shape[0], pad_frames, audio_codes.shape[2]), dtype=audio_codes.dtype)
        return np.concatenate([audio_codes, pad], axis=1)


class HMONNXGGUFStatefulDecoder:
    """HMONNX runtime adapter with the same public protocol as ``GGUFStatefulDecoder``."""

    def __init__(self, decoder_path: str, device: str = "cuda", chunk_size: int = 12) -> None:
        path = Path(decoder_path)
        if not path.exists():
            raise FileNotFoundError(f"stateful decoder path not found: {path}")
        if path.name == "meta.json":
            self.model = Qwen3TTSStatefulDecoderInference(model_cfg=str(path))
        else:
            self.model = Qwen3TTSStatefulDecoderInference(hmonnx_file=str(path), chunk_size=chunk_size)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.initialize()
        self.chunk_size = int(self.model.chunk_size)
        self.active_provider = f"HMONNX/{self.device.type}"
        self.dtype = np.float16
        self.decode(
            np.zeros((self.chunk_size, 16), dtype=np.int32),
            state=self.create_state(self.model.kv_cache_window),
            is_final=True,
        )

    def create_state(self, history_num: int = 0) -> HMONNXDecoderState:
        return self.model.create_state(history_num=history_num, device=self.device, dtype=torch.float16)

    def decode(
        self, audio_codes: np.ndarray, state: Optional[HMONNXDecoderState] = None, is_final: bool = False
    ) -> Tuple[np.ndarray, HMONNXDecoderState]:
        codes = torch.as_tensor(np.asarray(audio_codes), device=self.device, dtype=torch.int32)
        audio, next_state = self.model.decode(codes, state=state, is_final=is_final)
        return audio.detach().cpu().numpy().astype(np.float32), next_state


class StreamingQwen3TTS:
    """Streaming TTS wrapper with strict GGUF live mode."""

    def __init__(
        self,
        model: Qwen3TTSHMONNXInference,
        chunk_size: int = 12,
        stateful_decoder: Optional[object] = None,
    ):
        self.model = model
        self.chunk_size = int(chunk_size)
        self.stateful_decoder = stateful_decoder
        self.logger = get_root_logger()
        self._sr = SAMPLE_RATE
        self.task_counter = 0
        self.last_timing: Optional[Timing] = None
        self.last_decode_result: Optional[DecodeResult] = None
        self.last_loop_output: Optional[LoopOutput] = None

    @property
    def sample_rate(self) -> int:
        return int(getattr(self, "_sr", SAMPLE_RATE) or SAMPLE_RATE)

    @staticmethod
    def _normalize_codes(codes) -> torch.Tensor:
        if not torch.is_tensor(codes):
            codes = torch.as_tensor(np.asarray(codes), dtype=torch.long)
        else:
            codes = codes.detach().cpu().long()
        if codes.dim() == 3:
            codes = codes.reshape(-1, codes.shape[-1])
        if codes.dim() == 1:
            codes = codes.view(-1, 16)
        if codes.dim() != 2 or codes.shape[-1] != 16:
            raise ValueError(f"expected codec codes with shape [N, 16], got {tuple(codes.shape)}")
        return codes

    def _make_audio_response(
        self,
        task_id: str,
        index: int,
        audio: np.ndarray,
        compute_time: float,
        state: Optional[object] = None,
    ) -> DecoderResponse:
        return DecoderResponse(
            task_id=task_id,
            msg_type="AUDIO",
            index=index,
            audio=np.asarray(audio, dtype=np.float32).reshape(-1),
            compute_time=float(compute_time),
            state=state,
            recv_time=time.time(),
        )

    def generate_live_responses(
        self,
        tts_mode: str,
        text: str,
        language: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **gen_kwargs,
    ) -> Generator[DecoderResponse, None, None]:
        """Strict GGUF live path: stateful decoder only."""
        if self.stateful_decoder is None:
            raise RuntimeError(
                "--mode live is now strict GGUF alignment and requires --stateful-decoder. "
                "The bundled HMONNX speech_tokenizer is stateless and cannot provide GGUF-equivalent streaming."
            )

        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        session = DecoderSession()
        timing = Timing()
        decode_result = DecodeResult()
        all_codes: List[np.ndarray] = []
        total_frames = 0
        decoded_final_chunk = False
        last_chunk_time = time.time()

        self.logger.info(
            f"[live] strict GGUF streaming: task_id={task_id}, tts_mode={tts_mode}, text='{text}', "
            f"speaker={speaker}, instruct={bool(instruct)}, ref_audio={ref_audio or ''}, "
            f"chunk_size={self.chunk_size}, decoder_provider={self.stateful_decoder.active_provider}"
        )

        code_stream = self.model.generate_code_stream(
            mode=tts_mode,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            chunk_size=self.chunk_size,
            **gen_kwargs,
        )

        for codes in code_stream:
            now = time.time()
            timing.chunk_gen_times.append(now - last_chunk_time)
            last_chunk_time = now

            codes = self._normalize_codes(codes)
            if codes.shape[0] == 0:
                continue
            codes_np = codes.numpy().astype(np.int64)
            all_codes.append(codes_np)
            total_frames += int(codes_np.shape[0])

            # The code streamer emits full chunks as soon as they are ready. A short chunk can only be
            # the final residual chunk, matching GGUF's final decode call with is_final=True.
            is_final_chunk = codes_np.shape[0] < self.chunk_size
            decoded_final_chunk = decoded_final_chunk or is_final_chunk

            t_c = time.time()
            audio, session.state = self.stateful_decoder.decode(
                codes_np, state=session.state, is_final=is_final_chunk
            )
            decode_time = time.time() - t_c
            timing.decoder_compute_times.append(decode_time)

            response = self._make_audio_response(task_id, session.index, audio, decode_time)
            decode_result.responses.append(response)
            self.logger.info(
                f"[{task_id}] AUDIO index={response.index} frames={codes_np.shape[0]} "
                f"samples={len(response.audio)} ({len(response.audio)/self.sample_rate*1000:.0f}ms), "
                f"chunk_gen={timing.chunk_gen_times[-1]:.2f}s, decode={decode_time:.2f}s"
            )
            session.index += 1
            if response.audio is not None and len(response.audio) > 0:
                yield response

        # GGUF final packet: when the last emitted chunk was full-size, send an empty final decode to
        # flush latent audio. Short residual chunks have already been decoded with is_final=True above.
        if not decoded_final_chunk:
            t_c = time.time()
            final_audio, session.state = self.stateful_decoder.decode(
                np.zeros((0, 16), dtype=np.int64), state=session.state, is_final=True
            )
            final_decode_time = time.time() - t_c
            if len(final_audio) > 0:
                timing.decoder_compute_times.append(final_decode_time)
                response = self._make_audio_response(task_id, session.index, final_audio, final_decode_time)
                decode_result.responses.append(response)
                self.logger.info(
                    f"[{task_id}] AUDIO index={response.index} samples={len(response.audio)} "
                    f"({len(response.audio)/self.sample_rate*1000:.0f}ms), final state flush"
                )
                session.index += 1
                yield response

        finish = DecoderResponse(
            task_id=task_id,
            msg_type="FINISH",
            index=session.index,
            state=session.state,
            recv_time=time.time(),
        )
        decode_result.responses.append(finish)
        self.logger.info(f"[{task_id}] FINISH chunks={session.index}, total_frames={total_frames}")
        yield finish

        self.last_timing = timing
        self.last_decode_result = decode_result
        self.last_loop_output = LoopOutput(
            all_codes=all_codes,
            summed_embeds=[],
            timing=timing,
            decode_result=decode_result,
        )

    def generate_live_streaming(
        self,
        tts_mode: str,
        text: str,
        language: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **gen_kwargs,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        for response in self.generate_live_responses(
            tts_mode=tts_mode,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **gen_kwargs,
        ):
            if response.msg_type == "AUDIO" and response.audio is not None and len(response.audio) > 0:
                yield response.audio.astype(np.float32), self.sample_rate

    def generate_oneshot(
        self,
        tts_mode: str,
        text: str,
        language: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **gen_kwargs,
    ) -> Tuple[np.ndarray, int]:
        wavs, sr = self.model.generate_by_mode(
            mode=tts_mode,
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            **gen_kwargs,
        )
        wav = wavs[0]
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        self._sr = int(sr)
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def _write_and_report(
    logger,
    output_file: Path,
    all_chunks: List[np.ndarray],
    sr: int,
    start_time: float,
    first_chunk_latency: Optional[float],
    mode: str,
    timing: Optional[Timing] = None,
    decode_result: Optional[DecodeResult] = None,
) -> None:
    total_time = time.time() - start_time
    if not all_chunks:
        logger.error("no audio chunk was generated.")
        return

    full_audio = np.concatenate(all_chunks)
    sf.write(output_file, full_audio, sr)

    audio_dur = len(full_audio) / sr
    logger.info(f"[{mode}] streaming generation done!")
    logger.info(f"  total duration: {audio_dur:.2f}s")
    logger.info(f"  total time: {total_time:.2f}s")
    if first_chunk_latency is not None:
        logger.info(f"  first-packet latency (wall): {first_chunk_latency:.2f}s")
    if timing is not None and timing.first_audio_latency > 0:
        logger.info(
            f"  first-audio latency (GGUF timing): {timing.first_audio_latency:.2f}s "
            f"(chunk_gen {timing.first_chunk_latency:.2f}s + decode {timing.first_decode_latency:.2f}s)"
        )
        logger.info(f"  decoder compute total: {timing.total_decoder_time:.2f}s")
    logger.info(f"  RTF: {total_time / audio_dur:.3f}")
    logger.info(f"  Chunks: {len(all_chunks)}")
    if decode_result is not None:
        logger.info(f"  GGUF responses: {len(decode_result.responses)}")
        logger.info(f"  final_state: {'yes' if decode_result.final_state is not None else 'no'}")
    logger.info(f"  output file: {output_file}")


def _choose_stateful_decoder_backend(decoder_path: str, backend: str) -> str:
    if backend != "auto":
        return backend
    path = Path(decoder_path)
    lower_name = path.name.lower()
    lower_parts = {part.lower() for part in path.parts}
    if lower_name == "meta.json" or "hmonnx" in lower_parts or "xh2" in lower_name:
        return "hmonnx"
    return "onnxruntime"


def _build_stateful_decoder(args: argparse.Namespace, logger, device: str):
    backend = _choose_stateful_decoder_backend(args.stateful_decoder, args.stateful_decoder_backend)
    if backend == "hmonnx":
        logger.info(f"loading HMONNX stateful decoder: {args.stateful_decoder}")
        decoder = HMONNXGGUFStatefulDecoder(
            args.stateful_decoder,
            device=device,
            chunk_size=args.chunk_size,
        )
    else:
        logger.info(f"loading ONNX Runtime stateful decoder: {args.stateful_decoder}")
        decoder = GGUFStatefulDecoder(
            args.stateful_decoder,
            onnx_provider=args.onnx_provider,
            chunk_size=args.chunk_size,
        )
    if int(decoder.chunk_size) != int(args.chunk_size):
        logger.warning(
            f"stateful decoder chunk_size={decoder.chunk_size} overrides requested chunk_size={args.chunk_size}"
        )
        args.chunk_size = int(decoder.chunk_size)
    logger.info(
        f"stateful decoder ready: backend={backend}, provider={decoder.active_provider}, "
        f"dtype={decoder.dtype.__name__}, chunk_size={decoder.chunk_size}"
    )
    return decoder


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    from config.llm._components import apply_variant_hmonnx

    variant = _normalize_variant(args.variant)
    apply_variant_hmonnx(cfg, variant)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_streaming_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = cfg.device

    seed = cfg.get("seed", args.seed)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()
    logger.info(f"variant: {variant}, tts_mode: {getattr(cfg, 'tts_mode', 'custom_voice')}")
    logger.info(f"config:\n{cfg.pretty_text}")

    logger.info("loading HMONNX model...")
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, Qwen3TTSHMONNXInference)
    xh_model.to(cfg.exec_device)

    stateful_decoder = None
    if args.mode == "live":
        if not args.stateful_decoder:
            raise RuntimeError(
                "--mode live is strict GGUF alignment and requires --stateful-decoder. "
                "Pass the stateful decoder meta.json/HMONNX/ONNX exported by qwen3_tts_stateful_decoder_export.py."
            )
        stateful_decoder = _build_stateful_decoder(args, logger, cfg.exec_device)

    streaming_tts = StreamingQwen3TTS(
        xh_model,
        chunk_size=args.chunk_size,
        stateful_decoder=stateful_decoder,
    )

    tts_request = _resolve_tts_request(args, cfg)
    logger.info(f"TTS request: {tts_request}")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if tts_request.tts_mode == "voice_clone":
        voice_clone_frontend = VoiceCloneFrontendHMONNX(args, torch.device(cfg.exec_device), logger)
        gen_kwargs["voice_clone_prompt"] = voice_clone_frontend.build_prompt(
            tts_request.ref_audio,
            tts_request.ref_text or "",
            args.xvec_only,
        )
        tts_request.ref_audio = None
    output_file = Path(cfg.work_dir) / args.output

    if args.mode == "oneshot":
        logger.info("[oneshot] non-streaming reference decode...")
        t0 = time.time()
        full_audio, sr = streaming_tts.generate_oneshot(**tts_request.as_kwargs(), **gen_kwargs)
        total_time = time.time() - t0
        sf.write(output_file, full_audio, sr)
        logger.info(
            f"[oneshot] duration {len(full_audio)/sr:.2f}s, time {total_time:.2f}s, "
            f"RTF {total_time/(len(full_audio)/sr):.3f}, output {output_file}"
        )
        return

    all_chunks: List[np.ndarray] = []
    first_chunk_latency = None
    start_time = time.time()

    if args.mode == "live":
        chunk_dir = Path(cfg.work_dir) / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[live] strict GGUF mode: code stream -> stateful decoder -> AUDIO/FINISH")
        for response in streaming_tts.generate_live_responses(**tts_request.as_kwargs(), **gen_kwargs):
            if response.msg_type == "AUDIO" and response.audio is not None and len(response.audio) > 0:
                if first_chunk_latency is None:
                    first_chunk_latency = time.time() - start_time
                all_chunks.append(response.audio)
                chunk_file = chunk_dir / f"chunk_{response.index:04d}.wav"
                sf.write(chunk_file, response.audio, streaming_tts.sample_rate)
                cumulative = sum(len(c) for c in all_chunks) / streaming_tts.sample_rate
                logger.info(
                    f"Chunk {len(all_chunks)}: task_id={response.task_id}, index={response.index}, "
                    f"{len(response.audio)} samples ({len(response.audio)/streaming_tts.sample_rate*1000:.0f}ms), "
                    f"cumulative {cumulative:.2f}s, chunk file: {chunk_file}"
                )
            elif response.msg_type == "FINISH":
                logger.info(f"[{response.task_id}] FINISH received at index={response.index}")

        _write_and_report(
            logger,
            output_file,
            all_chunks,
            streaming_tts.sample_rate,
            start_time,
            first_chunk_latency,
            mode="live",
            timing=streaming_tts.last_timing,
            decode_result=streaming_tts.last_decode_result,
        )
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS strict GGUF streaming inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py",
        help="HMONNX config file path",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="customvoice",
        choices=[
            "customvoice",
            "base",
            "voicedesign",
            "custom_voice",
            "voice_clone",
            "voice_design",
            "0_6B_customvoice",
            "0_6B_base",
            "1_7B_voicedesign",
        ],
        help="Qwen3-TTS variant/task family",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地。",
        help="text to synthesize",
    )
    parser.add_argument("--language", type=str, default="Chinese", help="language, e.g. Chinese/English/Japanese/Korean")
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        choices=["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"],
        help="customvoice speaker id; defaults to the variant config",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="voicedesign instruction; defaults to the variant config",
    )
    parser.add_argument(
        "--ref-audio",
        dest="ref_audio",
        type=str,
        default=None,
        help="base/voice-clone reference audio; defaults to the variant config",
    )
    parser.add_argument(
        "--ref-text",
        dest="ref_text",
        type=str,
        default=None,
        help="base/voice-clone reference text; defaults to the variant config",
    )
    parser.add_argument(
        "--xvec-only",
        dest="xvec_only",
        action="store_true",
        help="base/voice-clone only: use x-vector speaker embedding mode",
    )
    parser.add_argument(
        "--frontend-hmonnx-dir",
        type=str,
        default=DEFAULT_FRONTEND_HMONNX_DIR,
        help="work dir containing exported voice-clone frontend HMONNX files",
    )
    parser.add_argument(
        "--speech-tokenizer-encode-hmonnx",
        type=str,
        default=None,
        help="explicit speech_tokenizer.encode HMONNX path; overrides --frontend-hmonnx-dir",
    )
    parser.add_argument(
        "--speaker-encoder-hmonnx",
        type=str,
        default=None,
        help="explicit speaker_encoder HMONNX path; overrides --frontend-hmonnx-dir",
    )
    parser.add_argument(
        "--frontend-sample-rate",
        type=int,
        default=24000,
        help="sample rate expected by exported voice-clone frontend modules",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "oneshot"],
        help="live=strict GGUF stateful decoder; oneshot=reference",
    )
    parser.add_argument(
        "--stateful-decoder",
        type=str,
        default="",
        help="stateful decoder meta.json/HMONNX/ONNX path required by --mode live",
    )
    parser.add_argument(
        "--stateful-decoder-backend",
        type=str,
        default="auto",
        choices=["auto", "hmonnx", "onnxruntime"],
        help="runtime backend for --stateful-decoder; auto uses HMONNX for meta.json or hmonnx paths",
    )
    parser.add_argument(
        "--onnx-provider",
        type=str,
        default="CUDA",
        choices=["CUDA", "CPU", "DML", "TRT", "TENSORRT"],
        help="ONNX Runtime provider for --stateful-decoder",
    )
    parser.add_argument("--chunk-size", type=int, default=12, help="code frames per decoded chunk (GGUF default 12)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="max talker generation frames")
    parser.add_argument("--device", type=str, default="cuda", help="inference device")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--output", type=str, default="output_streaming.wav", help="output audio filename")
    parser.add_argument("--name", type=str, default=None, help="work_dir name override")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    cfg_name = Path(args.config).stem
    variant = _normalize_variant(args.variant)
    work_name = args.name or f"{cfg_name}_{variant}_streaming"
    args.work_dir = str(Path("./work_dirs") / work_name)

    main(args)
