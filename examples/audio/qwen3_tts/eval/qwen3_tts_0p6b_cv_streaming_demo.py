#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS 0.6B CustomVoice streaming inference example (HMONNX / XH2a)

This script mirrors the decode loop of the community GGUF solution
https://github.com/HaujetZhao/Qwen3-TTS-GGUF: audio codes are fed to the
speech_tokenizer in frame chunks; each decoded chunk is yielded immediately,
with left-context overlap + crossfade at chunk boundaries; last chunk is is_final.

Differences vs GGUF (limited by the current HMONNX export):
  * GGUF rewrites everything in llama.cpp with per-frame interleaving and a
    cross-chunk streaming vocoder, achieving ~300ms first-packet latency.
  * Here qwen_tts talker generation is one-shot autoregressive (no streamer), and
    the HMONNX talker per-step forward does not expose full-frame codes, so
    per-frame interleaving is impossible; the exported speech_tokenizer is a
    stateless fixed-length decoder (1..300 frames). So this demo does
    'talker produces all codes once -> decoder streams by frame chunk'. Talker is
    the bottleneck; decoding is real chunked decoding, consumable chunk by chunk.

Usage:
    PYTHONPATH=/path/to/xh2modelzoo \
    python eval/qwen3_tts_0p6b_cv_streaming_demo.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py \
        --text "Hello, this is a streaming TTS test." \
        --speaker vivian --chunk-size 12 --overlap 4
"""

import argparse
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from xhquant.api import Config, set_random_seed
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_tts import Qwen3TTSHMONNXInference


CODE_RATE_HZ = 12  # Qwen3-TTS 12Hz: 12 code frames per second (for log estimates only)


class StreamingQwen3TTS:
    """Streaming TTS wrapper: talker produces all codes once; decoder streams by frame chunk."""

    def __init__(
        self,
        model: Qwen3TTSHMONNXInference,
        chunk_size: int = 12,
        overlap: int = 4,
        crossfade_ms: float = 8.0,
    ):
        """
        Args:
            model: HMONNX inference model (Qwen3TTSHMONNXInference).
            chunk_size: code frames per decoded chunk; smaller = lower latency. GGUF default 12.
            overlap: extra left-context frames per chunk (vocoder conv warmup only; that audio is dropped).
            crossfade_ms: crossfade length (ms) between adjacent chunks to remove boundary discontinuity.
        """
        self.model = model
        self.chunk_size = int(chunk_size)
        self.overlap = max(0, int(overlap))
        self.crossfade_ms = float(crossfade_ms)
        self.logger = get_root_logger()
        # real entry of speech_tokenizer.decode (qwen_tts tokenizer, backed by our HMONNX decoder)
        self._st = self.model.native_model.model.speech_tokenizer
        self._warmed = False

    def warmup(self) -> None:
        """Warm up speech_tokenizer: the HMONNX decoder does a one-time graph compile (~90s) on first call.
        Trigger it here so it does not pollute first-packet latency / first-chunk timing."""
        if self._warmed:
            return
        try:
            t = time.time()
            dummy = torch.zeros(min(self.chunk_size, 8), 16, dtype=torch.long)
            self._decode_chunk(dummy)
            self.logger.info(f"speech_tokenizer warmup done in {time.time() - t:.1f}s")
        except Exception as e:  # warmup failure is non-fatal; real decode will surface it
            self.logger.warning(f"speech_tokenizer warmup failed (ignored): {e}")
        self._warmed = True

    # ------------------------------------------------------------------ #
    # 1) get codes: run talker to completion, intercept speech_tokenizer.decode to grab all audio_codes
    # ------------------------------------------------------------------ #
    def _generate_codes(self, text: str, language: str, speaker: str, **gen_kwargs) -> torch.Tensor:
        """Run generate_custom_voice but intercept the decode input to get [T, 16] codes, skipping full decode."""
        captured = {}
        real_decode = self._st.decode

        def _capture_decode(encoded, *a, **k):
            entry = encoded[0] if isinstance(encoded, (list, tuple)) else encoded
            captured["codes"] = entry["audio_codes"]
            # return a minimal valid result so generate_custom_voice finishes (we ignore its wav)
            return [np.zeros(1, dtype=np.float32)], int(getattr(real_decode, "sample_rate", 24000) or 24000)

        self._st.decode = _capture_decode
        try:
            self.model.generate_custom_voice(text=text, language=language, speaker=speaker, **gen_kwargs)
        finally:
            self._st.decode = real_decode

        if "codes" not in captured:
            raise RuntimeError("failed to intercept audio_codes from speech_tokenizer.decode; cannot stream-decode.")
        codes = captured["codes"]
        if not torch.is_tensor(codes):
            codes = torch.as_tensor(np.asarray(codes))
        if codes.dim() == 3:  # [1, T, 16] -> [T, 16]
            codes = codes[0]
        return codes  # [T, 16]

    def _decode_chunk(self, codes_chunk: torch.Tensor) -> np.ndarray:
        """Decode one chunk of codes ([n, 16]) with speech_tokenizer -> 1D float32 waveform."""
        wavs, sr = self._st.decode([{"audio_codes": codes_chunk}])
        wav = wavs[0]
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        self._sr = int(sr)
        return wav

    # ------------------------------------------------------------------ #
    # 2) chunked decode + left-context warmup + crossfade, yielding chunk by chunk
    # ------------------------------------------------------------------ #
    def generate_streaming(
        self, text: str, language: str, speaker: str, **gen_kwargs
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """Stream audio, yielding (audio_chunk[float32, 1D], sample_rate) per chunk."""
        self.logger.info(
            f"start streaming: text='{text}', speaker={speaker}, "
            f"chunk_size={self.chunk_size} frames, overlap={self.overlap} frames"
        )

        t0 = time.time()
        codes = self._generate_codes(text, language, speaker, **gen_kwargs)
        gen_time = time.time() - t0
        total_frames = int(codes.shape[0])
        self.logger.info(f"talker done: {total_frames} code frames in {gen_time:.2f}s")
        if total_frames == 0:
            return

        # decode one chunk first to get sample rate / samples-per-frame (inferred, not hardcoded)
        self._sr = 24000
        xfade = 0
        prev_tail: Optional[np.ndarray] = None  # previous chunk tail kept for crossfade

        for start in range(0, total_frames, self.chunk_size):
            end = min(start + self.chunk_size, total_frames)
            ctx = min(self.overlap, start)  # left-context frames (0 for the first chunk)
            chunk_codes = codes[start - ctx : end]  # [ctx+C, 16]

            t_c = time.time()
            wav = self._decode_chunk(chunk_codes)
            decode_time = time.time() - t_c

            # drop the warmup audio for left context, keep only the new audio for [start, end)
            n_in = (end - start) + ctx
            spf = len(wav) / max(n_in, 1)  # samples per frame (measured from this decode)
            ctx_samples = int(round(ctx * spf))
            new_wav = wav[ctx_samples:]

            if xfade == 0:
                xfade = min(int(self._sr * self.crossfade_ms / 1000.0), max(1, len(new_wav) // 4))

            # crossfade join: linearly mix previous chunk's xfade tail with this chunk's head
            if prev_tail is None:
                emit = new_wav[:-xfade] if len(new_wav) > xfade else new_wav[:0]
                prev_tail = new_wav[-xfade:] if len(new_wav) > xfade else new_wav
            else:
                k = min(xfade, len(prev_tail), len(new_wav))
                ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
                blended = prev_tail[-k:] * (1.0 - ramp) + new_wav[:k] * ramp
                body = new_wav[k:-xfade] if len(new_wav) - k > xfade else new_wav[k:][:0]
                emit = np.concatenate([blended, body])
                prev_tail = new_wav[-xfade:] if len(new_wav) > xfade else new_wav[k:]

            is_final = end >= total_frames
            if is_final and prev_tail is not None and len(prev_tail) > 0:
                emit = np.concatenate([emit, prev_tail])
                prev_tail = None

            self.logger.info(
                f"chunk[{start:>4}:{end:>4}] frames -> {len(emit)} samples "
                f"({len(emit)/self._sr*1000:.0f}ms), decode {decode_time:.2f}s"
                + ("  [final]" if is_final else "")
            )
            if len(emit) > 0:
                yield emit.astype(np.float32), self._sr

        self.logger.info("streaming generation done")

    # non-streaming reference: decode the whole thing at once (for comparison)
    def generate_oneshot(self, text: str, language: str, speaker: str, **gen_kwargs) -> Tuple[np.ndarray, int]:
        wavs, sr = self.model.generate_custom_voice(text=text, language=language, speaker=speaker, **gen_kwargs)
        wav = wavs[0]
        if torch.is_tensor(wav):
            wav = wav.detach().cpu().numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    from config.llm._components import apply_variant_hmonnx
    apply_variant_hmonnx(cfg, "cv")  # streaming demo is custom-voice only
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
    logger.info(f"config:\n{cfg.pretty_text}")

    logger.info("loading HMONNX model...")
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, Qwen3TTSHMONNXInference)
    xh_model.to(cfg.exec_device)

    streaming_tts = StreamingQwen3TTS(
        xh_model, chunk_size=args.chunk_size, overlap=args.overlap, crossfade_ms=args.crossfade_ms
    )

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    output_file = Path(cfg.work_dir) / args.output

    mode = args.mode
    if mode == "live":
        logger.warning(
            "current HMONNX export does not support GGUF-style per-frame interleaved streaming "
            "(talker is one-shot, per-frame seam not exposed); falling back to decode (chunked streaming)."
        )
        mode = "decode"

    if mode == "oneshot":
        logger.info("[oneshot] non-streaming reference decode...")
        t0 = time.time()
        full_audio, sr = streaming_tts.generate_oneshot(args.text, args.language, args.speaker, **gen_kwargs)
        total_time = time.time() - t0
        sf.write(output_file, full_audio, sr)
        logger.info(f"[oneshot] duration {len(full_audio)/sr:.2f}s, time {total_time:.2f}s, "
                    f"RTF {total_time/(len(full_audio)/sr):.3f}, output {output_file}")
        return

    # decode: chunked streaming
    streaming_tts.warmup()  # one-time graph compile so first-packet/first-chunk timing is clean
    all_chunks: List[np.ndarray] = []
    sr = None
    chunk_count = 0
    first_chunk_latency = None
    start_time = time.time()

    for audio_chunk, _sr in streaming_tts.generate_streaming(
        text=args.text, language=args.language, speaker=args.speaker, **gen_kwargs
    ):
        chunk_count += 1
        if first_chunk_latency is None:
            first_chunk_latency = time.time() - start_time
        all_chunks.append(audio_chunk)
        sr = _sr
        # in real deployment you could play here, e.g. sounddevice.play(audio_chunk, sr)
        logger.info(
            f"Chunk {chunk_count}: {len(audio_chunk)} samples "
            f"({len(audio_chunk)/sr*1000:.0f}ms), cumulative {sum(len(c) for c in all_chunks)/sr:.2f}s"
        )

    total_time = time.time() - start_time
    if not all_chunks:
        logger.error("no audio chunk was generated.")
        return

    full_audio = np.concatenate(all_chunks)
    sf.write(output_file, full_audio, sr)

    audio_dur = len(full_audio) / sr
    logger.info("streaming generation done!")
    logger.info(f"  total duration: {audio_dur:.2f}s")
    logger.info(f"  total time: {total_time:.2f}s")
    logger.info(f"  first-packet latency: {first_chunk_latency:.2f}s")
    logger.info(f"  RTF: {total_time / audio_dur:.3f}")
    logger.info(f"  Chunks: {chunk_count}")
    logger.info(f"  output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 0.6B CustomVoice streaming inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py",
        help="HMONNX config file path",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地。",
        help="text to synthesize",
    )
    parser.add_argument("--language", type=str, default="Chinese", choices=["Chinese", "English"], help="language")
    parser.add_argument(
        "--speaker",
        type=str,
        default="vivian",
        choices=["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"],
        help="speaker id",
    )
    parser.add_argument("--mode", type=str, default="decode", choices=["decode", "oneshot", "live"],
                        help="decode=chunked streaming (default); oneshot=non-streaming reference; live=per-frame interleave (unsupported by current export, falls back to decode)")
    parser.add_argument("--chunk-size", type=int, default=12, help="code frames per decoded chunk (GGUF default 12)")
    parser.add_argument("--overlap", type=int, default=4, help="left-context frames per chunk (vocoder warmup only)")
    parser.add_argument("--crossfade-ms", type=float, default=8.0, help="crossfade length between adjacent chunks (ms)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="max talker generation frames")
    parser.add_argument("--device", type=str, default="cuda", help="inference device")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    parser.add_argument("--output", type=str, default="output_streaming.wav", help="output audio filename")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    cfg_name = Path(args.config).stem
    args.work_dir = str(Path("./work_dirs") / f"{cfg_name}_streaming")

    main(args)
