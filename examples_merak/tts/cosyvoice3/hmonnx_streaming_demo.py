"""CosyVoice3 HMONNX token-level streaming demo.

Implements intra-sentence 25-token chunk streaming on fixed-length HMONNX
graphs (ported from examples/audio/Cosyvoice3/cv3_stream.py). Each chunk
is written to disk as soon as it is generated, demonstrating real
first-packet latency below full-sentence generation time.

Usage:
  python hmonnx_streaming_demo.py --work-dir <export_dir> \
      --text "你好" --prompt-wav prompt.wav --prompt-text "prompt" \
      --device cuda:0 --v3-align --fade-ms 5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from hmonnx_utils import build_cosyvoice3_model, load_export_meta

from xhquant.api import set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="CosyVoice3 HMONNX token-level streaming demo.")
    parser.add_argument("--work-dir", type=str, required=True, help="Export output directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--prompt-wav", type=str, required=True, help="Prompt wav (16k).")
    parser.add_argument("--prompt-text", type=str, default="", help="Prompt text.")
    parser.add_argument("--output", type=str, default=None, help="Output wav path.")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--token-hop-len", type=int, default=25, help="Tokens per chunk.")
    parser.add_argument("--pre-lookahead-len", type=int, default=3, help="Pre-lookahead tokens.")
    parser.add_argument("--v3-align", action="store_true", help="Cut hift tail 3840 samples on intermediate chunks.")
    parser.add_argument("--fade-ms", type=float, default=5.0, help="Fade-in/out per chunk boundary (ms).")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    set_random_seed(args.seed)
    work_dir = Path(args.work_dir)

    load_export_meta(work_dir)
    model = build_cosyvoice3_model(work_dir, args.device)

    sample_rate = 24000
    out = Path(args.output) if args.output else work_dir / "streaming_demo.wav"
    chunks_dir = out.parent / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_idx = 0
    first_packet_time = None
    start = time.time()

    for chunk_wav, is_first, is_final in model.generate_stream(
        text=args.text,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        token_hop_len=args.token_hop_len,
        pre_lookahead_len=args.pre_lookahead_len,
        v3_align=args.v3_align,
        fade_ms=args.fade_ms,
        sample_rate=sample_rate,
    ):
        elapsed = time.time() - start
        if is_first and first_packet_time is None:
            first_packet_time = elapsed

        chunk_path = chunks_dir / f"chunk_{chunk_idx:03d}.wav"
        sf.write(str(chunk_path), chunk_wav.squeeze(0), sample_rate)
        all_chunks.append(chunk_wav)

        tag = "FIRST" if is_first else ("LAST " if is_final else "     ")
        duration = chunk_wav.shape[-1] / sample_rate
        print(
            f"[chunk {chunk_idx:2d} {tag}] wall={elapsed:.2f}s audio={duration:.2f}s -> {chunk_path}",
            flush=True,
        )
        chunk_idx += 1

    total_elapsed = time.time() - start
    if all_chunks:
        final_wav = np.concatenate(all_chunks, axis=-1)
        sf.write(str(out), final_wav.squeeze(0), sample_rate)
    else:
        final_wav = np.zeros((1, 0), dtype=np.float32)

    print(f"\noutput: {out}")
    print(f"chunks: {chunk_idx}")
    print(f"first-packet latency: {first_packet_time:.2f}s" if first_packet_time else "first-packet latency: N/A")
    print(f"total elapsed: {total_elapsed:.2f}s")
    print(f"total audio: {final_wav.shape[-1] / sample_rate:.2f}s")


if __name__ == "__main__":
    main()
