"""Run true stateful streaming alignment for VoxCPM2 AudioVAE decoder.

This script uses the newly exported
``AudioVAE_Decoder_StreamState_np1`` HMONNX graph. It feeds only the newest
latent patch into the decoder and carries the returned cache tensors between
chunks, matching the native ``audio_vae.streaming_decode()`` contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from onnx import TensorProto

from xhquant.api import HMONNXGoldenInference

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None


_NAME_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


def save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    if sf is not None:
        sf.write(str(path), audio, sample_rate)
        return
    from scipy.io import wavfile

    wavfile.write(str(path), sample_rate, (audio * 32767).astype(np.int16))


def audio_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(a.size, b.size)
    result: dict[str, float | int] = {
        "a_len": int(a.size),
        "b_len": int(b.size),
        "len_diff": int(a.size - b.size),
        "len_ratio": float(a.size / max(1, b.size)),
    }
    if n == 0:
        result.update(max_abs=float("inf"), mean_abs=float("inf"), cosine=0.0)
        return result
    ax = a[:n]
    bx = b[:n]
    diff = np.abs(ax - bx)
    result.update(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        cosine=float(np.dot(ax, bx) / ((np.linalg.norm(ax) * np.linalg.norm(bx)) + 1e-12)),
    )
    return result


def concat_or_empty(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in chunks])


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cast_tensor(tensor: torch.Tensor, dtype_name: str | None) -> torch.Tensor:
    if dtype_name is None:
        return tensor
    dtype = _NAME_TO_TORCH_DTYPE.get(dtype_name)
    if dtype is None or tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype=dtype)


def patch_to_latent(patch: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if patch.dim() != 4 or patch.shape[0] != 1 or patch.shape[1] != 1:
        raise ValueError(f"expected [1,1,P,D] patch, got {tuple(patch.shape)}")
    return patch.permute(0, 3, 1, 2).reshape(1, latent_dim, -1).contiguous()


class StatefulAudioVAEDecoderHMONNX:
    def __init__(self, work_dir: str | Path, device: torch.device):
        self.work_dir = Path(work_dir).expanduser().resolve()
        meta_files = sorted(self.work_dir.glob("audiovae_decoder_streaming_stateful_np*_meta_info.json"))
        if not meta_files:
            raise FileNotFoundError(f"stateful decoder meta not found in {self.work_dir}")
        self.meta_path = meta_files[0]
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        hmonnx_rel = self.meta.get("hmonnx_file")
        if not hmonnx_rel:
            raise FileNotFoundError(f"hmonnx_file is missing in {self.meta_path}")
        self.hmonnx_path = self.work_dir.parent / hmonnx_rel
        self.session = HMONNXGoldenInference(str(self.hmonnx_path))
        self.session.to(device)
        self.device = device
        self.graph_dtypes = self.meta.get("graph_input_dtype", {})
        self.input_names = self.meta["input_names"]
        self.state_specs = self.meta["state_specs"]
        self.sr_idx = int(self.meta["precomputed_sr_idx"])
        self.reset()

    def reset(self) -> None:
        self.states: list[torch.Tensor] = []
        for idx, spec in enumerate(self.state_specs):
            dtype_name = self.graph_dtypes.get(f"state_in_{idx}", self.graph_dtypes.get("z", "float16"))
            dtype = _NAME_TO_TORCH_DTYPE.get(dtype_name, torch.float16)
            self.states.append(torch.zeros(spec["shape"], device=self.device, dtype=dtype))

    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        z = cast_tensor(z.to(self.device), self.graph_dtypes.get("z"))
        sr_idx = torch.tensor([self.sr_idx], device=self.device, dtype=torch.int32)
        sr_idx = cast_tensor(sr_idx, self.graph_dtypes.get("sr_idx"))
        inputs = [z, sr_idx, *self.states]
        outputs = self.session(*inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        audio = outputs[0]
        self.states = [out.detach() for out in outputs[1:]]
        return audio


def build_gen_inputs(pipeline: Any, args: argparse.Namespace):
    target_text = args.text.replace("\n", " ")
    text_token, audio_feat, text_mask, audio_mask = pipeline._assemble_prefill_inputs(
        target_text=target_text,
        prompt_text=args.prompt_text or "",
        prompt_wav_path=args.prompt_wav or "",
        reference_wav_path=args.reference_wav or "",
    )
    combined_embed, feat_embed = pipeline._build_combined_embed(text_token, audio_feat, text_mask, audio_mask)
    target_text_length = len(pipeline.text_tokenizer(target_text))
    max_len = min(int(target_text_length * 6.0 + 10), int(args.max_len))
    return combined_embed, feat_embed, text_mask, audio_mask, audio_feat, max(1, max_len)


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    from xh_model_zoo.xh_llm.models.voxcpm2 import VoxCPM2HMONNXTTSPipeline

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    print(f"[stateful-align] loading pipeline on {device}", file=sys.stderr)
    pipeline = VoxCPM2HMONNXTTSPipeline(
        work_dir=args.work_dir,
        device=str(device),
        audio_encoder_backend=args.audio_encoder_backend,
        torch_audio_model_dir=args.torch_audio_model_dir,
    )
    print("[stateful-align] loading PyTorch AudioVAE", file=sys.stderr)
    audio_vae = pipeline._load_torch_audio_vae().to(pipeline.device, dtype=torch.float32).eval()
    print("[stateful-align] loading stateful HMONNX decoder", file=sys.stderr)
    h_decoder = StatefulAudioVAEDecoderHMONNX(args.stateful_decoder_dir, pipeline.device)

    combined_embed, feat_embed, text_mask, audio_mask, audio_feat, max_len = build_gen_inputs(pipeline, args)
    inference_gen = pipeline._inference_core(
        combined_embed=combined_embed,
        feat_embed=feat_embed,
        text_mask=text_mask,
        audio_mask=audio_mask,
        original_audio_feat=audio_feat,
        min_len=args.min_len,
        max_len=max_len,
        inference_timesteps=args.inference_timesteps,
        cfg_value=args.cfg_value,
        streaming=True,
        streaming_prefix_len=args.streaming_prefix_len,
    )

    torch_chunks: list[np.ndarray] = []
    hmonnx_chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    with audio_vae.streaming_decode() as torch_dec:
        h_decoder.reset()
        for idx, (_overlap_latent, pred_feat_seq, context_len) in enumerate(inference_gen):
            z_new = patch_to_latent(pred_feat_seq[-1], pipeline.latent_dim)
            torch_audio = torch_dec.decode_chunk(z_new.to(pipeline.device, dtype=torch.float32))
            h_audio = h_decoder.decode_chunk(z_new)

            torch_np = torch_audio.squeeze(1).reshape(-1).detach().float().cpu().numpy()
            h_np = h_audio.squeeze(1).reshape(-1).detach().float().cpu().numpy()
            torch_chunks.append(torch_np)
            hmonnx_chunks.append(h_np)
            row = {
                "chunk_index": idx,
                "context_len": int(context_len),
                "new_latent_T": int(z_new.shape[-1]),
                "torch_samples": int(torch_np.size),
                "hmonnx_samples": int(h_np.size),
                "hmonnx_stateful_vs_torch_true": audio_metrics(h_np, torch_np),
            }
            rows.append(row)
            if args.save_chunks:
                save_wav(torch_np, pipeline.sample_rate, output_dir / "chunks" / f"torch_true_{idx:04d}.wav")
                save_wav(h_np, pipeline.sample_rate, output_dir / "chunks" / f"hmonnx_stateful_{idx:04d}.wav")
            print(
                f"[stateful-align] chunk {idx:03d}: "
                f"mean_abs={row['hmonnx_stateful_vs_torch_true']['mean_abs']:.6f} "
                f"cos={row['hmonnx_stateful_vs_torch_true']['cosine']:.6f}",
                file=sys.stderr,
            )

    torch_audio = concat_or_empty(torch_chunks)
    hmonnx_audio = concat_or_empty(hmonnx_chunks)
    torch_wav = output_dir / "torch_true_streaming.wav"
    hmonnx_wav = output_dir / "hmonnx_stateful_streaming.wav"
    save_wav(torch_audio, pipeline.sample_rate, torch_wav)
    save_wav(hmonnx_audio, pipeline.sample_rate, hmonnx_wav)

    report = {
        "work_dir": str(Path(args.work_dir).expanduser().resolve()),
        "stateful_decoder_dir": str(Path(args.stateful_decoder_dir).expanduser().resolve()),
        "stateful_hmonnx": str(h_decoder.hmonnx_path),
        "stateful_meta": str(h_decoder.meta_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "text": args.text,
        "sample_rate": int(pipeline.sample_rate),
        "num_chunks": int(len(rows)),
        "elapsed_sec": float(time.time() - t0),
        "outputs": {
            "torch_true_streaming": str(torch_wav),
            "hmonnx_stateful_streaming": str(hmonnx_wav),
        },
        "metrics": {
            "hmonnx_stateful_streaming_vs_torch_true_streaming": audio_metrics(hmonnx_audio, torch_audio),
        },
        "chunks": rows,
    }
    report_path = output_dir / "stateful_streaming_alignment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(output_dir / "stateful_streaming_alignment_chunks.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    m = report["metrics"]["hmonnx_stateful_streaming_vs_torch_true_streaming"]
    print(
        "[stateful-align] total "
        f"len=({m['a_len']},{m['b_len']}) mean_abs={m['mean_abs']:.6f} cosine={m['cosine']:.6f}",
        file=sys.stderr,
    )
    print(f"[stateful-align] report saved: {report_path}", file=sys.stderr)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work_dir",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/voxcpm2/work_dirs/VoxCPM2_XH2a",
    )
    p.add_argument(
        "--stateful_decoder_dir",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/voxcpm2/work_dirs/VoxCPM2_XH2a/AudioVAE_Decoder_StreamState_np1",
    )
    p.add_argument("--torch_audio_model_dir", type=str, default="/data01/nfs_shared/ASR_TTS/VoxCPM2")
    p.add_argument("--text", type=str, default="这是一个真实流式解码对齐测试。")
    p.add_argument("--prompt_wav", type=str, default=None)
    p.add_argument("--prompt_text", type=str, default=None)
    p.add_argument("--reference_wav", type=str, default=None)
    p.add_argument("--audio_encoder_backend", type=str, default="torch", choices=["auto", "hmonnx", "torch"])
    p.add_argument("--cfg_value", type=float, default=2.0)
    p.add_argument("--inference_timesteps", type=int, default=10)
    p.add_argument("--min_len", type=int, default=2)
    p.add_argument("--max_len", type=int, default=80)
    p.add_argument("--streaming_prefix_len", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/voxcpm2/streaming_align_results/stateful_streaming_align",
    )
    p.add_argument("--save_chunks", action="store_true")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
