"""VoxCPM2 HMONNX demo。

用法:
    python hmonnx_demo.py \
        --work_dir /data01/home/binghu.ji/0401_modelzoo/xh2modelzoo/examples/audio/voxcpm2/work_dirs/VoxCPM2_XH2a \
        --text "后摩智能的小伙伴们你们好，我是哆啦 A 梦！" \
        --output zero_shot_output.wav

    # reference mode:
    python hmonnx_demo.py \
        --work_dir /data01/home/binghu.ji/0401_modelzoo/xh2modelzoo/examples/audio/voxcpm2/work_dirs/VoxCPM2_XH2a \
        --text "后摩智能的小伙伴们你们好，我是哆啦 A 梦！" \
        --reference_wav /data01/home/binghu.ji/0401_modelzoo/xh2modelzoo/examples/audio/qwen3_asr/dsj_20251212.wav \
        --output 0423_reference_output.wav \
        --align_torch \
        --model_dir /data01/home/binghu.ji/models/VoxCPM2

输出:
    写到 --output 指定的 WAV 文件(默认 output.wav)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import soundfile as sf
except ImportError:
    sf = None


def save_wav(audio: np.ndarray, sample_rate: int, path: str):
    """保存 1D 波形到 WAV。优先用 soundfile,失败则回退到 scipy.io.wavfile。"""
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    if sf is not None:
        sf.write(path, audio, sample_rate)
        return

    from scipy.io import wavfile
    int16 = (audio * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, int16)


def _calc_audio_metrics(hmonnx_audio: np.ndarray, torch_audio: np.ndarray):
    a = np.asarray(hmonnx_audio, dtype=np.float32).reshape(-1)
    b = np.asarray(torch_audio, dtype=np.float32).reshape(-1)
    min_len = min(a.size, b.size)
    if min_len == 0:
        return {
            "hmonnx_len": int(a.size),
            "torch_len": int(b.size),
            "len_ratio": float("inf") if b.size == 0 else float(a.size / max(1, b.size)),
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "cosine": 0.0,
        }
    ax = a[:min_len]
    bx = b[:min_len]
    abs_diff = np.abs(ax - bx)
    cosine = float(np.dot(ax, bx) / ((np.linalg.norm(ax) * np.linalg.norm(bx)) + 1e-12))
    return {
        "hmonnx_len": int(a.size),
        "torch_len": int(b.size),
        "len_ratio": float(a.size / max(1, b.size)),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "cosine": cosine,
    }


def main(args):
    from xhmodel_merak.xh_other_model.models.voxcpm2 import VoxCPM2HMONNXTTSPipeline

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[demo] loading pipeline from {args.work_dir} on {device}", file=sys.stderr)

    pipeline = VoxCPM2HMONNXTTSPipeline(
        work_dir=args.work_dir,
        device=device,
        audio_encoder_backend=args.audio_encoder_backend,
        torch_audio_model_dir=args.torch_audio_model_dir,
    )
    print(f"[demo] pipeline loaded. sample_rate={pipeline.sample_rate}", file=sys.stderr)

    gen_kwargs = dict(
        text=args.text,
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        reference_wav_path=args.reference_wav,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        min_len=args.min_len,
        max_len=args.max_len,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    if args.streaming:
        print(f"[demo] streaming mode backend={args.streaming_backend}", file=sys.stderr)
        chunks = []
        stream_iter = (
            pipeline.generate_streaming_legacy(**gen_kwargs)
            if args.streaming_backend == "overlap"
            else pipeline.generate_streaming(**gen_kwargs)
        )
        for i, chunk in enumerate(stream_iter):
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            print(f"  chunk {i}: {chunks[-1].shape[0]} samples", file=sys.stderr)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    else:
        audio = pipeline.generate(**gen_kwargs)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    t1 = time.time()

    duration = audio.shape[0] / pipeline.sample_rate
    print(
        f"[demo] done. elapsed={t1 - t0:.2f}s  audio_duration={duration:.2f}s"
        f"  rtf={((t1 - t0) / duration if duration > 0 else float('inf')):.3f}",
        file=sys.stderr,
    )

    out_path = args.output
    save_wav(audio, pipeline.sample_rate, out_path)
    print(f"[demo] saved to {out_path}", file=sys.stderr)

    if args.align_torch:
        try:
            from voxcpm import VoxCPM2Model
        except ImportError:
            from voxcpm.model.voxcpm2 import VoxCPM2Model

        model_dir = str(Path(args.model_dir).expanduser().resolve())
        print(f"[align] loading pytorch model from {model_dir}", file=sys.stderr)
        torch_model = VoxCPM2Model.from_local(model_dir, optimize=False, training=False)
        torch_model.to(device)
        torch_model.eval()

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        t2 = time.time()
        if args.streaming:
            pt_chunks = []
            for chunk in torch_model.generate_streaming(
                target_text=args.text,
                prompt_wav_path=args.prompt_wav or "",
                prompt_text=args.prompt_text or "",
                reference_wav_path=args.reference_wav or "",
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                min_len=args.min_len,
                max_len=args.max_len,
            ):
                pt_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            torch_audio = np.concatenate(pt_chunks) if pt_chunks else np.zeros(0, dtype=np.float32)
        else:
            torch_audio = torch_model.generate(
                target_text=args.text,
                prompt_wav_path=args.prompt_wav or "",
                prompt_text=args.prompt_text or "",
                reference_wav_path=args.reference_wav or "",
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                min_len=args.min_len,
                max_len=args.max_len,
            )
            torch_audio = np.asarray(torch_audio, dtype=np.float32).reshape(-1)
        t3 = time.time()
        print(f"[align] pytorch done. elapsed={t3 - t2:.2f}s", file=sys.stderr)

        metrics = _calc_audio_metrics(audio, torch_audio)
        print(
            "[align] hmonnx_vs_torch "
            f"len=({metrics['hmonnx_len']},{metrics['torch_len']}) "
            f"len_ratio={metrics['len_ratio']:.4f} "
            f"max_abs={metrics['max_abs']:.6f} "
            f"mean_abs={metrics['mean_abs']:.6f} "
            f"cosine={metrics['cosine']:.6f}",
            file=sys.stderr,
        )

        ok = (
            abs(metrics["len_ratio"] - 1.0) <= args.e2e_len_ratio_tol
            and metrics["mean_abs"] <= args.e2e_mean_abs_tol
            and metrics["cosine"] >= args.e2e_cosine_tol
        )
        report = {
            "work_dir": str(args.work_dir),
            "model_dir": model_dir,
            "seed": args.seed,
            "streaming": bool(args.streaming),
            "metrics": metrics,
            "thresholds": {
                "e2e_len_ratio_tol": args.e2e_len_ratio_tol,
                "e2e_mean_abs_tol": args.e2e_mean_abs_tol,
                "e2e_cosine_tol": args.e2e_cosine_tol,
            },
            "passed": bool(ok),
        }
        report_file = str(Path(out_path).with_suffix(Path(out_path).suffix + ".align.json"))
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[align] report saved to {report_file}", file=sys.stderr)
        if not ok and args.e2e_fail_on_mismatch:
            raise RuntimeError("E2E alignment failed, see align report for details.")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", type=str, required=True,
                   help="量化导出的工作目录(含 lm_export_meta_info.json 等)")
    p.add_argument("--text", type=str, required=True, help="目标合成文本")
    p.add_argument("--prompt_wav", type=str, default=None)
    p.add_argument("--prompt_text", type=str, default=None)
    p.add_argument("--reference_wav", type=str, default=None)
    p.add_argument("--cfg_value", type=float, default=2.0)
    p.add_argument("--inference_timesteps", type=int, default=10)
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--max_len", type=int, default=500)
    p.add_argument("--streaming", action="store_true")
    p.add_argument(
        "--streaming_backend",
        type=str,
        default="stateful",
        choices=["stateful", "overlap"],
        help="streaming decoder 后端: stateful 为真流式 cache 图; overlap 为旧 np3 overlap/crop",
    )
    p.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    p.add_argument("--output", type=str, default="output.wav")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--align_torch", action="store_true", help="追加 PyTorch 端到端对齐")
    p.add_argument("--model_dir", type=str, default="/data01/home/binghu.ji/models/VoxCPM2")
    p.add_argument(
        "--audio_encoder_backend",
        type=str,
        default="hmonnx",
        choices=["auto", "hmonnx", "torch"],
        help="prompt/reference wav 编码后端: auto 优先 torch,失败回落 hmonnx",
    )
    p.add_argument(
        "--torch_audio_model_dir",
        type=str,
        default=None,
        help="torch 音频编码器模型目录(默认取 lm_export_meta_info.json 的 hf_model)",
    )
    p.add_argument("--e2e_len_ratio_tol", type=float, default=0.20)
    p.add_argument("--e2e_mean_abs_tol", type=float, default=0.25)
    p.add_argument("--e2e_cosine_tol", type=float, default=0.60)
    p.add_argument("--e2e_fail_on_mismatch", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
