#!/usr/bin/env python3
"""Run the VoxCPM2 HMONNX demo cases and save all outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hmonnx_demo import _calc_audio_metrics, save_wav


def as_audio(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def run_case(name: str, pipeline, output_dir: Path, gen_kwargs: dict, streaming: bool = False) -> dict:
    output_path = output_dir / f"{name}.wav"
    t0 = time.time()
    chunk_sizes = []
    if streaming:
        chunks = []
        for chunk in pipeline.generate_streaming(**gen_kwargs):
            audio_chunk = as_audio(chunk)
            chunks.append(audio_chunk)
            chunk_sizes.append(int(audio_chunk.shape[0]))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    else:
        audio = as_audio(pipeline.generate(**gen_kwargs))
    elapsed = time.time() - t0
    save_wav(audio, int(pipeline.sample_rate), str(output_path))
    return {
        "name": name,
        "streaming": streaming,
        "wav": str(output_path),
        "num_samples": int(audio.shape[0]),
        "duration_sec": float(audio.shape[0] / int(pipeline.sample_rate)),
        "elapsed_sec": float(elapsed),
        "chunk_sizes": chunk_sizes,
    }


def run_align_case(name: str, pipeline, output_dir: Path, gen_kwargs: dict, args: argparse.Namespace) -> dict:
    try:
        from voxcpm import VoxCPM2Model
    except ImportError:
        from voxcpm.model.voxcpm2 import VoxCPM2Model

    hmonnx_path = output_dir / f"{name}.hmonnx.wav"
    torch_path = output_dir / f"{name}.torch.wav"
    report_path = output_dir / f"{name}.align.json"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    hmonnx_audio = as_audio(pipeline.generate(**gen_kwargs))
    hmonnx_elapsed = time.time() - t0
    save_wav(hmonnx_audio, int(pipeline.sample_rate), str(hmonnx_path))

    torch_model = VoxCPM2Model.from_local(args.model_dir, optimize=False, training=False)
    torch_model.to(pipeline.device)
    torch_model.eval()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t1 = time.time()
    torch_audio = as_audio(
        torch_model.generate(
            target_text=gen_kwargs["text"],
            prompt_wav_path=gen_kwargs["prompt_wav_path"] or "",
            prompt_text=gen_kwargs["prompt_text"] or "",
            reference_wav_path=gen_kwargs["reference_wav_path"] or "",
            cfg_value=gen_kwargs["cfg_value"],
            inference_timesteps=gen_kwargs["inference_timesteps"],
            min_len=gen_kwargs["min_len"],
            max_len=gen_kwargs["max_len"],
        )
    )
    torch_elapsed = time.time() - t1
    save_wav(torch_audio, int(pipeline.sample_rate), str(torch_path))

    metrics = _calc_audio_metrics(hmonnx_audio, torch_audio)
    report = {
        "name": name,
        "hmonnx_wav": str(hmonnx_path),
        "torch_wav": str(torch_path),
        "hmonnx_elapsed_sec": float(hmonnx_elapsed),
        "torch_elapsed_sec": float(torch_elapsed),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(args: argparse.Namespace) -> None:
    from xh_model_zoo.xh_llm.models.voxcpm2 import VoxCPM2HMONNXTTSPipeline

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pipeline = VoxCPM2HMONNXTTSPipeline(
        work_dir=args.work_dir,
        device=device,
        audio_encoder_backend=args.audio_encoder_backend,
        torch_audio_model_dir=args.model_dir,
    )

    base = {
        "prompt_wav_path": None,
        "prompt_text": None,
        "reference_wav_path": None,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "min_len": args.min_len,
        "max_len": args.max_len,
    }

    cases = [
        (
            "01_zero_shot",
            {
                **base,
                "text": "你好，这是一次 VoxCPM2 HMONNX 语音合成测试。我们会检查完整句子的自然度和结尾。",
            },
            False,
        ),
        (
            "02_reference_clone",
            {
                **base,
                "text": "这是一段使用参考音色合成的语音，用来检查音色克隆是否稳定。",
                "reference_wav_path": args.reference_wav,
            },
            False,
        ),
        (
            "03_prompt_continuation",
            {
                **base,
                "text": "现在继续生成后面的内容，检查提示音频和新文本之间是否衔接自然。",
                "prompt_wav_path": args.reference_wav,
                "prompt_text": "这是一段提示音频对应的文本。",
            },
            False,
        ),
        (
            "04_reference_prompt",
            {
                **base,
                "text": "这是目标合成文本，用来检查参考音色和提示文本组合模式。",
                "reference_wav_path": args.reference_wav,
                "prompt_wav_path": args.reference_wav,
                "prompt_text": "这是一段提示音频对应的文本。",
            },
            False,
        ),
        (
            "05_streaming_zero_shot",
            {
                **base,
                "text": "这是流式语音合成测试。请注意每个音频片段拼接后的连贯程度。",
            },
            True,
        ),
        (
            "06_backend_hmonnx_reference",
            {
                **base,
                "text": "这条样本使用 HMONNX 音频编码后端处理参考音频。",
                "reference_wav_path": args.reference_wav,
            },
            False,
        ),
    ]

    summary = {
        "work_dir": str(Path(args.work_dir).expanduser().resolve()),
        "model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "reference_wav": str(Path(args.reference_wav).expanduser().resolve()),
        "device": device,
        "sample_rate": int(pipeline.sample_rate),
        "audio_encoder_backend": args.audio_encoder_backend,
        "cases": [],
    }

    for name, kwargs, streaming in cases:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        summary["cases"].append(run_case(name, pipeline, output_dir, kwargs, streaming=streaming))

    align_kwargs = {
        **base,
        "text": "这是端到端 PyTorch 对齐测试，用来生成 HMONNX 和浮点模型的对比音频。",
    }
    summary["align"] = run_align_case("07_align_torch_zero_shot", pipeline, output_dir, align_kwargs, args)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="/data01/home/she.gao/xh2modelzoo/examples/audio/voxcpm2/work_dirs/VoxCPM2_XH2a")
    parser.add_argument("--model-dir", default="/data01/nfs_shared/ASR_TTS/VoxCPM2")
    parser.add_argument("--reference-wav", default="/data01/nfs_shared/ASR_TTS/CAM++/examples/speaker1_b_cn_16k.wav")
    parser.add_argument("--output-dir", default="/data01/home/she.gao/xh2modelzoo/examples/audio/voxcpm2/demo_results/full_demo_suite")
    parser.add_argument("--audio-encoder-backend", choices=["auto", "hmonnx", "torch"], default="hmonnx")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
