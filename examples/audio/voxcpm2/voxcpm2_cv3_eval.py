#!/usr/bin/env python3
"""VoxCPM2 CV3-Eval generation script.

This follows the style of examples/audio/qwen3_tts/eval/qwen3_tts_eval.py:
  * choose one inference mode per run: float or hmonnx
  * shard CV3-Eval samples across the requested GPUs
  * generate wav files only; no metric computation is done here

Examples:
    # HMONNX, generate all 500 zh + 500 en samples on all visible GPUs
    PYTHONPATH=/data01/home/she.gao/xh2modelzoo CUDA_VISIBLE_DEVICES=0,1,2,3 \
    /data01/home/she.gao/miniconda3/envs/xhquant/bin/python voxcpm2_cv3_eval.py \
      --mode hmonnx --languages zh,en --gpus auto

    # Float/PyTorch, first 2 samples per language
    PYTHONPATH=/data01/home/she.gao/xh2modelzoo CUDA_VISIBLE_DEVICES=0 \
    /data01/home/she.gao/miniconda3/envs/xhquant/bin/python voxcpm2_cv3_eval.py \
      --mode float --languages zh,en --gpus 0 --max-samples 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.multiprocessing import set_start_method, spawn
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hmonnx_demo import save_wav  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def load_voxcpm2_model(model_dir: str, device: torch.device):
    try:
        from voxcpm import VoxCPM2Model
    except ImportError:
        from voxcpm.model.voxcpm2 import VoxCPM2Model

    model = VoxCPM2Model.from_local(model_dir, optimize=False, training=False)
    model.to(device)
    model.eval()
    return model


def read_kaldi_map(path: Path, max_samples: int | None = None) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                logging.warning("skip malformed line in %s: %s", path, line.rstrip())
                continue
            result[parts[0]] = parts[1]
            if max_samples is not None and len(result) >= max_samples:
                break
    return result


def load_cv3_zero_shot(dataset_dir: Path, max_samples: int | None = None) -> List[dict]:
    required = ["text", "prompt_text", "prompt_wav.scp"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset_dir} missing files: {missing}")

    text = read_kaldi_map(dataset_dir / "text")
    prompt_text = read_kaldi_map(dataset_dir / "prompt_text")
    prompt_wav = read_kaldi_map(dataset_dir / "prompt_wav.scp", max_samples=max_samples)

    samples = []
    for utt, wav in prompt_wav.items():
        if utt not in text or utt not in prompt_text:
            logging.warning("skip %s because text/prompt_text is missing", utt)
            continue
        samples.append(
            {
                "uttid": utt,
                "text": text[utt],
                "prompt_text": prompt_text[utt],
                "prompt_wav": wav,
            }
        )
    return samples


def resolve_cv3_wav(cv3_root: Path, wav_path: str) -> str:
    path = Path(wav_path)
    if path.is_absolute():
        return str(path)
    return str(cv3_root / path)


def make_gen_kwargs(args: argparse.Namespace, sample: dict, cv3_root: Path) -> dict:
    prompt_wav = resolve_cv3_wav(cv3_root, sample["prompt_wav"])
    if args.cv3_mode == "prompt":
        return {
            "text": sample["text"],
            "prompt_wav_path": prompt_wav,
            "prompt_text": sample["prompt_text"],
            "reference_wav_path": None,
        }
    if args.cv3_mode == "reference":
        return {
            "text": sample["text"],
            "prompt_wav_path": None,
            "prompt_text": None,
            "reference_wav_path": prompt_wav,
        }
    if args.cv3_mode == "reference_prompt":
        return {
            "text": sample["text"],
            "prompt_wav_path": prompt_wav,
            "prompt_text": sample["prompt_text"],
            "reference_wav_path": prompt_wav,
        }
    raise ValueError(f"unknown cv3_mode: {args.cv3_mode}")


def flatten_audio(audio) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def generate_hmonnx(pipeline, kwargs: dict, args: argparse.Namespace, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    audio = pipeline.generate(
        text=kwargs["text"],
        prompt_wav_path=kwargs["prompt_wav_path"],
        prompt_text=kwargs["prompt_text"],
        reference_wav_path=kwargs["reference_wav_path"],
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    return flatten_audio(audio)


def generate_float(model, kwargs: dict, args: argparse.Namespace, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    audio = model.generate(
        target_text=kwargs["text"],
        prompt_wav_path=kwargs["prompt_wav_path"] or "",
        prompt_text=kwargs["prompt_text"] or "",
        reference_wav_path=kwargs["reference_wav_path"] or "",
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    return flatten_audio(audio)


def worker(
    rank: int,
    gpus: List[int],
    args: argparse.Namespace,
    language: str,
    samples: List[dict],
    output_dir: str,
) -> None:
    gpu_id = gpus[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    output_root = Path(output_dir)
    lang_dir = output_root / args.mode / language
    lang_dir.mkdir(parents=True, exist_ok=True)
    rank_manifest = output_root / f"generated.{args.mode}.{language}.rank{rank}.jsonl"

    if args.mode == "hmonnx":
        from xh_model_zoo.xh_llm.models.voxcpm2 import VoxCPM2HMONNXTTSPipeline

        logging.info("rank=%d gpu=%d loading HMONNX pipeline", rank, gpu_id)
        engine = VoxCPM2HMONNXTTSPipeline(
            work_dir=args.work_dir,
            device=str(device),
            audio_encoder_backend=args.audio_encoder_backend,
            torch_audio_model_dir=args.model_dir,
        )
        sample_rate = int(engine.sample_rate)
    else:
        logging.info("rank=%d gpu=%d loading float VoxCPM2 model", rank, gpu_id)
        engine = load_voxcpm2_model(args.model_dir, device)
        sample_rate = int(args.sample_rate)

    cv3_root = Path(args.cv3_root)
    with rank_manifest.open("w", encoding="utf-8") as f:
        for idx in tqdm(range(rank, len(samples), len(gpus)), desc=f"{args.mode} {language} gpu{gpu_id}"):
            sample = samples[idx]
            utt = sample["uttid"]
            wav_path = lang_dir / f"{utt}.wav"

            record = {
                "language": language,
                "uttid": utt,
                "text": sample["text"],
                "prompt_text": sample["prompt_text"],
                "prompt_wav": resolve_cv3_wav(cv3_root, sample["prompt_wav"]),
                "mode": args.mode,
                "cv3_mode": args.cv3_mode,
                "wav": str(wav_path),
                "skipped": False,
                "ok": False,
            }

            if wav_path.exists() and not args.overwrite:
                record.update({"skipped": True, "ok": True})
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                continue

            try:
                gen_kwargs = make_gen_kwargs(args, sample, cv3_root)
                t0 = time.time()
                if args.mode == "hmonnx":
                    audio = generate_hmonnx(engine, gen_kwargs, args, args.seed)
                else:
                    audio = generate_float(engine, gen_kwargs, args, args.seed)
                elapsed = time.time() - t0
                save_wav(audio, sample_rate, str(wav_path))
                record.update(
                    {
                        "ok": True,
                        "elapsed_sec": elapsed,
                        "sample_rate": sample_rate,
                        "num_samples": int(audio.shape[0]),
                        "duration_sec": float(audio.shape[0] / sample_rate) if sample_rate else 0.0,
                    }
                )
                logging.info("generated %s", wav_path)
            except Exception as exc:
                logging.exception("failed %s/%s", language, utt)
                record.update({"error": repr(exc)})

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def parse_gpus(gpus: str) -> List[int]:
    if gpus.strip().lower() == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        count = torch.cuda.device_count()
        if count <= 0:
            raise RuntimeError("no visible CUDA devices")
        return list(range(count))

    result = [int(item.strip()) for item in gpus.split(",") if item.strip()]
    if not result:
        raise ValueError("--gpus must contain at least one GPU id")
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if any(gpu < 0 or gpu >= visible for gpu in result):
            raise ValueError(f"GPU ids must be in [0, {visible}), got {result}")
    return result


def merge_manifests(output_dir: Path, mode: str, languages: List[str], num_ranks: int) -> dict:
    summary = {"mode": mode, "languages": {}, "total": 0, "ok": 0, "failed": 0, "skipped": 0}
    for language in languages:
        merged = output_dir / f"generated.{mode}.{language}.jsonl"
        rows = []
        for rank in range(num_ranks):
            part = output_dir / f"generated.{mode}.{language}.rank{rank}.jsonl"
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        with merged.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ok = sum(1 for row in rows if row.get("ok"))
        skipped = sum(1 for row in rows if row.get("skipped"))
        failed = len(rows) - ok
        summary["languages"][language] = {
            "num_samples": len(rows),
            "ok": ok,
            "failed": failed,
            "skipped": skipped,
            "manifest": str(merged),
            "wav_dir": str(output_dir / mode / language),
        }
        summary["total"] += len(rows)
        summary["ok"] += ok
        summary["failed"] += failed
        summary["skipped"] += skipped
    return summary


def validate_paths(args: argparse.Namespace) -> None:
    paths = [args.cv3_root, args.data_path, args.model_dir]
    if args.mode == "hmonnx":
        paths.append(args.work_dir)
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required paths:\n  " + "\n  ".join(missing))


def main(args: argparse.Namespace) -> None:
    validate_paths(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    if not languages:
        raise ValueError("--languages is empty")

    gpus = parse_gpus(args.gpus)
    output_dir = Path(args.exp_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["exp_dir"] = str(output_dir)
    (output_dir / f"run_config.{args.mode}.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    data_root = Path(args.data_path)
    for language in languages:
        dataset_dir = data_root / language
        samples = load_cv3_zero_shot(dataset_dir, max_samples=args.max_samples)
        logging.info("language=%s loaded %d samples from %s", language, len(samples), dataset_dir)
        if not samples:
            continue
        spawn(
            worker,
            args=(gpus, args, language, samples, str(output_dir)),
            nprocs=len(gpus),
            join=True,
        )

    summary = merge_manifests(output_dir, args.mode, languages, len(gpus))
    summary.update({"exp_dir": str(output_dir), "languages_requested": languages})
    (output_dir / f"summary.{args.mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("summary saved to %s", output_dir / f"summary.{args.mode}.json")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoxCPM2 CV3-Eval wav generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True, choices=["float", "hmonnx"], help="inference mode")
    parser.add_argument("--model-dir", default="/data01/nfs_shared/ASR_TTS/VoxCPM2")
    parser.add_argument(
        "--work-dir",
        default=str(SCRIPT_DIR / "work_dirs" / "VoxCPM2_XH2a"),
        help="VoxCPM2 HMONNX export directory; only used by --mode hmonnx",
    )
    parser.add_argument("--cv3-root", default="/data01/home/she.gao/CV3-Eval")
    parser.add_argument("--data-path", default="/data01/home/she.gao/CV3-Eval/data/zero_shot")
    parser.add_argument("--languages", default="zh,en")
    parser.add_argument("--gpus", default="auto", help="comma-separated visible GPU ids, or auto for all visible GPUs")
    parser.add_argument("--max-samples", type=int, default=None, help="None means all samples, usually 500/language")
    parser.add_argument("--exp-dir", default=str(SCRIPT_DIR / "cv3_eval_results" / "voxcpm2_cv3"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cv3-mode", choices=["prompt", "reference", "reference_prompt"], default="prompt")
    parser.add_argument("--audio-encoder-backend", choices=["auto", "hmonnx", "torch"], default="hmonnx")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=48000, help="float VoxCPM2 output sample rate")
    return parser


if __name__ == "__main__":
    main(build_argparser().parse_args())
