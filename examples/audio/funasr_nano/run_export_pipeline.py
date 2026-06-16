"""Run the standalone FunASR-Nano XH2a export pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", default="/data01/datasets/Funasr/Fun-ASR-Nano-2512")
    parser.add_argument("--qwen3-dir", default=None, help="Defaults to <model-dir>/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="work_dirs/funasr_nano_xh2a")
    parser.add_argument("--audio", default=None, help="Optional sample audio for audio ONNX dummy shape")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--quant-type", default="w8a8h1_sefp")
    parser.add_argument("--skip-audio-onnx", action="store_true")
    parser.add_argument("--skip-audio-hmonnx", action="store_true")
    parser.add_argument("--skip-qwen3", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    qwen3_dir = args.qwen3_dir or str(Path(args.model_dir).expanduser().resolve() / "Qwen3-0.6B")

    if not args.skip_audio_onnx:
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "export_audio_modules_onnx.py"),
            "--model-dir",
            args.model_dir,
            "--work-dir",
            args.work_dir,
            "--max-frames",
            str(args.max_frames),
        ]
        if args.audio:
            cmd.extend(["--audio", args.audio])
        _run(cmd)

    if not args.skip_audio_hmonnx:
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "convert_hmonnx.py"),
            "--work-dir",
            args.work_dir,
            "--quant-type",
            args.quant_type,
        ]
        if args.debug:
            cmd.append("--debug")
        _run(cmd)

    if not args.skip_qwen3:
        cmd = [
            PYTHON,
            str(SCRIPT_DIR / "export_qwen3_llm_hmonnx.py"),
            "--model-dir",
            qwen3_dir,
            "--work-dir",
            args.work_dir,
            "--context-length",
            str(args.context_length),
            "--input-sequence-length",
            str(args.input_sequence_length),
            "--quant-type",
            args.quant_type,
        ]
        if args.debug:
            cmd.append("--debug")
        _run(cmd)

    print(f"Pipeline finished. Work dir: {Path(args.work_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
