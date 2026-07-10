import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from xhquant.api import set_random_seed

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from hmonnx_utils import build_hmonnx_model, build_voice_clone_prompt, infer_request, load_export_meta


SAMPLES = [
    "后摩智能正在推进高效人工智能芯片和软件栈的协同优化。",
    "这个小样本评估只验证迁移后的 HMONNX 链路可以稳定生成音频。",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    set_random_seed(args.seed)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir) if args.output_dir else work_dir / "eval_small"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_meta = load_export_meta(work_dir)
    model = build_hmonnx_model(work_dir, args.device)
    results = []

    for idx, text in enumerate(SAMPLES[: max(1, args.max_samples)]):
        args.text = text
        request = infer_request(export_meta, args)
        voice_clone_prompt = build_voice_clone_prompt(work_dir, export_meta, request, args.device, args.xvec_only)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
        }
        if voice_clone_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = voice_clone_prompt
            request["ref_audio"] = None
        start = time.time()
        wavs, sr = model.generate_by_mode(**request, **gen_kwargs)
        elapsed = time.time() - start
        wav_file = output_dir / f"sample_{idx:03d}.wav"
        sf.write(wav_file, wavs[0], sr)
        results.append({"index": idx, "text": text, "output": str(wav_file), "sample_rate": sr, "elapsed": elapsed})
        print(f"sample {idx}: {wav_file} elapsed={elapsed:.2f}s")

    report = output_dir / "eval_small_report.json"
    report.write_text(json.dumps(results, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"report saved to {report}")


if __name__ == "__main__":
    main()
