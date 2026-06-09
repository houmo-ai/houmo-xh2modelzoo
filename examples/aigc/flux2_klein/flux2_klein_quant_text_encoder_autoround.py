import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "/data02/datasets/flux-4b"
DEFAULT_OUTPUT_DIR = "work_dirs/flux2-klein-4b/autoround-w8-text-encoder"
DEFAULT_PROMPTS = (
    "A cat holding a sign that says hello world",
    "A cinematic photo of a small robot reading a book in a neon library",
    "A watercolor landscape with mountains, river, and sunrise",
    "A product photo of a transparent glass teapot on a wooden table",
    "An astronaut riding a horse on Mars, detailed, sharp focus",
    "A cozy bedroom with warm lighting and a window looking out at snow",
    "A futuristic city skyline at night, ultra detailed",
    "A close-up portrait of a golden retriever wearing sunglasses",
)


DTYPE_MAP = {
    "auto": None,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use AutoRound to W8-quantize the Qwen3 text_encoder inside FLUX.2-klein.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR, help="FLUX.2-klein model directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output FLUX directory. Non-text_encoder assets are linked/copied from --model.",
    )
    parser.add_argument("--bits", type=int, default=8, help="Weight bits for text_encoder")
    parser.add_argument("--group-size", type=int, default=128, help="Weight group size")
    parser.add_argument("--sym", action="store_true", default=True, help="Use symmetric weight quantization")
    parser.add_argument("--asym", dest="sym", action="store_false", help="Use asymmetric weight quantization")
    parser.add_argument("--iters", type=int, default=0, help="AutoRound iters. 0 means RTN, recommended for W8")
    parser.add_argument("--nsamples", type=int, default=32, help="Calibration sample count")
    parser.add_argument("--seqlen", type=int, default=512, help="Calibration sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Calibration batch size")
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k", help="Fallback AutoRound dataset")
    parser.add_argument(
        "--calib-prompt",
        action="append",
        default=[],
        help="Calibration prompt. Can be passed multiple times. Uses Qwen3 chat template.",
    )
    parser.add_argument(
        "--calib-prompts-file",
        type=str,
        default=None,
        help="Optional txt/jsonl calibration prompts file. txt: one prompt per line; jsonl uses prompt/text/content field.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="AutoRound device_map, e.g. auto/cpu/cuda/cuda:0/0. For one GPU, cuda:0 or 0 is fine.",
    )
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="auto", help="Load dtype for text_encoder")
    parser.add_argument(
        "--format",
        type=str,
        default="fake",
        help="Save format. Keep fake for a normal HF text_encoder usable by Flux2KleinPipeline.",
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="Copy non-text_encoder Flux assets instead of symlinking them.",
    )
    parser.add_argument(
        "--no-assets",
        action="store_true",
        help="Only save quantized text_encoder to output-dir/text_encoder; do not link/copy other Flux assets.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        default=True,
        help="Enable AutoRound low CPU memory mode.",
    )
    parser.add_argument(
        "--no-low-cpu-mem-usage",
        dest="low_cpu_mem_usage",
        action="store_false",
        help="Disable AutoRound low CPU memory mode.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name != "auto":
        return DTYPE_MAP[dtype_name]
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_device_arg(device: str):
    if device == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return int(device)
    return device


def read_prompt_file(path: str) -> list[str]:
    prompt_path = Path(path)
    prompts: list[str] = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if prompt_path.suffix.lower() == ".jsonl":
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("text") or item.get("content")
                if prompt is None:
                    raise ValueError(f"No prompt/text/content field in {prompt_path}: {line[:120]}")
                prompts.append(str(prompt))
            else:
                prompts.append(line)
    return prompts


def build_chat_calib_dataset(tokenizer, prompts: Iterable[str], seqlen: int, nsamples: int) -> list[dict[str, torch.Tensor]]:
    prompt_list = list(prompts)
    if not prompt_list:
        prompt_list = list(DEFAULT_PROMPTS)
    if len(prompt_list) < nsamples:
        repeat = (nsamples + len(prompt_list) - 1) // len(prompt_list)
        prompt_list = (prompt_list * repeat)[:nsamples]
    else:
        prompt_list = prompt_list[:nsamples]

    dataset: list[dict[str, torch.Tensor]] = []
    for prompt in prompt_list:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seqlen,
        )
        dataset.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})
    return dataset


def link_or_copy(src: Path, dst: Path, copy_assets: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if copy_assets:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def prepare_output_flux_dir(source_dir: Path, output_dir: Path, copy_assets: bool, no_assets: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_encoder_out = output_dir / "text_encoder"
    if text_encoder_out.exists() or text_encoder_out.is_symlink():
        if text_encoder_out.is_dir() and not text_encoder_out.is_symlink():
            shutil.rmtree(text_encoder_out)
        else:
            text_encoder_out.unlink()

    if no_assets:
        return text_encoder_out

    for item in source_dir.iterdir():
        if item.name in {"text_encoder", "._____temp", ".msc", ".mv"}:
            continue
        link_or_copy(item, output_dir / item.name, copy_assets=copy_assets)
    return text_encoder_out


def main() -> None:
    args = parse_args()
    start_time = time.time()

    source_dir = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    text_encoder_dir = source_dir / "text_encoder"
    tokenizer_dir = source_dir / "tokenizer"
    if not source_dir.exists():
        raise SystemExit(f"模型目录不存在: {source_dir}")
    if not text_encoder_dir.exists():
        raise SystemExit(f"text_encoder 目录不存在: {text_encoder_dir}")
    if not tokenizer_dir.exists():
        raise SystemExit(f"tokenizer 目录不存在: {tokenizer_dir}")

    if args.format.lower() != "fake":
        print("WARNING: non-fake formats may not be directly loadable by Flux2KleinPipeline.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = resolve_dtype(args.dtype)
    device_map = resolve_device_arg(args.device)
    print(f"Loading tokenizer: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    print(f"Loading text_encoder: {text_encoder_dir}")
    text_encoder = AutoModelForCausalLM.from_pretrained(
        text_encoder_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    prompts = list(args.calib_prompt)
    if args.calib_prompts_file:
        prompts.extend(read_prompt_file(args.calib_prompts_file))
    dataset = build_chat_calib_dataset(tokenizer, prompts, args.seqlen, args.nsamples) if prompts else args.dataset

    print("\nAutoRound text_encoder quantization")
    print(f"  bits/group/sym: W{args.bits}G{args.group_size}, sym={args.sym}")
    print(f"  iters/nsamples/seqlen/batch: {args.iters}/{args.nsamples}/{args.seqlen}/{args.batch_size}")
    print(f"  dtype/device_map: {dtype}/{device_map}")
    print(f"  calibration: {'chat prompts' if prompts else args.dataset}")

    from auto_round import AutoRound

    autoround = AutoRound(
        model=text_encoder,
        tokenizer=tokenizer,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        act_bits=16,
        iters=args.iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        dataset=dataset,
        seed=args.seed,
        device_map=device_map,
        quant_lm_head=False,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    quant_start = time.time()
    autoround.quantize()
    print(f"Quantization finished in {time.time() - quant_start:.1f}s")

    text_encoder_out = prepare_output_flux_dir(source_dir, output_dir, args.copy_assets, args.no_assets)
    print(f"Saving quantized text_encoder to: {text_encoder_out}")
    autoround.save_quantized(str(text_encoder_out), format=args.format)

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "source_model_dir": str(source_dir),
        "source_text_encoder_dir": str(text_encoder_dir),
        "output_dir": str(output_dir),
        "text_encoder_dir": str(text_encoder_out),
        "autoround": {
            "bits": args.bits,
            "group_size": args.group_size,
            "sym": args.sym,
            "act_bits": 16,
            "iters": args.iters,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "format": args.format,
            "quant_lm_head": False,
        },
    }
    with (output_dir / "autoround_text_encoder_w8_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=4, ensure_ascii=False)

    print("\nDone")
    print(f"  output={output_dir}")
    print(f"  text_encoder={text_encoder_out}")
    print(f"  total_time={time.time() - start_time:.1f}s")
    if not args.no_assets:
        print("  You can test it with:")
        print(f"    python examples/aigc/flux2_klein/flux2_klein_fp_demo.py --model {output_dir}")


if __name__ == "__main__":
    main()
