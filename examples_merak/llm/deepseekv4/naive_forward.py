import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import PreTrainedTokenizerFast


DEFAULT_MODEL_PATH = "/data01/datasets/DeepSeek-V4-Flash"
DEFAULT_OUTPUT_PATH = "/data01/datasets/DeepSeek-V4-Flash-mp4"


def resolve_paths(model_path: str) -> tuple[Path, Path, Path, Path, Path]:
    model_root = Path(model_path).resolve()
    inference_dir = model_root / "inference"
    encoding_dir = model_root / "encoding"
    convert_script = inference_dir / "convert.py"
    generate_script = inference_dir / "generate.py"
    inference_config = inference_dir / "config.json"
    if not inference_dir.exists():
        raise FileNotFoundError(f"Missing inference directory: {inference_dir}")
    if not encoding_dir.exists():
        raise FileNotFoundError(f"Missing encoding directory: {encoding_dir}")
    if not convert_script.exists():
        raise FileNotFoundError(f"Missing convert script: {convert_script}")
    if not generate_script.exists():
        raise FileNotFoundError(f"Missing generate script: {generate_script}")
    if not inference_config.exists():
        raise FileNotFoundError(f"Missing inference config: {inference_config}")
    return model_root, inference_dir, encoding_dir, convert_script, generate_script


def load_encoder(model_path: str):
    model_root, _, encoding_dir, _, _ = resolve_paths(model_path)
    module_path = encoding_dir / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load encoder module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encode_messages = module.encode_messages
    return model_root, encode_messages


def build_python_command() -> list[str]:
    return [sys.executable]


def run_command(cmd: list[str], cwd: Path | None = None, env_updates: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def encode_prompt(model_path: str, prompt: str, thinking_mode: str) -> None:
    model_root, encode_messages = load_encoder(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_root))
    encoded_prompt = encode_messages([{"role": "user", "content": prompt}], thinking_mode=thinking_mode)
    token_ids = tokenizer.encode(encoded_prompt)
    print("Prompt string:\n")
    print(encoded_prompt)
    print("\nToken count:", len(token_ids))
    print("First 32 token ids:", token_ids[:32])


def prepare_ckpt(model_path: str, output_path: str, model_parallel: int, n_experts: int, expert_dtype: str | None) -> None:
    _, _, _, convert_script, _ = resolve_paths(model_path)
    cmd = build_python_command() + [
        str(convert_script),
        "--hf-ckpt-path",
        model_path,
        "--save-path",
        output_path,
        "--n-experts",
        str(n_experts),
        "--model-parallel",
        str(model_parallel),
    ]
    if expert_dtype:
        cmd.extend(["--expert-dtype", expert_dtype])
    run_command(cmd)


def interactive_chat(
    model_path: str,
    ckpt_path: str,
    model_parallel: int,
    max_new_tokens: int,
    temperature: float,
    cuda_visible_devices: str,
    device: str,
) -> None:
    if device != "cuda":
        raise RuntimeError(
            "DeepSeek-V4 official inference is GPU-only: upstream generate.py hardcodes CUDA/NCCL and depends on TileLang GPU kernels. "
            "CPU can be used for prepare/encode, but not for chat generation."
        )
    _, _, _, _, generate_script = resolve_paths(model_path)
    cmd = [
        "torchrun",
        "--nproc-per-node",
        str(model_parallel),
        str(generate_script),
        "--ckpt-path",
        ckpt_path,
        "--config",
        str(Path(model_path) / "inference" / "config.json"),
        "--interactive",
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
    ]
    env_updates = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices} if cuda_visible_devices else None
    run_command(cmd, env_updates=env_updates)


def single_turn_chat(
    model_path: str,
    ckpt_path: str,
    model_parallel: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    cuda_visible_devices: str,
    device: str,
) -> None:
    if device != "cuda":
        raise RuntimeError(
            "DeepSeek-V4 official inference is GPU-only: upstream generate.py hardcodes CUDA/NCCL and depends on TileLang GPU kernels. "
            "CPU can be used for prepare/encode, but not for chat generation."
        )
    _, _, _, _, generate_script = resolve_paths(model_path)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        handle.write("\n")
        input_file = handle.name
    try:
        cmd = [
            "torchrun",
            "--nproc-per-node",
            str(model_parallel),
            str(generate_script),
            "--ckpt-path",
            ckpt_path,
            "--config",
            str(Path(model_path) / "inference" / "config.json"),
            "--input-file",
            input_file,
            "--max-new-tokens",
            str(max_new_tokens),
            "--temperature",
            str(temperature),
        ]
        env_updates = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices} if cuda_visible_devices else None
        run_command(cmd, env_updates=env_updates)
    finally:
        os.unlink(input_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-V4 official local inference from a local model directory.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode one user prompt with the official DeepSeek-V4 encoder.")
    encode_parser.add_argument("--prompt", required=True)
    encode_parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")

    prepare_parser = subparsers.add_parser("prepare", help="Convert original Hugging Face weights to the official inference format.")
    prepare_parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    prepare_parser.add_argument("--model-parallel", type=int, default=4)
    prepare_parser.add_argument("--n-experts", type=int, default=256)
    prepare_parser.add_argument("--expert-dtype", choices=["fp4", "fp8"], default=None)

    chat_parser = subparsers.add_parser("chat", help="Run official DeepSeek-V4 inference.")
    chat_parser.add_argument("--ckpt-path", default=DEFAULT_OUTPUT_PATH)
    chat_parser.add_argument("--model-parallel", type=int, default=4)
    chat_parser.add_argument("--max-new-tokens", type=int, default=300)
    chat_parser.add_argument("--temperature", type=float, default=1.0)
    chat_parser.add_argument("--prompt", default="")
    chat_parser.add_argument("--cuda-visible-devices", default="")
    chat_parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "encode":
        encode_prompt(args.model_path, args.prompt, args.thinking_mode)
        return
    if args.command == "prepare":
        prepare_ckpt(args.model_path, args.output_path, args.model_parallel, args.n_experts, args.expert_dtype)
        return
    if args.command == "chat":
        if args.prompt:
            single_turn_chat(
                args.model_path,
                args.ckpt_path,
                args.model_parallel,
                args.prompt,
                args.max_new_tokens,
                args.temperature,
                args.cuda_visible_devices,
                args.device,
            )
        else:
            interactive_chat(
                args.model_path,
                args.ckpt_path,
                args.model_parallel,
                args.max_new_tokens,
                args.temperature,
                args.cuda_visible_devices,
                args.device,
            )
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()