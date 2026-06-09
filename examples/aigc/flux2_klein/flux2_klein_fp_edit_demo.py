import argparse
from pathlib import Path

import torch
from PIL import Image


DIFFUSERS_INSTALL_HINT = (
    "当前 diffusers 版本不包含 Flux2KleinPipeline。\n"
    "请先升级到支持 FLUX.2-klein 的版本，例如:\n"
    "  pip install git+https://github.com/huggingface/diffusers.git\n"
    "当前模型目录 /data02/datasets/flux-4b 的 model_index.json 标记为 0.37.0.dev0。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/flux-4b", help="FLUX.2-klein-4B 本地模型目录")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Edit the input image into a cinematic high quality version while preserving the main composition.",
    )
    parser.add_argument(
        "--image",
        action="append",
        # default=[str(Path(__file__).with_name("flux2_klein_hmonnx.png"))],
        help="参考图路径。可重复传入多次，用于 image editing / multi-reference editing。",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4, help="README 推荐的最小推理步数")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="浮点精度。auto 在 CUDA 上优先 bfloat16，其次 float16；CPU 上使用 float32。",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="推理设备。",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="启用模型 CPU offload，降低显存占用。仅 CUDA 下有效。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="work_dirs/flux2-klein-4b/fp_edit_demo/flux2_klein_edit.png",
        help="输出图片路径",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32

    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_images(image_paths: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
    return images


def import_flux2_klein_pipeline():
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise SystemExit(DIFFUSERS_INSTALL_HINT) from exc

    return Flux2KleinPipeline


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    Flux2KleinPipeline = import_flux2_klein_pipeline()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"模型目录不存在: {model_path}")

    if device == "cpu" and args.cpu_offload:
        raise SystemExit("--cpu-offload 仅在 CUDA 推理时有意义。")

    pipe = Flux2KleinPipeline.from_pretrained(str(model_path), torch_dtype=dtype)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    reference_images = load_images(args.image)
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    infer_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.steps,
        "generator": generator,
    }
    if reference_images:
        infer_kwargs["image"] = reference_images if len(reference_images) > 1 else reference_images[0]

    image = pipe(**infer_kwargs).images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"reference_images={len(reference_images)}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()