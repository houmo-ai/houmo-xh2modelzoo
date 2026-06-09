import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xh_model_zoo.xh_aigc.models.flux2_klein import (
    Flux2KleinHMONNXPipeline,
    Flux2KleinTextEncoderInference,
    attach_flux2_klein_hmonnx_components,
)


DIFFUSERS_INSTALL_HINT = (
    "当前 diffusers 版本不包含 Flux2KleinPipeline。\n"
    "请先升级到支持 FLUX.2-klein 的版本，例如:\n"
    "  pip install git+https://github.com/huggingface/diffusers.git\n"
    "当前模型目录 /data02/datasets/flux-4b 的 model_index.json 标记为 0.37.0.dev0。"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def import_flux2_klein_pipeline():
    vendored_diffusers = str(_repo_root() / "data" / "difs" / "diffusers-main" / "src")
    if vendored_diffusers in sys.path:
        sys.path.remove(vendored_diffusers)
    for module_name in list(sys.modules):
        if module_name == "diffusers" or module_name.startswith("diffusers."):
            del sys.modules[module_name]

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise SystemExit(DIFFUSERS_INSTALL_HINT) from exc

    return Flux2KleinPipeline


def parse_args() -> argparse.Namespace:
    default_image = _repo_root() / "examples" / "aigc" / "flux2_klein" / "flux2_klein_hmonnx.png"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/flux-4b")
    parser.add_argument("--meta", type=str, required=True, help="导出的 text_encoder meta.json 或根 meta.json 路径")
    parser.add_argument("--root-meta", type=str, default=None, help="导出的根 meta.json 路径，用于 transformer/vae")
    parser.add_argument(
        "--components",
        type=str,
        default="text_encoder,transformer,vae_encoder,vae",
        help="逗号分隔启用组件: text_encoder,transformer,vae_encoder,vae。",
    )
    parser.add_argument("--image", type=str, default=str(default_image), help="草稿图/参考图路径")
    parser.add_argument("--prompt", type=str, default="make the image more cinematic and detailed")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--skip-image", action="store_true", help="只做 embedding 对比，不跑图像编辑")
    parser.add_argument("--skip-compare", action="store_true", help="跳过 text_encoder embedding 对比")
    parser.add_argument(
        "--output",
        type=str,
        default="work_dirs/flux2-klein-4b/edit_hmonnx_demo/flux2_klein_hmonnx_edit.png",
        help="编辑后图片保存路径",
    )
    return parser.parse_args()


def parse_components(value: str) -> set[str]:
    components = {item.strip().lower() for item in value.split(",") if item.strip()}
    aliases = {"te": "text_encoder", "text": "text_encoder", "trans": "transformer", "vae_decoder": "vae", "vae_encode": "vae_encoder"}
    components = {aliases.get(item, item) for item in components}
    supported = {"text_encoder", "transformer", "vae", "vae_encoder"}
    unknown = components - supported
    if unknown:
        raise SystemExit(f"不支持的组件: {sorted(unknown)}，支持: {sorted(supported)}")
    return components


def resolve_text_meta(meta_path: Path, root_meta_path: Path | None) -> Path:
    meta_info = json_load(meta_path)
    if "text_encoder_meta" in meta_info:
        return meta_path.parent / meta_info["text_encoder_meta"]
    if root_meta_path is not None:
        root_info = json_load(root_meta_path)
        if "text_encoder_meta" in root_info:
            return root_meta_path.parent / root_info["text_encoder_meta"]
    return meta_path


def main() -> None:
    args = parse_args()
    components = parse_components(args.components)
    import_flux2_klein_pipeline()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    root_meta_path = Path(args.root_meta) if args.root_meta is not None else None
    image_path = Path(args.image)
    if not model_path.exists():
        raise SystemExit(f"模型目录不存在: {model_path}")
    if not meta_path.exists():
        raise SystemExit(f"meta.json 不存在: {meta_path}")
    if not image_path.exists():
        raise SystemExit(f"草稿图不存在: {image_path}")
    if root_meta_path is None and ({"transformer", "vae", "vae_encoder"} & components):
        root_meta_path = meta_path
    if root_meta_path is not None and not root_meta_path.exists():
        raise SystemExit(f"root meta.json 不存在: {root_meta_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = Flux2KleinHMONNXPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    pipe = pipe.to(device)

    text_meta_path = resolve_text_meta(meta_path, root_meta_path)
    text_meta = json_load(text_meta_path)

    hm_text_encoder = Flux2KleinTextEncoderInference.from_meta(
        text_meta_path,
        tokenizer=pipe.tokenizer,
        device=device,
        dtype=dtype,
    )
    hm_prompt_embeds, hm_model_inputs = hm_text_encoder.get_prompt_embeds(args.prompt)

    if components:
        attach_root_meta = root_meta_path if root_meta_path is not None else meta_path
        pipe, loaded_components = attach_flux2_klein_hmonnx_components(
            pipe,
            components=components,
            meta_path=text_meta_path,
            root_meta_path=attach_root_meta,
            device=device,
            dtype=dtype,
        )
        if "text_encoder" in loaded_components:
            hm_text_encoder = loaded_components["text_encoder"]
        for component_name in sorted(loaded_components):
            print(f"attached_{component_name}_hmonnx=True")

    if args.skip_image:
        return

    draft_image = Image.open(image_path).convert("RGB")
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    image = pipe(
        image=draft_image,
        prompt=None,
        prompt_embeds=hm_prompt_embeds,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
        use_hmonnx_text_encoder="text_encoder" in components,
        use_hmonnx_transformer="transformer" in components,
        use_hmonnx_vae="vae" in components,
    ).images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"saved={output_path}")
    print(f"draft_image={image_path}")
    print(f"input_ids_shape={list(hm_model_inputs['input_ids'].shape)}")
    print(f"attention_mask_shape={list(hm_model_inputs['attention_mask'].shape)}")


def json_load(path: Path):
    import json

    return json.load(open(path, "r"))


if __name__ == "__main__":
    main()
