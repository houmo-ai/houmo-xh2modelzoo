import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/flux-4b")
    parser.add_argument("--meta", type=str, required=True, help="导出的 text_encoder meta.json 或根 meta.json 路径")
    parser.add_argument("--root-meta", type=str, default=None, help="导出的根 meta.json 路径，用于 transformer/vae")
    parser.add_argument(
        "--components",
        type=str,
        default="text_encoder",
        help="逗号分隔启用组件: text_encoder,transformer,vae。vae 当前做独立组件验证。",
    )
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--min-cosine", type=float, default=0.999, help="embedding cosine similarity threshold")
    parser.add_argument("--skip-image", action="store_true", help="只做 embedding 对比，不跑整图生成")
    parser.add_argument("--skip-compare", action="store_true", help="跳过 text_encoder embedding 对比")
    parser.add_argument(
        "--output",
        type=str,
        default="work_dirs/flux2-klein-4b/text_encoder_demo/flux2_klein_hmonnx.png",
        help="生成图片保存路径",
    )
    return parser.parse_args()


def parse_components(value: str) -> set[str]:
    components = {item.strip().lower() for item in value.split(",") if item.strip()}
    aliases = {"te": "text_encoder", "text": "text_encoder", "trans": "transformer", "vae_decoder": "vae"}
    components = {aliases.get(item, item) for item in components}
    supported = {"text_encoder", "transformer", "vae"}
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
    if not model_path.exists():
        raise SystemExit(f"模型目录不存在: {model_path}")
    if not meta_path.exists():
        raise SystemExit(f"meta.json 不存在: {meta_path}")
    if root_meta_path is None and ("transformer" in components or "vae" in components):
        root_meta_path = meta_path
    if root_meta_path is not None and not root_meta_path.exists():
        raise SystemExit(f"root meta.json 不存在: {root_meta_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = Flux2KleinHMONNXPipeline.from_pretrained(str(model_path), torch_dtype=dtype)
    pipe = pipe.to(device)

    text_meta_path = resolve_text_meta(meta_path, root_meta_path)
    text_meta = json_load(text_meta_path)

    if not args.skip_compare or "text_encoder" in components:
        prompt_embeds_ref, _ = pipe._get_qwen3_prompt_embeds(
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            prompt=args.prompt,
            device=device,
            max_sequence_length=pipe.tokenizer_max_length,
            hidden_states_layers=tuple(text_meta["text_encoder_out_layers"]),
        ), None
    else:
        prompt_embeds_ref = None

    hm_text_encoder = Flux2KleinTextEncoderInference.from_meta(
        text_meta_path,
        tokenizer=pipe.tokenizer,
        device=device,
        dtype=dtype,
    )
    hm_prompt_embeds, hm_model_inputs = hm_text_encoder.get_prompt_embeds(args.prompt)

    if not args.skip_compare and prompt_embeds_ref is not None:
        valid_length = int((hm_model_inputs["attention_mask"][0] != 0).sum().item())
        prompt_embeds_ref_valid = prompt_embeds_ref[:, :valid_length, :]
        hm_prompt_embeds_valid = hm_prompt_embeds[:, :valid_length, :]

        prompt_embeds_ref_np = prompt_embeds_ref_valid.detach().to(dtype=torch.float32).cpu().numpy()
        hm_prompt_embeds_np = hm_prompt_embeds_valid.detach().to(dtype=torch.float32).cpu().numpy()
        max_abs_diff = float(np.max(np.abs(prompt_embeds_ref_np - hm_prompt_embeds_np)))
        mean_abs_diff = float(np.mean(np.abs(prompt_embeds_ref_np - hm_prompt_embeds_np)))
        allclose = bool(np.allclose(prompt_embeds_ref_np, hm_prompt_embeds_np, rtol=args.rtol, atol=args.atol))
        ref_tensor = torch.from_numpy(prompt_embeds_ref_np).flatten()
        hm_tensor = torch.from_numpy(hm_prompt_embeds_np).flatten()
        cosine = float(torch.nn.functional.cosine_similarity(ref_tensor.unsqueeze(0), hm_tensor.unsqueeze(0)).item())
        rmse = float(torch.sqrt(torch.mean((ref_tensor - hm_tensor) ** 2)).item())

        print(f"prompt_embeds_shape={list(hm_prompt_embeds.shape)}")
        print(f"valid_length={valid_length}")
        print(f"max_abs_diff={max_abs_diff:.6f}")
        print(f"mean_abs_diff={mean_abs_diff:.6f}")
        print(f"rmse={rmse:.6f}")
        print(f"cosine={cosine:.8f}")
        print(f"allclose={allclose}")

        # if cosine < args.min_cosine:
        #     raise SystemExit("HMONNX prompt_embeds 与 HF reference cosine similarity 不达标")

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

    # if "vae" in components:
    #     generator_device = "cuda" if device.type == "cuda" else "cpu"
    #     generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    #     vae_image = pipe(
    #         prompt=None,
    #         prompt_embeds=hm_prompt_embeds,
    #         height=args.height,
    #         width=args.width,
    #         guidance_scale=args.guidance_scale,
    #         num_inference_steps=args.steps,
    #         generator=generator,
    #         output_type="pil",
    #         use_hmonnx_text_encoder="text_encoder" in components,
    #         use_hmonnx_transformer="transformer" in components,
    #         use_hmonnx_vae="vae" in components,
    #     ).images[0]
    #     output_path = Path(args.output)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     # vae_image.save(output_path)
    #     print(f"saved_vae_hmonnx={output_path}")
    #     print("loaded_vae_hmonnx=True")

    if args.skip_image:
        return

    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    image = pipe(
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
    print(f"input_ids_shape={list(hm_model_inputs['input_ids'].shape)}")
    print(f"attention_mask_shape={list(hm_model_inputs['attention_mask'].shape)}")


def json_load(path: Path):
    import json

    return json.load(open(path, "r"))


if __name__ == "__main__":
    main()