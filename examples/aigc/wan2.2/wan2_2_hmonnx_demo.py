# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402, I001

# pyright: reportMissingImports=false

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.xh_aigc.models.wan2_2 import (
    Wan2_2DiTExportWrapper,
    Wan2_2DiTInference,
    Wan2_2T5EncoderExportWrapper,
    Wan2_2T5EncoderInference,
    Wan2_2VAEDecoderInference,
    Wan2_2VAEEncoderInference,
)
from xh_model_zoo.xh_aigc.models.wan2_2.common import ensure_wan2_2_repo
from xh_model_zoo.xh_aigc.models.wan2_2.pipeline_hmonnx import (
    WanI2VHMONNXPipeline,
    WanT2VHMONNXPipeline,
)
from xh_model_zoo.xh_aigc.models.wan2_2.vae_local import LocalWan2_1_VAE
from xh_model_zoo.xh_aigc.models.wan2_2.vae_wrapper import Wan2_2VAEDecoderExportWrapper
from xh_model_zoo.xh_aigc.models.wan2_2.wan2_2_converter import Wan2_2ConvertConfig, Wan2_2Converter

ensure_wan2_2_repo()
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.utils import save_video as wan_save_video  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2")
    parser.add_argument("--task", type=str, default="i2v-A14B")
    parser.add_argument("--meta-dir", type=str, required=True)
    parser.add_argument("--components", nargs="+", default=["high_noise_model","t5","vae_encode", "low_noise_model", "vae_decode"]) #  ，"t5","vae_encode", "low_noise_model", "vae_decode"
    parser.add_argument("--prompt", type=str, default="A calm seaside scene with gentle waves.")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--output", type=str, default="outputs/wan2_2_hmonnx_demo.mp4")
    parser.add_argument(
        "--latent",
        type=str,
        default="",
        help="Decode this latent tensor with only the VAE HMONNX decoder and save it as video.",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default="",
        help="Tensor key to read when --latent points to a dict checkpoint; defaults to the first tensor value.",
    )
    parser.add_argument("--size", nargs=2, type=int, default=[480,832], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--guide-scale", nargs="+", type=float, default=None)
    parser.add_argument("--sample-solver", type=str, default="euler")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-model", action="store_true", default=True)
    parser.add_argument("--no-offload-model", action="store_false", dest="offload_model")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-t5-check", action="store_true")
    parser.add_argument(
        "--use-vae-decode-wrapper",
        action="store_true",
        help="Use Wan2_2VAEDecoderExportWrapper.forward instead of the HMONNX VAE decoder in generation.",
    )
    parser.add_argument(
        "--use-low-noise-wrapper",
        action="store_true",
        help="Use Wan2_2DiTExportWrapper.forward instead of the HMONNX low_noise_model in generation.",
    )
    parser.add_argument("--t5-atol", type=float, default=5e-2)
    parser.add_argument("--t5-rtol", type=float, default=5e-2)
    return parser.parse_args()


def _resolve_sampling_args(args, cfg):
    sample_steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    sample_shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift

    if args.guide_scale is None:
        guide_scale = cfg.sample_guide_scale
    elif len(args.guide_scale) == 1:
        guide_scale = args.guide_scale[0]
    elif len(args.guide_scale) == 2:
        guide_scale = tuple(args.guide_scale)
    else:
        raise ValueError("--guide-scale expects one float or two floats")

    return sample_steps, sample_shift, guide_scale


def _save_video(video: torch.Tensor, output_path: Path, fps: int):
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims [C, T, H, W], got shape {tuple(video.shape)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wan_save_video(video.unsqueeze(0), save_file=str(output_path), fps=fps, nrow=1)
    return (video.shape[1], video.shape[2], video.shape[3], video.shape[0])


def _load_latent(latent_path: Path, latent_key: str) -> torch.Tensor:
    obj = torch.load(latent_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"Expected tensor or dict in latent file, got {type(obj).__name__}")
    if latent_key:
        latent = obj[latent_key]
        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"Expected tensor at latent key {latent_key!r}, got {type(latent).__name__}")
        return latent
    for value in obj.values():
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError(f"No tensor value found in latent dict: {latent_path}")


def _decode_latent_with_hmonnx_vae(args, meta_dir: Path):
    latent = _load_latent(Path(args.latent), args.latent_key)
    if args.use_vae_decode_wrapper:
        vae_decoder = _build_vae_decode_wrapper_from_model_path(args.model, args.task, torch.device("cuda"))
    else:
        vae_decoder = Wan2_2VAEDecoderInference.from_meta(meta_dir / "vae_decode_meta.json")
    with torch.no_grad():
        video = vae_decoder(latent)
    output_path = Path(args.output)
    saved_shape = _save_video(video, output_path, fps=args.fps)
    print(f"Loaded latent from: {args.latent}")
    print(f"Latent shape: {tuple(latent.shape)} dtype={latent.dtype}")
    decoder_name = "VAE wrapper.forward" if args.use_vae_decode_wrapper else "VAE HMONNX"
    print(f"Saved {decoder_name} decoded video to: {output_path}")
    print(f"Saved frames shape [T, H, W, C]: {saved_shape}")


def _build_vae_decode_wrapper(pipe, device: torch.device):
    return _build_vae_decode_wrapper_from_vae(pipe.vae, device)


class _LazyVaeDecodeWrapper:
    def __init__(self, pipe, device: torch.device):
        self.pipe = pipe
        self.device = device
        self.wrapper = None

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        if self.wrapper is None:
            self.wrapper = _build_vae_decode_wrapper(self.pipe, self.device)
        latent = latent.to(self.device, dtype=torch.float16)
        return self.wrapper(latent).float().clamp_(-1, 1)


class _LazyDiTExportWrapper:
    def __init__(self, pipe, noise_model_name: str, device: torch.device):
        self.pipe = pipe
        self.noise_model_name = noise_model_name
        self.device = device
        self.wrapper = None

    def __getattr__(self, name):
        if self.wrapper is None:
            model = getattr(self.pipe, self.noise_model_name)
            return getattr(model, name)
        return getattr(self.wrapper, name)

    def _build(self):
        if self.wrapper is None:
            source_model = getattr(self.pipe, self.noise_model_name)
            model = deepcopy(source_model).to(device=self.device, dtype=torch.float16).eval()
            self.wrapper = Wan2_2DiTExportWrapper(model).to(self.device).eval()
        return self.wrapper

    def __call__(self, latent, *, context, e, e0):
        wrapper = self._build()
        with torch.no_grad():
            return wrapper.model(
                [latent.to(self.device, dtype=torch.float16)],
                context=context.to(self.device, dtype=torch.float16),
                e=e.to(self.device, dtype=torch.float16),
                e0=e0.to(self.device, dtype=torch.float16),
            )[0]


def _build_vae_decode_wrapper_from_vae(vae, device: torch.device):
    validate_dtype = torch.float16
    vae_model = vae.model.to(device=device, dtype=validate_dtype).eval()
    scale = [
        item.to(device=device, dtype=validate_dtype) if isinstance(item, torch.Tensor) else item
        for item in vae.scale
    ]
    return Wan2_2VAEDecoderExportWrapper(vae_model, scale).to(device).eval()


def _build_vae_decode_wrapper_from_model_path(model_path: str, task: str, device: torch.device):
    cfg = WAN_CONFIGS[task]
    cfg.param_dtype = torch.float16
    cfg.t5_dtype = torch.float16
    converter = Wan2_2Converter(model_path, Wan2_2ConvertConfig(task=task, use_resolved_float_loader=True))
    vae = LocalWan2_1_VAE(
        vae_pth=str(converter._resolve_vae_path()),
        device=device,
        dtype=cfg.param_dtype,
    )
    vae.model = vae.model.to(cfg.param_dtype)
    return _build_vae_decode_wrapper_from_vae(vae, device)


def _validate_t5(pipe, hmonnx_t5, prompt: str, atol: float, rtol: float):
    device = torch.device("cuda")

    pipe.text_encoder.model.to(device)
    with torch.no_grad():
        float_context = pipe.text_encoder([prompt], device)
        hmonnx_context = hmonnx_t5([prompt], device)
    pipe.text_encoder.model.cpu()

    warp = Wan2_2T5EncoderExportWrapper(
        deepcopy(pipe.text_encoder.model),
        max_sequence_length=int(pipe.text_encoder.text_len),
    ).to(device).eval()
    with torch.no_grad():
        warp_context = warp.encode_texts(pipe.text_encoder.tokenizer, [prompt], device)
    del warp
    torch.cuda.empty_cache()

    if len(float_context) != len(hmonnx_context) or len(float_context) != len(warp_context):
        raise AssertionError(
            "T5 output batch size mismatch: "
            f"float={len(float_context)} vs warp={len(warp_context)} vs hmonnx={len(hmonnx_context)}"
        )

    metrics = []
    for idx, (float_item, warp_item, hmonnx_item) in enumerate(
        zip(float_context, warp_context, hmonnx_context, strict=True)
    ):
        float_item = float_item.detach().float().cpu()
        warp_item = warp_item.detach().float().cpu()
        hmonnx_item = hmonnx_item.detach().float().cpu()
        warp_item = warp_item[:123]
        if float_item.shape != warp_item.shape or float_item.shape != hmonnx_item.shape:
            raise AssertionError(
                "T5 output shape mismatch at batch "
                f"{idx}: float={tuple(float_item.shape)} vs warp={tuple(warp_item.shape)} "
                f"vs hmonnx={tuple(hmonnx_item.shape)}"
            )
        float_warp_diff = (float_item - warp_item).abs()
        float_hmonnx_diff = (float_item - hmonnx_item).abs()
        warp_hmonnx_diff = (warp_item - hmonnx_item).abs()
        float_warp_max_abs = float_warp_diff.max().item()
        float_warp_mean_abs = float_warp_diff.mean().item()
        float_hmonnx_max_abs = float_hmonnx_diff.max().item()
        float_hmonnx_mean_abs = float_hmonnx_diff.mean().item()
        warp_hmonnx_max_abs = warp_hmonnx_diff.max().item()
        warp_hmonnx_mean_abs = warp_hmonnx_diff.mean().item()
        float_warp_close = torch.allclose(float_item, warp_item, atol=atol, rtol=rtol)
        float_hmonnx_close = torch.allclose(float_item, hmonnx_item, atol=atol, rtol=rtol)
        metrics.append(
            {
                "batch": idx,
                "shape": tuple(float_item.shape),
                "float_warp_max_abs": float_warp_max_abs,
                "float_warp_mean_abs": float_warp_mean_abs,
                "float_warp_allclose": float_warp_close,
                "float_hmonnx_max_abs": float_hmonnx_max_abs,
                "float_hmonnx_mean_abs": float_hmonnx_mean_abs,
                "float_hmonnx_allclose": float_hmonnx_close,
                "warp_hmonnx_max_abs": warp_hmonnx_max_abs,
                "warp_hmonnx_mean_abs": warp_hmonnx_mean_abs,
                "warp_hmonnx_allclose": torch.allclose(warp_item, hmonnx_item, atol=atol, rtol=rtol),
            }
        )
        if not float_warp_close or not float_hmonnx_close:
            raise AssertionError(
                "T5 correctness check failed at batch "
                f"{idx}: float-vs-warp max_abs={float_warp_max_abs:.6f}, "
                f"mean_abs={float_warp_mean_abs:.6f}; "
                f"float-vs-hmonnx max_abs={float_hmonnx_max_abs:.6f}, "
                f"mean_abs={float_hmonnx_mean_abs:.6f}, "
                f"atol={atol}, rtol={rtol}"
            )
    return metrics


def main(args):
    meta_dir = Path(args.meta_dir)
    if args.latent:
        _decode_latent_with_hmonnx_vae(args, meta_dir)
        return

    cfg = WAN_CONFIGS[args.task]
    cfg.param_dtype = torch.float16
    cfg.t5_dtype = torch.float16
    sample_steps, sample_shift, guide_scale = _resolve_sampling_args(args, cfg)
    pipeline_cls = WanI2VHMONNXPipeline if args.task.startswith("i2v") else WanT2VHMONNXPipeline
    pipe = pipeline_cls(cfg, args.model)
    t5_hmonnx = None

    if "t5" in args.components:
        t5_hmonnx = Wan2_2T5EncoderInference.from_meta(
            meta_dir / "t5_meta.json",
            tokenizer=pipe.text_encoder.tokenizer,
            token_embedding=pipe.text_encoder.model.token_embedding,
            device=torch.device("cuda"),
        )
        pipe.set_hmonnx_components(t5=t5_hmonnx)
    if "vae_encode" in args.components:
        pipe.set_hmonnx_components(vae_encoder=Wan2_2VAEEncoderInference.from_meta(meta_dir / "vae_encode_meta.json"))
    if "vae_decode" in args.components:
        if args.use_vae_decode_wrapper:
            pipe.set_hmonnx_components(vae_decoder=_LazyVaeDecodeWrapper(pipe, torch.device("cuda")))
        else:
            pipe.set_hmonnx_components(
                vae_decoder=Wan2_2VAEDecoderInference.from_meta(meta_dir / "vae_decode_meta.json")
            )
    if "low_noise_model" in args.components:
        if args.use_low_noise_wrapper:
            pipe.set_hmonnx_components(
                low_noise_model=_LazyDiTExportWrapper(pipe, "low_noise_model", torch.device("cuda"))
            )
        else:
            pipe.set_hmonnx_components(low_noise_model=Wan2_2DiTInference.from_meta(meta_dir / "low_noise_model_meta.json"))
    if "high_noise_model" in args.components:
        pipe.set_hmonnx_components(
            high_noise_model=Wan2_2DiTInference.from_meta(meta_dir / "high_noise_model_meta.json")
        )

    if t5_hmonnx is not None and not args.skip_t5_check:
        metrics = _validate_t5(pipe, t5_hmonnx, args.prompt, atol=args.t5_atol, rtol=args.t5_rtol)
        print("T5 HMONNX correctness check passed.")
        for metric in metrics:
            print(
                "  "
                f"batch={metric['batch']} shape={metric['shape']} "
                f"float-vs-warp(max={metric['float_warp_max_abs']:.6f}, mean={metric['float_warp_mean_abs']:.6f}) "
                f"float-vs-hmonnx(max={metric['float_hmonnx_max_abs']:.6f}, "
                f"mean={metric['float_hmonnx_mean_abs']:.6f}) "
                f"warp-vs-hmonnx(max={metric['warp_hmonnx_max_abs']:.6f}, "
                f"mean={metric['warp_hmonnx_mean_abs']:.6f})"
            )

    if not args.skip_generate:
        negative_prompt = args.negative_prompt if args.negative_prompt else pipe.sample_neg_prompt
        if args.task.startswith("i2v"):
            if not args.image:
                raise ValueError("--image is required for i2v tasks")
            image = Image.open(args.image).convert("RGB")
            video = pipe.generate(
                args.prompt,
                image,
                max_area=args.size[0] * args.size[1],
                frame_num=args.frame_num,
                shift=sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=sample_steps,
                guide_scale=guide_scale,
                n_prompt=negative_prompt,
                seed=args.seed,
                offload_model=args.offload_model,
            )
        else:
            video = pipe.generate(
                args.prompt,
                size=tuple(args.size),
                frame_num=args.frame_num,
                shift=sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=sample_steps,
                guide_scale=guide_scale,
                n_prompt=negative_prompt,
                seed=args.seed,
                offload_model=args.offload_model,
            )
        output_path = Path(args.output)
        if len(video.shape) == 5:
            video = video.squeeze(0)
        saved_shape = _save_video(video, output_path, fps=args.fps)
        print(f"Saved video to: {output_path}")
        print(f"Saved frames shape [T, H, W, C]: {saved_shape}")

    print(f"Built Wan2.2 HMONNX pipeline: {pipeline_cls.__name__}")
    print(f"Enabled components: {args.components}")
    print(f"Use VAE decode wrapper forward: {args.use_vae_decode_wrapper}")
    print(f"Use low_noise wrapper forward: {args.use_low_noise_wrapper}")
    print(f"Prompt: {args.prompt}")
    print(f"Sampling steps: {sample_steps}")
    print(f"Sampling shift: {sample_shift}")
    print(f"Guide scale: {guide_scale}")
    if args.skip_generate:
        print("Skip video generation as requested.")


if __name__ == "__main__":
    main(parse_args())
