import argparse
import json
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from xhquant.common import PrecisionMode
from xhquant.xhonnxruntime import HMONNXCUDAGraphInference

HIDREAM_EXAMPLE_DIR = Path(__file__).resolve().parent / "HiDream-O1-Image"
if str(HIDREAM_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(HIDREAM_EXAMPLE_DIR))

from models.pipeline import DEFAULT_TIMESTEPS, PATCH_SIZE, T_EPS, build_scheduler  # pyright: ignore[reportMissingImports]  # noqa: E402
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # pyright: ignore[reportMissingImports]  # noqa: E402

from xh_model_zoo.xh_llm.models.hidream_o1 import (
    build_rotary_inputs,
    build_t2i_sample_inputs,
    ensure_hidream_o1_imports,
)


ensure_hidream_o1_imports()


FIXED_TIMESTEPS = torch.tensor(
    [
        0.001,
        0.007,
        0.014,
        0.021,
        0.029,
        0.036,
        0.044,
        0.052,
        0.060,
        0.069,
        0.078,
        0.087,
        0.096,
        0.105,
        0.115,
        0.126,
        0.136,
        0.147,
        0.159,
        0.170,
        0.182,
        0.195,
        0.208,
        0.222,
        0.236,
        0.251,
        0.266,
        0.282,
        0.298,
        0.316,
        0.334,
        0.353,
        0.373,
        0.393,
        0.415,
        0.438,
        0.462,
        0.487,
        0.514,
        0.542,
        0.572,
        0.604,
        0.637,
        0.672,
        0.710,
        0.751,
        0.794,
        0.840,
        0.889,
        0.943,
    ],
    dtype=torch.float32,
)


def add_special_tokens(tokenizer) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def timestep_to_index(timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
    timestep = timestep.reshape(-1).to(dtype=torch.float32)
    fixed_timesteps = FIXED_TIMESTEPS.to(device=timestep.device)
    index = torch.argmin(torch.abs(fixed_timesteps - timestep.reshape(-1, 1)), dim=1)
    return index.to(device=device, dtype=torch.int32)


@torch.no_grad()
def build_hmonnx_step_inputs(
    model: Qwen3VLForConditionalGeneration,
    sample: dict,
    z: torch.Tensor,
    t_pixeldit: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    input_names: list[str],
) -> list[torch.Tensor]:
    inputs_embeds = model.model.get_input_embeddings()(sample["input_ids"])
    t_emb = model.model.t_embedder1(t_pixeldit.reshape(-1).to(device))
    text_embeds_for_rope = torch.cat([inputs_embeds[:, :-1, :], t_emb.unsqueeze(1)], dim=1)
    vinputs_embedded = model.model.x_embedder(z.to(device)).to(inputs_embeds.dtype)
    rotary_dummy_embeds = torch.cat([text_embeds_for_rope, vinputs_embedded], dim=1)

    rotary_position_ids = sample["position_ids"]
    if rotary_position_ids.ndim == 2:
        rotary_position_ids = rotary_position_ids[None, ...].expand(3, rotary_position_ids.shape[0], -1)
    elif rotary_position_ids.ndim == 3 and rotary_position_ids.shape[0] == 4:
        rotary_position_ids = rotary_position_ids[1:]
    rotary_cos, rotary_sin = build_rotary_inputs(
        model.model.language_model.rotary_emb,
        rotary_dummy_embeds,
        rotary_position_ids,
        dtype=dtype,
    )

    tensors = {
        "inputs_embeds": inputs_embeds.to(device=device, dtype=dtype),
        "attention_mask": sample["attention_mask"].to(device=device, dtype=dtype),
        "rotary_cos": rotary_cos.to(device=device, dtype=dtype),
        "rotary_sin": rotary_sin.to(device=device, dtype=dtype),
        "vinputs": z.to(device=device, dtype=dtype),
        "timestep": timestep_to_index(t_pixeldit.to(device), device),
        "token_types": sample["token_types"].to(device=device, dtype=torch.int32),
    }
    return [tensors[name] for name in input_names]


@torch.no_grad()
def generate_image_hmonnx(
    model: Qwen3VLForConditionalGeneration,
    processor,
    hmonnx_path: str | Path,
    input_names: list[str],
    prompt: str,
    height: int,
    width: int,
    text_seq_len: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    shift: float = 3.0,
    timesteps_list=None,
    scheduler_name: str = "default",
    seed: int = 42,
    noise_scale_start: float = 8.0,
    noise_scale_end: float = 8.0,
    noise_clip_std: float = 0.0,
    enable_cuda_graph: bool = True,
    warmup_runs: int = 3,
    graph_warmup_runs: int = 6,
    precision_mode: str = "aligned",
) -> Image.Image:
    """Text-to-image generation based on `models.pipeline.generate_image`, with HMONNX denoise forward."""

    device = model.device
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    h_patches = height // PATCH_SIZE
    w_patches = width // PATCH_SIZE

    runtime = HMONNXCUDAGraphInference(
        str(hmonnx_path),
        enable_cuda_graph=enable_cuda_graph,
        warmup_runs=warmup_runs,
        graph_warmup_runs=graph_warmup_runs,
    )
    runtime.exec_device = device
    runtime.to(device)
    mode = PrecisionMode.FAST if precision_mode == "fast" else PrecisionMode.ALIGNED
    try:
        runtime.set_precision_mode(mode)
    except Exception:
        runtime._precision_mode = mode

    cond_sample, _ = build_t2i_sample_inputs(
        model=model,
        processor=processor,
        prompt=prompt,
        height=height,
        width=width,
        seed=seed,
        dtype=dtype,
        device=device,
        text_seq_len=text_seq_len,
    )
    samples = [cond_sample]
    if guidance_scale > 1.0:
        uncond_sample, _ = build_t2i_sample_inputs(
            model=model,
            processor=processor,
            prompt=" ",
            height=height,
            width=width,
            seed=seed,
            dtype=dtype,
            device=device,
            text_seq_len=text_seq_len,
        )
        samples.append(uncond_sample)

    noise = noise_scale_start * torch.randn(
        (1, 3, height, width),
        generator=torch.Generator("cpu").manual_seed(seed + 1),
    ).to(device, dtype)
    z = einops.rearrange(noise, "B C (H p1) (W p2) -> B (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE)

    sched = build_scheduler(num_inference_steps, timesteps_list, shift, device, scheduler_name)
    num_steps = len(sched.timesteps)
    if num_steps > 1:
        noise_scale_schedule = [
            noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1) for i in range(num_steps)
        ]
    else:
        noise_scale_schedule = [noise_scale_start]

    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1)

    def forward_once(sample: dict, z_in: torch.Tensor, t_pixeldit: torch.Tensor) -> torch.Tensor:
        inputs = build_hmonnx_step_inputs(
            model=model,
            sample=sample,
            z=z_in,
            t_pixeldit=t_pixeldit,
            dtype=dtype,
            device=device,
            input_names=input_names,
        )
        out = runtime(*inputs)
        x_pred = out[0] if isinstance(out, tuple) else out
        return x_pred.to(device=device, dtype=dtype)

    try:
        import tqdm

        step_iter = tqdm.tqdm(sched.timesteps, desc="Generating(HMONNX)")
    except Exception:
        step_iter = sched.timesteps

    for step_idx, step_t in enumerate(step_iter):
        t_pixeldit = 1.0 - step_t.float() / 1000.0
        sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

        x_pred_cond = forward_once(samples[0], z.clone(), t_pixeldit)
        v_cond = (x_pred_cond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma

        if len(samples) > 1:
            x_pred_uncond = forward_once(samples[1], z.clone(), t_pixeldit)
            v_uncond = (x_pred_uncond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma
            v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v_guided = v_cond

        model_output = -v_guided
        if scheduler_name == "flash":
            z = sched.step(
                model_output.float(),
                step_t.to(dtype=torch.float32),
                z.float(),
                s_noise=noise_scale_schedule[step_idx],
                noise_clip_std=noise_clip_std,
                return_dict=False,
            )[0].to(dtype)
        else:
            z = sched.step(model_output.float(), step_t.to(dtype=torch.float32), z.float(), return_dict=False)[0].to(
                dtype
            )

    img = (z + 1) / 2
    img = einops.rearrange(
        img.cpu().float(),
        "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
        H=h_patches,
        W=w_patches,
        p1=PATCH_SIZE,
        p2=PATCH_SIZE,
    )
    arr = np.round(np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    image = Image.fromarray(arr).convert("RGB")
    if enable_cuda_graph:
        print(f"[hmonnx] cuda_graph captured={runtime.has_captured_graph} reason={runtime.capture_unavailable_reason}")
    return image


def parse_args():
    parser = argparse.ArgumentParser("HiDream-O1 HMONNX text-to-image inference")
    parser.add_argument("--model_path", type=str, default="/data02/datasets/HiDream-O1-Image")
    parser.add_argument("--meta", type=str, default="work_dirs/hidream_o1_w8a16/meta.json")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output_image", type=str, default="output_hmonnx.png")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_type", choices=["full", "dev"], default="full")
    parser.add_argument("--scheduler_name", choices=["default", "flow_match", "flash"], default=None)
    parser.add_argument("--noise_scale_start", type=float, default=7.5)
    parser.add_argument("--noise_scale_end", type=float, default=7.5)
    parser.add_argument("--noise_clip_std", type=float, default=2.5)
    parser.add_argument("--precision_mode", choices=["aligned", "fast"], default="aligned")
    parser.add_argument("--warmup_runs", type=int, default=3)
    parser.add_argument("--graph_warmup_runs", type=int, default=6)
    parser.add_argument("--disable_cuda_graph", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required for HMONNX inference."

    meta_path = Path(args.meta).resolve()
    meta = json.load(open(meta_path, "r"))
    hmonnx_path = meta_path.parent / meta["hmonnx_file"]
    input_names = list(meta["input_names"])
    text_seq_len = int(meta.get("txt_seq_len", 512))
    height = int(args.height if args.height is not None else meta["height"])
    width = int(args.width if args.width is not None else meta["width"])
    seed = int(args.seed if args.seed is not None else meta.get("seed", 32))
    prompt = args.prompt if args.prompt is not None else meta.get("prompt", "A beautiful castle beside a lake")

    if height != int(meta["height"]) or width != int(meta["width"]):
        raise ValueError(
            f"HMONNX was exported for {meta['width']}x{meta['height']}, "
            f"but got {width}x{height}. Please export a matching graph."
        )

    print(
        "[hmonnx] config "
        f"prompt={prompt!r} seed={seed} size={width}x{height} "
        f"steps={args.num_inference_steps} guidance_scale={args.guidance_scale} "
        f"shift={args.shift} text_seq_len={text_seq_len}"
    )
    print(f"[hmonnx] Loading processor/model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    add_special_tokens(tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    if args.model_type == "full":
        num_inference_steps = args.num_inference_steps
        guidance_scale = args.guidance_scale
        shift = args.shift
        timesteps_list = None
        scheduler_name = args.scheduler_name or "default"
        extra_kwargs = {}
    else:
        num_inference_steps = 28
        guidance_scale = 0.0
        shift = 1.0
        timesteps_list = DEFAULT_TIMESTEPS
        scheduler_name = args.scheduler_name or "flash"
        extra_kwargs = {
            "noise_scale_start": args.noise_scale_start,
            "noise_scale_end": args.noise_scale_end,
            "noise_clip_std": args.noise_clip_std,
        }

    image = generate_image_hmonnx(
        model=model,
        processor=processor,
        hmonnx_path=hmonnx_path,
        input_names=input_names,
        prompt=prompt,
        height=height,
        width=width,
        text_seq_len=text_seq_len,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        timesteps_list=timesteps_list,
        scheduler_name=scheduler_name,
        seed=seed,
        enable_cuda_graph=not args.disable_cuda_graph,
        warmup_runs=args.warmup_runs,
        graph_warmup_runs=args.graph_warmup_runs,
        precision_mode=args.precision_mode,
        **extra_kwargs,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output_image)), exist_ok=True)
    image.save(args.output_image)
    print(f"[hmonnx] Saved -> {args.output_image}")


if __name__ == "__main__":
    main()
