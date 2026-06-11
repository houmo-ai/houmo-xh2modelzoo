import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from xhquant.api import HMONNXGoldenInference

from xh_model_zoo.xh_llm.models.hidream_o1 import (
    HiDreamO1DenoiseExportWrapper,
    build_rotary_inputs,
    build_t2i_sample_inputs,
    ensure_hidream_o1_imports,
    register_hidream_o1_wrap_modules,
)
from xh_model_zoo.xh_llm.models.hidream_o1.models.pipeline import PATCH_SIZE, T_EPS, build_scheduler


ensure_hidream_o1_imports()
from xh_model_zoo.xh_llm.models.hidream_o1.models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # pyright: ignore[reportMissingImports]  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--meta", type=str, default="work_dirs/hidream_o1_w8a16/meta.json")
    parser.add_argument("--model", type=str, default=None, help="HF model path; default uses meta source_model_dir")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt; default uses meta prompt")
    parser.add_argument("--seed", type=int, default=None, help="Seed; default uses meta seed")
    parser.add_argument("--height", type=int, default=None, help="Height; default uses meta height")
    parser.add_argument("--width", type=int, default=None, help="Width; default uses meta width")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--scheduler-name", choices=["default", "flow_match", "flash"], default="default")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--noise-scale-start", type=float, default=7.5)
    parser.add_argument("--noise-scale-end", type=float, default=7.5)
    parser.add_argument("--noise-clip-std", type=float, default=2.5)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--hmonnx-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--compare-uncond", action="store_true", help="Also compare unconditional branch when CFG is enabled"
    )
    return parser.parse_args()


def tensor_stats(name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().float()
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"mean={float(value.mean()):.8f} std={float(value.std()):.8f} "
        f"min={float(value.min()):.8f} max={float(value.max()):.8f}"
    )


def summarize_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float) -> None:
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    diff = (lhs_f - rhs_f).abs()
    cosine = torch.nn.functional.cosine_similarity(lhs_f.flatten()[None], rhs_f.flatten()[None]).item()
    allclose = torch.allclose(lhs_f, rhs_f, atol=atol, rtol=rtol)
    print(
        f"{name}: shape={tuple(lhs.shape)} max_abs={float(diff.max()):.8f} "
        f"mean_abs={float(diff.mean()):.8f} cosine={cosine:.8f} "
        f"allclose(atol={atol}, rtol={rtol})={bool(allclose)}"
    )


def build_step_inputs(
    model: Qwen3VLForConditionalGeneration,
    sample: dict[str, torch.Tensor],
    z: torch.Tensor,
    dtype: torch.dtype,
):
    inputs_embeds = model.model.get_input_embeddings()(sample["input_ids"])
    t_emb = model.model.t_embedder1(sample["timestep"].to(inputs_embeds.device))
    text_embeds_for_rope = torch.cat([inputs_embeds[:, :-1, :], t_emb.unsqueeze(1)], dim=1)
    vinputs_embedded = model.model.x_embedder(z.to(inputs_embeds.device)).to(inputs_embeds.dtype)
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
    return inputs_embeds, sample["attention_mask"], rotary_cos, rotary_sin, z, sample["timestep_index"]


@torch.no_grad()
def compare_branch(
    branch_name: str,
    float_wrapper: HiDreamO1DenoiseExportWrapper,
    hmonnx_runtime: HMONNXGoldenInference,
    input_names: list[str],
    model: Qwen3VLForConditionalGeneration,
    sample: dict[str, torch.Tensor],
    z: torch.Tensor,
    sigma: torch.Tensor,
    dtype: torch.dtype,
    hmonnx_dtype: torch.dtype,
    atol: float,
    rtol: float,
):
    inputs_embeds, attention_mask, rotary_cos, rotary_sin, vinputs, timestep_index = build_step_inputs(
        model,
        sample,
        z,
        dtype,
    )
    tensor_stats(f"{branch_name}.inputs_embeds", inputs_embeds)
    tensor_stats(f"{branch_name}.attention_mask", attention_mask)
    tensor_stats(f"{branch_name}.rotary_cos", rotary_cos)
    tensor_stats(f"{branch_name}.rotary_sin", rotary_sin)
    tensor_stats(f"{branch_name}.vinputs", vinputs)
    print(
        f"{branch_name}.timestep={sample['timestep'].detach().cpu().tolist()} "
        f"index={timestep_index.detach().cpu().tolist()}"
    )

    float_out = float_wrapper(inputs_embeds, attention_mask, rotary_cos, rotary_sin, vinputs, timestep_index)
    feed = {
        "inputs_embeds": inputs_embeds.to(dtype=hmonnx_dtype),
        "attention_mask": attention_mask.to(dtype=hmonnx_dtype),
        "rotary_cos": rotary_cos.to(dtype=hmonnx_dtype),
        "rotary_sin": rotary_sin.to(dtype=hmonnx_dtype),
        "vinputs": vinputs.to(dtype=hmonnx_dtype),
        "timestep": timestep_index.to(dtype=torch.int32),
        "token_types": sample["token_types"].to(dtype=torch.int32),
    }
    hm_inputs = [feed[name] for name in input_names]
    hmonnx_out = hmonnx_runtime(*hm_inputs)
    hmonnx_out = hmonnx_out[0] if isinstance(hmonnx_out, tuple) else hmonnx_out

    tensor_stats(f"{branch_name}.float_x_pred", float_out)
    tensor_stats(f"{branch_name}.hmonnx_x_pred", hmonnx_out)
    summarize_diff(f"{branch_name}.x_pred_diff", float_out, hmonnx_out, atol, rtol)

    float_v = (float_out.float() - z.float()) / sigma
    hmonnx_v = (hmonnx_out.float() - z.float()) / sigma
    summarize_diff(f"{branch_name}.v_pred_diff", float_v, hmonnx_v, atol, rtol)
    return float_out, hmonnx_out, float_v, hmonnx_v


@torch.no_grad()
def main():
    args = parse_args()
    meta_path = Path(args.meta).resolve()
    meta = json.load(open(meta_path, "r"))
    model_dir = Path(args.model or meta["source_model_dir"]).resolve()
    hmonnx_path = meta_path.parent / meta["hmonnx_file"]
    input_names = list(meta["input_names"])
    height = int(args.height if args.height is not None else meta["height"])
    width = int(args.width if args.width is not None else meta["width"])
    prompt = args.prompt if args.prompt is not None else meta["prompt"]
    seed = int(args.seed if args.seed is not None else meta["seed"])
    text_seq_len = int(meta.get("txt_seq_len", 512))

    if height != int(meta["height"]) or width != int(meta["width"]):
        raise ValueError(f"HMONNX was exported for {meta['width']}x{meta['height']}, got {width}x{height}")
    if height % PATCH_SIZE != 0 or width % PATCH_SIZE != 0:
        raise ValueError(f"height/width must be divisible by PATCH_SIZE={PATCH_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        hmonnx_dtype = torch.bfloat16 if args.hmonnx_dtype == "bfloat16" else torch.float16
    else:
        dtype = torch.float32
        hmonnx_dtype = torch.float32
    print(
        f"meta={meta_path} hmonnx={hmonnx_path} model={model_dir} size={width}x{height} "
        f"seed={seed} prompt={prompt!r} text_seq_len={text_seq_len} "
        f"float_dtype={dtype} hmonnx_dtype={hmonnx_dtype} input_names={input_names}"
    )

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        device_map="cuda" if device.type == "cuda" else None,
    ).eval()

    sched = build_scheduler(args.num_inference_steps, None, args.shift, device, args.scheduler_name)
    step_t = sched.timesteps[0]
    t_pixeldit = 1.0 - step_t.float() / 1000.0
    sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)
    print(f"first_step_t={float(step_t):.8f} t_pixeldit={float(t_pixeldit):.8f} sigma={float(sigma):.8f}")

    cond_sample, _ = build_t2i_sample_inputs(
        model=model,
        processor=processor,
        prompt=prompt,
        height=height,
        width=width,
        seed=seed,
        dtype=dtype,
        device=device,
        timestep=float(t_pixeldit),
        text_seq_len=text_seq_len,
    )
    uncond_sample = None
    if args.compare_uncond or args.guidance_scale > 1.0:
        uncond_sample, _ = build_t2i_sample_inputs(
            model=model,
            processor=processor,
            prompt=" ",
            height=height,
            width=width,
            seed=seed,
            dtype=dtype,
            device=device,
            timestep=float(t_pixeldit),
            text_seq_len=text_seq_len,
        )

    noise = args.noise_scale_start * torch.randn(
        (1, 3, height, width),
        generator=torch.Generator("cpu").manual_seed(seed + 1),
    ).to(device, dtype)
    z = noise.reshape(1, 3, height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE)
    z = z.permute(0, 2, 4, 1, 3, 5).reshape(1, -1, 3 * PATCH_SIZE * PATCH_SIZE)
    tensor_stats("step0.z", z)

    register_hidream_o1_wrap_modules(model)
    float_wrapper = HiDreamO1DenoiseExportWrapper(model, txt_seq_len=text_seq_len).to(device).eval()
    hmonnx_runtime = HMONNXGoldenInference(str(hmonnx_path))
    hmonnx_runtime.exec_device = device

    cond_float_x, cond_hm_x, cond_float_v, cond_hm_v = compare_branch(
        "cond",
        float_wrapper,
        hmonnx_runtime,
        input_names,
        model,
        cond_sample,
        z,
        sigma,
        dtype,
        hmonnx_dtype,
        args.atol,
        args.rtol,
    )

    if uncond_sample is not None:
        uncond_float_x, uncond_hm_x, uncond_float_v, uncond_hm_v = compare_branch(
            "uncond",
            float_wrapper,
            hmonnx_runtime,
            input_names,
            model,
            uncond_sample,
            z,
            sigma,
            dtype,
            hmonnx_dtype,
            args.atol,
            args.rtol,
        )
        float_guided_v = uncond_float_v + args.guidance_scale * (cond_float_v - uncond_float_v)
        hmonnx_guided_v = uncond_hm_v + args.guidance_scale * (cond_hm_v - uncond_hm_v)

    summarize_diff("guided_v_diff", float_guided_v, hmonnx_guided_v, args.atol, args.rtol)
    float_model_output = -float_guided_v
    hmonnx_model_output = -hmonnx_guided_v
    float_sched = copy.deepcopy(sched)
    hmonnx_sched = copy.deepcopy(sched)
    if args.scheduler_name == "flash":
        num_steps = len(sched.timesteps)
        if num_steps > 1:
            noise_scale = args.noise_scale_start + (args.noise_scale_end - args.noise_scale_start) * 0 / (num_steps - 1)
        else:
            noise_scale = args.noise_scale_start
        float_z_next = float_sched.step(
            float_model_output.float(),
            step_t.to(dtype=torch.float32),
            z.float(),
            s_noise=noise_scale,
            noise_clip_std=args.noise_clip_std,
            return_dict=False,
        )[0]
        hmonnx_z_next = hmonnx_sched.step(
            hmonnx_model_output.float(),
            step_t.to(dtype=torch.float32),
            z.float(),
            s_noise=noise_scale,
            noise_clip_std=args.noise_clip_std,
            return_dict=False,
        )[0]
    else:
        float_z_next = float_sched.step(
            float_model_output.float(), step_t.to(dtype=torch.float32), z.float(), return_dict=False
        )[0]
        hmonnx_z_next = hmonnx_sched.step(
            hmonnx_model_output.float(), step_t.to(dtype=torch.float32), z.float(), return_dict=False
        )[0]
    summarize_diff("z_next_diff", float_z_next, hmonnx_z_next, args.atol, args.rtol)


if __name__ == "__main__":
    main()
