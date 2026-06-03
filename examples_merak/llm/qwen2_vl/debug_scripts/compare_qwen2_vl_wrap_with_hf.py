import argparse
import sys
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMModelState
from xhquant.api import Config, xhquant_init
from xhquant.utils import ContextManagers


def _build_messages(prompt: str, image_path: str, image_size: int) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": image_size,
                    "resized_width": image_size,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _move_inputs(inputs, device: str):
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _summarize(name: str, hf_tensor: torch.Tensor, wrap_tensor: torch.Tensor) -> bool:
    hf = hf_tensor.detach().float().cpu()
    wrap = wrap_tensor.detach().float().cpu()
    diff = (hf - wrap).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = hf.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    cosine = torch.nn.functional.cosine_similarity(hf.flatten(), wrap.flatten(), dim=0).item()
    same_shape = tuple(hf.shape) == tuple(wrap.shape)
    exact_equal = bool(torch.equal(hf, wrap))
    close_fp16 = bool(torch.allclose(hf, wrap, rtol=1e-2, atol=1e-2))
    print(f"[{name}]")
    print(f"  hf_shape   = {tuple(hf.shape)}")
    print(f"  wrap_shape = {tuple(wrap.shape)}")
    print(f"  exact_equal= {exact_equal}")
    print(f"  allclose   = {close_fp16}  # rtol=1e-2, atol=1e-2")
    print(f"  max_abs    = {max_abs:.8f}")
    print(f"  mean_abs   = {mean_abs:.8f}")
    print(f"  max_rel    = {max_rel:.8f}")
    print(f"  cosine     = {cosine:.8f}")
    return same_shape


def _run_wrap_prefill(xh_model, input_ids, image_embeds, image_grid_thw):
    xh_data_input = xh_model.get_data_preprocessor()(
        {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "past_seq_length": 0,
            "image_grid_thw": image_grid_thw,
        }
    )
    return xh_model.forward(*xh_data_input)


def main(args):
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "compare_qwen2_vl_wrap_with_hf.log"), args.debug)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float16

    cfg = Config.fromfile(args.config)
    cfg.model.hf_model = args.model
    cfg.model.visual_config.hf_model = args.model
    cfg.model.visual_config.max_size_h = args.image_size
    cfg.model.visual_config.max_size_w = args.image_size
    cfg.model.prefill_chunk_length = max(cfg.model.prefill_chunk_length, args.min_prefill_chunk)

    processor = AutoProcessor.from_pretrained(args.model)
    messages = _build_messages(args.prompt, args.image_path, args.image_size)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = _move_inputs(model_inputs, device)
    pixel_values = model_inputs["pixel_values"].to(dtype)
    input_ids = model_inputs["input_ids"]
    image_grid_thw = model_inputs["image_grid_thw"]

    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    hf_model.eval()

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir / "wrap")
    model_cfg.visual_config.work_dir = str(work_dir / "wrap" / "visual")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.WRAP)
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    seq_len = input_ids.shape[1]
    xh_model.set_input_sequence_length(seq_len)
    xh_model.prepare_for_inference()

    with ContextManagers([torch.no_grad(), xh_model.get_kvcache_mixin().kv_cache_scope(device=device)]):
        hf_vit = hf_model.visual(pixel_values, grid_thw=image_grid_thw)
        wrap_vit = xh_model.visual.forward(pixel_values)
        wrap_logits_with_hf_vit = _run_wrap_prefill(xh_model, input_ids, hf_vit, image_grid_thw)
        wrap_logits_end_to_end = _run_wrap_prefill(xh_model, input_ids, wrap_vit, image_grid_thw)

        hf_outputs = hf_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            return_dict=True,
        )
        hf_logits = hf_outputs.logits[:, -1:, :]

    print(f"device      = {device}")
    print(f"dtype       = {dtype}")
    print(f"model       = {args.model}")
    print(f"image       = {args.image_path}")
    print(f"prompt      = {args.prompt}")
    print(f"image_size  = {args.image_size}")
    print(f"input_shape = {tuple(input_ids.shape)}")
    print(f"grid_thw    = {image_grid_thw.detach().cpu().tolist()}")

    ok = True
    ok = _summarize("vit", hf_vit, wrap_vit) and ok
    ok = _summarize("llm_prefill_last_logits_with_hf_vit", hf_logits, wrap_logits_with_hf_vit) and ok
    ok = _summarize("llm_prefill_last_logits_end_to_end", hf_logits, wrap_logits_end_to_end) and ok
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs_merak/xh2a/llm_models/qwen2_vl/2b/qwen2_vl_llm_2b_xh2a_4k.py")
    parser.add_argument("--model", default="/data02/datasets/Qwen2-VL-2B-Instruct")
    parser.add_argument("--image-path", default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", default="简单描述这张图片")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--min-prefill-chunk", type=int, default=512)
    parser.add_argument("--work-dir", default="work_dirs/qwen2_vl_wrap_hf_compare")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
