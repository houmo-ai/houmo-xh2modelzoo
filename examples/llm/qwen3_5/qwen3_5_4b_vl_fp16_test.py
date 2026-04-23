"""
Qwen3.5-4B VL fp16 浮点基线测试。

使用 transformers 原始模型 (fp16) 推理，验证 VL 基线结果是否正确。
用于 CMS-477 排查：先确认浮点模型结果正确，再对比量化模型。

CMS-477 复现场景：
  - 图片: data/images/cms477_0.jpg (3840x2160 监控/航拍图)
  - Prompt: "图中有什么？"
  - repetition_penalty: 1.1 / 1.2

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python examples/llm/qwen3_5/qwen3_5_4b_vl_fp16_test.py

    # CMS-477 复现
    CUDA_VISIBLE_DEVICES=0,1 python examples/llm/qwen3_5/qwen3_5_4b_vl_fp16_test.py \
        --image-path data/images/cms477_0.jpg --prompt "图中有什么？" \
        --repetition-penalty 1.2
"""

import argparse
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B VL fp16 baseline test (CMS-477)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/Qwen3.5-4B",
        help="Path to HF Qwen3.5-4B model",
    )
    parser.add_argument("--image-path", type=str, default="data/images/cms477_0.jpg")
    parser.add_argument("--prompt", type=str, default="图中有什么？")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-size-w", type=int, default=448)
    parser.add_argument("--max-size-h", type=int, default=448)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true", default=False)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (CMS-477 uses 1.1 or 1.2)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return parser


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def build_messages(prompt, image_path, system_prompt, max_size_h, max_size_w):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": max_size_h,
                    "resized_width": max_size_w,
                },
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def main():
    args = parse_arguments().parse_args()
    dtype = DTYPE_MAP[args.dtype]
    print(f"[Config] model={args.model_path}, dtype={args.dtype}, image={args.image_path}")
    print(f"[Config] prompt={args.prompt!r}, max_new_tokens={args.max_new_tokens}")
    print(f"[Config] repetition_penalty={args.repetition_penalty}, do_sample={args.do_sample}")

    # Load model
    print("[Loading] model + processor ...", flush=True)
    t0 = time.perf_counter()
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"[Loading] done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[Model] device_map: {getattr(model, 'hf_device_map', 'N/A')}", flush=True)

    # Build messages
    messages = build_messages(
        args.prompt, args.image_path, args.system_prompt,
        args.max_size_h, args.max_size_w,
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        min_pixels=args.max_size_h * args.max_size_w,
        max_pixels=args.max_size_h * args.max_size_w,
        patch_size=16,
        merge_size=2,
        padding=True,
        return_tensors="pt",
    )

    input_ids = model_inputs["input_ids"]
    print(f"[Input] prompt tokens: {input_ids.shape[1]}", flush=True)

    # Move inputs to model device
    device = next(model.parameters()).device
    model_inputs_device = {}
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            model_inputs_device[k] = v.to(device)
        else:
            model_inputs_device[k] = v

    # Generate
    print("[Generating] ...", flush=True)
    t1 = time.perf_counter()
    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )
        if args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty
        if args.do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
            gen_kwargs["top_k"] = args.top_k

        output_ids = model.generate(**model_inputs_device, **gen_kwargs)

    gen_time = time.perf_counter() - t1
    generated_ids = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    num_tokens = len(generated_ids)
    tps = num_tokens / gen_time if gen_time > 0 else 0

    print("=" * 60)
    print(f"[Result] fp16 VL output ({num_tokens} tokens, {gen_time:.2f}s, {tps:.1f} tok/s):")
    print(output_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
