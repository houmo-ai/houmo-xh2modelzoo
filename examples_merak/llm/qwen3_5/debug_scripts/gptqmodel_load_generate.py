import argparse

import torch
from gptqmodel import GPTQModel
from transformers import AutoTokenizer, TextStreamer


DEFAULT_PROMPT = "请介绍一下 GPTQ 量化的核心思想。"
DEFAULT_VISION_PROMPT = "请简要描述这张图片中的主要内容。"


def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()
    load_kwargs = dict(trust_remote_code=True)
    load_kwargs["device_map"] = "auto"
    model_id = args.model
    model = GPTQModel.load(model_id, **load_kwargs)
    runtime_device = str(model.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    inputs = build_chat_inputs(tokenizer, DEFAULT_PROMPT, runtime_device)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            streamer=streamer,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print("=== Prompt ===")
    print(DEFAULT_PROMPT)
    print("=== Answer ===")
    print(answer)
