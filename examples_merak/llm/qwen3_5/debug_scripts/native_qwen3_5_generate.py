import argparse
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    TextStreamer,
)


def main(args):
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()
    image = None
    if args.image_path is not None and len(args.image_path) > 0:
        image = Image.open(args.image_path).convert("RGB")

    hf_model_dir = args.model_dir
    processor = AutoProcessor.from_pretrained(hf_model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        hf_model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    message = {
        "role": "user",
        "content": [],
    }
    if image is not None:
        message["content"].append(
            {
                "type": "image",
                "image": image,
            },
        )
    message["content"].append(
        {"type": "text", "text": prompt},
    )
    messages = [
        message,
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = inputs.to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    logger.info(f"{'-' * 20} Stream output {'-' * 20}")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            # max_new_tokens=81920,
            max_new_tokens=1024,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = tokenizer.decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    logger.info(f"{'-' * 20} Output {'-' * 20}")
    logger.info(f"{output_text[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3.5 model")
    parser.add_argument("--model-dir", type=str, default="./data/models/Qwen3.5-9B")
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--prompt", type=str, default="你是谁？")

    args = parser.parse_args()
    main(args)
