import argparse
from pathlib import Path

import torch
from PIL import Image
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer


def main(args):
    model_dir = args.model_dir
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(args.image_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=args.think,
    )
    inputs = inputs.to(model.device)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.eos_token_id,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    logger.info(output_text[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./weights/gemma-4-31B-it")
    parser.add_argument("--image-path", type=str, default="data/images/qwen2_vl_demo.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()
    main(args)
