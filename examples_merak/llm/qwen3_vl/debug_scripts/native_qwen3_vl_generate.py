import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextStreamer


def _read_prompt(prompt: str) -> str:
    prompt_path = Path(prompt)
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")
    return prompt


def _load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def main(args):
    model_dir = args.model_dir
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype="auto", device_map="auto")

    processor = AutoProcessor.from_pretrained(model_dir)
    image = _load_image(args.image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # BGR to RGB
                },
                {"type": "text", "text": _read_prompt(args.prompt)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
    logger.info(f"{'-' * 20} Stream output {'-' * 20}")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            streamer=streamer,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logger.info(f"{'-' * 20} Output {'-' * 20}")
    logger.info(f"{output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3 model")
    parser.add_argument("--model-dir", type=str, default="./data/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--image-path", type=str, default="./data/images/demo_qwen3_vl.jpeg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    args = parser.parse_args()
    main(args)
