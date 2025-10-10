# from transformers import AutoProcessor
from xh_model_zoo.xh_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from xh_model_zoo.xh_llm.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor as AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os
from pathlib import Path

import torch
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def create_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
    )

    # default processor
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def create_template_messages(image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_dir", type=str, default="data/test")
    parser.add_argument("--prompt", type=str, default="详细描述下这张图片")
    parser.add_argument("--output_dir", type=str, default="data/output")
    return parser.parse_args()


def main():
    args = get_args()
    model, processor = create_model(args.model_path)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    logger.add(Path(args.output_dir) / "log.txt")
    for image_path in os.listdir(args.image_dir):
        messages = create_template_messages(os.path.join(args.image_dir, image_path), args.prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        inputs.pop("hm_pixel_values", None)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        logger.info(f"Image: {image_path}, Output: {output_text[0]}")
        with open(Path(args.output_dir) / f"{os.path.splitext(image_path)[0]}.txt", "w") as f:
            f.write(output_text[0])


if __name__ == "__main__":
    main()
