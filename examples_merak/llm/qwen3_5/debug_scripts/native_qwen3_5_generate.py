import argparse
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, TextStreamer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


def main(args):
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()
    if not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image file does not exist: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Failed to load image: {args.image_path}")

    hf_model_dir = args.model_dir
    processor = AutoProcessor.from_pretrained(hf_model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        hf_model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
        #                 },
        #             },
        #             {
        #                 "type": "text",
        #                 "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$",
        #             },
        #         ],
        #     },
        # {
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
        #             },
        #         },
        #         {"type": "text", "text": "Where is this?"},
        #     ],
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
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
    output_text = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    logger.info(f"{'-' * 20} Output {'-' * 20}")
    logger.info(f"{output_text[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3.5 model")
    parser.add_argument("--model-dir", type=str, default="./data/models/Qwen3.5-9B")
    parser.add_argument("--image-path", type=str, default="./data/images/RealWorld-04.png")
    parser.add_argument("--prompt", type=str, default="Describe this image.")

    args = parser.parse_args()
    main(args)
