from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os
from pathlib import Path
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
import math
import re
from ui_tars_utils import parse_output, COMPUTER_USE_DOUBAO

# Constants for Smart Resize
IMAGE_FACTOR = 28
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def create_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def create_template_messages(image_path, instruction):
    prompt = COMPUTER_USE_DOUBAO.format(instruction=instruction, language="English")
    
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
    parser.add_argument("--model_path", type=str, default="weights/UI-TARS-1.5-7B") # Default to UI-TARS model
    parser.add_argument("--image_path", type=str, default="data/test/example.png")
    parser.add_argument("--prompt", type=str, default="Could you help me set the Number Threads to Use to 4?") # Example instruction
    parser.add_argument("--output_dir", type=str, default="data/output")
    return parser.parse_args()

def visualize_action(image_path, output_dir, parsed_action, original_size, resized_size):
    """
    Visualizes the action on the image and saves it.
    """
    img = Image.open(image_path)
    width, height = original_size
    new_width, new_height = resized_size
    
    # Create figure
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(img)
    
    action_type = parsed_action.get("action_type")
    action_inputs = parsed_action.get("action_inputs", {})
    
    # Handle click/point actions
    start_box = action_inputs.get("start_box")
    if start_box:
        # start_box format is expected to be '(x,y)' string from parse_output
        try:
            if isinstance(start_box, str):
                match = re.search(r"\((\d+),(\d+)\)", start_box.replace(" ", ""))
                if match:
                    model_x, model_y = int(match.group(1)), int(match.group(2))
                    
                    real_x = int(model_x / new_width * width)
                    real_y = int(model_y / new_height * height)
                    
                    plt.scatter([real_x], [real_y], c='red', s=100, marker='o')
                    plt.title(f"Action: {action_type} at ({real_x}, {real_y})")
        except Exception as e:
            logger.error(f"Error parsing coordinates: {e}")

    plt.axis('off')
    
    output_filename = Path(output_dir) / f"{Path(image_path).stem}_annotated.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Saved annotated image to {output_filename}")

def main():
    args = get_args()
    model, processor = create_model(args.model_path)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    logger.add(Path(args.output_dir) / "log.txt")
    
    if not os.path.exists(args.image_path):
        logger.warning(f"Image {args.image_path} does not exist.")
        return

    image_path = args.image_path
    image_name = os.path.basename(image_path)
            
    with Image.open(image_path) as img:
        width, height = img.size
        resized_height, resized_width = smart_resize(height, width)

        messages = create_template_messages(image_path, args.prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        inputs.pop("hm_pixel_values", None) 
        
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Image: {image_name}, Output: {output_text}")
        
        # Save raw text output
        with open(Path(args.output_dir) / f"{Path(image_name).stem}.txt", "w") as f:
            f.write(output_text)
            
        # Parse and Visualize
        parsed_action = parse_output(output_text)
        if parsed_action:
            thought = parsed_action.get("thought")
            if thought:
                print("\n" + "="*20 + " Model Thought " + "="*20)
                print(thought)
                print("="*55 + "\n")

            visualize_action(
                image_path, 
                args.output_dir, 
                parsed_action, 
                original_size=(width, height), 
                resized_size=(resized_width, resized_height)
            )

if __name__ == "__main__":
    main()
