# Copyright 2025 HOUMO AI
#
# File: ui_tars1_5_xh2a_demo.py
# Description:
#   Example script: llm/uitars1_5/ui_tars1_5_xh2a_demo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import torch
from PIL import Image
import math
import json
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace

from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLONNXModel, Qwen2_5_VLProcessor
from xhquant.api import get_root_logger
import xhquant.utils.suppress_printing
from ui_tars_utils import parse_output, parse_coordinates, COMPUTER_USE_DOUBAO, GROUNDING_DOUBAO

# Constants for smart resize (same as eval/demo)
IMAGE_FACTOR = 28
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
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

def build_xh_model(model_dir_str, device="cuda"):
    model_dir = Path(model_dir_str)
    
    with open(model_dir / "meta.json", "r") as f:
        meta_info = json.load(f)
    
    wrap_cfg = meta_info["wrap_cfg"]
    visual_cfg = wrap_cfg["visual"]
    
    image_feature_cfg = SimpleNamespace(
        onnx=str(model_dir / meta_info["vision_onnx"]),
        patch_size=visual_cfg["patch_size"],
        image_size_w=visual_cfg["image_max_size_w"],
        image_size_h=visual_cfg["image_max_size_h"],
        image_max_size_t=visual_cfg["image_max_size_t"],
        temporal_patch_size=visual_cfg["temporal_patch_size"],
    )

    prefill_cfg = SimpleNamespace(
        onnx=str(model_dir / meta_info["prefill_onnx"]),
        input_sequence_length=wrap_cfg["input_sequence_length"],
    )

    decode_cfg = SimpleNamespace(
        onnx=str(model_dir / meta_info["decode_onnx"]),
    )

    kv_cache_shape = [int(x) for x in meta_info["kv_cache"]["shape"]]
    kv_cache_cfg = SimpleNamespace(
        num_decoder_layers=meta_info["kv_cache"]["num_decoder_layers"],
        num_hidden_layers=meta_info["kv_cache"]["num_decoder_layers"],
        shape=kv_cache_shape,
    )

    xh_model = Qwen2_5_VLONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        image_size_w=visual_cfg["image_max_size_w"],
        image_size_h=visual_cfg["image_max_size_h"],
        max_size_t=visual_cfg["image_max_size_t"],
        resize_v1=True,
        repetition_penalty=1.0,
        chat_template=None,
    )

    xhquant.utils.suppress_printing.disable_printing = True # Enable printing to see token generation
    
    torch.serialization.add_safe_globals([torch.nn.Embedding])
    token_embedding_path = model_dir / meta_info["token_embedding_file"]
    token_embedding = torch.load(token_embedding_path, weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()
    
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(device)
    
    hf_model_config_dir = model_dir / meta_info["hf_config"]
    processor = Qwen2_5_VLProcessor.from_pretrained(hf_model_config_dir)
    
    return xh_model, processor

def visualize_action(image_path, output_path, parsed_action, original_size, resized_size, resize_v1):
    """
    Visualizes the action on the image and saves it.
    Adapted from demo.py
    """
    img = Image.open(image_path)
    width, height = original_size
    new_width, new_height = resized_size
    
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(img)
    
    action_type = parsed_action.get("action_type")
    action_inputs = parsed_action.get("action_inputs", {})
    
    # Handle click/point actions
    start_box = action_inputs.get("start_box")
    if start_box:
        try:
            # parse_coordinates returns (x, y)
            x_model, y_model = parse_coordinates(start_box)
            if resize_v1:
                scale = min(new_width / width, new_height / height)
                real_x = int(x_model / scale)
                real_y = int(y_model / scale)
            else:
                real_x = int(x_model / new_width * width)
                real_y = int(y_model / new_height * height)
            real_x = max(0, min(real_x, width - 1))
            real_y = max(0, min(real_y, height - 1))
            
            print(f"Detected Coordinates (Model): ({x_model}, {y_model})")
            print(f"Mapped to Original Image: ({real_x}, {real_y})")
            
            plt.scatter([real_x], [real_y], c='red', s=100, marker='o')
            plt.title(f"Action: {action_type} at ({real_x}, {real_y})")
        except Exception as e:
            print(f"Error parsing coordinates: {e}")

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved annotated image to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to exported model")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--instruction", type=str, default="Could you help me set the Number Threads to Use to 4?", help="Instruction")
    parser.add_argument("--output_image", type=str, default="demo_result.png", help="Output image path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()
    print(f"Loading model from {args.model_dir} to {device}...")
    
    model, processor = build_xh_model(args.model_dir, device)
    
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    image_path = image_path
    image_name = os.path.basename(image_path)
    
    with Image.open(image_path) as img:
        width, height = img.size
        prompt = COMPUTER_USE_DOUBAO.format(instruction=args.instruction, language="English")
        # prompt = GROUNDING_DOUBAO.format(instruction=args.instruction, language="English")
        output_text = model.chat(prompt, image_path, processor, logger, use_fast=False)
        
        logger.info(f"Image: {image_name}, Output: {output_text}")
        
        output_image_path = Path(args.output_image)
        output_text_path = output_image_path.with_suffix(".txt")
        with open(output_text_path, "w") as f:
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
                str(output_image_path),
                parsed_action, 
                original_size=(width, height), 
                resized_size=(model.image_size_w, model.image_size_h),
                resize_v1=model.resize_v1,
            )


if __name__ == "__main__":
    main()
