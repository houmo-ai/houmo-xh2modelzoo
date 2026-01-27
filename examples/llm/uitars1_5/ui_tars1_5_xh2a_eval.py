# Copyright 2025 HOUMO AI
#
# File: ui_tars1_5_xh2a_eval.py
# Description:
#   Example script: llm/uitars1_5/ui_tars1_5_xh2a_eval.py
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
import json
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
import math

from xh_model_zoo.xh_llm.models.qwen2_5_vl import Qwen2_5_VLONNXModel, Qwen2_5_VLProcessor
import xhquant.utils.suppress_printing
from xhquant.api import get_root_logger
from ui_tars_utils import construct_prompt, parse_output, parse_coordinates

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28  
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

def build_xh_model(args, device="cuda"):
    model_dir = Path(args.model_dir)
    
    # Load Meta Info
    with open(model_dir / "meta.json", "r") as f:
        meta_info = json.load(f)
    
    wrap_cfg = meta_info["wrap_cfg"]
    visual_cfg = wrap_cfg["visual"]
    
    # Construct Config Objects
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

    kv_cache_shape = meta_info["kv_cache"]["shape"]
    # Ensure shape is int
    kv_cache_shape = [int(x) for x in kv_cache_shape]
    
    kv_cache_cfg = SimpleNamespace(
        num_decoder_layers=meta_info["kv_cache"]["num_decoder_layers"],
        num_hidden_layers=meta_info["kv_cache"]["num_decoder_layers"],
        shape=kv_cache_shape,
    )

    # Initialize Model
    xh_model = Qwen2_5_VLONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        image_size_w=visual_cfg["image_max_size_w"],
        image_size_h=visual_cfg["image_max_size_h"],
        max_size_t=visual_cfg["image_max_size_t"],
        resize_v1=True, # Assuming default behavior
        repetition_penalty=1.0,
        chat_template=None,
    )

    xhquant.utils.suppress_printing.disable_printing = True
    
    # Load Embeddings
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding_path = model_dir / meta_info["token_embedding_file"]
    token_embedding = torch.load(token_embedding_path, weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()
    
    # Explicitly move embedding to target device to avoid multi-device conflicts
    token_embedding = token_embedding.to(device)
    
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(device)
    
    # Load Processor
    hf_model_config_dir = model_dir / meta_info["hf_config"]
    processor = Qwen2_5_VLProcessor.from_pretrained(hf_model_config_dir)
    processor.image_processor.min_pixels = MIN_PIXELS
    processor.image_processor.max_pixels = MAX_PIXELS
    
    return SimpleNamespace(model=xh_model, processor=processor)

def ground_only_positive(model_bundle, instruction, image_path, device, language="en"):
    if isinstance(image_path, str):
        assert os.path.exists(image_path) and os.path.isfile(image_path)
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
        
    # Resize Logic
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=28,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    
    resized_image = image.resize((resized_width, resized_height))
    
    messages = construct_prompt(resized_image, instruction, language=language, mode="grounding")
    text_prompt = messages[0]["content"][1]["text"] # Extract the formatted text
    
    response_text = model_bundle.model.chat(
        text_prompt, 
        image_path if isinstance(image_path, str) else resized_image, # Pass path or PIL
        model_bundle.processor, 
        get_root_logger(),
        use_fast=True 
    )
    
    result_dict = {
        "result": "positive",
        "format": "x1y1x2y2",
        "raw_response": response_text,
        "bbox": None,
        "point": None
    }
    
    parsed_output = parse_output(response_text)
    
    if parsed_output["action_inputs"].get("start_box"):
        box_str = parsed_output["action_inputs"]["start_box"]
        x, y = parse_coordinates(box_str)
        
        orig_w, orig_h = image.size
        if orig_w > 0 and orig_h > 0:
            model_w = getattr(model_bundle.model, "image_size_w", orig_w)
            model_h = getattr(model_bundle.model, "image_size_h", orig_h)
            resize_v1 = getattr(model_bundle.model, "resize_v1", True)

            if resize_v1:
                scale = min(model_w / orig_w, model_h / orig_h)
                if scale > 0:
                    x_orig = x / scale
                    y_orig = y / scale
                else:
                    x_orig = x
                    y_orig = y
            else:
                scale_x = model_w / orig_w if orig_w > 0 else 0.0
                scale_y = model_h / orig_h if orig_h > 0 else 0.0
                x_orig = x / scale_x if scale_x > 0 else x
                y_orig = y / scale_y if scale_y > 0 else y

            x_orig = max(0.0, min(float(x_orig), float(orig_w - 1)))
            y_orig = max(0.0, min(float(y_orig), float(orig_h - 1)))
            result_dict["point"] = [x_orig / orig_w, y_orig / orig_h]
            
    return result_dict

def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample["img_size"]
    bbox = [
        bbox[0] / img_size[0],
        bbox[1] / img_size[1],
        bbox[2] / img_size[0],
        bbox[3] / img_size[1]
    ]
    click_point = response.get("point")
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    return "wrong"

def calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    text_results = [res for res in results if res.get("ui_type") == "text"]
    icon_results = [res for res in results if res.get("ui_type") == "icon"]
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0.0,
        "text_acc": text_correct / text_total if text_total > 0 else 0.0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0.0
    }
    return metrics

def evaluate_overall(results):
    metrics = calc_metric_for_result_list(results)
    return {"metrics": {"overall": metrics}, "details": results}

def load_tasks(screenspot_test_dir, task_arg):
    if task_arg == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(screenspot_test_dir)
            if f.endswith(".json")
        ]
    else:
        task_filenames = [t.strip() for t in task_arg.split(",") if t.strip()]
    samples = []
    for task_name in task_filenames:
        path = os.path.join(screenspot_test_dir, task_name + ".json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            task_data = json.load(f)
        for item in task_data:
            if "bbox" not in item or "img_size" not in item or "instruction" not in item:
                continue
            sample = dict(item)
            sample["task_filename"] = task_name
            sample["prompt_to_evaluate"] = item["instruction"]
            sample["language"] = "en"
            sample["instruction_style"] = "instruction"
            sample["gt_type"] = "positive"
            samples.append(sample)
    return samples

worker_model_bundle = None
worker_device = None

def worker_initializer(gpu_queue, model_dir):
    global worker_model_bundle
    global worker_device
    gpu_index = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = f"cuda:{gpu_index}"
    else:
        device = "cpu"
    
    worker_device = device
    args_sim = SimpleNamespace(model_dir=model_dir)
    worker_model_bundle = build_xh_model(args_sim, device)
    print(f"Worker initialized on {device}", file=sys.stderr)

def worker_process(sample):
    global worker_model_bundle
    global worker_device
    
    img_path = os.path.join(sample["screenspot_imgs"], sample["img_filename"])
    
    try:
        response = ground_only_positive(
            model_bundle=worker_model_bundle,
            instruction=sample["prompt_to_evaluate"],
            image_path=img_path,
            device=worker_device,
            language=sample.get("language", "en"),
        )
        
        point = response.get("point")
        img_size = sample["img_size"]
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None
        
        sample_result = {
            "id": sample.get("id"),
            "img_path": img_path,
            "group": sample.get("group"),
            "platform": sample.get("platform"),
            "application": sample.get("application"),
            "lang": sample.get("language"),
            "instruction_style": sample.get("instruction_style"),
            "prompt_to_evaluate": sample["prompt_to_evaluate"],
            "gt_type": sample.get("gt_type"),
            "ui_type": sample.get("ui_type"),
            "task_filename": sample["task_filename"],
            "pred": point_in_pixel,
            "raw_response": response.get("raw_response")
        }
        
        correctness = eval_sample_positive_gt(sample, response)
        sample_result["bbox"] = sample["bbox"]
        sample_result["correctness"] = correctness
        return sample_result
    except Exception as e:
        print(f"Error processing {sample['img_filename']}: {e}", file=sys.stderr)
        return {
            "id": sample.get("id"),
            "correctness": "error",
            "error": str(e)
        }

def run_single(args, device):
    model_bundle = build_xh_model(args, device)
    samples = load_tasks(args.screenspot_test, args.task)
    total_samples = len(samples)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
    
    results = []
    if len(samples) != total_samples:
        print(f"Start evaluation on {len(samples)}/{total_samples} samples...", file=sys.stderr)
    else:
        print(f"Start evaluation on {len(samples)} samples...", file=sys.stderr)
    
    for sample in tqdm(samples, file=sys.stdout, disable=False, dynamic_ncols=True):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        
        try:
            response = ground_only_positive(
                model_bundle=model_bundle,
                instruction=sample["prompt_to_evaluate"],
                image_path=img_path,
                device=device,
                language=sample.get("language", "en"),
            )
            
            point = response.get("point")
            img_size = sample["img_size"]
            point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point else None
            
            sample_result = {
                "id": sample.get("id"),
                "img_path": img_path,
                "group": sample.get("group"),
                "platform": sample.get("platform"),
                "application": sample.get("application"),
                "lang": sample.get("language"),
                "instruction_style": sample.get("instruction_style"),
                "prompt_to_evaluate": sample["prompt_to_evaluate"],
                "gt_type": sample.get("gt_type"),
                "ui_type": sample.get("ui_type"),
                "task_filename": sample["task_filename"],
                "pred": point_in_pixel,
                "raw_response": response.get("raw_response")
            }
            
            correctness = eval_sample_positive_gt(sample, response)
            sample_result["bbox"] = sample["bbox"]
            sample_result["correctness"] = correctness
            results.append(sample_result)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)
            
    report = evaluate_overall(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w") as f:
        json.dump(report, f, indent=4)
    
    print("\nEvaluation Results:", file=sys.stderr)
    print(json.dumps(report["metrics"], indent=4), file=sys.stderr)

def run_parallel(args, num_gpu):
    samples = load_tasks(args.screenspot_test, args.task)
    total_samples = len(samples)
    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]
    # Inject screenspot_imgs path into sample for worker
    for sample in samples:
        sample["screenspot_imgs"] = args.screenspot_imgs
        
    if len(samples) != total_samples:
        print(f"Start parallel evaluation on {len(samples)}/{total_samples} samples using {num_gpu} GPUs...", file=sys.stderr)
    else:
        print(f"Start parallel evaluation on {len(samples)} samples using {num_gpu} GPUs...", file=sys.stderr)
    
    gpu_queue = mp.Manager().Queue()
    for i in range(num_gpu):
        gpu_queue.put(i)
        
    results = []
    # Use 'spawn' to avoid CUDA initialization issues in fork
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
        
    with mp.Pool(
        processes=num_gpu,
        initializer=worker_initializer,
        initargs=(gpu_queue, args.model_dir)
    ) as pool:
        for sample_result in tqdm(
            pool.imap_unordered(worker_process, samples),
            total=len(samples),
            file=sys.stdout,
            disable=False,
            dynamic_ncols=True,
        ):
            results.append(sample_result)
            
    report = evaluate_overall(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w") as f:
        json.dump(report, f, indent=4)
    
    print("\nEvaluation Results:", file=sys.stderr)
    print(json.dumps(report["metrics"], indent=4), file=sys.stderr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the exported HMONNX model directory")
    parser.add_argument("--screenspot_imgs", type=str, required=True)
    parser.add_argument("--screenspot_test", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.num_gpu > 1 and torch.cuda.is_available():
        available = torch.cuda.device_count()
        num_gpu = min(args.num_gpu, available)
        if num_gpu < 1:
            run_single(args, args.device)
        else:
            run_parallel(args, num_gpu)
    else:
        run_single(args, args.device)

if __name__ == "__main__":
    main()
