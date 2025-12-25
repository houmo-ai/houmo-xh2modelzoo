import os
import json
import argparse
from types import SimpleNamespace
import multiprocessing as mp

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import math

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = 28,
                 min_pixels: int = 100 * 28 * 28,
                 max_pixels: int = 16384 * 28 * 28) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
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

from ui_tars_utils import construct_prompt, parse_output, parse_coordinates


def build_model(model_path, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16
        # dtype=torch.float32
        # dtype=torch.float16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor.min_pixels = 100 * 28 * 28
    processor.image_processor.max_pixels = 16384 * 28 * 28
    return SimpleNamespace(model=model, tokenizer=tokenizer, processor=processor)


def ground_only_positive(model_bundle, instruction, image_path, device, language="en"):
    if isinstance(image_path, str):
        assert os.path.exists(image_path) and os.path.isfile(image_path)
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
        
    processor = model_bundle.processor
    
    # Note: UI-TARS official code uses factor=28 and max_pixels=16384*28*28
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=28,
        min_pixels=100 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    )
    resized_image = image.resize((resized_width, resized_height))
    
    messages = construct_prompt(resized_image, instruction, language=language, mode="grounding")
    
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text_input],
        images=[resized_image],
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    generated_ids = model_bundle.model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    result_dict = {
        "result": "positive",
        "format": "x1y1x2y2",
        "raw_response": response_text,
        "bbox": None,
        "point": None
    }
    
    # Use the new output parser
    parsed_output = parse_output(response_text)
    
    if parsed_output["action_inputs"].get("start_box"):
        box_str = parsed_output["action_inputs"]["start_box"]
        x, y = parse_coordinates(box_str)
        
        if resized_width > 0 and resized_height > 0:
            result_dict["point"] = [x / resized_width, y / resized_height]
            
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


worker_model_bundle = None
worker_device = None


def worker_initializer(gpu_queue, model_path):
    global worker_model_bundle
    global worker_device
    gpu_index = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = f"cuda:{gpu_index}"
    else:
        device = "cpu"
    worker_device = device
    worker_model_bundle = build_model(model_path, device)


def worker_process(sample):
    global worker_model_bundle
    global worker_device
    img_path = os.path.join(sample["screenspot_imgs"], sample["img_filename"])
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
        "lang": sample["language"],
        "instruction_style": sample["instruction_style"],
        "prompt_to_evaluate": sample["prompt_to_evaluate"],
        "gt_type": sample["gt_type"],
        "ui_type": sample.get("ui_type"),
        "task_filename": sample["task_filename"],
        "pred": point_in_pixel,
        "raw_response": response.get("raw_response")
    }
    correctness = eval_sample_positive_gt(sample, response)
    sample_result["bbox"] = sample["bbox"]
    sample_result["correctness"] = correctness
    return sample_result


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--screenspot_imgs", type=str, required=True)
    parser.add_argument("--screenspot_test", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpu", type=int, default=1)
    return parser.parse_args()


def run_single(args, device):
    model_bundle = build_model(args.model_path, device)
    samples = load_tasks(args.screenspot_test, args.task)
    results = []
    for sample in tqdm(samples):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
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
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"],
            "gt_type": sample["gt_type"],
            "ui_type": sample.get("ui_type"),
            "task_filename": sample["task_filename"],
            "pred": point_in_pixel,
            "raw_response": response.get("raw_response")
        }
        correctness = eval_sample_positive_gt(sample, response)
        sample_result["bbox"] = sample["bbox"]
        sample_result["correctness"] = correctness
        results.append(sample_result)
    report = evaluate_overall(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w") as f:
        json.dump(report, f, indent=4)


def run_parallel(args, num_gpu):
    samples = load_tasks(args.screenspot_test, args.task)
    for sample in samples:
        sample["screenspot_imgs"] = args.screenspot_imgs
    gpu_queue = mp.Queue()
    for i in range(num_gpu):
        gpu_queue.put(i)
    results = []
    with mp.Pool(
        processes=num_gpu,
        initializer=worker_initializer,
        initargs=(gpu_queue, args.model_path)
    ) as pool:
        for sample_result in tqdm(
            pool.imap_unordered(worker_process, samples),
            total=len(samples)
        ):
            results.append(sample_result)
    report = evaluate_overall(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, "w") as f:
        json.dump(report, f, indent=4)


def main():
    args = parse_args()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if args.num_gpu > 1 and torch.cuda.is_available():
        available = torch.cuda.device_count()
        num_gpu = min(args.num_gpu, available)
        if num_gpu < 1:
            device = args.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            run_single(args, device)
        else:
            run_parallel(args, num_gpu)
    else:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        run_single(args, device)


if __name__ == "__main__":
    main()
