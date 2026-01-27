# Copyright 2025 HOUMO AI
#
# File: gen_calib_data.py
# Description:
#   Example script: llm/uitars1_5/gen_calib_data.py
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
from pathlib import Path
from tqdm import tqdm
from ui_tars_utils import COMPUTER_USE_DOUBAO

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenspot_imgs", type=str, required=True, help="ScreenSpot images directory")
    parser.add_argument("--screenspot_test", type=str, required=True, help="ScreenSpot task json directory")
    parser.add_argument("--task", type=str, default="all", help="Task name(s): all or comma-separated")
    parser.add_argument("--output_file", type=str, default="data/calib_data.json", help="Output JSON file")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of samples to generate")
    return parser.parse_args()

def create_template_messages(image_path, instruction):
    """
    Creates the message structure expected by Qwen2.5-VL processor.
    """
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


def load_tasks(screenspot_test_dir, task_arg):
    if task_arg == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(screenspot_test_dir)
            if f.endswith(".json")
        ]
        task_filenames.sort()
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
            if "instruction" not in item or "img_filename" not in item:
                continue
            samples.append(
                {
                    "task_filename": task_name,
                    "img_filename": item["img_filename"],
                    "instruction": item["instruction"],
                }
            )
    return samples


def collect_samples_from_screenspot(screenspot_test, screenspot_imgs, task, num_samples):
    if not os.path.exists(screenspot_test):
        raise FileNotFoundError(f"ScreenSpot task dir {screenspot_test} does not exist.")
    if not os.path.exists(screenspot_imgs):
        raise FileNotFoundError(f"ScreenSpot images dir {screenspot_imgs} does not exist.")

    samples = load_tasks(screenspot_test, task)
    if num_samples and num_samples > 0:
        samples = samples[:num_samples]
    resolved = []
    for s in samples:
        img_path = os.path.join(screenspot_imgs, s["img_filename"])
        if not os.path.exists(img_path):
            continue
        resolved.append({"img_path": img_path, "instruction": s["instruction"]})
    return resolved

def main():
    args = get_args()

    try:
        samples = collect_samples_from_screenspot(
            args.screenspot_test, args.screenspot_imgs, args.task, args.num_samples
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    if not samples:
        print("No valid ScreenSpot samples found.")
        return
    print(f"Collected {len(samples)} ScreenSpot samples.")
    data = []
    for s in tqdm(samples, desc="Generating samples"):
        abs_path = os.path.abspath(s["img_path"])
        messages = create_template_messages(abs_path, s["instruction"])
        data.append(messages)
        
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
    print(f"Successfully saved {len(data)} calibration samples to {output_path}")

if __name__ == "__main__":
    main()
