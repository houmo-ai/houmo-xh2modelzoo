# Copyright 2025 HOUMO AI
#
# File: vllm_custom_dataset.py
# Description:
#   Custom dataset class for VLLM calibration data.
#   This module provides dataset formatting and loading functionality for
#   Qwen3-VL and Qwen2-VL models, supporting various calibration datasets
#   including CMMMU, MMMU, OCRBench, DocVQA, and COCO.
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
from torch.utils.data import Dataset
import json
from pathlib import Path
from .builder import DATASETS


def format_qwen3_vl_dataset_for_calibration(item):
    content = []
    for p_input in item["struct"]:
        content.append({"type": p_input["type"], p_input["type"]: p_input["value"]})
    format_item = [{"role": "user", "content": content}, {"role": "assistant", "content": item["response"]}]
    return format_item


def format_qwen2_vl_dataset(image, text, assistant):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
        {"role": "assistant", "content": assistant},
    ]

def format_qwen2_vl_dataset_for_calibration(item):
    content = []
    for p_input in item["struct"]:
        content.append({"type": p_input["type"], p_input["type"]: p_input["value"]})
    format_item = [{"role": "user", "content": content}, {"role": "assistant", "content": item["response"]}]
    return format_item

@DATASETS.register_module()
class VLLMCustomDataset(Dataset):
    def __init__(self, data_files: list[str]):
        super().__init__()
        self.data_files = data_files
        
        if len(data_files) != 0:
            self.data = []

            for data_file in data_files:
                with open(data_file, "r") as f:
                    data = json.load(f)
                for item in data:
                    formatted = format_qwen3_vl_dataset_for_calibration(item)
                    for message in formatted:
                        content = message.get("content")
                        if not isinstance(content, list):
                            continue
                        for entry in content:
                            if entry.get("type") == "image" and isinstance(entry.get("image"), str):
                                entry["image"] = str(Path(entry["image"]).expanduser())
                    self.data.append(formatted)

            self.data = sorted(self.data, key=lambda x: len(x[1]["content"]), reverse=True)
        else:
            calibration_dataset = json.load(open("data/calib_data/result_base_case_qwen2_5_vl.json", "r"))
            calibration_dataset = [format_qwen2_vl_dataset(item["struct"][0]["content"][0]["image"], item["struct"][0]["content"][1]["text"], item["response"]) for item in calibration_dataset]

            cmmmu_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_CMMMU_VAL_20250923102519_struct.json", "r"))
            cmmmu_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in cmmmu_val_struct]

            cmmmu_val_struct = sorted(cmmmu_val_struct, key=lambda x: len(x[1]["content"]))[-128:]
            calibration_dataset = calibration_dataset + cmmmu_val_struct

            mmmu_dev_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_MMMU_DEV_VAL_20250923102615_struct.json", "r"))
            mmmu_dev_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in mmmu_dev_val_struct]
            mmmu_dev_val_struct = sorted(mmmu_dev_val_struct, key=lambda x: len(x[1]["content"]))[-64:]
            calibration_dataset = calibration_dataset + mmmu_dev_val_struct

            ocrbench_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_OCRBench_20250923102641_struct.json", "r"))
            ocrbench_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in ocrbench_struct]
            ocrbench_struct = sorted(ocrbench_struct, key=lambda x: len(x[1]["content"]))[-32:]
            calibration_dataset = calibration_dataset + ocrbench_struct

            docvqa_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_DocVQA_VAL_20250923102720_struct.json", "r")) 
            docvqa_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in docvqa_val_struct]
            docvqa_val_struct = sorted(docvqa_val_struct, key=lambda x: len(x[1]["content"]))[-32:]
            calibration_dataset = calibration_dataset + docvqa_val_struct

            coco_val_struct = json.load(open("data/calib_data/Qwen2.5-VL-7B-Instruct_COCO_VAL_20250923104643_struct.json", "r"))
            coco_val_struct = [format_qwen2_vl_dataset_for_calibration(item) for item in coco_val_struct]
            coco_val_struct = sorted(coco_val_struct, key=lambda x: len(x[1]["content"]))[-32:]
            calibration_dataset = calibration_dataset + coco_val_struct

            self.data = calibration_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
