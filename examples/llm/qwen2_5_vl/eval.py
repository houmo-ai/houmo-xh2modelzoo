# Copyright 2025 HOUMO AI
#
# File: eval.py
# Description:
#   Example script: llm/qwen2_5_vl/eval.py
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

from vlmeval.smp import dump, tabulate, pd
from vlmeval.config import qwen2vl_series, Qwen2VLChat
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin

import argparse
import torch
import datetime
import os
import json
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_dir", type=str, default="test")
    parser.add_argument("--dataset_name", type=str, default="CMMMU_VAL")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


@torch.no_grad()
def eval_model(model, dataset, dataset_name, model_name, verbose=False):
    """
    Evaluate the model on the dataset.
    Args:
        model: The model to evaluate.
        dataset: The dataset to evaluate on.
        dataset_name: The name of the dataset.
        model_name: The name of the model.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = f"output/{model_name}_{dataset_name}_{timestamp}.xlsx"
    os.makedirs("output", exist_ok=True)
    res = {}
    lt = len(dataset.data)

    struct_data_file = f"output/{model_name}_{dataset_name}_{timestamp}_struct.json"

    struct_data = []
    data_indices = [i for i in dataset.data["index"]]
    for i in tqdm(range(lt)):
        idx = dataset.data.iloc[i]["index"]
        if idx in res:
            continue

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])
        response = model.generate(message=struct, dataset=dataset_name)
        if verbose:
            print(response, flush=True)
        res[idx] = response

        struct_data.append({
            "index": idx,
            "struct": struct,
            "response": response,
        })
    dump(struct_data, struct_data_file)
    res = {k: res[k] for k in data_indices}

    data = dataset.data
    for x in data["index"]:
        assert x in res
    data["prediction"] = [str(res[x]) for x in data["index"]]
    if "image" in data:
        data.pop("image")

    dump(data, result_file)
    
    judge_kwargs = dict()
    eval_results = dataset.evaluate(result_file, **judge_kwargs)
    if eval_results is not None:
        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
        print(
            f"The evaluation of model {model_name} x dataset {dataset_name} has finished! "
        )
        print("Evaluation Results:")
    if isinstance(eval_results, dict):
        print("\n" + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        if len(eval_results) < len(eval_results.columns):
            eval_results = eval_results.T
        print("\n" + tabulate(eval_results))
        

def main():
    args = get_args()
    model = Qwen2VLChat(model_path=args.model_id, min_pixels=1280 * 28 * 28, max_pixels=16384 * 28 * 28, use_custom_prompt=False)
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(args.dataset_name)
    eval_model(model, dataset, args.dataset_name, args.model_name, args.verbose)

if __name__ == "__main__":
    main()