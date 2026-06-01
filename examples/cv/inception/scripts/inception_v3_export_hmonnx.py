# Copyright 2025 HOUMO AI
#
# File: inception_v3_export_hmonnx.py
# Description:
#   Example script: cv/inception/scripts/inception_v3_export_hmonnx.py
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

import torch
import torch.nn.functional as F
import xhquant.xhonnxruntime.config
from tqdm import tqdm
from xhquant.api import Config, HMONNXInference, get_root_logger, xhquant_init

from xh2_model_zoo.data_preprocessors import ClsDataPreprocessor
from xh2_model_zoo.registry import MODELS
from xh2_model_zoo.runner import Runner
from xh2_model_zoo.structures import DataSample


def _get_predictions(cls_score, data_samples):
    """Post-process the output of head.

    Including softmax and set ``pred_label`` of data samples.
    """
    pred_scores = F.softmax(cls_score, dim=1)
    pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

    out_data_samples = []
    if data_samples is None:
        data_samples = [None for _ in range(pred_scores.size(0))]

    for data_sample, score, label in zip(data_samples, pred_scores, pred_labels):
        if data_sample is None:
            data_sample = DataSample()

        data_sample.set_pred_score(score).set_pred_label(label)
        out_data_samples.append(data_sample)
    return out_data_samples


def main(args):
    xhquant_init(None, debug=args.debug)
    data_config = Config.fromfile(args.dataset)
    data_config["val_dataloader"]["batch_size"] = args.batch_size
    data_config["val_dataloader"]["drop_last"] = True
    val_dataloader = Runner.build_dataloader(data_config.get("val_dataloader"))

    data_preprocessor_cfg = dict(
        type="ClsDataPreprocessor",
        num_classes=1000,
        # RGB format normalization parameters
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        # convert image from BGR to RGB
        to_rgb=True,
    )
    data_preprocessor: ClsDataPreprocessor = MODELS.build(data_preprocessor_cfg)

    test_evaluator = Runner.build_evaluator(data_config.get("test_evaluator"))

    session = HMONNXInference(args.hmonnx)
    session.to_fast_mode()
    exec_device = torch.device("cuda")
    dtype = torch.float16
    session.to(exec_device)

    data_preprocessor.to(dtype=dtype, device=exec_device)

    # model = DataParallel(session, device_ids=range(4))
    model = session

    logger = get_root_logger()
    logger.info("session is created successfully")
    xhquant.xhonnxruntime.config.disable_progress = True
    for data_batch in tqdm(val_dataloader):
        data_batch = data_preprocessor(data_batch)
        inputs = data_batch["inputs"]
        inputs = inputs.to(dtype)
        data_samples = data_batch["data_samples"]
        cls_score = model(inputs)
        outputs = _get_predictions(cls_score, data_samples)
        test_evaluator.process(data_samples=outputs, data_batch=data_batch)

    val_dataset = val_dataloader.dataset
    metrics = test_evaluator.evaluate(len(val_dataset))  # type: ignore[arg-type]
    logger.info(f"Test metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmonnx",
        type=str,
        default="work_dirs/resnet50_224x224_batch_1/hmonnx/resnet50_224x224_batch_1_XH2a.onnx",
    )
    parser.add_argument("--dataset", type=str, default="configs/datasets/imagenet_224x224.py")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    main(args)
