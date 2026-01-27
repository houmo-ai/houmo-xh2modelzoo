# Copyright 2025 HOUMO AI
#
# File: yolov3_export.py
# Description:
#   Example script: cv/yolo/yolov3/yolov3_export.py
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
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from xhquant.api import (
    Config,
    ConfigDict,
    DeviceType,
    FrontendType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    export_onnx,
    get_root_logger,
    ptq_quantize,
    to_export_graph,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.common.types import DeviceType, FrontendType, PrecisionMode
from xhquant.utils.logger import padding_message
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

from xh2_model_zoo.xh_cv.models import yolov3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="yolov3_640x640", type=str, help="[yolov3_640x640, yolov3_416x416]")
    parser.add_argument("--onnx_without_postprocess", action="store_false")
    parser.add_argument(
        "--input_shape", default=[1, 3, 640, 640], type=int, nargs="+", help="[h,w] use custom onnx should apply"
    )
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    parser.add_argument("--save_golden", action="store_true", help="save golden model")
    parser.add_argument("--calib_num", type=int, default=32, help="calib num")
    parser.add_argument("--test_batch_size", type=int, default=32, help="test batch size")
    parser.add_argument("--test_num", type=int, default=None, help="test num")
    parser.add_argument("--no_eval", action="store_true", help="no eval")
    parser.add_argument("--test", action="store_true", default=False, help="test mode")
    args = parser.parse_args()
    return args


def main(args):
    logger = get_root_logger()
    torch.manual_seed(1024)
    hm_yolov3 = yolov3.HMYolov3(
        model_name=args.onnx_path,
        device=args.device,
        onnx_without_postprocess=args.onnx_without_postprocess,
        input_shape=args.input_shape,
    )

    onnx_name = args.onnx_path
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    start = time.time()
    logger.info(f"Start convert onnx to hmonnx, time: {start}")
    if not os.path.exists(out_hmonnx_file):
        input_names = ["images"]
        output_names = ["outs"]
        input_args: List[Tensor] = [torch.randn(args.input_shape, dtype=torch.float32)]

        fronted_graph_module = to_frontend_graph(hm_yolov3.model, FrontendType.ONNX, input_args)
        logger.debug(padding_message("Frontend graph"))
        logger.debug(f"{fronted_graph_module.graph}")
        _input_names = fronted_graph_module.get_input_names()

        if quant_config is None:
            quant_config = ConfigDict()
        if isinstance(quant_config, dict):
            quant_config = ConfigDict(quant_config)

        if "inputs" not in quant_config:
            quant_config.inputs = ConfigDict()

        for input_name, input_arg in zip(_input_names, input_args):
            input_qconfig = ConfigDict(
                dict(
                    quantizer=dict(
                        qspec=dict(),
                    ),
                )
            )
            if isinstance(input_arg, torch.Tensor):
                if input_arg.dtype in [torch.float32, torch.float64, torch.float16]:
                    input_qconfig.quantizer.qspec.fake_dtype = "float16"
                elif input_arg.dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
                    input_qconfig.quantizer.qspec.fake_dtype = "int32"
                else:
                    raise ValueError(f"Unsupported dtype: {input_arg.dtype}")

            quant_config.inputs[input_name] = input_qconfig
        logger.debug(f"{Config(quant_config, format_python_code=False).pretty_text}")

        quanted_graph_module = to_quant_graph(fronted_graph_module, str(DeviceType.XH2a), quant_config)
        logger.debug(padding_message("Quanted graph"))
        logger.debug(f"{quanted_graph_module.graph}")
        logger.debug(padding_message("Quanted graph"))
        execution_devce = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 对权重进行量化
        ptq_quantize(quanted_graph_module, [input_args], PrecisionMode.ALIGNED, execution_devce)
        exported_graph_module = to_export_graph(quanted_graph_module, input_args)
        # export_cfg = ConfigDict(
        #     dict(
        #         input_names=input_names,
        #         output_names=output_names,
        #     )
        # )
        export_onnx(exported_graph_module, out_hmonnx_file)
        end = time.time()
        logger.info(f"Convert onnx to hmonnx success, time: {end - start}")
        logger.info(f"out hmonnx file to: {out_hmonnx_file}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        session = HMONNXGoldenInference(out_hmonnx_file)
        session.to(device)
        session.save_golden = True
        session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
        session.step = 0
        session(input_args[0].to(torch.float16).to(device))

    session = HMONNXInference(out_hmonnx_file)
    if args.save_golden:
        start = time.time()
        logger.info(f"Start save golden model, time: {start}")
        session.save_golden = True
        session.save_golden_dir = work_dirs / "hmonnx" / "golden"
        end = time.time()
        logger.info(f"Save golden model success, time: {end - start}")
        logger.info(f"Save golden model to: {work_dirs / 'hmonnx' / 'golden' / f'{onnx_name}_{target_device}.onnx'}")

    torch.set_grad_enabled(False)
    if not args.no_eval:
        session.save_golden = False
        calib_dataset, test_dataset = hm_yolov3.dataset(
            calib_num=args.calib_num, test_batch_size=1, subset=args.test_num
        )
        for i, (input, target, img_path) in enumerate(tqdm(test_dataset)):
            pre_out, labels_out = hm_yolov3.pre_process(input, target)
            nn_out = session.cuda().forward((pre_out / 255.0).cuda().half())

            # # === ONNX 推理和误差对比 ===
            # ort_session = ort.InferenceSession('data/model_zoo2/houmo/yolov3/yolov3_without_ptprocess.onnx', providers=["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"])
            # input_name = ort_session.get_inputs()[0].name
            # ort_inputs = {input_name: (pre_out/255.).cpu().numpy() if hasattr(pre_out, 'cpu') else pre_out}
            # ort_outputs = ort_session.run(None, ort_inputs)
            # # 假设只对第一个输出做对比
            # onnx_out = ort_outputs[0]
            # if isinstance(nn_out, (list, tuple)):
            #     nn_out_tensor = nn_out[0]
            # else:
            #     nn_out_tensor = nn_out
            # if hasattr(nn_out_tensor, 'detach'):
            #     nn_out_tensor = nn_out_tensor.detach().cpu().numpy()
            # # 误差计算
            # abs_diff = np.max(np.abs(nn_out_tensor - onnx_out))
            # max_v = np.max(np.abs(onnx_out))
            # max_rel_error = abs_diff / max_v if max_v != 0 else 0.0
            # mse_error = np.mean((nn_out_tensor - onnx_out) ** 2)
            # print(f"[ONNX对比] max_abs_error={abs_diff:.6f}, max_rel_error={max_rel_error:.6f}, mse={mse_error:.6f}")

            post_out = hm_yolov3.post_process(pre_out, nn_out, labels_out, path=img_path)
            if i == 5:
                hm_yolov3.evaluate()
        hm_yolov3.evaluate()


if __name__ == "__main__":
    args = parse_args()
    main(args)
