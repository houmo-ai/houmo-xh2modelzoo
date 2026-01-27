# Copyright 2025 HOUMO AI
#
# File: onnx2hmonnx_debug.py
# Description:
#   Example script: debug/onnx2hmonnx_debug.py
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
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import xhquant.xhonnxruntime.config
from onnx import TensorProto
from torch import Tensor
from torch.fx import Interpreter
from xhquant import PrecisionMode, to_quant_graph
from xhquant.api import (
    Config,
    ConfigDict,
    FrontendType,
    HMONNXInference,
    export_onnx,
    get_root_logger,
    ptq_quantize,
    query_device,
    to_export_graph,
    to_frontend_graph,
    xhquant_init,
)
from xhquant.utils.utils import map_aggregate

TENSOR_TYPE_TO_TORCH_TYPE = {
    int(TensorProto.FLOAT): torch.float32,
    int(TensorProto.UINT8): torch.uint8,
    int(TensorProto.INT8): torch.int8,
    int(TensorProto.INT16): torch.int16,
    int(TensorProto.INT32): torch.int32,
    int(TensorProto.INT64): torch.int64,
    int(TensorProto.BOOL): torch.bool,
    int(TensorProto.FLOAT16): torch.float16,
    int(TensorProto.DOUBLE): torch.float64,
    int(TensorProto.COMPLEX64): torch.complex64,
    int(TensorProto.COMPLEX128): torch.complex128,
}


def main(args):
    cfg = Config.fromfile(args.config)
    target_device = query_device(cfg.target_device)
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    cfg_name = f"{onnx_name}_{target_device.name}"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    cfg.work_dir = str(work_dir)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"

    xhquant_init(log_file, debug=args.debug)

    device = torch.device("cuda:0")
    logger = get_root_logger()
    logger.info(f"Target device: {target_device}")
    logger.info("*************** config ***************\n{cfg.pretty_text}")
    logger.info(cfg)
    onnx_model = onnx.load(onnx_file)

    initializer_names = [init.name for init in onnx_model.graph.initializer]
    inilializer_as_inputs: List[str] = []
    for input in onnx_model.graph.input:
        if input.name in initializer_names:
            inilializer_as_inputs.append(input)
    for input in inilializer_as_inputs:
        onnx_model.graph.input.remove(input)

    inputs: List[Tensor] = []
    input_names: List[str] = []
    for input in onnx_model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input.type.tensor_type.shape.dim]
        dtype = TENSOR_TYPE_TO_TORCH_TYPE[input.type.tensor_type.elem_type]
        logger.info(f"inputs: {input.name}, {shape}, {dtype}")
        input_names.append(input.name)
        if dtype == torch.float32:
            inputs.append(torch.randn(shape, dtype=dtype))
        elif dtype in [torch.int32, torch.int64]:
            inputs.append(torch.randint(0, 10, shape, dtype=torch.int32))
        elif dtype == torch.bool:
            inputs.append(torch.randint(0, 2, shape, dtype=dtype))
        else:
            raise NotImplementedError(f"dtype {dtype} not supported")

    ort_check = False
    ort_outputs = None
    if ort_check:
        providers = []
        provider_options = []
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
            provider_options.append({"device_id": 0})
        else:
            providers.append("CPUExecutionProvider")
        so = ort.SessionOptions()

        ort_session = ort.InferenceSession(onnx_file, so, providers=providers, provider_options=provider_options)
        ort_inputs: Dict[str, np.ndarray] = {name: input.numpy() for name, input in zip(input_names, inputs)}
        ort_outputs = ort_session.run(None, ort_inputs)

    onnx_input_names = [input.name for input in onnx_model.graph.input]
    onnx_output_names = [output.name for output in onnx_model.graph.output]

    fronted_graph_module = to_frontend_graph(onnx_file, FrontendType.ONNX, inputs)

    _input_names = fronted_graph_module.get_input_names()
    quant_config = cfg.quant_config

    if "inputs" not in quant_config:
        quant_config.inputs = ConfigDict()

    for input_name, input_arg in zip(_input_names, inputs):
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
    logger.info(quant_config)

    quanted_graph_module = to_quant_graph(fronted_graph_module, target_device.name, quant_config)

    ptq_quantize(quanted_graph_module, [inputs], PrecisionMode.ALIGNED, [device])

    inputs = map_aggregate(inputs, lambda x: x.cpu())
    inputs = map_aggregate(inputs, lambda x: x.half() if x.dtype == torch.float32 else x)
    quanted_graph_module.to(torch.float16)
    quanted_graph_module.cpu()

    logger.info("**************** Exported to export_graph ****************")
    exported_graph_module = to_export_graph(
        quanted_graph_module,
        inputs,
    )
    logger.info("**************** Exported to onnx ****************")
    logger.info(exported_graph_module.graph)
    out_onnx_file = str(work_dir / f"{onnx_name}_{target_device.name}.onnx")

    export_cfg = dict(
        input_names=onnx_input_names,
        output_names=onnx_output_names,
    )
    export_onnx(exported_graph_module, out_onnx_file, export_cfg)
    logger.info(f"Save to {out_onnx_file}")

    xhquant.xhonnxruntime.config.verbose_progress = True
    session = HMONNXInference(out_onnx_file)
    session.to(device)
    session.exec_device = device
    session(*inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default="data/models/baidu/modified_model_c3_camera_v2_960x544_si.onnx", help="onnx file"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--config", default="./configs/xh2a/base_xh2a.py", help="config file")
    args = parser.parse_args()
    main(args)
