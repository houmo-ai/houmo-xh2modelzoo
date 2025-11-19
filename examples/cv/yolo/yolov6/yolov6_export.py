import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import torch
from torch import Tensor
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    ptq_quantize,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.common.types import DeviceType, FrontendType, PrecisionMode
from xhquant.export import export_onnx, to_export_graph
from xhquant.mix_precision.mix_precision import MixPrecisionSearch
from xhquant.utils.config import Config, ConfigDict
from xhquant.utils.logger import get_root_logger, padding_message
from yolov6_hmonnx_test import preprocess_image

from xh2_model_zoo.utils.onnx.onnx_shape_infer import replace_reshape_minus_one_by_infer


def main(args):
    onnx_file = args.onnx
    replace_reshape_minus_one_by_infer(
        onnx_path=onnx_file, output_path=onnx_file, input_shape=[1, 3, 640, 640]  # 可根据实际输入shape调整
    )

    onnx_name = Path(onnx_file).stem
    # 修改工作目录以匹配yolov6m
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{args.quant_type}_{target_device.name}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    execution_devce = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    fronted_graph_module = to_frontend_graph(
        onnx_file, FrontendType.ONNX, [torch.randn(1, 3, 640, 640, dtype=torch.float32)]
    )
    image = cv2.imread(args.image)
    results = preprocess_image(image, (640, 640))
    calib_dataloader = [results["inputs"].half().to(execution_devce)]
    label_dataloader = [fronted_graph_module.cuda().half()(i) for i in calib_dataloader]

    logger = get_root_logger()
    logger.debug(padding_message("Frontend graph"))
    logger.debug(f"{fronted_graph_module.graph}")
    _input_names = fronted_graph_module.get_input_names()

    logger.debug(f"{Config(quant_config, format_python_code=False).pretty_text}")

    quanted_graph_module = to_quant_graph(fronted_graph_module, target_device.name, quant_config)
    logger.debug(padding_message("Quanted graph"))
    logger.debug(f"{quanted_graph_module.graph}")
    logger.debug(padding_message("Quanted graph"))

    ## 将输入的List展开
    input_args: List[Tensor] = []
    for arg in _input_names:
        if isinstance(arg, (list, tuple)):
            input_args.extend(arg)
        else:
            input_args.append(arg)

    ptq_quantize(quanted_graph_module, [input_args], PrecisionMode.ALIGNED, execution_devce)

    if args.mix_search:
        quanted_graph_module.enable_quant()
        quanted_graph_module.enable_fast_precision_mode()

        ms_cfg = dict(
            topk=0.20,
            weight_bits=[4, 8],
            act_bits=[8],
            policy="topk",
            task="cv",
            metric="l1",
            key_name="loss",
        )
        ms = MixPrecisionSearch(ms_cfg, quanted_graph_module)

        ms.search([calib_dataloader], 2, label_dataloader)
        quanted_graph_module = ms.module

    exported_graph_module = to_export_graph(quanted_graph_module, [results["inputs"].half().to(execution_devce)])
    export_cfg = ConfigDict(
        dict(
            input_names=["images"],
            output_names=[],
        )
    )
    export_onnx(exported_graph_module, out_hmonnx_file, export_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / "hmonnx/golden_w8a16"
    session.step = 0
    session(torch.randn(1, 3, 640, 640, dtype=torch.float16).to(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改默认的onnx文件路径为yolov6m
    parser.add_argument("--onnx", type=str, default="data/models/yolo/yolov6m.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    parser.add_argument("--quant-type", default="w8a16_sefp", help="quant type, default is w8a8")
    parser.add_argument("--mix_search", default=False)
    args = parser.parse_args()
    main(args)
