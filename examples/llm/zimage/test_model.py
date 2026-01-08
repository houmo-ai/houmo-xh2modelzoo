import tempfile
from copy import deepcopy
from pathlib import Path

import cv2
import onnx
import torch
import torch.nn as nn
from torch import Tensor

from xhquant.api import (
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_onnx_to_hmonnx,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
)
from xhquant.api.ptq_export_hmonnx import FrontendType, _convert_model_to_quanted_model, normalized_onnx
from xhquant.export.onnx.onnx_opset.schema.xh2a_schema import RESIZE_OP_SCHEMA
from xhquant.api import DeviceType
from xhquant.nn.modules.resizer import OnnxResize
from xhquant.nn.modules.onnx_style_modules import ImageResize
import cv2
from xhquant.api.ptq_export_hmonnx import normalized_onnx, _convert_model_to_quanted_model, FrontendType
# from .common import assert_check_op, op_test_context
import cv2
import numpy as np


def _test_resize_onnx_export(work_dir: str, save_golden: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type="w4a8h1_ssfp")
    quant_config = create_quant_config(quant_scheme)
    quant_config = ConfigDict(quant_config)
    # x = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8 ,device=device)
    
    hidd = torch.rand((1, 31800, 2048), device=device)
    encoder_hid = torch.rand((1,1985,2048), device=device)
    temb = torch.rand((1, 2048), device=device)
    encoder_atten_mask = torch.ones((1, 1985), device=device)
    encoder_atten_mask[:, 1000:] = 0
    rotat_emb = torch.rand((31800,128), device=device)
    rotat_emb2 = torch.rand((31800,128), device=device)

    hm_onnx_file = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/zimage/transformer_first_layer.onnx"

    # onnx_model = onnx.load("/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/zimage/transformer_first_layer.onnx")
    
    model_inputs = [
        hidd.half(),
        encoder_hid.half(),
        temb.half(),
        encoder_atten_mask.bool(),
        rotat_emb.half(),
        rotat_emb2.half(),
    ]

    # onnx_model.to(device)

    model_name = f"zimage_transformer_first_layer"
    # Path(work_dir).mkdir(parents=True, exist_ok=True)
    # onnx_file = str(Path(work_dir) / "onnx" / f"{model_name}.onnx")
    # hm_onnx_file = str(Path(work_dir) / "hmonnx" / f"{model_name}_hm.onnx")


    # Path(work_dir + "/hmonnx").mkdir(parents=True, exist_ok=True)

    # import onnxsim
    # onnx_model, checked = onnxsim.simplify(onnx_file)
    
    # device_type = DeviceType(DeviceType.XH2a)

    # onnx_model = normalized_onnx(onnx_model)
    # input_names = [input.name for input in onnx_model.graph.input]
    # output_names = [output.name for output in onnx_model.graph.output]

    # quanted_graph_module = _convert_model_to_quanted_model(
    #     onnx_model, FrontendType.ONNX, model_inputs, device_type, quant_config, use_ptq=False
    # )

    # out = quanted_graph_module(model_inputs)

    # convert_onnx_to_hmonnx(onnx_model, model_inputs, target_device, hm_onnx_file)

    if save_golden:
        session = HMONNXGoldenInference(hm_onnx_file)
        session.to(device)
        session.save_golden = save_golden
        session.golden_dir = work_dir + f"/hmonnx/golden_{model_name}"
        session.step = 0
    else:
        session = HMONNXInference(hm_onnx_file)
        session.to(device)
    
    out = session(*model_inputs)



if __name__ == "__main__":
    _test_resize_onnx_export(work_dir="work_dirs",save_golden=True)
    # _test_groupnorm_affine_onnx_11_export(True)
