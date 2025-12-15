import tempfile
from copy import deepcopy
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torch import Tensor

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    ConfigDict
)
from xhquant.export.onnx.onnx_opset.schema.xh2a_schema import RESIZE_OP_SCHEMA
from xhquant.api import DeviceType
from xhquant.nn.modules.resizer import OnnxResize 
from xhquant.nn.modules.onnx_style_modules import DynamicImageResize
import cv2
from xhquant.api.ptq_export_hmonnx import normalized_onnx, _convert_model_to_quanted_model, FrontendType

# input_shapes = {
#     "x": [1, 256, 560],
#     "language": [1],
#     "text_norm": [1]
# }

# input_shapes = {
#     "encoder_attention_mask": [1, 256],
#     "input_ids": [1, 256],
#     "encoder_hidden_states": [1, 256,1024]
# }

# encoder
input_shapes = {
    "input_ids": [1, 256],
    "attention_mask": [1, 256]
}



def export_hmonnx_model(
    work_dir: str, save_golden: bool = True
):
    device = "cpu" # cuda" if torch.cuda.is_available() else 

    # onnx_file = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fix_decoder/fix_decoder.sim.onnx"
    onnx_file = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fix_encoder/fix_encoder.sim.onnx"

    # model_name = "fix_decoder"
    model_name = "fix_encoder"
    hm_onnx_file = "work_dirs/MIT" + model_name

    # encoder_attention_mask = torch.ones(input_shapes["encoder_attention_mask"]).to(device)
    # input_ids = torch.randint(0, 128, input_shapes["input_ids"]).to(device)
    # encoder_hidden_states = torch.randn(input_shapes["encoder_hidden_states"]).to(device)

    input_ids = torch.randint(0, 128, input_shapes["input_ids"]).to(device)
    attention_mask = torch.ones(input_shapes["attention_mask"]).to(device)


    onnx_model = onnx.load(onnx_file)
    convert_onnx_to_hmonnx(onnx_model, [input_ids, attention_mask], DeviceType.XH2a, hm_onnx_file)
    # convert_onnx_to_hmonnx(onnx_model, [encoder_attention_mask, input_ids, encoder_hidden_states], DeviceType.XH2a, hm_onnx_file)

    if save_golden:
        session = HMONNXGoldenInference(hm_onnx_file)
        session.to(device)
        session.save_golden = save_golden
        session.golden_dir = work_dir + f"/hmonnx/golden_{model_name}"
        session.step = 0
    else:
        session = HMONNXInference(hm_onnx_file)
        session.to(device)
    # out = session(encoder_attention_mask, input_ids, encoder_hidden_states)
    out = session(input_ids, attention_mask)

if __name__ == "__main__":
    export_hmonnx_model("work_dirs", True)
    # _test_resize_trace(True, "work_dir", 11, DeviceType.XH2a, True)
