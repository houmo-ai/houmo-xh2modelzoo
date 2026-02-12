import torch
import os.path as osp

model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_3.onnx"
output_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx"

from xhquant.api import convert_fx_model_to_hmonnx, convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType
input = torch.randn(1, 128, 3000)
mask = torch.randn(1, 20, 750, 750)
mask1 = torch.randn(1, 750, 1280)
quant_type = "w8a8_sefp"
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
convert_onnx_to_hmonnx(model_path, (input, mask, mask1), out_hmonnx_file=osp.join(output_path,"speech_tokenizer_v3_3000.onnx"), device_type="XH2A", quant_config=quant_config)