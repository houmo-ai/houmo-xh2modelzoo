
from xhquant.api import convert_fx_model_to_hmonnx, convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType

import torch
import os.path as osp

output_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx"

#llm_decoder
model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/llm_decoder.onnx"

input = torch.randn(1, 896)
quant_type = "w8a16h1_sefp"
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
convert_onnx_to_hmonnx(model_path, (input,), out_hmonnx_file=osp.join(output_path,"llm_decoder.onnx"), device_type="XH2A", quant_config=quant_config)

#spk_embed_affine_layer
model_path_spk = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/spk_embed_affine_layer.onnx"

input = torch.randn(1, 192)
quant_type = "w8a16h1_sefp"
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
convert_onnx_to_hmonnx(model_path_spk, (input,), out_hmonnx_file=osp.join(output_path,"spk.onnx"), device_type="XH2A", quant_config=quant_config)

#pre_lookahead_layer
model_path_pre = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/pre_lookahead_layer.onnx"
input = torch.randn(1, 1024, 80)
quant_type = "w8a16h1_sefp"
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
convert_onnx_to_hmonnx(model_path_pre, (input,), out_hmonnx_file=osp.join(output_path,"pre_lookahead_layer.onnx"), device_type="XH2A", quant_config=quant_config)