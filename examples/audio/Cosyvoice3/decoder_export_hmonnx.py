import os
import os.path as osp
import onnx
import torch
import onnxsim
import onnxruntime as ort
import numpy as np

from xhquant.api import (
   convert_onnx_to_hmonnx,
   QuantScheme,
   create_quant_config,
   DeviceType,
   HMONNXGoldenInference,
)

model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/flow_decoder_estimator_fp32.onnx"
model_path_simplify = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/decoder_simplify.onnx"
OUTPUT_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/decoder/prefill"
GOLDEN_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/decoder/prefill/step_0"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)

def get_dummy_input(batch_size, seq_len, out_channels):
   device = "cpu"
   x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
   mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
   mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
   t = torch.rand((batch_size), dtype=torch.float32, device=device)
   spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
   cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
   return x, mask, mu, t, spks, cond

# ======================
# 固定 shape
# ======================
FIXED_DIMS = {
   "seq_len": 2048,
}

def inspect_onnx(model_path):
   session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
   print("模型输入节点:")
   for inp in session.get_inputs():
      print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
   print("模型输出节点:")
   for out in session.get_outputs():
      print(f"  {out.name}: {out.type}, shape={out.shape}")

def fix_input_shape(model, fixed_dims):
   for input_node in model.graph.input:
      dims = []
      for d in input_node.type.tensor_type.shape.dim:
         if d.dim_value > 0:
               dims.append(d.dim_value)
         else:
               dims.append(d.dim_param)

      new_dims = []
      for dim in dims:
         if dim in fixed_dims:
               new_dims.append(fixed_dims[dim])
         elif isinstance(dim, int):
               new_dims.append(dim)
         else:
               raise ValueError(f"未处理的动态维度: {dim}")

      input_node.type.tensor_type.shape.dim.clear()
      for val in new_dims:
         input_node.type.tensor_type.shape.dim.add(dim_value=val)

   return model

def main():
   inspect_onnx(model_path)

   # fix shape
   model = onnx.load(model_path)
   model = fix_input_shape(model, FIXED_DIMS)

   # simplify（加存在判断）
   if not osp.exists(model_path_simplify):
      model_simplified, _ = onnxsim.simplify(model)
      onnx.save(model_simplified, model_path_simplify)

   # dummy input
   x, mask, mu, t, spks, cond = get_dummy_input(2, 2048, 80)

   # quant config
   quant_type = "w8a16_sefp"
   quant_scheme = QuantScheme(
      target_device=DeviceType.XH2a,
      quant_type=quant_type
   )
   quant_config = create_quant_config(quant_scheme)

   # convert（加存在判断）
   prefix = f"hmquant_xh2_decoder_w8a16_2048_20260320"
   output_file = osp.join(OUTPUT_DIR, f"{prefix}.onnx")

   if not osp.exists(output_file):
      convert_onnx_to_hmonnx(
         model_path_simplify,
         (x, mask, mu, t, spks, cond),
         out_hmonnx_file=output_file,
         device_type="XH2A",
         quant_config=quant_config
      )

   # golden
   model = HMONNXGoldenInference(output_file)
   model.save_golden = True
   model.exec_device = torch.device("cuda:0")

   x = x.to(torch.float16)
   mask = mask.to(torch.float16)
   mu = mu.to(torch.float16)
   t = t.to(torch.float16)
   spks = spks.to(torch.float16)
   cond = cond.to(torch.float16)

   input_args = (x, mask, mu, t, spks, cond)
   model.golden_dir = str(GOLDEN_DIR)

   with torch.no_grad():
      model.forward(*input_args)

if __name__ == "__main__":
   main()