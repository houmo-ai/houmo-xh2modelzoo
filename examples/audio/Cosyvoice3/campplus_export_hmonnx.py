
import onnx
import torch
import onnxsim
import onnxruntime as ort
import numpy as np
import os.path as osp

model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/campplus.onnx"
model_path_simplify = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/campplus_simplify.onnx"
output_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx"

#fix shape
fixed_d = {
   'batch_size': 1,
   'sequence_length' : 1000,
}

# 加载模型
model = onnx.load(model_path)
providers = ['CPUExecutionProvider']  # 替换为 ['CUDAExecutionProvider'] 启用 GPU
session = ort.InferenceSession(model_path, providers=providers)
print("模型输入节点:")
for input_node in session.get_inputs():
   print(f"  {input_node.name}: {input_node.type}, 形状: {input_node.shape}")
print("模型输出节点:")
for output_node in session.get_outputs():
   print(f"  {output_node.name}: {output_node.type}, 形状: {output_node.shape}")

# 修改输入节点的shape（将符号维度替换为固定值）
for input_node in model.graph.input:
   dims = [d.dim_value if d.dim_value != 0 else d.dim_param for d in input_node.type.tensor_type.shape.dim]
   # 根据fixed_dims替换符号维度
   new_dims = []
   for dim in dims:
      if dim in fixed_d:
            new_dims.append(fixed_d[dim])  # 替换为固定值
      else:
            # 如果是数字维度（如80），保持不变
            new_dims.append(dim if isinstance(dim, int) else 0)  # 0表示动态，这里应避免
   
   # 更新输入节点的shape
   input_node.type.tensor_type.shape.dim.clear()
   for dim_val in new_dims:
      input_node.type.tensor_type.shape.dim.add(dim_value=dim_val)

slimmed_model, _ = onnxsim.simplify(model)
onnx.save(slimmed_model, model_path_simplify)

#convert hmonnx
from xhquant.api import convert_fx_model_to_hmonnx, convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType
input = torch.randn(1, 1000, 80)
quant_type = "w8a8h1_sefp"
quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
quant_config = create_quant_config(quant_scheme)
convert_onnx_to_hmonnx(model_path_simplify, (input,), out_hmonnx_file=osp.join(output_path,"campplus_1000.onnx"), device_type="XH2A", quant_config=quant_config)