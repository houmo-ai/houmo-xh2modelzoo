# -*- coding: utf-8 -*-
# Convert fixed-shape CAMPPlus ONNX to HMONNX.

import argparse
import os
import os.path as osp

import onnx
import onnxruntime as ort
import onnxsim
import torch

from xhquant.api import (
   convert_onnx_to_hmonnx,
   QuantScheme,
   create_quant_config,
   DeviceType,
   HMONNXGoldenInference,
)

HERE = osp.dirname(osp.abspath(__file__))
DEFAULT_FIXED_T = 1000
DEFAULT_QUANT_TYPE = "w8a8_sefp"


def default_model_path(fixed_t):
   return osp.join(HERE, "onnx", f"campplus_{fixed_t}.onnx")


def default_simplified_path(fixed_t):
   if fixed_t == DEFAULT_FIXED_T:
      return osp.join(HERE, "onnx", "campplus_simplify.onnx")
   return osp.join(HERE, "onnx", f"campplus_{fixed_t}_simplify.onnx")


def default_output_root():
   return osp.join(HERE, "campplus")


def default_output_file(output_root, fixed_t, quant_type):
   if fixed_t == DEFAULT_FIXED_T:
      output_dir = osp.join(output_root, "prefill")
   else:
      output_dir = osp.join(output_root, f"prefill_{fixed_t}")
   return osp.join(output_dir, f"hmquant_xh2_campplus_{quant_type}_{fixed_t}.onnx")


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


def main(argv=None):
   parser = argparse.ArgumentParser(description="Convert fixed-shape CAMPPlus ONNX to HMONNX")
   parser.add_argument("--fixed-t", type=int, default=DEFAULT_FIXED_T, help="Fixed fbank frame length")
   parser.add_argument("--model-path", default=None, help="Input fixed ONNX path")
   parser.add_argument("--simplified-path", default=None, help="Output simplified ONNX path")
   parser.add_argument("--output-root", default=default_output_root(), help="HMONNX output root directory")
   parser.add_argument("--output-file", default=None, help="Output HMONNX file path")
   parser.add_argument("--golden-dir", default=None, help="Golden output directory")
   parser.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE, help="Quant type")
   args = parser.parse_args(argv)
   if args.fixed_t <= 0:
      raise ValueError("--fixed-t must be positive")

   fixed_t = args.fixed_t
   model_path = args.model_path or default_model_path(fixed_t)
   simplified_path = args.simplified_path or default_simplified_path(fixed_t)
   output_file = args.output_file or default_output_file(args.output_root, fixed_t, args.quant_type)
   golden_dir = args.golden_dir or osp.join(osp.dirname(output_file), "step_0")

   if not osp.exists(model_path):
      raise FileNotFoundError(f"Missing fixed ONNX: {model_path}. Run campplus_export_onnx.py --fixed-t {fixed_t} first.")

   os.makedirs(osp.dirname(output_file), exist_ok=True)
   os.makedirs(golden_dir, exist_ok=True)

   inspect_onnx(model_path)

   model = onnx.load(model_path)
   model = fix_input_shape(model, {"batch": 1, "time": fixed_t, "batch_size": 1, "sequence_length": fixed_t})

   model_simplified, _ = onnxsim.simplify(model)
   onnx.save(model_simplified, simplified_path)

   dummy_input = torch.randn(1, fixed_t, 80)

   quant_scheme = QuantScheme(
      target_device=DeviceType.XH2a,
      quant_type=args.quant_type,
   )
   quant_config = create_quant_config(quant_scheme)

   if not osp.exists(output_file):
      convert_onnx_to_hmonnx(
         simplified_path,
         (dummy_input,),
         out_hmonnx_file=output_file,
         device_type="XH2A",
         quant_config=quant_config,
      )

   model = HMONNXGoldenInference(output_file)
   model.save_golden = True
   model.exec_device = torch.device("cuda:0")

   model.golden_dir = str(golden_dir)
   with torch.no_grad():
      model.forward(dummy_input.to(torch.float16))


if __name__ == "__main__":
   main()
