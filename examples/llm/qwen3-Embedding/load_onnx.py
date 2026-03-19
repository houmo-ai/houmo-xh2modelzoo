import glob
import os

import onnx

dir_path = "/data01/home/feilong.kong/xh2modelzoo/work_dirs/hmquant_xh2_qwen3_embedding_8b_w4a8h0_256_2k_20260309/prefill"

onnx_files = sorted(glob.glob(os.path.join(dir_path, "*_prefill_with_act.onnx")))
assert onnx_files, f"No .onnx file found in {dir_path}"
onnx_path = onnx_files[0]

print("Loading:", onnx_path)
model = onnx.load(onnx_path, load_external_data=True)
print("Load OK.")

print("Checking (path mode):", onnx_path)
onnx.checker.check_model(onnx_path)  # 大模型用路径检查
print("Check OK.")
