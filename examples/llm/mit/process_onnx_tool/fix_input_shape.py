import onnx
from onnxsim import simplify

# 加载模型
model = onnx.load("/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/model.fp_all.onnx")

# 固定输入形状（字典格式：输入名 -> 形状列表）
input_shapes = {
    "x": [1, 256, 560],
    "language": [1],
    "text_norm": [1],
    "x_length": [1],
}

# 简化+固定输入
model_simp, check = simplify(model, input_shapes=input_shapes)
assert check, "简化后模型验证失败"

# 保存
onnx.save(model_simp, "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fp/model_all.sim.onnx")
print("模型简化+固定输入完成！")