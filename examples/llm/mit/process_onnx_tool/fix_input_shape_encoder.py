import onnx
from onnxsim import simplify

# 加载模型
model = onnx.load("/data02/datasets/MIT/demo_model/MT/encoder_model.onnx")

# 固定输入形状（字典格式：输入名 -> 形状列表）
# input_shapes = {
#     "encoder_attention_mask": [1, 256],
#     "input_ids": [1, 256],
#     "encoder_hidden_states": [1, 256,1024]
# }

input_shapes = {
    "input_ids": [1, 256],
    "attention_mask": [1, 256]
}

# 简化+固定输入
model_simp, check = simplify(model, input_shapes=input_shapes)
assert check, "简化后模型验证失败"

# 保存
onnx.save(model_simp, "/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/mit/fix_encoder/fix_encoder.sim.onnx")
print("模型简化+固定输入完成！")