import onnx
import numpy as np
from onnx import numpy_helper

IN = "weights/huachuang/encoder.onnx"
OUT = "weights/huachuang/encoder_fix_len.onnx"

model = onnx.load(IN)
g = model.graph

name = "speech_lengths"

# 1) 从输入里删掉 speech_lengths（避免“同名 input + initializer”冲突）
kept_inputs = []
for inp in g.input:
    if inp.name != name:
        kept_inputs.append(inp)
del g.input[:]
g.input.extend(kept_inputs)

# 2) 删掉旧 initializer（如果有）
kept_inits = []
for init in g.initializer:
    if init.name != name:
        kept_inits.append(init)
del g.initializer[:]
g.initializer.extend(kept_inits)

# 3) 加入常量 initializer：speech_lengths = [334]
arr = np.array([334], dtype=np.int32)
g.initializer.append(numpy_helper.from_array(arr, name=name))

onnx.save(model, OUT)
print("saved:", OUT)