import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import paddle

# 验证CUDA是否可用
print("CUDA available:", paddle.device.is_compiled_with_cuda())
# 创建一个简单张量并移到GPU
x = paddle.to_tensor([1,2,3], place=paddle.CUDAPlace(0))
print("GPU tensor:", x)
# 执行简单运算
y = x * 2
print("GPU运算结果:", y)