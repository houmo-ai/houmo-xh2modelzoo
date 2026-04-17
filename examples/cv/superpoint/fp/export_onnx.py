import torch
import torch.nn as nn
import onnx
import onnxsim
from fp_superpoint import SuperPoint


class SuperPointONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model({"image": image})


weights_path = "superpoint_v6_from_tf.pth"
output_path = "superpoint_480x640.onnx"

device = torch.device('cpu')
model = SuperPoint()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

onnx_model = SuperPointONNX(model)
dummy_input = torch.randn(1, 1, 480, 640)

torch.onnx.export(
    onnx_model,
    dummy_input,
    output_path,
    input_names=["image"],
    output_names=["scores", "descriptors_dense"],
    dynamic_axes=None,
    opset_version=16,
)

print(f"Exported SuperPoint ONNX model to {output_path}")
print(f"Input shape: {dummy_input.shape}")

onnx_model = onnx.load(output_path)
model_simplified, check = onnxsim.simplify(onnx_model)
assert check, "Simplified model verification failed"
onnx.save(model_simplified, output_path)

print(f"Simplified ONNX model saved to {output_path}")