from pathlib import Path

import onnx
import onnxsim
import torch
import torch.hub
import torchvision
import torchvision.models.squeezenet

if __name__ == "__main__":
    model = torch.hub.load("moskomule/senet.pytorch", "se_resnet50", num_classes=1000)
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    out_onnx_file = "data/models/senet/se_resnet50_224x224.onnx"
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(model, input, out_onnx_file, input_names=["images"], output_names=["cls_score"])
    onnx_model = onnx.load(out_onnx_file)
    onnx_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(onnx_model, out_onnx_file)
    print("Export onnx success, out onnx file to: ", out_onnx_file)
