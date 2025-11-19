from pathlib import Path

import onnx
import onnxsim
import torch
import torchvision

if __name__ == "__main__":
    model = torchvision.models.vit_b_32()
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    out_onnx_file = "data/model_zoo2/houmo/vit/vit_small_patch16_224.onnx"
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)
    torch.onnx.export(model, input, out_onnx_file, input_names=["images"], output_names=["cls_score"])
    onnx_model = onnx.load(out_onnx_file)
    onnx_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(onnx_model, out_onnx_file)
    print("Export onnx success, out onnx file to: ", out_onnx_file)
