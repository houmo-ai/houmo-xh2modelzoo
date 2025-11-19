import torch
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
input_tensor = torch.randn(1, 3, 256, 256)
onnx_file_path = "maskrcnn.onnx"

# Export the PyTorch model to ONNX format
torch.onnx.export(
    model.cpu(),
    input_tensor.cpu(),
    onnx_file_path,
    export_params=True,
    do_constant_folding=False,
    input_names=["input"],
    output_names=["boxes", "labels", "scores", "masks"],
)
