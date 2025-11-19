from pathlib import Path

import onnx
import onnxruntime as ort
import onnxsim
import requests
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from sympy import loggamma
from transformers import AutoImageProcessor, DetrForObjectDetection
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

"""
pip install requests
pip install transformers
pip install timm
pip install onnxruntime-gpu 
pip install onnxsim
pip install loguru

"""


def main(args):
    out_dir = args.out_dir
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    class DetrModel(nn.Module):
        def __init__(self, model: DetrForObjectDetection):
            super().__init__()
            self.model = model

        def forward(self, pixel_values, pixel_mask):
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            return outputs.logits, outputs.pred_boxes

    inputs = image_processor(images=image, return_tensors="pt")
    detr_model = DetrModel(model)
    outputs = detr_model(
        inputs["pixel_values"],
        inputs["pixel_mask"],
    )

    outputs = DetrObjectDetectionOutput(logits=outputs[0], pred_boxes=outputs[1])
    # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    inputs = (inputs["pixel_values"], inputs["pixel_mask"])
    detr_model.eval()
    onnx_file = str(Path(out_dir) / "detr.onnx")
    Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        detr_model,
        inputs,
        onnx_file,
        opset_version=18,
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
    )
    onnx_model, check = onnxsim.simplify(onnx_file)
    onnx.save(
        onnx_model,
        onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{Path(onnx_file).stem}_external_data",
    )
    session = ort.InferenceSession(onnx_file)
    outputs = session.run(
        None,
        {
            "pixel_values": inputs[0].cpu().numpy(),
            "pixel_mask": inputs[1].cpu().numpy(),
        },
    )
    outputs = DetrObjectDetectionOutput(logits=torch.from_numpy(outputs[0]), pred_boxes=torch.from_numpy(outputs[1]))
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    logger.info(f"Save to {onnx_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/models/detr")
    args = parser.parse_args()
    main(args)
