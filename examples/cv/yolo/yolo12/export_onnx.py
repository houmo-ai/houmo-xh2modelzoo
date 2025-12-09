import re
from pathlib import Path

import cv2
import onnx
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO

# from xhquant.utils.config import yaml_load


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict
        if append_filename:
            data["yaml_file"] = str(file)
        return data


# Load a COCO-pretrained YOLO12n model
model = YOLO("data/model_zoo2/houmo/yolo12m/yolo12m.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

coco_cfg_file = str(Path(__file__).parent / "coco8.yaml")
classes = yaml_load(coco_cfg_file)["names"]

mode = "onnx"
results = model("data/images/ILSVRC2012_val_00002031.JPEG")
if mode == "torch":
    # Run inference with the YOLO12n model on the 'bus.jpg' image
    results = model("data/images/ILSVRC2012_val_00002031.JPEG")

    # im_dummy = torch.randn(1, 3, 640, 640, requires_grad=True)
else:

    image_to_draw = cv2.imread("data/images/ILSVRC2012_val_00002031.JPEG")
    image_to_draw_re = cv2.resize(image_to_draw, (640, 640))  # Resize to match model input size
    img_inp = model.predictor.preprocess([image_to_draw_re])
    sess_ori = ort.InferenceSession(
        "data/model_zoo2/houmo/yolo12m/yolo12m.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_feed = {}
    input_names = [inp.name for inp in sess_ori.get_inputs()]
    input_feed[input_names[0]] = img_inp.cpu().numpy()  # Convert to numpy array for ONNX runtime
    results = sess_ori.run(None, input_feed)
    results = model.predictor.postprocess(torch.from_numpy(results[0]), img_inp, [image_to_draw_re])

det_boxes = results[0].boxes  # batch 0
for box, score, label in zip(det_boxes.xyxy, det_boxes.conf, det_boxes.cls):
    if score < 0.1:  # Use the same confidence threshold
        continue
    box = box.int().tolist()
    x1, y1, x2, y2 = box
    class_id = int(label)
    cv2.rectangle(image_to_draw_re, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create the label text with class name and score
    label_text = f"{classes[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw the label text on the image
    cv2.putText(
        image_to_draw_re, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )

out_file = f"examples/cv/yolo/yolo12/ILSVRC2012_val_00002031_det_vis.jpg"
cv2.imwrite(str(out_file), image_to_draw_re)
# logger.info(f"Detect result is saved to {out_file}")

path = model.export(format="onnx")

# torch.onnx.export(
#     model,
#     im_dummy,
#     tmp_onnx_file,
#     input_names=["images"],
#     output_names=["logits", "boxes"],
#     opset_version=16,
#     verbose=False,
#     do_constant_folding=True,
# )
# onnx_model, _ = onnxsim.simplify(tmp_onnx_file)
# onnx.save(
#     onnx_model,
#     out_onnx_file,
#     save_as_external_data=True,
#     all_tensors_to_one_file=True,
#     location=f"{Path(out_onnx_file).stem}_external_data",
# )
