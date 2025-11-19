import argparse
import re
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(
    old_size: tuple,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(f"Scale must be a number or tuple of int, but got {type(scale)}")

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def preprocess_image(image, input_img_size) -> Dict[str, Any]:
    # 图像预处理 (Letterbox)
    h, w = image.shape[:2]
    # target_w, target_h = input_img_size
    new_size, scale_factor, pad_w, pad_h = letterbox(image, input_img_size)

    resized_img = cv2.resize(image, (new_size[0], new_size[1]), interpolation=cv2.INTER_LINEAR)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # BGR to RGB

    # Add padding
    pad_img = np.full((input_img_size[1], input_img_size[0], 3), 114, dtype=np.uint8)
    pad_img[pad_h : (pad_h + new_size[1]), pad_w : (pad_w + new_size[0])] = resized_img

    # HWC to CHW, BGR to RGB, to float
    pad_img = pad_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    pad_img = np.ascontiguousarray(pad_img)
    pad_img = torch.from_numpy(pad_img)

    pad_img = pad_img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    pad_img = pad_img.unsqueeze(0).to(torch.float16)

    return {
        "metas": [{"ori_shape": image.shape[:2], "scale_factor": scale_factor, "pad_wh": (pad_w, pad_h)}],  # h, w
        "inputs": pad_img,
    }


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    return new_unpad, r, left, top


def postprocess_yolov6(output: torch.Tensor, batch_img_metas: Dict[str, Any]):
    """
    Post-processes the output of a YOLOv6 model to generate bounding box detections.
    """
    confidence_thres = 0.1
    iou_thres = 0.45

    img_meta = batch_img_metas[0]
    ori_h, ori_w = img_meta["ori_shape"]
    scale_factor = img_meta["scale_factor"]
    pad_w, pad_h = img_meta["pad_wh"]

    # The output of YOLOv6 ONNX is [1, 8400, 85]
    # 85 = 4 (cx, cy, w, h) + 1 (obj_conf) + 80 (cls_conf)
    outputs = np.squeeze(output[0])  # Shape: (8400, 85)

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Iterate over each prediction
    for i in range(outputs.shape[0]):
        prediction = outputs[i]
        obj_conf = prediction[4]

        # Filter out predictions with low object confidence
        if obj_conf < confidence_thres:
            continue

        class_scores = prediction[5:]
        class_id = np.argmax(class_scores)
        max_class_score = class_scores[class_id]

        # Calculate the final score
        final_score = obj_conf * max_class_score

        # Filter out predictions with low final score
        if final_score < confidence_thres:
            continue

        # Extract the bounding box coordinates
        cx, cy, w, h = prediction[0], prediction[1], prediction[2], prediction[3]

        # Convert from cx,cy,w,h to x1,y1,x2,y2
        x1 = int((cx - w / 2 - pad_w) / scale_factor)
        y1 = int((cy - h / 2 - pad_h) / scale_factor)
        x2 = int((cx + w / 2 - pad_w) / scale_factor)
        y2 = int((cy + h / 2 - pad_h) / scale_factor)

        boxes.append([x1, y1, x2 - x1, y2 - y1])  # NMSBoxes wants x,y,w,h
        scores.append(final_score)
        class_ids.append(class_id)

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    det_boxes = [{"boxes": [], "scores": [], "labels": []}]
    if len(indices) > 0:
        for i in indices.flatten():
            # Get the box, score, and class ID corresponding to the index
            x, y, w, h = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Append to result structure
            det_boxes[0]["boxes"].append([x, y, x + w, y + h])
            det_boxes[0]["scores"].append(score)
            det_boxes[0]["labels"].append(class_id)

    # Convert lists to tensors
    det_boxes[0]["boxes"] = torch.tensor(det_boxes[0]["boxes"])
    det_boxes[0]["scores"] = torch.tensor(det_boxes[0]["scores"])
    det_boxes[0]["labels"] = torch.tensor(det_boxes[0]["labels"])
    return det_boxes


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


def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("session is created successfully")

    input_shape = (640, 640)
    # 读取图像并进行预处理
    image_file = args.image
    image = cv2.imread(image_file)
    results = preprocess_image(image, input_shape)
    inputs = results["inputs"]
    batch_img_metas = results["metas"]

    input_name = session.get_input_names()[0]
    inputs = inputs.to(exec_device)  # Dtype was already set to float16 in preprocess

    batch_nn_out = session.run(
        {
            input_name: inputs,
        }
    )
    batch_nn_out = batch_nn_out.cpu().numpy()

    # 使用为YOLOv6适配的后处理函数
    det_boxes = postprocess_yolov6(batch_nn_out, batch_img_metas)

    det_boxes = det_boxes[0]  # batch 0

    # 修改输出目录
    out_dir = Path("work_dirs/yolov6m")
    out_dir.mkdir(exist_ok=True, parents=True)  # 确保目录存在
    image_fname = Path(image_file).stem
    image_to_draw = cv2.imread(image_file)

    # 加载类别名称
    coco_cfg_file = str(Path(__file__).parent / "coco8.yaml")
    classes = yaml_load(coco_cfg_file)["names"]

    for box, score, label in zip(det_boxes["boxes"], det_boxes["scores"], det_boxes["labels"]):
        if score < 0.1:  # Use the same confidence threshold
            continue
        box = box.int().tolist()
        x1, y1, x2, y2 = box
        class_id = int(label)
        cv2.rectangle(image_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create the label text with class name and score
        label_text = f"{classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw the label text on the image
        cv2.putText(
            image_to_draw, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    out_file = out_dir / f"{image_fname}_det_vis.jpg"
    cv2.imwrite(str(out_file), image_to_draw)
    logger.info(f"Detect result is saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改默认的hmonnx文件路径
    parser.add_argument(
        "--hmonnx",
        type=str,
        default="/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/yolov6m/hmonnx/yolov6m_w8a8_sefp_XH2a.onnx",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    args = parser.parse_args()
    main(args)
