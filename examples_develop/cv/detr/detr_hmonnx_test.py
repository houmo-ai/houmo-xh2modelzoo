# detr_hmonnx_test.py
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init

# DETR的颜色和类别，COCO 80个类别
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def preprocess_image_detr(image_bgr, input_size: int) -> dict:
    """为DETR模型预处理图像"""
    # 1. 直接Resize到目标尺寸
    resized_img = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # 2. BGR to RGB, HWC to CHW
    img_rgb = resized_img[:, :, ::-1]
    img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))

    # 3. To Tensor并归一化到[0, 1]
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float() / 255.0

    # 4. 标准化 (mean/std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized_tensor = (img_tensor - mean) / std

    return {
        "metas": [{"ori_shape": image_bgr.shape[:2]}],
        "inputs": normalized_tensor.to(torch.float16),  # HMONNX 通常使用 float16 输入
    }


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def postprocess_detr(out_logits, out_bbox, ori_shape, conf_thres=0.7):
    """DETR后处理函数"""
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)  # 忽略最后一个"no object"类别

    # 将中心点+宽高格式的box转换为左上角+右下角格式
    boxes = box_cxcywh_to_xyxy(out_bbox)

    # 将归一化的坐标[0, 1]缩放到原始图像尺寸
    img_h, img_w = ori_shape
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
    boxes = boxes * scale_fct

    # 过滤掉低置信度的结果
    keep = scores > conf_thres
    results = {
        "scores": scores[keep],
        "labels": labels[keep],
        "boxes": boxes[keep],
    }
    return results


def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("Session created successfully")

    # --- 主要修改: 从data YAML文件加载配置和类别 ---
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    # 使用与模型训练时一致的完整类别列表 (91个类别)
    model_classes = data_cfg["model_classes"]
    logger.info(f"Loaded {len(model_classes)} classes from {args.data}")

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        logger.error(f"Failed to read image: {args.image}")
        return

    # Preprocess
    proc_results = preprocess_image_detr(image_bgr, args.imgsz)
    inputs = proc_results["inputs"].to(exec_device)
    img_meta = proc_results["metas"][0]

    # Inference (DETR有两个输出)
    input_name = session.get_input_names()[0]
    out_logits, out_bbox = session.run({input_name: inputs})
    out_logits = out_logits.to(exec_device)
    out_bbox = out_bbox.to(exec_device)

    # Postprocess
    results = postprocess_detr(out_logits[0], out_bbox[0], img_meta["ori_shape"], conf_thres=args.conf_thres)

    # Visualization
    out_dir = Path(f"work_dirs/{Path(args.hmonnx).parent.parent.stem}")
    out_dir.mkdir(exist_ok=True, parents=True)
    image_fname = Path(args.image).stem

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box]
        x1, y1, x2, y2 = box
        class_id = label.item()

        if class_id < len(model_classes):
            class_name = model_classes[class_id]
        else:
            class_name = f"Unknown: {class_id}"

        color = COLORS[class_id % len(COLORS)]
        color = [c * 255 for c in color]

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{class_name}: {score:.2f}"
        cv2.putText(image_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_file = out_dir / f"{image_fname}_detr_vis.jpg"
    cv2.imwrite(str(out_file), image_bgr)
    logger.info(f"Detection result saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmonnx", type=str, required=True, help="Path to the HMONNX model file.")
    parser.add_argument(
        "--data",
        type=str,
        default="examples/cv/detr/coco_detr_eval.yaml",
        help="Path to the dataset config file (e.g., coco_detr_eval.yaml)",
    )
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg", help="Path to the input image.")
    parser.add_argument("--imgsz", type=int, default=1312, help="Input image size used during export.")
    parser.add_argument("--conf-thres", type=float, default=0.7, help="Confidence threshold for visualization.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()
    main(args)
