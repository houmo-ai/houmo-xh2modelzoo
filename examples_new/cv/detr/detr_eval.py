# detr_eval.py
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def preprocess_image_detr(image_bgr, input_size: int):
    h, w = image_bgr.shape[:2]
    resized_img = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = resized_img[:, :, ::-1]
    img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized_tensor = (img_tensor - mean) / std

    return {"inputs": normalized_tensor, "metas": [{"ori_shape": (h, w)}]}


def postprocess_detr_batch(batch_logits, batch_boxes, batch_metas, conf_thres=0.001):
    batch_results = []
    for logits, boxes, meta in zip(batch_logits, batch_boxes, batch_metas):
        prob = F.softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        img_h, img_w = meta["ori_shape"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=boxes_xyxy.device)
        scaled_boxes = boxes_xyxy * scale_fct

        keep = scores > conf_thres

        batch_results.append(
            {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": scaled_boxes[keep],
            }
        )
    return batch_results


class InferenceEngine:
    def __init__(self, model_path, model_type="onnx"):
        self.model_type = model_type
        self.device = torch.device("cuda")

        if self.model_type == "hmonnx":
            from xhquant.api import HMONNXInference, xhquant_init

            xhquant_init()
            self.session = HMONNXInference(model_path)
            self.session.to(self.device)
            self.input_names = self.session.get_input_names()
        else:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]

    def __call__(self, inputs_tensor) -> list[torch.Tensor]:
        input_feed = {}
        if self.model_type == "hmonnx":
            input_feed[self.input_names[0]] = inputs_tensor.to(self.device)
            outputs = self.session.run(input_feed)
            return [o.to(self.device) for o in outputs]
        else:
            input_feed[self.input_names[0]] = inputs_tensor.cpu().numpy()
            outputs_np = self.session.run(None, input_feed)
            return [torch.from_numpy(o).to(self.device) for o in outputs_np]


# --- 主函数 ---
def main(args):
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    model_classes = data_cfg["model_classes"]
    imgsz = data_cfg.get("imgsz", 1312)  # 从yaml获取imgsz

    base_path = Path(data_cfg["path"])
    img_dir = base_path / data_cfg["val"]
    annotation_file = base_path / data_cfg["annotations"]

    engine = InferenceEngine(args.model, args.model_type)
    image_files = sorted(list(img_dir.glob("*.jpg")))

    if args.limit > 0:
        print(f"--- Limiting evaluation to the first {args.limit} images. ---")
        image_files = image_files[: args.limit]

    coco_gt = COCO(str(annotation_file))

    cats = coco_gt.loadCats(coco_gt.getCatIds())
    coco_name_to_id = {cat["name"]: cat["id"] for cat in cats}

    results = []
    print(f"Evaluating model: {args.model} on {len(image_files)} images with batch size {args.batch_size}")

    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Processing batches"):
        batch_paths = image_files[i : i + args.batch_size]
        if len(batch_paths) != args.batch_size:
            print(f"\nSkipping last incomplete batch of size {len(batch_paths)}.")
            continue

        batch_inputs, batch_metas = [], []
        for img_path in batch_paths:
            proc_results = preprocess_image_detr(cv2.imread(str(img_path)), imgsz)
            batch_inputs.append(proc_results["inputs"])
            meta = proc_results["metas"][0]
            meta["image_id"] = int(img_path.stem)
            batch_metas.append(meta)

        inputs = torch.cat(batch_inputs, dim=0)
        input_tensor = inputs.to(torch.float16 if args.model_type == "hmonnx" else torch.float32)

        batch_logits, batch_boxes = engine(input_tensor)
        det_results_batch = postprocess_detr_batch(batch_logits, batch_boxes, batch_metas)

        for j, det_results in enumerate(det_results_batch):
            img_meta = batch_metas[j]
            for score, label, box in zip(det_results["scores"], det_results["labels"], det_results["boxes"]):
                model_pred_label = label.item()

                class_name = model_classes[model_pred_label]

                if class_name in coco_name_to_id:
                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]

                    results.append(
                        {
                            "image_id": img_meta["image_id"],
                            "category_id": coco_name_to_id[class_name],
                            "bbox": bbox,
                            "score": score.item(),
                        }
                    )

    if not results:
        print("\nNo valid detections were generated. Skipping COCO evaluation.")
        return

    result_json_path = Path(f"work_dirs/detections_{Path(args.model).stem}_b{args.batch_size}_limit{args.limit}.json")
    with open(result_json_path, "w") as f:
        json.dump(results, f)
    print(f"\nDetection results saved to {result_json_path}\nRunning COCO-style evaluation...")

    coco_dt = coco_gt.loadRes(str(result_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DETR ONNX/HMONNX models on COCO dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--model-type", type=str, default="onnx", choices=["onnx", "hmonnx"])
    parser.add_argument(
        "--data",
        type=str,
        default="examples/cv/detr/coco_detr_eval.yaml",
        help="Path to the dataset config file (e.g., coco_detr_eval.yaml)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit evaluation to the first N images. 0 means use all images."
    )
    args = parser.parse_args()
    main(args)
