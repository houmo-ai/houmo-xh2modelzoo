# yolov5_eval.py (最终、API调用正确版)
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# --- 辅助函数 (保持不变) ---
def _scale_size(size, scale):
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


# ... (其他辅助函数 xywh2xyxy, postprocess_v5 等都复制过来) ...
def rescale_size(old_size, scale, return_scale=False):
    w, h = old_size
    if isinstance(scale, (float, int)):
        scale_factor = scale
    else:
        max_long_edge, max_short_edge = max(scale), min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    new_size = _scale_size((w, h), scale_factor)
    return (new_size, scale_factor) if return_scale else new_size


def preprocess_image(image, input_img_size):
    h, w = image.shape[:2]
    new_size, scale_factor = rescale_size((w, h), input_img_size, return_scale=True)
    resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    target_w, target_h = input_img_size
    pad_w, pad_h = target_w - new_size[0], target_h - new_size[1]
    padded_img = cv2.copyMakeBorder(resized_img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    padded_img = padded_img[:, :, ::-1].transpose(2, 0, 1)
    padded_img = np.ascontiguousarray(padded_img)
    img_tensor = torch.from_numpy(padded_img).unsqueeze(0).float() / 255.0
    return {"metas": [{"ori_shape": (h, w), "scale_factor": scale_factor}], "inputs": img_tensor}


def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def postprocess_v5(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]
        if not n:
            continue
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output


# --- 推理引擎包装类 (最终修正版) ---
class InferenceEngine:
    def __init__(self, model_path, model_type="onnx", batch_size=1, imgsz=640):
        self.model_type = model_type
        self.batch_size = batch_size
        s = [imgsz // stride for stride in [8, 16, 32]]
        self.num_predictions = sum([nx * ny * 3 for nx, ny in zip(s, s)])
        self.expected_last_dim = 85

        if self.model_type == "hmonnx":
            from xhquant.api import HMONNXInference, xhquant_init

            xhquant_init()
            self.session = HMONNXInference(model_path)
            self.session.to(torch.device("cuda"))
            self.input_names = self.session.get_input_names()
        else:
            self.session = onnxruntime.InferenceSession(
                model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_names = [inp.name for inp in self.session.get_inputs()]

    def __call__(self, *inputs_tensors) -> list[torch.Tensor]:
        if self.model_type == "hmonnx":
            # HMONNXInference 直接接收张量作为位置参数
            outputs = self.session(*[t.to(torch.device("cuda")) for t in inputs_tensors])

            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            # 对 YOLOv5 的第一个输出进行强制重塑
            output_tensor = outputs[0]
            if output_tensor.ndim == 2:
                output_tensor = output_tensor.view(self.batch_size, self.num_predictions, self.expected_last_dim)
            return [output_tensor] + list(outputs[1:])
        else:
            # ONNX Runtime 需要字典输入
            input_feed = {name: tensor.cpu().numpy() for name, tensor in zip(self.input_names, inputs_tensors)}
            outputs_np = self.session.run(None, input_feed)
            return [torch.from_numpy(o).to("cuda") for o in outputs_np]


# --- 主函数 ---
def main(args):
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    base_path = Path(data_cfg["path"])
    img_dir = base_path / data_cfg["val"]
    annotation_file = base_path / data_cfg["annotations"]
    engine = InferenceEngine(args.model, args.model_type, batch_size=args.batch_size, imgsz=args.imgsz)
    image_files = sorted(list(img_dir.glob("*.jpg")))
    coco_gt = COCO(str(annotation_file))
    try:
        class_names = data_cfg["names"]
        coco_cat_ids = coco_gt.getCatIds(catNms=class_names)
        model_cls_to_coco_cat = {i: coco_id for i, coco_id in enumerate(coco_cat_ids)}
    except:
        cat_ids = coco_gt.getCatIds()
        model_cls_to_coco_cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    results = []
    print(f"Evaluating model: {args.model} with batch size {args.batch_size}")

    for i in tqdm(range(0, len(image_files), args.batch_size), desc="Processing batches"):
        batch_paths = image_files[i : i + args.batch_size]
        if len(batch_paths) != args.batch_size:
            print(f"\nSkipping the last incomplete batch of size {len(batch_paths)}.")
            continue

        batch_inputs, batch_metas = [], []
        for img_path in batch_paths:
            proc_results = preprocess_image(cv2.imread(str(img_path)), (args.imgsz, args.imgsz))
            batch_inputs.append(proc_results["inputs"])
            meta = proc_results["metas"][0]
            meta["image_id"] = int(img_path.stem)
            batch_metas.append(meta)

        inputs = torch.cat(batch_inputs, dim=0)
        if args.model_type == "hmonnx":
            input_tensor = inputs.to(torch.float16)
        else:
            input_tensor = inputs.to(torch.float32)

        # 对于 YOLOv5，只有一个输入
        outputs_list = engine(input_tensor)
        batch_nn_out = outputs_list[0]

        det_results = postprocess_v5(batch_nn_out.float(), conf_thres=0.001, iou_thres=0.65)

        for j, det in enumerate(det_results):
            img_meta = batch_metas[j]
            if det is None or len(det) == 0:
                continue
            coords = det[:, :4]
            coords /= img_meta["scale_factor"]
            coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img_meta["ori_shape"][1])
            coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img_meta["ori_shape"][0])
            for *xyxy, conf, cls in det:
                bbox = [xyxy[0].item(), xyxy[1].item(), (xyxy[2] - xyxy[0]).item(), (xyxy[3] - xyxy[1]).item()]
                results.append(
                    {
                        "image_id": img_meta["image_id"],
                        "category_id": model_cls_to_coco_cat[int(cls)],
                        "bbox": bbox,
                        "score": conf.item(),
                    }
                )

    result_json_path = Path(f"detections_{Path(args.model).stem}_b{args.batch_size}.json")
    with open(result_json_path, "w") as f:
        json.dump(results, f)
    print(f"Detection results saved to {result_json_path}\nRunning COCO-style evaluation...")
    coco_dt = coco_gt.loadRes(str(result_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 ONNX/HMONNX models on COCO dataset.")
    SCRIPT_DIR = Path(__file__).parent.resolve()
    DEFAULT_DATA_CONFIG = SCRIPT_DIR / "coco_yolov5_eval.yaml"
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--model-type", type=str, default="onnx", choices=["onnx", "hmonnx"])
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_CONFIG, help="Path to the dataset config file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    args = parser.parse_args()
    main(args)
