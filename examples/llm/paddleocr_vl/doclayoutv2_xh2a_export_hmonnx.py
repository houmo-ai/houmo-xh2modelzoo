"""
基于原始 PP-DocLayoutV2.onnx (paddle2onnx 导出) 内部存在动态 Expand/Range 节点，
本脚本通过 onnxsim + 常量替换的方式将其静态化，然后转 HMONNX 进行导出和推理。

  1. onnxsim 固定输入 shape (batch=1, 800x800)
  2. 用 dummy inference 求解残余动态张量的实际值
  3. 将动态张量替换为 initializer 常量
  4. 再做一次 onnxsim 清理
  5. 验证静态 ONNX 推理结果与官方一致
  6. convert_onnx_to_hmonnx + HMONNXGoldenInference
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnxsim import simplify

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_QUANT_CONFIG_PATH = "xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl/paddleocr_vl_doclayoutv2_config.py"

DEFAULT_INPUT_SHAPES = {
    "im_shape": [1, 2],
    "image": [1, 3, 800, 800],
    "scale_factor": [1, 2],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_QUANT_CONFIG_PATH,
    )
    p.add_argument(
        "--src_onnx",
        type=str,
        default="work_dirs/paddleocr_vl/inference.onnx",
        help="paddle2onnx 直接导出的 ONNX",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="work_dirs/paddleocr_vl/doclayoutv2_hmonnx_export",
    )
    p.add_argument(
        "--image",
        type=str,
        default="xh2modelzoo/data/images/layout.png",
    )
    p.add_argument("--layout_nms", action="store_true", default=False)
    p.add_argument(
        "--exec_device",
        type=str,
        default="auto",
        help="HMONNX 执行设备: auto/cpu/cuda:0",
    )
    p.add_argument("--skip_verify", action="store_true", help="跳过推理验证")
    p.add_argument("--skip_hmonnx", action="store_true", help="跳过 HMONNX 转换")
    return p.parse_args()


def _load_doclayout_config(config_path: Optional[str] = None):
    import importlib.util

    if config_path is None:
        config_path = DEFAULT_QUANT_CONFIG_PATH
    spec = importlib.util.spec_from_file_location("_doclayoutv2_cfg", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_input_shapes(config_path: Optional[str] = None) -> Dict[str, List[int]]:
    mod = _load_doclayout_config(config_path)
    input_shapes = getattr(mod, "input_shapes", None)
    if input_shapes is None:
        return DEFAULT_INPUT_SHAPES
    return {k: [int(x) for x in v] for k, v in input_shapes.items()}


def _simplify(
    src_onnx: str, dst_onnx: str, input_shapes: Dict[str, List[int]]
) -> onnx.ModelProto:
    """onnxsim + 固定输入 shape，消除大部分动态节点。"""
    print("[Step 1] onnxsim simplify ...")
    model = onnx.load(src_onnx)
    model_sim, ok = simplify(model, overwrite_input_shapes=input_shapes)
    assert ok, "onnxsim check failed"

    # 固定输入维度 dim_value
    for inp in model_sim.graph.input:
        if inp.name in input_shapes:
            for i, dim_val in enumerate(input_shapes[inp.name]):
                inp.type.tensor_type.shape.dim[i].dim_param = ""
                inp.type.tensor_type.shape.dim[i].dim_value = dim_val

    onnx.save(model_sim, dst_onnx)
    n_before = len(model.graph.node)
    n_after = len(model_sim.graph.node)
    print(f" nodes: {n_before} → {n_after}")
    return model_sim


def _collect_dynamic_tensors(
    model: onnx.ModelProto,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    init_names = {x.name for x in model.graph.initializer}

    # initializer 或者是constant节点输出是常量
    def is_const(name: str) -> bool:
        if name in init_names:
            return True
        for node in model.graph.node:
            for out in node.output:
                if out == name and node.op_type == "Constant":
                    return True
        return False

    expand_reshape_targets: Dict[str, List[str]] = {}  # tensor_name -> [node_names]
    range_targets: Dict[str, List[str]] = {}

    for node in model.graph.node:
        if node.op_type in ("Expand", "Reshape") and len(node.input) > 1:
            shape_in = node.input[1]
            if shape_in and not is_const(shape_in):
                expand_reshape_targets.setdefault(shape_in, []).append(
                    f"{node.op_type}:{node.name or 'unnamed'}"
                )
        if node.op_type == "Range" and len(node.input) == 3:
            for idx, inp in enumerate(node.input):
                if inp and not is_const(inp):
                    range_targets.setdefault(inp, []).append(
                        f"Range:{node.name or 'unnamed'}:input{idx}"
                    )

    return expand_reshape_targets, range_targets


def find_dynamic_tensors(model: onnx.ModelProto) -> Dict[str, Any]:
    """返回残余动态张量信息。"""
    print("[Step 2] Find remaining dynamic tensors ...")
    expand_targets, range_targets = _collect_dynamic_tensors(model)
    all_targets = sorted(set(expand_targets.keys()) | set(range_targets.keys()))
    print(f"  dynamic Expand/Reshape shape inputs: {len(expand_targets)}")
    print(f"  dynamic Range inputs: {len(range_targets)}")
    print(f"  unique tensors to resolve: {len(all_targets)}")
    for t in all_targets:
        users = expand_targets.get(t, []) + range_targets.get(t, [])
        print(f"    {t} ← used by {users}")
    return {
        "expand_targets": expand_targets,
        "range_targets": range_targets,
        "all_targets": all_targets,
    }


# ─────────────── dummy inference  ───────────────
def _build_dummy_feed(input_shapes: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    image_shape = input_shapes["image"]
    h = float(image_shape[2])
    w = float(image_shape[3])
    return {
        "im_shape": np.array([[h, w]], dtype=np.float32),
        "image": np.zeros(image_shape, dtype=np.float32),
        "scale_factor": np.array([[1.0, 1.0]], dtype=np.float32),
    }


def evaluate_tensors(
    model: onnx.ModelProto,
    target_names: List[str],
    input_shapes: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """跑dummy inference，获取实际值"""
    print("[Step 3] Evaluate dynamic tensors via dummy inference ...")
    if not target_names:
        print("  nothing to evaluate")
        return {}

    # 推断 value_info 获取 dtype
    inferred = shape_inference.infer_shapes(model)
    vi_map = {}
    for vi in (
        list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    ):
        vi_map[vi.name] = vi

    eval_model = copy.deepcopy(model)
    existing_out_names = {o.name for o in eval_model.graph.output}
    for name in target_names:
        if name in existing_out_names:
            continue
        vi = vi_map.get(name)
        if vi is not None:
            eval_model.graph.output.append(vi)
        else:
            eval_model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.INT64, None)
            )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        onnx.save(eval_model, tmp_path)
        sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        out_values = sess.run(out_names, _build_dummy_feed(input_shapes))
        out_map = dict(zip(out_names, out_values))
    finally:
        os.unlink(tmp_path)

    result = {}
    for name in target_names:
        if name in out_map:
            val = np.asarray(out_map[name])
            result[name] = val
            print(
                f"  {name}: value={val.tolist() if val.ndim else val.item()}, "
                f"shape={val.shape}, dtype={val.dtype}"
            )
        else:
            print(f"  {name}: FAILED to evaluate")
    return result


# ─────────────── 替换动态张量为常量 ───────────────
def replace_with_constants(
    sim_onnx: str,
    out_onnx: str,
    expand_targets: Dict[str, List[str]],
    range_targets: Dict[str, List[str]],
    tensor_values: Dict[str, np.ndarray],
    input_shapes: Dict[str, List[int]],
) -> Dict[str, Any]:
    """
    在 onnx 图中把动态张量替换为 initializer 常量。
    替换完后再跑一次 onnxsim 进一步常量折叠。
    """
    print("[Step 4] Replace dynamic tensors with constants ...")
    model = onnx.load(sim_onnx)
    replacements = []

    for node in model.graph.node:
        # Expand / Reshape 的 shape 输入
        if node.op_type in ("Expand", "Reshape") and len(node.input) > 1:
            shape_in = node.input[1]
            if shape_in in expand_targets and shape_in in tensor_values:
                const_name = f"_static_{shape_in}"
                val = tensor_values[shape_in]
                if val.dtype != np.int64:
                    val = val.astype(np.int64)
                model.graph.initializer.append(numpy_helper.from_array(val, const_name))
                node.input[1] = const_name
                replacements.append(
                    {
                        "node": f"{node.op_type}:{node.name}",
                        "old": shape_in,
                        "new": const_name,
                        "value": val.tolist(),
                    }
                )

        # Range 的非常量输入
        if node.op_type == "Range" and len(node.input) == 3:
            for idx, inp in enumerate(node.input):
                if inp in range_targets and inp in tensor_values:
                    const_name = f"_static_{inp}"
                    val = tensor_values[inp]
                    model.graph.initializer.append(
                        numpy_helper.from_array(val, const_name)
                    )
                    node.input[idx] = const_name
                    replacements.append(
                        {
                            "node": f"Range:{node.name}:input{idx}",
                            "old": inp,
                            "new": const_name,
                            "value": val.tolist() if val.ndim else val.item(),
                        }
                    )

    # 保存替换后的模型
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    onnx.save(model, tmp_path)

    # 再做一次 onnxsim 进一步折叠
    print("  running onnxsim again for constant folding ...")
    model2 = onnx.load(tmp_path)
    model_final, ok = simplify(model2)
    os.unlink(tmp_path)

    if not ok:
        print("  WARNING: second onnxsim check failed, using pre-sim model")
        model_final = model2

    # 固定输出 shape（用 dummy inference 获取实际 shape）
    sess = ort.InferenceSession(
        model_final.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feed = _build_dummy_feed(input_shapes)
    outputs = sess.run(None, feed)
    for i, out_meta in enumerate(sess.get_outputs()):
        graph_out = model_final.graph.output[i]
        for j, d in enumerate(outputs[i].shape):
            graph_out.type.tensor_type.shape.dim[j].dim_param = ""
            graph_out.type.tensor_type.shape.dim[j].dim_value = int(d)

    onnx.save(model_final, out_onnx)

    # 验证
    e2, r2 = _collect_dynamic_tensors(model_final)
    n_nodes = len(model_final.graph.node)
    print(f"  replacements: {len(replacements)}")
    print(f"  remaining dynamic Expand/Reshape: {len(e2)}, Range: {len(r2)}")
    print(f"  final nodes: {n_nodes}")
    print(f"  saved: {out_onnx}")

    return {
        "replacements": replacements,
        "remaining_expand": len(e2),
        "remaining_range": len(r2),
        "final_nodes": n_nodes,
    }


# ─────────────── 独立预处理 (无 PaddleOCR 依赖) ───────────────


def _preprocess_image(
    image_path: str,
    target_size: int = 800,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """RT-DETR 前处理: ReadImage → Resize → Normalize → ToCHW → ToBatch.

    与 PaddleX 官方 DetPredictor 的 pre_ops 流程一致，但不依赖 PaddleOCR 库。

    Returns:
        batch_inputs: [im_shape(1,2), image(1,3,H,W), scale_factor(1,2)]
        ori_img: 原始 BGR 图像 (用于可视化)
    """
    import cv2

    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = ori_img.shape[:2]

    # BGR → RGB → resize → float32 → /255
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size)).astype(np.float32) / 255.0

    # PP-DocLayoutV2 官方 inference.yml: mean=[0,0,0], std=[1,1,1], norm_type=none
    # 只需 /255 归一化到 [0,1]，不做 ImageNet normalize

    # HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    im_shape = np.array([[float(target_size), float(target_size)]], dtype=np.float32)
    scale_factor = np.array(
        [[float(target_size) / h, float(target_size) / w]],
        dtype=np.float32,
    )

    return [im_shape, img, scale_factor], ori_img


# ─────────────── 独立后处理 (无 PaddleOCR 依赖) ───────────────

# PP-DocLayoutV2 官方 25 类标签 (来自 inference.yml label_list)
DOCLAYOUT_LABELS: Dict[int, str] = {
    0: "abstract",
    1: "algorithm",
    2: "aside_text",
    3: "chart",
    4: "content",
    5: "display_formula",
    6: "doc_title",
    7: "figure_title",
    8: "footer",
    9: "footer_image",
    10: "footnote",
    11: "formula_number",
    12: "header",
    13: "header_image",
    14: "image",
    15: "inline_formula",
    16: "number",
    17: "paragraph_title",
    18: "reference",
    19: "reference_content",
    20: "seal",
    21: "table",
    22: "text",
    23: "vertical_text",
    24: "vision_footnote",
}

_DEFAULT_THRESHOLD = 0.5

# 与 PaddleX processors.py SKIP_ORDER_LABELS 保持一致 —— 这些标签不参与阅读排序，order 置 null
SKIP_ORDER_LABELS = {
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
}


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个 [x1,y1,x2,y2] 框的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _layout_nms(
    boxes: np.ndarray,
    iou_same: float = 0.6,
    iou_diff: float = 0.98,
) -> np.ndarray:
    """Layout NMS: 同类 IoU 阈值 0.6，异类 IoU 阈值 0.98。

    boxes: [N, 6] = [cls_id, score, x1, y1, x2, y2]
    """
    if len(boxes) == 0:
        return boxes
    order = np.argsort(-boxes[:, 1])  # 按 score 降序
    keep = []
    suppressed = set()
    for i in order:
        if int(i) in suppressed:
            continue
        keep.append(int(i))
        for j in order:
            if int(j) in suppressed or int(j) == int(i):
                continue
            iou = _compute_iou(boxes[i, 2:6], boxes[j, 2:6])
            threshold = iou_same if boxes[i, 0] == boxes[j, 0] else iou_diff
            if iou > threshold:
                suppressed.add(int(j))
    return boxes[keep]


def _postprocess_detections(
    raw_outputs: List[np.ndarray],
    ori_h: int,
    ori_w: int,
    threshold: float = _DEFAULT_THRESHOLD,
    layout_nms: bool = False,
) -> List[Dict[str, Any]]:
    """将 RT-DETR 原始输出转为结构化检测框列表。

    raw_outputs[0]: [N, 8] = [cls_id, score, x1, y1, x2, y2, order_col6, order_col7]
                    或 [N, 6] (旧版无 order 列)
    raw_outputs[1]: [batch] = bbox_num

    Returns: [{cls_id, label, score, coordinate:[x1,y1,x2,y2], order:int|None}, ...]
             按模型内置阅读顺序排列。
    """
    bboxes = raw_outputs[0]
    bbox_num = int(raw_outputs[1][0]) if len(raw_outputs) > 1 else len(bboxes)
    bboxes = bboxes[:bbox_num]

    if len(bboxes) == 0:
        return []

    # 0. 阅读顺序排序 (PP-DocLayoutV2 输出 8 列，第 6/7 列为阅读顺序索引)
    #    与 PaddleX processors.py 保持一致:
    #    shape[1]==8 → lexsort(-col7, col6), shape[1]==7 → argsort(col6)
    ncols = bboxes.shape[1] if bboxes.ndim == 2 else 0
    if ncols == 8:
        order_idx = np.lexsort((-bboxes[:, 7], bboxes[:, 6]))
        bboxes = bboxes[order_idx]
    elif ncols == 7:
        order_idx = np.argsort(bboxes[:, 6])
        bboxes = bboxes[order_idx]

    # 1. 阈值过滤
    mask = bboxes[:, 1] > threshold
    bboxes = bboxes[mask]

    # 2. Layout NMS
    if layout_nms and len(bboxes) > 0:
        bboxes = _layout_nms(bboxes)

    # 3. 坐标裁剪到图像范围
    if len(bboxes) > 0:
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, ori_w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, ori_h)
        bboxes[:, 4] = np.clip(bboxes[:, 4], 0, ori_w)
        bboxes[:, 5] = np.clip(bboxes[:, 5], 0, ori_h)

    # 4. 过滤微小框 (w<6 or h<6)
    result_boxes = []
    for row in bboxes:
        cls_id = int(row[0])
        score = float(row[1])
        x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
        if (x2 - x1) < 6 or (y2 - y1) < 6:
            continue
        result_boxes.append(
            {
                "cls_id": cls_id,
                "label": DOCLAYOUT_LABELS.get(cls_id, f"cls_{cls_id}"),
                "score": score,
                "coordinate": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

    if len(result_boxes) > 1:
        filtered = []
        coords = np.array([b["coordinate"] for b in result_boxes], dtype=np.float32)
        areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
        removed = set()
        for i in range(len(result_boxes)):
            if i in removed:
                continue
            for j in range(i + 1, len(result_boxes)):
                if j in removed:
                    continue
                iou = _compute_iou(coords[i], coords[j])
                if iou > 0.7:
                    # 移除面积较小的
                    if areas[i] >= areas[j]:
                        removed.add(j)
                    else:
                        removed.add(i)
                        break
        result_boxes = [b for idx, b in enumerate(result_boxes) if idx not in removed]

    # 6. 分配阅读顺序 order (与 PaddleX update_order_index 一致)
    #    SKIP_ORDER_LABELS 中的标签 order=None，其余按出现顺序递增
    order_counter = 1
    for b in result_boxes:
        if b["label"] in SKIP_ORDER_LABELS:
            b["order"] = None
        else:
            b["order"] = order_counter
            order_counter += 1

    return result_boxes


# ─────────────── 独立可视化 ───────────────

# PaddleOCR/PaddleX 官方 20 色调色板 (BGR 格式，与 OpenCV 一致)
# 来自 paddlex.inference.utils.color_map.get_colormap
_VIS_COLORS_BGR = [
    (0, 0, 255),  # 0:  RGB(255,0,0)
    (0, 255, 204),  # 1:  RGB(204,255,0)
    (102, 255, 0),  # 2:  RGB(0,255,102)
    (255, 102, 0),  # 3:  RGB(0,102,255)
    (255, 0, 204),  # 4:  RGB(204,0,255)
    (0, 77, 255),  # 5:  RGB(255,77,0)
    (0, 255, 128),  # 6:  RGB(128,255,0)
    (178, 255, 0),  # 7:  RGB(0,255,178)
    (255, 26, 0),  # 8:  RGB(0,26,255)
    (229, 0, 255),  # 9:  RGB(255,0,229)
    (0, 153, 255),  # 10: RGB(255,153,0)
    (0, 255, 51),  # 11: RGB(51,255,0)
    (255, 255, 0),  # 12: RGB(0,255,255)
    (255, 0, 51),  # 13: RGB(51,0,255)
    (153, 0, 255),  # 14: RGB(255,0,153)
    (0, 229, 255),  # 15: RGB(255,229,0)
    (26, 255, 0),  # 16: RGB(0,255,26)
    (255, 178, 0),  # 17: RGB(0,178,255)
    (255, 0, 128),  # 18: RGB(128,0,255)
    (77, 0, 255),  # 19: RGB(255,0,77)
]

_VIS_LABEL_COLOR_INDEX: Dict[str, int] = {
    "figure_title": 0,
    "table": 1,
    "paragraph_title": 2,
    "text": 3,
}

# 与 paddlex.inference.utils.color_map.font_colormap 对齐
_VIS_LIGHT_FONT_INDEXES = {0, 3, 4, 8, 9, 13, 14, 18, 19}


def _font_color_from_palette_index(color_idx: int) -> Tuple[int, int, int]:
    if color_idx in _VIS_LIGHT_FONT_INDEXES:
        return (255, 255, 255)
    # 官方 dark=RGB(0x14, 0x0E, 0x35)，OpenCV 使用 BGR
    return (53, 14, 20)


def _visualize_boxes(
    ori_img: np.ndarray,
    boxes: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """用 OpenCV 在原图上画检测框并保存。"""
    import cv2

    vis = ori_img.copy()
    label2style: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}
    for i, b in enumerate(boxes):
        cls_id = b["cls_id"]
        x1, y1, x2, y2 = b["coordinate"]
        score = b["score"]
        label = b["label"]

        # 优先使用固定标签映射，保证 text/paragraph/table 的颜色稳定一致。
        # 其它标签回退到 cls_id 对应的颜色。
        if label not in label2style:
            color_idx = _VIS_LABEL_COLOR_INDEX.get(label, cls_id % len(_VIS_COLORS_BGR))
            color = _VIS_COLORS_BGR[color_idx]
            font_color = _font_color_from_palette_index(color_idx)
            label2style[label] = (color, font_color)
        color, font_color = label2style[label]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            vis,
            text,
            (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            font_color,
            1,
        )

        # 右上角顺序号：与官方 draw_box 一致，始终使用列表位置号 (i+1)
        order_text = str(i + 1)
        (ow, oh), _ = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = x2 + 2
        if vis.shape[1] - x2 < ow + 4:
            tx = max(0, x2 - ow - 2)
        ty = max(oh + 2, y1 + oh // 2)
        cv2.putText(
            vis,
            order_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


# ─────────────── 验证推理正确性 ───────────────


def verify(
    static_onnx: str,
    src_onnx: str,
    image: str,
    layout_nms: bool,
    output_dir: str,
    input_shapes: Dict[str, List[int]],
) -> Dict[str, Any]:
    """用独立前后处理对比 静态ONNX vs 原始ONNX 的推理结果 (无 PaddleOCR 依赖)。"""
    print("[Step 5] Verify static ONNX vs original ONNX ...")

    # 预处理
    target_size = input_shapes["image"][2]  # 800
    batch_inputs, ori_img = _preprocess_image(image, target_size)
    ori_h, ori_w = ori_img.shape[:2]

    feed = {
        "im_shape": batch_inputs[0],
        "image": batch_inputs[1],
        "scale_factor": batch_inputs[2],
    }

    # --- 静态 ONNX 推理 ---
    sess_static = ort.InferenceSession(static_onnx, providers=["CPUExecutionProvider"])
    static_outputs = sess_static.run(None, feed)
    static_boxes = _postprocess_detections(
        static_outputs,
        ori_h,
        ori_w,
        threshold=_DEFAULT_THRESHOLD,
        layout_nms=layout_nms,
    )

    # --- 原始 ONNX 推理 (paddle2onnx 导出的动态 shape ONNX) ---
    sess_orig = ort.InferenceSession(src_onnx, providers=["CPUExecutionProvider"])
    orig_outputs = sess_orig.run(None, feed)
    orig_boxes = _postprocess_detections(
        orig_outputs,
        ori_h,
        ori_w,
        threshold=_DEFAULT_THRESHOLD,
        layout_nms=layout_nms,
    )

    # --- 保存可视化 ---
    os.makedirs(output_dir, exist_ok=True)
    _visualize_boxes(
        ori_img, static_boxes, str(Path(output_dir) / "static_onnx_layout_res.jpg")
    )
    _visualize_boxes(
        ori_img, orig_boxes, str(Path(output_dir) / "original_onnx_layout_res.jpg")
    )

    # --- 数值对比: 原始输出张量 ---
    tensor_diffs = []
    for i, (ref, out) in enumerate(zip(orig_outputs, static_outputs)):
        if ref.shape != out.shape:
            tensor_diffs.append(
                {
                    "output": i,
                    "error": "shape mismatch",
                    "orig_shape": list(ref.shape),
                    "static_shape": list(out.shape),
                }
            )
            continue
        abs_diff = np.abs(ref.astype(np.float64) - out.astype(np.float64))
        tensor_diffs.append(
            {
                "output": i,
                "shape": list(ref.shape),
                "max_abs_diff": float(abs_diff.max()),
                "mean_abs_diff": float(abs_diff.mean()),
            }
        )

    # --- box 级对比 ---
    n = min(len(static_boxes), len(orig_boxes))
    max_score_diff = 0.0
    max_coord_diff = 0
    for i in range(n):
        max_score_diff = max(
            max_score_diff, abs(static_boxes[i]["score"] - orig_boxes[i]["score"])
        )
        max_coord_diff = max(
            max_coord_diff,
            max(
                abs(a - b)
                for a, b in zip(
                    static_boxes[i]["coordinate"], orig_boxes[i]["coordinate"]
                )
            ),
        )

    summary = {
        "static_box_count": len(static_boxes),
        "original_box_count": len(orig_boxes),
        "max_score_diff": max_score_diff,
        "max_coord_diff": max_coord_diff,
        "tensor_diffs": tensor_diffs,
        "pass": (
            len(static_boxes) == len(orig_boxes)
            and max_coord_diff == 0
            and max_score_diff < 1e-4
        ),
    }
    print(f"  boxes: static={len(static_boxes)}, original={len(orig_boxes)}")
    for j, b in enumerate(static_boxes):
        order_str = b.get("order", "-")
        print(
            f"    [{j+1:2d}] cls={b['cls_id']:2d} ({b['label']:18s}) "
            f"coord={b['coordinate']}  order={order_str}"
        )
    print(f"  max_score_diff={max_score_diff:.6e}, max_coord_diff={max_coord_diff}")
    for td in tensor_diffs:
        if "error" in td:
            print(f"  output[{td['output']}]: {td['error']}")
        else:
            print(
                f"  output[{td['output']}]: max_diff={td['max_abs_diff']:.6e}, "
                f"mean_diff={td['mean_abs_diff']:.6e}"
            )
    print(f"  PASS: {summary['pass']}")

    (Path(output_dir) / "verify_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    (Path(output_dir) / "static_onnx_boxes.json").write_text(
        json.dumps(static_boxes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "batch_inputs": batch_inputs,
    }


# ─────────────── HMONNX 转换 ───────────────


def _load_quant_config(config_path: Optional[str] = None) -> dict:
    mod = _load_doclayout_config(config_path)
    return mod.quant_config


def convert_hmonnx(
    static_onnx: str,
    hmonnx_path: str,
    batch_inputs,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """调用 xhquant convert_onnx_to_hmonnx。"""
    print("[Step 6] Convert to HMONNX ...")
    import torch
    from xhquant.api import DeviceType, convert_onnx_to_hmonnx

    input_args = [torch.from_numpy(x).float().contiguous() for x in batch_inputs]

    # Monkeypatch protobuf remove() 避免重复删除报错
    from google.protobuf.internal import containers as pb_containers

    _original_remove = pb_containers.RepeatedCompositeFieldContainer.remove

    def _safe_remove(self, elem):
        try:
            return _original_remove(self, elem)
        except ValueError:
            return None

    pb_containers.RepeatedCompositeFieldContainer.remove = _safe_remove
    try:
        Path(hmonnx_path).parent.mkdir(parents=True, exist_ok=True)

        # fetch_name_1 (bbox_num=[300]) 是常量，HMONNX 转换会将其优化掉导致
        # KeyError: 'fetch_name_1'。需要从图输出中移除它，后处理中再补充。
        _static_model = onnx.load(static_onnx)
        _outputs_to_remove = [
            o for o in _static_model.graph.output if o.name != "fetch_name_0"
        ]
        for o in _outputs_to_remove:
            _static_model.graph.output.remove(o)

        _tmp_onnx = str(Path(hmonnx_path).parent / "_tmp_single_output.onnx")
        onnx.save(_static_model, _tmp_onnx)

        convert_onnx_to_hmonnx(
            _tmp_onnx,
            input_args,
            DeviceType.XH2a,
            hmonnx_path,
            quant_config=_load_quant_config(config_path),
            input_names=["im_shape", "image", "scale_factor"],
            output_names=["fetch_name_0"],
        )

        # 清理临时文件
        if os.path.exists(_tmp_onnx):
            os.unlink(_tmp_onnx)
        ok = Path(hmonnx_path).exists()
        print(f"  HMONNX saved: {hmonnx_path} (exists={ok})")
        return {"ok": ok, "path": hmonnx_path, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  HMONNX conversion failed: {e}")
        return {"ok": False, "path": hmonnx_path, "error": str(e), "traceback": tb}
    finally:
        pb_containers.RepeatedCompositeFieldContainer.remove = _original_remove


# ─────────────── HMONNX 推理验证 ───────────────


def verify_hmonnx(
    hmonnx_path: str,
    static_onnx: str,
    batch_inputs: List[np.ndarray],
    image_path: str,
    layout_nms: bool,
    output_dir: str,
    exec_device: str = "auto",
) -> Dict[str, Any]:
    """用 HMONNXGoldenInference 推理 HMONNX，与静态 ONNX 数值对比 + 独立可视化 (无 PaddleOCR 依赖)。"""
    print("[Step 7] Verify HMONNX inference ...")
    import torch
    from xhquant.api import HMONNXGoldenInference

    try:
        # --- ONNX reference ---
        sess = ort.InferenceSession(static_onnx, providers=["CPUExecutionProvider"])
        feed = {
            "im_shape": batch_inputs[0],
            "image": batch_inputs[1],
            "scale_factor": batch_inputs[2],
        }
        onnx_outputs = sess.run(None, feed)

        # --- HMONNX inference ---
        hm = HMONNXGoldenInference(hmonnx_path)

        if exec_device == "auto":
            exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if exec_device.startswith("cuda") and not torch.cuda.is_available():
            print("  WARN: CUDA not available, fallback to cpu")
            exec_device = "cpu"

        hm.exec_device = exec_device
        input_args = [
            torch.from_numpy(x).half().cpu().contiguous() for x in batch_inputs
        ]
        hm_outputs_raw = hm.forward(*input_args)

        # 转 numpy
        if isinstance(hm_outputs_raw, torch.Tensor):
            hm_np = [hm_outputs_raw.detach().cpu().float().numpy()]
        elif isinstance(hm_outputs_raw, (list, tuple)):
            hm_np = [
                (
                    x.detach().cpu().float().numpy()
                    if isinstance(x, torch.Tensor)
                    else np.asarray(x)
                )
                for x in hm_outputs_raw
            ]
        else:
            hm_np = [np.asarray(hm_outputs_raw)]

        # HMONNX 只输出 fetch_name_0 (bboxes)，补充 fetch_name_1 (bbox_num)
        if len(hm_np) == 1:
            bbox_num = np.array([hm_np[0].shape[0]], dtype=np.int32)
            hm_np.append(bbox_num)

        # --- 数值对比: HMONNX vs ONNX ---
        diffs = []
        for i, (ref, out) in enumerate(zip(onnx_outputs, hm_np)):
            if ref.shape != out.shape:
                diffs.append(
                    {
                        "output": i,
                        "error": "shape mismatch",
                        "ref_shape": list(ref.shape),
                        "hm_shape": list(out.shape),
                    }
                )
                continue
            abs_diff = np.abs(ref.astype(np.float64) - out.astype(np.float64))
            diffs.append(
                {
                    "output": i,
                    "shape": list(ref.shape),
                    "max_abs_diff": float(abs_diff.max()),
                    "mean_abs_diff": float(abs_diff.mean()),
                    "ref_range": [float(ref.min()), float(ref.max())],
                    "hm_range": [float(out.min()), float(out.max())],
                }
            )

        for d in diffs:
            if "error" in d:
                print(
                    f"  Output[{d['output']}]: {d['error']} "
                    f"ref={d['ref_shape']} hm={d['hm_shape']}"
                )
            else:
                print(
                    f"  Output[{d['output']}]: shape={d['shape']}, "
                    f"max_diff={d['max_abs_diff']:.6e}, mean_diff={d['mean_abs_diff']:.6e}"
                )

        # --- 后处理 + 可视化 (独立实现) ---
        import cv2

        ori_img = cv2.imread(image_path)
        box_count = 0
        if ori_img is not None:
            ori_h, ori_w = ori_img.shape[:2]
            hm_boxes = _postprocess_detections(
                hm_np,
                ori_h,
                ori_w,
                threshold=_DEFAULT_THRESHOLD,
                layout_nms=layout_nms,
            )
            box_count = len(hm_boxes)
            _visualize_boxes(
                ori_img, hm_boxes, str(Path(output_dir) / "hmonnx_layout_res.jpg")
            )
            (Path(output_dir) / "hmonnx_boxes.json").write_text(
                json.dumps(hm_boxes, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  HMONNX detected {box_count} boxes")

        (Path(output_dir) / "hmonnx_diff.json").write_text(
            json.dumps(diffs, indent=2),
            encoding="utf-8",
        )

        return {"ok": True, "diffs": diffs, "box_count": box_count, "error": None}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  HMONNX verify failed: {e}")
        (Path(output_dir) / "hmonnx_verify_error.log").write_text(tb, encoding="utf-8")
        return {"ok": False, "error": str(e), "traceback": tb}


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    input_shapes = _load_input_shapes(args.config)

    sim_onnx = str(Path(output_dir) / "PP-DocLayoutV2sim.onnx")
    static_onnx = str(Path(output_dir) / "PP-DocLayoutV2static.onnx")
    hmonnx_path = str(Path(output_dir) / "PP-DocLayoutV2hm.onnx")

    report: Dict[str, Any] = {"src_onnx": args.src_onnx}

    model_sim = _simplify(args.src_onnx, sim_onnx, input_shapes)

    dynamic_info = find_dynamic_tensors(model_sim)

    if not dynamic_info["all_targets"]:
        print("  No dynamic tensors remaining, skip rewrite")
        shutil.copy2(sim_onnx, static_onnx)
    else:
        tensor_values = evaluate_tensors(
            model_sim, dynamic_info["all_targets"], input_shapes
        )

        # 替换 + 二次 onnxsim
        rewrite_info = replace_with_constants(
            sim_onnx,
            static_onnx,
            dynamic_info["expand_targets"],
            dynamic_info["range_targets"],
            tensor_values,
            input_shapes,
        )
        report["rewrite"] = rewrite_info

    final_dyn = find_dynamic_tensors(onnx.load(static_onnx))
    report["final_dynamic_count"] = len(final_dyn["all_targets"])

    # 推理验证: 静态 ONNX vs 原始 ONNX
    verify_ctx = None
    if not args.skip_verify:
        verify_ctx = verify(
            static_onnx,
            args.src_onnx,
            args.image,
            args.layout_nms,
            output_dir,
            input_shapes,
        )
        report["verify"] = verify_ctx["summary"]

    # HMONNX
    if not args.skip_hmonnx:
        # batch_inputs: 优先复用 Step 5 的结果，否则独立预处理
        if verify_ctx is not None:
            batch_inputs = verify_ctx["batch_inputs"]
        else:
            target_size = input_shapes["image"][2]
            batch_inputs, _ = _preprocess_image(args.image, target_size)

        hmonnx_result = convert_hmonnx(
            static_onnx,
            hmonnx_path,
            batch_inputs,
            config_path=args.config,
        )
        report["hmonnx"] = hmonnx_result

        # HMONNX 推理验证: 与静态 ONNX 数值对比 + 独立后处理可视化
        if hmonnx_result["ok"]:
            hm_verify = verify_hmonnx(
                hmonnx_path,
                static_onnx,
                batch_inputs,
                args.image,
                args.layout_nms,
                output_dir,
                exec_device=args.exec_device,
            )
            report["hmonnx_verify"] = hm_verify

    report_path = Path(output_dir) / "export_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\n{'='*60}")
    print(f"Report: {report_path}")
    print(f"Static ONNX: {static_onnx}")
    if not args.skip_hmonnx:
        print(f"HMONNX: {hmonnx_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
