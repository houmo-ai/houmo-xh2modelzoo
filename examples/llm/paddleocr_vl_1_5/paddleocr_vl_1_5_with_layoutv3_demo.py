"""
DocLayoutV3 布局检测 → 按多边形区域裁剪 → PaddleOCR-VL 分区识别

V3 相比 V2 的核心变化:
  - 模型输出 3 个张量: bboxes[N,7] + bbox_num[1] + masks[N,200,200]
  - 检测结果为不规则多边形 (而非矩形)
  - 裁剪时使用多边形最小外接矩形, 并 mask 掉其他检测区域的内容

使用方式 (示例):
  python examples/paddleocr_vl_1_5/paddleocr_vl_1_5_with_layoutv3_demo.py \\
    --image /path/to/doc_image.jpg \\
    --layout_onnx  <PP-DocLayoutV3 静态 ONNX 路径> \\
    --hf_model_config_dir  <paddleocr_vl_llm_config_dir> \\
    --visual_onnx_path     <vision.onnx> \\
    --prefill_onnx_path    <prefill.onnx> \\
    --decode_onnx_path     <decode.onnx> \\
    --meta_info            <export_meta_info.json>
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import xhquant.utils.suppress_printing

from xhquant.api import Config, ConfigDict
try:
    from .common import xhquant_llm_init, get_root_logger
except ImportError:
    from common import xhquant_llm_init, get_root_logger  # pyright: ignore[reportMissingImports]
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.paddleocr_vl import PaddleOCRVLONNXModel, PaddleOCRVLProcessor

# PP-DocLayoutV3 官方 25 类标签
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

# 与 PaddleX processors.py SKIP_ORDER_LABELS 保持一致 ,这些标签不参与阅读排序, order=null
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


def parse_args():
    p = argparse.ArgumentParser(
        description="DocLayoutV3 布局检测 + PaddleOCR-VL 分区识别 ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 图像
    p.add_argument("--image", type=str, required=True, help="输入文档图像路径")

    # Layout 检测
    p.add_argument(
        "--layout_onnx",
        type=str,
        default="xh2modelzoo/work_dirs/paddleocr_vl_1_5/doclayoutv3_hmonnx_export/PP-DocLayoutV3static.onnx",
        help="PP-DocLayoutV3 模型路径 (ONNX 或 HMONNX)",
    )
    p.add_argument(
        "--layout_exec_device",
        type=str,
        default="auto",
        help="layout hmonnx 执行设备: auto/cpu/cuda:0",
    )
    p.add_argument(
        "--layout_target_size", type=int, default=800, help="布局检测模型的输入尺寸"
    )
    p.add_argument(
        "--layout_nms", action="store_true", default=False, help="是否启用 Layout NMS"
    )
    p.add_argument(
        "--score_threshold", type=float, default=0.5, help="布局检测置信度阈值"
    )
    p.add_argument("--crop_padding", type=int, default=2, help="裁剪区域 padding 像素")
    p.add_argument(
        "--layout_unclip_ratio",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_RATIO", "H_RATIO"),
        help="按 (宽比例, 高比例) 展开 bbox，如 1.05 1.05",
    )

    # PaddleOCR-VL 模型
    p.add_argument("--hf_model_config_dir", type=str, required=True)
    p.add_argument("--visual_onnx_path", type=str, required=True)
    p.add_argument("--prefill_onnx_path", type=str, required=True)
    p.add_argument("--decode_onnx_path", type=str, required=True)
    p.add_argument(
        "--meta_info", type=str, default=None, help="export_meta_info.json path"
    )
    p.add_argument("--cache_len", type=int, default=8192)
    p.add_argument("--num_hidden_layers", type=int, default=None)
    p.add_argument("--num_key_value_heads", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--output_json", type=str, default=None, help="结果 JSON 输出路径")
    p.add_argument("--save_crops", action="store_true", help="保存裁剪子图")
    p.add_argument(
        "--save_layout_vis",
        action="store_true",
        default=True,
        help="保存布局检测可视化",
    )
    p.add_argument(
        "--output_dir", type=str, default=None, help="输出目录 (裁剪子图、可视化、JSON)"
    )
    return p.parse_args()


# ─────────────── 标签 → Prompt 映射 ───────────────
LABEL_PROMPT_MAP = {
    # ── 文本类 ──
    "text": "OCR:",
    "paragraph_title": "OCR:",
    "header": "OCR:",
    "footer": "OCR:",
    "reference": "OCR:",
    "figure_title": "OCR:",
    "abstract": "OCR:",
    "content": "OCR:",
    "number": "OCR:",
    "doc_title": "OCR:",
    "reference_content": "OCR:",
    "vertical_text": "OCR:",
    "aside_text": "OCR:",
    "formula_number": "OCR:",
    "footnote": "OCR:",
    "vision_footnote": "OCR:",
    # ── 表格类 ──
    "table": "Table Recognition:",
    # ── 公式类 ──
    "display_formula": "Formula Recognition:",
    "algorithm": "Formula Recognition:",
    "inline_formula": "Formula Recognition:",
    # ── 图表类 ──
    "chart": "Chart Recognition:",
    # "image": "Chart Recognition:",
    # ── 印章/特殊 ──
    "seal": "Seal Recognition:",
}

DEFAULT_PROMPT = "OCR:"
# 可配置: 跳过不需要 OCR 的标签
# 注意: 这些标签仍会保留裁剪切片保存（用于可视化/后处理）
SKIP_LABELS = {"header_image", "footer_image", "image"}


def get_prompt_for_label(label: str) -> Optional[str]:
    """根据 layout 标签返回对应 prompt, 返回 None 表示跳过该区域。"""
    if label in SKIP_LABELS:
        return None
    return LABEL_PROMPT_MAP.get(label, DEFAULT_PROMPT)


def _get_task_from_prompt(prompt: Optional[str]) -> str:
    if prompt == "Table Recognition:":
        return "table"
    if prompt == "Formula Recognition:":
        return "formula"
    if prompt == "Chart Recognition:":
        return "chart"
    if prompt == "Spotting:":
        return "spotting"
    if prompt == "Seal Recognition:":
        return "seal"
    return "ocr"


def _preprocess_crop_for_task(
    crop_img: Image.Image, task: str
) -> Tuple[Image.Image, int]:
    """按 task 做图像预处理，返回 (处理后图像, max_pixels)。"""
    image = crop_img
    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500

    if (
        task == "spotting"
        and orig_w < spotting_upscale_threshold
        and orig_h < spotting_upscale_threshold
    ):
        process_w, process_h = orig_w * 2, orig_h * 2
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image = image.resize((process_w, process_h), resample_filter)

    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    return image, max_pixels


# ─────────────── 布局检测前处理 ───────────────


def _preprocess_image(
    image_path: str,
    target_size: int = 800,
) -> Tuple[List[np.ndarray], np.ndarray]:
    import cv2

    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = ori_img.shape[:2]

    # BGR → RGB → resize → float32 → /255
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size)).astype(np.float32) / 255.0

    # PP-DocLayoutV2 官方 inference.yml: mean=[0,0,0], std=[1,1,1], norm_type=none
    # 只需 /255 归一化到 [0,1], 不做 ImageNet normalize

    # HWC → CHW → NCHW
    img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    im_shape = np.array([[float(target_size), float(target_size)]], dtype=np.float32)
    scale_factor = np.array(
        [[float(target_size) / h, float(target_size) / w]],
        dtype=np.float32,
    )

    return [im_shape, img, scale_factor], ori_img


# ─────────────── 布局检测后处理 ───────────────
# IoU = 两个框的交集面积 / 两个框的并集面积
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


# 分别针对相同类别和不同类别进行布局非极大值抑制
def _layout_nms(
    boxes: np.ndarray,
    iou_same: float = 0.6,
    iou_diff: float = 0.98,
) -> Tuple[np.ndarray, List[int]]:
    """Layout NMS: 同类 IoU=0.6, 异类 IoU=0.98。返回 (filtered_boxes, keep_indices)。"""
    if len(boxes) == 0:
        return boxes, []
    order = np.argsort(-boxes[:, 1])
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
    return boxes[keep], keep


# ─────────────── mask → polygon (与 doclayoutv3_xh2a_export_hmonnx.py 一致) ───────────────


def _is_convex(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> bool:
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    return (v1[0] * v2[1] - v1[1] * v2[0]) < 0


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    u1 = v1 / (np.linalg.norm(v1) + 1e-12)
    u2 = v2 / (np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))))


def _extract_custom_vertices(
    polygon: np.ndarray,
    max_allowed_dist: float,
    sharp_angle_thresh: float = 45.0,
    max_dist_ratio: float = 0.3,
) -> List[Tuple[float, float]]:
    poly = np.array(polygon, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return [tuple(p) for p in poly]
    max_allowed_dist = max_allowed_dist * max_dist_ratio

    point_info = []
    for i in range(n):
        p_prev, p_curr, p_next = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        v1, v2 = p_prev - p_curr, p_next - p_curr
        point_info.append(
            {
                "index": i,
                "is_convex": _is_convex(p_prev, p_curr, p_next),
                "angle": _angle_between_vectors(v1, v2),
                "v1": v1,
                "v2": v2,
            }
        )

    concave_indices = [i for i, info in enumerate(point_info) if not info["is_convex"]]
    preserve_concave = set()
    if concave_indices:
        groups, current_group = [], [concave_indices[0]]
        for i in range(1, len(concave_indices)):
            if concave_indices[i] - concave_indices[i - 1] == 1 or (
                concave_indices[i - 1] == n - 1 and concave_indices[i] == 0
            ):
                current_group.append(concave_indices[i])
            else:
                if len(current_group) >= 2:
                    groups.extend(current_group)
                current_group = [concave_indices[i]]
        if len(current_group) >= 2:
            groups.extend(current_group)
        preserve_concave.update(groups)

    kept_points = [
        i
        for i, info in enumerate(point_info)
        if info["is_convex"] or (i in preserve_concave and info["angle"] >= 120)
    ]
    if not kept_points:
        kept_points = list(range(n))

    final_points = []
    for idx_i in range(len(kept_points)):
        ci, ni = kept_points[idx_i], kept_points[(idx_i + 1) % len(kept_points)]
        final_points.append(ci)
        dist = np.linalg.norm(poly[ci] - poly[ni])
        if dist > max_allowed_dist:
            intermediate = (
                list(range(ci + 1, ni))
                if ni > ci
                else list(range(ci + 1, n)) + list(range(0, ni))
            )
            if intermediate:
                num_needed = int(np.ceil(dist / max_allowed_dist)) - 1
                if len(intermediate) <= num_needed:
                    final_points.extend(intermediate)
                else:
                    step = len(intermediate) / num_needed
                    final_points.extend(
                        [intermediate[int(k * step)] for k in range(num_needed)]
                    )

    final_points = sorted(set(final_points))
    return [tuple(poly[i]) for i in final_points]


def _convert_polygon_to_quad(polygon) -> Optional[np.ndarray]:
    import cv2

    if polygon is None or len(polygon) < 3:
        return None
    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    min_rect = cv2.minAreaRect(points)
    quad = cv2.boxPoints(min_rect)
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]
    sums = quad[:, 0] + quad[:, 1]
    quad = np.roll(quad, -np.argmin(sums), axis=0)
    return quad


def _calculate_polygon_overlap_ratio(poly1_pts, poly2_pts, mode="union") -> float:
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
    except ImportError:
        return 0.0
    p1, p2 = ShapelyPolygon(poly1_pts), ShapelyPolygon(poly2_pts)
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    inter = p1.intersection(p2).area
    if mode == "union":
        u = p1.union(p2).area
        return inter / u if u > 0 else 0.0
    elif mode == "small":
        s = min(p1.area, p2.area)
        return inter / s if s > 0 else 0.0
    return 0.0


def _mask2polygon(mask, max_allowed_dist, epsilon_ratio=0.004, extract_custom=True):
    import cv2

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = np.atleast_2d(approx.squeeze())
    if extract_custom:
        pts = _extract_custom_vertices(pts, max_allowed_dist)
    return np.array(pts) if isinstance(pts, list) else pts


def _extract_polygons_from_masks(
    bboxes,
    masks,
    ori_h,
    ori_w,
    input_size=800,
    layout_shape_mode="auto",
):
    import cv2

    if masks is None or len(masks) == 0:
        return [None] * len(bboxes)
    mask_h, mask_w = masks.shape[1], masks.shape[2]
    scale_w = input_size / (4.0 * ori_w)
    scale_h = input_size / (4.0 * ori_h)
    max_box_w = float(np.max(bboxes[:, 4] - bboxes[:, 2])) if len(bboxes) > 0 else 1.0

    polygons = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i, 2:6].astype(np.int32)
        box_w, box_h = max(x2 - x1, 1), max(y2 - y1, 1)
        rect = [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ]

        if box_w <= 0 or box_h <= 0:
            polygons.append(rect)
            continue

        mx1 = int(np.clip(round(x1 * scale_w), 0, mask_w))
        mx2 = int(np.clip(round(x2 * scale_w), 0, mask_w))
        my1 = int(np.clip(round(y1 * scale_h), 0, mask_h))
        my2 = int(np.clip(round(y2 * scale_h), 0, mask_h))

        cropped = masks[i, my1:my2, mx1:mx2] if (mx2 > mx1 and my2 > my1) else None
        if cropped is None or cropped.size == 0:
            polygons.append(rect)
            continue
        cropped_bin = (cropped > 0.5).astype(np.uint8)
        if np.sum(cropped_bin) == 0:
            polygons.append(rect)
            continue
        if layout_shape_mode == "rect":
            polygons.append(rect)
            continue

        resized = cv2.resize(
            cropped_bin, (box_w, box_h), interpolation=cv2.INTER_NEAREST
        )
        mad = float(box_w) if box_w > max_box_w * 0.6 else float(max_box_w)
        poly = _mask2polygon(resized, mad)
        if poly is None or len(poly) < 4:
            polygons.append(rect)
            continue

        poly_abs = np.array(poly, dtype=np.float64) + np.array([x1, y1])
        poly_abs[:, 0] = np.clip(poly_abs[:, 0], 0, ori_w)
        poly_abs[:, 1] = np.clip(poly_abs[:, 1], 0, ori_h)

        if layout_shape_mode == "poly":
            polygons.append(poly_abs.tolist())
        elif layout_shape_mode == "quad":
            q = _convert_polygon_to_quad(poly_abs)
            polygons.append(q.tolist() if q is not None else rect)
        elif layout_shape_mode == "auto":
            rect_list = rect
            q = _convert_polygon_to_quad(poly_abs)
            if q is not None:
                q_list = q.tolist()
                iou_qr = _calculate_polygon_overlap_ratio(rect_list, q_list, "union")
                if iou_qr >= 0.95:
                    q_list = rect_list
                p_list = poly_abs.tolist()
                iou_pq = _calculate_polygon_overlap_ratio(p_list, q_list, "union")
                pre = polygons[-1] if polygons else None
                iou_pre = (
                    _calculate_polygon_overlap_ratio(pre, rect_list, "small")
                    if pre
                    else 0.0
                )
                if iou_pq >= 0.8 and iou_pre < 0.01:
                    polygons.append(q_list)
                    continue
            polygons.append(poly_abs.tolist())
        else:
            polygons.append(poly_abs.tolist())
    return polygons


def _unclip_boxes(bboxes, unclip_ratio, ori_h, ori_w):
    if unclip_ratio is None or (unclip_ratio[0] == 1.0 and unclip_ratio[1] == 1.0):
        return bboxes
    out = bboxes.copy()
    w = out[:, 4] - out[:, 2]
    h = out[:, 5] - out[:, 3]
    nw, nh = w * unclip_ratio[0], h * unclip_ratio[1]
    cx, cy = out[:, 2] + w / 2, out[:, 3] + h / 2
    out[:, 2] = np.clip(cx - nw / 2, 0, ori_w)
    out[:, 3] = np.clip(cy - nh / 2, 0, ori_h)
    out[:, 4] = np.clip(cx + nw / 2, 0, ori_w)
    out[:, 5] = np.clip(cy + nh / 2, 0, ori_h)
    return out


def _postprocess_detections(
    raw_outputs: List[np.ndarray],
    ori_h: int,
    ori_w: int,
    threshold: float = _DEFAULT_THRESHOLD,
    layout_nms: bool = False,
    input_size: int = 800,
    layout_unclip_ratio: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """V3 后处理: bboxes[N,7] + bbox_num[1] + masks[N,200,200]。

    Returns: [{cls_id, label, score, coordinate, polygon, order}, ...] 按阅读顺序。
    """
    bboxes = raw_outputs[0]
    bbox_num = int(raw_outputs[1][0]) if len(raw_outputs) > 1 else len(bboxes)
    bboxes = bboxes[:bbox_num]
    raw_masks = raw_outputs[2][:bbox_num] if len(raw_outputs) > 2 else None

    if len(bboxes) == 0:
        return []

    # 0. 阅读顺序排序
    ncols = bboxes.shape[1] if bboxes.ndim == 2 else 0
    if ncols == 8:
        order_idx = np.lexsort((-bboxes[:, 7], bboxes[:, 6]))
    elif ncols == 7:
        order_idx = np.argsort(bboxes[:, 6])
    else:
        order_idx = np.arange(len(bboxes))
    bboxes = bboxes[order_idx]
    if raw_masks is not None:
        raw_masks = raw_masks[order_idx]

    # 1. 阈值过滤
    score_mask = bboxes[:, 1] > threshold
    bboxes = bboxes[score_mask]
    if raw_masks is not None:
        raw_masks = raw_masks[score_mask]

    # 2. Layout NMS
    if layout_nms and len(bboxes) > 0:
        bboxes, keep_idx = _layout_nms(bboxes)
        if raw_masks is not None:
            raw_masks = raw_masks[keep_idx]

    # 3. 坐标裁剪
    if len(bboxes) > 0:
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, ori_w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, ori_h)
        bboxes[:, 4] = np.clip(bboxes[:, 4], 0, ori_w)
        bboxes[:, 5] = np.clip(bboxes[:, 5], 0, ori_h)

    # 4. 多边形提取 (V3)
    polygons = (
        _extract_polygons_from_masks(
            bboxes,
            raw_masks,
            ori_h,
            ori_w,
            input_size,
        )
        if raw_masks is not None
        else [None] * len(bboxes)
    )

    # 4b. bbox 展开
    if layout_unclip_ratio is not None and len(bboxes) > 0:
        bboxes = _unclip_boxes(bboxes, layout_unclip_ratio, ori_h, ori_w)

    # 5. 过滤微小框 + 组装
    result_boxes = []
    for row_idx, row in enumerate(bboxes):
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
                "polygon": polygons[row_idx],
            }
        )

    # 5b. IoU 去重
    if len(result_boxes) > 1:
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
                    if areas[i] >= areas[j]:
                        removed.add(j)
                    else:
                        removed.add(i)
                        break
        result_boxes = [b for idx, b in enumerate(result_boxes) if idx not in removed]

    # 6. 阅读顺序
    order_counter = 1
    for b in result_boxes:
        if b["label"] in SKIP_ORDER_LABELS:
            b["order"] = None
        else:
            b["order"] = order_counter
            order_counter += 1

    return result_boxes


# ─────────────── 布局检测可视化 ───────────────

# PaddleOCR/PaddleX 官方 20 色调色板 (BGR 格式)
_VIS_COLORS_BGR = [
    (0, 0, 255),
    (0, 255, 204),
    (102, 255, 0),
    (255, 102, 0),
    (255, 0, 204),
    (0, 77, 255),
    (0, 255, 128),
    (178, 255, 0),
    (255, 26, 0),
    (229, 0, 255),
    (0, 153, 255),
    (0, 255, 51),
    (255, 255, 0),
    (255, 0, 51),
    (153, 0, 255),
    (0, 229, 255),
    (26, 255, 0),
    (255, 178, 0),
    (255, 0, 128),
    (77, 0, 255),
]

_VIS_LABEL_COLOR_INDEX: Dict[str, int] = {
    "figure_title": 0,
    "table": 1,
    "paragraph_title": 2,
    "text": 3,
}

_VIS_LIGHT_FONT_INDEXES = {0, 3, 4, 8, 9, 13, 14, 18, 19}


def _font_color_from_palette_index(color_idx: int) -> Tuple[int, int, int]:
    if color_idx in _VIS_LIGHT_FONT_INDEXES:
        return (255, 255, 255)
    return (53, 14, 20)  # dark=RGB(0x14,0x0E,0x35) in BGR


def _visualize_boxes(
    ori_img: np.ndarray,
    boxes: List[Dict[str, Any]],
    output_path: str,
    draw_polygon: bool = True,
) -> None:
    """在原图上画多边形/矩形检测框 + 标签 + 顺序号并保存。"""
    import cv2

    vis = ori_img.copy()
    label2style: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}
    for i, b in enumerate(boxes):
        cls_id = b["cls_id"]
        x1, y1, x2, y2 = b["coordinate"]
        score = b["score"]
        label = b["label"]

        if label not in label2style:
            color_idx = _VIS_LABEL_COLOR_INDEX.get(label, cls_id % len(_VIS_COLORS_BGR))
            color = _VIS_COLORS_BGR[color_idx]
            font_color = _font_color_from_palette_index(color_idx)
            label2style[label] = (color, font_color)
        color, font_color = label2style[label]

        polygon = b.get("polygon")
        if draw_polygon and polygon is not None and len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color)
            cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 标签: polygon 模式用 left_top 多边形顶点
        if draw_polygon and polygon is not None and len(polygon) >= 3:
            pts_arr = np.array(polygon, dtype=np.float64)
            lt_pt = pts_arr[np.argmin(np.sum(pts_arr**2, axis=1))]
            lx, ly = int(lt_pt[0]), int(lt_pt[1])
        else:
            lx, ly = x1, y1

        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if ly < th + 6:
            cv2.rectangle(vis, (lx, ly), (lx + tw + 2, ly + th + 6), color, -1)
            cv2.putText(
                vis,
                text,
                (lx + 1, ly + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                font_color,
                1,
            )
        else:
            cv2.rectangle(vis, (lx, ly - th - 6), (lx + tw + 2, ly), color, -1)
            cv2.putText(
                vis,
                text,
                (lx + 1, ly - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                font_color,
                1,
            )

        # 顺序号: polygon 模式找 right_top 多边形顶点
        order_val = b.get("order")
        order_text = str(order_val) if order_val is not None else str(i + 1)
        (ow, oh), _ = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        if draw_polygon and polygon is not None and len(polygon) >= 3:
            img_rt = np.array([vis.shape[1], 0], dtype=np.float64)
            pts_arr = np.array(polygon, dtype=np.float64)
            rt_pt = pts_arr[np.argmin(np.sum((pts_arr - img_rt) ** 2, axis=1))]
            rx, ry = int(rt_pt[0]), int(rt_pt[1])
        else:
            rx, ry = x2, y1

        tx = rx + 2
        if vis.shape[1] - rx < ow + 4:
            tx = max(0, int(rx - ow - 2))
        ty = max(oh + 2, ry + oh // 2)
        cv2.putText(
            vis, order_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


# ═══════════════════════════════════════════════════════════
# 布局检测主入口 (ONNX Runtime, 不依赖 PaddleOCR)
# ═══════════════════════════════════════════════════════════


def run_layout_detection(
    image_path: str,
    layout_onnx: str,
    layout_exec_device: str = "auto",
    target_size: int = 800,
    threshold: float = _DEFAULT_THRESHOLD,
    layout_nms: bool = False,
    layout_unclip_ratio: Optional[Tuple[float, float]] = None,
    providers: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    用 PP-DocLayoutV3 ONNX 做布局检测 (支持多边形输出)。

    Returns:
        boxes: [{cls_id, label, score, coordinate, polygon, order}, ...] 按阅读顺序
        ori_img: 原始 BGR 图像 (供后续可视化)
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    batch_inputs, ori_img = _preprocess_image(image_path, target_size)
    ori_h, ori_w = ori_img.shape[:2]

    feed = {
        "im_shape": batch_inputs[0],
        "image": batch_inputs[1],
        "scale_factor": batch_inputs[2],
    }
    try:
        sess = ort.InferenceSession(layout_onnx, providers=providers)
        raw_outputs = sess.run(None, feed)
    except Exception:
        from xhquant.api import HMONNXGoldenInference

        hm = HMONNXGoldenInference(layout_onnx)
        if layout_exec_device == "auto":
            layout_exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if layout_exec_device.startswith("cuda") and not torch.cuda.is_available():
            layout_exec_device = "cpu"
        hm.exec_device = layout_exec_device

        input_args = [
            torch.from_numpy(x).half().cpu().contiguous() for x in batch_inputs
        ]
        hm_outputs = hm.forward(*input_args)

        if isinstance(hm_outputs, torch.Tensor):
            raw_outputs = [hm_outputs.detach().cpu().float().numpy()]
        elif isinstance(hm_outputs, (list, tuple)):
            raw_outputs = [
                (
                    x.detach().cpu().float().numpy()
                    if isinstance(x, torch.Tensor)
                    else np.asarray(x)
                )
                for x in hm_outputs
            ]
        else:
            raw_outputs = [np.asarray(hm_outputs)]

        if len(raw_outputs) == 1:
            bbox_num = np.array([raw_outputs[0].shape[0]], dtype=np.int32)
            raw_outputs.append(bbox_num)
        elif len(raw_outputs) == 2:
            # HMONNX 可能输出 [bboxes, masks] (无 bbox_num)
            # 如果第二个输出不是标量/1维，则视为 mask，需插入 bbox_num
            if raw_outputs[1].ndim >= 2:
                hm_bboxes = raw_outputs[0]
                hm_masks = raw_outputs[1]
                bbox_num = np.array([hm_bboxes.shape[0]], dtype=np.int32)
                raw_outputs = [hm_bboxes, bbox_num, hm_masks]

    boxes = _postprocess_detections(
        raw_outputs,
        ori_h,
        ori_w,
        threshold=threshold,
        layout_nms=layout_nms,
        input_size=target_size,
        layout_unclip_ratio=layout_unclip_ratio,
    )
    return boxes, ori_img


def crop_region(
    image: Image.Image, coordinate: List[int], padding: int = 2
) -> Image.Image:
    """根据 [x1, y1, x2, y2] 裁剪图像子区域 (矩形回退)。"""
    x1, y1, x2, y2 = coordinate
    w, h = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image.crop((x1, y1, x2, y2))


def crop_polygon_region(
    image: Image.Image,
    polygon: List[List[float]],
    all_boxes: List[Dict[str, Any]],
    current_idx: int,
    padding: int = 2,
    mask_others: bool = True,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """根据多边形的最小外接矩形裁剪图像，并 mask 掉属于其他检测区域的像素。

    V3 核心裁剪逻辑:
      1. 计算当前多边形的最小外接矩形 (axis-aligned bounding rect)
      2. 裁剪该矩形区域
      3. 在裁剪图中创建当前多边形的 mask
      4. 将属于其他检测区域且不属于当前区域的像素填白
    """
    import cv2

    w, h = image.size
    pts = np.array(polygon, dtype=np.float32)

    # 最小外接矩形 (axis-aligned)
    px1 = max(0, int(np.floor(pts[:, 0].min())) - padding)
    py1 = max(0, int(np.floor(pts[:, 1].min())) - padding)
    px2 = min(w, int(np.ceil(pts[:, 0].max())) + padding)
    py2 = min(h, int(np.ceil(pts[:, 1].max())) + padding)
    crop_w, crop_h = px2 - px1, py2 - py1
    if crop_w <= 0 or crop_h <= 0:
        return image.crop((0, 0, 1, 1))

    # 裁剪
    crop = image.crop((px1, py1, px2, py2))

    if not mask_others:
        return crop

    # 构建当前多边形 mask (在 crop 坐标系中)
    local_pts = pts - np.array([px1, py1], dtype=np.float32)
    cur_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(cur_mask, [local_pts.astype(np.int32)], 255)

    # 构建其他检测区域的联合 mask
    others_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for j, other_box in enumerate(all_boxes):
        if j == current_idx:
            continue
        other_poly = other_box.get("polygon")
        if other_poly is not None and len(other_poly) >= 3:
            other_pts = np.array(other_poly, dtype=np.float32) - np.array([px1, py1])
            cv2.fillPoly(others_mask, [other_pts.astype(np.int32)], 255)
        else:
            ox1, oy1, ox2, oy2 = other_box["coordinate"]
            lx1 = max(0, ox1 - px1)
            ly1 = max(0, oy1 - py1)
            lx2 = min(crop_w, ox2 - px1)
            ly2 = min(crop_h, oy2 - py1)
            if lx2 > lx1 and ly2 > ly1:
                others_mask[ly1:ly2, lx1:lx2] = 255

    # 需要填白的区域: 属于其他区域 且 不属于当前区域
    erase_mask = cv2.bitwise_and(others_mask, cv2.bitwise_not(cur_mask))

    if np.sum(erase_mask) > 0:
        crop_np = np.array(crop)
        crop_np[erase_mask > 0] = bg_color
        crop = Image.fromarray(crop_np)

    return crop


def crop_polygon_region_irregular(
    image: Image.Image,
    polygon: List[List[float]],
    padding: int = 2,
) -> Image.Image:
    """按多边形区域抠图并输出 RGBA（多边形外透明）。"""
    w, h = image.size
    pts = np.array(polygon, dtype=np.float32)

    px1 = max(0, int(np.floor(pts[:, 0].min())) - padding)
    py1 = max(0, int(np.floor(pts[:, 1].min())) - padding)
    px2 = min(w, int(np.ceil(pts[:, 0].max())) + padding)
    py2 = min(h, int(np.ceil(pts[:, 1].max())) + padding)
    crop_w, crop_h = px2 - px1, py2 - py1
    if crop_w <= 0 or crop_h <= 0:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 0))

    crop = image.crop((px1, py1, px2, py2)).convert("RGBA")
    local_pts = (pts - np.array([px1, py1], dtype=np.float32)).tolist()

    mask = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(p) for p in local_pts], fill=255)

    crop_np = np.array(crop)
    crop_np[..., 3] = np.array(mask, dtype=np.uint8)
    return Image.fromarray(crop_np, mode="RGBA")


# ─────────────── Vision ONNX 尺寸适配 ───────────────


def _get_vision_onnx_hw(vision_onnx_path: str) -> Tuple[int, int]:
    """从 vision ONNX 的 pixel_values 输入声明中读取静态 (H, W)。

    ONNX 输入形状: [1, 1, 3, H, W]
    如果有维度是动态的 (dim_param), 返回 None 表示不需要 pad。
    """
    import onnx

    model = onnx.load(vision_onnx_path, load_external_data=False)
    for inp in model.graph.input:
        if inp.name == "pixel_values":
            dims = inp.type.tensor_type.shape.dim
            # 检查 H, W 是否为静态值
            h_dim, w_dim = dims[-2], dims[-1]
            if h_dim.dim_value > 0 and w_dim.dim_value > 0:
                return int(h_dim.dim_value), int(w_dim.dim_value)
    return None  # type: ignore[return-value]


def pad_crop_to_vision_size(
    crop: Image.Image,
    target_h: int,
    target_w: int,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """将裁剪区域等比缩放后居中/左上 pad 到 (target_w, target_h)。

    步骤:
        1. 按 min(target_w/crop_w, target_h/crop_h) 等比缩放, 保持内容比例
        2. 在 target_w × target_h 白底画布上粘贴 (左上对齐)
        3. 将输入预对齐到 processor 的目标尺寸, 减少额外 resize 带来的形变
    """
    cw, ch = crop.size  # PIL: (width, height)
    scale = min(target_w / cw, target_h / ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    resized = crop.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    canvas.paste(resized, (0, 0))  # 左上对齐
    return canvas


# 初始化 PaddleOCR-VL 模型


def build_ocrvl_model(args):
    """
    根据命令行参数构建 PaddleOCR-VL ONNX 模型 + processor。
    与 paddleocr_vl_demo.py 逻辑一致。
    """
    cfg = Config()
    hf_model_dir = args.hf_model_config_dir
    meta = None
    if args.meta_info is not None:
        with open(args.meta_info, "r") as f:
            meta = json.load(f)
        hf_model_dir = meta.get("hf_model", hf_model_dir)
        num_hidden_layers = meta.get("num_hidden_layers")
        kv_cache_shape = meta.get("kv_cache_shape")
        if kv_cache_shape is not None:
            kv_cache_shape = [int(x) for x in kv_cache_shape]
    else:
        num_hidden_layers = args.num_hidden_layers
        if args.num_key_value_heads is None or args.head_dim is None:
            raise ValueError(
                "Please provide --num_key_value_heads and --head_dim when meta_info is not set"
            )
        kv_cache_shape = [1, args.num_key_value_heads, args.cache_len, args.head_dim]

    # hf_config 目录
    hf_config_dir = None
    candidates = [
        Path(args.hf_model_config_dir) / "hf_config",
        Path(hf_model_dir) / "hf_config",
        Path(args.hf_model_config_dir),
        Path(hf_model_dir),
    ]
    for c in candidates:
        if (c / "config.json").exists():
            hf_config_dir = c
            break
    if hf_config_dir is None:
        hf_config_dir = Path(args.hf_model_config_dir) / "hf_config"

    token_embedding_path = Path(args.hf_model_config_dir) / "token_embedding.pt"
    if not token_embedding_path.exists() and args.meta_info is not None:
        token_embedding_path = Path(args.meta_info).parent / "token_embedding.pt"

    cfg.hf_model_config_dir = str(hf_config_dir)
    cfg.embed_tokens = str(token_embedding_path)

    if num_hidden_layers is None or kv_cache_shape is None:
        raise ValueError("num_hidden_layers/kv_cache_shape is required")

    cfg.model = ConfigDict()
    cfg.model.type = "PaddleOCRVLONNXModel"
    cfg.model.hf_model_dir = hf_model_dir
    cfg.model.image_feature = ConfigDict()
    cfg.model.image_feature.onnx = args.visual_onnx_path
    cfg.model.prefill = ConfigDict()
    cfg.model.prefill.onnx = args.prefill_onnx_path
    cfg.model.prefill.input_sequence_length = 256
    cfg.model.decode = ConfigDict()
    cfg.model.decode.onnx = args.decode_onnx_path
    cfg.model.kv_cache = ConfigDict()
    cfg.model.kv_cache.num_hidden_layers = num_hidden_layers
    cfg.model.kv_cache.shape = kv_cache_shape
    cfg.model.cache_len = args.cache_len

    cfg.work_dir = str(Path("./work_dirs") / "paddleocr_vl_1_5" / "layout_vl_demo")
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"

    xhquant_llm_init(Path(cfg.work_dir) / "debug.log", False)
    logger = get_root_logger()

    xhquant.utils.suppress_printing.disable_printing = True

    exec_device = cfg.exec_device

    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding = torch.load(
        cfg.embed_tokens, weights_only=False, map_location="cpu"
    )
    torch.serialization.clear_safe_globals()

    # Processor
    processor_dir = Path(str(hf_config_dir))
    if not (processor_dir / "image_processing.py").exists():
        processor_dir = Path(hf_model_dir)
    processor = PaddleOCRVLProcessor.from_pretrained(
        str(processor_dir), trust_remote_code=True
    )

    from xh_model_zoo.xh_llm.models.paddleocr_vl.image_processing import SiglipImageProcessor

    processor.image_processor = SiglipImageProcessor.from_pretrained(str(processor_dir))

    if args.min_pixels is not None:
        processor.image_processor.min_pixels = args.min_pixels
    if args.max_pixels is not None:
        processor.image_processor.max_pixels = args.max_pixels

    xh_model: PaddleOCRVLONNXModel = MODELS.build(cfg.model)
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)
    xh_model.to(exec_device)

    return xh_model, processor, logger


def main():
    args = parse_args()
    image_path = args.image
    image = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path} ({image.size[0]}x{image.size[1]})")

    # 输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("./work_dirs/paddleocr_vl_1_5/layout_vl_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Step 1] Running DocLayoutV3 layout detection (ONNX) ...")
    print(f"  Layout ONNX: {args.layout_onnx}")
    unclip = tuple(args.layout_unclip_ratio) if args.layout_unclip_ratio else None
    t0 = time.time()
    boxes, ori_img_bgr = run_layout_detection(
        image_path,
        layout_onnx=args.layout_onnx,
        layout_exec_device=args.layout_exec_device,
        target_size=args.layout_target_size,
        threshold=args.score_threshold,
        layout_nms=args.layout_nms,
        layout_unclip_ratio=unclip,
    )
    t_layout = time.time() - t0
    print(f"  Detected {len(boxes)} regions in {t_layout:.2f}s")
    for i, box in enumerate(boxes):
        poly_info = f"  poly={len(box['polygon'])}pts" if box.get("polygon") else ""
        print(
            f"    [{i + 1:2d}] label={box['label']:18s} score={box['score']:.4f} "
            f"coord={box['coordinate']}  order={box.get('order')}{poly_info}"
        )

    if not boxes:
        print("No regions detected. Done.")
        return

    # 保存布局检测可视化
    if args.save_layout_vis:
        vis_path = str(output_dir / "layout_detection.jpg")
        _visualize_boxes(ori_img_bgr, boxes, vis_path)
        print(f"  Layout visualization: {vis_path}")

    print("\n" + "=" * 60)
    print("[Step 2] Building PaddleOCR-VL model ...")
    t0 = time.time()
    xh_model, processor, logger = build_ocrvl_model(args)
    t_init = time.time() - t0
    print(f"  Model ready in {t_init:.2f}s")

    # 以 vision onnx 的静态输入尺寸作为 crop padding 目标
    vision_hw = _get_vision_onnx_hw(args.visual_onnx_path)
    if vision_hw is not None:
        vis_h, vis_w = vision_hw
        print(f"  Vision ONNX static input: H={vis_h}, W={vis_w}")
        print(f"  → All crops will be padded to {vis_w}x{vis_h} before processor")
    else:
        vis_h, vis_w = None, None
        print("  Vision ONNX has dynamic input, no pre-padding needed")

    print("\n" + "=" * 60)
    print("[Step 3] Cropping regions and running PaddleOCR-VL ...")

    results = []
    total_ocr_time = 0.0
    # 分别推理
    for i, box in enumerate(boxes):
        label = box["label"]
        prompt = get_prompt_for_label(label)

        # V3 多边形裁剪: 使用多边形最小外接矩形, 并 mask 掉其他检测区域
        polygon = box.get("polygon")
        if polygon is not None and len(polygon) >= 3:
            crop_img = crop_polygon_region(
                image,
                polygon,
                boxes,
                i,
                padding=args.crop_padding,
                mask_others=True,
            )
        else:
            crop_img = crop_region(image, box["coordinate"], padding=args.crop_padding)

        raw_crop_size = f"{crop_img.size[0]}x{crop_img.size[1]}"
        crop_path = output_dir / f"crop_{i + 1:03d}_{label}.jpg"

        # 跳过 OCR 的标签: 仍保留切片（若未开启 save_crops 也单独保存）
        if prompt is None:
            saved_path = crop_path
            if polygon is not None and len(polygon) >= 3:
                irregular_crop = crop_polygon_region_irregular(
                    image,
                    polygon,
                    padding=args.crop_padding,
                )
                saved_path = output_dir / f"crop_{i + 1:03d}_{label}.png"
                irregular_crop.save(str(saved_path))
            else:
                crop_img.save(str(saved_path))
            print(
                f"\n  [{i + 1:2d}] label={label:18s} -> SKIP (crop saved: {saved_path})"
            )
            results.append(
                {
                    "index": i + 1,
                    "label": label,
                    "score": box["score"],
                    "coordinate": box["coordinate"],
                    "polygon": box.get("polygon"),
                    "order": box.get("order"),
                    "prompt": None,
                    "recognition": None,
                    "skipped": True,
                    "saved_crop": str(saved_path),
                }
            )
            continue

        if args.save_crops:
            crop_img.save(str(crop_path))

        task = _get_task_from_prompt(prompt)
        crop_img, task_max_pixels = _preprocess_crop_for_task(crop_img, task)
        effective_max_pixels = (
            args.max_pixels if args.max_pixels is not None else task_max_pixels
        )
        processor.image_processor.max_pixels = effective_max_pixels

        # ── 将 crop pad 到 vision ONNX 期望的静态尺寸 ──
        if vis_h is not None and vis_w is not None:
            crop_img = pad_crop_to_vision_size(crop_img, vis_h, vis_w)

        print(
            f'\n  [{i + 1:2d}] label={label:18s} prompt="{prompt}" '
            f"task={task} max_pixels={effective_max_pixels} "
            f"crop={raw_crop_size} -> padded={crop_img.size[0]}x{crop_img.size[1]}"
        )

        # PaddleOCR-VL 推理
        t0 = time.time()
        try:
            text = xh_model.chat(
                prompt,
                crop_img,
                processor,
                logger,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            text = f"[ERROR] {e}"
        t_ocr = time.time() - t0
        total_ocr_time += t_ocr

        preview = str(text)[:200]
        if len(str(text)) > 200:
            preview += "..."
        print(f"    Result ({t_ocr:.2f}s): {preview}")

        results.append(
            {
                "index": i + 1,
                "label": label,
                "score": box["score"],
                "coordinate": box["coordinate"],
                "polygon": box.get("polygon"),
                "order": box.get("order"),
                "prompt": prompt,
                "recognition": text,
                "skipped": False,
                "time_sec": round(t_ocr, 2),
            }
        )

    print("\n" + "=" * 60)
    print("[Step 4] Summary")

    recognized = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    print(f"  Total regions:  {len(results)}")
    print(f"  Recognized:     {len(recognized)}")
    print(f"  Skipped:        {len(skipped)}")
    print(f"  Layout time:    {t_layout:.2f}s")
    print(f"  Model init:     {t_init:.2f}s")
    print(f"  OCR total:      {total_ocr_time:.2f}s")

    output = {
        "image": image_path,
        "image_size": [image.size[0], image.size[1]],
        "layout_onnx": args.layout_onnx,
        "layout_boxes": len(boxes),
        "timing": {
            "layout_sec": round(t_layout, 2),
            "model_init_sec": round(t_init, 2),
            "ocr_total_sec": round(total_ocr_time, 2),
        },
        "regions": results,
    }

    print()
    for r in results:
        status = "SAVE" if r.get("skipped") else "OK"
        text_preview = (str(r.get("recognition") or ""))[:100]
        print(f"  [{r['index']:2d}] {r['label']:18s} [{status:4s}] {text_preview}")

    json_path = args.output_json or str(output_dir / "layout_ocrvl_result.json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Result JSON: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
