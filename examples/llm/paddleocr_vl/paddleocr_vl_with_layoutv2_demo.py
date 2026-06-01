"""
DocLayoutV2 布局检测 → 按区域裁剪 → PaddleOCR-VL 分区识别

使用方式 (示例):
  python examples/paddleocr_vl/paddleocr_vl_with_layoutv2_demo.py \
    --image /path/to/doc_image.jpg \
    --layout_onnx  <PP-DocLayoutV2 静态 ONNX 路径> \
    --hf_model_config_dir  <paddleocr_vl_llm_config_dir> \
    --visual_onnx_path     <vision.onnx> \
    --prefill_onnx_path    <prefill.onnx> \
    --decode_onnx_path     <decode.onnx> \
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
from PIL import Image

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

# PP-DocLayoutV2 官方 25 类标签
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
        description="DocLayoutV2 布局检测 + PaddleOCR-VL 分区识别 ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 图像
    p.add_argument("--image", type=str, required=True, help="输入文档图像路径")

    # Layout 检测
    p.add_argument(
        "--layout_onnx",
        type=str,
        default="/data01/home/linxiang.wang/xhquant_llm/work_dirs/paddleocr_vl/PP-DocLayoutV2.onnx",
        help="PP-DocLayoutV2 模型路径 (ONNX 或 HMONNX)",
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
    "seal": "OCR:",
}

DEFAULT_PROMPT = "OCR:"

# 可配置: 跳过不需要 OCR 的标签
SKIP_LABELS = {"header_image", "footer_image", "image"}


def get_prompt_for_label(label: str) -> Optional[str]:
    """根据 layout 标签返回对应 prompt, 返回 None 表示跳过该区域。"""
    if label in SKIP_LABELS:
        return None
    return LABEL_PROMPT_MAP.get(label, DEFAULT_PROMPT)


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


# 分别针对相同类别和不同类别进行布局非极大值抑制，对检测到的文本框/单元格进行去重，保留最佳检测结果。
def _layout_nms(
    boxes: np.ndarray,
    iou_same: float = 0.6,
    iou_diff: float = 0.98,
) -> np.ndarray:
    """Layout NMS: 同类 IoU 阈值 0.6, 异类 IoU 阈值 0.98。

    boxes: [N, >=6] — 前 6 列为 [cls_id, score, x1, y1, x2, y2]
    """
    if len(boxes) == 0:
        return boxes
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
                # 置信度较低的框 j 抑制掉
                suppressed.add(int(j))
    return boxes[keep]


def _postprocess_detections(
    raw_outputs: List[np.ndarray],
    ori_h: int,
    ori_w: int,
    threshold: float = _DEFAULT_THRESHOLD,
    layout_nms: bool = False,
) -> List[Dict[str, Any]]:
    """
    raw_outputs[0]: [N, 8] = [cls_id, score, x1, y1, x2, y2, order_col6, order_col7]
                    或 [N, 6] (旧版无 order 列)
    raw_outputs[1]: [batch] = bbox_num

    Returns: [{cls_id, label, score, coordinate:[x1,y1,x2,y2], order:int|None}, ...]
             按模型内置阅读顺序排列。
    """
    bboxes = raw_outputs[0]
    bbox_num = int(raw_outputs[1][0]) if len(raw_outputs) > 1 else len(bboxes)
    # print('&&&&&&&&$$$$$$$$$$$',bboxes.shape)
    bboxes = bboxes[:bbox_num]
    # print('&&&&&&&&$$$$$$$$$$$',bboxes.shape)
    # print('&&&&&&&&$$$$$$$$$$$',bbox_num)

    if len(bboxes) == 0:
        return []

    # 0. 阅读顺序排序 (PP-DocLayoutV2 输出 8 列, 第 6/7 列为阅读顺序索引)
    ncols = bboxes.shape[1] if bboxes.ndim == 2 else 0
    if ncols == 8:
        # print('#####',bboxes[:, 7],'##', bboxes[:, 6])
        order_idx = np.lexsort((-bboxes[:, 7], bboxes[:, 6]))
        bboxes = bboxes[order_idx]
        # print('#####',-bboxes[:, 7],'##', bboxes[:, 6],'##', order_idx)
        # print('#####',bboxes[:, 7],'##', bboxes[:, 6])
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

    # 5. 去除高重叠框
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

    # 6. 分配阅读顺序 order (与 PaddleX update_order_index 一致)
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
) -> None:
    """在原图上画检测框 + 标签 + 顺序号并保存。"""
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

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            vis, text, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 1
        )

        # 右上角顺序号
        order_text = str(i + 1)
        (ow, oh), _ = cv2.getTextSize(order_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = x2 + 2
        if vis.shape[1] - x2 < ow + 4:
            tx = max(0, x2 - ow - 2)
        ty = max(oh + 2, y1 + oh // 2)
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
    providers: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    用 PP-DocLayoutV2 ONNX 做布局检测。

    Returns:
        boxes: [{cls_id, label, score, coordinate, order}, ...] 按阅读顺序
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

    boxes = _postprocess_detections(
        raw_outputs,
        ori_h,
        ori_w,
        threshold=threshold,
        layout_nms=layout_nms,
    )
    return boxes, ori_img


def crop_region(
    image: Image.Image, coordinate: List[int], padding: int = 2
) -> Image.Image:
    """根据 [x1, y1, x2, y2] 裁剪图像子区域。"""
    x1, y1, x2, y2 = coordinate
    w, h = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return image.crop((x1, y1, x2, y2))


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

    cfg.work_dir = str(Path("./work_dirs") / "paddleocr_vl" / "layout_vl_demo")
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
        output_dir = Path("./work_dirs/paddleocr_vl/layout_vl_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Step 1] Running DocLayoutV2 layout detection (ONNX) ...")
    print(f"  Layout ONNX: {args.layout_onnx}")
    t0 = time.time()
    boxes, ori_img_bgr = run_layout_detection(
        image_path,
        layout_onnx=args.layout_onnx,
        layout_exec_device=args.layout_exec_device,
        target_size=args.layout_target_size,
        threshold=args.score_threshold,
        layout_nms=args.layout_nms,
    )
    t_layout = time.time() - t0
    print(f"  Detected {len(boxes)} regions in {t_layout:.2f}s")
    for i, box in enumerate(boxes):
        print(
            f"    [{i + 1:2d}] label={box['label']:18s} score={box['score']:.4f} "
            f"coord={box['coordinate']}  order={box.get('order')}"
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
        if prompt is None:
            crop_img = crop_region(image, box["coordinate"], padding=args.crop_padding)
            crop_path = output_dir / f"crop_{i + 1:03d}_{label}.jpg"
            crop_img.save(str(crop_path))
            print(
                f"\n  [{i + 1:2d}] label={label:18s} -> SKIP (crop saved: {crop_path})"
            )
            results.append(
                {
                    "index": i + 1,
                    "label": label,
                    "score": box["score"],
                    "coordinate": box["coordinate"],
                    "order": box.get("order"),
                    "prompt": None,
                    "recognition": None,
                    "skipped": True,
                    "saved_crop": str(crop_path),
                }
            )
            continue

        # 裁剪子区域
        crop_img = crop_region(image, box["coordinate"], padding=args.crop_padding)
        raw_crop_size = f"{crop_img.size[0]}x{crop_img.size[1]}"
        if args.save_crops:
            crop_path = output_dir / f"crop_{i + 1:03d}_{label}.jpg"
            crop_img.save(str(crop_path))

        # ── 将 crop pad 到 vision ONNX 期望的静态尺寸 ──
        if vis_h is not None and vis_w is not None:
            crop_img = pad_crop_to_vision_size(crop_img, vis_h, vis_w)

        print(
            f'\n  [{i + 1:2d}] label={label:18s} prompt="{prompt}" '
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
        status = "SKIP" if r.get("skipped") else "OK"
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
