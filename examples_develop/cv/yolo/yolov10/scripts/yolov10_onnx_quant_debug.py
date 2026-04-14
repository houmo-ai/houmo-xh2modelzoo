# Copyright 2025 HOUMO AI
#
# File: yolov10_onnx_quant_debug.py
# Description:
#   Example script: cv/yolo/yolov10/scripts/yolov10_onnx_quant_debug.py
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT



import argparse
import re
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import tabulate
import torch
import torch.nn.functional as F
import yaml
from onnx import TensorProto
from torch import Tensor, is_floating_point
from torch.fx import Node
from xhquant import disable_quant
from xhquant import nn as xhnn
from xhquant.api import (
    Config,
    FrontendType,
    FXInterpreter,
    PrecisionMode,
    QBaseModule,
    QTensor,
    QuantizerState,
    get_root_logger,
    ptq_quantize,
    query_device,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.debug import GraphSnapshot
from xhquant.utils import map_aggregate

TENSOR_TYPE_TO_TORCH_TYPE = {
    int(TensorProto.FLOAT): torch.float32,
    int(TensorProto.UINT8): torch.uint8,
    int(TensorProto.INT8): torch.int8,
    int(TensorProto.INT16): torch.int16,
    int(TensorProto.INT32): torch.int32,
    int(TensorProto.INT64): torch.int64,
    int(TensorProto.BOOL): torch.bool,
    int(TensorProto.FLOAT16): torch.float16,
    int(TensorProto.DOUBLE): torch.float64,
    int(TensorProto.COMPLEX64): torch.complex64,
    int(TensorProto.COMPLEX128): torch.complex128,
}


def get_onnx_node_out(onnx_model: onnx.ModelProto):
    native_out_names = [output.name for output in onnx_model.graph.output]
    out_names = []
    tensors = {v_info.name: v_info for v_info in onnx_model.graph.value_info}
    for i, node in enumerate(onnx_model.graph.node):
        for node_out_name in node.output:
            if node_out_name in native_out_names:
                continue
            out_names.append(node_out_name)
    for out_name in out_names:
        if out_name in tensors:
            intermediate_layer_value_info = tensors[out_name]
        else:
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = out_name
        onnx_model.graph.output.append(intermediate_layer_value_info)

    return onnx_model


def kl_div_manual(p: Tensor, q: Tensor):
    # 确保p和q都是概率分布（和为1）
    p_normalized = p / p.sum()
    q_normalized = q / q.sum()

    # 手动计算KL散度公式: sum(p * log(p/q))
    kl_div = torch.sum(p_normalized * torch.log(p_normalized / q_normalized))
    return kl_div.item()


class LayerwiseQuantizeErrorSnapshot(FXInterpreter):
    def __init__(
        self,
        *args,
        snap_dir: str = None,
    ):
        super().__init__(*args)
        self.snap_dir = snap_dir
        Path(self.snap_dir).mkdir(exist_ok=True, parents=True)

    def run_node(self, n: Node) -> Any:
        if n.op == "call_module":
            m = self.module.get_submodule(n.target)
            if isinstance(m, QBaseModule) and not isinstance(m, xhnn.InputStub):
                m._old_forward = m.forward

                def _quanted_forward(self, *args, **kwargs):
                    old_quant_state = m.quant_state
                    old_precision_mode = m.precision_mode
                    assert old_quant_state == QuantizerState.NONE
                    m.quant_state = QuantizerState.Fixed
                    m.precision_mode = PrecisionMode.ALIGNED
                    args = map_aggregate(args, lambda x: QTensor(x) if isinstance(x, Tensor) else x)
                    kwargs = map_aggregate(kwargs, lambda x: QTensor(x) if isinstance(x, Tensor) else x)
                    outputs = m._old_forward(*args, **kwargs)
                    m.quant_state = old_quant_state
                    m.precision_mode = old_precision_mode
                    return outputs

                m.forward = MethodType(_quanted_forward, m)

                output = super().run_node(n)
                m.forward = m._old_forward
                del m._old_forward

                out = None
                if isinstance(output, QTensor):
                    out = output.dequantize()
                elif isinstance(output, Tensor):
                    out = output
                if out is not None:
                    out_file = Path(self.snap_dir) / f"{n.name}.pt"
                    torch.save(out.detach().cpu(), out_file)

        disable_outputs = super().run_node(n)
        out = None
        if isinstance(disable_outputs, QTensor):
            out = disable_outputs.dequantize()
        elif isinstance(disable_outputs, Tensor):
            out = disable_outputs
        if out is not None:
            out_file = Path(self.snap_dir) / f"{n.name}_float.pt"
            torch.save(out.detach().cpu(), out_file)

        return disable_outputs


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
    # 图像预处理
    h, w = image.shape[:2]
    target_w, target_h = input_img_size
    new_size, scale_factor = rescale_size((w, h), input_img_size, return_scale=True)
    resized_img = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    resized_img = torch.from_numpy(resized_img).permute(2, 0, 1).contiguous()
    resized_img = resized_img.float() / 255.0
    pad_h = target_h - h
    pad_w = target_w - w
    pad_img = F.pad(
        resized_img,
        (
            0,
            pad_w,
            0,
            pad_h,
        ),
        "constant",
        0,
    )
    pad_img = pad_img.unsqueeze(0)
    return {
        "metas": [
            {
                "ori_shape": image.shape[-2:],
                "scale_factor": (scale_factor, scale_factor),
            }
        ],
        "inputs": pad_img,
    }


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.
        end2end (bool): If the model doesn't require NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            assert False
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output


def show_result(image_file, batch_nn_out):
    # 加载类别名称
    coco_cfg_file = str(Path(__file__).parent / "scripts" / "coco8.yaml")
    classes = yaml_load(coco_cfg_file)["names"]

    preds = non_max_suppression(
        batch_nn_out,
        0.25,
        0.45,
        None,
        None,
        max_det=300,
        nc=len(classes),
        end2end=False,
        rotated=False,
    )
    pred = preds[0]
    image_fname = Path(image_file).stem
    image = cv2.imread(image_file)

    for box in pred:
        x1, y1, x2, y2, score, label = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def main(args):
    input_shape = (640, 640)
    # 读取图像并进行预处理
    image_file = args.image
    image_fname = Path(image_file).stem
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = preprocess_image(image, input_shape)
    net_inputs = results["inputs"]

    cfg = Config.fromfile(args.config)
    target_device = query_device(cfg.target_device)
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    cfg_name = f"{onnx_name}_{target_device.name}"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    cfg.work_dir = str(work_dir)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"

    xhquant_init(log_file, debug=args.debug)

    device = torch.device("cuda:0")
    logger = get_root_logger()
    logger.info(f"Target device: {target_device}")
    onnx_model = onnx.load(onnx_file)

    initializer_names = [init.name for init in onnx_model.graph.initializer]
    inilializer_as_inputs: List[str] = []
    for input in onnx_model.graph.input:
        if input.name in initializer_names:
            inilializer_as_inputs.append(input)
    for input in inilializer_as_inputs:
        onnx_model.graph.input.remove(input)

    inputs: List[Tensor] = []
    input_names: List[str] = []
    input_infos = []
    for idx, input in enumerate(onnx_model.graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input.type.tensor_type.shape.dim]
        dtype = TENSOR_TYPE_TO_TORCH_TYPE[input.type.tensor_type.elem_type]
        input_infos.append([idx, input.name, shape, dtype])
        input_names.append(input.name)
        if dtype == torch.float32:
            inputs.append(torch.randn(shape, dtype=dtype))
        elif dtype in [torch.int32, torch.int64]:
            inputs.append(torch.randint(0, 10, shape, dtype=dtype))
        elif dtype == torch.bool:
            inputs.append(torch.randint(0, 2, shape, dtype=dtype))
        else:
            raise NotImplementedError(f"dtype {dtype} not supported")
    headers = ["idx", "name", "shape", "dtype"]
    inputs_str = tabulate.tabulate(input_infos, headers=headers, tablefmt="simple")

    logger.info(f"onnx inputs:\n{inputs_str}")
    inputs = [net_inputs]

    # 获取onnx所有节点的输出
    if args.compare_all_nodes:
        onnx_model = get_onnx_node_out(onnx_model)

    output_infos = []
    for idx, output in enumerate(onnx_model.graph.output):
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in output.type.tensor_type.shape.dim]
        dtype = TENSOR_TYPE_TO_TORCH_TYPE[output.type.tensor_type.elem_type]
        output_infos.append([idx, output.name, shape, dtype])
    headers = ["idx", "name", "shape", "dtype"]
    outputs_str = tabulate.tabulate(output_infos, headers=headers, tablefmt="simple")
    logger.info(f"onnx outputs:\n{outputs_str}")

    providers = []
    provider_options = []
    if torch.cuda.is_available():
        providers.append("CUDAExecutionProvider")
        provider_options.append(
            {
                "device_id": 0,
                # "fp16": True,
            }
        )
    else:
        providers.append("CPUExecutionProvider")
    so = ort.SessionOptions()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx_file = Path(tmp_dir) / f"{onnx_name}.onnx"
        onnx.save(onnx_model, str(tmp_onnx_file))
        ort_session = ort.InferenceSession(tmp_onnx_file, so, providers=providers, provider_options=provider_options)
    ort_inputs: Dict[str, np.ndarray] = {name: input.float().numpy() for name, input in zip(input_names, inputs)}
    ort_outputs = ort_session.run(None, ort_inputs)
    image_vis = show_result(image_file, torch.from_numpy(ort_outputs[0]))

    out_dir = Path("work_dirs/yolov10m")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{image_fname}_vis_ort.jpg"), image_vis)

    ## 先评测onnx解析的输出和onnxruntime的输出是否一致
    fronted_graph_module = to_frontend_graph(onnx_model, FrontendType.ONNX, inputs, enable_fuse=True)
    fronted_graph_module.eval()
    fronted_graph_module.to(device)
    inputs = map_aggregate(inputs, lambda x: x.to(device))

    with torch.no_grad():
        fronted_outputs = fronted_graph_module(*inputs)
    if isinstance(fronted_outputs, Tensor):
        fronted_outputs = [fronted_outputs]

    output_names = [output.name for output in onnx_model.graph.output]
    assert len(output_names) == len(fronted_outputs)
    assert len(output_names) == len(ort_outputs)

    gt_outputs = ort_outputs
    pred_outputs = fronted_outputs

    errors_infos = []
    for output_name, pred_output, gt_output in zip(output_names, pred_outputs, gt_outputs):
        if isinstance(gt_output, np.ndarray):
            gt_output = torch.from_numpy(gt_output)
        gt_output = gt_output.to(pred_output.device)

        if gt_output.is_floating_point():
            abs_diff = (pred_output - gt_output).abs().max().item()
            cos_simiarity = torch.cosine_similarity(pred_output, gt_output, dim=-1).mean().item()
            mse_error = torch.mean((pred_output - gt_output) ** 2).item()
        else:
            cos_simiarity: str = "/"
            mse_error: str = "/"
            abs_diff: str = "/"
        str_dtype = f"{pred_output.dtype} vs {gt_output.dtype}"
        errors_infos.append(
            [
                output_name,
                abs_diff,
                cos_simiarity,
                mse_error,
                str_dtype,
                list(pred_output.shape),
                list(gt_output.shape),
                pred_output.shape == gt_output.shape,
            ]
        )

    headers = ["name", "Abs", "cos_similarity", "MSE", "dtype", "onnx shape", "pred shape", "matched"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple")
    logger.info(f"fronted vs onnxruntime errors:\n{errors_str}")

    ## 评测量化前后的输出是否一致
    quant_config = cfg.quant_config
    quanted_graph_module = to_quant_graph(fronted_graph_module, target_device.name, quant_config)

    # 保存quanted_graph_module在disable_quant的情况下的输出, 用于后续的量化误差分析
    quanted_graph_module.to(torch.float32).to(device)
    inputs = map_aggregate(inputs, lambda x: x.to(device).to(torch.float32) if is_floating_point(x) else x.to(device))
    quanted_graph_module.disable_quant()
    disable_snap_dir = work_dir / "quanted_graph_module_disable_quant"
    disable_snap_shot = GraphSnapshot(quanted_graph_module, disable_snap_dir, "disable_quant")
    disable_snap_shot.run(*inputs)

    ptq_quantize(quanted_graph_module, [inputs], PrecisionMode.ALIGNED, [device], auto_release_unused_parameters=False)
    aligend_snap_dir = work_dir / "quanted_graph_module_quant_aligned"
    quant_aligned_snap_shot = GraphSnapshot(quanted_graph_module, aligend_snap_dir, "quant_aligned")
    inputs = map_aggregate(inputs, lambda x: x.to(device).to(torch.float16) if is_floating_point(x) else x.to(device))
    aligned_outs = quant_aligned_snap_shot.run(*inputs)

    image_vis = show_result(image_file, aligned_outs)
    cv2.imwrite(str(out_dir / f"{image_fname}_vis_aligned.jpg"), image_vis)

    ## 逐层计算量化累计误差
    errors_infos = []
    for node in quanted_graph_module.graph.nodes:
        node_name = node.name
        aligned_tensor_file = aligend_snap_dir / f"{node_name}.pt"
        disable_tensor_file = disable_snap_dir / f"{node_name}.pt"
        if aligned_tensor_file.exists() and disable_tensor_file.exists():
            op_type = node.op
            if node.op == "call_module":
                m = quanted_graph_module.get_submodule(node.target)
                if hasattr(m, "op_type"):
                    op_type = m.op_type
                else:
                    op_type = m.__class__.__name__

            aligned_tensor = torch.load(aligned_tensor_file, weights_only=True)
            disable_tensor = torch.load(disable_tensor_file, weights_only=True)
            if aligned_tensor.is_floating_point():
                aligned_tensor = aligned_tensor.to(torch.float32)
                disable_tensor = disable_tensor.to(torch.float32)

            if disable_tensor.is_floating_point():
                abs_diff = (disable_tensor - aligned_tensor).abs().max().item()
                cos_simiarity = (
                    torch.cosine_similarity(aligned_tensor.flatten(), disable_tensor.flatten(), dim=-1).mean().item()
                )
                mse_error = torch.mean((aligned_tensor - disable_tensor) ** 2).item()
            else:
                cos_simiarity: str = "/"
                mse_error: str = "/"
                abs_diff: str = "/"

            min_v = disable_tensor.min().item()
            max_v = disable_tensor.max().item()
            errors_infos.append(
                [
                    node.name,
                    abs_diff,
                    cos_simiarity,
                    mse_error,
                    disable_tensor.min(),
                    disable_tensor.max(),
                    min_v < -65504 or max_v > 65504,
                    op_type,
                ]
            )

    headers = ["name", "Abs", "cos_similarity", "MSE", "min", "max", "overflow", "op_type"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple")
    logger.info(f"quanted_aligned accumulated errors:\n{errors_str}")

    df = pd.DataFrame(np.array(errors_infos))
    df.columns = headers
    df.to_csv(work_dir / "layerwise_accumulated_quantizer_error.csv", index=True)
    del df

    ## 逐层计算量化误差:输入与float模型一致，相同输入下，量化误差
    layerwise_error_dir = work_dir / "layerwise_quantize_error_snapshot"
    errors_infos = []
    assert quanted_graph_module.is_fixed(), "quanted_graph_module is not fixed"
    quanted_graph_module.disable_quant()
    snapshot_interpreter = LayerwiseQuantizeErrorSnapshot(quanted_graph_module, snap_dir=str(layerwise_error_dir))
    disable_quant_outputs = snapshot_interpreter.run(*inputs)

    errors_infos = []
    for node in quanted_graph_module.graph.nodes:
        node_name = node.name
        quanted_tensor_file = layerwise_error_dir / f"{node_name}.pt"
        disable_tensor_file = layerwise_error_dir / f"{node_name}_float.pt"
        if quanted_tensor_file.exists() and disable_tensor_file.exists():
            op_type = node.op
            if node.op == "call_module":
                m = quanted_graph_module.get_submodule(node.target)
                if hasattr(m, "op_type"):
                    op_type = m.op_type
                else:
                    op_type = m.__class__.__name__
            aligned_tensor = torch.load(quanted_tensor_file, weights_only=True)
            disable_tensor = torch.load(disable_tensor_file, weights_only=True)
            if aligned_tensor.is_floating_point():
                aligned_tensor = aligned_tensor.to(torch.float32)
                disable_tensor = disable_tensor.to(torch.float32)

            if disable_tensor.is_floating_point():
                abs_diff = (disable_tensor - aligned_tensor).abs().max().item()
                cos_simiarity = (
                    torch.cosine_similarity(aligned_tensor.flatten(), disable_tensor.flatten(), dim=-1).mean().item()
                )
                mse_error = torch.mean((aligned_tensor - disable_tensor) ** 2).item()
            else:
                abs_diff: str = "/"
                cos_simiarity: str = "/"
                mse_error: str = "/"

            min_v = disable_tensor.min().item()
            max_v = disable_tensor.max().item()
            errors_infos.append(
                [
                    node.name,
                    abs_diff,
                    cos_simiarity,
                    mse_error,
                    disable_tensor.min(),
                    disable_tensor.max(),
                    min_v < -65504 or max_v > 65504,
                    op_type,
                ]
            )

    headers = ["name", "Abs", "cos_similarity", "MSE", "min", "max", "overflow", "op_type"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple")
    logger.info(f"quanted_aligned layerwise errors:\n{errors_str}")
    df = pd.DataFrame(np.array(errors_infos))
    df.columns = headers
    df.to_csv(work_dir / "layerwise_quantizer_error.csv", index=True)
    del df

    # 校验disable_quant输出和前端输出是否一致
    gt_outputs = fronted_outputs
    pred_outputs = disable_quant_outputs

    errors_infos = []
    for output_name, pred_output, gt_output in zip(output_names, pred_outputs, gt_outputs):
        if isinstance(gt_output, np.ndarray):
            gt_output = torch.from_numpy(gt_output)
        gt_output = gt_output.to(pred_output.device)
        abs_diff = (pred_output - gt_output).abs().max().item()

        if gt_output.is_floating_point():
            cos_simiarity = torch.cosine_similarity(pred_output, gt_output, dim=-1).mean().item()
            mse_error = torch.mean((pred_output - gt_output) ** 2).item()
        else:
            cos_simiarity = 0.0
            mse_error = 0.0
        str_dtype = f"{pred_output.dtype} vs {gt_output.dtype}"
        errors_infos.append(
            [
                output_name,
                abs_diff,
                cos_simiarity,
                mse_error,
                str_dtype,
                list(pred_output.shape),
                list(gt_output.shape),
                pred_output.shape == gt_output.shape,
            ]
        )

    headers = ["name", "Abs", "cos_similarity", "MSE", "dtype", "onnx shape", "pred shape", "matched"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple")
    logger.info(f"errors:\n{errors_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/models/yolo/yolov10m.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    parser.add_argument("--config", default="./configs/xh2a/base_xh2a.py", help="config file")
    parser.add_argument("--compare-all-nodes", action="store_true", help="compare all nodes")
    args = parser.parse_args()
    main(args)
