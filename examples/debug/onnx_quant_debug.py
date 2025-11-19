import argparse
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import tabulate
import torch
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
    to_export_graph,
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


def convert_unsupported_dtype(tensor: Tensor) -> Tensor:
    """将不支持的数据类型转换为支持的数据类型"""
    if tensor.dtype == torch.uint16:
        # uint16转换为int32，因为uint16的范围是0-65535，可以用int32表示
        return tensor.to(torch.int32)
    elif tensor.dtype == torch.uint32:
        # uint32转换为int64
        return tensor.to(torch.int64)
    elif tensor.dtype == torch.uint64:
        # uint64转换为float64，因为int64可能无法表示所有uint64值
        return tensor.to(torch.float64)
    else:
        return tensor


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
                    # 转换不支持的数据类型
                    out_converted = convert_unsupported_dtype(out)
                    torch.save(out_converted.detach().cpu(), out_file)

        disable_outputs = super().run_node(n)
        out = None
        if isinstance(disable_outputs, QTensor):
            out = disable_outputs.dequantize()
        elif isinstance(disable_outputs, Tensor):
            out = disable_outputs
        if out is not None:
            out_file = Path(self.snap_dir) / f"{n.name}_float.pt"
            # 转换不支持的数据类型
            out_converted = convert_unsupported_dtype(out)
            torch.save(out_converted.detach().cpu(), out_file)

        return disable_outputs


def output_compare(output_names, pred_outputs, gt_outputs):
    errors_infos = []
    for output_name, pred_output, gt_output in zip(output_names, pred_outputs, gt_outputs):
        if isinstance(gt_output, np.ndarray):
            gt_output = torch.from_numpy(gt_output)
        gt_output = gt_output.to(pred_output.device)

        if gt_output.is_floating_point():
            # 将除了batch维度外的所有维度展平
            pred_flat = pred_output.reshape(pred_output.shape[0], -1)
            gt_flat = gt_output.reshape(gt_output.shape[0], -1)
            cos_simiarity = torch.cosine_similarity(pred_flat, gt_flat, dim=1).mean().item()
            mse_error = torch.mean((pred_output - gt_output) ** 2).item()
            abs_diff = (pred_output - gt_output).abs().max().item()
            max_v = gt_output.abs().max().item()
            if max_v != 0:
                max_rel_error = abs_diff / max_v
            else:
                max_rel_error = 0.0
        else:
            abs_diff: str = "/"
            cos_simiarity: str = "/"
            mse_error: str = "/"
            max_rel_error: str = "/"
        str_dtype = f"{pred_output.dtype} vs {gt_output.dtype}"
        errors_infos.append(
            [
                output_name,
                abs_diff,
                max_rel_error,
                cos_simiarity,
                mse_error,
                str_dtype,
                list(pred_output.shape),
                list(gt_output.shape),
                pred_output.shape == gt_output.shape,
            ]
        )
        del abs_diff
        del cos_simiarity
        del mse_error
        del max_rel_error

    headers = ["name", "Abs", "abs_relative", "cos_similarity", "MSE", "dtype", "onnx shape", "pred shape", "matched"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple", floatfmt=".6f")
    return errors_str


def main(args):
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
    logger.info(f"*************** config ***************\n{cfg.pretty_text}")
    logger.info(f"onnx: {onnx_file}")
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
        if dtype in [torch.float32, torch.float16]:
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
    ort_inputs: Dict[str, np.ndarray] = {name: input.cpu().numpy() for name, input in zip(input_names, inputs)}
    ort_outputs = ort_session.run(None, ort_inputs)

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

    errors_str = output_compare(output_names, fronted_outputs, ort_outputs)
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
    aligned_output = quant_aligned_snap_shot.run(*inputs)
    if isinstance(aligned_output, Tensor):
        aligned_output = [aligned_output]

    errors_str = output_compare(output_names, aligned_output, ort_outputs)
    logger.info(f"aligned vs onnxruntime errors:\n{errors_str}")

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
            if not (isinstance(aligned_tensor, Tensor) and isinstance(disable_tensor, Tensor)):
                continue

            if aligned_tensor.is_floating_point():
                aligned_tensor = aligned_tensor.to(torch.float32)
                disable_tensor = disable_tensor.to(torch.float32)

            pred_output = aligned_tensor
            gt_output = disable_tensor
            if gt_output.is_floating_point():
                # 将除了batch维度外的所有维度展平
                if disable_tensor.ndim > 0:
                    pred_flat = pred_output.reshape(pred_output.shape[0], -1)
                    gt_flat = gt_output.reshape(gt_output.shape[0], -1)
                else:
                    pred_flat = pred_output
                    gt_flat = gt_output
                cos_simiarity = torch.cosine_similarity(pred_flat, gt_flat, dim=-1).mean().item()
                mse_error = torch.mean((pred_output - gt_output) ** 2).item()
                abs_diff = (pred_output - gt_output).abs().max().item()
                max_v = gt_output.abs().max().item()
                if max_v != 0:
                    max_rel_error = abs_diff / max_v
                else:
                    max_rel_error = 0.0
            else:
                cos_simiarity: str = "/"
                mse_error: str = "/"
                abs_diff: str = "/"
                max_rel_error: str = "/"

            min_v = gt_output.min().item()
            max_v = gt_output.max().item()
            errors_infos.append(
                [
                    node.name,
                    abs_diff,
                    max_rel_error,
                    cos_simiarity,
                    mse_error,
                    min_v,
                    max_v,
                    min_v < -65504 or max_v > 65504,
                    op_type,
                ]
            )
            del abs_diff
            del cos_simiarity
            del mse_error
            del max_rel_error

    headers = ["name", "Abs", "abs_relative", "cos_similarity", "MSE", "min", "max", "overflow", "op_type"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple", floatfmt=".6f")
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
    with torch.no_grad():
        disable_quant_outputs = snapshot_interpreter.run(*inputs)

    if isinstance(disable_quant_outputs, Tensor):
        disable_quant_outputs = [disable_quant_outputs]

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

            if not (isinstance(aligned_tensor, Tensor) and isinstance(disable_tensor, Tensor)):
                continue

            if aligned_tensor.is_floating_point():
                aligned_tensor = aligned_tensor.to(torch.float32)
                disable_tensor = disable_tensor.to(torch.float32)

            pred_output = aligned_tensor
            gt_output = disable_tensor

            if gt_output.is_floating_point():
                # 将除了batch维度外的所有维度展平
                if disable_tensor.ndim > 0:
                    pred_flat = pred_output.reshape(pred_output.shape[0], -1)
                    gt_flat = gt_output.reshape(gt_output.shape[0], -1)
                else:
                    pred_flat = pred_output
                    gt_flat = gt_output
                cos_simiarity = torch.cosine_similarity(pred_flat, gt_flat, dim=-1).mean().item()
                mse_error = torch.mean((pred_output - gt_output) ** 2).item()
                abs_diff = (pred_output - gt_output).abs().max().item()
                max_v = gt_output.abs().max().item()
                if max_v != 0:
                    max_rel_error = abs_diff / max_v
                else:
                    max_rel_error = 0.0
            else:
                abs_diff: str = "/"
                cos_simiarity: str = "/"
                mse_error: str = "/"
                max_rel_error: str = "/"

            min_v = gt_output.min().item()
            max_v = gt_output.max().item()
            errors_infos.append(
                [
                    node.name,
                    abs_diff,
                    max_rel_error,
                    cos_simiarity,
                    mse_error,
                    min_v,
                    max_v,
                    min_v < -65504 or max_v > 65504,
                    op_type,
                ]
            )
            del abs_diff
            del cos_simiarity
            del mse_error
            del max_rel_error

    headers = ["name", "Abs", "abs_relative", "cos_similarity", "MSE", "min", "max", "overflow", "op_type"]
    errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple", floatfmt=".6f")
    logger.info(f"quanted_aligned layerwise errors:\n{errors_str}")
    df = pd.DataFrame(np.array(errors_infos))
    df.columns = headers
    df.to_csv(work_dir / "layerwise_quantizer_error.csv", index=True)
    del df

    # 校验disable_quant输出和前端输出是否一致
    # gt_outputs = fronted_outputs
    # pred_outputs = disable_quant_outputs

    # errors_infos = []
    # for output_name, pred_output, gt_output in zip(output_names, pred_outputs, gt_outputs):
    #     if isinstance(gt_output, np.ndarray):
    #         gt_output = torch.from_numpy(gt_output)
    #     gt_output = gt_output.to(pred_output.device)

    #     if gt_output.is_floating_point():
    #         # 将除了batch维度外的所有维度展平
    #         pred_flat = pred_output.reshape(pred_output.shape[0], -1)
    #         gt_flat = gt_output.reshape(gt_output.shape[0], -1)
    #         cos_simiarity = torch.cosine_similarity(pred_flat, gt_flat, dim=1).mean().item()
    #         mse_error = torch.mean((pred_output - gt_output) ** 2).item()
    #         abs_diff = (pred_output - gt_output).abs().max().item()
    #         max_v = gt_output.abs().max().item()
    #         if max_v != 0:
    #             max_rel_error = abs_diff / max_v
    #         else:
    #             max_rel_error = 0.0
    #     else:
    #         abs_diff: str = "/"
    #         cos_simiarity: str = "/"
    #         mse_error: str = "/"
    #         max_rel_error: str = "/"
    #     str_dtype = f"{pred_output.dtype} vs {gt_output.dtype}"
    #     errors_infos.append(
    #         [
    #             output_name,
    #             abs_diff,
    #             max_rel_error,
    #             cos_simiarity,
    #             mse_error,
    #             str_dtype,
    #             list(pred_output.shape),
    #             list(gt_output.shape),
    #             pred_output.shape == gt_output.shape,
    #         ]
    #     )
    #     del abs_diff
    #     del cos_simiarity
    #     del mse_error
    #     del max_rel_error

    # headers = ["name", "Abs", "abs_relative", "cos_similarity", "MSE", "dtype", "onnx shape", "pred shape", "matched"]
    # errors_str = tabulate.tabulate(errors_infos, headers=headers, tablefmt="simple", floatfmt=".6f")
    errors_str = output_compare(output_names, disable_quant_outputs, fronted_outputs)
    logger.info(f"disable_quant vs front_model errors:\n{errors_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default="/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/resnet50.onnx", help="onnx file"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--config", default="./configs/xh2a/base_xh2a.py", help="config file")
    parser.add_argument("--compare-all-nodes", action="store_true", help="compare all nodes")
    args = parser.parse_args()
    main(args)
