#!/usr/bin/env python3
# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""COCO bbox mAP 评测脚本，支持 Torch / ONNX / frontend / quant graph / HMONNX 后端。

用法:
    # 评测 HMONNX 量化模型
    python examples/cv/dinov3/3.hmonnx_eval_coco.py \
        --hmonnx mixed_precision_search/auto_run/hmonnx/xxx_XH2a.onnx \
        --images-dir /data01/datasets/coco/val2017 \
        --annotations /data01/datasets/coco/annotations/instances_val2017.json \
        --backend hmonnx \
        --device cuda \
        --sample-size 200

    # 评测 Torch 原始模型
    python examples/cv/dinov3/3.hmonnx_eval_coco.py \
        --images-dir /data01/datasets/coco/val2017 \
        --annotations /data01/datasets/coco/annotations/instances_val2017.json \
        --backend torch \
        --device cuda \
        --sample-size 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dinov3_common import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_DATA_CACHE_DIR,
    DEFAULT_ONNX,
    DEFAULT_QUANT_TYPE,
    build_quant_config,
    detect_onnx_io,
    load_mixed_precision_config,
    load_lightly_model,
    map_contiguous_label_to_category_id,
    make_hmonnx_runner,
    make_onnx_runner,
    outputs_to_coco_rows,
    preprocess_image_batch,
    resolve_runtime_device,
    run_torch_with_original_size,
    scale_resized_boxes_to_original,
    select_coco_image_ids,
)


QUANTGRAPH_TORCH_SIGMOID_PRESETS = {
    "decoder-bbox-refine-top3": r"^(_decoder_decoder_sigmoid_2|_decoder_decoder_sigmoid_3|_decoder_decoder_sigmoid_4)$",
}

DEFAULT_COCO_IMG = Path("/data01/datasets/coco2017/val2017")
DEFAULT_COCO_ANN = Path("/data01/datasets/coco2017/annotations/instances_val2017.json")
DEFAULT_HMONNX = (
    SCRIPT_DIR
    / "mixed_precision_search"
    / "auto_search_sample8_top40_v2"
    / "hmonnx"
    / "dinov3-vitt16-ltdetr-coco_is640_mix_search_w8a8h1_sefp_XH2a.onnx"
)
DEFAULT_HMONNX_EVAL_DIR = SCRIPT_DIR / "coco_eval" / "hmonnx_mix77_sample200_seed4200"


def make_frontend_runner(onnx_path: str | Path, *, batch_size: int, image_size: int, device: str):
    """Create a runner for the xhquant frontend graph before quantization."""
    import numpy as np
    import torch
    from xhquant.common.types import FrontendType
    from xhquant.frontend import to_frontend_graph

    input_name, input_shape, _output_names = detect_onnx_io(
        onnx_path,
        batch_size=batch_size,
        image_size=image_size,
    )
    example_input = torch.randn(*input_shape, dtype=torch.float32)
    frontend_graph = to_frontend_graph(
        str(Path(onnx_path).expanduser().resolve()),
        FrontendType.ONNX,
        [example_input],
        input_names=[input_name],
    )
    runtime_device = resolve_runtime_device(device)
    frontend_graph.eval().to(runtime_device)

    def run(batch):
        with torch.no_grad():
            output = frontend_graph(batch.to(device=runtime_device, dtype=torch.float32))
        if not isinstance(output, (tuple, list)):
            output = (output,)
        return tuple(
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
            for item in output
        )

    return run


def make_quantgraph_runner(
    onnx_path: str | Path,
    *,
    batch_size: int,
    image_size: int,
    device: str,
    quant_type: str,
    quantgraph_mode: str,
    mixed_precision_config: str | Path | None,
    input_enable_fp32: bool,
    output_enable_fp32: bool,
    torch_softmax_pattern: str | None,
    torch_matmul_pattern: str | None,
    torch_bypass_groups: list[str] | None,
    torch_linear_pattern: str | None,
    torch_layernorm_pattern: str | None,
    torch_conv_pattern: str | None,
    torch_gelu_pattern: str | None,
    torch_sigmoid_presets: list[str] | None,
    torch_sigmoid_pattern: str | None,
    torch_sigmoid_dtype: str,
    torch_add_pattern: str | None,
    torch_mul_pattern: str | None,
):
    """Create a runner for the xhquant QuantGraph stage before HMONNX export."""
    import argparse

    import numpy as np
    import torch
    from xhquant.api import DeviceType
    from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model
    from xhquant.common.types import FrontendType, PrecisionMode
    from xhquant.quantization import ptq_quantize

    def dequantize_if_needed(value):
        return value.dequantize() if hasattr(value, "dequantize") else value

    def quantize_output_if_needed(module, value):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            return value
        value = value.to(torch.float16)
        if getattr(module, "is_quanting", False) and hasattr(module, "o_quantizer"):
            return module.o_quantizer(value)
        return value

    def binary_float_inputs(x1, x2):
        x1 = dequantize_if_needed(x1)
        x2 = dequantize_if_needed(x2)
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            return x1, x2, False
        if not x1.is_floating_point() and not x2.is_floating_point():
            return x1, x2, False
        return x1.float(), x2.float(), True

    def patch_softmax(module):
        def torch_softmax_forward(self, x):
            x = dequantize_if_needed(x)
            return torch.softmax(x.float(), dim=self.dim)

        return types.MethodType(torch_softmax_forward, module)

    def patch_softmax_qout(module):
        def torch_softmax_qout_forward(self, x):
            x = dequantize_if_needed(x)
            out = torch.softmax(x.float(), dim=self.dim)
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_softmax_qout_forward, module)

    def patch_matmul(module):
        def torch_matmul_forward(self, x, y):
            x = dequantize_if_needed(x)
            y = dequantize_if_needed(y)
            return torch.matmul(x.float(), y.float())

        return types.MethodType(torch_matmul_forward, module)

    def patch_matmul_qout(module):
        def torch_matmul_qout_forward(self, x, y):
            x = dequantize_if_needed(x)
            y = dequantize_if_needed(y)
            out = torch.matmul(x.float(), y.float())
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_matmul_qout_forward, module)

    def patch_matmul_fp16(module):
        def torch_matmul_fp16_forward(self, x, y):
            x = dequantize_if_needed(x)
            y = dequantize_if_needed(y)
            return torch.matmul(x.float(), y.float()).half()

        return types.MethodType(torch_matmul_fp16_forward, module)

    def patch_linear(module):
        def torch_linear_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            bias = self.bias.float() if self.bias is not None else None
            return F.linear(x.float(), self.weight.float(), bias)

        return types.MethodType(torch_linear_forward, module)

    def patch_linear_qout(module):
        def torch_linear_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            bias = self.bias.float() if self.bias is not None else None
            out = F.linear(x.float(), self.weight.float(), bias)
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_linear_qout_forward, module)

    def patch_layernorm(module):
        def torch_layernorm_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)

        return types.MethodType(torch_layernorm_forward, module)

    def patch_layernorm_qout(module):
        def torch_layernorm_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            out = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_layernorm_qout_forward, module)

    def patch_conv2d(module):
        def torch_conv2d_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            bias = self.bias.float() if self.bias is not None else None
            return F.conv2d(
                x.float(),
                self.weight.float(),
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        return types.MethodType(torch_conv2d_forward, module)

    def patch_conv2d_qout(module):
        def torch_conv2d_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            bias = self.bias.float() if self.bias is not None else None
            out = F.conv2d(
                x.float(),
                self.weight.float(),
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_conv2d_qout_forward, module)

    def patch_gelu(module):
        def torch_gelu_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            approximate = "tanh" if getattr(self, "approximate", "none") == "tanh" else "none"
            return F.gelu(x.float(), approximate=approximate)

        return types.MethodType(torch_gelu_forward, module)

    def patch_gelu_qout(module):
        def torch_gelu_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            approximate = "tanh" if getattr(self, "approximate", "none") == "tanh" else "none"
            out = F.gelu(x.float(), approximate=approximate)
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_gelu_qout_forward, module)

    def patch_relu_qout(module):
        def torch_relu_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            out = F.relu(x.float())
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_relu_qout_forward, module)

    def patch_silu_qout(module):
        def torch_silu_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            out = F.silu(x.float())
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_silu_qout_forward, module)

    def patch_sigmoid_qout(module):
        def torch_sigmoid_qout_forward(self, x):
            x = dequantize_if_needed(x)
            out = torch.sigmoid(x.float())
            if torch_sigmoid_dtype == "fp16":
                out = out.half()
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_sigmoid_qout_forward, module)

    def patch_add(module):
        def torch_add_forward(self, x1, x2=None):
            x1 = dequantize_if_needed(x1)
            x2 = dequantize_if_needed(x2)
            return x1.float() + x2.float()

        return types.MethodType(torch_add_forward, module)

    def patch_add_qout(module):
        def torch_add_qout_forward(self, x1, x2=None):
            x1, x2, is_float = binary_float_inputs(x1, x2)
            out = x1 + x2
            return quantize_output_if_needed(self, out) if is_float else out

        return types.MethodType(torch_add_qout_forward, module)

    def patch_mul(module):
        def torch_mul_forward(self, x1, x2=None):
            x1 = dequantize_if_needed(x1)
            x2 = dequantize_if_needed(x2)
            return x1.float() * x2.float()

        return types.MethodType(torch_mul_forward, module)

    def patch_mul_qout(module):
        def torch_mul_qout_forward(self, x1, x2=None):
            x1, x2, is_float = binary_float_inputs(x1, x2)
            out = x1 * x2
            return quantize_output_if_needed(self, out) if is_float else out

        return types.MethodType(torch_mul_qout_forward, module)

    def patch_sub_qout(module):
        def torch_sub_qout_forward(self, x1, x2=None):
            x1, x2, is_float = binary_float_inputs(x1, x2)
            out = x1 - x2
            return quantize_output_if_needed(self, out) if is_float else out

        return types.MethodType(torch_sub_qout_forward, module)

    def patch_div_qout(module):
        def torch_div_qout_forward(self, x1, x2=None):
            x1, x2, is_float = binary_float_inputs(x1, x2)
            out = x1 / x2
            return quantize_output_if_needed(self, out) if is_float else out

        return types.MethodType(torch_div_qout_forward, module)

    def patch_maxpool2d_qout(module):
        def torch_maxpool2d_qout_forward(self, x):
            import torch.nn.functional as F

            x = dequantize_if_needed(x)
            out = F.max_pool2d(x.float(), self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_maxpool2d_qout_forward, module)

    def patch_grid_sample_qout(module):
        def torch_grid_sample_qout_forward(self, input, grid):
            import torch.nn.functional as F

            input = dequantize_if_needed(input)
            grid = dequantize_if_needed(grid)
            out = F.grid_sample(
                input.float(),
                grid.float(),
                mode=self.mode,
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_grid_sample_qout_forward, module)

    def patch_reduce_sum_qout(module):
        def torch_reduce_sum_qout_forward(self, x):
            x = dequantize_if_needed(x)
            axes = [self.axes] if isinstance(self.axes, int) else self.axes
            out = torch.sum(x.float(), dim=tuple(axes), keepdim=bool(self.keepdims))
            return quantize_output_if_needed(self, out)

        return types.MethodType(torch_reduce_sum_qout_forward, module)

    bypass_group_specs = {
        "backbone_qkv": [("linear", r"^_backbone_blocks_[0-9]+_attn_qkv_mat_mul$")],
        "self_attn_qkv": [(
            "linear",
            r"^(_encoder_encoder_0_layers_0_self_attn_mat_mul(_[0-2])?|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul(_[0-2])?)$",
        )],
        "qkv": [(
            "linear",
            r"^(_backbone_blocks_[0-9]+_attn_qkv_mat_mul|"
            r"_encoder_encoder_0_layers_0_self_attn_mat_mul(_[0-2])?|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul(_[0-2])?)$",
        )],
        "qkv_qout": [(
            "linear_qout",
            r"^(_backbone_blocks_[0-9]+_attn_qkv_mat_mul|"
            r"_encoder_encoder_0_layers_0_self_attn_mat_mul(_[0-2])?|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul(_[0-2])?)$",
        )],
        "q": [(
            "linear",
            r"^(_encoder_encoder_0_layers_0_self_attn_mat_mul|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul)$",
        )],
        "k": [(
            "linear",
            r"^(_encoder_encoder_0_layers_0_self_attn_mat_mul_1|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_1)$",
        )],
        "v": [(
            "linear",
            r"^(_encoder_encoder_0_layers_0_self_attn_mat_mul_2|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_2)$",
        )],
        "o": [(
            "linear",
            r"^(_backbone_blocks_[0-9]+_attn_proj_mat_mul|"
            r"_encoder_encoder_0_layers_0_self_attn_gemm|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_gemm)$",
        )],
        "o_qout": [(
            "linear_qout",
            r"^(_backbone_blocks_[0-9]+_attn_proj_mat_mul|"
            r"_encoder_encoder_0_layers_0_self_attn_gemm|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_gemm)$",
        )],
        "up": [("linear", r"^(_backbone_blocks_[0-9]+_mlp_fc1_mat_mul|.*_linear1_mat_mul)$")],
        "down": [("linear", r"^(_backbone_blocks_[0-9]+_mlp_fc2_mat_mul|.*_linear2_mat_mul)$")],
        "up_qout": [("linear_qout", r"^(_backbone_blocks_[0-9]+_mlp_fc1_mat_mul|.*_linear1_mat_mul)$")],
        "down_qout": [("linear_qout", r"^(_backbone_blocks_[0-9]+_mlp_fc2_mat_mul|.*_linear2_mat_mul)$")],
        "gate": [("linear", r"gate")],
        "qk_matmul": [(
            "matmul",
            r"^(_backbone_blocks_[0-9]+_attn_mat_mul|"
            r"_encoder_encoder_0_layers_0_self_attn_mat_mul_3|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_3)$",
        )],
        "qk_matmul_qout": [(
            "matmul_qout",
            r"^(_backbone_blocks_[0-9]+_attn_mat_mul|"
            r"_encoder_encoder_0_layers_0_self_attn_mat_mul_3|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_3)$",
        )],
        "ov_matmul": [(
            "matmul",
            r"^(_backbone_blocks_[0-9]+_attn_mat_mul_1|"
            r"_encoder_encoder_0_layers_0_self_attn_mat_mul_4|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_4)$",
        )],
        "ov_matmul_qout": [(
            "matmul_qout",
            r"^(_backbone_blocks_[0-9]+_attn_mat_mul_1|"
            r"_encoder_encoder_0_layers_0_self_attn_mat_mul_4|"
            r"_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_4)$",
        )],
        "softmax": [("softmax", r".*softmax$")],
        "softmax_qout": [("softmax_qout", r".*softmax$")],
        "rmsnorm": [("layernorm", r".*layer_normalization$")],
        "layernorm": [("layernorm", r".*layer_normalization$")],
        "rmsnorm_qout": [("layernorm_qout", r".*layer_normalization$")],
        "layernorm_qout": [("layernorm_qout", r".*layer_normalization$")],
        "add": [("add", r".*add.*")],
        "mul": [("mul", r".*mul.*")],
        "gelu_qout": [("gelu_qout", r".*gelu.*")],
        "relu_qout": [("relu_qout", r".*relu.*")],
        "silu_qout": [("silu_qout", r".*silu.*")],
        "sigmoid_qout": [("sigmoid_qout", r".*sigmoid.*")],
        "activation_qout": [("gelu_qout", r".*gelu.*"), ("relu_qout", r".*relu.*"), ("silu_qout", r".*silu.*"), ("sigmoid_qout", r".*sigmoid.*")],
        "elementwise_qout": [("add_qout", r".*add.*"), ("mul_qout", r".*mul.*"), ("sub_qout", r".*sub.*")],
        "conv_qout": [("conv_qout", r".*conv.*")],
        "pool_qout": [("maxpool2d_qout", r".*max_pool.*")],
        "sampling_qout": [("grid_sample_qout", r".*grid_sample.*"), ("reduce_sum_qout", r".*reduce_sum.*")],
        "attention_qout": [
            ("linear_qout", r"^(_backbone_blocks_[0-9]+_attn_qkv_mat_mul|_encoder_encoder_0_layers_0_self_attn_mat_mul(_[0-2])?|_decoder_decoder_layers_[0-9]+_self_attn_mat_mul(_[0-2])?)$"),
            ("matmul_qout", r"^(_backbone_blocks_[0-9]+_attn_mat_mul|_encoder_encoder_0_layers_0_self_attn_mat_mul_3|_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_3)$"),
            ("softmax_qout", r".*softmax$"),
            ("matmul_qout", r"^(_backbone_blocks_[0-9]+_attn_mat_mul_1|_encoder_encoder_0_layers_0_self_attn_mat_mul_4|_decoder_decoder_layers_[0-9]+_self_attn_mat_mul_4)$"),
            ("linear_qout", r"^(_backbone_blocks_[0-9]+_attn_proj_mat_mul|_encoder_encoder_0_layers_0_self_attn_gemm|_decoder_decoder_layers_[0-9]+_self_attn_gemm)$"),
        ],
        "mlp_qout": [
            ("linear_qout", r"^(_backbone_blocks_[0-9]+_mlp_fc1_mat_mul|.*_linear1_mat_mul)$"),
            ("linear_qout", r"^(_backbone_blocks_[0-9]+_mlp_fc2_mat_mul|.*_linear2_mat_mul)$"),
        ],
    }

    bypass_patchers = {
        "softmax": ("QSoftmax", patch_softmax),
        "softmax_qout": ("QSoftmax", patch_softmax_qout),
        "matmul": ("QMatMul", patch_matmul),
        "matmul_qout": ("QMatMul", patch_matmul_qout),
        "matmul_fp16": ("QMatMul", patch_matmul_fp16),
        "linear": ("QLinear", patch_linear),
        "linear_qout": ("QLinear", patch_linear_qout),
        "layernorm": ("QLayerNorm", patch_layernorm),
        "layernorm_qout": ("QLayerNorm", patch_layernorm_qout),
        "conv": ("QConv2d", patch_conv2d),
        "conv_qout": ("QConv2d", patch_conv2d_qout),
        "gelu": ("QGELU", patch_gelu),
        "gelu_qout": ("QGELU", patch_gelu_qout),
        "relu_qout": ("QReLU", patch_relu_qout),
        "silu_qout": ("QSiLU", patch_silu_qout),
        "sigmoid_qout": ("QSigmoid", patch_sigmoid_qout),
        "add": ("QAdd", patch_add),
        "add_qout": ("QAdd", patch_add_qout),
        "mul": ("QMul", patch_mul),
        "mul_qout": ("QMul", patch_mul_qout),
        "sub_qout": ("QSub", patch_sub_qout),
        "div_qout": ("QDiv", patch_div_qout),
        "maxpool2d_qout": ("QMaxPool2d", patch_maxpool2d_qout),
        "grid_sample_qout": ("QGridSample", patch_grid_sample_qout),
        "reduce_sum_qout": ("QReduceSum", patch_reduce_sum_qout),
    }

    input_name, input_shape, _output_names = detect_onnx_io(
        onnx_path,
        batch_size=batch_size,
        image_size=image_size,
    )
    example_input = torch.randn(*input_shape, dtype=torch.float32)
    mixed_precision = load_mixed_precision_config(mixed_precision_config)
    quant_config_args = argparse.Namespace(
        quant_type=quant_type,
        input_enable_fp32=input_enable_fp32,
        output_enable_fp32=output_enable_fp32,
    )
    quant_config = build_quant_config(quant_config_args, mixed_precision)
    quanted_graph = _convert_model_to_quanted_model(
        str(Path(onnx_path).expanduser().resolve()),
        FrontendType.ONNX,
        [example_input.cpu()],
        DeviceType.XH2a,
        quant_config=quant_config,
        use_ptq=False,
        input_names=[input_name],
    )

    runtime_device = torch.device(resolve_runtime_device(device))
    if quantgraph_mode == "none":
        quanted_graph.disable_quant()
        input_dtype = torch.float32
    else:
        precision_mode = {
            "fast": PrecisionMode.FAST,
            "aligned": PrecisionMode.ALIGNED,
        }[quantgraph_mode]
        ptq_quantize(quanted_graph, [[example_input.cpu()]], precision_mode, runtime_device)
        quanted_graph.set_precision_mode(precision_mode)
        input_dtype = torch.float16

    torch_bypass_nodes: dict[str, list[str]] = {}

    def apply_bypass(kind: str, pattern_text: str, label: str) -> None:
        class_suffix, patcher = bypass_patchers[kind]
        pattern = re.compile(pattern_text)
        matched_nodes = []
        for module_name, module in quanted_graph.named_modules():
            if pattern.search(module_name) and module.__class__.__name__.endswith(class_suffix):
                module.forward = patcher(module)
                matched_nodes.append(module_name)
        if not matched_nodes:
            raise RuntimeError(f"No QuantGraph {kind} modules matched {label}: {pattern_text}")
        torch_bypass_nodes[label] = matched_nodes
        print(f"Bypassed {len(matched_nodes)} QuantGraph {kind} modules for {label}")

    for group_name in torch_bypass_groups or []:
        if group_name not in bypass_group_specs:
            raise RuntimeError(
                f"Unknown --quantgraph-torch-bypass-group={group_name}; "
                f"available groups: {', '.join(sorted(bypass_group_specs))}"
            )
        for index, (kind, pattern_text) in enumerate(bypass_group_specs[group_name]):
            apply_bypass(kind, pattern_text, f"group:{group_name}:{index}:{kind}")

    if torch_linear_pattern:
        apply_bypass("linear", torch_linear_pattern, "linear_pattern")
    if torch_layernorm_pattern:
        apply_bypass("layernorm", torch_layernorm_pattern, "layernorm_pattern")
    if torch_conv_pattern:
        apply_bypass("conv", torch_conv_pattern, "conv_pattern")
    if torch_gelu_pattern:
        apply_bypass("gelu", torch_gelu_pattern, "gelu_pattern")
    for preset_name in torch_sigmoid_presets or []:
        apply_bypass(
            "sigmoid_qout",
            QUANTGRAPH_TORCH_SIGMOID_PRESETS[preset_name],
            f"sigmoid_preset:{preset_name}",
        )
    if torch_sigmoid_pattern:
        apply_bypass("sigmoid_qout", torch_sigmoid_pattern, "sigmoid_pattern")
    if torch_add_pattern:
        apply_bypass("add", torch_add_pattern, "add_pattern")
    if torch_mul_pattern:
        apply_bypass("mul", torch_mul_pattern, "mul_pattern")

    torch_softmax_nodes: list[str] = []
    if torch_softmax_pattern:
        pattern = re.compile(torch_softmax_pattern)

        def make_torch_softmax_forward(module):
            def torch_softmax_forward(self, x):
                if hasattr(x, "dequantize"):
                    x = x.dequantize()
                out = torch.softmax(x.float(), dim=self.dim)
                if getattr(self, "is_quanting", False) and hasattr(self, "o_quantizer"):
                    out = self.o_quantizer(out)
                return out

            return types.MethodType(torch_softmax_forward, module)

        for module_name, module in quanted_graph.named_modules():
            if pattern.search(module_name) and module.__class__.__name__.endswith("Softmax"):
                module.forward = make_torch_softmax_forward(module)
                torch_softmax_nodes.append(module_name)
        if not torch_softmax_nodes:
            raise RuntimeError(f"No QuantGraph softmax modules matched pattern: {torch_softmax_pattern}")
        print(f"Replaced {len(torch_softmax_nodes)} QuantGraph softmax modules with torch.softmax")

    torch_matmul_nodes: list[str] = []
    if torch_matmul_pattern:
        pattern = re.compile(torch_matmul_pattern)

        def make_torch_matmul_forward(module):
            def torch_matmul_forward(self, x, y):
                if hasattr(x, "dequantize"):
                    x = x.dequantize()
                if hasattr(y, "dequantize"):
                    y = y.dequantize()
                return torch.matmul(x.float(), y.float())

            return types.MethodType(torch_matmul_forward, module)

        for module_name, module in quanted_graph.named_modules():
            if pattern.search(module_name) and module.__class__.__name__.endswith("MatMul"):
                module.forward = make_torch_matmul_forward(module)
                torch_matmul_nodes.append(module_name)
        if not torch_matmul_nodes:
            raise RuntimeError(f"No QuantGraph matmul modules matched pattern: {torch_matmul_pattern}")
        print(f"Replaced {len(torch_matmul_nodes)} QuantGraph matmul modules with torch.matmul")

    quanted_graph.eval().to(runtime_device)

    def to_numpy(item):
        if hasattr(item, "dequantize"):
            item = item.dequantize()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
        return np.asarray(item)

    def run(batch):
        with torch.no_grad():
            output = quanted_graph(batch.to(device=runtime_device, dtype=input_dtype))
        if not isinstance(output, (tuple, list)):
            output = (output,)
        return tuple(to_numpy(item) for item in output)

    run.torch_bypass_nodes = torch_bypass_nodes
    return run


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 模型
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_MODEL_CACHE_DIR))
    parser.add_argument("--data-cache-dir", default=str(DEFAULT_DATA_CACHE_DIR))

    # 数据
    parser.add_argument("--images-dir", default=str(DEFAULT_COCO_IMG), help="COCO val2017 图片目录")
    parser.add_argument("--annotations", default=str(DEFAULT_COCO_ANN), help="COCO instances_val2017.json")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    # 后端
    parser.add_argument("--backend", default="hmonnx", choices=["torch", "onnx", "frontend", "quantgraph", "hmonnx"])
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX), help="ONNX 路径（backend=onnx 时必需）")
    parser.add_argument("--hmonnx", default=str(DEFAULT_HMONNX), help="HMONNX 路径（backend=hmonnx 时必需）")
    parser.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE, help="Quant type for backend=quantgraph.")
    parser.add_argument(
        "--quantgraph-mode",
        default="aligned",
        choices=["none", "fast", "aligned"],
        help="QuantGraph execution stage: none only wraps the graph, fast/aligned runs PTQ first.",
    )
    parser.add_argument(
        "--mixed-precision-config",
        default=None,
        help="Optional mixed precision JSON/YAML config for backend=quantgraph.",
    )
    parser.add_argument("--input-enable-fp32", action="store_true", help="Keep quant graph input stubs in fp32.")
    parser.add_argument(
        "--output-enable-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep quant graph output stubs in fp32. Enabled by default for detection boxes/scores.",
    )
    parser.add_argument(
        "--quantgraph-torch-softmax-pattern",
        default=None,
        help=(
            "Regex for QuantGraph softmax module names to replace with torch.softmax. "
            "Only valid for backend=quantgraph; useful for isolating XH2aQuantQSoftmax error."
        ),
    )
    parser.add_argument(
        "--quantgraph-torch-matmul-pattern",
        default=None,
        help=(
            "Regex for QuantGraph MatMul module names to replace with torch.matmul. "
            "Only valid for backend=quantgraph; useful for isolating pre-softmax QK MatMul clamp error."
        ),
    )
    parser.add_argument(
        "--quantgraph-torch-bypass-group",
        action="append",
        default=[],
        help=(
            "Named QuantGraph operator group to run with plain torch implementation and fp32 output. "
            "Can be repeated. Supported groups include q, k, v, qkv, backbone_qkv, self_attn_qkv, o, up, down, "
            "gate, qk_matmul, ov_matmul, softmax, rmsnorm/layernorm, add, mul."
        ),
    )
    parser.add_argument(
        "--quantgraph-torch-linear-pattern",
        default=None,
        help="Regex for QuantGraph Linear modules to replace with torch.nn.functional.linear.",
    )
    parser.add_argument(
        "--quantgraph-torch-layernorm-pattern",
        default=None,
        help="Regex for QuantGraph LayerNorm modules to replace with torch.nn.functional.layer_norm.",
    )
    parser.add_argument(
        "--quantgraph-torch-conv-pattern",
        default=None,
        help="Regex for QuantGraph Conv2d modules to replace with torch.nn.functional.conv2d.",
    )
    parser.add_argument(
        "--quantgraph-torch-gelu-pattern",
        default=None,
        help="Regex for QuantGraph GELU modules to replace with torch.nn.functional.gelu.",
    )
    parser.add_argument(
        "--quantgraph-torch-sigmoid-pattern",
        default=None,
        help="Regex for QuantGraph Sigmoid modules to replace with torch.sigmoid and quantized output boundary.",
    )
    parser.add_argument(
        "--quantgraph-torch-sigmoid-preset",
        action="append",
        default=[],
        choices=sorted(QUANTGRAPH_TORCH_SIGMOID_PRESETS),
        help=(
            "Named QuantGraph Sigmoid preset to replace with torch.sigmoid and quantized output boundary. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--quantgraph-torch-sigmoid-dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Internal dtype for torch sigmoid fallback before re-applying the quantized output boundary.",
    )
    parser.add_argument(
        "--quantgraph-torch-add-pattern",
        default=None,
        help="Regex for QuantGraph Add modules to replace with torch add.",
    )
    parser.add_argument(
        "--quantgraph-torch-mul-pattern",
        default=None,
        help="Regex for QuantGraph Mul modules to replace with torch mul.",
    )
    parser.add_argument("--torch-quarot", action="store_true", help="Apply attention QuaRot before Torch backend eval.")
    parser.add_argument(
        "--quarot-scope",
        default="all-attention",
        choices=["all-attention", "dinov3-attention", "detector-attention"],
        help="QuaRot scope used when --torch-quarot is set.",
    )
    parser.add_argument(
        "--dinov3-qk-rotation",
        default="rope-pairs",
        choices=["none", "rope-pairs"],
        help="DINOv3 Q/K rotation used when --torch-quarot is set.",
    )
    parser.add_argument("--quarot-seed", type=int, default=4200, help="QuaRot random seed.")

    # 评测
    parser.add_argument("--out-dir", default=str(DEFAULT_HMONNX_EVAL_DIR))
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--sample-size", type=int, default=200, help="随机采样 N 张图")
    parser.add_argument("--sample-seed", type=int, default=4200)
    parser.add_argument("--limit", type=int, default=None, help="最多评测 N 张图")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for split evaluation.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of evaluation shards.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--skip-missing-images", action="store_true")

    args = parser.parse_args(argv)

    # ── 校验 ──
    if args.backend in {"onnx", "frontend", "quantgraph"} and not args.onnx:
        parser.error(f"--onnx is required for --backend {args.backend}")
    if args.backend == "hmonnx" and not args.hmonnx:
        parser.error("--hmonnx is required for --backend hmonnx")
    if args.torch_quarot and args.backend != "torch":
        parser.error("--torch-quarot is only valid with --backend torch")
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must satisfy 0 <= shard-index < shard-count")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 COCO GT ──
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(args.annotations))
    image_ids = coco_gt.getImgIds()

    if args.skip_missing_images:
        image_root = Path(args.images_dir)
        image_ids = [iid for iid in image_ids if (image_root / coco_gt.loadImgs([iid])[0]["file_name"]).exists()]

    image_ids = select_coco_image_ids(image_ids, limit=args.limit, sample_size=args.sample_size, sample_seed=args.sample_seed)
    if not image_ids:
        raise RuntimeError("No COCO images selected.")

    full_image_ids = list(image_ids)
    if args.shard_count > 1:
        shard_size = (len(full_image_ids) + args.shard_count - 1) // args.shard_count
        shard_start = args.shard_index * shard_size
        shard_end = min(len(full_image_ids), shard_start + shard_size)
        image_ids = full_image_ids[shard_start:shard_end]
        if not image_ids:
            raise RuntimeError(f"No COCO images selected for shard {args.shard_index}/{args.shard_count}.")

    category_ids = sorted(coco_gt.getCatIds())
    print(f"Evaluating {len(image_ids)} images on backend={args.backend}")
    if args.shard_count > 1:
        print(f"Shard: {args.shard_index}/{args.shard_count}; full_sample_images={len(full_image_ids)}")

    # ── 加载模型 ──
    model = load_lightly_model(args)
    model.eval()
    quarot_report = None
    if args.torch_quarot:
        from dataclasses import asdict

        quarot_module_path = Path(__file__).resolve().parent / "debug" / "legacy_scripts" / "export_quarot_onnx.py"
        if not quarot_module_path.exists():
            raise FileNotFoundError(f"Archived QuaRot helper not found: {quarot_module_path}")
        spec = importlib.util.spec_from_file_location("dinov3_archived_export_quarot_onnx", quarot_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load archived QuaRot helper: {quarot_module_path}")
        quarot_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = quarot_module
        spec.loader.exec_module(quarot_module)
        apply_attention_quarot = quarot_module.apply_attention_quarot

        entries = apply_attention_quarot(
            model,
            scope=args.quarot_scope,
            dinov3_qk_rotation=args.dinov3_qk_rotation,
            seed=args.quarot_seed,
        )
        quarot_report = {
            "scope": args.quarot_scope,
            "seed": args.quarot_seed,
            "dinov3_qk_rotation": args.dinov3_qk_rotation,
            "num_rotated_modules": len(entries),
            "rotated_modules": [asdict(entry) for entry in entries],
        }
        (out_dir / "torch_quarot_report.json").write_text(
            json.dumps(quarot_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    model.deploy()

    runner = None
    if args.backend == "onnx":
        runner = make_onnx_runner(args.onnx)
    elif args.backend == "frontend":
        runner = make_frontend_runner(
            args.onnx,
            batch_size=1,
            image_size=args.image_size,
            device=args.device,
        )
    elif args.backend == "quantgraph":
        runner = make_quantgraph_runner(
            args.onnx,
            batch_size=1,
            image_size=args.image_size,
            device=args.device,
            quant_type=args.quant_type,
            quantgraph_mode=args.quantgraph_mode,
            mixed_precision_config=args.mixed_precision_config,
            input_enable_fp32=args.input_enable_fp32,
            output_enable_fp32=args.output_enable_fp32,
            torch_softmax_pattern=args.quantgraph_torch_softmax_pattern,
            torch_matmul_pattern=args.quantgraph_torch_matmul_pattern,
            torch_bypass_groups=args.quantgraph_torch_bypass_group,
            torch_linear_pattern=args.quantgraph_torch_linear_pattern,
            torch_layernorm_pattern=args.quantgraph_torch_layernorm_pattern,
            torch_conv_pattern=args.quantgraph_torch_conv_pattern,
            torch_gelu_pattern=args.quantgraph_torch_gelu_pattern,
            torch_sigmoid_presets=args.quantgraph_torch_sigmoid_preset,
            torch_sigmoid_pattern=args.quantgraph_torch_sigmoid_pattern,
            torch_sigmoid_dtype=args.quantgraph_torch_sigmoid_dtype,
            torch_add_pattern=args.quantgraph_torch_add_pattern,
            torch_mul_pattern=args.quantgraph_torch_mul_pattern,
        )
    elif args.backend == "hmonnx":
        runner = make_hmonnx_runner(args.hmonnx, args.device)

    # ── 推理 ──
    predictions: list[dict] = []
    for idx, image_id in enumerate(image_ids, 1):
        info = coco_gt.loadImgs([image_id])[0]
        image_path = Path(args.images_dir) / info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        batch, metadata = preprocess_image_batch(model, image_path)

        if args.backend == "torch":
            labels, boxes, scores = run_torch_with_original_size(model, batch, metadata)
        elif runner is not None:
            labels, boxes, scores = runner(batch)
            labels, boxes, scores = scale_resized_boxes_to_original(
                (labels, boxes, scores), metadata, (args.image_size, args.image_size),
            )

        predictions.extend(outputs_to_coco_rows(
            (labels, boxes, scores),
            image_id=image_id,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
            category_ids=category_ids,
        ))

        if idx % args.log_every == 0:
            print(f"  [{idx}/{len(image_ids)}]")
        if args.backend == "hmonnx":
            import gc
            import torch

            del batch, labels, boxes, scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── COCO eval ──
    if not predictions:
        raise RuntimeError("No predictions generated.")

    predictions_path = out_dir / f"{args.backend}_predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ── 保存指标 ──
    metrics = {
        "backend": args.backend,
        "num_images": len(image_ids),
        "predictions": str(predictions_path),
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed if args.sample_size is not None else None,
        "full_num_images": len(full_image_ids),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "image_ids": image_ids,
        "torch_quarot": bool(args.torch_quarot),
        "quant_type": args.quant_type if args.backend == "quantgraph" else None,
        "quantgraph_mode": args.quantgraph_mode if args.backend == "quantgraph" else None,
        "mixed_precision_config": args.mixed_precision_config if args.backend == "quantgraph" else None,
        "input_enable_fp32": bool(args.input_enable_fp32) if args.backend == "quantgraph" else None,
        "output_enable_fp32": bool(args.output_enable_fp32) if args.backend == "quantgraph" else None,
        "quantgraph_torch_softmax_pattern": args.quantgraph_torch_softmax_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_matmul_pattern": args.quantgraph_torch_matmul_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_bypass_group": args.quantgraph_torch_bypass_group if args.backend == "quantgraph" else None,
        "quantgraph_torch_linear_pattern": args.quantgraph_torch_linear_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_layernorm_pattern": args.quantgraph_torch_layernorm_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_conv_pattern": args.quantgraph_torch_conv_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_gelu_pattern": args.quantgraph_torch_gelu_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_sigmoid_preset": args.quantgraph_torch_sigmoid_preset if args.backend == "quantgraph" else None,
        "quantgraph_torch_sigmoid_pattern": args.quantgraph_torch_sigmoid_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_sigmoid_dtype": args.quantgraph_torch_sigmoid_dtype if args.backend == "quantgraph" else None,
        "quantgraph_torch_add_pattern": args.quantgraph_torch_add_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_mul_pattern": args.quantgraph_torch_mul_pattern if args.backend == "quantgraph" else None,
        "quantgraph_torch_bypass_nodes": getattr(runner, "torch_bypass_nodes", None) if args.backend == "quantgraph" else None,
        "quarot_scope": args.quarot_scope if args.torch_quarot else None,
        "dinov3_qk_rotation": args.dinov3_qk_rotation if args.torch_quarot else None,
        "quarot_seed": args.quarot_seed if args.torch_quarot else None,
        "quarot_report": str(out_dir / "torch_quarot_report.json") if quarot_report else None,
        "AP_50_95": float(coco_eval.stats[0]),
        "AP_50": float(coco_eval.stats[1]),
        "AP_75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
        "reference_lightly_train_coco_val_AP_50_95": 0.498,
    }
    metrics_path = out_dir / f"{args.backend}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"AP_50_95:  {metrics['AP_50_95']:.4f}")
    print(f"AP_50:     {metrics['AP_50']:.4f}")
    print(f"AP_75:     {metrics['AP_75']:.4f}")
    print(f"AP_small:  {metrics['AP_small']:.4f}")
    print(f"AP_medium: {metrics['AP_medium']:.4f}")
    print(f"AP_large:  {metrics['AP_large']:.4f}")
    print(f"Metrics:   {metrics_path}")
    return metrics_path


if __name__ == "__main__":
    main()
