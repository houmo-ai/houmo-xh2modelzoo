# Copyright 2025 HOUMO AI
# File: groundingdino_export.py

import argparse
from pathlib import Path

import onnx
import torch
from onnx import TensorProto
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

DEFAULT_DIM_BY_NAME = {
    "image": [1, 3, 800, 1200],
    "input_ids": [1, 256],
    "attention_mask": [1, 256],
    "position_ids": [1, 256],
    "token_type_ids": [1, 256],
}


def _tensorproto_to_torch_dtype(elem_type: int) -> torch.dtype:
    mapping = {
        TensorProto.FLOAT: torch.float32,
        TensorProto.FLOAT16: torch.float16,
        TensorProto.INT64: torch.int64,
        TensorProto.INT32: torch.int32,
        TensorProto.BOOL: torch.bool,
    }
    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX input dtype: {TensorProto.DataType.Name(elem_type)}")
    return mapping[elem_type]


def _resolve_dim(dim, fallback: int) -> int:
    if dim.dim_value > 0:
        return dim.dim_value
    return fallback


def load_onnx_io_spec(onnx_file: str):
    model = onnx.load(onnx_file)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    input_specs = []
    for input_info in model.graph.input:
        if input_info.name in initializer_names:
            continue

        default_shape = DEFAULT_DIM_BY_NAME.get(input_info.name, [1])
        shape = [
            _resolve_dim(dim, default_shape[idx] if idx < len(default_shape) else 1)
            for idx, dim in enumerate(input_info.type.tensor_type.shape.dim)
        ]
        input_specs.append(
            {
                "name": input_info.name,
                "shape": shape,
                "dtype": _tensorproto_to_torch_dtype(input_info.type.tensor_type.elem_type),
            }
        )

    output_names = [output.name for output in model.graph.output]
    return input_specs, output_names


def build_dummy_input(spec, device="cpu", dtype_img=torch.float32):
    name = spec["name"]
    shape = spec["shape"]
    dtype = spec["dtype"]

    if name == "image":
        return torch.randn(*shape, dtype=dtype_img if dtype.is_floating_point else dtype, device=device)
    if name == "input_ids":
        return torch.randint(0, 30522, shape, dtype=dtype, device=device)
    if name == "attention_mask":
        return torch.ones(shape, dtype=dtype, device=device)
    if name == "position_ids":
        seq_len = shape[-1]
        return torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(0).expand(*shape)
    if name == "token_type_ids":
        return torch.zeros(shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.zeros(shape, dtype=dtype, device=device)
    if dtype.is_floating_point:
        return torch.randn(*shape, dtype=dtype, device=device)
    return torch.zeros(shape, dtype=dtype, device=device)


def get_dummy_inputs(onnx_file: str, device="cpu", dtype_img=torch.float32):
    input_specs, output_names = load_onnx_io_spec(onnx_file)
    dummy_inputs = [build_dummy_input(spec, device=device, dtype_img=dtype_img) for spec in input_specs]
    input_names = [spec["name"] for spec in input_specs]
    return dummy_inputs, input_names, output_names

def main(args):
    # 1. 路径设置
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs_v2") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file_str = str(out_hmonnx_file)

    # 2. 初始化 xhquant
    xhquant_init(None, debug=args.debug)

    # 3. 配置量化参数
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    # 4. 准备转换用的输入 (FP32)
    # 注意：这里生成的 dummy inputs 用于校准/Tracing，图像保持 FP32，文本输入跟随 ONNX 实际 dtype
    convert_inputs, input_names, output_names = get_dummy_inputs(
        onnx_file,
        device=args.device,#"cpu",
        dtype_img=torch.float32,
    )

    print(f"Starting conversion for {onnx_file}...")
    
    # 5. 执行转换
    convert_onnx_to_hmonnx(
        onnx_file,
        convert_inputs, # 传入包含5个Tensor的列表
        DeviceType.XH2a,
        out_hmonnx_file_str,
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file_str}")

    # 6. 运行 Golden Inference (验证)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # session = HMONNXGoldenInference(out_hmonnx_file_str)
    # session.to(device)
    # session.save_golden = True
    # session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    # session.step = 0
    
    # inference_inputs = get_dummy_inputs(device=device, dtype_img=torch.float16)
    
    # print("Running Golden Inference...")
    # # 使用 * 解包列表，传入多个参数
    # session(*inference_inputs)
    # print("Golden Inference Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径指向你刚才导出的静态 ONNX
    parser.add_argument(
        "--onnx", type=str, default="/data01/home/chenzx/project/xh2modelzoo/GroundingDINO/outputs/groundingdino.onnx"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--device", type=str, default="cpu", help="device for conversion and inference")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    args = parser.parse_args()
    main(args)