import argparse
import logging  # 1. 导入标准 logging 库
from pathlib import Path

# 导入 onnx 库，用于加载和修改 ONNX 模型
import onnx
import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
    xhquant_init_logger,
)


def _onnx_dtype_to_torch(elem_type, fallback_dtype):
    if elem_type == onnx.TensorProto.FLOAT:
        return torch.float32
    if elem_type == onnx.TensorProto.FLOAT16:
        return torch.float16
    if elem_type == onnx.TensorProto.DOUBLE:
        return torch.float64
    if elem_type == onnx.TensorProto.INT64:
        return torch.int64
    if elem_type == onnx.TensorProto.INT32:
        return torch.int32
    if elem_type == onnx.TensorProto.INT16:
        return torch.int16
    if elem_type == onnx.TensorProto.INT8:
        return torch.int8
    if elem_type == onnx.TensorProto.UINT8:
        return torch.uint8
    if elem_type == onnx.TensorProto.BOOL:
        return torch.bool
    if fallback_dtype is not None:
        return fallback_dtype
    raise ValueError(f"Unsupported ONNX elem_type: {elem_type}")


def _parse_input_dtype(arg: str):
    arg = arg.lower()
    if arg in ("fp16", "float16"):
        return torch.float16
    if arg in ("fp32", "float32"):
        return torch.float32
    if arg in ("fp64", "float64"):
        return torch.float64
    if arg in ("int64",):
        return torch.int64
    if arg in ("int32",):
        return torch.int32
    if arg in ("int16",):
        return torch.int16
    if arg in ("int8",):
        return torch.int8
    if arg in ("uint8",):
        return torch.uint8
    if arg in ("bool",):
        return torch.bool
    return None


def _make_inputs_from_onnx(onnx_path: str, fallback_dtype: torch.dtype):
    model = onnx.load(onnx_path)
    init_names = {init.name for init in model.graph.initializer}
    inputs = []
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        t = inp.type.tensor_type
        if not t.HasField("shape"):
            raise ValueError(f"Input {inp.name} has no tensor shape")
        shape = []
        for d in t.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(1)
        dtype = _onnx_dtype_to_torch(t.elem_type, fallback_dtype)
        if dtype == torch.int64:
            dtype = torch.int32
        if dtype == torch.bool:
            data = torch.rand(*shape) > 0.5
        elif dtype.is_floating_point:
            data = torch.randn(*shape, dtype=dtype)
        else:
            data = torch.randint(0, 3, shape, dtype=dtype)
        inputs.append((inp.name, data))
    return inputs


def main(args):
    # --- 日志配置部分 ---
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    # --------------------

    # 获取 logger
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{quant_type}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)
    xhquant_init_logger()
    logger = get_root_logger()
    # try:
    logger.info("开始将 ONNX 转换为 hmonnx...")
    
    fallback_dtype = _parse_input_dtype(args.input_dtype)
    named_inputs = _make_inputs_from_onnx(onnx_file, fallback_dtype)
    input_tensors = [t for _, t in named_inputs]

    convert_onnx_to_hmonnx(
        onnx_file,
        input_tensors,
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
    )
    logger.info(f"原始 hmonnx 文件已成功生成: {out_hmonnx_file}")

    logger.info(f"最终转换流程完成，修复后的 hmonnx 文件位于: {out_hmonnx_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    input_tensors = [
        (t.to(torch.float16) if t.is_floating_point() else t).to(device)
        for t in input_tensors
    ]
    if len(input_tensors) == 1:
        session(input_tensors[0])
    else:
        session(*input_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="weights/zte/model_timi_cjb_sim.onnx")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--input-dtype", type=str, default="fp16", help="input dtype, default is fp16")
    args = parser.parse_args()
    main(args)
