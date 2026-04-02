# Copyright 2025 HOUMO AI
# File: groundingdino_export.py

import argparse
from pathlib import Path

import onnx
import torch
from onnx import TensorProto
from xhquant.api import (
    DeviceType,
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


def _collect_tensor_infos(model: onnx.ModelProto):
    tensor_infos = {}
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type == 0:
            continue

        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
        tensor_infos[value_info.name] = (tensor_type.elem_type, shape)
    return tensor_infos


def _append_tensor_value_info(graph, existing_names, name: str, elem_type: int, shape):
    if name in existing_names or shape is None or any(dim is None for dim in shape):
        return
    graph.value_info.extend([onnx.helper.make_tensor_value_info(name, elem_type, shape)])
    existing_names.add(name)


def _rewrite_external_data_location(onnx_file: str, location: str) -> None:
    model = onnx.load(onnx_file, load_external_data=False)
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if not onnx.external_data_helper.uses_external_data(tensor):
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = location
    onnx.save(model, onnx_file)


def rewrite_hmonnx_gridsample_to_fp32(hmonnx_file: str) -> int:
    model = onnx.load(hmonnx_file)
    graph = model.graph
    tensor_infos = _collect_tensor_infos(model)
    existing_value_info_names = {value_info.name for value_info in graph.value_info}

    rewritten = 0
    rewritten_nodes = []
    for node in graph.node:
        copied_node = onnx.NodeProto()
        copied_node.CopyFrom(node)

        if node.domain != "ai.houmo.xh2a" or node.op_type != "GridSample":
            rewritten_nodes.append(copied_node)
            continue

        promoted_inputs = []
        original_inputs = list(node.input)
        original_output = node.output[0]
        for input_index, original_input in enumerate(original_inputs[:2]):
            promoted_input = f"{original_input}_fp32_for_{node.name}_{input_index}"
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Cast",
                    inputs=[original_input],
                    outputs=[promoted_input],
                    name=f"{node.name}_cast_input_{input_index}",
                    to=TensorProto.FLOAT,
                )
            )
            input_info = tensor_infos.get(original_input)
            if input_info is not None:
                _append_tensor_value_info(
                    graph,
                    existing_value_info_names,
                    promoted_input,
                    TensorProto.FLOAT,
                    input_info[1],
                )
            promoted_inputs.append(promoted_input)

        promoted_inputs.extend(original_inputs[2:])

        promoted_output = f"{original_output}_fp32_from_{node.name}"
        output_info = tensor_infos.get(original_output)
        _append_tensor_value_info(
            graph,
            existing_value_info_names,
            promoted_output,
            TensorProto.FLOAT,
            output_info[1] if output_info is not None else None,
        )

        del copied_node.input[:]
        copied_node.input.extend(promoted_inputs)
        del copied_node.output[:]
        copied_node.output.extend([promoted_output])
        rewritten_nodes.append(copied_node)

        restore_dtype = output_info[0] if output_info is not None else TensorProto.FLOAT16
        rewritten_nodes.append(
            onnx.helper.make_node(
                "Cast",
                inputs=[promoted_output],
                outputs=[original_output],
                name=f"{node.name}_cast_output_restore",
                to=restore_dtype,
            )
        )
        rewritten += 1

    if rewritten == 0:
        return 0

    del graph.node[:]
    graph.node.extend(rewritten_nodes)

    hmonnx_path = Path(hmonnx_file)
    external_data_path = Path(hmonnx_file).with_name(f"{Path(hmonnx_file).stem}_external_data")
    temp_hmonnx_path = hmonnx_path.with_suffix(f"{hmonnx_path.suffix}.tmp")
    temp_external_data_path = external_data_path.with_name(f"{external_data_path.name}.tmp")
    temp_hmonnx_path.unlink(missing_ok=True)
    temp_external_data_path.unlink(missing_ok=True)
    onnx.save_model(
        model,
        str(temp_hmonnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=temp_external_data_path.name,
        size_threshold=1024,
    )
    _rewrite_external_data_location(str(temp_hmonnx_path), external_data_path.name)
    external_data_path.unlink(missing_ok=True)
    temp_external_data_path.replace(external_data_path)
    temp_hmonnx_path.replace(hmonnx_path)
    return rewritten


def convert_onnx_to_hmonnx_cpu_safe(
    onnx_file: str,
    convert_inputs,
    target_device: DeviceType,
    out_hmonnx_file: str,
    quant_config,
    input_names,
    output_names,
):
    logger = get_root_logger()
    logger.info(
        "CPU-safe GridSample export enabled: rewrite emitted HMONNX so GridSample executes in float32 on CPU and casts back to the original dtype."
    )

    convert_onnx_to_hmonnx(
        onnx_file,
        convert_inputs,
        target_device,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )

    rewritten = rewrite_hmonnx_gridsample_to_fp32(out_hmonnx_file)
    logger.info(f"Rewrote {rewritten} GridSample nodes in {out_hmonnx_file} for CPU-safe runtime.")

def main(args):
    # 1. 路径设置
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
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
    use_cpu_safe_gridsample = args.cpu_safe_gridsample and args.device.lower() == "cpu"
    if use_cpu_safe_gridsample:
        convert_onnx_to_hmonnx_cpu_safe(
            onnx_file,
            convert_inputs,
            DeviceType.XH2a,
            out_hmonnx_file_str,
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )
    else:
        convert_onnx_to_hmonnx(
            onnx_file,
            convert_inputs,
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
    parser.add_argument(
        "--cpu-safe-gridsample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when device=cpu, rewrite emitted HMONNX so GridSample runs in float32 and casts back to the original dtype",
    )
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    args = parser.parse_args()
    main(args)