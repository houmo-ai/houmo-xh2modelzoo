import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("XHQUANT_STRICT_EXPORT", "0")

import onnx
import torch
from onnx import TensorProto
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    xhquant_init,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO_ROOT / "work_dirs/zhongxin/dlrkc-1 (1).onnx"
DEFAULT_OUTPUT = REPO_ROOT / "work_dirs/zhongxin/dlrkc-1_XH2a_w8a8_sefp.onnx"
TORCH_DTYPES = {
    TensorProto.FLOAT: torch.float32,
    TensorProto.FLOAT16: torch.float16,
    TensorProto.INT32: torch.int32,
    TensorProto.INT64: torch.int64,
    TensorProto.BOOL: torch.bool,
}


def tensor_shape(value_info):
    return [
        dimension.dim_value if dimension.dim_value > 0 else dimension.dim_param
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def freeze_batch(source: Path, destination: Path, batch_size: int):
    model = onnx.load(source)
    changed = []
    value_infos = [*model.graph.input, *model.graph.output, *model.graph.value_info]
    for value_info in value_infos:
        dimensions = value_info.type.tensor_type.shape.dim
        if dimensions and dimensions[0].dim_param == "batch_size":
            dimensions[0].ClearField("dim_param")
            dimensions[0].dim_value = batch_size
            changed.append(value_info.name)

    if not changed:
        raise ValueError("No batch_size dimensions were found in the ONNX model")

    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return model, changed


def make_inputs(model, batch_size: int):
    generator = torch.Generator().manual_seed(20260804)
    inputs = []
    input_names = []
    input_shapes = {}
    for value_info in model.graph.input:
        shape = tensor_shape(value_info)
        if (
            not shape
            or shape[0] != batch_size
            or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
        ):
            raise ValueError(
                f"Input {value_info.name!r} is not fully static batch={batch_size}: {shape}"
            )

        dtype = TORCH_DTYPES.get(value_info.type.tensor_type.elem_type)
        if dtype is None:
            raise TypeError(
                f"Unsupported input dtype for {value_info.name!r}: "
                f"{value_info.type.tensor_type.elem_type}"
            )

        if dtype.is_floating_point:
            tensor = torch.randn(shape, dtype=dtype, generator=generator)
        elif dtype == torch.bool:
            tensor = torch.randint(0, 2, shape, dtype=dtype, generator=generator)
        else:
            tensor = torch.randint(0, 10, shape, dtype=dtype, generator=generator)

        inputs.append(tensor)
        input_names.append(value_info.name)
        input_shapes[value_info.name] = shape
    return inputs, input_names, input_shapes


def patch_constant_of_shape():
    from xhquant.nn.modules.onnx_style_modules import OnnxConstantOfShape

    def forward(module, shape):
        fill_value = module.value.item()
        shape_values = shape.tolist() if isinstance(shape, torch.Tensor) else list(shape)
        return torch.full(
            size=torch.Size(shape_values),
            fill_value=int(fill_value) if isinstance(fill_value, bool) else fill_value,
            dtype=module.value.dtype,
            device=module.value.device,
        )

    OnnxConstantOfShape.forward = forward


def make_quant_config(quant_type: str):
    sum_config = {
        "sum_dtype": "float16",
    }
    quant_config = create_quant_config(
        QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    )
    # Configure Norm precision through xhquant's quantization config. The
    # exporter consumes these values when lowering LayerNorm; no ONNX node
    # attributes are patched before or after export.
    quant_config["sum_quant"] = dict(sum_config)
    quant_config.ops_cfg["LayerNorm"] = {
        "force_fp32": False,
        "sum_dtype": "float16",
        "sum_cfg": dict(sum_config),
    }
    return quant_config, sum_config


def graph_summary(model):
    return {
        "inputs": {
            value_info.name: tensor_shape(value_info)
            for value_info in model.graph.input
        },
        "outputs": {
            value_info.name: tensor_shape(value_info)
            for value_info in model.graph.output
        },
        "op_counts": dict(
            sorted(Counter(node.op_type for node in model.graph.node).items())
        ),
    }


def hmonnx_audit(model, batch_size: int):
    value_infos = [*model.graph.input, *model.graph.value_info, *model.graph.output]
    shapes = {value_info.name: tensor_shape(value_info) for value_info in value_infos}
    initializer_shapes = {
        initializer.name: list(initializer.dims) for initializer in model.graph.initializer
    }

    layernorm_rows = []
    for node in model.graph.node:
        if node.op_type != "LayerNorm":
            continue
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        sum_dtype = attributes.get("sum_dtype")
        if isinstance(sum_dtype, bytes):
            sum_dtype = sum_dtype.decode()
        layernorm_rows.append(
            {
                "name": node.name,
                "activation_shape": shapes.get(node.input[0]),
                "force_fp32": attributes.get("force_fp32"),
                "sum_dtype": sum_dtype,
                "keep_fp32_variance": attributes.get("keep_fp32_variance"),
                "keep_fp32_mean": attributes.get("keep_fp32_mean"),
            }
        )

    te_rows = []
    for node in model.graph.node:
        if node.op_type not in {"Linear", "MatMul"}:
            continue
        activation_shape = shapes.get(node.input[0], initializer_shapes.get(node.input[0]))
        direct_batch = activation_shape is not None and batch_size in activation_shape[:-1]
        flattened_batch = (
            activation_shape is not None
            and len(activation_shape) == 2
            and isinstance(activation_shape[0], int)
            and activation_shape[0] % batch_size == 0
        )
        te_rows.append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "activation_shape": activation_shape,
                "output_shape": shapes.get(node.output[0]),
                "batch_encoding": (
                    "direct"
                    if direct_batch
                    else "flattened"
                    if flattened_batch
                    else "unexpected"
                ),
                "flattened_sequence": (
                    activation_shape[0] // batch_size
                    if flattened_batch and not direct_batch
                    else None
                ),
            }
        )

    unexpected_te = [row for row in te_rows if row["batch_encoding"] == "unexpected"]
    return {
        "layernorm": {
            "count": len(layernorm_rows),
            "all_force_fp32_false": bool(layernorm_rows)
            and all(row["force_fp32"] == 0 for row in layernorm_rows),
            "all_sum_dtype_float16": bool(layernorm_rows)
            and all(row["sum_dtype"] == "float16" for row in layernorm_rows),
            "all_keep_fp32_variance_false": bool(layernorm_rows)
            and all(row["keep_fp32_variance"] == 0 for row in layernorm_rows),
            "all_keep_fp32_mean_false": bool(layernorm_rows)
            and all(row["keep_fp32_mean"] == 0 for row in layernorm_rows),
            "nodes": layernorm_rows,
        },
        "te": {
            "count": len(te_rows),
            "op_counts": dict(Counter(row["op_type"] for row in te_rows)),
            "batch_encoding_counts": dict(
                Counter(row["batch_encoding"] for row in te_rows)
            ),
            "all_batch_48": not unexpected_te,
            "unexpected_nodes": unexpected_te,
            "nodes": te_rows,
        },
    }


def validate_models(static_model_path, hmonnx_path, inputs, input_names, output_names):
    import onnxruntime as ort

    input_map = {
        name: tensor.detach().cpu() for name, tensor in zip(input_names, inputs)
    }
    ort_session = ort.InferenceSession(
        str(static_model_path), providers=["CPUExecutionProvider"]
    )
    reference_outputs = ort_session.run(
        output_names, {name: tensor.numpy() for name, tensor in input_map.items()}
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HMONNX golden validation")
    hmonnx_session = HMONNXGoldenInference(str(hmonnx_path))
    hmonnx_session.initialize()
    hmonnx_inputs = []
    for name in hmonnx_session.get_input_names():
        input_info = hmonnx_session.get_input(name)
        hmonnx_inputs.append(input_map[name].to(device="cuda", dtype=input_info.dtype))
    hmonnx_session.to("cuda")
    hmonnx_outputs = hmonnx_session(*hmonnx_inputs)
    if not isinstance(hmonnx_outputs, (tuple, list)):
        hmonnx_outputs = [hmonnx_outputs]

    metrics = []
    for name, reference, actual in zip(output_names, reference_outputs, hmonnx_outputs):
        reference_tensor = torch.from_numpy(reference).float().reshape(-1)
        actual_tensor = actual.detach().float().cpu().reshape(-1)
        difference = (reference_tensor - actual_tensor).abs()
        cosine = torch.nn.functional.cosine_similarity(
            reference_tensor, actual_tensor, dim=0
        )
        metrics.append(
            {
                "name": name,
                "reference_shape": list(reference.shape),
                "hmonnx_shape": list(actual.shape),
                "cosine_similarity": cosine.item(),
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
            }
        )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Quantize the ZTE DLRKC ONNX model for XH2a."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--quant-type", default="w8a8_sefp")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    source = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    static_model_path = source.with_name(
        f"{source.stem}_batch{args.batch_size}.onnx"
    )
    report_path = output.with_suffix(".audit.json")

    static_model, changed = freeze_batch(
        source, static_model_path, args.batch_size
    )
    inputs, input_names, input_shapes = make_inputs(static_model, args.batch_size)
    output_names = [value_info.name for value_info in static_model.graph.output]
    quant_config, sum_config = make_quant_config(args.quant_type)

    patch_constant_of_shape()
    xhquant_init()
    output.parent.mkdir(parents=True, exist_ok=True)
    convert_onnx_to_hmonnx(
        str(static_model_path),
        inputs,
        DeviceType.XH2a,
        str(output),
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )

    hmonnx_model = onnx.load(output, load_external_data=False)
    audit = hmonnx_audit(hmonnx_model, args.batch_size)
    layernorm_audit = audit["layernorm"]
    if not all(
        layernorm_audit[key]
        for key in (
            "all_force_fp32_false",
            "all_sum_dtype_float16",
            "all_keep_fp32_variance_false",
            "all_keep_fp32_mean_false",
        )
    ):
        raise RuntimeError("The exported HMONNX does not satisfy the LayerNorm policy")
    if not audit["te"]["all_batch_48"]:
        raise RuntimeError("The exported HMONNX contains an unexpected batch encoding")

    validation = None
    if not args.skip_validation:
        validation = validate_models(
            static_model_path, output, inputs, input_names, output_names
        )

    report = {
        "source": str(source),
        "static_model": str(static_model_path),
        "hmonnx": str(output),
        "batch_size": args.batch_size,
        "batch_dimensions_frozen": changed,
        "conversion_input_shapes": input_shapes,
        "quant_type": args.quant_type,
        "layernorm": {
            "force_fp32": False,
            "keep_fp32_variance": False,
            "keep_fp32_mean": False,
            **sum_config,
        },
        "source_graph": graph_summary(static_model),
        "hmonnx_graph": graph_summary(hmonnx_model),
        "hmonnx_audit": audit,
        "validation": validation,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()