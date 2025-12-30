import onnx
import torch
import argparse
from pathlib import Path
from xh_model_zoo_new.xh_base import BaseSingleModelConverter, BaseSingleModelConverterConfig, BaseSingleModelInfer
from xh_model_zoo_new.utils.time_profiler import TimeProfiler


def main(args):
    batch_size = args.batch_size
    onnx_file = args.onnx

    onnx_model = onnx.load(onnx_file)
    input_shape = [int(dim.dim_value) for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    input_shape[0] = batch_size
    str_shape = "x".join([str(dim) for dim in input_shape])

    model_name = Path(onnx_file).stem + f"_{str_shape}"
    work_dir = Path("work_dirs") / model_name

    with TimeProfiler("convert and export") as tp:
        quant_scheme = dict(target_device=args.target_device, quant_type=args.quant_type)
        config = BaseSingleModelConverterConfig(
            quant_scheme=quant_scheme, model_name=model_name, inputs=[torch.randn(input_shape, dtype=torch.float16)]
        )
        output_hmonnx_path = BaseSingleModelConverter.convert_and_export(
            onnx_file, config, work_dir, generate_golden=True
        )
        print(f"Convert and export success, output hmonnx file to: {output_hmonnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/models/resnet/resnet50_224x224.onnx")
    parser.add_argument("--target-device", type=str, default="XH2a", help="target device, default is XH2a")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    main(args)
