import argparse
from pathlib import Path

import onnx
import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def main(args):
    xhquant_init(None, debug=args.debug)

    batch_size = args.batch_size
    onnx_file = args.onnx

    onnx_model = onnx.load(onnx_file)
    input_shape = [int(dim.dim_value) for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    input_shape[0] = batch_size
    str_shape = "x".join([str(dim) for dim in input_shape])

    onnx_name = Path(onnx_file).stem
    onnx_name = f"{onnx_name}_{str_shape}"
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(input_shape, dtype=torch.float32)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(
        onnx_file,
        [input],
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=["images"],
        output_names=["cls_score"],
    )

    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / "hmonnx/golden"
    session.step = 0
    session(input.half().to(device))
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default="/data01/home/xuchen/xh2/xhquant_examples/data/models/resnet50_224x224.onnx"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    main(args)
