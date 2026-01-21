import argparse
from pathlib import Path

import torch
from xhquant.api import (
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    input = torch.randn(1, 3, 1024, 2048, dtype=torch.float32)
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
        output_names=["output"],
    )
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default="data/models/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.onnx"
    )
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
