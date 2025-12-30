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
    xhquant_init(None, debug=args.debug)
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    onnx_name = f"{onnx_name}"
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    image = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    input = [
        image,
    ]
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(
        onnx_file,
        input,
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=["images"],
        output_names=["cls_score"],
    )
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-no-post_process.onnx")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
