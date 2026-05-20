import argparse
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    ResizerScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

MEAN = [0, 0, 0]
STD = [255.0, 255.0, 255.0]
INPUT_NAME = "onnx::Add_0"


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    out_hmonnx_file = (
        work_dirs / "hmonnx" / f"{onnx_name}_{quant_type}_{target_device}.onnx"
    )
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file_str = str(out_hmonnx_file)

    xhquant_init(None, debug=False)
    logger = get_root_logger()

    MODEL_IN_H = 128
    MODEL_IN_W = 128
    RESIZER_IN_H = 1080
    RESIZER_IN_W = 1920

    input_ppc_config = [
        ResizerScheme(
            size=(MODEL_IN_H, MODEL_IN_W),
            mode="bilinear",
            align_corners=False,
            fmt="yuv420",
            int_trans=True,
            crop_size=[RESIZER_IN_H, RESIZER_IN_W],
            crop_offset=[0, 0],
            pad_size=[0, 0, 0, 0],
            pad_value=114,
            mean=[v / 255.0 for v in MEAN],
            std=[v / 255.0 for v in STD],
            dynamic_crop=True,
            model_inp_fmt="rgb",
        ).to_dict()
    ]
    quant_scheme = QuantScheme(
        target_device=target_device,
        quant_type=quant_type,
        input_ppc_config=input_ppc_config,
    )
    quant_config = create_quant_config(quant_scheme)

    convert_onnx_to_hmonnx(
        onnx_file,
        [
            torch.randint(
                low=0,
                high=255,
                size=(1, 3, RESIZER_IN_H, RESIZER_IN_W),
                dtype=torch.uint8,
            ),
            torch.tensor(
                [
                    [
                        0,
                        0,
                        RESIZER_IN_H,
                        RESIZER_IN_W,
                        MODEL_IN_H,
                        MODEL_IN_W,
                        0,
                        0,
                        0,
                        0,
                    ]
                ],
                dtype=torch.int32,
            ),
        ],
        target_device,
        out_hmonnx_file_str,
        quant_config=quant_config,
        input_names=[INPUT_NAME, f"resizer_crop_{INPUT_NAME}"],
        output_names=["703"],
    )

    logger.info(
        f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file_str}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=str,
        default="vest_cls_20260403_opset13.onnx",
    )
    parser.add_argument(
        "--quant-type", default="w8a8_sefp", help="quant type, default is w8a8_sefp"
    )
    args = parser.parse_args()
    main(args)
