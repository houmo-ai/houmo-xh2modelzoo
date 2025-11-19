import argparse
from pathlib import Path

import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
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
    out_hmonnx_file: str = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    quant_config = create_quant_config(quant_scheme)

    convert_onnx_to_hmonnx(
        onnx_file,
        [torch.randn(1, 3, 640, 640, dtype=torch.float32)],
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=["images"],
        output_names=["outs"],
    )
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    session.step = 0
    session(torch.randn(1, 3, 640, 640, dtype=torch.float16).to(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default="/data01/home/xuchen/xh2/xh2_model_zoo/data/models/yolo11m.sim.onnx"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    args = parser.parse_args()
    main(args)
