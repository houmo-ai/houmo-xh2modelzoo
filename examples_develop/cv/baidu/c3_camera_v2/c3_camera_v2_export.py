import argparse
from pathlib import Path

import torch
from xhquant.api import (
    Config,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    query_device,
    xhquant_init,
)


def main(args):
    cfg = Config.fromfile(args.config)
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    cfg_name = f"{onnx_name}"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    cfg.work_dir = str(work_dir)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"

    xhquant_init(log_file, debug=args.debug)

    target_device = query_device(cfg.target_device)
    logger = get_root_logger()
    logger.info(f"Target device: {target_device}")
    # out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    # out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    # out_hmonnx_file: str = str(out_hmonnx_file)

    # input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
    # quant_type = args.quant_type
    # quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    # quant_config = create_quant_config(quant_scheme)
    # convert_onnx_to_hmonnx(
    #     onnx_file,
    #     [input],
    #     DeviceType.XH2a,
    #     out_hmonnx_file,
    #     quant_config=quant_config,
    #     input_names=["images"],
    #     output_names=["cls_score"],
    # )
    # logger = get_root_logger()
    # logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx", type=str, default="data/models/baidu/modified_model_c3_camera_v2_960x544_si.onnx", help="onnx file"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--config", default="configs/xh2a/cv/baidu/c3_camera_v2_xh2a_224x224.py", help="config file")
    args = parser.parse_args()
    main(args)
