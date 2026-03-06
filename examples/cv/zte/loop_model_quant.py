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
)
import numpy as np


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

    logger = get_root_logger()
    # try:
    logger.info("开始将 ONNX 转换为 hmonnx...")
    
    convert_onnx_to_hmonnx(
        onnx_file,
        [torch.randn(32, 9, 13, dtype=torch.float16)],
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
    input_data = torch.randn(32, 9, 13, dtype=torch.float16).to(device)
    session(input_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="weights/zte/model_hexiaoxi_sim_unloop_sim.onnx")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--input-dtype", type=str, default="fp16", help="input dtype, default is fp16")
    args = parser.parse_args()
    main(args)
