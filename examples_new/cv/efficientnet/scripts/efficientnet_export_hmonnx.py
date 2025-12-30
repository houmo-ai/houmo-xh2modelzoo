import argparse
from pathlib import Path

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


def main(args):
    # 初始化 xhquant 环境
    xhquant_init(None, debug=args.debug)

    batch_size = args.batch_size
    onnx_file = args.onnx
    quant_type = args.quant_type  # 将 quant_type 的获取提前

    # 从 ONNX 模型动态读取输入形状
    print(f"Loading ONNX model from: {onnx_file}")
    onnx_model = onnx.load(onnx_file)
    input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]

    # 根据命令行参数重写 batch_size
    input_shape[0] = batch_size
    str_shape = "x".join([str(dim) for dim in input_shape])
    print(f"Target input shape set to: {input_shape}")

    # --- 自动化构建输出路径 ---
    onnx_name = Path(onnx_file).stem
    # 路径名中包含模型名、形状、量化类型等信息，方便管理
    output_folder_name = f"{onnx_name}_{quant_type}"
    work_dirs = Path("work_dirs/efficientnet") / output_folder_name
    work_dirs.mkdir(exist_ok=True, parents=True)

    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / f"{output_folder_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)

    print(f"Output HMONNX will be saved to: {out_hmonnx_file}")

    # --- 模型量化转换 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用目标形状创建用于量化校准的随机输入数据
    input_tensor = torch.randn(input_shape, dtype=torch.float32)

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    # 调用核心 API，将 ONNX 转换为量化后的 HMONNX
    convert_onnx_to_hmonnx(
        onnx_file,
        [input_tensor],
        target_device,
        out_hmonnx_file,
        quant_config=quant_config,
        # 确保这里的 input/output names 和您导出 ONNX 时设置的一致
        input_names=["images"],
        output_names=["cls_score"],
    )

    # --- 生成 Golden Data 用于后续精度验证 ---
    print("Generating golden data...")
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / "golden"
    session.step = 0

    # 运行一次推理，输入和输出将被自动保存到 golden_dir
    session(input_tensor.half().to(device))

    logger = get_root_logger()
    logger.info(f"Successfully converted ONNX to HMONNX: {out_hmonnx_file}")
    logger.info(f"Golden data saved in: {session.golden_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize an ONNX model using xhquant.")
    # 将默认的 ONNX 文件路径修改为 EfficientNet B4 的路径
    parser.add_argument(
        "--onnx",
        type=str,
        default="data/models/efficientnet/efficientnet_b4.onnx",
        help="Path to the input ONNX model file.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--quant-type", default="w8a16_sefp", help="Quantization type, e.g., w8a8h1_sefp, w8a8.")
    parser.add_argument("--batch-size", type=int, default=1, help="Target batch size for quantization.")

    args = parser.parse_args()
    main(args)
