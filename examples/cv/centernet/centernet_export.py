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


# 2. 新增一个函数，专门用于设置日志记录
def setup_logging(log_file_path: Path, debug_mode: bool):
    """
    配置日志系统，使其同时输出到控制台和文件。

    Args:
        log_file_path (Path): 日志文件的完整路径。
        debug_mode (bool): 是否启用调试模式。如果为 True，控制台和文件都记录 DEBUG 级别；
                           否则控制台记录 INFO 级别，文件仍然记录 DEBUG 级别。
    """
    # 根据 debug 参数设置日志级别
    console_log_level = logging.DEBUG if debug_mode else logging.INFO
    file_log_level = logging.DEBUG  # 文件中始终记录最详细的 DEBUG 级别

    # 获取根 logger
    root_logger = logging.getLogger()
    # 设置 logger 的总级别为最低级别，以确保所有消息都能被 handler 接收
    root_logger.setLevel(logging.DEBUG)

    # 清除任何可能已经存在的 handlers，避免日志重复输出
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 创建一个 Formatter，定义日志的输出格式
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 配置控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # 配置文件 Handler，用于将日志写入文件
    # 使用 'a' 模式，表示追加日志，不清空旧文件
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    root_logger.info(f"日志系统已配置。日志将被保存到: {log_file_path}")


def patch_leakyrelu_in_onnx(model_path: str):
    """
    加载一个ONNX模型，将其中的 'LeakyRelu' 算子全部重命名为 'LeakyReLU'，
    并覆盖保存原文件。

    Args:
        model_path (str): 需要被修改的 ONNX 模型文件路径。
    """
    logger = get_root_logger()
    try:
        model = onnx.load(model_path)
    except FileNotFoundError:
        logger.error(f"模型文件未找到: {model_path}")
        return

    node_changed_count = 0
    for node in model.graph.node:
        if node.op_type == "LeakyRelu":
            logger.info(f"发现节点 '{node.name}' 的算子类型为 'LeakyRelu'，正在修改为 'LeakyReLU'...")
            node.op_type = "LeakyReLU"
            node_changed_count += 1

    if node_changed_count > 0:
        logger.warning(f"共修改了 {node_changed_count} 个节点。正在覆盖保存模型...")
        onnx.save(model, model_path)
        logger.info(f"模型已成功修复并保存至: {model_path}")
    else:
        logger.info("模型中未发现 'LeakyRelu' 算子，无需修改。")


def main(args):
    # --- 日志配置部分 ---
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)

    # 3. 在执行任何操作前，先设置好日志
    log_file = work_dirs / f"{onnx_name}_quant.log"
    setup_logging(log_file, args.debug)
    # --------------------

    # 获取 logger
    logger = get_root_logger()

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{quant_type}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)

    # xhquant_init 可能会影响日志配置，所以我们在它之前完成自己的设置
    xhquant_init(None, debug=args.debug)

    output_names = ["boxes", "confs"]

    # try:
    logger.info("开始将 ONNX 转换为 hmonnx...")
    convert_onnx_to_hmonnx(
        onnx_file,
        [torch.randn(1, 3, 512, 512, dtype=torch.float32)],
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        # input_names=["input"],
        # output_names=output_names,
    )
    logger.info(f"原始 hmonnx 文件已成功生成: {out_hmonnx_file}")

    # logger.warning("=" * 20 + " 开始执行 'LeakyRelu' 算子修复 " + "=" * 20)
    # patch_leakyrelu_in_onnx(out_hmonnx_file)
    # logger.warning("=" * 20 + " 'LeakyRelu' 算子修复完成 " + "=" * 20)

    logger.info(f"最终转换流程完成，修复后的 hmonnx 文件位于: {out_hmonnx_file}")

    # except Exception as e:
    #     # 捕获异常并记录到日志中
    #     logger.error("程序执行过程中发生严重错误！")
    #     logger.exception(e)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    session.step = 0
    session(torch.randn(1, 3, 512, 512, dtype=torch.float16).to(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=str,
        default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco_d.onnx",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    args = parser.parse_args()
    main(args)
