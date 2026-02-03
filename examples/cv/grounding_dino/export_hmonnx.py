# Copyright 2025 HOUMO AI
# File: groundingdino_export.py

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

def get_dummy_inputs(device="cpu", dtype_img=torch.float32):
    """
    构造 Grounding DINO 需要的 5 个固定输入
    Shape 必须与导出时 static_800x800.onnx 保持一致
    """
    # 1. Image: [1, 3, 800, 800]
    img = torch.randn(1, 3, 800, 1200, dtype=dtype_img, device=device)
    
    # 2. Text Inputs: 固定长度 256
    seq_len = 256
    # 模拟随机的 input_ids (Bert Vocab size ~30522)
    input_ids = torch.randint(0, 30522, (1, seq_len), dtype=torch.int32, device=device)
    # attention_mask 全 1
    
    attention_mask = torch.ones((1, seq_len), dtype=torch.int32, device=device)
    # position_ids: 0, 1, 2, ... 255
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)
    # token_type_ids 全 0
    token_type_ids = torch.zeros((1, seq_len), dtype=torch.int32, device=device)
    
    # 返回列表，顺序必须与 ONNX 导出时的 input_names 一致
    return [img, input_ids, attention_mask,token_type_ids]

def main(args):
    # 1. 路径设置
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file_str = str(out_hmonnx_file)

    # 2. 初始化 xhquant
    xhquant_init(None, debug=args.debug)

    # 3. 配置量化参数
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    # 4. 准备转换用的输入 (FP32)
    # 注意：这里生成的 dummy inputs 用于校准/Tracing，通常使用 FP32
    convert_inputs = get_dummy_inputs(device="cpu", dtype_img=torch.float32)

    print(f"Starting conversion for {onnx_file}...")
    
    # 5. 执行转换
    convert_onnx_to_hmonnx(
        onnx_file,
        convert_inputs, # 传入包含5个Tensor的列表
        DeviceType.XH2a,
        out_hmonnx_file_str,
        quant_config=quant_config,
        # 这里的名字必须和 export_onnx_final.py 里指定的 input_names 完全一致
        input_names=["image", "input_ids", "attention_mask", "token_type_ids"],
        output_names=["pred_logits", "pred_boxes"],
    )
    
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file_str}")

    # 6. 运行 Golden Inference (验证)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # session = HMONNXGoldenInference(out_hmonnx_file_str)
    # session.to(device)
    # session.save_golden = True
    # session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    # session.step = 0
    
    # inference_inputs = get_dummy_inputs(device=device, dtype_img=torch.float16)
    
    # print("Running Golden Inference...")
    # # 使用 * 解包列表，传入多个参数
    # session(*inference_inputs)
    # print("Golden Inference Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径指向你刚才导出的静态 ONNX
    parser.add_argument(
        "--onnx", type=str, default="/data01/home/chenzx/project/xh2modelzoo/GroundingDINO/outputs/groundingdino.onnx"
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type")
    args = parser.parse_args()
    main(args)