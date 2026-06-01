import torch
from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    convert_onnx_to_quanted_model,
    convert_onnx_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    HMONNXGoldenInference
)
import argparse
import os
from pathlib import Path
import onnxruntime as ort
import numpy as np
from PIL import Image
import ast

def tensor_to_image(tensor, save_path="output.png"):
    """
    将形状为 (1, 1, H, W) 的张量保存为图片
    
    Args:
        tensor: 输入张量，形状为 (1, 1, 512, 896)
        save_path: 图片保存路径，支持 png/jpg 等格式
    """
    # 1. 移除批次维度和通道维度 (1,1,512,896) -> (512,896)
    img_tensor = tensor.squeeze()  # 移除所有维度为1的轴
    
    # 2. 转换为 NumPy 数组（如果是 CUDA tensor 先转到 CPU）
    if img_tensor.is_cuda:
        img_np = img_tensor.cpu().numpy()
    else:
        img_np = img_tensor.numpy()
    
    # 3. 归一化到 [0, 255]（关键步骤，避免像素值异常）
    # 如果你的张量已经是 [0,255] 范围，可以跳过这一步
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)  # 归一化到 [0,1]
    img_np = (img_np * 255).astype(np.uint8)  # 转换为 0-255 的整数
    
    # 4. 保存为图片
    img = Image.fromarray(img_np)
    img.save(save_path)
    print(f"图片已保存到: {save_path}")

def main(args):
    device = args.device

    work_dirs = Path(args.output_path) 
    work_dirs.mkdir(exist_ok=True, parents=True)

    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type) # , ops=ops
    quant_config = create_quant_config(quant_scheme)

    det_onnx_path = args.onnx_model_path + "/det.onnx"
    det_input = torch.randn(1, 3, 512, 896) 

    rec_onnx_path = args.onnx_model_path + "/rec.onnx"
    
    # 解析 rec_shape 参数，支持字符串格式 "[6, 3, 48, 320]"
    if isinstance(args.rec_shape, str):
        rec_shape = ast.literal_eval(args.rec_shape)
    else:
        rec_shape = args.rec_shape
    
    rec_input = torch.randn(*rec_shape)

    quant_det_model_path = work_dirs / "hmquant_xh2_paddleocr_det.onnx"
    quant_rec_model_path = work_dirs / "hmquant_xh2_paddleocr_rec.onnx"

    if True:
        quanted_model = convert_onnx_to_hmonnx(
            det_onnx_path,
            [det_input],
            device_type=DeviceType.XH2a,
            out_hmonnx_file=quant_det_model_path,
            quant_config=quant_config,
            input_names=["input"],
            output_names=["output"],
        )

        session = HMONNXGoldenInference(quant_det_model_path)
        session.to(device)
        session.save_golden = True #False
        session.golden_dir = work_dirs / "hmonnx/golden"
        session.golden_dir.mkdir(exist_ok=True, parents=True)
        session.step = 0
        hm_out = session(det_input.half())

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_onnx = ort.InferenceSession(det_onnx_path, providers=providers)
        input_name = session_onnx.get_inputs()[0].name
        outputs = session_onnx.run(None, {input_name:det_input.cpu().numpy()})
        # cos_sim = torch.cosine_similarity(hm_out, torch.from_numpy(outputs[0]).to(hm_out[0].device) ).mean()
        # print("cos_sim:", cos_sim)



    if True:
        quanted_model = convert_onnx_to_hmonnx(
            rec_onnx_path,
            [rec_input],
            device_type=DeviceType.XH2a,
            out_hmonnx_file=quant_rec_model_path,
            quant_config=quant_config,
            input_names=["input"],
            output_names=["output"],
        )

        session = HMONNXGoldenInference(quant_rec_model_path)
        session.to(device)
        session.save_golden = True
        session.golden_dir = work_dirs / "hmonnx/golden"
        session.golden_dir.mkdir(exist_ok=True, parents=True)
        session.step = 0
        hm_out = session(rec_input.half())

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session_onnx = ort.InferenceSession(rec_onnx_path, providers=providers)
        input_name = session_onnx.get_inputs()[0].name
        outputs = session_onnx.run(None, {input_name:rec_input.cpu().numpy()})

        # cos_sim = torch.cosine_similarity(hm_out, torch.from_numpy(outputs[0]).to(hm_out[0].device) ).mean()
        # print("cos_sim:", cos_sim)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--onnx_model_path", type=str, default="data/models/ocr_onnx")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant_weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument("--output_path", type=str, default="work_dirs/paddle_312")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rec_shape", default="[6, 3, 48, 320]", help="rec shape, default is [3, 48, 320]")
    args = parser.parse_args()
    main(args)
