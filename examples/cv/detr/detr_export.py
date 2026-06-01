# Copyright 2025 HOUMO AI
#
# File: detr_export.py
# Description:
#   Example script: cv/detr/detr_export.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# detr_export.py
import argparse
from pathlib import Path

import onnx
import torch
from onnx import version_converter
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def convert_onnx_opset(onnx_file_path: str, target_opset: int):
    """
    检查ONNX模型的Opset版本，并在必要时进行转换。
    转换后的模型会覆盖原始文件。
    """
    print("--- [步骤 0] 开始检查并转换 ONNX Opset 版本 ---")
    try:
        model = onnx.load(onnx_file_path)
        current_opset = model.opset_import[0].version

        if current_opset == target_opset:
            print(f"模型已经是目标 Opset 版本 {target_opset}，无需转换。")
            return

        print(f"当前 Opset: {current_opset}，目标 Opset: {target_opset}。开始转换...")
        converted_model = version_converter.convert_version(model, target_opset)
        onnx.save(converted_model, onnx_file_path)
        print(f"成功将模型转换为 Opset {target_opset} 并覆盖原文件: {onnx_file_path}")

    except Exception as e:
        print(f"ONNX Opset 版本转换失败: {e}")
        raise e
    finally:
        print("--- [步骤 0] ONNX Opset 检查完成 ---\n")


def main(args):
    xhquant_init(None, debug=args.debug)

    batch_size = args.batch_size
    onnx_file = args.onnx
    imgsz = args.imgsz
    onnx_name = Path(onnx_file).stem
    quant_type = args.quant_type

    convert_onnx_opset(onnx_file, target_opset=13)

    # 1. 准备输出路径
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = "XH2a"
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}c_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    # 2. 准备量化配置
    print(f"Using quantization type: {quant_type}")
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    # 此处可以根据需要加入更复杂的量化配置，如你同事脚本中的DetrConfig

    # 3. 创建与静态模型批次大小和输入尺寸匹配的随机输入
    print(f"Creating a random input tensor with shape: [{batch_size}, 3, {imgsz}, {imgsz}]")
    dummy_input = torch.randn(batch_size, 3, imgsz, imgsz, dtype=torch.float32)

    # 4. --- API 调用 ---
    # 调用转换API，注意为DETR指定正确的输入输出节点名
    print("--- [步骤 1] 开始使用 xhquanttool 转换模型 ---")
    convert_onnx_to_hmonnx(
        onnx_file,
        [dummy_input],
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=["input"],  # 假设DETR模型的输入名为'input'
        output_names=["pred_logits", "pred_boxes"],  # DETR的两个主要输出
    )

    logger = get_root_logger()
    logger.info(f"HMONNX model converted successfully to: {out_hmonnx_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / f"hmonnx/golden_{quant_type}"
    session.step = 0
    session(dummy_input.to(torch.float16).to(device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DETR ONNX to HMONNX with specified quantization.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the static ONNX model file.")
    parser.add_argument("--imgsz", type=int, default=1312, help="The static input image size (height and width).")
    parser.add_argument("--batch-size", type=int, default=1, help="The static batch size of the ONNX model.")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="Quantization type (e.g. w8a8_sefp).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()
    main(args)
