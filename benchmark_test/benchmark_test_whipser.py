import os
from typing import Literal
import allure
import pytest
from pathlib import Path
import onnx
import sys
import subprocess

@allure.title("Whisper测试")
@pytest.mark.parametrize(
    "trace_method",
    [
        "FX",
        # "DFX",
        # "ONNX",
    ],
)
@pytest.mark.parametrize("w_bit", [8]) # 
@pytest.mark.parametrize("a_bit", [8]) # , 16
def test_whisper(trace_method: Literal["FX", "DFX", "ONNX"], w_bit: int, a_bit: int):
    # onnx_file = None
    # if onnx_file is None:
    #     allure.attach(str(f"该模型未完成迁移"), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)
    #     return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from xhquant.api import (
        DeviceType,
        HMONNXGoldenInference,
        HMONNXInference,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        get_root_logger,
        xhquant_init,
    )
    import onnxruntime as ort

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = Path(__file__).parent.parent.resolve()  
    model_file = os.path.join(current_dir, "examples/llm/whisper/hmonnx_export.py")

    # ========================== export model ==========================
    if w_bit < 8:
        fp_mode="ssfp"
        # TODO
    else:
        fp_mode="sefp"
        command = [
            sys.executable,  # 使用当前Python解释器，避免环境不一致
            model_file,  # 注意：你原命令里多了一个.py后缀，需修正
            "--model", "/data02/datasets/whisper_medium",
            "--gen_golden",
            # "--context-length", "2048",
            # "--input-sequence-length", "2048",
            # "--quant-type", "w8a8h1_sefp",
            # "--num_logits_to_keep", "0",
        ]
    print(model_file)

    try:
        result = subprocess.run(
            command,
            check=True,  # 命令执行失败时抛出异常
            stdout=None,
            stderr=None,
            encoding="utf-8"
        )
        print("Export 脚本执行成功！")
        # print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败！错误信息：{e.stdout}")

    # ========================== eval model ==========================
    # TODO
    acc = "导出成功"

    allure.attach(trace_method, "TraceMethod", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(w_bit), "Wbit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(a_bit), "Abit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(acc), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)

    # if need_rollback:
    #   rollback_transformers_version(original_version)
    # rollback_transformers_version("4.57.3")


if __name__ == "__main__":
    test_whisper("ONNX", w_bit=8, a_bit=8)
