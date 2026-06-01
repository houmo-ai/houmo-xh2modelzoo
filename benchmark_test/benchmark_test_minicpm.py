import os
from typing import Literal
import allure
import pytest
from pathlib import Path
import onnx
import sys
import subprocess

@allure.title("Minicpm 模型测试")
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
def test_qwen2(trace_method: Literal["FX", "DFX", "ONNX"], w_bit: int, a_bit: int):
    minicpm_data_file = None
    if minicpm_data_file is None:
        allure.attach(str(f"该模型未完成迁移"), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)
        return

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
    model_file = os.path.join(current_dir, "examples/llm/qwen3_legacy/qwen3_legacy_xh2a_export_hmonnx.py")

    # ========================== export model ==========================
    if w_bit < 8:
        fp_mode="ssfp"
        # TODO
    else:
        fp_mode="sefp"
        command = [
            sys.executable,  # 使用当前Python解释器，避免环境不一致
            model_file,  # 注意：你原命令里多了一个.py后缀，需修正
            "--model", "/data02/datasets/Qwen3-8B",
            "--context-length", "2048",
            "--input-sequence-length", "2048",
            "--quant-type", "w8a8h1_sefp",
            "--num_logits_to_keep", "0",
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
    model_file = os.path.join(current_dir, "examples/llm/qwen3_legacy/qwen3_legacy_eval.py")
    meta_path = os.path.join(current_dir, "work_dirs/Qwen3-8B-XH2a-2k-w8a8h0_sefp/meta.json")
    hf_model_path = os.path.join(current_dir, "work_dirs/Qwen3-8B-XH2a-2k-w8a8h0_sefp/")
    eval_ppl_path = os.path.join(current_dir, "work_dirs/Qwen3-8B-XH2a-2k-w8a8h0_sefp/eval_ppl.txt")
    command = [
        sys.executable,  # 使用当前Python解释器，避免环境不一致
        model_file,  # 注意：你原命令里多了一个.py后缀，需修正
        "--config", meta_path,
        "--hf-model", hf_model_path,
        "--eval_ppl", eval_ppl_path,
    ]

    result = subprocess.run(
        command,
        check=True,  # 命令执行失败时抛出异常
        stdout=None,
        stderr=None,
        encoding="utf-8"
    )

    with open(f"{eval_ppl_path}", "r", encoding="utf-8") as f:
        read_content = f.read()

    allure.attach(trace_method, "TraceMethod", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(w_bit), "Wbit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(a_bit), "Abit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(read_content), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)


if __name__ == "__main__":
    test_qwen2("ONNX", w_bit=8, a_bit=8)
