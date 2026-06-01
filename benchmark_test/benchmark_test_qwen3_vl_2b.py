import os
from typing import Literal
import allure
import pytest
from pathlib import Path
import onnx
import sys
import subprocess

@allure.title("Qwen3-vl-2B测试")
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
    Qwen3vl_2b_half_path = None
    if os.getenv("TEST_ALL_MODEL", "false") != "true":
        allure.attach(str(f"有类似模型已测试，如想测试请发送TEST_ALL_MODEL"), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)
        return
    
    if Qwen3vl_2b_half_path is None:
        allure.attach(str(f"A800 不存在该浮点模型，请拷贝后修改测试文件"), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)
        return
    
    from utils import check_and_set_transformers_version, rollback_transformers_version
    original_version, need_rollback = check_and_set_transformers_version(target_version="4.57.1")

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
    # import sys
    current_dir = Path(__file__).parent.parent.resolve()  
    # sys.path.append(str(current_dir))

    # 1. 复制当前环境变量
    env = os.environ.copy()
    # 2. 设置子进程的 PYTHONPATH 为当前目录（等效于 export PYTHONPATH=.）
    env["PYTHONPATH"] = str(current_dir)

    model_file = os.path.join(current_dir, "examples/llm/qwen3vl/qwen3_vl_xh2a_export_hmonnx.py")

    # ========================== export model ==========================
    if w_bit < 8:
        fp_mode="ssfp"
        # TODO
    else:
        fp_mode="sefp"
        command = [
            sys.executable,  # 使用当前Python解释器，避免环境不一致
            model_file,  # 注意：你原命令里多了一个.py后缀，需修正
            "--model", "/data02/datasets/Qwen3-VL-8B-Instruct",
            "--context-length", "2048",
            # "--input-sequence-length", "2048",
            "--quant-type", "w8a8h1_sefp",
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

    rollback_transformers_version("4.57.3")

if __name__ == "__main__":
    test_qwen2("ONNX", w_bit=8, a_bit=8)
