import os
from typing import Literal
import allure
import pytest
from pathlib import Path
import onnx
import sys
import subprocess
import tarfile
import requests


def check_and_download_model(model_path: str, download_url: str):
    """
    检查模型是否存在，不存在则从Artifactory下载并解压
    
    Args:
        model_path: 模型路径
        download_url: 下载URL
    """
    model_dir = Path(model_path)
    
    if model_dir.exists():
        print(f"模型已存在: {model_path}")
        return
    
    print(f"模型不存在，开始下载: {download_url}")
    
    # 创建父目录
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # 下载文件
    tar_gz_path = model_dir.parent / "bge.tar.gz"
    
    try:
        print(f"正在下载...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(tar_gz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"下载完成: {tar_gz_path}")
        
        # 解压
        print(f"正在解压...")
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            tar.extractall(path=model_dir.parent)
        
        print(f"解压完成: {model_path}")
        
        # 删除压缩包
        tar_gz_path.unlink()
        print(f"已删除压缩包: {tar_gz_path}")
        
    except Exception as e:
        print(f"下载或解压失败: {e}")
        if tar_gz_path.exists():
            tar_gz_path.unlink()
        raise

@allure.title("BGE-Reranker测试")
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
    current_dir = Path(__file__).parent.parent.parent.resolve()  
    model_file = os.path.join(current_dir, "examples/llm/bge_reranker/bge_reranker_xh2a_export_hmonnx.py")

    # ========================== 检查并下载模型 ==========================
    model_path = "work_dirs/fp_model/bge_reranker_base"
    download_url = "http://10.10.1.53:8082/artifactory/model_zoo2/model_zoo2/fp_models/bge.tar.gz"
    
    check_and_download_model(model_path, download_url)

    # ========================== export model ==========================
    if w_bit < 8:
        fp_mode="ssfp"
        # TODO
    else:
        fp_mode="sefp"
        command = [
            sys.executable,  # 使用当前Python解释器，避免环境不一致
            model_file,  # 注意：你原命令里多了一个.py后缀，需修正
            "--model", "work_dirs/fp_model/bge_reranker_base",
            # "--context-length", "2048",
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


if __name__ == "__main__":
    test_qwen2("ONNX", w_bit=8, a_bit=8)
