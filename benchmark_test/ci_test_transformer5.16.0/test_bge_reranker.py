import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Literal

import pytest
import requests
from _allure_compat import allure


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
        print("正在下载...")
        response = requests.get(download_url, stream=True, timeout=(10, 600))
        response.raise_for_status()

        with open(tar_gz_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"下载完成: {tar_gz_path}")

        # 解压
        print("正在解压...")
        with tarfile.open(tar_gz_path, "r:gz") as tar:
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
@pytest.mark.parametrize("w_bit", [8])
@pytest.mark.parametrize("a_bit", [8])
def test_qwen2(trace_method: Literal["FX", "DFX", "ONNX"], w_bit: int, a_bit: int):
    assert (w_bit, a_bit) == (8, 8), "PR CI keeps BGE on the full W8A8 HMONNX/golden path"
    current_dir = Path(__file__).parent.parent.parent.resolve()
    model_file = os.path.join(current_dir, "examples/llm/bge_reranker/bge_reranker_xh2a_export_hmonnx.py")

    # ========================== 检查并下载模型 ==========================
    model_path = "work_dirs/fp_model/bge_reranker_base"
    download_url = "http://10.10.1.53:8082/artifactory/model_zoo2/model_zoo2/fp_models/bge.tar.gz"

    check_and_download_model(model_path, download_url)

    # ========================== export model ==========================
    command = [
        sys.executable,
        model_file,
        "--model",
        model_path,
        "--quant-type",
        "w8a8h1_sefp",
    ]
    print(model_file)

    subprocess.run(command, check=True)
    print("Export 脚本执行成功！")

    acc = "导出成功"

    allure.attach(trace_method, "TraceMethod", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(w_bit), "Wbit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(a_bit), "Abit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(acc), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)


if __name__ == "__main__":
    test_qwen2("ONNX", w_bit=8, a_bit=8)
