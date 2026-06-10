from dataclasses import dataclass
from typing import Any


@dataclass
class QuantResult:
    hf_model_dir: str # 原始HF模型路径
    skipped: bool = False # 量化是否被跳过。如果被跳过，相当于直接从原始HF模型导出hmonnx
    quanted_model_dir: str | None = None # 量化模型路径
    is_quant_weight_format: bool = False # 产物是否为量化权重文件，这种情况无法直接加载量化模型到内存中


@dataclass
class ExportResult:
    # exported_model_dir: str # 导出的hmonnx模型路径
    work_dir: str # 导出的工作目录，包含log，config，hmonnx产物文件夹，
    config_file: str
    meta: Any | None = None
