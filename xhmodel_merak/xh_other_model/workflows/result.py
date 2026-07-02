from dataclasses import dataclass
from typing import Any


"""
如果量化被跳过，按照如下方式构造quant_result
>>> quant_result = QuantResult(
        raw_model_dir = self.model_dir,
        skipped = True,
    )
"""
@dataclass
class QuantResult:
    raw_model_dir: str  # 原始模型路径
    skipped: bool = False  # 量化是否被跳过。如果被跳过，相当于直接从原始模型导出hmonnx
    quanted_model_dir: str | None = None  # 量化模型路径，None表示量化被跳过
    is_quant_weight_format: bool = False  # 产物是否为量化权重文件，这种情况无法直接加载量化模型到内存中。一般情况下使用默认值即可
    meta: Any | None = None  # 元数据。如有额外信息需要存储，使用此字段


@dataclass
class ExportResult:
    work_dir: str  # 导出的工作目录，包含log，config，hmonnx等产物
    config_file: str  # 导出时要求将override后的config文件dump到磁盘上，该成员用于记录config文件路径
    meta: Any | None = None  # 元数据。如有额外信息需要存储，使用此字段
