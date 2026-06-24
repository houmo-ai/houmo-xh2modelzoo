from dataclasses import dataclass
from typing import Optional

from ..qwen3_5_moe.qwen3_5_moe_convert_config import Qwen3_5MoeConvertConfig


@dataclass
class Qwen3_5MoePruneConvertConfig(Qwen3_5MoeConvertConfig):
    threshold: float = 0.0
    s_scalar_path: Optional[str] = None
