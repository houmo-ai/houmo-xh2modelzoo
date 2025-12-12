from dataclasses import dataclass
from typing import Tuple

from ...llm_convert_config import LLMConvertConfig


@dataclass
class MinicpmoTTSDVAEConvertConfig(LLMConvertConfig):
    audio: str = ""
    video: str = ""
    debug: bool = False
    valid: bool = False
    image_slice_max_size: Tuple[int, int] = (40, 40)

