# ================================================================== #
#  File: deepseek_v4_convert_config.py                                #
#  Description:                                                       #
#    DeepSeek-V4 convert configuration dataclass.                    #
# ================================================================== #

from dataclasses import dataclass
from typing import Optional

from xh_model_zoo.xh_llm.llm_convert_config import LLMConvertConfig


@dataclass
class DeepseekV4ConvertConfig(LLMConvertConfig):
    num_logits_to_keep: Optional[int] = 1
