# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass, field

from xhquant.api import QuantScheme

from ...llm_convert_config import BaseConvertConfig


@dataclass
class StellaMRLConvertConfig(BaseConvertConfig):
    batch_size: int = 1
    context_length: int = 512
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    output_normalized: bool = False

    def to_dict(self):
        return dataclasses.asdict(self)
