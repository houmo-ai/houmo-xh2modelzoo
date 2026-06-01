import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

from xhquant.api import ConfigDict, QuantScheme, create_quant_config

from xh_model_zoo_develop.core import ConverterConfig


@dataclass
class BaseLLMConverterConfig(ConverterConfig):
    """General Config"""  # 一般用来和 args 进行合并

    batch_size: int = 1
    context_length: int = 2048  # 上下文长度
    input_sequence_length: int = 256  # prefill阶段的输入的最大序列长度

    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    quant_weight: Optional[str] = None

    """Only For Debug"""
    # Quant Config
    mix_search: bool = False
    quarot = False
    gptq = False

    # Wrap Config
    only_first_block: bool = False
    num_logits_to_keep: int = 1  # 1表示取最后一个,
    use_cache: bool = True
    cache_axis: int = 2

    # Eval Config
    eval_ppl: bool = False  # TODO not used now

    def __post_init__(self):
        if isinstance(self.quant_scheme, dict):
            self.quant_scheme = QuantScheme(**self.quant_scheme)

    def merge_with_args(self, args: argparse.Namespace):
        for key, value in args.__dict__.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_wrap_cfg(self) -> ConfigDict:
        wrap_cfg = ConfigDict(
            dict(
                batch_size=self.batch_size,
                max_sequence_length=self.context_length,
                input_sequence_length=self.input_sequence_length,
                use_cache=self.use_cache,
                num_logits_to_keep=self.num_logits_to_keep,
                only_first_block=self.only_first_block,
            )
        )
        return wrap_cfg

    def to_wrap_quant_cfg(self) -> Tuple[ConfigDict, ConfigDict]:
        wrap_cfg = ConfigDict(
            dict(
                batch_size=self.batch_size,
                max_sequence_length=self.context_length,
                input_sequence_length=self.input_sequence_length,
                use_cache=self.use_cache,
                num_logits_to_keep=self.num_logits_to_keep,
                only_first_block=self.only_first_block,
                kv_cache=dict(
                    cache_axis=self.cache_axis,
                ),
            )
        )
        quant_cfg = ConfigDict(create_quant_config(self.quant_scheme))
        return wrap_cfg, quant_cfg
