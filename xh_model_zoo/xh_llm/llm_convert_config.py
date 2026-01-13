import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from xhquant.api import QuantScheme


@dataclass
class BaseConvertConfig:
    """Base configuration class for model conversion."""

    pass


@dataclass
class LLMConvertConfig(BaseConvertConfig):
    """Configuration class for Large Language Model conversion.

    Attributes:
        batch_size: Batch size for inference
        context_length: Maximum context length
        input_sequence_length: Maximum input sequence length for prefill stage
        quant_scheme: Quantization scheme
        quant_weight: Path to quantization weight file
        eval_ppl: Whether to evaluate perplexity
    """

    batch_size: int = 1
    context_length: int = 2048
    input_sequence_length: int = 256
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    quant_weight: Optional[str] = None
    eval_ppl: bool = False

    def to_dict(self):
        """Convert configuration to dictionary format."""
        return dataclasses.asdict(self)
