from pathlib import Path
import torch
import transformers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from xhquant.api import xhquant_init
from xhquant.utils.version_utils import digit_version
from xh_model_zoo.utils.config import Config, ConfigDict
from ..utils.logger import get_root_logger, xhquant_llm_init_logger


def xhquant_llm_init(log_file=None, debug=False, file_mode="w"):
    xhquant_log_file = None
    if log_file is not None:
        log_fname = Path(log_file).stem
        log_suffix = Path(log_file).suffix
        xhquant_log_name = f"{log_fname}_xhquant{log_suffix}"
        xhquant_log_file = str(Path(log_file).with_name(xhquant_log_name))

    xhquant_init(xhquant_log_file, debug=debug)
    xhquant_llm_init_logger(log_file, "DEBUG" if debug else "INFO", "xhquant_llm", file_mode=file_mode)
    logger = get_root_logger()

    logger.info(f"transformers version: {transformers.__version__}")
    if digit_version(transformers.__version__) < digit_version("4.45.0"):
        logger.warning("transformers version is less than 4.45.0, which may cause some issues.")


__all__ = [
    "Config",
    "ConfigDict",
    "xhquant_llm_init",
    "get_root_logger",
]


def decode_next_token(tokenizer: PreTrainedTokenizerBase, logits: torch.Tensor):
    # logits: (batch_size, 1, vocab_size)
    next_token_id = torch.argmax(logits, dim=-1)
    next_token_str = tokenizer.batch_decode(next_token_id, skip_special_tokens=True)
    return next_token_id, next_token_str
