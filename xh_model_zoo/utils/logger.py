from enum import Enum
from typing import Union

from xhquant.utils.logging import MMLogger

_root_logger = None


def get_root_logger():
    global _root_logger
    if _root_logger is None:
        raise ValueError("Logger has not been initialized yet.")
    return _root_logger


def xhquant_llm_init_logger(
    log_file=None, log_level: Union[int, str] = "INFO", name="xhquant_llm", file_mode="w", **kwargs
):
    global _root_logger
    log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
    log_cfg.setdefault("name", name)
    log_cfg.setdefault("logger_name", name)
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.

    log_cfg.setdefault("file_mode", file_mode)
    _root_logger = MMLogger.get_instance(**log_cfg)  # type: ignore
    return _root_logger
