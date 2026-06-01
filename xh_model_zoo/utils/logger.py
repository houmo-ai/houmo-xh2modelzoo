# Copyright 2025 HOUMO AI
#
# File: logger.py
# Description:
#   Logger initialization and management utilities.
#   This module provides functions for initializing and accessing the root logger.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Union

from xhquant.utils.logging import MMLogger

_root_logger = None


def get_root_logger():
    global _root_logger
    if _root_logger is None:
        raise ValueError("Logger has not been initialized yet.")
    return _root_logger


def xh2modelzoo_init_logger(
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


# Backward-compatible alias for legacy API callers.
xhquant_llm_init_logger = xh2modelzoo_init_logger
