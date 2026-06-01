# Copyright 2025 HOUMO AI
#
# File: logger.py
# Description:
#   Logger implementation with colored output support.
#   This module provides MMLogger class for logging with
#   colored console output and file handlers.
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
import inspect
import logging
import os
import os.path as osp
import pathlib
import sys
import warnings
from enum import Enum
from getpass import getuser
from logging import Logger, LogRecord, handlers
from socket import gethostname
from typing import Dict, Optional, Union

from termcolor import colored

from ..manager import ManagerMixin, _accquire_lock, _release_lock

BASE_PATH = pathlib.Path(inspect.getabsfile(inspect.currentframe()))


class LoggerColor(str, Enum):
    BOLD = "\033[1m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    GREEN = "\033[32m"
    WHITE = "\033[34m"
    YELLOW = "\033[33m"
    NO_COLOR = "\033[0m"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class RelativePathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        filename = record.filename
        try:
            if str(BASE_PATH.parent.parent.parent.parent) in record.pathname:
                filename = str(pathlib.PurePath(record.pathname).relative_to(BASE_PATH.parent.parent.parent.parent))
        except Exception as e:
            pass
        record.filename = filename
        return True


class XHFileHandler(logging.FileHandler):
    def emit(self, record: LogRecord):
        # record.msg = (
        #     record.msg.replace(LoggerColor.RED, "")
        #     .replace(LoggerColor.BOLD, "")
        #     .replace(LoggerColor.BLUE, "")
        #     .replace(LoggerColor.GREEN, "")
        #     .replace(LoggerColor.WHITE, "")
        #     .replace(LoggerColor.YELLOW, "")
        #     .replace(LoggerColor.NO_COLOR, "")
        # )
        super().emit(record)


class FilterDuplicateWarning(logging.Filter):
    """Filter the repeated warning message.

    Args:
        name (str): name of the filter.
    """

    def __init__(self, name: str = "xhquant"):
        super().__init__(name)
        self.seen: set = set()

    def filter(self, record: LogRecord) -> bool:
        """Filter the repeated warning message.

        Args:
            record (LogRecord): The log record.

        Returns:
            bool: Whether to output the log record.
        """
        if record.levelno != logging.WARNING:
            return True

        if record.msg not in self.seen:
            self.seen.add(record.msg)
            return True
        return False


class MMFormatter(logging.Formatter):
    """Colorful format for MMLogger. If the log level is error, the logger will
    additionally output the location of the code.

    Args:
        color (bool): Whether to use colorful format. filehandler is not
            allowed to use color format, otherwise it will be garbled.
        blink (bool): Whether to blink the ``INFO`` and ``DEBUG`` logging
            level.
        **kwargs: Keyword arguments passed to
            :meth:`logging.Formatter.__init__`.
    """

    _color_mapping: dict = dict(ERROR="red", WARNING="yellow", INFO="white", DEBUG="green")

    def __init__(self, color: bool = True, blink: bool = False, **kwargs):
        super().__init__(**kwargs)
        assert not (not color and blink), "blink should only be available when color is True"
        # Get prefix format according to color.
        error_prefix = self._get_prefix("ERROR", color, blink=True)
        warn_prefix = self._get_prefix("WARNING", color, blink=True)
        info_prefix = self._get_prefix("INFO", color, blink)
        debug_prefix = self._get_prefix("DEBUG", color, blink)

        # 增加文件名和行号信息到输出格式中
        self.err_format = f"%(asctime)s {error_prefix} " "%(pathname)s - %(funcName)s - %(lineno)d : " "%(message)s"
        self.warn_format = f"%(asctime)s {warn_prefix} " "%(filename)s:%(lineno)d => " "%(message)s"
        self.info_format = f"%(asctime)s {info_prefix} " "%(filename)s:%(lineno)d => " "%(message)s"
        self.debug_format = f"%(asctime)s {debug_prefix} " "%(filename)s:%(lineno)d => " "%(message)s"

    def _get_prefix(self, level: str, color: bool, blink=False) -> str:
        """Get the prefix of the target log level.

        Args:
            level (str): log level.
            color (bool): Whether to get colorful prefix.
            blink (bool): Whether the prefix will blink.

        Returns:
            str: The plain or colorful prefix.
        """
        if color:
            attrs = ["underline"]
            if blink:
                attrs.append("blink")
            prefix = colored(level, self._color_mapping[level], attrs=attrs)
        else:
            prefix = level
        return prefix

    def format(self, record: LogRecord) -> str:
        """Override the `logging.Formatter.format`` method `. Output the
        message according to the specified log level.

        Args:
            record (LogRecord): A LogRecord instance represents an event being
                logged.

        Returns:
            str: Formatted result.
        """
        if record.levelno == logging.ERROR:
            self._style._fmt = self.err_format
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_format
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_format
        elif record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format

        result = logging.Formatter.format(self, record)
        return result


class MMLogger(Logger, ManagerMixin):
    def __init__(
        self,
        name: str,
        logger_name="xhquant",
        log_file: Optional[str] = None,
        log_level: Union[int, str] = "INFO",
        file_mode: str = "w",
        distributed=False,
        file_handler_cfg: Optional[dict] = None,
    ):
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        # Get rank in DDP mode.
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()
        device_id = _get_device_id()

        # Config stream_handler. If `rank != 0`. stream_handler can only
        # export ERROR logs.
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # `StreamHandler` record month, day, hour, minute, and second
        # timestamp.
        stream_handler.setFormatter(MMFormatter(color=True, datefmt="%m/%d %H:%M:%S"))
        # Only rank0 `StreamHandler` will log messages below error level.
        if global_rank == 0:
            stream_handler.setLevel(log_level)
        else:
            stream_handler.setLevel(logging.ERROR)
        stream_handler.addFilter(FilterDuplicateWarning(logger_name))
        stream_handler.addFilter(RelativePathFilter())
        self.handlers.append(stream_handler)

        if log_file is not None:
            world_size = _get_world_size()
            is_distributed = (log_level <= logging.DEBUG or distributed) and world_size > 1
            if is_distributed:
                filename, suffix = osp.splitext(osp.basename(log_file))
                hostname = _get_host_info()
                if hostname:
                    filename = f"{filename}_{hostname}_device{device_id}_" f"rank{global_rank}{suffix}"
                else:
                    # Omit hostname if it is empty
                    filename = f"{filename}_device{device_id}_" f"rank{global_rank}{suffix}"
                log_file = osp.join(osp.dirname(log_file), filename)
            # Save multi-ranks logs if distributed is True. The logs of rank0
            # will always be saved.
            if global_rank == 0 or is_distributed:
                if file_handler_cfg is not None:
                    assert "type" in file_handler_cfg
                    file_handler_type = file_handler_cfg.pop("type")
                    file_handlers_map = _get_logging_file_handlers()
                    if file_handler_type in file_handlers_map:
                        file_handler_cls = file_handlers_map[file_handler_type]
                        file_handler_cfg.setdefault("filename", log_file)
                        file_handler = file_handler_cls(**file_handler_cfg)
                    else:
                        raise ValueError("`logging.handlers` does not " f"contain {file_handler_type}")
                else:
                    # Here, the default behavior of the official
                    # logger is 'a'. Thus, we provide an interface to
                    # change the file mode to the default behavior.
                    # `FileHandler` is not supported to have colors,
                    # otherwise it will appear garbled.
                    # file_handler = logging.FileHandler(log_file, file_mode)
                    file_handler = XHFileHandler(log_file, file_mode)

                # `StreamHandler` record year, month, day hour, minute,
                # and second timestamp. file_handler will only record logs
                # without color to avoid garbled code saved in files.
                file_handler.setFormatter(MMFormatter(color=False, datefmt="%Y/%m/%d %H:%M:%S"))
                file_handler.setLevel(log_level)
                file_handler.addFilter(FilterDuplicateWarning(logger_name))
                file_handler.addFilter(RelativePathFilter())
                self.handlers.append(file_handler)
        self._log_file = log_file

    @classmethod
    def clear_loggers(cls):
        cls._instance_dict.clear()

    @classmethod
    def clear_logger_by_name(cls, name: str = None):
        cls._instance_dict.pop(name, None)

    @classmethod
    @property
    def log_file(self):
        return self._log_file

    @classmethod
    def get_current_instance(cls) -> "MMLogger":
        """Get latest created ``MMLogger`` instance.

        :obj:`MMLogger` can call :meth:`get_current_instance` before any
        instance has been created, and return a logger with the instance name
        "xhquant".

        Returns:
            MMLogger: Configured logger instance.
        """
        if not cls._instance_dict:
            cls.get_instance("xhquant")
        return super().get_current_instance()

    def callHandlers(self, record: LogRecord) -> None:
        """Pass a record to all relevant handlers.

        Override ``callHandlers`` method in ``logging.Logger`` to avoid
        multiple warning messages in DDP mode. Loop through all handlers of
        the logger instance and its parents in the logger hierarchy. If no
        handler was found, the record will not be output.

        Args:
            record (LogRecord): A ``LogRecord`` instance contains logged
                message.
        """
        for handler in self.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def setLevel(self, level):
        """Set the logging level of this logger.

        If ``logging.Logger.selLevel`` is called, all ``logging.Logger``
        instances managed by ``logging.Manager`` will clear the cache. Since
        ``MMLogger`` is not managed by ``logging.Manager`` anymore,
        ``MMLogger`` should override this method to clear caches of all
        ``MMLogger`` instance which is managed by :obj:`ManagerMixin`.

        level must be an int or a str.
        """
        self.level = logging._checkLevel(level)
        _accquire_lock()
        # The same logic as `logging.Manager._clear_cache`.
        for logger in MMLogger._instance_dict.values():
            logger._cache.clear()
        _release_lock()


def print_current_log(msg, level=logging.INFO):
    print_log(msg, "current", level=level)


def print_log(msg, logger: Optional[Union[Logger, str]] = None, level=logging.INFO) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif logger == "current":
        logger_instance = MMLogger.get_current_instance()
        logger_instance.log(level, msg)
    elif isinstance(logger, str):
        # If the type of `logger` is `str`, but not with value of `current` or
        # `silent`, we assume it indicates the name of the logger. If the
        # corresponding logger has not been created, `print_log` will raise
        # a `ValueError`.
        if MMLogger.check_instance_created(logger):
            logger_instance = MMLogger.get_instance(logger)
            logger_instance.log(level, msg)
        else:
            raise ValueError(f"MMLogger: {logger} has not been created!")
    else:
        raise TypeError(
            "`logger` should be either a logging.Logger object, str, "
            f'"silent", "current" or None, but got {type(logger)}'
        )


def _get_world_size():
    """Support using logging module without torch."""
    try:
        # requires torch
        from ..dist import get_world_size
    except ImportError:
        return 1
    else:
        return get_world_size()


def _get_rank():
    """Support using logging module without torch."""
    try:
        # requires torch
        from ..dist import get_rank
    except ImportError:
        return 0
    else:
        return get_rank()


def _get_device_id():
    """Get device id of current machine."""
    try:
        import torch
    except ImportError:
        return 0
    else:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # TODO: return device id of npu and mlu.
        if not torch.cuda.is_available():
            return local_rank
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            num_device = torch.cuda.device_count()
            cuda_visible_devices = list(range(num_device))
        else:
            cuda_visible_devices = cuda_visible_devices.split(",")
        try:
            return int(cuda_visible_devices[local_rank])
        except ValueError:
            # handle case for Multi-Instance GPUs
            # see #1148 for details
            return cuda_visible_devices[local_rank]


def _get_host_info() -> str:
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ""
    try:
        host = f"{getuser()}@{gethostname()}"
    except Exception as e:
        warnings.warn(f"Host or user not found: {str(e)}")
    finally:
        return host


def _get_logging_file_handlers() -> Dict:
    """Get additional file_handlers in ``logging.handlers``.

    Returns:
        Dict: A map of file_handlers.
    """
    file_handlers_map = {}
    for module_name in dir(handlers):
        if module_name.startswith("__"):
            continue
        _fh = getattr(handlers, module_name)
        if inspect.isclass(_fh) and issubclass(_fh, logging.FileHandler):
            file_handlers_map[module_name] = _fh
    return file_handlers_map
