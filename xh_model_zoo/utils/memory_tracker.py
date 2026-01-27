# Copyright 2025 HOUMO AI
#
# File: memory_tracker.py
# Description:
#   Memory usage tracking utility.
#   This module provides MemoryTracker class for tracking
#   GPU and CPU memory usage during code execution.
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
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Union

import psutil
import torch


class MemoryTracker(AbstractContextManager):
    """内存使用追踪器
    with MemoryTracker("cuda:0", 'Before code execution'):
        # 执行代码
    """

    def __init__(self, device: Union[int, str, torch.device], name: str = None, logger=None):
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.rank = device.index
        self.process = psutil.Process()
        self.reset()
        self._step = name
        self._logger = logger

    def reset(self):
        torch.cuda.reset_peak_memory_stats()

    def get_memory_usage(self):
        """获取当前内存使用情况"""
        # CPU 内存
        cpu_mem = self.process.memory_info().rss / 1024**2  # MB

        # GPU 内存
        gpu_allocated = torch.cuda.memory_allocated(self.rank) / 1024**2  # MB
        gpu_reserved = torch.cuda.memory_reserved(self.rank) / 1024**2  # MB
        gpu_max_allocated = torch.cuda.max_memory_allocated(self.rank) / 1024**2  # MB

        return {
            "cpu_mem": cpu_mem,
            "gpu_allocated": gpu_allocated,
            "gpu_reserved": gpu_reserved,
            "gpu_max_allocated": gpu_max_allocated,
        }

    def log_memory(self, name="", logger=None):
        """记录内存使用"""
        mem = self.get_memory_usage()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if logger is not None:
            logger.info(
                f"""Rank {self.rank} - {name}
    CPU Memory: {mem['cpu_mem']:.2f}MB
    GPU Memory Allocated: {mem['gpu_allocated']:.2f}MB
    GPU Memory Reserved: {mem['gpu_reserved']:.2f}MB
    GPU Max Memory Allocated: {mem['gpu_max_allocated']:.2f}MB
{'='*50}"""
            )
        else:
            print(
                f"""
    [{timestamp}] Rank {self.rank} - {name}
    CPU Memory: {mem['cpu_mem']:.2f}MB
    GPU Memory Allocated: {mem['gpu_allocated']:.2f}MB
    GPU Memory Reserved: {mem['gpu_reserved']:.2f}MB
    GPU Max Memory Allocated: {mem['gpu_max_allocated']:.2f}MB
    {'='*50}"""
            )

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_memory(self._step, self._logger)
