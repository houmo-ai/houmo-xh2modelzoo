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

    def __init__(self, rank: Union[int, str, torch.device], name: str, logger=None):
        if isinstance(rank, str):
            device = torch.device(rank)
        elif isinstance(rank, int):
            device = torch.device(f"cuda:{rank}")
        else:
            device = rank
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
        gpu_max_memory_reserved = torch.cuda.max_memory_reserved(self.rank) / 1024**2

        return {
            "cpu_mem": cpu_mem,
            "gpu_allocated": gpu_allocated,
            "gpu_reserved": gpu_reserved,
            "gpu_max_allocated": gpu_max_allocated,
            "gpu_max_memory_reserved": gpu_max_memory_reserved,
        }

    def log_memory(self, name="", logger=None):
        """记录内存使用"""
        mem = self.get_memory_usage()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if logger is not None:
            logger.info(
                f"""Rank {self.rank} - {name}
    CPU Memory: {mem['cpu_mem']:.2f}MBR
    GPU Memory Allocated: {mem['gpu_allocated']:.2f}MB
    GPU Memory Reserved: {mem['gpu_reserved']:.2f}MB
    GPU Max Memory Allocated: {mem['gpu_max_allocated']:.2f}MB
    GPU Max Memory Reserved: {mem['gpu_max_memory_reserved']:.2f}MB
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
    GPU Max Memory Reserved: {mem['gpu_max_memory_reserved']:.2f}MB
    {'='*50}"""
            )

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_memory(self._step, self._logger)
