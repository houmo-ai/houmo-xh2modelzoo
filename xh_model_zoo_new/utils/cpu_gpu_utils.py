import os
import torch
import psutil


GB = 1024 * 1024 * 1024


def print_gpu_info(logger=None):
    if not torch.cuda.is_available():
        return
    if logger is None:
        from .logger import get_root_logger

        logger = get_root_logger()
    import pynvml

    devices = range(torch.cuda.device_count())
    if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"]) > 0:
        devices = os.environ["CUDA_VISIBLE_DEVICES"]
        devices = [int(d) for d in devices.split(",")]
    pynvml.nvmlInit()
    for device in devices:
        logger.info(f"GPU {device} Info:")
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info(f"  GPU Total Memory: {meminfo.total / (1024**3):.4f} GB")  # 总的显存大小
        logger.info(f"  GPU Used Memory: {meminfo.used / (1024**3):.4f} GB")  # 已用显存大小
        logger.info(f"  GPU Free Memory: {meminfo.free / (1024**3):.4f} GB")  # 剩余显存大小


def print_cpu_memory_info(logger=None):
    if logger is None:
        from .logger import get_root_logger

        logger = get_root_logger()

    mem_info = psutil.virtual_memory()
    total = mem_info.total / GB
    used = mem_info.used / GB
    free = mem_info.available / GB
    logger.info(f"CPU Memory Info:")
    logger.info(f"\tTotal Memory: {total:.4f} GB")  # 总的显存大小
    logger.info(f"\tUsed Memory: {used:.4f} GB")  # 已用显存大小
    logger.info(f"\tAvailable Memory: {free:.4f} GB")  # 剩余显存大小
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Current Process Memory Info:")
    logger.info(f"\tMemory: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
    return
