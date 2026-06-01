import os
import torch


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
        logger.info(f"  GPU Total Memory: {meminfo.total / (1024**3):.4f} GB")
        logger.info(f"  GPU Used Memory: {meminfo.used / (1024**3):.4f} GB")
        logger.info(f"  GPU Free Memory: {meminfo.free / (1024**3):.4f} GB")