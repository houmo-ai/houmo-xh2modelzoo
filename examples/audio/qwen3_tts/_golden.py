from pathlib import Path

import torch

from xh_model_zoo.api import get_root_logger


def _flatten_inputs(input_args):
    """Flatten inputs containing list/tuple (e.g., kv-cache lists) into positional argument sequences."""
    flat = []
    for arg in input_args:
        if isinstance(arg, (list, tuple)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _align_dtype(input_args, float_dtype=torch.float16):
    """Align floating-point Tensors to export dtype (fp16 for xh2a).

    Non-torch.Tensor subclasses like CacheTensor and integer Tensors remain unchanged.
    """
    out = []
    for arg in input_args:
        if type(arg) is torch.Tensor and arg.is_floating_point():
            arg = arg.to(float_dtype)
        out.append(arg)
    return out


def run_hmonnx_golden(hmonnx_file, golden_dir, input_args, device="cuda:0"):
    """
    Args:
        hmonnx_file: Path to exported HMONNX file.
        golden_dir: Directory to save golden data.
        input_args: Model input, nested list/tuple (e.g., kv-cache lists) allowed, internally flattened.
        device: Device for golden inference, fallback to cpu if no GPU available.

    Returns:
        String path of golden_dir.
    """
    from xhquant.api import HMONNXGoldenInference

    logger = get_root_logger()
    Path(golden_dir).mkdir(exist_ok=True, parents=True)

    input_args = _align_dtype(_flatten_inputs(input_args))

    session = HMONNXGoldenInference(str(hmonnx_file))
    session.save_golden = True
    session.exec_device = torch.device(device if torch.cuda.is_available() else "cpu")
    session.golden_dir = str(golden_dir)

    with torch.no_grad():
        session.forward(*input_args)

    logger.info(f"export hmonnx golden to {golden_dir}")
    return str(golden_dir)
