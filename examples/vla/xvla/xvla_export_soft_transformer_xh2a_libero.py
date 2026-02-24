
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import logging
import argparse
import numpy as np
# --- 量化工具导入 ---
from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType
# Add src to sys.path to allow importing lerobot modules
# Assuming we are running from xh2modelzoo root or examples directory
# The lerobot path seems to be at /data01/home/chenzx/project/xh2_release/lerobot
LEROBOT_PATH = "/data01/home/chenzx/project/xh2_release/lerobot/src"
if LEROBOT_PATH not in sys.path:
    sys.path.append(LEROBOT_PATH)

try:
    from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer
except ImportError:
    # Fallback if the path structure is different
    sys.path.append("/data01/home/chenzx/project/xh2_release/lerobot")
    from src.lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer

from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType, get_root_logger

def main(args):
    logger = get_root_logger()
    logger.setLevel(logging.INFO)

    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        logger.error(f"ONNX file not found at {onnx_path}")
        return

    work_dir = Path("work_dirs/xvla_soft_transformer")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Prepare Calibration Data ---
    # Need to match the input shapes used during export
    batch_size = 1
    num_actions = 30
    vlm_seq_len = 50
    aux_visual_seq_len = 20
    
    hidden_size = 1024
    multi_modal_input_size = 1024
    dim_action = 20
    dim_propio = 20
    
    # Inputs:
    domain_id = torch.zeros((batch_size,), dtype=torch.long)
    vlm_features = torch.randn(batch_size, vlm_seq_len, multi_modal_input_size)
    aux_visual_inputs = torch.randn(batch_size, aux_visual_seq_len, multi_modal_input_size)
    action_with_noise = torch.randn(batch_size, num_actions, dim_action)
    proprio = torch.randn(batch_size, dim_propio)
    t = torch.rand(batch_size)

    # Note: convert_onnx_to_hmonnx expects calibration_data to be a list of tuples/lists if input is positional
    # or list of dicts if named? The previous example used tuple `(input_features,)`.
    # Let's check `convert_onnx_to_hmonnx` signature usage in `xvla_export_vision_xh2a_libero.py`:
    # convert_onnx_to_hmonnx(simplified_onnx_file, (input_features,), ...)
    # So it takes a tuple of tensors for a single batch.
    
    calib_data = [
        (
            domain_id,
            vlm_features,
            aux_visual_inputs,
            action_with_noise,
            proprio,
            t
        )
    ]

    # --- 2. Convert to HMONNX ---
    hmonnx_path = work_dir / "soft_prompted_transformer.onnx"
    logger.info(f"Converting ONNX {onnx_path} to HMONNX at {hmonnx_path}...")

    quant_type = "w8a8h1_sefp" 
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    convert_onnx_to_hmonnx(
        str(onnx_path),
        calib_data[0],
        out_hmonnx_file=str(hmonnx_path),
        device_type="XH2A",
        quant_config=quant_config,
    )
    logger.info("HMONNX conversion finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, default="/data01/home/chenzx/project/xh2_release/lerobot/soft_prompted_transformer.onnx", help="Path to the input ONNX file")
    args = parser.parse_args()
    main(args)
