#!/usr/bin/env python3
"""
Export OpenVLA-OFT Action Head to ONNX and HMONNX format.

This script exports both L1RegressionActionHead and DiffusionActionHead to ONNX and HMONNX format.
For DiffusionActionHead, it exports the noise_predictor component.

Usage:
    # Export L1RegressionActionHead
    python openvla_oft_action_head_export_hmonnx.py --type l1 --hidden-dim 4096 --action-dim 7 --output action_head_l1.onnx --hmonnx action_head_l1_hm.onnx

    # Export DiffusionActionHead (noise_predictor)
    python openvla_oft_action_head_export_hmonnx.py --type diffusion --hidden-dim 4096 --action-dim 7 --output action_head_diffusion.onnx --hmonnx action_head_diffusion_hm.onnx

    # Export from pretrained checkpoint
    python openvla_oft_action_head_export_hmonnx.py --type l1 --checkpoint /path/to/action_head.pt --output action_head_l1.onnx --hmonnx action_head_l1_hm.onnx
"""

import argparse
from pathlib import Path
import os
import os.path as osp
import logging
from typing import Dict

import torch
import torch.nn as nn
import onnx
import onnxslim

from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForVision2Seq, AutoProcessor

from xhquant.api import convert_onnx_to_hmonnx, HMONNXGoldenInference, QuantScheme, DeviceType, create_quant_config

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


def export_l1_action_head(
    hidden_dim: int,
    action_dim: int,
    output_path: str,
    hmonnx_path: str,
    checkpoint_path: str = None,
    batch_size: int = 1,
    opset_version: int = 14,
    dtype: torch.dtype = torch.float32,
):
    """
    Export L1RegressionActionHead to ONNX.
    
    Args:
        hidden_dim: Hidden dimension (typically 4096 for OpenVLA)
        action_dim: Action dimension (typically 7 for robot arms)
        output_path: Path to save ONNX file
        checkpoint_path: Path to load weights (optional)
        batch_size: Batch size for export
        opset_version: ONNX opset version
        dtype: Data type for export
    """
    print(f"Exporting L1RegressionActionHead to {output_path}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Dtype: {dtype}")
    
    # Create wrapper model with forward method for ONNX export
    class L1ActionHeadWrapper(nn.Module):
        """Wrapper for L1RegressionActionHead with forward method."""
        def __init__(self, input_dim, hidden_dim, action_dim):
            super().__init__()
            self.model = L1RegressionActionHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            )
        
        def forward(self, actions_hidden_states):
            """
            Args:
                actions_hidden_states: (batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
            Returns:
                action: (batch_size, action_dim)
            """
            return self.model.predict_action(actions_hidden_states)
    
    # Create model
    model = L1ActionHeadWrapper(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )
    
    # Load weights if provided
    if checkpoint_path:
        print(f"  - Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Map checkpoint keys to wrapper keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key] = value
            else:
                new_state_dict['model.' + key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(dtype).eval()
    
    # Create dummy input
    # Input shape: (batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
    # Note: L1RegressionActionHead.predict_action expects this shape
    dummy_input = torch.randn(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim, dtype=dtype)
    
    # Export (static shape, no dynamic axes)
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['actions_hidden_states'],
        output_names=['action'],
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Successfully exported to: {output_path}")
    
    # Verify
    verify_onnx_export(output_path, model, dummy_input)
    
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    
    dummy_input = dummy_input.to(torch.float16)
    convert_onnx_to_hmonnx(
        str(output_path),
        (dummy_input,),
        out_hmonnx_file=str(hmonnx_path),
        device_type="XH2A",
        quant_config=quant_config,
    )


def export_diffusion_action_head(
    hidden_dim: int,
    action_dim: int,
    output_path: str,
    hmonnx_path: str,
    checkpoint_path: str = None,
    batch_size: int = 1,
    opset_version: int = 14,
    dtype: torch.dtype = torch.float32,
):
    """
    Export DiffusionActionHead's noise_predictor to ONNX.
    
    Note: We export the noise_predictor component which is used during inference.
    The diffusion process (scheduler) remains in Python.
    
    Args:
        hidden_dim: Hidden dimension (typically 4096 for OpenVLA)
        action_dim: Action dimension (typically 7 for robot arms)
        output_path: Path to save ONNX file
        checkpoint_path: Path to load weights (optional)
        batch_size: Batch size for export
        opset_version: ONNX opset version
        dtype: Data type for export
    """
    print(f"Exporting DiffusionActionHead (noise_predictor) to {output_path}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Action dim: {action_dim}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Dtype: {dtype}")
    
    # Create wrapper module for ONNX export
    class DiffusionNoisePredictorWrapper(nn.Module):
        """Wrapper to make noise_predictor compatible with ONNX export."""
        def __init__(self, action_dim, hidden_dim):
            super().__init__()
            # Import here to avoid circular imports
            from prismatic.models.action_heads import NoisePredictionModel
            self.noise_predictor = NoisePredictionModel(
                transformer_hidden_dim=hidden_dim * ACTION_DIM,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            )
        
        def forward(self, rearranged_actions_hidden_states):
            """
            Args:
                rearranged_actions_hidden_states: (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM * hidden_dim)
            Returns:
                noise_pred: (batch_size, action_dim)
            """
            return self.noise_predictor(rearranged_actions_hidden_states)
    
    # Create model
    model = DiffusionNoisePredictorWrapper(action_dim=action_dim, hidden_dim=hidden_dim)
    
    # Load weights if provided
    if checkpoint_path:
        print(f"  - Loading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Map checkpoint keys to wrapper keys
        # Checkpoint keys are like: "noise_predictor.mlp_resnet.xxx"
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('noise_predictor.'):
                # Already has noise_predictor prefix, use directly
                new_state_dict[key] = value
            elif 'mlp_resnet' in key or 'layer_norm' in key:
                # Keys like "mlp_resnet.xxx" -> "noise_predictor.mlp_resnet.xxx"
                new_state_dict['noise_predictor.' + key] = value
            else:
                new_state_dict[key] = value
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  ⚠ Missing keys: {missing}")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {unexpected}")
    
    model = model.to(dtype).eval()
    
    # Create dummy input
    # Input shape: (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM * hidden_dim)
    dummy_input = torch.randn(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM * hidden_dim, dtype=dtype)
    
    # Export (static shape, no dynamic axes)
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['rearranged_actions_hidden_states'],
        output_names=['noise_pred'],
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    print(f"✓ Successfully exported to: {output_path}")
    
    # Verify
    verify_onnx_export(output_path, model, dummy_input)
    
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    
    dummy_input = dummy_input.to(torch.float16)
    convert_onnx_to_hmonnx(
        str(output_path),
        (dummy_input,),
        out_hmonnx_file=str(hmonnx_path),
        device_type="XH2A",
        quant_config=quant_config,
    )


def verify_onnx_export(onnx_path: str, model: nn.Module, dummy_input: torch.Tensor, method_name: str = None):
    """Verify ONNX export by comparing outputs."""
    print(f"\nVerifying ONNX export...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model is valid")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Get PyTorch output
        with torch.no_grad():
            if method_name:
                torch_output = getattr(model, method_name)(dummy_input).numpy()
            else:
                torch_output = model(dummy_input).numpy()
        
        # Get ONNX output
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]
        
        # Compare
        import numpy as np
        max_diff = np.max(np.abs(torch_output - onnx_output))
        print(f"  ✓ Max difference between PyTorch and ONNX: {max_diff:.6e}")
        
        if max_diff < 1e-5:
            print("  ✓ Outputs match within tolerance!")
        else:
            print(f"  ⚠ Warning: outputs differ by {max_diff:.6e}")
            
    except ImportError as e:
        print(f"  ⚠ Skipping verification (onnx/onnxruntime not installed): {e}")
    except Exception as e:
        print(f"  ⚠ Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export OpenVLA-OFT Action Head to ONNX")
    
    parser.add_argument(
        "--type",
        type=str,
        choices=["l1", "diffusion"],
        required=True,
        help="Type of action head to export"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Hidden dimension (default: 4096 for OpenVLA)"
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Action dimension (default: 7 for robot arms)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="action_head.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--hmonnx",
        type=str,
        default="action_head_hm.onnx",
        help="Output HMONNX file path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for export"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export based on type
    if args.type == "l1":
        export_l1_action_head(
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim,
            output_path=args.output,
            hmonnx_path=args.hmonnx,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            opset_version=args.opset_version,
            dtype=dtype,
        )
    elif args.type == "diffusion":
        export_diffusion_action_head(
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim,
            output_path=args.output,
            hmonnx_path=args.hmonnx,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            opset_version=args.opset_version,
            dtype=dtype,
        )


if __name__ == "__main__":
    main()