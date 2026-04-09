#!/usr/bin/env python3
"""
Debug script to verify that the wrapped RMSNorm implementation matches
HF's native Gemma4RMSNorm for both with_scale=True and with_scale=False cases.
"""

import sys
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm


# Import xhquant RMSNorm
try:
    from xhquant.nn import RMSNorm
    XHQUANT_AVAILABLE = True
except ImportError:
    print("Warning: xhquant.nn not available, creating fallback RMSNorm")
    XHQUANT_AVAILABLE = False
    
    class RMSNorm(nn.Module):
        """Fallback RMSNorm implementation for testing"""
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
            self.eps = eps
            self.hidden_size = hidden_size

        def forward(self, x):
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output * self.weight


class _Gemma4RMSNorm(nn.Module):
    """Wrapped Gemma4RMSNorm that uses xhquant's RMSNorm"""
    
    def __init__(self, hf_module: Gemma4RMSNorm):
        super().__init__()
        # Copy attributes from HF module
        self.eps = hf_module.eps
        self.with_scale = getattr(hf_module, "with_scale", True)
        self.dim = getattr(hf_module, "dim", hf_module.weight.shape[0] if hasattr(hf_module, "weight") else None)
        
        # Copy weight if it exists
        if hasattr(hf_module, "weight"):
            self.weight = nn.Parameter(hf_module.weight.data.clone())
        
        # Setup wrapped RMSNorm
        self._setup()
    
    def _setup(self):
        """Setup the wrapped RMSNorm"""
        hidden_size = self.weight.shape[0] if hasattr(self, "weight") else self.dim
        self.norm = RMSNorm(hidden_size, self.eps)
        if getattr(self, "with_scale", True) and hasattr(self, "weight"):
            self.norm.weight = nn.Parameter(deepcopy(self.weight.data))
        else:
            self.norm.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32), requires_grad=False)
    
    def forward(self, hidden_states):
        return self.norm(hidden_states)


def compute_cosine_similarity(x, y):
    """Compute cosine similarity between two tensors"""
    x_flat = x.flatten()
    y_flat = y.flatten()
    return F.cosine_similarity(x_flat.unsqueeze(0), y_flat.unsqueeze(0), dim=1).item()


def compute_max_abs_diff(x, y):
    """Compute max absolute difference between two tensors"""
    return (x - y).abs().max().item()


def test_rmsnorm_case(case_name: str, with_scale: bool):
    """Test a single RMSNorm case"""
    print(f"\n{'='*60}")
    print(f"Testing: {case_name} (with_scale={with_scale})")
    print(f"{'='*60}")
    
    # Config
    hidden_size = 3072
    eps = 1e-6
    batch_size = 2
    seq_len = 10
    
    # Create HF Gemma4RMSNorm
    hf_norm = Gemma4RMSNorm(hidden_size, eps=eps)
    hf_norm.with_scale = with_scale
    hf_norm.dim = hidden_size
    
    if not with_scale:
        # For with_scale=False, remove the weight parameter
        if hasattr(hf_norm, "weight"):
            delattr(hf_norm, "weight")
    
    print(f"HF RMSNorm attributes:")
    print(f"  - with_scale: {with_scale}")
    print(f"  - has weight: {hasattr(hf_norm, 'weight')}")
    if hasattr(hf_norm, "weight"):
        print(f"  - weight shape: {hf_norm.weight.shape}")
    
    # Create wrapped version
    wrapped_norm = _Gemma4RMSNorm(hf_norm)
    print(f"\nWrapped RMSNorm attributes:")
    print(f"  - with_scale: {wrapped_norm.with_scale}")
    print(f"  - has weight: {hasattr(wrapped_norm, 'weight')}")
    print(f"  - norm.weight shape: {wrapped_norm.norm.weight.shape}")
    
    # Create random input
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    
    # Compute outputs
    with torch.no_grad():
        hf_output = hf_norm(input_tensor)
        wrapped_output = wrapped_norm(input_tensor)
    
    # Compute metrics
    cosine_sim = compute_cosine_similarity(hf_output, wrapped_output)
    max_diff = compute_max_abs_diff(hf_output, wrapped_output)
    
    print(f"\nOutput comparison:")
    print(f"  - HF output shape: {hf_output.shape}")
    print(f"  - Wrapped output shape: {wrapped_output.shape}")
    print(f"  - Cosine similarity: {cosine_sim:.6f}")
    print(f"  - Max absolute difference: {max_diff:.10f}")
    
    # Pass/Fail criteria
    # Cosine similarity should be very close to 1.0
    # Max absolute difference should be very small
    sim_threshold = 0.999
    diff_threshold = 1e-5
    
    sim_pass = cosine_sim > sim_threshold
    diff_pass = max_diff < diff_threshold
    
    print(f"\nCriteria:")
    print(f"  - Cosine similarity > {sim_threshold}: {'PASS' if sim_pass else 'FAIL'}")
    print(f"  - Max diff < {diff_threshold}: {'PASS' if diff_pass else 'FAIL'}")
    
    overall_pass = sim_pass and diff_pass
    print(f"\nResult: {'✓ PASS' if overall_pass else '✗ FAIL'}")
    
    return overall_pass


def main():
    """Main test function"""
    print("=" * 60)
    print("Gemma4 RMSNorm Verification Debug Script")
    print("=" * 60)
    
    results = {}
    
    # Test case 1: with_scale=True (standard RMSNorm with weight)
    results["with_scale=True"] = test_rmsnorm_case(
        "Standard RMSNorm with scale",
        with_scale=True
    )
    
    # Test case 2: with_scale=False (normalize only, no weight scaling)
    results["with_scale=False"] = test_rmsnorm_case(
        "RMSNorm without scale",
        with_scale=False
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for case, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{case}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
