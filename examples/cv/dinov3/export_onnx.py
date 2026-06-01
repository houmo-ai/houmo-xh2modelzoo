# Copyright 2025 HOUMO AI
#
# File: examples/cv/dinov3/export_onnx.py
# Description:
#   Export DINOv3 models (ViT and ConvNeXt) to ONNX format.
#   Supports both HuggingFace transformers DINOv3ViTModel and DINOv3ConvNextModel.
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

"""Export DINOv3 (ViT / ConvNeXt) to ONNX format.

Usage:
    # Export ViT-S/16 (distilled) to ONNX
    python export_onnx.py --model-type vit --model-name facebook/dinov3-vits16-pretrain-lvd1689m

    # Export ConvNeXt Tiny to ONNX
    python export_onnx.py --model-type convnext --model-name facebook/dinov3-convnext-tiny-pretrain-lvd1689m

    # Export ViT-L/16 with custom image size
    python export_onnx.py --model-type vit --model-name facebook/dinov3-vitl16-pretrain-lvd1689m --image-size 518

    # Export with dynamic batch size
    python export_onnx.py --model-type vit --model-name facebook/dinov3-vits16-pretrain-lvd1689m --dynamic-batch
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.onnx
from transformers.models.dinov3_convnext import DINOv3ConvNextConfig, DINOv3ConvNextModel
from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel

# Output directory relative to this script
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "onnx"


def _patch_embeddings_for_export(model):
    """Monkey-patch the DINOv3 ViT embeddings to avoid zero-size torch.cat.

    When num_register_tokens == 0, concatenating a zero-size register token
    tensor via torch.cat causes TorchScript JIT tracer to miscount tokens
    (off by 1), producing invalid ONNX Reshape target shapes.

    This patch skips the register token concatenation entirely when
    num_register_tokens == 0, which avoids the JIT tracer bug.
    The patch is applied in-place on the model instance.
    """
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings

    original_forward = DINOv3ViTEmbeddings.forward

    def patched_forward(self, pixel_values, bool_masked_pos=None):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        if self.config.num_register_tokens > 0:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)
        else:
            embeddings = torch.cat([cls_token, patch_embeddings], dim=1)
        return embeddings

    DINOv3ViTEmbeddings.forward = patched_forward
    # Store original for potential restoration
    model._original_embeddings_forward = original_forward


def _restore_embeddings_patch(model):
    """Restore the original embeddings forward method after export."""
    if hasattr(model, "_original_embeddings_forward"):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings
        DINOv3ViTEmbeddings.forward = model._original_embeddings_forward
        del model._original_embeddings_forward


def _export_vit(
    model_name: str,
    onnx_path: str,
    image_size: int = 224,
    dynamic_batch: bool = False,
    opset_version: int = 17,
    simplify: bool = True,
) -> str:
    """Export a DINOv3 ViT model to ONNX.

    Uses eager attention (not SDPA/Flash) for reliable ONNX trace compatibility.
    Monkey-patches embeddings to avoid zero-size torch.cat JIT tracer bug.

    Args:
        model_name: HuggingFace model name or path.
        onnx_path: Output ONNX file path.
        image_size: Input image size (square, e.g., 224).
        dynamic_batch: Whether to use dynamic batch size.
        opset_version: ONNX opset version.
        simplify: Whether to run onnx-simplifier.

    Returns:
        Path to the exported ONNX file.
    """
    print(f"[ViT] Loading model: {model_name}")
    # Use eager attention for reliable ONNX tracing (SDPA tracing can produce
    # incomplete graphs that fail at ORT runtime with shape inference errors).
    config = DINOv3ViTConfig.from_pretrained(model_name)
    config._attn_implementation = "eager"
    model = DINOv3ViTModel.from_pretrained(model_name, config=config, torch_dtype=torch.float32)
    model.eval()

    # Monkey-patch embeddings to avoid zero-size torch.cat JIT tracer bug
    _patch_embeddings_for_export(model)

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)

    # NOTE: output_names must match the field order of BaseModelOutputWithPooling
    # (alphabetical: last_hidden_state, pooler_output). Swapping them causes
    # incorrect name-to-shape mapping in the ONNX graph.
    input_names = ["pixel_values"]
    output_names = ["last_hidden_state", "pooler_output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        }

    print(f"[ViT] Exporting with image_size={image_size}, dynamic_batch={dynamic_batch}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Restore original embeddings forward
    _restore_embeddings_patch(model)

    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            onnx_model = onnx.load(onnx_path)
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, onnx_path)
                print(f"[ViT] ONNX simplified successfully")
            else:
                print(f"[ViT] ONNX simplification check failed, keeping original")
        except ImportError:
            print("[ViT] onnxsim not installed, skipping simplification")

    # Verify with ONNX checker
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Quick ORT inference smoke test
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        ort_outputs = session.run(None, {"pixel_values": test_input})
        print(f"[ViT] ORT smoke test passed: "
              f"last_hidden_state={ort_outputs[0].shape}, pooler_output={ort_outputs[1].shape}")
    except Exception as e:
        print(f"[ViT] WARNING: ORT smoke test failed: {e}")

    print(f"[ViT] ONNX exported and verified: {onnx_path}")
    return onnx_path


def _export_convnext(
    model_name: str,
    onnx_path: str,
    image_size: int = 224,
    dynamic_batch: bool = False,
    opset_version: int = 17,
    simplify: bool = True,
) -> str:
    """Export a DINOv3 ConvNeXt model to ONNX.

    Args:
        model_name: HuggingFace model name or path.
        onnx_path: Output ONNX file path.
        image_size: Input image size (square, e.g., 224).
        dynamic_batch: Whether to use dynamic batch size.
        opset_version: ONNX opset version.
        simplify: Whether to run onnx-simplifier.

    Returns:
        Path to the exported ONNX file.
    """
    print(f"[ConvNeXt] Loading model: {model_name}")
    model = DINOv3ConvNextModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)

    # NOTE: output_names must match the field order of BaseModelOutputWithPooling
    # (alphabetical: last_hidden_state, pooler_output).
    input_names = ["pixel_values"]
    output_names = ["last_hidden_state", "pooler_output"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        }

    print(f"[ConvNeXt] Exporting with image_size={image_size}, dynamic_batch={dynamic_batch}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            onnx_model = onnx.load(onnx_path)
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified_model, onnx_path)
                print(f"[ConvNeXt] ONNX simplified successfully")
            else:
                print(f"[ConvNeXt] ONNX simplification check failed, keeping original")
        except ImportError:
            print("[ConvNeXt] onnxsim not installed, skipping simplification")

    # Quick ORT smoke test
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        ort_outputs = session.run(None, {"pixel_values": test_input})
        print(f"[ConvNeXt] ORT smoke test passed: "
              f"last_hidden_state={ort_outputs[0].shape}, pooler_output={ort_outputs[1].shape}")
    except Exception as e:
        print(f"[ConvNeXt] WARNING: ORT smoke test failed: {e}")

    print(f"[ConvNeXt] ONNX exported to: {onnx_path}")
    return onnx_path


def main(args):
    model_name = args.model_name
    model_type = args.model_type.lower()
    image_size = args.image_size
    dynamic_batch = args.dynamic_batch
    opset = args.opset
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine image_size from model config if not specified
    if image_size is None:
        if model_type == "vit":
            cfg = DINOv3ViTConfig.from_pretrained(model_name)
            image_size = cfg.image_size if isinstance(cfg.image_size, int) else cfg.image_size[0]
        else:
            cfg = DINOv3ConvNextConfig.from_pretrained(model_name)
            image_size = cfg.image_size if isinstance(cfg.image_size, int) else cfg.image_size[0]
        print(f"Auto-detected image_size={image_size} from model config")

    # Generate output path — use directory name for local paths, full name for HF models
    if Path(model_name).exists():
        safe_name = Path(model_name).name
    else:
        safe_name = model_name.replace("/", "_")
    suffix = f"_is{image_size}"
    if dynamic_batch:
        suffix += "_dbatch"
    onnx_filename = f"{safe_name}{suffix}.onnx"
    onnx_path = str(outdir / onnx_filename)

    if model_type == "vit":
        _export_vit(model_name, onnx_path, image_size=image_size,
                     dynamic_batch=dynamic_batch, opset_version=opset,
                     simplify=not args.no_simplify)
    elif model_type == "convnext":
        _export_convnext(model_name, onnx_path, image_size=image_size,
                          dynamic_batch=dynamic_batch, opset_version=opset,
                          simplify=not args.no_simplify)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: vit, convnext")

    print(f"\nDone! ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DINOv3 to ONNX")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["vit", "convnext"],
                        help="Model architecture type")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name or local path, "
                             "e.g., facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input image size (square). Auto-detected if not specified.")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Export with dynamic batch size dimension")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Skip onnx-simplifier pass")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR),
                        help=f"Output directory (default: {DEFAULT_OUTDIR})")
    args = parser.parse_args()
    main(args)
