# Copyright 2025 HOUMO AI
#
# File: examples/cv/dinov3/test_accuracy.py
# Description:
#   Test DINOv3 HMONNX model accuracy against PyTorch float model.
#   Compares numerical outputs (cosine similarity, MSE) and optionally
#   evaluates classification accuracy on ImageNet using k-NN or linear probe.
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

"""Test DINOv3 HMONNX accuracy against PyTorch float model.

Usage:
    # Numerical accuracy test (random inputs)
    python test_accuracy.py \
        --model-name /data01/datasets/dinov3-vits16-pretrain-lvd1689m \
        --model-type vit \
        --hmonnx work_dirs/.../hmonnx/xxx_XH2a.onnx

    # With ImageNet validation set
    python test_accuracy.py \
        --model-name facebook/dinov3-vits16-pretrain-lvd1689m \
        --model-type vit \
        --hmonnx work_dirs/.../hmonnx/xxx_XH2a.onnx \
        --imagenet-val /data02/datasets/imagenet/val \
        --batch-size 32 --num-samples 1000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_float_model(model_name: str, model_type: str, device: str = "cpu"):
    """Load PyTorch float model."""
    if model_type == "vit":
        from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel
        config = DINOv3ViTConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        model = DINOv3ViTModel.from_pretrained(model_name, config=config, torch_dtype=torch.float32)
    elif model_type == "convnext":
        from transformers.models.dinov3_convnext import DINOv3ConvNextModel
        model = DINOv3ConvNextModel.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.eval()
    model.to(device)
    return model


def load_hmonnx_model(hmonnx_path: str, device: str = "cuda"):
    """Load HMONNX model."""
    from xhquant.api import HMONNXInference
    session = HMONNXInference(hmonnx_path)
    session.to(torch.device(device))
    return session


def compute_metrics(float_outputs: torch.Tensor, hmonnx_outputs: torch.Tensor):
    """Compute numerical accuracy metrics between float and HMONNX outputs."""
    float_flat = float_outputs.float().flatten()
    hmonnx_flat = hmonnx_outputs.float().flatten()

    # Cosine similarity
    cos_sim = F.cosine_similarity(float_flat.unsqueeze(0), hmonnx_flat.unsqueeze(0)).item()

    # MSE
    mse = ((float_flat - hmonnx_flat) ** 2).mean().item()

    # Max absolute difference
    max_abs_diff = (float_flat - hmonnx_flat).abs().max().item()

    # Mean absolute difference
    mean_abs_diff = (float_flat - hmonnx_flat).abs().mean().item()

    # Relative error
    float_norm = float_flat.norm().item()
    if float_norm > 0:
        rel_error = (float_flat - hmonnx_flat).norm().item() / float_norm
    else:
        rel_error = 0.0

    return {
        "cosine_similarity": cos_sim,
        "mse": mse,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_error": rel_error,
    }


def test_numerical_accuracy(
    model_name: str,
    model_type: str,
    hmonnx_path: str,
    image_size: int = 224,
    num_samples: int = 100,
    device: str = "cuda",
):
    """Test numerical accuracy: compare HMONNX outputs against PyTorch float outputs."""
    print(f"\n{'='*60}")
    print(f"Numerical Accuracy Test")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"HMONNX: {hmonnx_path}")
    print(f"Samples: {num_samples}, Image size: {image_size}")
    print()

    # Load models
    print("Loading PyTorch float model...")
    float_model = load_float_model(model_name, model_type, device="cpu")
    float_model.eval()

    print("Loading HMONNX model...")
    hmonnx_model = load_hmonnx_model(hmonnx_path, device=device)

    # Collect metrics
    all_metrics = {"pooler_output": [], "last_hidden_state": []}

    print(f"\nRunning {num_samples} samples...")
    for i in range(num_samples):
        # Generate random input
        pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

        # Float model inference
        with torch.no_grad():
            float_out = float_model(pixel_values)

        # HMONNX inference
        hmonnx_input = pixel_values.to(device).to(torch.float16)
        hmonnx_out = hmonnx_model(hmonnx_input)

        # Compare outputs
        if isinstance(hmonnx_out, tuple):
            hmonnx_last_hidden, hmonnx_pooler = hmonnx_out[0], hmonnx_out[1]
        else:
            hmonnx_last_hidden = hmonnx_out
            hmonnx_pooler = None

        # Pooler output metrics
        pooler_metrics = compute_metrics(float_out.pooler_output, hmonnx_pooler.cpu())
        all_metrics["pooler_output"].append(pooler_metrics)

        # Last hidden state metrics
        lhs_metrics = compute_metrics(float_out.last_hidden_state, hmonnx_last_hidden.cpu())
        all_metrics["last_hidden_state"].append(lhs_metrics)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{num_samples} samples")

    # Aggregate metrics
    print(f"\n{'='*60}")
    print(f"Results (averaged over {num_samples} samples)")
    print(f"{'='*60}")

    for output_name in ["pooler_output", "last_hidden_state"]:
        print(f"\n--- {output_name} ---")
        metrics_list = all_metrics[output_name]
        for metric_name in ["cosine_similarity", "mse", "max_abs_diff", "mean_abs_diff", "relative_error"]:
            values = [m[metric_name] for m in metrics_list]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"  {metric_name:20s}: mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")

    return all_metrics


def test_imagenet_accuracy(
    model_name: str,
    model_type: str,
    hmonnx_path: str,
    imagenet_val_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_samples: int = None,
    device: str = "cuda",
):
    """Test classification accuracy on ImageNet validation set using k-NN on CLS embeddings."""
    print(f"\n{'='*60}")
    print(f"ImageNet k-NN Accuracy Test")
    print(f"{'='*60}")

    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset

    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Load dataset
    val_dataset = ImageFolder(imagenet_val_path, transform=val_transform)
    if num_samples is not None:
        indices = list(range(min(num_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    print(f"Dataset: {len(val_dataset)} images")

    # Load models
    print("Loading models...")
    float_model = load_float_model(model_name, model_type, device="cpu")
    float_model.eval()
    hmonnx_model = load_hmonnx_model(hmonnx_path, device=device)

    # Collect embeddings and labels
    float_embeddings = []
    hmonnx_embeddings = []
    all_labels = []

    print("Extracting embeddings...")
    for batch_idx, (images, labels) in enumerate(val_loader):
        with torch.no_grad():
            float_out = float_model(images)
            float_embeddings.append(float_out.pooler_output)

        hmonnx_out = hmonnx_model(images.to(device).to(torch.float16))
        if isinstance(hmonnx_out, tuple):
            hmonnx_embeddings.append(hmonnx_out[1].cpu())
        else:
            hmonnx_embeddings.append(hmonnx_out.cpu())

        all_labels.append(labels)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * batch_size}/{len(val_dataset)} images")

    float_embeddings = torch.cat(float_embeddings, dim=0).numpy()
    hmonnx_embeddings = torch.cat(hmonnx_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute embedding similarity
    print(f"\n--- Embedding Quality ---")
    cos_sim = np.mean([
        np.dot(float_embeddings[i], hmonnx_embeddings[i]) /
        (np.linalg.norm(float_embeddings[i]) * np.linalg.norm(hmonnx_embeddings[i]) + 1e-8)
        for i in range(len(float_embeddings))
    ])
    print(f"  Mean cosine similarity (float vs hmonnx): {cos_sim:.6f}")

    # k-NN classification (k=20, standard for DINOv2/v3 evaluation)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    for k in [1, 5, 20]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(float_embeddings, all_labels)
        float_pred = knn.predict(float_embeddings)
        float_acc = accuracy_score(all_labels, float_pred)

        knn_hmonnx = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn_hmonnx.fit(hmonnx_embeddings, all_labels)
        hmonnx_pred = knn_hmonnx.predict(hmonnx_embeddings)
        hmonnx_acc = accuracy_score(all_labels, hmonnx_pred)

        print(f"  k-NN (k={k}) Float accuracy: {float_acc:.4f}")
        print(f"  k-NN (k={k}) HMONNX accuracy: {hmonnx_acc:.4f}")
        print(f"  k-NN (k={k}) Accuracy drop: {float_acc - hmonnx_acc:.4f}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.imagenet_val:
        test_imagenet_accuracy(
            model_name=args.model_name,
            model_type=args.model_type,
            hmonnx_path=args.hmonnx,
            imagenet_val_path=args.imagenet_val,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            device=device,
        )
    else:
        test_numerical_accuracy(
            model_name=args.model_name,
            model_type=args.model_type,
            hmonnx_path=args.hmonnx,
            image_size=args.image_size,
            num_samples=args.num_samples or 100,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DINOv3 HMONNX accuracy")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["vit", "convnext"],
                        help="Model architecture type")
    parser.add_argument("--hmonnx", type=str, required=True,
                        help="Path to HMONNX model file")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size (default: 224)")
    parser.add_argument("--imagenet-val", type=str, default=None,
                        help="Path to ImageNet validation directory (optional)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for ImageNet evaluation")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to test (default: all for ImageNet, 100 for numerical)")
    args = parser.parse_args()
    main(args)
