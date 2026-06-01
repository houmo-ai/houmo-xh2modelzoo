# Copyright 2025 HOUMO AI
#
# File: examples/cv/dinov3/download_models.py
# Description:
#   Download DINOv3 models from HuggingFace Hub to local directory.
#   Requires HuggingFace authentication (gated repo access).
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

"""Download DINOv3 models and convert to HuggingFace format.

Two download methods:
1. **timm (recommended, no auth)**: Downloads via timm and converts to HF format.
   Supports ViT-S/16 and ConvNeXt-Tiny.
2. **HuggingFace Hub (requires auth)**: Direct download from gated HF repos.
   Supports all DINOv3 variants.

Usage:
    # Download via timm (no auth needed, recommended)
    python download_models.py --method timm --all-small

    # Download specific model via timm
    python download_models.py --method timm --model vits16

    # Download via HuggingFace Hub (requires login)
    python download_models.py --method hf --model facebook/dinov3-vits16-pretrain-lvd1689m

    # Custom output directory
    python download_models.py --method timm --all-small --out-dir /data01/datasets
"""

import argparse
import os
from pathlib import Path

# Default output directory
DEFAULT_OUT_DIR = "/data01/datasets"

# timm model names -> output directory names
TIMM_MODELS = {
    "vits16": ("vit_small_patch16_dinov3", "dinov3-vits16-pretrain-lvd1689m"),
    "convnext-tiny": ("convnext_tiny.dinov3_lvd1689m", "dinov3-convnext-tiny-pretrain-lvd1689m"),
    "vits16plus": ("vit_small_plus_patch16_dinov3", "dinov3-vits16plus-pretrain-lvd1689m"),
    "vitb16": ("vit_base_patch16_dinov3", "dinov3-vitb16-pretrain-lvd1689m"),
    "convnext-small": ("convnext_small.dinov3_lvd1689m", "dinov3-convnext-small-pretrain-lvd1689m"),
    "convnext-base": ("convnext_base.dinov3_lvd1689m", "dinov3-convnext-base-pretrain-lvd1689m"),
    "convnext-large": ("convnext_large.dinov3_lvd1689m", "dinov3-convnext-large-pretrain-lvd1689m"),
}

# Small models suitable for quick testing
TIMM_SMALL = ["vits16", "convnext-tiny"]

# HuggingFace model names (gated, require auth)
HF_MODELS = [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
]


def download_via_timm(model_key: str, out_dir: str):
    """Download a DINOv3 model via timm and convert to HuggingFace format."""
    import timm
    import torch
    from safetensors.torch import save_file

    if model_key not in TIMM_MODELS:
        print(f"  ✗ Unknown model key: {model_key}")
        print(f"    Available: {', '.join(TIMM_MODELS.keys())}")
        return None

    timm_name, hf_dirname = TIMM_MODELS[model_key]
    local_dir = Path(out_dir) / hf_dirname
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {timm_name} via timm...")
    try:
        model = timm.create_model(timm_name, pretrained=True)
    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
        return None

    model.eval()
    timm_state = model.state_dict()

    # Determine model type
    is_convnext = "convnext" in timm_name

    if is_convnext:
        mapped = _map_convnext_weights(timm_state)
        from transformers.models.dinov3_convnext import DINOv3ConvNextConfig
        hidden_sizes = {
            "convnext_tiny.dinov3_lvd1689m": [96, 192, 384, 768],
            "convnext_small.dinov3_lvd1689m": [96, 192, 384, 768],
            "convnext_base.dinov3_lvd1689m": [128, 256, 512, 1024],
            "convnext_large.dinov3_lvd1689m": [192, 384, 768, 1536],
        }
        depths = {
            "convnext_tiny.dinov3_lvd1689m": [3, 3, 9, 3],
            "convnext_small.dinov3_lvd1689m": [3, 3, 27, 3],
            "convnext_base.dinov3_lvd1689m": [3, 3, 27, 3],
            "convnext_large.dinov3_lvd1689m": [3, 3, 27, 3],
        }
        config = DINOv3ConvNextConfig(
            hidden_sizes=hidden_sizes[timm_name],
            depths=depths[timm_name],
            image_size=224, patch_size=4,
        )
        config.save_pretrained(str(local_dir))
    else:
        mapped = _map_vit_weights(timm_state)
        from transformers.models.dinov3_vit import DINOv3ViTConfig
        # Extract config from timm model
        embed_dim = timm_state['cls_token'].shape[-1]
        num_layers = sum(1 for k in timm_state if k.startswith('blocks.') and k.endswith('.norm1.weight'))
        num_heads = {
            384: 6, 768: 12, 1024: 16, 1280: 16, 1536: 24,
        }.get(embed_dim, 12)
        intermediate_size = timm_state['blocks.0.mlp.fc1.weight'].shape[0]
        num_reg = timm_state.get('reg_token', torch.zeros(1, 0, embed_dim)).shape[1]

        config = DINOv3ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            image_size=224, patch_size=16,
            num_register_tokens=num_reg,
            _attn_implementation="eager",
        )
        config.save_pretrained(str(local_dir))

    save_file(mapped, str(local_dir / "model.safetensors"))
    print(f"  ✓ Saved to: {local_dir} ({len(mapped)} keys)")
    return str(local_dir)


def _map_vit_weights(timm_state):
    """Map timm ViT DINOv3 weights to HuggingFace format."""
    import torch
    mapped = {}
    embed_dim = timm_state['cls_token'].shape[-1]

    mapped['embeddings.cls_token'] = timm_state['cls_token']
    mapped['embeddings.mask_token'] = torch.zeros(1, 1, embed_dim)
    mapped['embeddings.register_tokens'] = timm_state['reg_token']
    mapped['embeddings.patch_embeddings.weight'] = timm_state['patch_embed.proj.weight']
    mapped['embeddings.patch_embeddings.bias'] = timm_state['patch_embed.proj.bias']

    num_layers = sum(1 for k in timm_state if k.startswith('blocks.') and k.endswith('.norm1.weight'))
    for i in range(num_layers):
        p, h = f'blocks.{i}', f'model.layer.{i}'
        mapped[f'{h}.norm1.weight'] = timm_state[f'{p}.norm1.weight']
        mapped[f'{h}.norm1.bias'] = timm_state[f'{p}.norm1.bias']
        mapped[f'{h}.norm2.weight'] = timm_state[f'{p}.norm2.weight']
        mapped[f'{h}.norm2.bias'] = timm_state[f'{p}.norm2.bias']

        qkv_w = timm_state[f'{p}.attn.qkv.weight']
        mapped[f'{h}.attention.q_proj.weight'] = qkv_w[:embed_dim, :]
        mapped[f'{h}.attention.k_proj.weight'] = qkv_w[embed_dim:2*embed_dim, :]
        mapped[f'{h}.attention.v_proj.weight'] = qkv_w[2*embed_dim:, :]
        mapped[f'{h}.attention.q_proj.bias'] = torch.zeros(embed_dim)
        mapped[f'{h}.attention.v_proj.bias'] = torch.zeros(embed_dim)

        mapped[f'{h}.attention.o_proj.weight'] = timm_state[f'{p}.attn.proj.weight']
        mapped[f'{h}.attention.o_proj.bias'] = timm_state[f'{p}.attn.proj.bias']
        mapped[f'{h}.layer_scale1.lambda1'] = timm_state[f'{p}.gamma_1']
        mapped[f'{h}.layer_scale2.lambda1'] = timm_state[f'{p}.gamma_2']
        mapped[f'{h}.mlp.up_proj.weight'] = timm_state[f'{p}.mlp.fc1.weight']
        mapped[f'{h}.mlp.up_proj.bias'] = timm_state[f'{p}.mlp.fc1.bias']
        mapped[f'{h}.mlp.down_proj.weight'] = timm_state[f'{p}.mlp.fc2.weight']
        mapped[f'{h}.mlp.down_proj.bias'] = timm_state[f'{p}.mlp.fc2.bias']

    mapped['norm.weight'] = timm_state['norm.weight']
    mapped['norm.bias'] = timm_state['norm.bias']
    return mapped


def _map_convnext_weights(timm_state):
    """Map timm ConvNeXt DINOv3 weights to HuggingFace format."""
    mapped = {}
    for timm_key, val in timm_state.items():
        parts = timm_key.split('.')
        if timm_key.startswith('stem.'):
            idx = parts[1]; rest = '.'.join(parts[2:])
            mapped[f'model.stages.0.downsample_layers.{idx}.{rest}'] = val
        elif timm_key.startswith('stages.'):
            si = parts[1]
            if 'blocks' in timm_key:
                bi = parts[3]; comp = parts[4:]
                if comp[0] == 'gamma':
                    mapped[f'model.stages.{si}.layers.{bi}.gamma'] = val
                elif comp[0] == 'conv_dw':
                    mapped[f'model.stages.{si}.layers.{bi}.depthwise_conv.{ ".".join(comp[1:]) }'] = val
                elif comp[0] == 'norm':
                    mapped[f'model.stages.{si}.layers.{bi}.layer_norm.{ ".".join(comp[1:]) }'] = val
                elif comp[0] == 'mlp':
                    pn = 'pointwise_conv1' if comp[1] == 'fc1' else 'pointwise_conv2'
                    mapped[f'model.stages.{si}.layers.{bi}.{pn}.{ ".".join(comp[2:]) }'] = val
            elif 'downsample.' in timm_key:
                di = parts[3]; rest = '.'.join(parts[4:])
                mapped[f'model.stages.{si}.downsample_layers.{di}.{rest}'] = val
            elif 'norm' in timm_key:
                rest = '.'.join(parts[2:])
                mapped[f'model.stages.{si}.norm.{rest}'] = val
        elif timm_key.startswith('head.norm.'):
            rest = '.'.join(parts[2:])
            mapped[f'layer_norm.{rest}'] = val
    return mapped


def download_via_hf(model_name: str, out_dir: str):
    """Download a single model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    local_dir = Path(out_dir) / model_name.split("/")[-1]
    print(f"\nDownloading {model_name} -> {local_dir}")

    try:
        path = snapshot_download(model_name, local_dir=str(local_dir))
        print(f"  ✓ Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"  Make sure you have accepted the license and logged in:")
        print(f"    1. Visit https://huggingface.co/{model_name}")
        print(f"    2. Accept the license agreement")
        print(f"    3. Run: huggingface-cli login")
        return None


def main(args):
    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    method = args.method or "timm"

    if method == "timm":
        if args.all_small:
            models = TIMM_SMALL
        elif args.all:
            models = list(TIMM_MODELS.keys())
        elif args.model:
            models = [args.model]
        else:
            print("Specify --model, --all-small, or --all")
            print(f"Available timm models: {', '.join(TIMM_MODELS.keys())}")
            return

        print(f"Downloading {len(models)} model(s) via timm to {out_dir}")
        results = {}
        for model_key in models:
            result = download_via_timm(model_key, out_dir)
            results[model_key] = result

    elif method == "hf":
        if args.all_small:
            models = [m for m in HF_MODELS if "vits16" in m and "plus" not in m
                       or "convnext-tiny" in m]
        elif args.all:
            models = HF_MODELS
        elif args.model:
            models = [args.model]
        else:
            print("Specify --model, --all-small, or --all")
            return

        print(f"Downloading {len(models)} model(s) via HuggingFace Hub to {out_dir}")
        results = {}
        for model_name in models:
            result = download_via_hf(model_name, out_dir)
            results[model_name] = result

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}")
    print()
    successful = sum(1 for v in results.values() if v)
    print(f"  {successful}/{len(results)} models downloaded successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DINOv3 models")
    parser.add_argument("--method", type=str, choices=["timm", "hf"], default="timm",
                        help="Download method: 'timm' (no auth needed) or 'hf' (requires HF login)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to download. For timm: vits16, convnext-tiny, etc. For HF: full HF name")
    parser.add_argument("--all-small", action="store_true",
                        help="Download all small models (ViT-S/16, ConvNeXt-Tiny)")
    parser.add_argument("--all", action="store_true",
                        help="Download all DINOv3 models")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()
    main(args)
