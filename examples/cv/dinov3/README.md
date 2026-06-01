# DINOv3 Quantization Example

Export and quantize [DINOv3](https://github.com/facebookresearch/dinov3) models (ViT and ConvNeXt) to HMONNX format for XH2a deployment.

DINOv3 is Meta AI's self-supervised vision foundation model with two architecture families:
- **ViT (Vision Transformer)** — with RoPE positional embeddings and register tokens
- **ConvNeXt** — efficient CNN-based alternative distilled from ViT-7B

## Supported Models

### ViT Backbones
| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| ViT-S/16 distilled | 21M | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| ViT-S+/16 distilled | 29M | `facebook/dinov3-vits16plus-pretrain-lvd1689m` |
| ViT-B/16 distilled | 86M | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| ViT-L/16 distilled | 300M | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| ViT-H+/16 distilled | 840M | `facebook/dinov3-vith16plus-pretrain-lvd1689m` |
| ViT-7B/16 | 6.7B | `facebook/dinov3-vit7b16-pretrain-lvd1689m` |

### ConvNeXt Backbones
| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| ConvNeXt Tiny | 29M | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` |
| ConvNeXt Small | 50M | `facebook/dinov3-convnext-small-pretrain-lvd1689m` |
| ConvNeXt Base | 89M | `facebook/dinov3-convnext-base-pretrain-lvd1689m` |
| ConvNeXt Large | 198M | `facebook/dinov3-convnext-large-pretrain-lvd1689m` |

## Pipeline

```
Download ──► PyTorch Model ──► ONNX ──► HMONNX ──► Accuracy Test
(download_models.py) (export_onnx.py) (quantize.py) (test_accuracy.py)
```

## Prerequisites

Install dependencies in the `xh2` conda environment:
```bash
conda activate xh2
pip install timm transformers onnx onnxruntime onnxsim datasets peft
```

## Usage

### 1. Download Models

**Method 1: Via timm (recommended, no authentication needed)**

```bash
# Download small models (ViT-S/16 + ConvNeXt-Tiny)
python examples/cv/dinov3/download_models.py --method timm --all-small

# Download a specific model
python examples/cv/dinov3/download_models.py --method timm --model vits16
python examples/cv/dinov3/download_models.py --method timm --model convnext-tiny
```

**Method 2: Via HuggingFace Hub (requires authentication)**

DINOv3 models are **gated** on HuggingFace — you must first accept the license at the model page and login.
```bash
huggingface-cli login
python examples/cv/dinov3/download_models.py --method hf --model facebook/dinov3-vits16-pretrain-lvd1689m
```

### 2. Export PyTorch to ONNX

```bash
# ViT-S/16 (local path)
python examples/cv/dinov3/export_onnx.py \
    --model-type vit \
    --model-name /data01/datasets/dinov3-vits16-pretrain-lvd1689m

# ConvNeXt Tiny
python examples/cv/dinov3/export_onnx.py \
    --model-type convnext \
    --model-name /data01/datasets/dinov3-convnext-tiny-pretrain-lvd1689m

# ViT-L/16 with custom image size
python examples/cv/dinov3/export_onnx.py \
    --model-type vit \
    --model-name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --image-size 518
```

### 3. Quantize to HMONNX

```bash
# ViT-S/16 → w8a8_sefp HMONNX
python examples/cv/dinov3/quantize.py \
    --onnx examples/cv/dinov3/onnx/dinov3-vits16-pretrain-lvd1689m_is224.onnx

# ConvNeXt Tiny → w8a8_sefp HMONNX
python examples/cv/dinov3/quantize.py \
    --onnx examples/cv/dinov3/onnx/dinov3-convnext-tiny-pretrain-lvd1689m_is224.onnx
```

### 4. Test Accuracy

```bash
# Numerical accuracy test (compare HMONNX vs PyTorch float on random inputs)
python examples/cv/dinov3/test_accuracy.py \
    --model-name /data01/datasets/dinov3-vits16-pretrain-lvd1689m \
    --model-type vit \
    --hmonnx work_dirs/dinov3-vits16-pretrain-lvd1689m_is224/hmonnx/dinov3-vits16-pretrain-lvd1689m_is224_w8a8_sefp_XH2a.onnx

# With ImageNet validation set (k-NN accuracy)
python examples/cv/dinov3/test_accuracy.py \
    --model-name /data01/datasets/dinov3-vits16-pretrain-lvd1689m \
    --model-type vit \
    --hmonnx work_dirs/.../xxx_w8a8_sefp_XH2a.onnx \
    --imagenet-val /path/to/imagenet/val \
    --num-samples 1000
```

## Model I/O

- **Input**: `pixel_values` — RGB image tensor `(B, 3, H, W)`, values in [0, 1]
- **Outputs**:
  - `pooler_output` — CLS/pooled token `(B, hidden_size)`
  - `last_hidden_state` — All patch tokens `(B, N+1, hidden_size)` where N = num_patches

## Verified Numerical Accuracy (w8a8_sefp quantization)

### Numerical Accuracy (100 random samples)

| Model | Pooler CosSim | Hidden CosSim | Pooler MSE |
|-------|---------------|---------------|------------|
| ViT-S/16 | 0.9919 | 0.9935 | 0.0044 |
| ConvNeXt-Tiny | 0.9972 | 0.9946 | 0.0191 |

### CIFAR-10 k-NN Classification Accuracy (500 samples)

Tested on CIFAR-10 test set with k-NN classifier using CLS embeddings:

| Model | k=1 Acc | k=5 Acc | k=20 Acc | Embed CosSim |
|-------|---------|---------|----------|--------------|
| ViT-S/16 Float | 100.00% | 93.80% | 90.40% | — |
| ViT-S/16 HMONNX | 100.00% | 94.20% | 91.40% | 0.9929 |

**Note**: The original DINOv3 checkpoints from HuggingFace have a `model.` prefix in safetensors keys that is incompatible with transformers' expected key format. Run `download_models.py` which automatically remaps keys, or manually fix with:
```python
from safetensors import safe_open
from safetensors.torch import save_file
# Remap keys by removing 'model.' prefix
```

## Official Accuracy Reference (DINOv3 ViT Family)

| Model | IN-ReaL | IN-R | Obj.Net | ADE20k |
|-------|---------|------|---------|--------|
| ViT-S/16 | 87.0 | 60.4 | 50.9 | 47.0 |
| ViT-B/16 | 89.3 | 76.7 | 64.1 | 51.8 |
| ViT-L/16 | 90.2 | 88.1 | 74.8 | 54.9 |
| ViT-7B/16 | 90.4 | 91.1 | 91.1 | 55.9 |
