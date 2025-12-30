import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import xhquant.xhonnxruntime.config
from tqdm import tqdm
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init

from xh_model_zoo_new.xh_base import BaseSingleModelInfer


def build_dataloader(data_root: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    """Build ImageNet-style dataloader using torchvision ImageFolder."""
    data_root_path = Path(data_root)
    assert data_root_path.is_dir(), f"dataset dir not found: {data_root}"

    # Match preprocessing in resnet50_hmonnx_test.py:
    # resize to 224x224 and normalize with ImageNet mean/std.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def evaluate(args):
    xhquant_init(None, debug=args.debug)
    logger = get_root_logger()

    val_dataloader = build_dataloader(args.dataset, args.batch_size, args.num_workers)

    # Build inference model from HMONNX using BaseSingleModelInfer
    infer = BaseSingleModelInfer.from_hmonnx(args.hmonnx)
    session: HMONNXInference = infer.model  # underlying HMONNXInference

    if args.fast:
        session.to_fast_mode()

    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if exec_device.type == "cuda" else torch.float32
    session.to(exec_device)

    logger.info(f"session is created successfully on device: {exec_device}, dtype: {dtype}")
    xhquant.xhonnxruntime.config.disable_progress = True

    correct_top1 = 0
    total = 0

    pbar = tqdm(val_dataloader, desc="Evaluating")
    for images, targets in pbar:
        images = images.to(device=exec_device, dtype=dtype)
        targets = targets.to(exec_device)

        with torch.inference_mode():
            # BaseSingleModelInfer forwards to underlying HMONNXInference
            logits = infer(images)

        # Top-1 accuracy
        pred = logits.argmax(dim=1)
        correct_top1 += (pred == targets).sum().item()
        total += targets.size(0)

        # Update progress bar with current top1_acc
        current_top1_acc = correct_top1 / total * 100.0 if total > 0 else 0.0
        pbar.set_postfix({"top1_acc": f"{current_top1_acc:.4f}%"})

    top1_acc = correct_top1 / total * 100.0 if total > 0 else 0.0
    logger.info(f"Top-1 accuracy on {total} samples: {top1_acc:.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmonnx",
        type=str,
        default="work_dirs/resnet50_224x224/hmonnx/resnet50_224x224-XH2a.onnx",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default = "/data01/datasets/imagenet/val",
        help="Path to classification dataset root (ImageFolder style).",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fast", action="store_true", help="use HMONNX fast mode")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
