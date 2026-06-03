# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Run image classification with ONNX and HMONNX MobileNetV2 models.

Example:
    python examples/cv/mobilnet_v2/classify_onnx_hmonnx.py \
        --image examples/cv/openpose/demo.jpg \
        --onnx examples/cv/mobilnet_v2/mobilenetv2_224x224.onnx \
        --build-hmonnx

The script uses ImageNet preprocessing and prints top-k predictions from both
the original ONNX model and the converted HMONNX model, plus numeric metrics
between their logits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="Image path to classify.",
    )
    parser.add_argument(
        "--onnx",
        default="examples/cv/mobilnet_v2/mobilenetv2_224x224.onnx",
        help="Original ONNX model path.",
    )
    parser.add_argument(
        "--hmonnx",
        default=None,
        help="HMONNX model path. If omitted with --build-hmonnx, it is generated under --work-dir.",
    )
    parser.add_argument(
        "--build-hmonnx",
        action="store_true",
        help="Convert --onnx to HMONNX before running classification.",
    )
    parser.add_argument(
        "--with-resizer",
        action="store_true",
        help=(
            "Build/run HMONNX with an input resizer op. ONNX baseline still uses standard "
            "ImageNet preprocessing; HMONNX takes a uint8 RGB image tensor plus crop info."
        ),
    )
    parser.add_argument(
        "--resizer-mode",
        choices=["dynamic-full", "static-center-crop"],
        default="static-center-crop",
        help=(
            "Resizer input mode used with --with-resizer. static-center-crop first resizes the "
            "image to 256x256 and lets the HMONNX resizer crop [28:252, 28:252], matching "
            "the ONNX ImageNet baseline. dynamic-full feeds the original image plus crop_info."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="work_dirs/mobilenetv2_224x224_cls",
        help="Directory used for generated HMONNX artifacts.",
    )
    parser.add_argument(
        "--quant-type",
        default="w8a8h1_sefp",
        help="Quant type passed to QuantScheme when --build-hmonnx is set.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for HMONNX execution.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of predictions to print.",
    )
    return parser.parse_args()


def load_imagenet_categories() -> list[str]:
    try:
        from torchvision.models import MobileNet_V2_Weights

        return list(MobileNet_V2_Weights.DEFAULT.meta["categories"])
    except Exception:
        return [str(i) for i in range(1000)]


def preprocess_image(image_path: str | Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.BILINEAR)
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(array, dtype=np.float32)


def load_rgb_uint8_image(image_path: str | Path, resizer_mode: str) -> tuple[np.ndarray, np.ndarray | None]:
    image = Image.open(image_path).convert("RGB")
    if resizer_mode == "static-center-crop":
        image = image.resize((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
        chw = np.transpose(array, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(chw), None

    if resizer_mode != "dynamic-full":
        raise ValueError(f"Unsupported resizer_mode: {resizer_mode}")

    array = np.asarray(image, dtype=np.uint8)
    chw = np.transpose(array, (2, 0, 1))[None, ...]
    h, w = array.shape[:2]
    crop_info = np.array([[0, 0, h, w, 224, 224, 0, 0, 0, 0]], dtype=np.int32)
    return np.ascontiguousarray(chw), crop_info


def run_onnx(onnx_path: str | Path, image_tensor: np.ndarray) -> tuple[np.ndarray, str, str]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run(None, {input_name: image_tensor})[0]
    return output, input_name, output_name


def build_hmonnx(
    onnx_path: str | Path,
    hmonnx_path: str | Path,
    image_tensor: np.ndarray,
    input_name: str,
    output_name: str,
    quant_type: str,
    with_resizer: bool,
    resizer_mode: str,
    resizer_image: np.ndarray | None = None,
    crop_info: np.ndarray | None = None,
) -> None:
    from xhquant.api import DeviceType, QuantScheme, ResizerScheme, convert_onnx_to_hmonnx, create_quant_config

    if with_resizer:
        if resizer_image is None:
            raise ValueError("resizer_image must be provided when with_resizer=True")
        dynamic_crop = resizer_mode == "dynamic-full"
        if dynamic_crop and crop_info is None:
            raise ValueError("crop_info must be provided when resizer_mode=dynamic-full")
        if resizer_mode == "static-center-crop":
            crop_size = (224, 224)
            crop_offset = (28, 28)
        elif resizer_mode == "dynamic-full":
            crop_size = tuple(int(v) for v in resizer_image.shape[-2:])
            crop_offset = (0, 0)
        else:
            raise ValueError(f"Unsupported resizer_mode: {resizer_mode}")
        input_ppc_config = [
            ResizerScheme(
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                fmt="rgb",
                int_trans=True,
                crop_size=crop_size,
                crop_offset=crop_offset,
                pad_size=(0, 0, 0, 0),
                pad_value=0,
                mean=IMAGENET_MEAN.tolist(),
                std=IMAGENET_STD.tolist(),
                dynamic_crop=dynamic_crop,
                model_inp_fmt="rgb",
            ).to_dict()
        ]
    else:
        input_ppc_config = None

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type, input_ppc_config=input_ppc_config)
    quant_config = create_quant_config(quant_scheme)
    Path(hmonnx_path).parent.mkdir(parents=True, exist_ok=True)
    convert_inputs = [torch.from_numpy(image_tensor)]
    input_names = [input_name]
    if with_resizer:
        if resizer_mode == "dynamic-full":
            convert_inputs = [torch.from_numpy(resizer_image), torch.from_numpy(crop_info)]
            input_names = [input_name, f"resizer_crop_{input_name}"]
        else:
            convert_inputs = [torch.from_numpy(resizer_image)]
            input_names = [input_name]
    convert_onnx_to_hmonnx(
        str(onnx_path),
        convert_inputs,
        DeviceType.XH2a,
        str(hmonnx_path),
        quant_config=quant_config,
        input_names=input_names,
        output_names=[output_name],
    )


def run_hmonnx(
    hmonnx_path: str | Path,
    image_tensor: np.ndarray,
    device: str,
    crop_info: np.ndarray | None = None,
) -> np.ndarray:
    from xhquant.api import HMONNXGoldenInference

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    session = HMONNXGoldenInference(str(hmonnx_path))
    session.initialize()
    input_tensors = []
    for index, input_name in enumerate(session.get_input_names()):
        input_info = session.get_input(input_name)
        if index == 0:
            array = image_tensor
        elif crop_info is not None and "crop" in input_name:
            array = crop_info
        else:
            raise ValueError(f"No input array for HMONNX input {input_name!r}")
        input_tensors.append(torch.from_numpy(array).to(device=device, dtype=input_info.dtype))
    session.to(device)
    output = session(*input_tensors)
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().float().cpu().numpy()


def softmax_topk(logits: np.ndarray, categories: Sequence[str], top_k: int) -> list[tuple[int, str, float, float]]:
    logits_t = torch.from_numpy(logits).float().reshape(-1)
    probs = torch.softmax(logits_t, dim=-1)
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
    rows = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        label = categories[index] if index < len(categories) else str(index)
        rows.append((index, label, value, logits_t[index].item()))
    return rows


def print_topk(title: str, rows: Sequence[tuple[int, str, float, float]]) -> None:
    print(f"\n{title}")
    print("rank | class_id | probability | logit | label")
    for rank, (class_id, label, prob, logit) in enumerate(rows, start=1):
        print(f"{rank:>4} | {class_id:>8} | {prob:>11.6f} | {logit:>8.4f} | {label}")


def print_metrics(onnx_logits: np.ndarray, hmonnx_logits: np.ndarray) -> None:
    onnx_t = torch.from_numpy(onnx_logits).float().reshape(-1)
    hmonnx_t = torch.from_numpy(hmonnx_logits).float().reshape(-1)
    cos = F.cosine_similarity(onnx_t[None], hmonnx_t[None]).item()
    diff = (onnx_t - hmonnx_t).abs()
    print("\nOutput metrics")
    print(f"cosine_similarity: {cos:.8f}")
    print(f"max_abs_diff     : {diff.max().item():.8f}")
    print(f"mean_abs_diff    : {diff.mean().item():.8f}")
    print(f"rmse             : {torch.sqrt((diff ** 2).mean()).item():.8f}")
    print(f"same_top1        : {int(onnx_t.argmax()) == int(hmonnx_t.argmax())}")


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    image_path = Path(args.image)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_tensor = preprocess_image(image_path)
    resizer_image, crop_info = load_rgb_uint8_image(image_path, args.resizer_mode)
    onnx_logits, input_name, output_name = run_onnx(onnx_path, image_tensor)

    suffix = f"XH2a_{args.quant_type}"
    if args.with_resizer:
        suffix += "_resizer"
    hmonnx_path = Path(args.hmonnx) if args.hmonnx else Path(args.work_dir) / f"{onnx_path.stem}_{suffix}.onnx"
    if args.build_hmonnx or not hmonnx_path.exists():
        print(f"Building HMONNX: {hmonnx_path}")
        build_hmonnx(
            onnx_path,
            hmonnx_path,
            image_tensor,
            input_name,
            output_name,
            args.quant_type,
            args.with_resizer,
            args.resizer_mode,
            resizer_image=resizer_image,
            crop_info=crop_info,
        )
    if not hmonnx_path.exists():
        raise FileNotFoundError(f"HMONNX model not found: {hmonnx_path}")

    hmonnx_input = resizer_image if args.with_resizer else image_tensor
    hmonnx_logits = run_hmonnx(hmonnx_path, hmonnx_input, args.device, crop_info=crop_info if args.with_resizer else None)
    categories = load_imagenet_categories()
    onnx_topk = softmax_topk(onnx_logits, categories, args.top_k)
    hmonnx_topk = softmax_topk(hmonnx_logits, categories, args.top_k)

    print(f"Image : {image_path}")
    print(f"ONNX  : {onnx_path}")
    print(f"HMONNX: {hmonnx_path}")
    print(f"Resizer enabled: {args.with_resizer}")
    if args.with_resizer:
        print(f"Resizer mode   : {args.resizer_mode}")
    print_topk("ONNX top-k", onnx_topk)
    print_topk("HMONNX top-k", hmonnx_topk)
    print_metrics(onnx_logits, hmonnx_logits)


if __name__ == "__main__":
    main()