import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init

from xh2_model_zoo.utils.time_profiler import TimeProfiler


def preprocess_image(image: np.ndarray, input_size: int) -> torch.Tensor:
    """
    Preprocesses an image for EfficientNet models with different input sizes.

    - Applies model-specific resizing and cropping (B4 vs V2 logic).
    - Normalizes using standard ImageNet mean and std.
    - Permutes to CHW format and converts to a float16 tensor.

    Args:
        image (np.ndarray): Input image in HWC, RGB format.
        input_size (int): The target square dimension for the model input (e.g., 380, 480).

    Returns:
        torch.Tensor: The preprocessed image tensor ready for inference.
    """
    logger = get_root_logger()
    logger.info("[DEBUG] --- Entering preprocess_image ---")
    logger.info(f"[DEBUG] Initial image shape: {image.shape}, dtype: {image.dtype}")

    # --- 1. Resizing and Cropping ---
    if input_size == 380:
        logger.info(f"[DEBUG] Applying EfficientNet-B4 preprocessing for size {input_size}")
        h, w, _ = image.shape
        crop_size = 380
        resize_size = 384

        if h < w:
            new_h = resize_size
            new_w = int(w * resize_size / h)
        else:
            new_w = resize_size
            new_h = int(h * resize_size / w)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.info(f"[DEBUG] Image shape after resizing to short edge {resize_size}: {image.shape}")

        top = (new_h - crop_size) // 2
        left = (new_w - crop_size) // 2
        image = image[top : top + crop_size, left : left + crop_size]
        logger.info(f"[DEBUG] Image shape after center cropping to {crop_size}x{crop_size}: {image.shape}")

    elif input_size == 480:
        logger.info(f"[DEBUG] Applying EfficientNet-V2-M preprocessing for size {input_size}")
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        logger.info(f"[DEBUG] Image shape after resizing to {input_size}x{input_size}: {image.shape}")

    else:
        logger.warning(
            f"Unsupported input size {input_size}. Applying simple resize. "
            f"For best accuracy, use 380 (EffNet-B4) or 480 (EffNet-V2-M)."
        )
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        logger.info(f"[DEBUG] Image shape after fallback resizing to {input_size}x{input_size}: {image.shape}")

    # --- 2. Normalization ---
    # Convert image to float32 for calculations
    image = image.astype(np.float32)
    logger.info(
        f"[DEBUG] Before normalization: dtype={image.dtype}, min={image.min():.2f}, max={image.max():.2f}, mean={image.mean():.2f}"
    )

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    # Normalize the image
    image = (image / 255.0 - mean) / std  # A more standard and stable way to normalize
    logger.info(
        f"[DEBUG] After normalization: dtype={image.dtype}, min={image.min():.2f}, max={image.max():.2f}, mean={image.mean():.2f}"
    )

    # --- 3. Convert to Tensor ---
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    logger.info(
        f"[DEBUG] After permute and unsqueeze (torch.float32): shape={image_tensor.shape}, dtype={image_tensor.dtype}"
    )

    # Convert to half precision for inference
    image_tensor = image_tensor.to(torch.float16)
    logger.info(f"[DEBUG] Final tensor (torch.float16): shape={image_tensor.shape}, dtype={image_tensor.dtype}")
    logger.info("[DEBUG] --- Exiting preprocess_image ---")

    return image_tensor


def main(args):
    # Initialize xhquant and the inference session
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    if args.fast:
        session.to_fast_mode()

    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session.to(exec_device)

    logger = get_root_logger()
    logger.info(f"Session for {args.hmonnx} created successfully on {exec_device}.")

    # --- Image Preprocessing ---
    image_file = args.image
    logger.info(f"Loading and preprocessing image: {image_file}")

    image = cv2.imread(image_file)
    if image is None:
        logger.error(f"Failed to read image file: {image_file}")
        return
    logger.info(f"[DEBUG] Image loaded with shape (H,W,C as BGR): {image.shape}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info("[DEBUG] Image converted from BGR to RGB.")

    # Use the improved preprocessing function with the specified input size
    input_tensor = preprocess_image(image, input_size=args.input_size)
    input_tensor = input_tensor.to(exec_device)
    logger.info(
        f"Final input tensor sent to model: shape={input_tensor.shape}, device='{input_tensor.device}', dtype={input_tensor.dtype}"
    )

    # --- Inference ---
    input_name = session.get_input_names()[0]

    with TimeProfiler("Inference", logger):
        cls_score = session.run({input_name: input_tensor})[0]

    logger.info(f"[DEBUG] Raw model output (cls_score): shape={cls_score.shape}, dtype={cls_score.dtype}")
    logger.info(
        f"[DEBUG] Raw model output stats: min={cls_score.min():.4f}, max={cls_score.max():.4f}, mean={cls_score.mean():.4f}"
    )
    logger.info(f"[DEBUG] Raw model output first 10 values: {cls_score.flatten()[:10].cpu().numpy()}")

    # --- Post-processing ---
    if cls_score.ndim == 1:
        cls_score = cls_score.unsqueeze(0)

    pred_scores = F.softmax(cls_score, dim=1)

    # Get the Top 5 predictions
    top5_scores, top5_labels = torch.topk(pred_scores, 5, dim=1)

    logger.info("Inference finished. Top 5 Results:")
    for i in range(top5_labels.shape[0]):
        logger.info(f"--- Image {i} ---")
        for j in range(5):
            label = top5_labels[i, j].item()
            score = top5_scores[i, j].item()
            logger.info(f"  - Rank {j+1}: Label Index: {label:4d}, Confidence: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and debug a quantized EfficientNet HMONNX model.")
    parser.add_argument("--hmonnx", type=str, required=True, help="Path to the quantized HMONNX model file.")
    parser.add_argument(
        "--image", type=str, default="data/images/ILSVRC2012_val_00002031.JPEG", help="Path to the input image."
    )
    parser.add_argument(
        "--input-size",
        type=int,
        required=True,
        help="The target input size for the model (e.g., 380 for EfficientNet-B4, 480 for EfficientNet-V2-M).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode for inference.")

    args = parser.parse_args()
    main(args)
