import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from tqdm import tqdm

# --- 0. 配置日志 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# 尝试导入HMONNX，如果失败则优雅降级
try:
    from xhquant.api import HMONNXInference
except ImportError:
    logging.warning("xhquant.api 未找到。HMONNX 模型将无法评估。")
    HMONNXInference = None


# --- 1. 修正后的预处理函数 ---
def preprocess_image(image_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    """
    对图像进行预处理，适配不同版本的EfficientNet。
    - BGR -> RGB
    - 应用模型特定的缩放和裁剪
    - 标准化
    - 转换为 NCHW 格式的 FP32 Tensor
    """
    # BGR to RGB
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- 关键修正：应用与训练时一致的缩放/裁剪策略 ---
    if input_size == 380:
        # EfficientNet-B4 策略: 短边缩放到384，中心裁剪到380
        h, w, _ = image.shape
        resize_size = 384
        if h < w:
            new_h = resize_size
            new_w = int(w * resize_size / h)
        else:
            new_w = resize_size
            new_h = int(h * resize_size / w)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        top = (new_h - input_size) // 2
        left = (new_w - input_size) // 2
        image = image[top : top + input_size, left : left + input_size]
    elif input_size == 480:
        # EfficientNet-V2 策略: 直接缩放到480x480
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    else:
        # 通用回退策略，并给出警告
        logging.warning(f"输入尺寸 {input_size} 没有特定的预处理策略，将使用简单的双线性缩放。")
        image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Normalize: [0, 255] -> [0, 1] -> 标准化
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    # HWC to NCHW (N=1), 确保为 float32
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    return image_tensor


# --- 2. 推理器封装类 ---
class ImageClassifier:
    """一个封装ONNX和HMONNX推理逻辑的通用分类器。"""

    def __init__(self, model_path: str, model_type: str, device: str = "cuda"):
        self.model_type = model_type.lower()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.session = None
        self.input_name = None

        logging.info(f"--- 正在加载模型 ---")
        logging.info(f"  - 类型: {self.model_type.upper()}")
        logging.info(f"  - 路径: {model_path}")

        if self.model_type == "onnx":
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        elif self.model_type == "hmonnx":
            if HMONNXInference is None:
                raise ImportError("无法评估HMONNX模型，因为xhquant库未安装。")
            self.session = HMONNXInference(model_path)
            self.session.to(self.device)
            self.input_name = self.session.get_input_names()[0]
        else:
            raise ValueError(f"不支持的模型类型: '{self.model_type}'. 请选择 'onnx' 或 'hmonnx'。")

        logging.info(f"--- 模型加载成功 on {self.device} ---")

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """执行推理并返回一个位于CPU上的logits张量。"""
        if self.model_type == "onnx":
            input_feed = {self.input_name: image_tensor.cpu().numpy()}
            outputs = self.session.run(None, input_feed)
            return torch.from_numpy(outputs[0])
        else:  # HMONNX
            input_tensor_device = image_tensor.to(self.device)
            input_tensor_device = input_tensor_device.half()
            outputs = self.session.run({self.input_name: input_tensor_device})
            logits = outputs[0]
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            return logits.float().cpu()


# --- 3. 主评估函数 ---
def main(args):
    val_dir = Path(args.data_dir)
    if not val_dir.is_dir():
        logging.error(f"[错误] ImageNet验证集目录未找到: {val_dir}")
        return

    # 1. 加载数据集并构建标签映射
    logging.info(f"--- 正在扫描ImageNet验证集: {val_dir} ---")
    image_paths = sorted(list(val_dir.rglob("*.JPEG")))

    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    synset_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    if not image_paths or not synset_to_idx:
        logging.error(f"在 {val_dir} 中未找到图片或类别子目录。请检查路径。")
        return

    if args.limit > 0:
        image_paths = image_paths[: args.limit]
        logging.info(f"  - [注意] 已将评估样本数量限制为: {args.limit}")

    logging.info(f"  - 找到 {len(image_paths)} 张图片和 {len(synset_to_idx)} 个类别。")

    # 2. 初始化模型
    classifier = ImageClassifier(args.model_path, args.model_type)

    # 3. 循环评估
    top1_correct, top5_correct, total_images = 0, 0, 0

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f"评估 ({args.model_type.upper()})"):
            total_images += 1

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                logging.warning(f"无法读取图片 {img_path}, 跳过。")
                total_images -= 1
                continue

            input_tensor = preprocess_image(image_bgr, args.input_size)

            gt_synset = img_path.parent.name
            gt_label = synset_to_idx[gt_synset]

            logits = classifier(input_tensor)

            _, pred_indices = torch.topk(logits, 5, dim=1)
            pred_indices = pred_indices.squeeze(0)

            if gt_label in pred_indices:
                top5_correct += 1
            if gt_label == pred_indices[0]:
                top1_correct += 1

    # 4. 打印结果
    if total_images == 0:
        logging.error("没有成功处理任何图片，无法计算准确率。")
        return

    top1_accuracy = (top1_correct / total_images) * 100
    top5_accuracy = (top5_correct / total_images) * 100

    print("\n" + "=" * 50)
    print(" " * 18 + "评估结果")
    print("=" * 50)
    print(f"  模型路径: {Path(args.model_path).name}")
    print(f"  模型类型: {args.model_type.upper()}")
    print(f"  输入尺寸: {args.input_size}x{args.input_size}")
    print(f"  总计评估图片: {total_images}")
    print("-" * 50)
    print(f"  Top-1 准确率: {top1_accuracy:.2f}% ({top1_correct}/{total_images})")
    print(f"  Top-5 准确率: {top5_accuracy:.2f}% ({top5_correct}/{total_images})")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在ImageNet验证集上评估分类模型的Top-1和Top-5准确率。",
        formatter_class=argparse.RawTextHelpFormatter,  # 保持帮助信息格式
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["onnx", "hmonnx"],
        help="要评估的模型类型: 'onnx' (FP32) 或 'hmonnx' (量化后)。",
    )
    parser.add_argument("--model-path", type=str, required=True, help="ONNX 或 HMONNX 模型的路径。")
    parser.add_argument(
        "--input-size",
        type=int,
        required=True,
        help="模型的输入尺寸 (例如, 380 for EfficientNet-B4, 480 for EfficientNet-V2-M)。",
    )
    parser.add_argument("--data-dir", type=str, required=True, help="ImageNet验证集的根目录路径。")
    parser.add_argument(
        "--limit", type=int, default=5000, help="限制评估的图片数量，用于快速测试 (0表示使用全部验证集)。"
    )

    # 添加调用示例
    parser.epilog = """
调用示例:
--------------------------------------------------------------------------------
1. 评估 FP32 EfficientNet-B4 模型 (使用500张图片进行快速测试):
   python %(prog)s \\
     --model-type onnx \\
     --model-path /path/to/your/efficientnet_b4_pretrained.onnx \\
     --input-size 380 \\
     --data-dir /path/to/imagenet/val \\
     --limit 500

2. 评估量化后的 EfficientNet-V2-M 模型 (在整个验证集上):
   python %(prog)s \\
     --model-type hmonnx \\
     --model-path /path/to/your/quantized_efficientnet_v2_m.onnx \\
     --input-size 480 \\
     --data-dir /path/to/imagenet/val \\
     --limit 0
--------------------------------------------------------------------------------
"""

    args = parser.parse_args()
    main(args)
