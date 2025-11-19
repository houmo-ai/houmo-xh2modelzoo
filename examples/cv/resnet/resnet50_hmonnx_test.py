import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from xhquant.api import HMONNXInference, get_root_logger, xhquant_init

from xh2_model_zoo.utils.time_profiler import TimeProfiler


def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(torch.float16)
    return image


def main(args):
    xhquant_init(None, debug=args.debug)
    session = HMONNXInference(args.hmonnx)
    if args.fast:
        session.to_fast_mode()
    exec_device = torch.device("cuda")
    session.to(exec_device)
    logger = get_root_logger()
    logger.info("session is created successfully")
    image_file = args.image

    # 读取图像并进行预处理
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input = preprocess_image(image)
    input = input.to(exec_device)

    input_name = session.get_input_names()[0]

    with TimeProfiler("Inference", logger):
        cls_score = session.run({input_name: input})
    pred_scores = F.softmax(cls_score, dim=1)
    pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

    for score, label in zip(pred_scores, pred_labels):
        logger.info(f"Predicted label: {label.item()}, Score: {score[label].item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hmonnx",
        type=str,
        default="work_dirs/resnet50_224x224_1x3x224x224/hmonnx/resnet50_224x224_1x3x224x224_XH2a.onnx",
    )
    parser.add_argument("--image", type=str, default="data/images/ILSVRC2012_val_00002031.JPEG")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    main(args)
