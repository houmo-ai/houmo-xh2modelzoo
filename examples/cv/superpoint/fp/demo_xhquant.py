import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
import argparse
import random
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from xhquant.api import (
    Config,
    ConfigDict,
    DeviceType,
    FrontendType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    export_onnx,
    get_root_logger,
    ptq_quantize,
    to_export_graph,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.common.types import DeviceType, FrontendType, PrecisionMode
from xhquant.utils.logger import padding_message
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = (scores == max_pool(scores)).float()
    for _ in range(2):
        supp_mask = (max_pool(max_mask) > 0).float()
        supp_scores = supp_mask * zeros + (1 - supp_mask) * scores
        new_max_mask = (supp_scores == max_pool(supp_scores)).float()
        not_supp_mask = 1 - supp_mask
        max_mask = max_mask + new_max_mask * not_supp_mask - max_mask * new_max_mask * not_supp_mask
    return max_mask * scores + (1 - max_mask) * zeros


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, stride: int = 8):
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * stride)
    keypoints = keypoints * 2 - 1
    descriptors = F.grid_sample(descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False)
    descriptors = F.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def extract_superpoint_keypoints_and_descriptors(keypoints, scores, descriptors, keep_k_points=1000):
    keypoints = keypoints.cpu().numpy()
    scores = scores.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    if len(scores) > keep_k_points:
        sorted_idx = np.argsort(scores)[::-1][:keep_k_points]
    else:
        sorted_idx = np.arange(len(scores))

    keypoints = keypoints[sorted_idx]
    desc = descriptors[sorted_idx]

    keypoints_cv = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints_cv, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    H, inliers = cv2.findHomography(matched_pts1,
                                    matched_pts2,
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


class SuperPointONNX:
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "stride": 8,
    }

    def __init__(self, onnx_path: str, device: str = "cuda", input_shape: List[int] = [480, 640]):
        self.onnx_path = onnx_path
        self.device = device
        self.input_shape = [1, 1, input_shape[0], input_shape[1]] if isinstance(input_shape, list) else input_shape
        self.stride = 8
        self.nms_radius = 4
        self.detection_threshold = 0.005
        self.remove_borders = 4
        self.max_num_keypoints = None

    def pre_process(self, img: Tensor) -> Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.shape[1] == 3:
            scale = img.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            img = (img * scale).sum(1, keepdim=True)
        img = img / 255.0 if img.max() > 1.0 else img
        return img

    def post_process(self, outputs: List[Tensor]) -> dict:
        scores, descriptors_dense = outputs
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
            descriptors_dense = torch.from_numpy(descriptors_dense)
        scores = scores.to(self.device)
        descriptors_dense = descriptors_dense.to(self.device)

        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * self.stride, w * self.stride)
        scores = batched_nms(scores, self.nms_radius)

        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        if b > 1:
            idxs = torch.where(scores > self.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.detection_threshold)

        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores_list = []
        descriptors = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], self.stride)
            keypoints.append(k)
            scores_list.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return {
            "keypoints": keypoints,
            "keypoint_scores": scores_list,
            "descriptors": descriptors,
        }

    def dataset(self, calib_num: int = 32, test_batch_size: int = 1, subset: int = 1000):
        return None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", default="superpoint_480x640.onnx", type=str)
    parser.add_argument("--input_shape", default=[1, 1, 480, 640], type=int, nargs="+", help="[b,c,h,w]")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--img1_path", type=str, default="/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/1.ppm")
    parser.add_argument("--img2_path", type=str, default="/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/2.ppm")
    parser.add_argument("--save_golden", action="store_true", help="save golden model")
    parser.add_argument("--calib_num", default=32, type=int)
    parser.add_argument("--test_num", default=1000, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger = get_root_logger()
    torch.manual_seed(1024)

    onnx_path = Path(__file__).parent / args.onnx_path
    sp_onnx = SuperPointONNX(
        onnx_path=str(onnx_path),
        device=args.device,
        input_shape=args.input_shape[-2:],
    )
    onnx_name = args.onnx_path.split("/")[-1][:-5]
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a

    xhquant_init(None, debug=args.debug)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    start = time.time()
    logger.info(f"Start convert onnx to hmonnx, time: {start}")

    if not os.path.exists(out_hmonnx_file):
        input_names = ["image"]
        output_names = ["scores", "descriptors_dense"]
        input_args: List[Tensor] = [torch.randn(args.input_shape, dtype=torch.float32)]

        convert_onnx_to_hmonnx(
            sp_onnx.onnx_path,
            input_args,
            DeviceType.XH2a,
            out_hmonnx_file,
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )

    session = HMONNXInference(out_hmonnx_file)
    if args.save_golden:
        start = time.time()
        logger.info(f"Start save golden model, time: {start}")
        session.save_golden = True
        session.save_golden_dir = (
            work_dirs / "hmonnx" / f"{onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}_golden"
        )
        end = time.time()
        logger.info(f"Save golden model success, time: {end - start}")

        with torch.no_grad():
            img = cv2.imread(args.img1_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (640, 480))
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(args.device)
            pre_out = sp_onnx.pre_process(img_tensor)
            nn_out = session.forward(pre_out.half())
            if isinstance(nn_out, (tuple, list)):
                nn_out = [o.to(pre_out.dtype) for o in nn_out]
            else:
                nn_out = nn_out.to(pre_out)
            post_out = sp_onnx.post_process(nn_out)
            print(f"Golden saved, keypoints: {len(post_out['keypoints'][0])}")

    torch.set_grad_enabled(False)
    if not args.no_eval:
        session.save_golden = False
        session.exec_device = args.device
        session.to(args.device)

        img1 = cv2.imread(args.img1_path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (640, 480))
        img1_tensor = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0) / 255.0
        img1_tensor = img1_tensor.to(args.device)

        img2 = cv2.imread(args.img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (640, 480))
        img2_tensor = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0) / 255.0
        img2_tensor = img2_tensor.to(args.device)

        pre_out1 = sp_onnx.pre_process(img1_tensor)
        pre_out2 = sp_onnx.pre_process(img2_tensor)

        nn_out1 = session.forward(pre_out1.half())
        nn_out2 = session.forward(pre_out2.half())

        if isinstance(nn_out1, (tuple, list)):
            nn_out1 = [o.to(pre_out1.dtype) for o in nn_out1]
            nn_out2 = [o.to(pre_out2.dtype) for o in nn_out2]
        else:
            nn_out1 = nn_out1.to(pre_out1)
            nn_out2 = nn_out2.to(pre_out2)

        post_out1 = sp_onnx.post_process(nn_out1)
        post_out2 = sp_onnx.post_process(nn_out2)

        print(f"Image1 keypoints: {len(post_out1['keypoints'][0])}")
        print(f"Image2 keypoints: {len(post_out2['keypoints'][0])}")

        img1_color = cv2.imread(args.img1_path)
        img1_color = cv2.resize(img1_color, (640, 480))
        img2_color = cv2.imread(args.img2_path)
        img2_color = cv2.resize(img2_color, (640, 480))

        kp1_cv, desc1_cv = extract_superpoint_keypoints_and_descriptors(
            post_out1["keypoints"][0],
            post_out1["keypoint_scores"][0],
            post_out1["descriptors"][0],
        )
        kp2_cv, desc2_cv = extract_superpoint_keypoints_and_descriptors(
            post_out2["keypoints"][0],
            post_out2["keypoint_scores"][0],
            post_out2["descriptors"][0],
        )

        m_kp1, m_kp2, matches = match_descriptors(kp1_cv, desc1_cv, kp2_cv, desc2_cv)
        H, inliers = compute_homography(m_kp1, m_kp2)

        matches = np.array(matches)[inliers.astype(bool)].tolist()
        matched_img = cv2.drawMatches(img1_color, kp1_cv, img2_color, kp2_cv, matches,
                                      None, matchColor=(0, 255, 0),
                                      singlePointColor=(0, 0, 255))

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "superpoint_onnx_matches.png"), matched_img)
        print(f"Saved results to {output_dir}")