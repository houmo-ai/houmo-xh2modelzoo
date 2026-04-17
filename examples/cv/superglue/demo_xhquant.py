import os
import sys
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


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def read_image(path, device, resize, rotation=0, resize_float=False):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

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


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def normalize_keypoints(kpts, image_shape):
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class SuperPointONNX:
    default_conf = {
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "stride": 8,
    }

    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
        input_shape: List[int] = [480, 640],
        nms_radius: int = 4,
        detection_threshold: float = 0.005,
        remove_borders: int = 4,
        max_num_keypoints: int = -1,
    ):
        self.onnx_path = onnx_path
        self.device = device
        self.input_shape = [1, 1, input_shape[0], input_shape[1]] if isinstance(input_shape, list) else input_shape
        self.stride = 8
        self.nms_radius = nms_radius
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders
        self.max_num_keypoints = max_num_keypoints

    def post_process(self, outputs: List[Tensor], device: str = "cuda") -> dict:
        scores, descriptors_dense = outputs
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
            descriptors_dense = torch.from_numpy(descriptors_dense)
        scores = scores.to(device)
        descriptors_dense = descriptors_dense.to(device)

        b, h, w = scores.shape
        scores = simple_nms(scores, self.nms_radius)

        keypoints = [
            torch.nonzero(s > self.detection_threshold)
            for s in scores]
        scores_init = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        keypoints, scores_init = list(zip(*[
            remove_borders(k, s, self.remove_borders, h, w)
            for k, s in zip(keypoints, scores_init)]))

        if self.max_num_keypoints is not None and self.max_num_keypoints > 0:
            keypoints, scores_init = list(zip(*[
                select_top_k_keypoints(k, s, self.max_num_keypoints)
                for k, s in zip(keypoints, scores_init)]))

        keypoints = [k.flip(1).float() for k in keypoints]

        descriptors_dense = F.normalize(descriptors_dense, p=2, dim=1)
        descriptors = [sample_descriptors(k[None], d[None], self.stride)[0]
                       for k, d in zip(keypoints, descriptors_dense)]

        return {
            'keypoints': keypoints,
            'scores': scores_init,
            'descriptors': descriptors,
        }


class SuperGlueONNX:
    def __init__(self, onnx_path: str, device: str = "cuda", max_keypoints: int = 800,
                 sinkhorn_iterations: int = 20, match_threshold: float = 0.2,
                 bin_score: float = 1.0):
        self.onnx_path = onnx_path
        self.device = device
        self.max_keypoints = max_keypoints
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.bin_score = float(bin_score)

    def match(self, keypoints0, scores0, descriptors0, image_shape0,
              keypoints1, scores1, descriptors1, image_shape1):
        kpts0 = keypoints0.unsqueeze(0)
        kpts1 = keypoints1.unsqueeze(0)
        scores0 = scores0.unsqueeze(0)
        scores1 = scores1.unsqueeze(0)
        desc0 = descriptors0.unsqueeze(0)
        desc1 = descriptors1.unsqueeze(0)
        img0 = torch.zeros(1, 1, image_shape0[0], image_shape0[1], device=kpts0.device)
        img1 = torch.zeros(1, 1, image_shape1[0], image_shape1[1], device=kpts1.device)

        kpts0_n = normalize_keypoints(kpts0, img0.shape)
        kpts1_n = normalize_keypoints(kpts1, img1.shape)

        n0 = min(kpts0.shape[1], self.max_keypoints)
        n1 = min(kpts1.shape[1], self.max_keypoints)

        kpts0_n = kpts0_n[:, :n0]
        kpts1_n = kpts1_n[:, :n1]
        scores0 = scores0[:, :n0]
        scores1 = scores1[:, :n1]
        desc0 = desc0[:, :, :n0]
        desc1 = desc1[:, :, :n1]

        if n0 < self.max_keypoints:
            pad0 = self.max_keypoints - n0
            kpts0_n = F.pad(kpts0_n, (0, 0, 0, pad0))
            scores0 = F.pad(scores0, (0, pad0), value=-1)
            desc0 = F.pad(desc0, (0, pad0), value=0)
        if n1 < self.max_keypoints:
            pad1 = self.max_keypoints - n1
            kpts1_n = F.pad(kpts1_n, (0, 0, 0, pad1))
            scores1 = F.pad(scores1, (0, pad1), value=-1)
            desc1 = F.pad(desc1, (0, pad1), value=0)

        return {
            "keypoints0_n": kpts0_n,
            "keypoints1_n": kpts1_n,
            "scores0": scores0,
            "scores1": scores1,
            "descriptors0": desc0,
            "descriptors1": desc1,
        }

    @staticmethod
    def arange_like(x, dim):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1

    def post_process(self, scores):
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bin_score = self.bin_score
        bins0 = bin_score * scores.new_ones(b, m, 1)
        bins1 = bin_score * scores.new_ones(b, 1, n)
        alpha = bin_score * scores.new_ones(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = -(ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = couplings
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(self.sinkhorn_iterations):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        Z = Z + u.unsqueeze(2) + v.unsqueeze(1)
        Z = Z - norm

        max0, max1 = Z[:, :-1, :-1].max(2), Z[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = Z.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.match_threshold)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        indices0 = indices0.squeeze(0).cpu().numpy()
        indices1 = indices1.squeeze(0).cpu().numpy()
        mscores0 = mscores0.squeeze(0).cpu().numpy()
        mscores1 = mscores1.squeeze(0).cpu().numpy()
        valid0 = indices0 >= 0
        valid1 = indices1 >= 0
        return indices0, indices1, mscores0, mscores1, valid0, valid1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--superpoint_onnx", default="superpoint_dense.onnx", type=str)
    parser.add_argument("--superglue_onnx", default="superglue.onnx", type=str)
    parser.add_argument("--input_shape", default=[480, 640], type=int, nargs="+", help="[h, w]")
    parser.add_argument("--quant_type", default="w8a8_sefp", help="quant type")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--img1_path", type=str, default="/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/1.ppm")
    parser.add_argument("--img2_path", type=str, default="/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/2.ppm")
    parser.add_argument("--save_golden", action="store_true")
    parser.add_argument("--calib_num", default=32, type=int)
    parser.add_argument("--max_keypoints", default=800, type=int)
    parser.add_argument("--keypoint_threshold", default=0.005, type=float)
    parser.add_argument("--nms_radius", default=4, type=int)
    parser.add_argument("--remove_borders", default=4, type=int)
    parser.add_argument("--sinkhorn_iterations", default=20, type=int)
    parser.add_argument("--match_threshold", default=0.2, type=float)
    parser.add_argument("--superglue_weight", default="examples/cv/superglue/superglue_outdoor.pth", type=str)
    parser.add_argument(
        "--bin_score",
        default=None,
        type=float,
        help="Override SuperGlue dustbin score; if unset, auto-load from --superglue_weight.",
    )
    parser.add_argument("--no_eval", action="store_true", default=False)
    args = parser.parse_args()
    return args


def load_superglue_bin_score(weight_path: str, default: float = 1.0) -> float:
    if not weight_path or (not os.path.exists(weight_path)):
        return default
    try:
        state = torch.load(weight_path, map_location="cpu")
        if isinstance(state, dict) and "bin_score" in state:
            value = state["bin_score"]
            if isinstance(value, torch.Tensor):
                return float(value.item())
            return float(value)
    except Exception as exc:
        print(f"Warning: failed to load bin_score from {weight_path}: {exc}")
    return default


if __name__ == "__main__":
    args = parse_args()
    logger = get_root_logger()

    work_dirs = Path("work_dirs") / "superpoint_superglue"
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a

    xhquant_init(None, debug=args.debug)

    sp_onnx_path = Path(__file__).parent / args.superpoint_onnx
    sg_onnx_path = Path(__file__).parent / args.superglue_onnx

    sp_onnx_name = args.superpoint_onnx.split("/")[-1][:-5]
    sg_onnx_name = args.superglue_onnx.split("/")[-1][:-5]
    quant_type = args.quant_type

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    sp_out_file = work_dirs / "hmonnx" / f"{sp_onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}.onnx"
    sg_out_file = work_dirs / "hmonnx" / f"{sg_onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}.onnx"
    sp_out_file.parent.mkdir(exist_ok=True, parents=True)
    sg_out_file.parent.mkdir(exist_ok=True, parents=True)
    sp_out_file = str(sp_out_file)
    sg_out_file = str(sg_out_file)

    sp_session = None
    sg_session = None

    if not os.path.exists(sp_out_file):
        logger.info(f"Converting SuperPoint ONNX to hmonnx: {sp_out_file}")
        convert_onnx_to_hmonnx(
            str(sp_onnx_path),
            [torch.randn(1, 1, args.input_shape[0], args.input_shape[1], dtype=torch.float16)],
            DeviceType.XH2a,
            sp_out_file,
            quant_config=quant_config,
            input_names=["image"],
            output_names=["scores", "descriptors_dense"],
        )

    if not os.path.exists(sg_out_file):
        logger.info(f"Converting SuperGlue ONNX to hmonnx: {sg_out_file}")
        convert_onnx_to_hmonnx(
            str(sg_onnx_path),
            [
             torch.randn(1, args.max_keypoints, 2, dtype=torch.float16),
             torch.randn(1, args.max_keypoints, dtype=torch.float16),
             torch.randn(1, 256, args.max_keypoints, dtype=torch.float16),
             torch.randn(1, args.max_keypoints, 2, dtype=torch.float16),
             torch.randn(1, args.max_keypoints, dtype=torch.float16),
             torch.randn(1, 256, args.max_keypoints, dtype=torch.float16)
            ],
            DeviceType.XH2a,
            sg_out_file,
            quant_config=quant_config,
            input_names=["keypoints0", "scores0", "descriptors0",
                         "keypoints1", "scores1", "descriptors1"],
            output_names=["score"],
        )

    sp_session = HMONNXInference(sp_out_file)
    sg_session = HMONNXInference(sg_out_file)

    if args.save_golden:
        sp_onnx_model = SuperPointONNX(
            str(sp_onnx_path),
            device=args.device,
            input_shape=args.input_shape,
            nms_radius=args.nms_radius,
            detection_threshold=args.keypoint_threshold,
            remove_borders=args.remove_borders,
            max_num_keypoints=args.max_keypoints,
        )
        logger.info("Saving SuperPoint and SuperGlue golden...")

        sp_session.save_golden = True
        sp_session.save_golden_dir = work_dirs / "hmonnx" / f"{sp_onnx_name}_golden"

        sg_session.save_golden = True
        sg_session.save_golden_dir = work_dirs / "hmonnx" / f"{sg_onnx_name}_golden"

        bin_score = args.bin_score
        if bin_score is None:
            bin_score = load_superglue_bin_score(args.superglue_weight, default=1.0)

        sg_onnx_model = SuperGlueONNX(
            str(sg_onnx_path),
            device=args.device,
            max_keypoints=args.max_keypoints,
            sinkhorn_iterations=args.sinkhorn_iterations,
            match_threshold=args.match_threshold,
            bin_score=bin_score,
        )

        with torch.no_grad():
            img1_raw, inp1, scales1 = read_image(args.img1_path, args.device, (args.input_shape[1], args.input_shape[0]))
            img2_raw, inp2, scales2 = read_image(args.img2_path, args.device, (args.input_shape[1], args.input_shape[0]))

            nn_out1 = sp_session.forward(inp1.half())
            nn_out2 = sp_session.forward(inp2.half())

            if isinstance(nn_out1, (tuple, list)):
                nn_out1 = [o.float() for o in nn_out1]
                nn_out2 = [o.float() for o in nn_out2]
            else:
                nn_out1 = nn_out1.float()
                nn_out2 = nn_out2.float()

            post_out1 = sp_onnx_model.post_process(nn_out1, device=args.device)
            post_out2 = sp_onnx_model.post_process(nn_out2, device=args.device)

            print(f"Image1 keypoints: {len(post_out1['keypoints'][0])}")
            print(f"Image2 keypoints: {len(post_out2['keypoints'][0])}")

            match_data = sg_onnx_model.match(
                post_out1["keypoints"][0], post_out1["scores"][0], post_out1["descriptors"][0], args.input_shape,
                post_out2["keypoints"][0], post_out2["scores"][0], post_out2["descriptors"][0], args.input_shape,
            )

            nn_out_sg = sg_session.forward(
                match_data["keypoints0_n"].half(),
                match_data["scores0"].half(),
                match_data["descriptors0"].half(),
                match_data["keypoints1_n"].half(),
                match_data["scores1"].half(),
                match_data["descriptors1"].half(),
            )

            print("Golden saved for SuperPoint and SuperGlue")

    torch.set_grad_enabled(False)
    if not args.no_eval:
        sp_onnx_model = SuperPointONNX(
            str(sp_onnx_path),
            device=args.device,
            input_shape=args.input_shape,
            nms_radius=args.nms_radius,
            detection_threshold=args.keypoint_threshold,
            remove_borders=args.remove_borders,
            max_num_keypoints=args.max_keypoints,
        )

        sp_session.exec_device = args.device
        sp_session.to(args.device)
        sg_session.exec_device = args.device
        sg_session.to(args.device)

        img1_color, inp1, scales1 = read_image(args.img1_path, args.device, (args.input_shape[1], args.input_shape[0]))
        img2_color, inp2, scales2 = read_image(args.img2_path, args.device, (args.input_shape[1], args.input_shape[0]))

        nn_out1 = sp_session.forward(inp1.half())
        nn_out2 = sp_session.forward(inp2.half())

        if isinstance(nn_out1, (tuple, list)):
            nn_out1 = [o.float() for o in nn_out1]
            nn_out2 = [o.float() for o in nn_out2]
        else:
            nn_out1 = nn_out1.float()
            nn_out2 = nn_out2.float()

        post_out1 = sp_onnx_model.post_process(nn_out1, device=args.device)
        post_out2 = sp_onnx_model.post_process(nn_out2, device=args.device)

        print(f"Image1 keypoints: {len(post_out1['keypoints'][0])}")
        print(f"Image2 keypoints: {len(post_out2['keypoints'][0])}")

        bin_score = args.bin_score
        if bin_score is None:
            bin_score = load_superglue_bin_score(args.superglue_weight, default=1.0)

        sg_onnx_model = SuperGlueONNX(
            str(sg_onnx_path),
            device=args.device,
            max_keypoints=args.max_keypoints,
            sinkhorn_iterations=args.sinkhorn_iterations,
            match_threshold=args.match_threshold,
            bin_score=bin_score,
        )

        match_data = sg_onnx_model.match(
            post_out1["keypoints"][0], post_out1["scores"][0], post_out1["descriptors"][0], args.input_shape,
            post_out2["keypoints"][0], post_out2["scores"][0], post_out2["descriptors"][0], args.input_shape,
        )

        nn_out_sg = sg_session.forward(
            match_data["keypoints0_n"].half(),
            match_data["scores0"].half(),
            match_data["descriptors0"].half(),
            match_data["keypoints1_n"].half(),
            match_data["scores1"].half(),
            match_data["descriptors1"].half(),
        )

        matches0, matches1, mscores0, mscores1, valid0, valid1 = sg_onnx_model.post_process(nn_out_sg)

        img1_color = cv2.imread(args.img1_path)
        img1_color = cv2.resize(img1_color, (args.input_shape[1], args.input_shape[0]))
        img2_color = cv2.imread(args.img2_path)
        img2_color = cv2.resize(img2_color, (args.input_shape[1], args.input_shape[0]))

        kp1 = post_out1["keypoints"][0].cpu().numpy()
        kp2 = post_out2["keypoints"][0].cpu().numpy()
        n0 = min(len(kp1), args.max_keypoints)
        n1 = min(len(kp2), args.max_keypoints)

        matches_draw = []
        for i in range(n0):
            if valid0[i] and matches0[i] < n1:
                matches_draw.append(cv2.DMatch(i, int(matches0[i]), float(mscores0[i])))

        matched_img = cv2.drawMatches(
            img1_color,
            [cv2.KeyPoint(float(kp1[i, 0]), float(kp1[i, 1]), 1) for i in range(n0)],
            img2_color,
            [cv2.KeyPoint(float(kp2[i, 0]), float(kp2[i, 1]), 1) for i in range(n1)],
            matches_draw,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
        )

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "superpoint_superglue_matches.png"), matched_img)
        print(f"Saved results to {output_dir / 'superpoint_superglue_matches.png'}")
        print(f"Total matches: {len(matches_draw)}")