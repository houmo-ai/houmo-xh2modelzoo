import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# --- Mock HMONNXInference for standalone execution if xhquant is not available ---
# (如果您的环境中有xhquant，请确保它可以被导入)
try:
    from xhquant.api import HMONNXInference
except ImportError:
    print("[警告] xhquant.api 未找到。将使用一个模拟的 HMONNXInference 类。")

    class HMONNXInference:
        def __init__(self, model_path, providers=None):
            self.session = onnxruntime.InferenceSession(
                model_path, providers=providers if providers else ["CPUExecutionProvider"]
            )
            self._input_names = [i.name for i in self.session.get_inputs()]
            self._output_names = [o.name for o in self.session.get_outputs()]

        def get_input_names(self):
            return self._input_names

        def get_output_names(self):
            return self._output_names

        def run(self, input_dict):
            input_feed = {k: v.cpu().numpy() for k, v in input_dict.items()}
            outputs = self.session.run(self._output_names, input_feed)
            return [torch.from_numpy(o) for o in outputs]

        def to(self, device):
            print(f"[模拟] 模型被移动到 {device}")
            # 在模拟类中，这是一个空操作，但返回self以保持链式调用
            return self


# --- Core Processing Functions (adapted from your scripts) ---


def preprocess_with_padding(image_bgr: np.ndarray, target_height: int, target_width: int) -> dict:
    h, w, _ = image_bgr.shape
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
    top_pad = (target_height - new_h) // 2
    left_pad = (target_width - new_w) // 2
    padded_image[top_pad : top_pad + new_h, left_pad : left_pad + new_w] = image_resized

    img_tensor_np = padded_image.astype(np.float32) / 256.0 - 0.5
    img_tensor_np = img_tensor_np.transpose(2, 0, 1)
    img_tensor_np = np.ascontiguousarray(img_tensor_np)[np.newaxis, ...]

    return {
        "inputs": img_tensor_np,
        "metas": {
            "ori_shape": (h, w),
            "new_shape": (new_h, new_w),
            "pad_info": (top_pad, left_pad),
            "target_shape": (target_height, target_width),
        },
    }


def get_result(heatmap_avg, paf_avg, ori_h, ori_w, thre1=0.1, thre2=0.05):
    all_peaks, peak_counter = [], 0
    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)
        if np.max(one_heatmap) < thre1:
            all_peaks.append([])
            continue
        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]
        peaks_binary = np.logical_and.reduce(
            (
                one_heatmap >= map_left,
                one_heatmap >= map_right,
                one_heatmap >= map_up,
                one_heatmap >= map_down,
                one_heatmap > thre1,
            )
        )
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]
    mapIdx = [
        [31, 32],
        [39, 40],
        [33, 34],
        [35, 36],
        [41, 42],
        [43, 44],
        [19, 20],
        [21, 22],
        [23, 24],
        [25, 26],
        [27, 28],
        [29, 30],
        [47, 48],
        [49, 50],
        [53, 54],
        [51, 52],
        [55, 56],
        [37, 38],
        [45, 46],
    ]
    connection_all, special_k, mid_num = [], [], 10
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA, candB = all_peaks[limbSeq[k][0] - 1], all_peaks[limbSeq[k][1] - 1]
        nA, nB = len(candA), len(candB)
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)
                    startend = list(
                        zip(
                            np.linspace(candA[i][0], candB[j][0], num=mid_num),
                            np.linspace(candA[i][1], candB[j][1], num=mid_num),
                        )
                    )
                    vec_x = np.array(
                        [
                            score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                            for I in range(len(startend))
                        ]
                    )
                    vec_y = np.array(
                        [
                            score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                            for I in range(len(startend))
                        ]
                    )
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * ori_h / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]]
                        )
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(nA, nB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs, partBs = connection_all[k][:, 0], connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1
            for i in range(len(connection_all[k])):
                found, subset_idx = 0, [-1, -1]
                for j in range(len(subset)):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found], found = j, found + 1
                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB], subset[j][-1], subset[j][-2] = (
                            partBs[i],
                            subset[j][-1] + 1,
                            subset[j][-2] + candidate[partBs[i].astype(int), 2] + connection_all[k][i][2],
                        )
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][indexB], subset[j1][-1], subset[j1][-2] = (
                            partBs[i],
                            subset[j1][-1] + 1,
                            subset[j1][-2] + candidate[partBs[i].astype(int), 2] + connection_all[k][i][2],
                        )
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA], row[indexB], row[-1], row[-2] = (
                        partAs[i],
                        partBs[i],
                        2,
                        sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2],
                    )
                    subset = np.vstack([subset, row])
    deleteIdx = [i for i in range(len(subset)) if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4]
    subset = np.delete(subset, deleteIdx, axis=0)
    return candidate, subset


def post_process(paf_output, heatmap_output, metas):
    # 将输入统一转为numpy进行处理
    if isinstance(paf_output, torch.Tensor):
        paf_output = paf_output.cpu().numpy()
    if isinstance(heatmap_output, torch.Tensor):
        heatmap_output = heatmap_output.cpu().numpy()

    paf_output = paf_output.astype(np.float32)
    heatmap_output = heatmap_output.astype(np.float32)

    ori_h, ori_w = metas["ori_shape"]
    new_h, new_w = metas["new_shape"]
    top_pad, left_pad = metas["pad_info"]
    target_h, target_w = metas["target_shape"]

    paf = paf_output[0]
    heatmap = heatmap_output[0]

    heatmap = np.transpose(heatmap, (1, 2, 0))
    paf = np.transpose(paf, (1, 2, 0))

    heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    heatmap = heatmap[top_pad : top_pad + new_h, left_pad : left_pad + new_w, :]
    paf = paf[top_pad : top_pad + new_h, left_pad : left_pad + new_w, :]

    heatmap = cv2.resize(heatmap, (ori_w, ori_h), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (ori_w, ori_h), interpolation=cv2.INTER_CUBIC)

    return get_result(heatmap, paf, ori_h, ori_w)


# --- COCO Evaluation Specific Functions ---

# OpenPose 18个部位到 COCO 17个关键点的映射
# COCO keypoints:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear, 5: left_shoulder, 6: right_shoulder,
# 7: left_elbow, 8: right_elbow, 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
#
# OpenPose parts:
# 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist,
# 8: RHip, 9: RKnee, 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle, 14: REye, 15: LEye, 16: REar, 17: LEar
OPENPOSE_TO_COCO_MAP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


def format_for_coco(image_id, candidate, subset):
    """
    Converts OpenPose output format to COCO keypoint detection format.
    """
    coco_results = []
    for i in range(len(subset)):
        person = subset[i]
        keypoints = np.zeros(17 * 3, dtype=np.float32)

        # 遍历 OpenPose 的18个身体部位
        for part_idx in range(18):
            coco_idx = OPENPOSE_TO_COCO_MAP[part_idx]
            if coco_idx == -1:  # Neck is not in COCO
                continue

            candidate_idx = int(person[part_idx])
            if candidate_idx != -1:
                x, y, score = candidate[candidate_idx][:3]
                keypoints[coco_idx * 3 + 0] = x
                keypoints[coco_idx * 3 + 1] = y
                keypoints[coco_idx * 3 + 2] = 2  # COCO 'v' visibility flag (2=visible)

        coco_results.append(
            {
                "image_id": image_id,
                "category_id": 1,  # 'person' category in COCO
                "keypoints": keypoints.tolist(),
                "score": person[-2],  # Use the overall score for the person
            }
        )
    return coco_results


# --- Inference Engine Wrapper ---


class PoseEstimator:
    def __init__(self, model_path, model_type="onnx", device="cpu"):
        self.model_type = model_type.lower()
        self.device = torch.device(device)
        self.session = None
        self.input_name = None
        self.paf_out_name = None
        self.heatmap_out_name = None
        self.model_h = 184  # 预设或从模型获取
        self.model_w = 128  # 预设或从模型获取

        if self.model_type == "onnx":
            self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            output_names = [o.name for o in self.session.get_outputs()]
            self.paf_out_name = next((name for name in output_names if "L1" in name or "paf" in name), output_names[0])
            self.heatmap_out_name = next(
                (name for name in output_names if "L2" in name or "heat" in name), output_names[1]
            )
            input_shape = self.session.get_inputs()[0].shape
            self.model_h, self.model_w = input_shape[2], input_shape[3]

        elif self.model_type == "hmonnx":
            self.session = HMONNXInference(model_path)
            self.session.to(self.device)
            self.input_name = self.session.get_input_names()[0]
            output_names = self.session.get_output_names()
            self.paf_out_name = next((name for name in output_names if "L1" in name or "paf" in name), output_names[0])
            self.heatmap_out_name = next(
                (name for name in output_names if "L2" in name or "heat" in name), output_names[1]
            )
            # HMONNX 模型尺寸通常是固定的，这里硬编码
            self.model_h, self.model_w = 184, 128

        else:
            raise ValueError(f"不支持的模型类型: {model_type}. 请选择 'onnx' 或 'hmonnx'.")

        print(f"--- 模型加载成功 ---")
        print(f"   - 类型: {self.model_type.upper()}")
        print(f"   - 路径: {model_path}")
        print(f"   - 输入尺寸 (H, W): ({self.model_h}, {self.model_w})")
        print(f"   - 输入节点: {self.input_name}")
        print(f"   - PAF输出: {self.paf_out_name}, Heatmap输出: {self.heatmap_out_name}")

    def __call__(self, image_bgr):
        preprocessed = preprocess_with_padding(image_bgr, self.model_h, self.model_w)
        input_data = preprocessed["inputs"]
        metas = preprocessed["metas"]

        if self.model_type == "onnx":
            outputs = self.session.run([self.paf_out_name, self.heatmap_out_name], {self.input_name: input_data})
            paf_output, heatmap_output = outputs[0], outputs[1]

        elif self.model_type == "hmonnx":
            input_tensor = torch.from_numpy(input_data).to(self.device).to(torch.float16)
            outputs = self.session.run({self.input_name: input_tensor})

            output_names = self.session.get_output_names()
            paf_idx = output_names.index(self.paf_out_name)
            heatmap_idx = output_names.index(self.heatmap_out_name)
            paf_output, heatmap_output = outputs[paf_idx], outputs[heatmap_idx]

        candidate, subset = post_process(paf_output, heatmap_output, metas)
        return candidate, subset


# --- Main Evaluation Function ---


def main(args):
    # 1. 加载配置和数据集
    with open(args.data_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    data_root = Path(data_config["path"])
    val_images_path = data_root / data_config["val"]
    val_annot_path = data_root / data_config["annotations"]

    print(f"--- 正在加载COCO数据集 ---")
    print(f"  - 标注文件: {val_annot_path}")
    coco_gt = COCO(str(val_annot_path))
    image_ids = sorted(coco_gt.getImgIds())

    if args.limit > 0:
        image_ids = image_ids[: args.limit]
        print(f"  - [注意] 已将评估样本数量限制为: {args.limit}")

    # 2. 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = PoseEstimator(args.model_path, args.model_type, device)

    # 3. 循环推理并收集结果
    all_predictions = []
    print(f"\n--- 开始在 {len(image_ids)} 张图片上进行评估 ---")
    for img_id in tqdm(image_ids, desc=f"评估中 ({args.model_type.upper()})"):
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = val_images_path / img_info["file_name"]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"警告: 无法读取图片 {image_path}, 跳过。")
            continue

        candidate, subset = estimator(image_bgr)

        if candidate.shape[0] > 0 and subset.shape[0] > 0:
            coco_formatted_results = format_for_coco(img_id, candidate, subset)
            all_predictions.extend(coco_formatted_results)

    # 4. 保存结果并运行评估
    if not all_predictions:
        print("\n--- [错误] ---")
        print("模型未在任何图片上检测到关键点，无法进行评估。")
        return

    results_dir = Path("eval_results")
    results_dir.mkdir(exist_ok=True)
    model_name = Path(args.model_path).stem
    results_file = results_dir / f"{model_name}_{args.model_type}_coco_results.json"

    print(f"\n--- 推理完成 ---")
    print(f"  - 总共生成了 {len(all_predictions)} 条预测结果。")
    print(f"  - 正在将结果保存到: {results_file}")

    with open(results_file, "w") as f:
        json.dump(all_predictions, f, indent=4)

    print("\n--- 正在使用 pycocotools 进行精度评估 ---")
    coco_dt = coco_gt.loadRes(str(results_file))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在COCO数据集上评估ONNX或HMONNX OpenPose模型")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["onnx", "hmonnx"],
        help="要评估的模型类型: 'onnx' 或 'hmonnx'。",
    )
    parser.add_argument("--model-path", type=str, required=True, help="ONNX 或 HMONNX 模型的路径。")
    parser.add_argument(
        "--data-yaml",
        type=str,
        default="examples/cv/openpose/coco_eval.yaml",
        help="指向 coco_eval.yaml 数据配置文件的路径。",
    )
    parser.add_argument("--limit", type=int, default=0, help="限制评估的图片数量，用于快速测试 (0表示使用全部验证集)。")
    args = parser.parse_args()

    main(args)
