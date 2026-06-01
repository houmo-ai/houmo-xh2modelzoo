# openpose_origin_onnx_test.py
import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

# scipy.ndimage.filters is deprecated, import from scipy.ndimage
from scipy.ndimage import gaussian_filter

DEBUG_MODE = False
DEBUG_DIR = Path("work_dirs/openpose_body/onnx/debug_output")

PART_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "Background",
]


def setup_debug_mode(is_debug):
    global DEBUG_MODE
    DEBUG_MODE = is_debug
    if DEBUG_MODE:
        DEBUG_DIR.mkdir(exist_ok=True, parents=True)
        print(f"--- [DEBUG] 调试模式已开启。中间图像将保存至: {DEBUG_DIR} ---")


def debug_save_image(image_data, filename, normalize=False, is_heatmap=False):
    if not DEBUG_MODE:
        return

    save_data = image_data.copy()

    if normalize:
        save_data = (save_data + 0.5) * 256.0
        save_data = np.clip(save_data, 0, 255).astype(np.uint8)

    if is_heatmap:
        if save_data.ndim == 3:
            save_data = np.amax(save_data, axis=2)

        min_val, max_val = np.min(save_data), np.max(save_data)
        # print(f"--- [DEBUG] Saving heatmap '{filename}'. Input data range: [{min_val:.4f}, {max_val:.4f}]")

        if max_val > 0:
            save_data = (save_data / max_val) * 255
        save_data = save_data.astype(np.uint8)
        save_data = cv2.applyColorMap(save_data, cv2.COLORMAP_JET)

    try:
        cv2.imwrite(str(DEBUG_DIR / filename), save_data)
        print(f"--- [DEBUG] 已保存图像: {filename}")
    except Exception as e:
        print(f"--- [DEBUG] [ERROR] 保存图像失败: {filename}, 错误: {e}")


def preprocess_with_padding(image_bgr: np.ndarray, target_height: int, target_width: int) -> dict:
    print("\n--- [ACTION] 正在进行预处理：保持长宽比缩放并填充 ---")

    h, w, _ = image_bgr.shape
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
    top_pad = (target_height - new_h) // 2
    left_pad = (target_width - new_w) // 2
    padded_image[top_pad : top_pad + new_h, left_pad : left_pad + new_w] = image_resized

    print(f"   - 原始尺寸 (H,W): ({h}, {w})")
    print(f"   - 缩放后尺寸 (H,W): ({new_h}, {new_w})")
    print(f"   - 填充后尺寸 (H,W): ({target_height}, {target_width})")
    print(f"   - 填充信息 (top, left): ({top_pad}, {left_pad})")
    debug_save_image(padded_image, "01_preprocessed_input_padded.jpg")

    img_tensor_np = padded_image.astype(np.float32) / 256.0 - 0.5
    print(f"   - 归一化后数据范围: [{np.min(img_tensor_np):.2f}, {np.max(img_tensor_np):.2f}]")
    debug_save_image(img_tensor_np, "02_preprocessed_normalized.jpg", normalize=True)

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
    print("\n--- [ACTION] 正在从Heatmap和PAF中解析骨骼 ---")
    print(f"   - 使用的关节点检测阈值 (thre1): {thre1}")

    if DEBUG_MODE:
        print("--- [DEBUG] 正在保存最终的、恢复尺寸后的各部分热力图...")
        for part in range(18):
            single_heatmap = heatmap_avg[:, :, part]
            debug_filename = f"05_final_heatmap_part_{part:02d}_{PART_NAMES[part]}.jpg"
            debug_save_image(single_heatmap, debug_filename, is_heatmap=True)

        print("--- [DEBUG] 正在单独保存背景通道的热力图...")
        background_heatmap = heatmap_avg[:, :, 18]
        debug_save_image(background_heatmap, "05_final_heatmap_part_18_Background.jpg", is_heatmap=True)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)

        max_val_orig = np.max(map_ori)
        max_val_filtered = np.max(one_heatmap)
        print(
            f"    - [Part {part:2d} - {PART_NAMES[part]:9s}] "
            f"原始热力图最大值: {max_val_orig:.6f} | "
            f"高斯滤波后最大值: {max_val_filtered:.6f}"
        )

        if max_val_filtered < thre1:
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

        if DEBUG_MODE and len(peaks) > 0:
            print(f"      -> SUCCESS: 为此部分找到 {len(peaks)} 个峰值 (得分 > {thre1})")

        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    print(f"   - 在18个热力图中总共找到 {peak_counter} 个候选关节点。")

    # (代码的其余部分保持不变)
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
    print(f"   - 成功找到 {sum(len(conn) for conn in connection_all)} 条有效的肢体连接。")
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
    print(f"   - 最终组装成 {len(subset)} 个有效的人体骨骼。")
    return candidate, subset


def post_process(paf_output, heatmap_output, metas, stride=8):
    print("\n--- [ACTION] 正在进行后处理：恢复尺寸和移除填充 ---")

    ori_h, ori_w = metas["ori_shape"]
    new_h, new_w = metas["new_shape"]
    top_pad, left_pad = metas["pad_info"]
    target_h, target_w = metas["target_shape"]

    paf = paf_output[0]
    heatmap = heatmap_output[0]
    print(f"   - 原始PAF输出形状: {paf.shape}, 数据范围: [{paf.min():.4f}, {paf.max():.4f}], 均值: {paf.mean():.4f}")
    print(
        f"   - 原始Heatmap输出形状: {heatmap.shape}, 数据范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}"
    )

    raw_heatmap_for_vis = np.transpose(heatmap, (1, 2, 0))
    raw_heatmap_resized_vis = cv2.resize(
        raw_heatmap_for_vis,
        (raw_heatmap_for_vis.shape[1] * 10, raw_heatmap_for_vis.shape[0] * 10),
        interpolation=cv2.INTER_NEAREST,
    )
    debug_save_image(raw_heatmap_resized_vis, "03_raw_model_output_heatmap_NEAREST.jpg", is_heatmap=True)

    heatmap = np.transpose(heatmap, (1, 2, 0))
    paf = np.transpose(paf, (1, 2, 0))

    heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    print(f"   - (步骤1) 上采样后Heatmap形状: {heatmap.shape}")
    print(
        f"     - [DEBUG] 上采样后Heatmap数据范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}"
    )

    heatmap = heatmap[top_pad : top_pad + new_h, left_pad : left_pad + new_w, :]
    paf = paf[top_pad : top_pad + new_h, left_pad : left_pad + new_w, :]
    print(f"   - (步骤2) 移除填充后Heatmap形状: {heatmap.shape}")
    print(
        f"     - [DEBUG] 移除填充后Heatmap数据范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}"
    )

    heatmap = cv2.resize(heatmap, (ori_w, ori_h), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (ori_w, ori_h), interpolation=cv2.INTER_CUBIC)
    print(f"   - (步骤3) 恢复至原始尺寸后Heatmap形状: {heatmap.shape}")
    print(f"     - [DEBUG] 最终Heatmap数据范围: [{heatmap.min():.4f}, {heatmap.max():.4f}], 均值: {heatmap.mean():.4f}")

    debug_save_image(heatmap, "04_postprocessed_heatmap_vis.jpg", is_heatmap=True)

    return get_result(heatmap, paf, ori_h, ori_w)


def draw_bodypose(image, candidate, subset):
    print("\n--- [ACTION] 正在绘制结果 ---")

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

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

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(image, (int(x), int(y)), 4, colors[i], thickness=-1)

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = image.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            image = cv2.addWeighted(image, 0.4, cur_canvas, 0.6, 0)

    print(f"   - 已将关节点和肢体绘制到图像上。")
    return image


def main(args):
    setup_debug_mode(args.debug)

    print("\n--- [ACTION] 正在加载ONNX模型 ---")
    session = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    paf_out_name = next((name for name in output_names if "L1" in name or "paf" in name), output_names[0])
    heatmap_out_name = next((name for name in output_names if "L2" in name or "heat" in name), output_names[1])

    print(f"   - 模型路径: {args.onnx}")
    print(f"   - 输入节点: {input_name}")
    print(f"   - 输出节点 (PAF): {paf_out_name}")
    print(f"   - 输出节点 (Heatmap): {heatmap_out_name}")

    print(f"\n--- [ACTION] 正在加载图像: {args.image} ---")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"[ERROR] 无法读取图像文件: {args.image}")
        return

    input_shape = session.get_inputs()[0].shape
    model_h, model_w = input_shape[2], input_shape[3]
    print(f"   - 模型期望的输入尺寸 (H, W): ({model_h}, {model_w})")

    preprocessed = preprocess_with_padding(image_bgr, model_h, model_w)
    input_tensor = preprocessed["inputs"]
    metas = preprocessed["metas"]
    print(f"   - 送入模型的张量形状: {input_tensor.shape}, 类型: {input_tensor.dtype}")

    print("\n--- [ACTION] 正在执行ONNX推理 ---")
    outputs = session.run([paf_out_name, heatmap_out_name], {input_name: input_tensor})
    paf_output, heatmap_output = outputs[0], outputs[1]
    print("   - 推理完成。")

    candidate, subset = post_process(paf_output, heatmap_output, metas)

    if candidate.shape[0] > 0 and subset.shape[0] > 0:
        output_image = draw_bodypose(image_bgr.copy(), candidate, subset)
        save_path = "work_dirs/openpose_body/onnx/output_result.jpg"
        cv2.imwrite(save_path, output_image)
        print(f"\n--- [SUCCESS] ---")
        print(f"结果已成功保存到: {save_path}")
    else:
        print("\n--- [INFO] ---")
        print("未检测到任何人。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用ONNX OpenPose模型进行人体姿态估计，并提供详细调试信息。")
    parser.add_argument(
        "--onnx", type=str, default="data/models/openpose/openpose_body.onnx", help="ONNX模型文件的路径。"
    )
    parser.add_argument("--image", type=str, default="examples/cv/openpose/demo.jpg", help="输入图像的路径。")
    parser.add_argument("--debug", action="store_true", help="激活调试模式，会保存处理过程中的中间图像。")
    args = parser.parse_args()

    main(args)
