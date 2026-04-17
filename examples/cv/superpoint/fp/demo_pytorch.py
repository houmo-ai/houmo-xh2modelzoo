import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from fp_superpoint import SuperPoint


def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)
    return kp, desc


def select_k_best(points, k):
    sorted_prob = points[points[:, 2].argsort(), :2]
    start = min(k, points.shape[0])
    return sorted_prob[-start:, :]


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


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img)[None, None]

    return img, img_orig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('--weights_name', type=str, default="/data01/home/xuchen/xh2/xh2_model_zoo/examples/cv/superpoint/superpoint_v6_from_tf.pth")
    parser.add_argument('--img1_path', type=str, default="/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/1.ppm")
    parser.add_argument('--img2_path', type=str, default="/data02/datasets/hpatches/raw/hpatches_seq/hpatches/hpatches-sequences-release/i_pool/2.ppm")
    parser.add_argument('--H', type=int, default=480,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()

    weights_name = args.weights_name
    img1_file = args.img1_path
    img2_file = args.img2_path
    img_size = (args.W, args.H)
    keep_k_best = args.k_best

    weights_path = Path(__file__).parent / weights_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SuperPoint()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device).float()
    model.eval()

    with torch.no_grad():
        img1, img1_orig = preprocess_image(img1_file, img_size)
        img1 = img1.to(device)
        pred1 = model({"image": img1})
        kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
            pred1["keypoints"][0],
            pred1["keypoint_scores"][0],
            pred1["descriptors"][0],
            keep_k_best
        )

        img2, img2_orig = preprocess_image(img2_file, img_size)
        img2 = img2.to(device)
        pred2 = model({"image": img2})
        kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
            pred2["keypoints"][0],
            pred2["keypoint_scores"][0],
            pred2["descriptors"][0],
            keep_k_best
        )

    m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
    H, inliers = compute_homography(m_kp1, m_kp2)

    matches = np.array(matches)[inliers.astype(bool)].tolist()
    matched_img = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, matches,
                                  None, matchColor=(0, 255, 0),
                                  singlePointColor=(0, 0, 255))

    sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(img1_orig)
    sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(img2_orig)
    sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
        sift_kp1, sift_desc1, sift_kp2, sift_desc2)
    sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)

    if sift_H is not None and sift_inliers is not None:
        sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
    else:
        sift_matches = []
    sift_matched_img = cv2.drawMatches(img1_orig, sift_kp1, img2_orig,
                                       sift_kp2, sift_matches, None,
                                       matchColor=(0, 255, 0),
                                       singlePointColor=(0, 0, 255))

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "superpoint_matches.png"), matched_img)
    cv2.imwrite(str(output_dir / "sift_matches.png"), sift_matched_img)
    print(f"Saved results to {output_dir}")