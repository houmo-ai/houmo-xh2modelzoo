# Copyright 2025 HOUMO AI
#
# File: onnx_run_modify_onnx_no_post_process.py
# Description:
#   Example script: cv/rtdetr/scripts/onnx_run_modify_onnx_no_post_process.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import cv2
import numpy as np
import onnxruntime as rt
from gguf import Path
from loguru import logger


def main(args):
    onnx_name = Path(args.onnx).stem
    work_dir = Path("./work_dirs") / onnx_name
    work_dir.mkdir(parents=True, exist_ok=True)
    sess = rt.InferenceSession(args.onnx)
    img = cv2.imread(args.image)
    logger.info(f"image shape: {img.shape}")
    org_img = img
    h, w = img.shape[:2]
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    input_img = np.transpose(img, [2, 0, 1])
    image = input_img[np.newaxis, :, :, :]

    results = sess.run(["scores", "boxes"], {"image": image})

    logger.info(f"scores: {tuple(results[0].shape)}")
    logger.info(f"bboxes: {tuple(results[1].shape)}")

    scores, boxes = [o[0] for o in results]

    index = scores.max(-1)
    boxes, scores = boxes[index > 0.5], scores[index > 0.5]
    labels = scores.argmax(-1)
    scores = scores.max(-1)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        cv2.rectangle(org_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            org_img, f"{int(label)}: {score:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    out_file = str(work_dir / "result.png")
    cv2.imwrite(out_file, org_img)
    logger.info(f"result image saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx run")
    parser.add_argument(
        "--onnx", type=str, default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-no-post_process.onnx", help="onnx file"
    )
    parser.add_argument("--image", type=str, default="data/images/dog.jpg", help="image file")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold")
    args = parser.parse_args()
    print(args)
    main(args)
