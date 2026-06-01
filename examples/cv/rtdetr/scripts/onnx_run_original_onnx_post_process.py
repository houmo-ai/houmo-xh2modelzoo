# Copyright 2025 HOUMO AI
#
# File: onnx_run_original_onnx_post_process.py
# Description:
#   Example script: cv/rtdetr/scripts/onnx_run_original_onnx_post_process.py
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
    im_shape = np.array([[float(img.shape[0]), float(img.shape[1])]]).astype("float32")
    img = cv2.resize(img, (640, 640))
    scale_factor = np.array([[float(640 / img.shape[0]), float(640 / img.shape[1])]]).astype("float32")
    img = img.astype(np.float32) / 255.0
    input_img = np.transpose(img, [2, 0, 1])
    image = input_img[np.newaxis, :, :, :]
    logger.info(f"im_shape: {im_shape.shape}")
    logger.info(f"image: {image.shape}")
    logger.info(f"scale_factor: {scale_factor.shape}")
    result = sess.run(None, {"im_shape": im_shape, "image": image, "scale_factor": scale_factor})
    logger.info(f"output shape: {np.array(result[0].shape)}")
    for value in result[0]:
        if value[1] > args.threshold:
            x1, y1, x2, y2 = value[2:]
            cv2.rectangle(org_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(
                org_img,
                f"{int(value[0])}: {value[1]:.04f}",
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
    out_file = str(work_dir / "result.png")
    cv2.imwrite(out_file, org_img)
    logger.info(f"result image saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx run")
    parser.add_argument(
        "--onnx", type=str, default="data/models/rtdetr/rtdetr_hgnetv2_l_6x_coco-sim.onnx", help="onnx file"
    )
    parser.add_argument("--image", type=str, default="data/images/dog.jpg", help="image file")
    parser.add_argument("--threshold", type=float, default=0.5, help="threshold")
    args = parser.parse_args()
    print(args)
    main(args)
