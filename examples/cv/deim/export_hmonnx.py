# Copyright 2025 HOUMO AI
#
# File: export_hmonnx.py
# Description:
#   Example script: cv/deim/export_hmonnx.py
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
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from curses import meta
from pathlib import Path
from re import S
from typing import List

import onnx
import onnxsim
import torch
from more_itertools import map_reduce
from xhquant import PrecisionMode
from xhquant import nn as xhnn
from xhquant import to_frontend_graph, to_quant_graph
from xhquant.api import DeviceType, HMONNXInference, convert_onnx_to_hmonnx, get_root_logger, xhquant_init
from xhquant.utils import map_aggregate

# isort: skip_file
from pathlib import Path
from tkinter import Image

import _init_path  # isort:skip
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as T
from engine.core import YAMLConfig
from dataclasses import dataclass


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: int
    score: float


# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw_keep_ratio(images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline="red")
            draw.text((bb[0], bb[1]), text=str(lbl), fill="blue")

        result_images.append(im)
    return result_images


def draw(images, labels, boxes, scores, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]
        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()}[{round(scrs[j].item(), 2)}]",
                fill="blue",
            )

        # im.save("torch_results.jpg")
        result_images.append(im)
    return result_images


def preprocess(im_pil: Image):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)

    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]])
    input_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    # transforms = T.Compose(
    #     [
    #         T.Resize((640, 640)),
    #         T.ToTensor(),
    #     ]
    # )
    im_data = transforms(resized_im_pil).unsqueeze(0)

    # output = sess.run(output_names=None, input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()})

    # labels, boxes, scores = output

    # result_images = draw([im_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)])
    # result_images[0].save("onnx_result.jpg")
    # print("Image processing complete. Result saved as 'result.jpg'.")
    return {
        "inputs": [im_data],
        "meta": {
            "im_pils": [im_pil],  # Keep the original image for drawing
            "input_sizes": input_size,
            "orig_sizes": orig_size,
            "scale_factors": torch.tensor([[ratio, ratio]]),
            "paddings": torch.tensor([[pad_w, pad_h]]),
            # "ratio": ratio,
            # "pad_w": pad_w,
            # "pad_h": pad_h,
        },
    }


def postprocess(
    pred_logits,
    pred_boxes,
    postprocessor,
    meta_info,
    thrh=0.4,
) -> List[List[Box]]:
    output = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
    }
    labels, boxes, scores = postprocessor(
        output,
        meta_info["input_sizes"].to(pred_logits.device),
    )
    im_pils = meta_info["im_pils"]
    results = []
    for i, im in enumerate(im_pils):
        ratio_w, ratio_h = meta_info["scale_factors"][i]
        pad_w = meta_info["paddings"][i][0]
        pad_h = meta_info["paddings"][i][1]
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]
        boxes = []
        for j, b in enumerate(box):
            x1, y1, x2, y2 = b
            x1 = x1.item()
            y1 = y1.item()
            x2 = x2.item()
            y2 = y2.item()
            x1 = (x1 - pad_w) / ratio_w
            y1 = (y1 - pad_h) / ratio_h
            x2 = (x2 - pad_w) / ratio_w
            y2 = (y2 - pad_h) / ratio_h
            label = int(lab[j].item())
            score = scrs[j].item()
            x_c = (x1 + x2) / 2
            y_c = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            box = Box(x_c, y_c, w, h, label, score)
            boxes.append(box)
        results.append(boxes)

    return results


def draw_results(
    images: List[Image.Image],
    results: List[List[Box]],
):
    result_images: List[Image.Image] = []
    for i, im in enumerate(images):
        im_draw = deepcopy(im)
        draw = ImageDraw.Draw(im_draw)
        boxes = results[i]
        for box in boxes:
            x1 = box.x - box.w / 2
            y1 = box.y - box.h / 2
            x2 = box.x + box.w / 2
            y2 = box.y + box.h / 2
            label = box.label
            score = box.score
            draw.rectangle([x1, y1, x2, y2], outline="red")
            draw.text(
                (x1, y1),
                text=f"{label}[{round(score, 2)}]",
                fill="blue",
            )
        result_images.append(im_draw)
    return result_images


def main(args):
    onnx_name = "deim"
    target_device = DeviceType.XH2a
    cfg_name = f"{onnx_name}_{target_device}"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = Path(work_dir) / f"{cfg_name}.log"
    xhquant_init(log_file, debug=args.debug)

    out_hmonnx_file = work_dir / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    logger = get_root_logger()
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            # self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images):
            outputs = self.model(images)
            # outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs["pred_logits"], outputs["pred_boxes"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model()
    model.eval()
    model.to(device)

    postprocessor = cfg.postprocessor.deploy()

    input_path = args.input
    fname = Path(input_path).stem
    im_pil = Image.open(input_path).convert("RGB")
    data_sample = preprocess(im_pil)
    inputs = data_sample["inputs"]
    inputs = [x.to(device) for x in inputs]
    pred_logits, pred_boxes = model(*inputs)
    torch_results = postprocess(pred_logits, pred_boxes, postprocessor, data_sample["meta"])
    result_images = draw_results([im_pil], torch_results)
    result_images[0].save(Path(work_dir) / f"{fname}_torch.jpg")

    im_dummy_cpu = torch.rand(1, 3, 640, 640)
    im_dummy = im_dummy_cpu.to(device)
    # size = torch.tensor([[640, 640]])
    _ = model(im_dummy)

    model.cpu()
    out_onnx_file = str(work_dir / "onnx" / f"{onnx_name}.onnx")
    Path(out_onnx_file).parent.mkdir(parents=True, exist_ok=True)
    if not Path(out_onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_onnx_file = os.path.join(tmpdir, "tmp.onnx")
            torch.onnx.export(
                model,
                im_dummy_cpu,
                tmp_onnx_file,
                input_names=["images"],
                output_names=["logits", "boxes"],
                opset_version=16,
                verbose=False,
                do_constant_folding=True,
            )
            onnx_model, _ = onnxsim.simplify(tmp_onnx_file)
            onnx.save(
                onnx_model,
                out_onnx_file,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{Path(out_onnx_file).stem}_external_data",
            )

    if args.debug:
        from xhquant.api import (
            FrontendType,
            FXInterpreter,
            ptq_quantize,
            to_export_graph,
            to_export_hmonnx,
        )

        fronted_graph_module = to_frontend_graph(out_onnx_file, FrontendType.ONNX, [im_dummy_cpu])

        fronted_graph_module.to(device)
        pred_logits, pred_boxes = fronted_graph_module(*inputs)

        fronted_result = postprocess(pred_logits, pred_boxes, postprocessor, data_sample["meta"])

        result_images = draw_results([im_pil], fronted_result)
        result_images[0].save(Path(work_dir) / f"{fname}_fronted.jpg")
        del pred_logits
        del pred_boxes

        quant_graph_module = to_quant_graph(fronted_graph_module, target_device)
        ptq_quantize(quant_graph_module, [inputs], PrecisionMode.ALIGNED, [device], auto_release_unused_parameters=True)
        quant_graph_module.to(device)
        inputs = map_aggregate(inputs, lambda x: x.to(device).to(torch.float16))
        interpreter = FXInterpreter(quant_graph_module)
        pred_logits, pred_boxes = interpreter.run(*inputs)
        quanted_result = postprocess(pred_logits, pred_boxes, postprocessor, data_sample["meta"], 0.5)

        result_images = draw_results([im_pil], quanted_result)
        result_images[0].save(Path(work_dir) / f"{fname}_quanted.jpg")
        del pred_logits
        del pred_boxes

        exported_graph_module = to_export_graph(quant_graph_module, [im_dummy_cpu])
        inputs = map_aggregate(inputs, lambda x: x.to(device).to(torch.float16))
        pred_logits, pred_boxes = exported_graph_module(*inputs)
        exported_result = postprocess(pred_logits, pred_boxes, postprocessor, data_sample["meta"], 0.5)
        result_images = draw_results([im_pil], exported_result)
        result_images[0].save(Path(work_dir) / f"{fname}_exported.jpg")
        del pred_logits
        del pred_boxes

        to_export_hmonnx(exported_graph_module, out_hmonnx_file)

    else:
        convert_onnx_to_hmonnx(out_onnx_file, [im_dummy_cpu], target_device, out_hmonnx_file)

    meta_file = str(Path(work_dir) / "meta.json")
    meta_info = {
        "input_shape": im_dummy_cpu.shape,
        "model_name": "deim",
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hmonnx": str(Path(out_hmonnx_file).relative_to(Path(meta_file).parent)),
    }

    with open(meta_file, "w") as fp:
        json.dump(meta_info, fp, indent=4)

    session = HMONNXInference(out_hmonnx_file)
    session.to(device)
    inputs = map_aggregate(inputs, lambda x: x.to(device).to(torch.float16))
    pred_logits, pred_boxes = session(*inputs)
    hmonnx_result = postprocess(pred_logits, pred_boxes, postprocessor, data_sample["meta"], 0.5)
    for box in hmonnx_result[0]:
        print(box)
    result_images = draw_results([im_pil], hmonnx_result)
    result_images[0].save(Path(work_dir) / f"{fname}_hmonnx.jpg")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="examples/cv/deim/DEIM/configs/deim_dfine/deim_hgnetv2_m_coco.yml",
        type=str,
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        default="data/models/deim/deim_dfine_hgnetv2_m_coco_90e.pth",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/images/000000001490.jpg",
        help="Path to the input image or video file.",
    )
    args = parser.parse_args()
    main(args)
