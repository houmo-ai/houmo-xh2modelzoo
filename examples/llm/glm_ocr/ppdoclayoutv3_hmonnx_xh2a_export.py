# Copyright 2025 HOUMO AI
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

"""Export and quantize PP-DocLayoutV3 to HMONNX.

This script converts the local Transformers safetensors PP-DocLayoutV3 model
used by the GLM-OCR full SDK pipeline into an XH2a HMONNX model.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import onnx
import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from PIL import Image
from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor
from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from xh_model_zoo.utils.memory_tracker import MemoryTracker  # noqa: E402
from xh_model_zoo.utils.time_profiler import TimeProfiler  # noqa: E402


class PPDocLayoutV3ExportWrapper(nn.Module):
    """Return tensor-only outputs needed by PPDocLayoutV3 post-processing."""

    def __init__(self, model: PPDocLayoutV3ForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.logits, outputs.pred_boxes, outputs.order_logits, outputs.out_masks


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", default="/data01/datasets/ppdoclayoutv3_safetensors", help="PP-DocLayoutV3 safetensors directory")
    parser.add_argument("--work_dir", default="work_dirs/ppdoclayoutv3_xh2a_export_hmonnx", help="output work directory")
    parser.add_argument("--image", default="examples/llm/glm_ocr/data/18UF.pdf", help="calibration image or PDF path")
    parser.add_argument("--pdf_page", type=int, default=1, help="1-based PDF page used for calibration when --image is a PDF")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--quant_type", default="w16a16_sefp", help="xhquant QuantScheme quant_type")
    parser.add_argument("--device_type", default="XH2a", choices=["XH2a"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force_export", action="store_true", help="re-export intermediate ONNX even if it exists")
    parser.add_argument("--force_convert", action="store_true", help="re-convert HMONNX even if it exists")
    parser.add_argument("--skip_golden", action="store_true", help="skip HMONNX golden sanity run")
    return parser.parse_args()


def _load_calibration_image(path: str, pdf_page: int) -> Image.Image:
    source = Path(path)
    if source.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(source))
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {source}")
        page_index = max(0, min(int(pdf_page) - 1, len(doc) - 1))
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return Image.open(source).convert("RGB")


def _prepare_inputs(processor: PPDocLayoutV3ImageProcessor, image_path: str, pdf_page: int, batch_size: int) -> Dict[str, torch.Tensor]:
    image = _load_calibration_image(image_path, pdf_page)
    inputs = processor(images=[image], return_tensors="pt")
    if batch_size > 1:
        inputs["pixel_values"] = inputs["pixel_values"].repeat(batch_size, 1, 1, 1)
    return inputs


def _export_onnx(model: nn.Module, pixel_values: torch.Tensor, onnx_file: Path, opset: int, force: bool, logger) -> None:
    if onnx_file.exists() and not force:
        logger.info(f"ONNX already exists: {onnx_file}")
        return

    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    model.float().eval().cpu()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx = Path(tmp_dir) / "ppdoclayoutv3.onnx"
        logger.info(f"Exporting ONNX to {tmp_onnx}")
        torch.onnx.export(
            model,
            (pixel_values.float().cpu(),),
            str(tmp_onnx),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes", "order_logits", "out_masks"],
            dynamic_axes=None,
            verbose=False,
        )
        onnx_model = onnx.load(str(tmp_onnx), load_external_data=True)

    onnx.save_model(
        onnx_model,
        str(onnx_file),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{onnx_file.stem}_external_data",
        size_threshold=1024,
    )
    logger.info(f"Saved ONNX to {onnx_file}")


def _device_type(name: str):
    if name == "XH2a":
        return DeviceType.XH2a
    raise ValueError(f"Unsupported device_type: {name}")


def _convert_hmonnx(onnx_file: Path, pixel_values: torch.Tensor, hmonnx_file: Path, quant_type: str, device_type: str, force: bool, logger) -> None:
    if hmonnx_file.exists() and not force:
        logger.info(f"HMONNX already exists: {hmonnx_file}")
        return

    hmonnx_file.parent.mkdir(parents=True, exist_ok=True)
    target_device = _device_type(device_type)
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [pixel_values.float().cpu()],
        target_device,
        str(hmonnx_file),
        quant_config=quant_config,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes", "order_logits", "out_masks"],
    )
    logger.info(f"Convert ONNX to HMONNX success: {hmonnx_file}")


def _run_golden(hmonnx_file: Path, pixel_values: torch.Tensor, logger) -> None:
    from xhquant.api import HMONNXGoldenInference

    device = "cuda" if torch.cuda.is_available() else "cpu"
    golden_dir = hmonnx_file.parent / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(str(hmonnx_file))
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.step = 0
    outputs = session(pixel_values.half().to(device))
    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]
    logger.info(f"Golden generated at: {golden_dir}")
    logger.info("HMONNX output shapes: " + json.dumps([list(out.shape) for out in outputs if isinstance(out, torch.Tensor)]))


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    xhquant_init(str(work_dir / "ppdoclayoutv3_quant.log"), debug=args.debug)
    xhquant.utils.suppress_printing.disable_printing = True
    logger = get_root_logger()
    logger.info(f"Args: {args}")

    with TimeProfiler("ppdoclayoutv3 hmonnx export", logger), MemoryTracker(0, "ppdoclayoutv3 hmonnx export", logger):
        processor = PPDocLayoutV3ImageProcessor.from_pretrained(args.model_dir)
        inputs = _prepare_inputs(processor, args.image, args.pdf_page, args.batch_size)
        pixel_values = inputs["pixel_values"]
        logger.info(f"Calibration pixel_values shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype}")

        hf_model = PPDocLayoutV3ForObjectDetection.from_pretrained(args.model_dir).eval()
        wrapper = PPDocLayoutV3ExportWrapper(hf_model)

        onnx_file = work_dir / "onnx" / "ppdoclayoutv3.onnx"
        hmonnx_file = work_dir / "hmonnx" / f"ppdoclayoutv3_{args.quant_type}_{args.device_type}.onnx"

        _export_onnx(wrapper, pixel_values, onnx_file, args.opset, args.force_export, logger)
        _convert_hmonnx(onnx_file, pixel_values, hmonnx_file, args.quant_type, args.device_type, args.force_convert, logger)

        meta = {
            "model_dir": args.model_dir,
            "onnx_file": str(onnx_file),
            "hmonnx_file": str(hmonnx_file),
            "quant_type": args.quant_type,
            "device_type": args.device_type,
            "input_shape": list(pixel_values.shape),
            "output_names": ["logits", "pred_boxes", "order_logits", "out_masks"],
        }
        (work_dir / "export_meta_info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved meta info to {work_dir / 'export_meta_info.json'}")

        if not args.skip_golden:
            _run_golden(hmonnx_file, pixel_values, logger)


if __name__ == "__main__":
    main()
