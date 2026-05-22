# Copyright 2025 HOUMO AI
#
# File: glm_ocr_fp_demo.py
# Description:
#   GLM-OCR float-point demo: load HF model and run inference on a single image.
#   Example script: examples/llm/glm_ocr/glm_ocr_fp_demo.py
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
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Allow importing common.py from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_inputs, build_messages, load_quarot_gptq_state_dict, render_pdf_to_images, resolve_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/GLM-OCR/")
    parser.add_argument("--image", type=str, default="examples/llm/glm_ocr/data/img3.png")
    parser.add_argument("--pdf", type=str, default=None,
                        help="PDF input path; pages are rendered to images before HF OCR")
    parser.add_argument("--pdf_output_dir", type=str, default="work_dirs/glm_ocr_pdf_pages",
                        help="directory used to store rendered PDF page images")
    parser.add_argument("--pdf_dpi", type=int, default=200,
                        help="DPI used to render PDF pages")
    parser.add_argument("--pdf_pages", type=str, default=None,
                        help="1-based PDF page selection, for example '1', '1,3', or '1-3,5'")
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--quarot_gptq_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = resolve_torch_dtype(args.dtype)
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    if args.quarot_gptq_path is not None:
        load_quarot_gptq_state_dict(model, args.quarot_gptq_path, strict=False)

    model = model.to(device)

    def run_single(image_path: str) -> str:
        messages = build_messages(image_path, args.prompt)
        inputs = build_inputs(processor, messages, device=device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        return processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    if args.pdf is not None:
        pdf_path = Path(args.pdf).expanduser().resolve()
        pdf_output_dir = Path(args.pdf_output_dir).expanduser().resolve() / pdf_path.stem
        page_images = render_pdf_to_images(
            pdf_path=pdf_path,
            output_dir=pdf_output_dir,
            dpi=args.pdf_dpi,
            pages=args.pdf_pages,
        )
        output_path = args.output_path or str(Path("work_dirs/glm_ocr_fp_demo") / f"{pdf_path.stem}_fp_ocr.txt")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f_txt:
            for i, page_image in enumerate(page_images, 1):
                print(f"Processing PDF page {i}/{len(page_images)}: {page_image}", flush=True)
                output_text = run_single(str(page_image))
                f_txt.write(f"===== Page {i} ({page_image.name}) =====\n")
                f_txt.write(output_text + "\n\n")
                f_txt.flush()
                print(output_text, flush=True)
        return

    output_text = run_single(args.image)
    if args.output_path is not None:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(output_text + "\n")
    print(output_text)


if __name__ == "__main__":
    main()
