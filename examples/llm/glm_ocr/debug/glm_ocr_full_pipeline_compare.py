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

"""Evaluate GLM-OCR FP/HMONNX under the official SDK layout pipeline.

The upstream GLM-OCR SDK pipeline is kept as a git submodule under
``examples/llm/glm_ocr/GLM-OCR``.  This script reuses its PageLoader,
PP-DocLayoutV3 detector, and ResultFormatter, but replaces the SDK remote
OCRClient with local recognizers:

* ``fp``: Hugging Face GLM-OCR ``AutoModelForImageTextToText``.
* ``hmonnx``: this repository's ``GlmOcrONNXModel``.

This makes the comparison cover the complete document flow:
PDF/image -> layout detection -> region crops -> OCR -> formatted Markdown/JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Local helpers from the GLM-OCR examples directory.
EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(REPO_ROOT))
from common import build_inputs, build_messages, resolve_torch_dtype  # noqa: E402

# Upstream SDK submodule.
SDK_ROOT = EXAMPLE_DIR / "GLM-OCR"
if not SDK_ROOT.exists():
    raise RuntimeError(
        f"Official GLM-OCR SDK submodule not found at {SDK_ROOT}. "
        "Run: git submodule update --init examples/llm/glm_ocr/GLM-OCR"
    )
sys.path.insert(0, str(SDK_ROOT))

from glmocr.config import PipelineConfig  # noqa: E402
from glmocr.pipeline import Pipeline  # noqa: E402


class FPLocalOCRClient:
    """OpenAI-response-compatible OCR client backed by HF GLM-OCR."""

    api_host = "local"
    api_port = 0

    def __init__(self, model_path: str, device: str, dtype: str, max_new_tokens: int, attn_implementation: str):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = resolve_torch_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.processor = None
        self.model = None

    def start(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
        ).eval().to(self.device)

    def stop(self):
        self.model = None
        self.processor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _extract_prompt(request_data: Dict[str, Any]) -> str:
        for message in request_data.get("messages", []):
            for item in message.get("content", []):
                if item.get("type") == "text":
                    return item.get("text", "")
        return "Text Recognition:"

    def process_image(self, image: Image.Image, task_type: str = "text", prompt: Optional[str] = None) -> str:  # noqa: ARG002
        if self.processor is None or self.model is None:
            raise RuntimeError("FPLocalOCRClient is not started")
        prompt = prompt or "Text Recognition:"
        messages = build_messages(image, prompt)
        inputs = build_inputs(self.processor, messages, device=self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    def process(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        # The SDK pipeline path calls process_image() directly via LocalLayoutPipeline below.
        return {"choices": [{"message": {"content": ""}}]}, 200


class HmonnxLocalOCRClient:
    """OpenAI-response-compatible OCR client backed by local HMONNX GLM-OCR."""

    api_host = "local"
    api_port = 0

    def __init__(
        self,
        model_path: str,
        vision_onnx: str,
        prefill_onnx: str,
        decode_onnx: str,
        token_embedding: str,
        hf_config_path: str,
        prefill_input_length: int,
        max_new_tokens: int,
    ):
        self.model_path = model_path
        self.vision_onnx = vision_onnx
        self.prefill_onnx = prefill_onnx
        self.decode_onnx = decode_onnx
        self.token_embedding = token_embedding
        self.hf_config_path = hf_config_path
        self.prefill_input_length = prefill_input_length
        self.max_new_tokens = max_new_tokens
        self.image_size_w = 672
        self.image_size_h = 672
        self.num_hidden_layers = 16
        self.num_kv_heads = 8
        self.head_dim = 128
        self.cache_len = 2048
        self.model = None

    def start(self):
        from xh_model_zoo.xh_llm.models.glm_ocr import GlmOcrProcessor
        from xh_model_zoo.xh_llm.models.glm_ocr.glm_ocr_onnx_model import GlmOcrONNXModel

        self.model = GlmOcrONNXModel(
            image_feature={"onnx": self.vision_onnx},
            prefill={"onnx": self.prefill_onnx, "input_sequence_length": self.prefill_input_length},
            decode={"onnx": self.decode_onnx},
            kv_cache={
                "num_hidden_layers": self.num_hidden_layers,
                "shape": [1, self.num_kv_heads, self.cache_len, self.head_dim],
            },
            cache_len=self.cache_len,
            image_size_w=self.image_size_w,
            image_size_h=self.image_size_h,
        )
        self.model.set_input_embeddings(torch.load(self.token_embedding, weights_only=False, map_location="cpu"))
        self.model.set_exec_device("cuda")
        self.model.to("cuda")
        self.processor = GlmOcrProcessor.from_pretrained(self.hf_config_path)
        self.model.init_image_feature()
        self.model.init_prefill()
        self.model.init_decode()
        self.model.to("cuda")
        self.model.set_exec_device("cuda")

    def stop(self):
        if self.model is not None:
            self.model.release_all_sessions()
        self.model = None
        self.processor = None

    def is_alive(self, timeout: float = 5.0) -> bool:  # noqa: ARG002
        return True

    def process_image(self, image: Image.Image, task_type: str = "text", prompt: Optional[str] = None) -> str:  # noqa: ARG002
        if self.model is None:
            raise RuntimeError("HmonnxLocalOCRClient is not started")
        prompt = prompt or "Text Recognition:"
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image = image.convert("RGB")
            target_w = int(getattr(self.model, "image_size_w", 1024))
            target_h = int(getattr(self.model, "image_size_h", 1024))
            if image.size != (target_w, target_h):
                scale = min(target_w / image.width, target_h / image.height)
                new_w = max(1, int(image.width * scale))
                new_h = max(1, int(image.height * scale))
                image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
                canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
                canvas.paste(image, (0, 0))
                image = canvas
            image.save(tmp.name)
            return self.model.chat(
                prompt,
                tmp.name,
                self.processor,
                max_new_tokens=self.max_new_tokens,
                keep_sessions=True,
            )

    def process(self, request_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        return {"choices": [{"message": {"content": ""}}]}, 200


class LocalLayoutPipeline(Pipeline):
    """Upstream layout pipeline with in-process OCR instead of HTTP OCRClient."""

    def __init__(self, config: PipelineConfig, ocr_client: Any):
        super().__init__(config=config)
        self.ocr_client = ocr_client

    def start(self):
        self.layout_detector.start()
        self.ocr_client.start()

    def stop(self):
        self.ocr_client.stop()
        self.layout_detector.stop()

    def process_local(self, source: str, save_layout_visualization: bool = True):
        pages = self.page_loader.load_pages(source)
        layout_results, layout_vis_images = self.layout_detector.process(
            pages,
            save_visualization=save_layout_visualization,
            use_polygon=self.config.layout.use_polygon,
        )
        grouped_results: List[List[Dict[str, Any]]] = []
        for page, page_layout in zip(pages, layout_results):
            page_items = []
            for region in page_layout:
                item = dict(region)
                if item.get("task_type") == "skip":
                    item["content"] = None
                    page_items.append(item)
                    continue
                from glmocr.utils.image_utils import crop_image_region

                polygon = item.get("polygon") if self.config.layout.use_polygon else None
                cropped = crop_image_region(page, item["bbox_2d"], polygon)
                item["content"] = self.ocr_client.process_image(
                    cropped,
                    task_type=item.get("task_type", "text"),
                )
                page_items.append(item)
            grouped_results.append(page_items)

        json_str, markdown_str, image_files = self.result_formatter.process(grouped_results)
        from glmocr.parser_result import PipelineResult

        return PipelineResult(
            json_result=json.loads(json_str),
            markdown_result=markdown_str,
            original_images=[source],
            image_files=image_files,
            layout_vis_images=layout_vis_images,
        )


def build_pipeline_config(args) -> PipelineConfig:
    config = PipelineConfig()
    config.max_workers = 1
    config.page_loader.pdf_dpi = args.pdf_dpi
    config.page_loader.pdf_max_pages = args.pdf_max_pages
    config.page_loader.max_tokens = args.max_new_tokens
    config.page_loader.task_prompt_mapping = {
        "text": args.prompt,
        "table": args.prompt,
        "formula": args.prompt,
    }
    config.layout.model_dir = args.layout_model
    config.layout.device = args.layout_device
    config.layout.batch_size = args.layout_batch_size
    return config


def save_result(result, output_dir: Path, name: str):
    mode_dir = output_dir / name
    mode_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir=str(mode_dir), save_layout_visualization=True)
    (mode_dir / "markdown.md").write_text(result.markdown_result, encoding="utf-8")
    (mode_dir / "result.json").write_text(json.dumps(result.json_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return mode_dir / "markdown.md"


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", default="examples/llm/glm_ocr/data/18UF.pdf")
    parser.add_argument("--output_dir", default="work_dirs/glm_ocr_full_pipeline_compare")
    parser.add_argument("--model", default="/data01/datasets/GLM-OCR")
    parser.add_argument("--layout_model", default="/data01/datasets/ppdoclayoutv3_safetensors")
    parser.add_argument("--layout_device", default="cpu")
    parser.add_argument("--layout_batch_size", type=int, default=1)
    parser.add_argument("--pdf_dpi", type=int, default=200)
    parser.add_argument("--pdf_max_pages", type=int, default=None)
    parser.add_argument("--prompt", default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--mode", choices=["fp", "hmonnx", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", default="eager")

    parser.add_argument("--vision_onnx", default="work_dirs/glm_ocr_vision_xh2a_export_hmonnx/vision/glm_ocr_vision_xh2a_export_hmonnx.onnx")
    parser.add_argument("--prefill_onnx", default="work_dirs/glm_ocr_llm_xh2a_2k_export/prefill_onnx/glm_ocr_llm_xh2a_2k_export_prefill.onnx")
    parser.add_argument("--decode_onnx", default="work_dirs/glm_ocr_llm_xh2a_2k_export/decode_onnx/glm_ocr_llm_xh2a_2k_export_decode.onnx")
    parser.add_argument("--token_embedding", default="work_dirs/glm_ocr_llm_xh2a_2k_export/token_embedding.pt")
    parser.add_argument("--hf_config_path", default="work_dirs/glm_ocr_llm_xh2a_2k_export/hf_config")
    parser.add_argument("--prefill_input_length", type=int, default=256)
    parser.add_argument("--cache_len", type=int, default=2048)
    parser.add_argument("--hmonnx_image_size_w", type=int, default=672)
    parser.add_argument("--hmonnx_image_size_h", type=int, default=672)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_pipeline_config(args)

    outputs: Dict[str, Path] = {}
    if args.mode in ("fp", "both"):
        fp_client = FPLocalOCRClient(
            args.model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )
        with LocalLayoutPipeline(config, fp_client) as pipeline:
            result = pipeline.process_local(args.input, save_layout_visualization=True)
        outputs["fp"] = save_result(result, output_dir, "fp_ocr")

    if args.mode in ("hmonnx", "both"):
        hmonnx_client = HmonnxLocalOCRClient(
            args.model,
            vision_onnx=args.vision_onnx,
            prefill_onnx=args.prefill_onnx,
            decode_onnx=args.decode_onnx,
            token_embedding=args.token_embedding,
            hf_config_path=args.hf_config_path,
            prefill_input_length=args.prefill_input_length,
            max_new_tokens=args.max_new_tokens,
        )
        hmonnx_client.image_size_w = args.hmonnx_image_size_w
        hmonnx_client.image_size_h = args.hmonnx_image_size_h
        hmonnx_client.num_hidden_layers = args.num_hidden_layers
        hmonnx_client.num_kv_heads = args.num_kv_heads
        hmonnx_client.head_dim = args.head_dim
        hmonnx_client.cache_len = args.cache_len
        with LocalLayoutPipeline(config, hmonnx_client) as pipeline:
            result = pipeline.process_local(args.input, save_layout_visualization=True)
        outputs["hmonnx"] = save_result(result, output_dir, "hmonnx_ocr")

    if "fp" in outputs and "hmonnx" in outputs:
        fp_text = outputs["fp"].read_text(encoding="utf-8")
        hmonnx_text = outputs["hmonnx"].read_text(encoding="utf-8")
        ratio = SequenceMatcher(None, fp_text, hmonnx_text).ratio()
        diff = "".join(
            unified_diff(
                fp_text.splitlines(True),
                hmonnx_text.splitlines(True),
                fromfile="fp_ocr_full_pipeline",
                tofile="hmonnx_ocr_full_pipeline",
                n=3,
            )
        )
        (output_dir / "fp_vs_hmonnx.diff").write_text(diff, encoding="utf-8")
        summary = {
            "fp_markdown": str(outputs["fp"]),
            "hmonnx_markdown": str(outputs["hmonnx"]),
            "char_similarity": ratio,
            "diff": str(output_dir / "fp_vs_hmonnx.diff"),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({k: str(v) for k, v in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
