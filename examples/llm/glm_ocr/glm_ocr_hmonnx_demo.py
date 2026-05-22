import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
import xhquant.xhonnxruntime.config as xhonnxruntime_config
from PIL import Image
from xhquant.api import HMONNXInference

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import get_root_logger, render_pdf_to_images, xhquant_llm_init
from xh_model_zoo.xh_llm.models.glm_ocr import GlmOcrONNXModel, GlmOcrProcessor

SDK_ROOT = Path(__file__).resolve().parent / "GLM-OCR"
if SDK_ROOT.exists():
    sys.path.insert(0, str(SDK_ROOT))


def _resolve_path(path_str: str | None, workspace_root: Path) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return str(cwd_path)

    workspace_path = (workspace_root / path).resolve()
    return str(workspace_path)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, default="work_dirs/glm_ocr_llm_xh2a_2k_export",
                        help="LLM export directory containing hf_config/ and token_embedding.pt")
    parser.add_argument("--vision_export_dir", type=str, default="work_dirs/glm_ocr_vision_xh2a_export_hmonnx",
                        help="Vision export directory")
    parser.add_argument("--visual_onnx", type=str, default=None,
                        help="vision onnx path; default: <vision_export_dir>/vision/<vision_dir_name>.onnx")
    parser.add_argument("--prefill_onnx", type=str, default=None,
                        help="prefill onnx path; default: <model_dir>/prefill_onnx/<model_dir_name>_prefill.onnx")
    parser.add_argument("--decode_onnx", type=str, default=None,
                        help="decode onnx path; default: <model_dir>/decode_onnx/<model_dir_name>_decode.onnx")
    parser.add_argument("--image", type=str, default='examples/llm/glm_ocr/data/img3.png')
    parser.add_argument("--pdf", type=str, default=None,
                        help="PDF input path; pages are rendered to images before HMONNX OCR")
    parser.add_argument("--pdf_output_dir", type=str, default="work_dirs/glm_ocr_pdf_pages",
                        help="directory used to store rendered PDF page images")
    parser.add_argument("--pdf_dpi", type=int, default=200,
                        help="DPI used to render PDF pages")
    parser.add_argument("--pdf_pages", type=str, default=None,
                        help="1-based PDF page selection, for example '1', '1,3', or '1-3,5'")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--images_json", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--use_fast", default=True, action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--cache_len", type=int, default=2048)
    parser.add_argument("--input_sequence_length", type=int, default=256)
    parser.add_argument("--image_size_w", type=int, default=672)
    parser.add_argument("--image_size_h", type=int, default=672)
    parser.add_argument("--eos_token_id", type=int, nargs="+", default=[151329])
    parser.add_argument(
        "--vision_mode",
        type=str,
        default="unquantized",
        choices=["quantized", "unquantized"],
        help="select vision model mode: quantized hmonnx-converted onnx or original unquantized onnx",
    )
    parser.add_argument(
        "--vision_onnx_path",
        type=str,
        default=None,
        help="optional manual override for vision ONNX path; has highest priority",
    )
    parser.add_argument("--full_pipeline", action="store_true",
                        help="run official SDK full pipeline: PDF/image -> PP-DocLayoutV3 -> region OCR -> markdown/json")
    parser.add_argument("--full_output_dir", type=str, default="work_dirs/glm_ocr_hmonnx_full_pipeline_demo",
                        help="output directory for --full_pipeline")
    parser.add_argument("--layout_model", type=str, default="/data01/datasets/ppdoclayoutv3_safetensors",
                        help="PP-DocLayoutV3 model/processor directory for full pipeline")
    parser.add_argument("--layout_backend", type=str, default="hmonnx", choices=["hf", "hmonnx"],
                        help="layout backend for --full_pipeline")
    parser.add_argument("--vision_backend", type=str, default="hmonnx", choices=["hf", "hmonnx"],
                        help="vision backend for --full_pipeline OCR")
    parser.add_argument("--llm_backend", type=str, default="hmonnx", choices=["hf", "hmonnx"],
                        help="LLM backend for --full_pipeline OCR")
    parser.add_argument("--layout_hmonnx", type=str,
                        default="work_dirs/ppdoclayoutv3_xh2a_export_hmonnx/hmonnx/ppdoclayoutv3_w16a16_sefp_XH2a.onnx",
                        help="PP-DocLayoutV3 HMONNX path used by --full_pipeline")
    parser.add_argument("--layout_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="execution device for PP-DocLayoutV3 HMONNX")
    parser.add_argument("--layout_batch_size", type=int, default=1,
                        help="PP-DocLayoutV3 HMONNX layout batch size")
    parser.add_argument("--layout_threshold", type=float, default=0.3,
                        help="PP-DocLayoutV3 post-process threshold")
    parser.add_argument("--layout_use_polygon", action="store_true",
                        help="use layout polygons for visualization/crop in full pipeline")
    parser.add_argument("--pdf_max_pages", type=int, default=None,
                        help="maximum PDF pages for official SDK full pipeline")
    parser.add_argument("--diagnose_components", action="store_true",
                        help="run all-HF reference plus one-at-a-time HMONNX->HF ablations for layout/vision/llm")
    parser.add_argument("--diagnose_max_cases", type=int, default=None,
                        help="optional limit for diagnosis cases, useful for smoke tests")
    parser.add_argument("--hf_model_dir", type=str, default="/data01/datasets/GLM-OCR",
                        help="HF GLM-OCR directory used by component diagnosis")
    parser.add_argument("--hf_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                        help="HF dtype used by component diagnosis")
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        help="HF attention implementation used by component diagnosis")
    return parser


def _resolve_torch_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


class PPDocLayoutV3HmonnxDetector:
    """Official SDK-compatible PP-DocLayoutV3 detector backed by HMONNX."""

    def __init__(self, config, hmonnx_path: str, device: str = "cuda", batch_size: int | None = None):
        self.config = config
        self.model_dir = config.model_dir
        self.hmonnx_path = hmonnx_path
        self.device = device
        self.batch_size = batch_size or config.batch_size
        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.label_task_mapping = config.label_task_mapping
        self.id2label = getattr(config, "id2label", None)
        self._session = None
        self._image_processor = None
        self._device = None

    def start(self):
        from transformers import PPDocLayoutV3ImageProcessor

        self._image_processor = PPDocLayoutV3ImageProcessor.from_pretrained(self.model_dir)
        self._session = HMONNXInference(self.hmonnx_path)
        self._session.to("cpu")
        self._session.exec_device = self.device
        self._device = torch.device(self.device if torch.cuda.is_available() or not str(self.device).startswith("cuda") else "cpu")

        if self.id2label is None:
            cfg = getattr(self._image_processor, "config", None)
            self.id2label = getattr(cfg, "id2label", None)
        if self.id2label is None:
            import json as _json

            config_file = Path(self.model_dir) / "config.json"
            self.id2label = _json.loads(config_file.read_text(encoding="utf-8"))["id2label"]
        self.id2label = {int(key): value for key, value in self.id2label.items()}
        if self.label_task_mapping is None:
            self.label_task_mapping = {"text": list(self.id2label.values())}
        self._patch_safe_polygon_extract()

    def stop(self):
        self._session = None
        self._image_processor = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _patch_safe_polygon_extract(self):
        import cv2

        def _safe_extract(boxes, masks, scale_ratio):
            scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
            mask_h, mask_w = masks.shape[1:]
            polygon_points = []
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].astype(np.int32)
                box_w, box_h = x_max - x_min, y_max - y_min
                rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
                if box_w <= 0 or box_h <= 0:
                    polygon_points.append(rect)
                    continue
                x_start = int(round((x_min * scale_w).item()))
                x_end = int(round((x_max * scale_w).item()))
                y_start = int(round((y_min * scale_h).item()))
                y_end = int(round((y_max * scale_h).item()))
                x_start, x_end = np.clip([x_start, x_end], 0, mask_w)
                y_start, y_end = np.clip([y_start, y_end], 0, mask_h)
                cropped_mask = masks[i, y_start:y_end, x_start:x_end]
                if cropped_mask.size == 0:
                    polygon_points.append(rect)
                    continue
                resized = cv2.resize(cropped_mask.astype(np.uint8), (box_w, box_h), interpolation=cv2.INTER_NEAREST)
                polygon = self._image_processor._mask2polygon(resized)
                if polygon is not None and len(polygon) < 4:
                    polygon_points.append(rect)
                    continue
                if polygon is not None and len(polygon) > 0:
                    polygon = polygon + np.array([x_min, y_min])
                polygon_points.append(polygon)
            return polygon_points

        self._image_processor._extract_polygon_points_by_masks = _safe_extract

    def _apply_per_class_threshold(self, raw_results: List[Dict]):
        if not self.threshold_by_class:
            return raw_results
        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}
        class_thresholds = {}
        for key, value in self.threshold_by_class.items():
            class_thresholds[label2id[key] if isinstance(key, str) and key in label2id else int(key)] = float(value)
        filtered = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]
            thresholds = torch.full_like(scores, self.threshold)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh
            keep = scores >= thresholds
            item = {"scores": scores[keep], "labels": labels[keep], "boxes": result["boxes"][keep]}
            if "order_seq" in result:
                item["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                item["polygon_points"] = [p for p, k in zip(result["polygon_points"], keep_list) if k]
            filtered.append(item)
        return filtered

    def process(self, images: List[Image.Image], save_visualization: bool = False, global_start_idx: int = 0, use_polygon: bool = False):
        if self._session is None or self._image_processor is None:
            raise RuntimeError("Layout detector not started. Call start() first.")
        from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
        from glmocr.utils.visualization_utils import draw_layout_boxes

        pil_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        all_paddle_format_results = []

        for chunk_start in range(0, len(pil_images), self.batch_size):
            chunk_pil = pil_images[chunk_start:chunk_start + self.batch_size]
            inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].half().to(self._device)
            with torch.no_grad():
                outputs = self._session(pixel_values)
            if not isinstance(outputs, (tuple, list)):
                outputs = [outputs]
            model_outputs = SimpleNamespace(
                logits=outputs[0],
                pred_boxes=outputs[1],
                order_logits=outputs[2],
                out_masks=outputs[3],
            )
            target_sizes = torch.tensor([img.size[::-1] for img in chunk_pil], device=self._device)
            pre_threshold = min(self.threshold, min(self.threshold_by_class.values())) if self.threshold_by_class else self.threshold
            raw_results = self._image_processor.post_process_object_detection(
                model_outputs,
                threshold=pre_threshold,
                target_sizes=target_sizes,
            )
            raw_results = self._apply_per_class_threshold(raw_results)
            all_paddle_format_results.extend(
                apply_layout_postprocess(
                    raw_results=raw_results,
                    id2label=self.id2label,
                    img_sizes=[img.size for img in chunk_pil],
                    layout_nms=self.layout_nms,
                    layout_unclip_ratio=self.layout_unclip_ratio,
                    layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
                )
            )

        vis_images: Dict[int, Image.Image] = {}
        if save_visualization:
            for img_idx, img_results in enumerate(all_paddle_format_results):
                vis_images[global_start_idx + img_idx] = draw_layout_boxes(
                    image=np.array(pil_images[img_idx]),
                    boxes=img_results,
                    use_polygon=use_polygon,
                )

        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width, image_height = pil_images[img_idx].size
            results = []
            valid_index = 0
            for item in paddle_results:
                label = item["label"]
                task_type = None
                for task_item, labels in self.label_task_mapping.items():
                    if isinstance(labels, list) and label in labels:
                        task_type = task_item
                        break
                if task_type is None or task_type == "abandon":
                    continue
                x1, y1, x2, y2 = item["coordinate"]
                polygon = [
                    [int(float(point[0]) / image_width * 1000), int(float(point[1]) / image_height * 1000)]
                    for point in item["polygon_points"]
                ]
                results.append(
                    {
                        "index": valid_index,
                        "label": label,
                        "score": float(item["score"]),
                        "bbox_2d": [
                            int(float(x1) / image_width * 1000),
                            int(float(y1) / image_height * 1000),
                            int(float(x2) / image_width * 1000),
                            int(float(y2) / image_height * 1000),
                        ],
                        "polygon": polygon,
                        "task_type": task_type,
                    }
                )
                valid_index += 1
            all_results.append(results)

        return all_results, vis_images


class HmonnxLocalOCRClient:
    def __init__(self, model: GlmOcrONNXModel, processor: GlmOcrProcessor, max_new_tokens: int, use_fast: bool, do_sample: bool, logger):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.use_fast = use_fast
        self.do_sample = do_sample
        self.logger = logger

    def start(self):
        self.model.init_image_feature()
        self.model.init_prefill()
        self.model.init_decode()

    def stop(self):
        self.model.release_all_sessions()

    def process_image(self, image: Image.Image, task_type: str = "text", prompt: str | None = None) -> str:  # noqa: ARG002
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.convert("RGB").save(tmp.name)
            return self.model.chat(
                prompt or "Text Recognition:",
                tmp.name,
                self.processor,
                self.logger,
                use_fast=self.use_fast,
                do_sample=self.do_sample,
                max_new_tokens=self.max_new_tokens,
                keep_sessions=True,
            )


class FPLocalOCRClient:
    def __init__(self, model_path: str, device: str, dtype: str, max_new_tokens: int, attn_implementation: str):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = _resolve_torch_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.processor = None
        self.model = None

    def start(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

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

    def process_image(self, image: Image.Image, task_type: str = "text", prompt: str | None = None) -> str:  # noqa: ARG002
        from common import build_inputs, build_messages

        if self.processor is None or self.model is None:
            raise RuntimeError("FPLocalOCRClient is not started")
        messages = build_messages(image, prompt or "Text Recognition:")
        inputs = build_inputs(self.processor, messages, device=self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


class HybridVisionHmonnxOCRClient(HmonnxLocalOCRClient):
    def __init__(self, *args, hf_model_dir: str, hf_dtype: str, attn_implementation: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_model_dir = hf_model_dir
        self.hf_dtype = _resolve_torch_dtype(hf_dtype)
        self.attn_implementation = attn_implementation
        self.hf_model = None

    def start(self):
        from transformers import AutoModelForImageTextToText

        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            self.hf_model_dir,
            dtype=self.hf_dtype,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
        ).eval().to(self.model._exec_device)
        self.model.init_prefill()
        self.model.init_decode()

    def stop(self):
        self.hf_model = None
        self.model.release_prefill_session()
        self.model.release_decode_session()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_image(self, image: Image.Image, task_type: str = "text", prompt: str | None = None) -> str:  # noqa: ARG002
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.convert("RGB").save(tmp.name)
            return self.model.chat(
                prompt or "Text Recognition:",
                tmp.name,
                self.processor,
                self.logger,
                use_fast=self.use_fast,
                do_sample=self.do_sample,
                max_new_tokens=self.max_new_tokens,
                keep_sessions=True,
                image_feature_fn=self._extract_hf_image_features,
            )

    def _extract_hf_image_features(self, pixel_values, image_grid_thw):
        with torch.no_grad():
            outputs = self.hf_model.get_image_features(
                pixel_values=pixel_values.to(self.hf_model.device),
                image_grid_thw=image_grid_thw.to(self.hf_model.device),
            )
        image_embeds = torch.cat(outputs.pooler_output, dim=0)
        return image_embeds.to(pixel_values.device, dtype=torch.float16)


class LocalLayoutPipeline:
    def __init__(self, config, layout_detector, ocr_client):
        from glmocr.pipeline import Pipeline

        self._pipeline = Pipeline(config=config, layout_detector=layout_detector)
        self.page_loader = self._pipeline.page_loader
        self.layout_detector = self._pipeline.layout_detector
        self.result_formatter = self._pipeline.result_formatter
        self.config = config
        self.ocr_client = ocr_client

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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
                item["content"] = self.ocr_client.process_image(cropped, task_type=item.get("task_type", "text"))
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


def _save_pipeline_result(result, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir=str(output_dir), save_layout_visualization=True)
    markdown_path = output_dir / "markdown.md"
    result_json_path = output_dir / "result.json"
    markdown_path.write_text(result.markdown_result, encoding="utf-8")
    result_json_path.write_text(json.dumps(result.json_result, ensure_ascii=False, indent=2), encoding="utf-8")
    region_counts = [len(page) for page in result.json_result] if isinstance(result.json_result, list) else []
    return {
        "markdown": str(markdown_path),
        "result_json": str(result_json_path),
        "markdown_chars": len(result.markdown_result),
        "region_counts": region_counts,
    }


def _build_pipeline_config(args, workspace_root: Path):
    from glmocr.config import PipelineConfig

    config = PipelineConfig()
    config.max_workers = 1
    config.page_loader.pdf_dpi = args.pdf_dpi
    config.page_loader.pdf_max_pages = args.pdf_max_pages
    config.page_loader.max_tokens = args.max_new_tokens
    config.page_loader.task_prompt_mapping = {
        "text": args.prompt or "Text Recognition:",
        "table": args.prompt or "Text Recognition:",
        "formula": args.prompt or "Text Recognition:",
    }
    config.layout.model_dir = _resolve_path(args.layout_model, workspace_root)
    config.layout.device = args.layout_device
    config.layout.batch_size = args.layout_batch_size
    config.layout.threshold = args.layout_threshold
    config.layout.use_polygon = args.layout_use_polygon
    return config


def _make_hmonnx_layout_detector(args, config, workspace_root: Path, exec_device):
    layout_hmonnx = _resolve_path(args.layout_hmonnx, workspace_root)
    if layout_hmonnx is None or not Path(layout_hmonnx).exists():
        raise FileNotFoundError(f"PP-DocLayoutV3 HMONNX not found: {layout_hmonnx}")
    return PPDocLayoutV3HmonnxDetector(
        config.layout,
        hmonnx_path=layout_hmonnx,
        device=str(exec_device),
        batch_size=args.layout_batch_size,
    )


def _make_hf_layout_detector(config):
    from glmocr.layout import PPDocLayoutDetector

    return PPDocLayoutDetector(config.layout)


def _run_full_pipeline_case(case_name: str, source: str, output_root: Path, config, layout_detector, ocr_client) -> dict:
    case_dir = output_root / case_name
    with LocalLayoutPipeline(config, layout_detector, ocr_client) as pipeline:
        result = pipeline.process_local(source, save_layout_visualization=True)
    summary = _save_pipeline_result(result, case_dir)
    summary["case"] = case_name
    return summary


def _resolve_vision_onnx_path(args, default_visual_onnx, workspace_root: Path) -> tuple[str, str]:
    if args.vision_onnx_path is not None:
        path = _resolve_path(args.vision_onnx_path, workspace_root)
        return path, "manual"

    use_unquantized = args.vision_mode == "unquantized"

    if not use_unquantized:
        path = _resolve_path(default_visual_onnx, workspace_root)
        return path, "quantized"

    # Try to find unquantized onnx in the vision export dir
    candidate_paths = []
    vision_export_dir = Path(args.vision_export_dir)
    candidate_paths.append(vision_export_dir / "onnx" / "visual_1.onnx")

    quantized_path = Path(default_visual_onnx)
    if "vision" in quantized_path.parts:
        idx = quantized_path.parts.index("vision")
        root = Path(*quantized_path.parts[:idx])
        candidate_paths.append(root / "onnx" / "visual_1.onnx")

    for candidate in candidate_paths:
        resolved = _resolve_path(str(candidate), workspace_root)
        if resolved is not None and Path(resolved).exists():
            return resolved, "unquantized"

    raise FileNotFoundError(
        "Cannot find unquantized vision ONNX. Please pass --vision_onnx_path explicitly, "
        "or ensure <vision_export_dir>/onnx/visual_1.onnx exists."
    )


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    workspace_root = Path(__file__).resolve().parents[3]

    model_dir = Path(args.model_dir)
    vision_export_dir = Path(args.vision_export_dir)
    model_dir_name = model_dir.name
    vision_dir_name = vision_export_dir.name

    # Derive default ONNX paths
    default_visual_onnx = args.visual_onnx or str(vision_export_dir / "vision" / f"{vision_dir_name}.onnx")
    prefill_onnx = args.prefill_onnx or str(model_dir / "prefill_onnx" / f"{model_dir_name}_prefill.onnx")
    decode_onnx = args.decode_onnx or str(model_dir / "decode_onnx" / f"{model_dir_name}_decode.onnx")
    hf_model_config_dir = str(model_dir / "hf_config")
    embed_tokens_path = str(model_dir / "token_embedding.pt")

    work_dir = str(Path("./work_dirs") / "glm_ocr_xh2a_hmonnx_demo")
    log_file = Path(work_dir) / "demo_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)
    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xhquant_llm_init(log_file, False)
    logger = get_root_logger()
    logger.info(f"Args: {args}")

    xhquant.utils.suppress_printing.disable_printing = True
    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False

    # Load token embedding
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding_path = _resolve_path(embed_tokens_path, workspace_root)
    token_embedding = torch.load(token_embedding_path, weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    hf_model_config_dir = _resolve_path(hf_model_config_dir, workspace_root)
    processor = GlmOcrProcessor.from_pretrained(hf_model_config_dir)

    # Resolve vision onnx path (quantized vs unquantized)
    vision_onnx_path, vision_mode = _resolve_vision_onnx_path(args, default_visual_onnx, workspace_root)
    logger.info(f"Vision mode: {vision_mode}, onnx={vision_onnx_path}")

    image_feature_cfg = {"onnx": vision_onnx_path}
    prefill_cfg = {
        "onnx": _resolve_path(prefill_onnx, workspace_root),
        "input_sequence_length": args.input_sequence_length,
    }
    decode_cfg = {"onnx": _resolve_path(decode_onnx, workspace_root)}
    kv_cache_cfg = {
        "num_hidden_layers": args.num_hidden_layers,
        "shape": [1, args.num_kv_heads, args.cache_len, args.head_dim],
    }

    xh_model: GlmOcrONNXModel = GlmOcrONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        cache_len=args.cache_len,
        image_size_w=args.image_size_w,
        image_size_h=args.image_size_h,
        presence_penalty=args.presence_penalty,
        eos_token_id=args.eos_token_id,
    )
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)
    xh_model.to(exec_device)

    if args.full_pipeline:
        if not SDK_ROOT.exists():
            raise RuntimeError(f"Official GLM-OCR SDK not found at {SDK_ROOT}")

        source = args.pdf if args.pdf is not None else args.image
        source = _resolve_path(source, workspace_root)
        full_output_dir = Path(_resolve_path(args.full_output_dir, workspace_root))
        full_output_dir.mkdir(parents=True, exist_ok=True)

        def make_hmonnx_client():
            return HmonnxLocalOCRClient(
                xh_model,
                processor,
                max_new_tokens=args.max_new_tokens,
                use_fast=args.use_fast,
                do_sample=args.do_sample,
                logger=logger,
            )

        def make_hf_llm_client():
            return FPLocalOCRClient(
                _resolve_path(args.hf_model_dir, workspace_root),
                device=str(exec_device),
                dtype=args.hf_dtype,
                max_new_tokens=args.max_new_tokens,
                attn_implementation=args.attn_implementation,
            )

        def make_hf_vision_client():
            return HybridVisionHmonnxOCRClient(
                xh_model,
                processor,
                max_new_tokens=args.max_new_tokens,
                use_fast=args.use_fast,
                do_sample=args.do_sample,
                logger=logger,
                hf_model_dir=_resolve_path(args.hf_model_dir, workspace_root),
                hf_dtype=args.hf_dtype,
                attn_implementation=args.attn_implementation,
            )

        def make_ocr_client(vision_backend: str, llm_backend: str):
            if llm_backend == "hf":
                return make_hf_llm_client()
            if vision_backend == "hf":
                return make_hf_vision_client()
            return make_hmonnx_client()

        def make_layout_detector(layout_backend: str, config):
            if layout_backend == "hf":
                return _make_hf_layout_detector(config)
            return _make_hmonnx_layout_detector(args, config, workspace_root, exec_device)

        if args.diagnose_components:
            cases = [
                ("000_hf_all", "hf", "hf", "hf"),
                ("100_hmonnx_all", "hmonnx", "hmonnx", "hmonnx"),
                ("110_hf_layout", "hf", "hmonnx", "hmonnx"),
                ("120_hf_vision", "hmonnx", "hf", "hmonnx"),
                ("130_hf_llm", "hmonnx", "hmonnx", "hf"),
            ]
            if args.diagnose_max_cases is not None:
                cases = cases[: args.diagnose_max_cases]
            summaries = []
            hf_reference_text = None
            for case_name, layout_backend, vision_backend, llm_backend in cases:
                logger.info(
                    f"Running diagnosis case: {case_name} "
                    f"layout={layout_backend} vision={vision_backend} llm={llm_backend}"
                )
                case_config = _build_pipeline_config(args, workspace_root)
                layout_detector = make_layout_detector(layout_backend, case_config)
                ocr_client = make_ocr_client(vision_backend, llm_backend)
                case_summary = _run_full_pipeline_case(case_name, source, full_output_dir, case_config, layout_detector, ocr_client)
                case_summary.update(
                    {
                        "layout_backend": layout_backend,
                        "vision_backend": vision_backend,
                        "llm_backend": llm_backend,
                    }
                )
                case_text = Path(case_summary["markdown"]).read_text(encoding="utf-8")
                if case_name == "000_hf_all":
                    hf_reference_text = case_text
                    case_summary["similarity_to_hf_all"] = 1.0
                elif hf_reference_text is not None:
                    case_summary["similarity_to_hf_all"] = SequenceMatcher(None, hf_reference_text, case_text).ratio()
                summaries.append(case_summary)
            summary = {
                "input": source,
                "layout_hmonnx": _resolve_path(args.layout_hmonnx, workspace_root),
                "layout_model": _resolve_path(args.layout_model, workspace_root),
                "cases": summaries,
            }
            (full_output_dir / "diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return

        config = _build_pipeline_config(args, workspace_root)
        layout_detector = make_layout_detector(args.layout_backend, config)
        ocr_client = make_ocr_client(args.vision_backend, args.llm_backend)
        summary = _run_full_pipeline_case("hmonnx_full", source, full_output_dir, config, layout_detector, ocr_client)
        summary.update({
            "input": source,
            "layout_backend": args.layout_backend,
            "vision_backend": args.vision_backend,
            "llm_backend": args.llm_backend,
            "layout_hmonnx": _resolve_path(args.layout_hmonnx, workspace_root),
            "layout_model": config.layout.model_dir,
        })
        (full_output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    output_path = args.output_path
    if output_path is None:
        output_path = str(Path(work_dir) / "demo_output.txt")
    output_path = _resolve_path(output_path, workspace_root)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    default_prompt = "Text Recognition:"

    def run_single(image_path: str, prompt: str) -> tuple[str, float]:
        image_path = _resolve_path(image_path, workspace_root)
        start = time.perf_counter()
        out = xh_model.chat(
            prompt,
            image_path,
            processor,
            logger,
            use_fast=args.use_fast,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - start
        return out, elapsed

    if args.pdf is not None:
        pdf_path = _resolve_path(args.pdf, workspace_root)
        pdf_output_dir = Path(_resolve_path(args.pdf_output_dir, workspace_root)) / Path(pdf_path).stem
        page_images = render_pdf_to_images(
            pdf_path=pdf_path,
            output_dir=pdf_output_dir,
            dpi=args.pdf_dpi,
            pages=args.pdf_pages,
        )
        if not page_images:
            raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")

        prompt = args.prompt if args.prompt is not None else default_prompt
        with open(output_path, "w", encoding="utf-8") as f_txt:
            for i, page_image in enumerate(page_images, 1):
                print(f"Processing PDF page {i}/{len(page_images)}: {page_image}", flush=True)
                out, elapsed = run_single(str(page_image), prompt)
                f_txt.write(f"===== Page {i} ({page_image.name}) =====\n")
                f_txt.write(out + "\n\n")
                f_txt.flush()
                print(f"[PDF page {i}] elapsed: {elapsed:.3f}s", flush=True)
                print(out, flush=True)
    elif args.images_json is not None:
        images_json_path = _resolve_path(args.images_json, workspace_root)
        with open(output_path, "w", encoding="utf-8") as f_txt:
            with open(images_json_path, "r", encoding="utf-8") as f:
                images_json = json.load(f)
            if isinstance(images_json, dict):
                items = list(images_json.items())
            elif isinstance(images_json, list):
                items = [(str(i), item) for i, item in enumerate(images_json)]
            else:
                raise TypeError(f"Unsupported images_json format: {type(images_json)}")

            for i, (image_name, image_info) in enumerate(items, 1):
                image_path = image_info.get("url") or image_info.get("image")
                prompt = image_info.get("prompt", default_prompt)
                if image_path is None:
                    logger.warning(f"skip sample {image_name}: missing 'url'/'image' field")
                    continue

                print(f"Processing {i}: {image_path}", flush=True)
                out, elapsed = run_single(image_path, prompt)
                f_txt.write(f"{image_name}\n{out}\n")
                f_txt.flush()
                print(f"[{i}] {image_name}", flush=True)
                print(f"elapsed: {elapsed:.3f}s", flush=True)
                print(out, flush=True)
    else:
        image_path = args.image if args.image is not None else "examples/llm/glm_ocr/data/img3.png"
        prompt = args.prompt if args.prompt is not None else default_prompt
        out, elapsed = run_single(image_path, prompt)
        with open(output_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(out + "\n")
        print(f"elapsed: {elapsed:.3f}s", flush=True)
        print(out, flush=True)


if __name__ == "__main__":
    main()
