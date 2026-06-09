import argparse
import copy
import json
import math
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE, smart_resize
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl import XHQwen2VLProcessor
    from xhmodel_merak.xh_llm.models.qwen2_vl.qwen2_vl_hmonnx_inference import XHQwen2VLHMONNXModel


MINERU_VISUAL_BUCKETS_MANIFEST = "mineru_visual_buckets.json"


class StaticBucketProcessorAdapter:
    """Keep MinerU crops on a finite set of static Qwen2-VL visual buckets."""

    def __init__(
        self,
        processor: "XHQwen2VLProcessor",
        buckets: list[tuple[int, int]],
        fallback_bucket: tuple[int, int],
        patch_size: int,
        spatial_merge_size: int,
        logger,
        max_upscale: float = 2.0,
    ) -> None:
        self.processor = processor
        self.buckets = sorted(set(buckets), key=lambda item: (item[0] * item[1], item[0], item[1]))
        self.fallback_bucket = fallback_bucket
        self.patch_size = patch_size
        self.patch_factor = patch_size * spatial_merge_size
        self.logger = logger
        if max_upscale <= 0:
            raise ValueError("max_upscale must be positive.")
        self.max_upscale = max_upscale

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        return Qwen2VLProcessor.apply_chat_template(
            self.processor,
            conversation,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        expected_buckets = None
        images = kwargs.get("images")
        if images is not None:
            kwargs["images"], expected_buckets = self._bucket_images(images)
        model_inputs = self.processor(*args, **kwargs)
        if expected_buckets:
            self._validate_image_grid(model_inputs, expected_buckets)
        pixel_values = model_inputs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor):
            model_inputs["hm_pixel_values"] = [pixel_values.contiguous().float()]
        return model_inputs

    def _bucket_images(self, images):
        single_image = isinstance(images, Image.Image)
        image_list = [images] if single_image else list(images)
        bucketed_images = []
        expected_buckets = []
        for image in image_list:
            if not isinstance(image, Image.Image):
                bucketed_images.append(image)
                continue
            bucketed_image, bucket, native_size, render_size, score = self._bucket_image(image)
            bucketed_images.append(bucketed_image)
            expected_buckets.append(bucket)
            self.logger.info(
                f"MinerU static visual bucket: native={native_size}, bucket={bucket}, "
                f"render={render_size}, score={score:.4f}"
            )
        return (bucketed_images[0] if single_image else bucketed_images), expected_buckets

    def _bucket_image(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, tuple[int, int], tuple[int, int], tuple[int, int], float]:
        image = image.convert("RGB")
        native_h, native_w = smart_resize(image.height, image.width, factor=self.patch_factor)
        bucket, score = self._select_bucket(native_h, native_w)
        native_resized = image.resize((native_w, native_h), Image.Resampling.BICUBIC)
        bucketed, render_size = self._letterbox(native_resized, bucket)
        return bucketed, bucket, (native_h, native_w), render_size, score

    def _select_bucket(self, native_h: int, native_w: int) -> tuple[tuple[int, int], float]:
        scored = [(self._bucket_score(native_h, native_w, bucket), bucket) for bucket in self.buckets]
        score, bucket = min(scored, key=lambda item: (item[0], item[1][0] * item[1][1], item[1]))
        return bucket, score

    def _bucket_score(self, native_h: int, native_w: int, bucket: tuple[int, int]) -> float:
        bucket_h, bucket_w = bucket
        native_ratio = native_w / native_h
        bucket_ratio = bucket_w / bucket_h
        aspect_cost = abs(math.log(native_ratio / bucket_ratio))
        scale = min(bucket_h / native_h, bucket_w / native_w)
        effective_scale = min(scale, self.max_upscale)
        render_h = max(1, min(bucket_h, round(native_h * effective_scale)))
        render_w = max(1, min(bucket_w, round(native_w * effective_scale)))
        padding_cost = 1.0 - (render_h * render_w) / (bucket_h * bucket_w)
        downscale_cost = max(0.0, -math.log(scale))
        return 2.0 * aspect_cost + 0.8 * downscale_cost + 0.2 * padding_cost

    def _letterbox(self, image: Image.Image, bucket: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
        bucket_h, bucket_w = bucket
        scale = min(bucket_h / image.height, bucket_w / image.width)
        scale = min(scale, self.max_upscale)
        render_w = max(1, min(bucket_w, round(image.width * scale)))
        render_h = max(1, min(bucket_h, round(image.height * scale)))
        if (render_w, render_h) != image.size:
            image = image.resize((render_w, render_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (bucket_w, bucket_h), (255, 255, 255))
        canvas.paste(image, ((bucket_w - image.width) // 2, (bucket_h - image.height) // 2))
        return canvas, (render_h, render_w)

    def _validate_image_grid(self, model_inputs, expected_buckets: list[tuple[int, int]]) -> None:
        image_grid_thw = model_inputs.get("image_grid_thw")
        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeError("Expected image_grid_thw after static bucket preprocessing.")
        if image_grid_thw.shape[0] != len(expected_buckets):
            raise RuntimeError(
                f"image_grid_thw count {image_grid_thw.shape[0]} does not match bucket count {len(expected_buckets)}."
            )
        for idx, bucket in enumerate(expected_buckets):
            expected_h = bucket[0] // self.patch_size
            expected_w = bucket[1] // self.patch_size
            actual_h = int(image_grid_thw[idx, 1].item())
            actual_w = int(image_grid_thw[idx, 2].item())
            if (actual_h, actual_w) != (expected_h, expected_w):
                raise RuntimeError(
                    f"Static bucket grid mismatch: bucket={bucket}, "
                    f"expected grid={(expected_h, expected_w)}, actual grid={(actual_h, actual_w)}"
                )


class StaticHMONNXVisualRouter:
    def __init__(
        self,
        hmonnx_model: "XHQwen2VLHMONNXModel",
        manifest: dict,
        manifest_dir: Path,
        device: str,
        dtype: torch.dtype,
        fast: bool,
        golden: bool,
        auto_offload: bool,
        logger,
    ) -> None:
        self.hmonnx_model = hmonnx_model
        self.manifest_dir = manifest_dir
        self.device = device
        self.dtype = dtype
        self.patch_size = int(manifest["patch_size"])
        self.fallback_bucket = _parse_bucket(manifest["fallback_bucket"])
        self.logger = logger
        self.visual_models: dict[tuple[int, int], HMONNXModel] = {}
        self.buckets = []
        self._build_all(manifest, fast=fast, golden=golden, auto_offload=auto_offload)

    def activate_for_grid(self, image_grid_thw) -> tuple[int, int]:
        bucket = self._bucket_from_grid(image_grid_thw)
        visual_model = self.visual_models.get(bucket)
        if visual_model is None:
            self.logger.warning(f"Missing static visual bucket {bucket}; fallback to {self.fallback_bucket}")
            bucket = self.fallback_bucket
            visual_model = self.visual_models[bucket]
        if self.hmonnx_model.visual is not visual_model:
            self.hmonnx_model.visual = visual_model
        self.logger.info(f"MinerU active static visual bucket: {bucket}")
        return bucket

    def _build_all(self, manifest: dict, fast: bool, golden: bool, auto_offload: bool) -> None:
        default_hmonnx = Path(self.hmonnx_model.visual_meta.hmonnx).resolve()
        for item in manifest["buckets"]:
            bucket = _parse_bucket(item)
            hmonnx_path = self._resolve_manifest_path(item["hmonnx"])
            if hmonnx_path.resolve() == default_hmonnx:
                visual_model = self.hmonnx_model.visual
            else:
                visual_model = HMONNXModel(str(hmonnx_path))
            visual_model.to(self.device)
            visual_model._set_dtype(self.dtype)
            if fast and not golden:
                visual_model.to_fast()
            if golden:
                visual_model.enable_golden = True
            if auto_offload:
                visual_model.hmonnx_session.enable_auto_offload = True
            self.visual_models[bucket] = visual_model
            self.buckets.append(bucket)
            self.logger.info(f"Loaded static visual HMONNX bucket {bucket}: {hmonnx_path}")
        if self.fallback_bucket not in self.visual_models:
            raise RuntimeError(f"Fallback static visual bucket {self.fallback_bucket} is missing from manifest.")
        self.buckets = sorted(set(self.buckets), key=lambda item: (item[0] * item[1], item[0], item[1]))

    def _resolve_manifest_path(self, hmonnx_path: str) -> Path:
        path = Path(hmonnx_path)
        if path.is_absolute():
            return path
        manifest_relative = self.manifest_dir / path
        if manifest_relative.exists():
            return manifest_relative
        cwd_relative = Path.cwd() / path
        if cwd_relative.exists():
            return cwd_relative
        return manifest_relative

    def _bucket_from_grid(self, image_grid_thw) -> tuple[int, int]:
        if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.numel() == 0:
            return self.fallback_bucket
        grid_h = int(image_grid_thw[0, 1].item())
        grid_w = int(image_grid_thw[0, 2].item())
        return (grid_h * self.patch_size, grid_w * self.patch_size)


class MinerUHMONNXModelAdapter:
    """Expose the HF model surface MinerU's transformers backend expects."""

    def __init__(
        self,
        hmonnx_model: "XHQwen2VLHMONNXModel",
        visual_router: StaticHMONNXVisualRouter,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.hmonnx_model = hmonnx_model
        self.visual_router = visual_router
        self._device = device
        self._dtype = dtype
        self.config = AutoConfig.from_pretrained(hmonnx_model.hf_model_dir)
        if not hasattr(self.config, "max_position_embeddings"):
            self.config.max_position_embeddings = hmonnx_model.meta_info.model_config.context_max_length

    def __getattr__(self, name):
        return getattr(self.hmonnx_model, name)

    @property
    def device(self):
        return torch.device(self._device)

    @property
    def dtype(self):
        return self._dtype

    def generate(self, *args, **kwargs):
        self._prepare_hmonnx_visual_inputs(kwargs)
        self.visual_router.activate_for_grid(kwargs.get("image_grid_thw"))
        _install_chunked_qwen2vl_forward(self.hmonnx_model)
        with ContextManagers([LLMInferenceContextManager(self.hmonnx_model), torch.no_grad()]):
            return self.hmonnx_model.generate(*args, **kwargs)

    @staticmethod
    def _prepare_hmonnx_visual_inputs(kwargs):
        hm_pixel_values = kwargs.get("hm_pixel_values")
        pixel_values = kwargs.get("pixel_values")
        if hm_pixel_values is None and isinstance(pixel_values, torch.Tensor):
            kwargs["hm_pixel_values"] = [pixel_values.contiguous().float()]
        elif isinstance(hm_pixel_values, torch.Tensor):
            kwargs["hm_pixel_values"] = [hm_pixel_values.contiguous().float()]


def _install_chunked_qwen2vl_forward(hmonnx_model: "XHQwen2VLHMONNXModel") -> None:
    if hmonnx_model.hf_compatible_model is None:
        llm_model_cls = hmonnx_model.LLM_MODEL_CLS
        hf_model = llm_model_cls._get_hf_model_for_compatible(hmonnx_model.hf_model_dir)
        hf_compatible_model = llm_model_cls.build_hf_compatible_model(hf_model, hmonnx_model)
        hf_compatible_model.to(device=hmonnx_model.device, dtype=hmonnx_model.dtype)
        hmonnx_model.hf_compatible_model = hf_compatible_model
    infer_model = hmonnx_model.hf_compatible_model
    if getattr(infer_model, "_mineru_chunked_qwen2vl_forward", False):
        return
    infer_model.forward = MethodType(_chunked_qwen2vl_forward, infer_model)
    infer_model._mineru_chunked_qwen2vl_forward = True


def _chunked_qwen2vl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    hm_pixel_values: Optional[list[torch.Tensor]] = None,
    **kwargs,
) -> Union[tuple, Qwen2VLCausalLMOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if input_ids is None:
        raise ValueError("MinerU HMONNX chunked Qwen2-VL forward requires input_ids.")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_embeds = None
    if hm_pixel_values is None and isinstance(pixel_values, (list, tuple)):
        hm_pixel_values = pixel_values
    if hm_pixel_values is not None:
        image_embeds = []
        for pixel_values_i in hm_pixel_values:
            image_embeds_i = self._llm_model.visual.forward(
                pixel_values_i.type(self._llm_model.visual.dtype).to(self._llm_model.visual.device)
            )
            image_embeds.append(image_embeds_i)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

    data_processor = self._llm_model.get_data_preprocessor()
    seq_length = int(input_ids.shape[-1])
    net_input_seq_len = int(self._llm_model.get_input_sequence_length())
    original_input_sequence_length = int(data_processor.input_sequence_length)
    data_processor.input_sequence_length = max(seq_length, 1)
    try:
        data_input = data_processor(
            {
                "input_ids": input_ids,
                "image_embeds": image_embeds,
                "past_seq_length": self._past_seq_length,
                "image_grid_thw": image_grid_thw,
            }
        )
    finally:
        data_processor.input_sequence_length = original_input_sequence_length

    (
        full_inputs_embeds,
        full_time_position_ids,
        full_height_position_ids,
        full_width_position_ids,
        past_seq_length,
        _current_seq_length,
        past_key_caches,
        past_value_caches,
    ) = data_input

    steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len
    padded_length = steps * net_input_seq_len
    if padded_length > seq_length:
        pad_len = padded_length - seq_length
        full_inputs_embeds = torch.cat(
            [
                full_inputs_embeds,
                torch.zeros(
                    full_inputs_embeds.shape[0],
                    pad_len,
                    full_inputs_embeds.shape[-1],
                    dtype=full_inputs_embeds.dtype,
                    device=full_inputs_embeds.device,
                ),
            ],
            dim=1,
        )
        pos_pad = torch.zeros(pad_len, dtype=full_time_position_ids.dtype, device=full_time_position_ids.device)
        full_time_position_ids = torch.cat([full_time_position_ids, pos_pad], dim=0)
        full_height_position_ids = torch.cat([full_height_position_ids, pos_pad], dim=0)
        full_width_position_ids = torch.cat([full_width_position_ids, pos_pad], dim=0)

    num_logits_to_keep = self._llm_model.get_num_logits_to_keep()
    outputs_logits = []
    for idx in range(steps):
        start = idx * net_input_seq_len
        end = (idx + 1) * net_input_seq_len
        current_length = min(end, seq_length) - start
        output = self._llm_model.forward(
            full_inputs_embeds[:, start:end, :],
            full_time_position_ids[start:end],
            full_height_position_ids[start:end],
            full_width_position_ids[start:end],
            past_seq_length + start,
            torch.tensor([current_length], dtype=torch.int32, device=full_inputs_embeds.device),
            past_key_caches,
            past_value_caches,
        )
        logits = output if isinstance(output, torch.Tensor) else output.logits
        outputs_logits.append(logits)

    if num_logits_to_keep != 0:
        logits = outputs_logits[-1]
    else:
        logits = torch.cat(outputs_logits, dim=1)[:, :seq_length, :]

    return Qwen2VLCausalLMOutputWithPast(
        logits=logits,
        past_key_values=past_key_caches,
        rope_deltas=data_processor.rope_deltas,
    )


def _parse_bucket(bucket) -> tuple[int, int]:
    if isinstance(bucket, dict):
        return int(bucket["max_size_h"]), int(bucket["max_size_w"])
    return int(bucket[0]), int(bucket[1])


def _load_visual_bucket_manifest(args) -> tuple[dict, Path]:
    if args.visual_buckets_manifest:
        manifest_path = Path(args.visual_buckets_manifest)
    else:
        manifest_path = Path(args.config).parent / MINERU_VISUAL_BUCKETS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Static visual bucket manifest not found: {manifest_path}. "
            "Run mineru2_5_xh_export_hmonnx.py with --visual-buckets-config first."
        )
    manifest = json.load(open(manifest_path))
    _validate_manifest(manifest)
    return manifest, manifest_path.parent


def _validate_manifest(manifest: dict) -> None:
    required = {"buckets", "fallback_bucket", "patch_size", "spatial_merge_size", "temporal_patch_size"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Invalid static visual bucket manifest, missing keys: {sorted(missing)}")
    factor = int(manifest["patch_size"]) * int(manifest["spatial_merge_size"])
    for item in manifest["buckets"]:
        bucket = _parse_bucket(item)
        if bucket[0] % factor != 0 or bucket[1] % factor != 0:
            raise ValueError(f"Static visual bucket {bucket} must be divisible by {factor}")
        if "hmonnx" not in item:
            raise ValueError(f"Static visual bucket {bucket} is missing hmonnx path")


def _build_sampling_params(max_new_tokens: int | None):
    sampling_params = copy.deepcopy(DEFAULT_SAMPLING_PARAMS)
    if max_new_tokens is not None:
        for params in sampling_params.values():
            params.max_new_tokens = max_new_tokens
    return sampling_params


def _set_hmonnx_model_dtype(model, dtype: torch.dtype) -> None:
    model._dtype = dtype
    for sub_model in getattr(model, "_models", {}).values():
        sub_model._set_dtype(dtype)


def main(args):
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model: XHQwen2VLHMONNXModel = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHQwen2VLHMONNXModel", (
        f"Expected model type XHQwen2VLHMONNXModel, but got {type(hmonnx_model).__name__}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    logger.info(f"Using device: {device}, dtype: {dtype}")
    hmonnx_model.to(device)
    _set_hmonnx_model_dtype(hmonnx_model, dtype)
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True
    if args.golden:
        hmonnx_model.enable_golden = True
    if args.fast and not args.golden:
        hmonnx_model.to_fast()

    manifest, manifest_dir = _load_visual_bucket_manifest(args)
    visual_router = StaticHMONNXVisualRouter(
        hmonnx_model=hmonnx_model,
        manifest=manifest,
        manifest_dir=manifest_dir,
        device=device,
        dtype=dtype,
        fast=args.fast,
        golden=args.golden,
        auto_offload=args.auto_offload,
        logger=logger,
    )
    processor = hmonnx_model.get_tf_processor()
    fallback_bucket = visual_router.fallback_bucket
    logger.info(f"Static visual buckets: {visual_router.buckets}; fallback: {fallback_bucket}")

    max_new_tokens = 2 if args.golden else args.max_new_tokens
    client = MinerUClient(
        backend="transformers",
        model=MinerUHMONNXModelAdapter(hmonnx_model, visual_router, device, dtype),
        processor=StaticBucketProcessorAdapter(
            processor=processor,
            buckets=visual_router.buckets,
            fallback_bucket=fallback_bucket,
            patch_size=visual_router.patch_size,
            spatial_merge_size=int(manifest["spatial_merge_size"]),
            logger=logger,
            max_upscale=args.static_vit_max_upscale,
        ),
        image_analysis=args.image_analysis,
        sampling_params=_build_sampling_params(max_new_tokens),
        layout_image_size=(args.layout_image_size, args.layout_image_size),
        batch_size=1,
        use_tqdm=not args.no_tqdm,
        debug=args.debug,
    )

    contexts = [
        TimeProfiler("mineru_hmonnx_two_step_extract", logger),
        MemoryTracker(device=device, name="mineru_hmonnx_extract", logger=logger),
    ]
    with ContextManagers(contexts):
        result = client.two_step_extract(
            Image.open(args.image_path),
            image_analysis=args.image_analysis,
        )
    logger.info(f"{'-' * 20} MinerU HMONNX Output {'-' * 20}")
    logger.info(f"{result}")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--visual-buckets-manifest",
        type=str,
        default="",
        help="Defaults to mineru_visual_buckets.json beside --config.",
    )
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument(
        "--image-path",
        type=str,
        default="/data01/home/chuyuan.wei/code/xh2modelzoo/data/images/houmo_logo.jpg",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--layout-image-size", type=int, default=1036)
    parser.add_argument("--static-vit-max-upscale", type=float, default=2.0)
    parser.add_argument("--mineru-batch-size", type=int, default=1, help="Kept for compatibility; forced to 1")
    parser.add_argument("--image-analysis", action="store_true", help="Whether to enable MinerU image/chart analysis")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable MinerU progress bars")
    parser.add_argument("--golden", action="store_true", help="Whether to save golden outputs for testing.")
    parser.add_argument("--auto-offload", action="store_true", help="Whether to enable auto offload")
    args = parser.parse_args()
    main(args)
