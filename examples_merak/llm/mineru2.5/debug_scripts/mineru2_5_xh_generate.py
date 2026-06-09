import argparse
import copy
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE, smart_resize
from transformers import AutoConfig
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl import (
        XHQwen2VLModel,
        XHQwen2VLProcessor,
        XHQwen2VLVisualModel,
    )


class StaticBucketProcessorAdapter:
    """Adapt XHQwen2VLProcessor to MinerU's transformers VLM client."""

    def __init__(
        self,
        processor: "XHQwen2VLProcessor",
        buckets: list[tuple[int, int]],
        fallback_bucket: tuple[int, int],
        logger,
        max_upscale: float = 2.0,
    ) -> None:
        self.processor = processor
        self.buckets = sorted(set(buckets), key=lambda item: (item[0] * item[1], item[0], item[1]))
        self.fallback_bucket = fallback_bucket
        self.patch_size = processor.config.patch_size
        self.patch_factor = self.patch_size * SPATIAL_MERGE_SIZE
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


class StaticVisualRouter:
    def __init__(
        self,
        xh_model: "XHQwen2VLModel",
        visual_cfg_model: Any,
        buckets: list[tuple[int, int]],
        fallback_bucket: tuple[int, int],
        eval_state: LLMModelState,
        work_dir: Path,
        device: str,
        dtype: torch.dtype,
        logger,
    ) -> None:
        self.xh_model = xh_model
        self.visual_cfg_model = copy.deepcopy(visual_cfg_model)
        self.buckets = sorted(set(buckets), key=lambda item: (item[0] * item[1], item[0], item[1]))
        self.fallback_bucket = fallback_bucket
        self.eval_state = eval_state
        self.work_dir = work_dir
        self.device = device
        self.dtype = dtype
        self.patch_size = xh_model.visual.config.patch_size
        self.logger = logger
        self.visual_models: dict[tuple[int, int], XHQwen2VLVisualModel] = {}
        self._build_all()

    def activate_for_grid(self, image_grid_thw) -> tuple[int, int]:
        bucket = self._bucket_from_grid(image_grid_thw)
        visual_model = self.visual_models.get(bucket)
        if visual_model is None:
            self.logger.warning(f"Missing static visual bucket {bucket}; fallback to {self.fallback_bucket}")
            bucket = self.fallback_bucket
            visual_model = self.visual_models[bucket]
        if self.xh_model.visual is not visual_model:
            self.xh_model.visual = visual_model
        self.logger.info(f"MinerU active static visual bucket: {bucket}")
        return bucket

    def _build_all(self) -> None:
        base_bucket = (self.xh_model.visual.config.max_size_h, self.xh_model.visual.config.max_size_w)
        if base_bucket in self.buckets:
            self.visual_models[base_bucket] = self.xh_model.visual
        for bucket in self.buckets:
            if bucket in self.visual_models:
                continue
            self.visual_models[bucket] = self._build_visual_model(bucket)

    def _build_visual_model(self, bucket: tuple[int, int]) -> "XHQwen2VLVisualModel":
        max_size_h, max_size_w = bucket
        cfg_model = copy.deepcopy(self.visual_cfg_model)
        cfg_model.max_size_h = max_size_h
        cfg_model.max_size_w = max_size_w
        cfg_model.model_name = f"{cfg_model.model_name}_{max_size_h}x{max_size_w}"
        model_cfg = AutoLLMConfig.from_pretrained(cfg_model)
        bucket_dir = self.work_dir / f"static_visual_{max_size_h}x{max_size_w}"
        model_cfg.work_dir = str(bucket_dir)
        model = AutoLLMModel.from_pretrained(config=model_cfg)
        model.set_state(self.eval_state)
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        self.logger.info(f"Built static visual bucket {bucket}")
        return model

    def _bucket_from_grid(self, image_grid_thw) -> tuple[int, int]:
        if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.numel() == 0:
            return self.fallback_bucket
        grid_h = int(image_grid_thw[0, 1].item())
        grid_w = int(image_grid_thw[0, 2].item())
        return (grid_h * self.patch_size, grid_w * self.patch_size)


class MinerUXHModelAdapter:
    """Expose the small HF surface that mineru_vl_utils.TransformersVlmClient expects."""

    def __init__(
        self,
        xh_model: "XHQwen2VLModel",
        visual_router: StaticVisualRouter,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.xh_model = xh_model
        self.visual_router = visual_router
        self._device = device
        self._dtype = dtype
        self.config = AutoConfig.from_pretrained(xh_model.hf_model_dir)
        if not hasattr(self.config, "max_position_embeddings"):
            self.config.max_position_embeddings = xh_model.config.context_max_length

    def __getattr__(self, name):
        return getattr(self.xh_model, name)

    @property
    def device(self):
        return torch.device(self._device)

    @property
    def dtype(self):
        return self._dtype

    def generate(self, *args, **kwargs):
        self._prepare_xh_visual_inputs(kwargs)
        self.visual_router.activate_for_grid(kwargs.get("image_grid_thw"))
        with ContextManagers([LLMInferenceContextManager(self.xh_model), torch.no_grad()]):
            return self.xh_model.generate(*args, **kwargs)

    @staticmethod
    def _prepare_xh_visual_inputs(kwargs):
        hm_pixel_values = kwargs.get("hm_pixel_values")
        pixel_values = kwargs.get("pixel_values")
        if hm_pixel_values is None and isinstance(pixel_values, torch.Tensor):
            kwargs["hm_pixel_values"] = [pixel_values.contiguous().float()]
        elif isinstance(hm_pixel_values, torch.Tensor):
            kwargs["hm_pixel_values"] = [hm_pixel_values.contiguous().float()]


def _build_sampling_params(max_new_tokens: int | None):
    sampling_params = copy.deepcopy(DEFAULT_SAMPLING_PARAMS)
    if max_new_tokens is not None:
        for params in sampling_params.values():
            params.max_new_tokens = max_new_tokens
    return sampling_params


def _load_visual_bucket_config(args, cfg) -> tuple[Any, list[tuple[int, int]], tuple[int, int]]:
    visual_cfg = Config.fromfile(args.visual_buckets_config)
    if args.model:
        visual_cfg.model.hf_model = args.model
    fallback_bucket = (cfg.model.visual_config.max_size_h, cfg.model.visual_config.max_size_w)
    buckets = [_parse_bucket(bucket) for bucket in visual_cfg.visual_buckets]
    if fallback_bucket not in buckets:
        buckets.append(fallback_bucket)
    _validate_buckets(buckets, cfg.model.visual_config.patch_size)
    return visual_cfg.model, buckets, fallback_bucket


def _parse_bucket(bucket) -> tuple[int, int]:
    if isinstance(bucket, dict):
        return int(bucket["max_size_h"]), int(bucket["max_size_w"])
    return int(bucket[0]), int(bucket[1])


def _validate_buckets(buckets: list[tuple[int, int]], patch_size: int) -> None:
    factor = patch_size * SPATIAL_MERGE_SIZE
    for bucket_h, bucket_w in buckets:
        if bucket_h <= 0 or bucket_w <= 0:
            raise ValueError(f"Invalid static visual bucket {(bucket_h, bucket_w)}")
        if bucket_h % factor != 0 or bucket_w % factor != 0:
            raise ValueError(f"Static visual bucket {(bucket_h, bucket_w)} must be divisible by {factor}")


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type
    if args.debug:
        cfg_name += "_debug"

    work_dir = Path("./work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / f"generate_{eval_type}.log"), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    if args.model:
        cfg.model.hf_model = args.model
        cfg.model.visual_config.hf_model = args.model
    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / Path(args.config).name)

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}, dtype: {dtype}")
    model_cfg: XHQwen2VLModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir)
    model_cfg.visual_config.work_dir = str(work_dir / "visual")

    xh_model: XHQwen2VLModel = AutoLLMModel.from_pretrained(config=model_cfg)
    eval_state = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_state)

    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    processor: XHQwen2VLProcessor = xh_model.get_tf_processor()
    visual_cfg_model, buckets, fallback_bucket = _load_visual_bucket_config(args, cfg)
    logger.info(f"Static visual buckets: {buckets}; fallback: {fallback_bucket}")
    visual_router = StaticVisualRouter(
        xh_model=xh_model,
        visual_cfg_model=visual_cfg_model,
        buckets=buckets,
        fallback_bucket=fallback_bucket,
        eval_state=eval_state,
        work_dir=work_dir,
        device=device,
        dtype=dtype,
        logger=logger,
    )
    client = MinerUClient(
        backend="transformers",
        model=MinerUXHModelAdapter(xh_model, visual_router, device, dtype),
        processor=StaticBucketProcessorAdapter(
            processor,
            buckets,
            fallback_bucket,
            logger,
            max_upscale=args.static_vit_max_upscale,
        ),
        image_analysis=args.image_analysis,
        sampling_params=_build_sampling_params(args.max_new_tokens),
        layout_image_size=(args.layout_image_size, args.layout_image_size),
        batch_size=1,
        use_tqdm=not args.no_tqdm,
        debug=args.debug,
    )

    contexts = [
        TimeProfiler("mineru_two_step_extract", logger),
        MemoryTracker(device=device, name="mineru_extract", logger=logger),
    ]
    with ContextManagers(contexts):
        result = client.two_step_extract(
            Image.open(args.image_path),
            image_analysis=args.image_analysis,
        )
    logger.info(f"{'-' * 20} MinerU {eval_state} Output {'-' * 20}")
    logger.info(f"{result}")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py",
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--eval-type", type=str, default="wrap", choices=LLMModelState.get_all_values())
    parser.add_argument(
        "--image-path",
        type=str,
        default="/data01/home/chuyuan.wei/code/xh2modelzoo/data/images/houmo_logo.jpg",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--layout-image-size", type=int, default=1036)
    parser.add_argument(
        "--visual-buckets-config",
        type=str,
        default="configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py",
    )
    parser.add_argument("--static-vit-max-upscale", type=float, default=2.0)
    parser.add_argument("--mineru-batch-size", type=int, default=1, help="Kept for compatibility; forced to 1")
    parser.add_argument("--image-analysis", action="store_true", help="Whether to enable MinerU image/chart analysis")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable MinerU progress bars")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    args = parser.parse_args()
    main(args)
