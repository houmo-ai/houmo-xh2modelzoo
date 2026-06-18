import argparse
import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS
from transformers import AutoConfig

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

SCRIPT_DIR = Path(__file__).resolve().parent
MINERU_DIR = SCRIPT_DIR.parent
if str(MINERU_DIR) not in sys.path:
    sys.path.insert(0, str(MINERU_DIR))

from static_vit_utils import (  # noqa: E402
    DEFAULT_VISUAL_BUCKETS_CONFIG,
    StaticBucketProcessorAdapter,
    parse_bucket,
    validate_static_buckets,
)


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen2_vl import (
        XHQwen2VLModel,
        XHQwen2VLVisualModel,
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
    buckets = [parse_bucket(bucket) for bucket in visual_cfg.visual_buckets]
    if fallback_bucket not in buckets:
        buckets.append(fallback_bucket)
    validate_static_buckets(buckets, cfg.model.visual_config.patch_size)
    return visual_cfg.model, buckets, fallback_bucket


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
            processor.config.patch_size,
            max_upscale=args.static_vit_max_upscale,
            score_mode=args.static_vit_score_mode,
            alpha_down=args.static_vit_alpha_down,
            beta_up=args.static_vit_beta_up,
            gamma_pad=args.static_vit_gamma_pad,
            ref_area=args.static_vit_ref_area,
            allow_content_fallback_bucket=args.allow_content_fallback_bucket,
            logger=logger,
            add_hm_pixel_values=True,
            log_prefix="MinerU static visual bucket",
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
        default=DEFAULT_VISUAL_BUCKETS_CONFIG,
    )
    parser.add_argument("--static-vit-max-upscale", type=float, default=2.5)
    parser.add_argument(
        "--static-vit-score-mode",
        type=str,
        default="fit_padding",
        choices=["fit_padding", "ratio"],
        help="Static visual bucket routing score. ratio is disabled because tests showed poor stability; use fit_padding.",
    )
    parser.add_argument("--static-vit-alpha-down", type=float, default=10.0)
    parser.add_argument("--static-vit-beta-up", type=float, default=1.0)
    parser.add_argument("--static-vit-gamma-pad", type=float, default=3.0)
    parser.add_argument("--static-vit-ref-area", type=float, default=448 * 448)
    parser.add_argument(
        "--allow-content-fallback-bucket",
        action="store_true",
        help="Allow non-layout content crops to route to the square fallback bucket.",
    )
    parser.add_argument("--mineru-batch-size", type=int, default=1, help="Kept for compatibility; forced to 1")
    parser.add_argument("--image-analysis", action="store_true", help="Whether to enable MinerU image/chart analysis")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable MinerU progress bars")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    args = parser.parse_args()
    main(args)
