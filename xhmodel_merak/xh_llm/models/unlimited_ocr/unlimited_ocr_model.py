import copy
import json
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from xhquant.api import get_xhquant_logger, ptq_quantize, PrecisionMode, to_frontend_graph, to_quant_graph
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import LLMModelState, ModelSwitcher, VLLMModelMeta
from ...utils import unfold_args
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import UnlimitedOCRDataPreprocess
from .modeling_unlimitedocr import UnlimitedOCRConfig, UnlimitedOCRForCausalLM
from .modeling_unlimitedocr_patch import unlimited_ocr_patch
from .unlimited_ocr_hmonnx_inference import XHUnlimitedOCRHMONNXModel
from .xh_unlimited_ocr_config import XHUnlimitedOCRModelConfig
from .unlimited_ocr_visual_model import XHUnlimitedOCRVisualModel


def _copy_model_shared_params(model: nn.Module) -> nn.Module:
    """Deep-copy module structure while sharing parameters and buffers."""
    memo: dict[int, Any] = {}
    for param in model.parameters():
        if id(param) not in memo:
            memo[id(param)] = nn.Parameter(param.data, requires_grad=param.requires_grad)
    for buffer in model.buffers():
        if id(buffer) not in memo:
            memo[id(buffer)] = buffer
    return copy.deepcopy(model, memo)


class _UnlimitedOCRHFCompatible(TextLLMHFCompatible):
    """HF-compatible wrapper that runs the visual branch, scatters image
    embeddings into the prompt, then delegates to the chunked text forward.

    Base/no-crop only. The visual sibling sub-model turns the global-view image
    tensor into image embeddings; ``UnlimitedOCRDataPreprocess`` scatters them at
    ``<image>`` positions. Decode steps drop the image tensors and keep the last
    token id.
    """

    def _setup(self, xh_model: "XHUnlimitedOCRModel"):
        m = super()._setup(xh_model)
        if m is not None:
            if hasattr(m, "model"):
                del m.model
            if hasattr(m, "lm_head"):
                del m.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        images=None,
        images_ori=None,
        images_crop=None,
        images_seq_mask=None,
        images_spatial_crop=None,
        **kwargs,
    ):
        # HF calls this once for the full prompt and then once per decoded token.
        # Only decoded-token steps drop the image tensors and keep the last id.
        has_past = False
        if past_key_values is not None:
            if cache_position is not None:
                has_past = cache_position[0].item() != 0
            elif hasattr(past_key_values, "get_seq_length"):
                has_past = past_key_values.get_seq_length() != 0
            else:
                has_past = True

        if has_past:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
            images = None
            images_ori = None
            images_crop = None
            images_seq_mask = None
            images_spatial_crop = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
            "use_cache": use_cache,
            "images": images,
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

    def _run_visual(
        self,
        images_ori: torch.Tensor,
        images_crop: Optional[torch.Tensor] = None,
        images_spatial_crop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        visual = self._llm_model.visual
        visual_config = getattr(visual, "config", None)
        crop_mode = bool(getattr(visual_config, "crop_mode", False))
        image_embeds = []
        if crop_mode:
            # crop/gundam: single image, global view + dynamic local crops.
            image_ori = images_ori[0].unsqueeze(0).type(visual.dtype).to(visual.device)
            if images_spatial_crop is not None and images_spatial_crop.shape[0] > 0:
                width_crop_num = int(images_spatial_crop[0][0])
                height_crop_num = int(images_spatial_crop[0][1])
            else:
                width_crop_num = height_crop_num = 1
            crop = images_crop
            if crop is not None and crop.shape[0] > 0:
                crop = crop.type(visual.dtype).to(visual.device)
            else:
                crop = torch.zeros(
                    (0, 3, visual.config.image_size, visual.config.image_size),
                    dtype=visual.dtype,
                    device=visual.device,
                )
            out = visual.forward_crop(image_ori, crop, width_crop_num, height_crop_num)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out.reshape(-1, out.shape[-1])
        for image_i in images_ori:
            out = visual.forward(image_i.unsqueeze(0).type(visual.dtype).to(visual.device))
            if isinstance(out, (list, tuple)):
                out = out[0]
            image_embeds.append(out.reshape(-1, out.shape[-1]))
        return torch.cat(image_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,
        images_ori: Optional[torch.Tensor] = None,
        images_crop: Optional[torch.Tensor] = None,
        images_seq_mask: Optional[torch.Tensor] = None,
        images_spatial_crop: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        embed_tokens = self.get_input_embeddings()
        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids)

        # Prefill step: turn raw global-view image(s) into image embeddings and
        # scatter them into the prompt at <image> positions.
        if images_ori is None and images is not None:
            # Original modeling packs images as [(images_crop, images_ori)].
            if isinstance(images, (list, tuple)) and len(images) > 0:
                first = images[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    images_ori = first[1]
                else:
                    images_ori = images if isinstance(images, torch.Tensor) else torch.stack(list(images), dim=0)
            elif isinstance(images, torch.Tensor):
                images_ori = images

        if images_ori is not None and images_ori.shape[0] > 0:
            image_embeds = self._run_visual(images_ori, images_crop, images_spatial_crop).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            if images_seq_mask is None:
                images_seq_mask = input_ids == self._llm_model.meta_info.image_token_id
            image_mask = images_seq_mask.to(inputs_embeds.device).unsqueeze(-1).expand_as(inputs_embeds)
            n_image_tokens = int(images_seq_mask.sum().item())
            n_image_features = int(image_embeds.shape[0])
            if n_image_tokens != n_image_features:
                raise ValueError(
                    "Unlimited-OCR image features and image tokens do not match: "
                    f"tokens={n_image_tokens}, features={n_image_features}."
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


def build_unlimited_ocr_hf_compatible_model(hf_model: UnlimitedOCRForCausalLM, xh_model: "XHUnlimitedOCRModel"):
    from transformers import GenerationConfig, GenerationMixin

    # Unlimited-OCR's HF class does not inherit GenerationMixin (transformers
    # >=4.50 no longer auto-injects it into PreTrainedModel). generate() lives on
    # the HF-compatible wrapper, so ensure GenerationMixin is in the MRO.
    hf_model_cls = type(hf_model)
    if not issubclass(hf_model_cls, GenerationMixin):
        patched_cls = type(hf_model_cls.__name__, (hf_model_cls, GenerationMixin), {})
        hf_model.__class__ = patched_cls
        hf_model_cls = patched_cls

    # The empty HF model may have no generation_config (exported hf_config has no
    # generation_config.json). Provide a default so HF generate() can run.
    if getattr(hf_model, "generation_config", None) is None:
        hf_model.generation_config = GenerationConfig.from_model_config(hf_model.config)

    compatible_modules = _DMRegistryCls("XHCompatible")
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _UnlimitedOCRHFCompatible)
    compatible_modules.convert(hf_model, xh_model=xh_model)
    return hf_model


@register_llm_model("UnlimitedOCRForCausalLM")
class XHUnlimitedOCRModel(VisionLLMModel):
    CONFIG_CLS = XHUnlimitedOCRModelConfig
    HF_MODEL_CLS = UnlimitedOCRForCausalLM
    HF_AUTO_MODEL_CLS = UnlimitedOCRForCausalLM
    HMONNXINFERENCE_CLS = XHUnlimitedOCRHMONNXModel
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.unlimited_ocr.workflow:XHUnlimitedOCRWorkflow"
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_unlimited_ocr_hf_compatible_model)

    def __init__(self, config: XHUnlimitedOCRModelConfig):
        super().__init__(config)
        self.config = config
        self.visual = XHUnlimitedOCRVisualModel(config.visual_config)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs: Any):
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = cls.get_hf_model_dtype()
        kwargs.setdefault("trust_remote_code", False)
        native_model = UnlimitedOCRForCausalLM.from_pretrained(hf_model_dir, **kwargs)
        native_model = cls.untied_weights(native_model)
        if quant_weight is not None and len(quant_weight) > 0:
            cls._load_quant_weight(quant_weight, native_model)
        return unlimited_ocr_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir: str, **kwargs: Any):
        from accelerate import init_empty_weights
        from transformers import GenerationConfig

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            no_init_weights = init_empty_weights

        kwargs.pop("torch_dtype", None)
        kwargs.pop("dtype", None)
        kwargs.pop("trust_remote_code", None)
        config = UnlimitedOCRConfig.from_pretrained(hf_model_dir)
        with no_init_weights(), init_empty_weights():
            native_model = UnlimitedOCRForCausalLM(config)
            try:
                native_model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
            except OSError:
                pass
        return unlimited_ocr_patch(native_model)

    def _get_language_model(self, hf_model: Any) -> Any:
        if not hasattr(hf_model, "model"):
            raise TypeError(f"Unlimited-OCR HF model should expose .model, got {type(hf_model)!r}")
        return hf_model.model

    def init_wrap_model(self, hf_model: Any = None):
        from ._llm_model_impl import register_wrap_cls

        register_wrap_cls(hf_model)
        super().init_wrap_model(hf_model)

    def to_wrap(self):
        if self._state == LLMModelState.WRAP:
            return
        if self._state != LLMModelState.NONE:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.WRAP}")

        hf_model = self.get_native_model()
        self._to_wrap(hf_model)
        self._state = LLMModelState.WRAP

    def set_prefill(self):
        if self._state == LLMModelState.FRONTED and hasattr(self._frontend_model, "set_activate_model"):
            self._frontend_model.set_activate_model("prefill")
        elif self._state in [
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
            LLMModelState.QUANTED_DISABLE,
        ] and hasattr(self._quanted_model, "set_activate_model"):
            self._quanted_model.set_activate_model("prefill")
        super().set_prefill()

    def set_decode(self):
        if self._state == LLMModelState.FRONTED and hasattr(self._frontend_model, "set_activate_model"):
            self._frontend_model.set_activate_model("decode")
        elif self._state in [
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
            LLMModelState.QUANTED_DISABLE,
        ] and hasattr(self._quanted_model, "set_activate_model"):
            self._quanted_model.set_activate_model("decode")
        super().set_decode()

    def _to_fronted_single(self, wrap_model):
        with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
            data_processor = self.get_data_preprocessor()
            dummy_inputs = self.get_dummy_inputs()
            inputs = data_processor(dummy_inputs)
            if not isinstance(inputs, (list, tuple)):
                raise TypeError(f"Processed dummy inputs should be list or tuple, got {type(inputs)}.")
            return to_frontend_graph(wrap_model, self.frontend_type, list(inputs))

    def _to_fronted(self, wrap_model):
        stage = "prefill"
        try:
            self.set_prefill()
            prefill_wrap_model = wrap_model
            decode_wrap_model = _copy_model_shared_params(wrap_model)

            self._wrap_model = prefill_wrap_model
            prefill_frontend_model = self._to_fronted_single(prefill_wrap_model)

            stage = "decode"
            self._wrap_model = decode_wrap_model
            self.set_decode()
            decode_frontend_model = self._to_fronted_single(decode_wrap_model)

            stage = "switcher"
            self._wrap_model = prefill_wrap_model
            self.set_prefill()
            frontend_model = ModelSwitcher({"prefill": prefill_frontend_model, "decode": decode_frontend_model})
            frontend_model.set_activate_model("prefill")
            return frontend_model
        except Exception as exc:
            raise RuntimeError(f"Unlimited-OCR LLM to_fronted failed during {stage} stage: {exc}") from exc

    def _quantize_frontend_model(self, frontend_model, state, calibration_batches=None):
        target_device = self.config.chip_arch
        quant_cfg = self.get_quant_cfg()
        quanted_model = to_quant_graph(frontend_model, target_device, quant_cfg)
        if state in [LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_ALIGNED]:
            if calibration_batches is None:
                with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
                    data_processor = self.get_data_preprocessor()
                    dummy_inputs = self.get_dummy_inputs()
                    assert isinstance(dummy_inputs, (dict,)), (
                        f"Dummy inputs should be a dictionary of tensors, but get {type(dummy_inputs)}."
                    )
                    calib_data = data_processor(dummy_inputs)
                    assert isinstance(calib_data, (list, tuple)), (
                        f"Processed dummy inputs should be a list or tuple of tensors, but get {type(calib_data)}."
                    )
                    calibration_batches = [unfold_args(calib_data)]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ptq_quantize(
                quanted_model,
                calibration_batches,
                PrecisionMode.ALIGNED if state == LLMModelState.QUANTED_ALIGNED else PrecisionMode.FAST,
                [device],
                auto_release_unused_parameters=True,
            )
        elif state == LLMModelState.QUANTED_DISABLE:
            pass
        else:
            raise ValueError(f"Invalid quantization state: {state}")
        return quanted_model

    def _to_quanted(self, frontend_model, state):
        stage = "prefill"
        try:
            prefill_frontend_model = frontend_model.prefill if hasattr(frontend_model, "prefill") else frontend_model
            self.set_prefill()
            calib_batches = None
            images = self._collect_calib_images()
            if images and self._llm_prefill:
                logger = get_xhquant_logger()
                logger.info(f"Unlimited-OCR PTQ using {len(images)} real calibration image(s).")
                calib_batches = self._build_real_prefill_calib(images)
            prefill_quanted_model = self._quantize_frontend_model(prefill_frontend_model, state, calib_batches)

            stage = "decode"
            decode_frontend_model = frontend_model.decode if hasattr(frontend_model, "decode") else frontend_model
            self.set_decode()
            decode_calib_batches = None
            if images:
                logger = get_xhquant_logger()
                logger.info(f"Unlimited-OCR PTQ using real decode calibration from {len(images)} image(s).")
                decode_calib_batches = self._build_real_decode_calib(images)
            decode_quanted_model = self._quantize_frontend_model(decode_frontend_model, state, decode_calib_batches)

            stage = "switcher"
            self.set_prefill()
            quanted_model = ModelSwitcher({"prefill": prefill_quanted_model, "decode": decode_quanted_model})
            quanted_model.set_activate_model("prefill")
            return quanted_model
        except Exception as exc:
            raise RuntimeError(f"Unlimited-OCR LLM quant alignment failed during {stage} stage: {exc}") from exc

    def _get_data_preprocessor(self):
        visual_config = self.config.visual_config
        return UnlimitedOCRDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            image_token_id=self.config.image_token_id,
            image_size=visual_config.image_size,
            patch_size=visual_config.patch_size,
            downsample_ratio=visual_config.downsample_ratio,
            crop_mode=visual_config.crop_mode,
            pad_token_id=self.pad_token_id if self.pad_token_id is not None else 0,
        )

    def _collect_calib_images(self) -> list[str]:
        calib_config = getattr(self.config, "calib_config", None)
        if not calib_config or not calib_config.get("enable", True):
            return []
        images: list[str] = []
        for img in calib_config.get("images", []) or []:
            if Path(img).is_file():
                images.append(str(img))
        image_dir = calib_config.get("image_dir")
        if image_dir and Path(image_dir).is_dir():
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            for p in sorted(Path(image_dir).iterdir()):
                if p.suffix.lower() in exts:
                    images.append(str(p))
        num_samples = int(calib_config.get("num_samples", len(images)))
        return images[:num_samples] if num_samples > 0 else images

    def _get_calib_prompts(self) -> list[str]:
        calib_config = self.config.calib_config or {}
        prompts = calib_config.get("prompts")
        if prompts is None:
            prompts = [calib_config.get("prompt", "<image>\\nFree OCR. ")]
        elif isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(prompt) for prompt in prompts if str(prompt)]
        return prompts or ["<image>\\nFree OCR. "]

    def _build_real_prefill_calib(self, images: list[str]) -> list:
        """Build real prefill calibration batches from document images.

        Each batch is a 5-tuple matching the LLM wrap forward signature, with
        real visual features scattered at <image> positions. The base image
        prompt (~278 tokens) is chunked to the prefill graph length, mirroring
        the runtime chunked prefill.
        """
        from transformers import AutoTokenizer

        from .unlimited_ocr_processor import XHUnlimitedOCRProcessor

        visual_config = self.config.visual_config
        calib_config = self.config.calib_config or {}
        prompts = self._get_calib_prompts()

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        processor = XHUnlimitedOCRProcessor(
            tokenizer,
            image_token_id=self.config.image_token_id,
            image_size=visual_config.image_size,
            base_size=visual_config.base_size,
            patch_size=visual_config.patch_size,
            downsample_ratio=visual_config.downsample_ratio,
            crop_mode=False,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_len = int(self.wrap_cfg.input_sequence_length)
        eager_visual = self.visual._get_eager_visual(device, self.visual.dtype)

        batches = []
        embed_tokens = self.embed_tokens
        embed_device = next(embed_tokens.parameters()).device
        for image_path in images:
            for prompt in prompts:
                inputs = processor.process(prompt, image_path, device=device)
                with torch.no_grad():
                    image_embeds = eager_visual.forward(inputs["images_ori"].to(self.visual.dtype))
                image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

                # Scatter real visual features into the full prompt embeddings,
                # then emit every runtime prefill chunk. The old calibration used
                # only the first 256-token chunk, while base/no-crop prompts are
                # usually ~278 tokens and runtime runs a second short chunk.
                input_ids = inputs["input_ids"].to(embed_device)
                images_seq_mask = inputs["images_seq_mask"].to(embed_device)
                inputs_embeds = embed_tokens(input_ids)
                mask = images_seq_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(
                    mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                )
                full_length = int(input_ids.shape[1])
                for start in range(0, full_length, seq_len):
                    current_len = min(seq_len, full_length - start)
                    chunk = inputs_embeds[:, start : start + current_len, :]
                    if chunk.shape[1] < seq_len:
                        pad = embed_tokens(
                            torch.zeros(1, seq_len - chunk.shape[1], dtype=torch.long, device=embed_device)
                        )
                        chunk = torch.cat([chunk, pad], dim=1)

                    with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
                        past_key_caches = self.past_key_caches
                        past_value_caches = self.past_value_caches
                        calib = unfold_args(
                            [
                                chunk,
                                torch.tensor([start], dtype=torch.int32, device=embed_device),
                                torch.tensor([current_len], dtype=torch.int32, device=embed_device),
                                past_key_caches,
                                past_value_caches,
                            ]
                        )
                    batches.append(calib)
        return batches

    def _build_real_decode_calib(self, images: list[str]) -> list:
        """Build decode calibration batches from real prompt continuations."""
        from transformers import AutoTokenizer

        from .unlimited_ocr_processor import XHUnlimitedOCRProcessor

        visual_config = self.config.visual_config
        calib_config = self.config.calib_config or {}
        prompts = self._get_calib_prompts()
        decode_steps = int(calib_config.get("decode_steps", 8))
        max_images = int(calib_config.get("decode_num_samples", min(4, len(images))))
        if decode_steps <= 0 or max_images == 0:
            return []

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        processor = XHUnlimitedOCRProcessor(
            tokenizer,
            image_token_id=self.config.image_token_id,
            image_size=visual_config.image_size,
            base_size=visual_config.base_size,
            patch_size=visual_config.patch_size,
            downsample_ratio=visual_config.downsample_ratio,
            crop_mode=False,
        )
        hf_model = type(self).get_hf_model(self.hf_model_dir, dtype=dtype).to(device).eval()
        from transformers import GenerationConfig, GenerationMixin

        hf_model_cls = type(hf_model)
        if not issubclass(hf_model_cls, GenerationMixin):
            patched_cls = type(
                f"{hf_model_cls.__name__}WithGeneration",
                (hf_model_cls, GenerationMixin),
                {"__module__": hf_model_cls.__module__},
            )
            hf_model.__class__ = patched_cls
        if getattr(hf_model, "generation_config", None) is None:
            try:
                hf_model.generation_config = GenerationConfig.from_pretrained(self.hf_model_dir)
            except OSError:
                hf_model.generation_config = GenerationConfig.from_model_config(hf_model.config)
        original_sliding_window = getattr(hf_model.config, "sliding_window", None)
        original_sliding_window_size = getattr(hf_model.config, "sliding_window_size", None)
        hf_model.config._ring_window = original_sliding_window_size or original_sliding_window
        hf_model.config.sliding_window = None

        batches = []
        for image_path in images[:max_images]:
            for prompt in prompts:
                inputs = processor.process(prompt, image_path, device=device)
                images_ori = inputs["images_ori"].to(dtype)
                images_crop = torch.zeros(
                    (1, 3, visual_config.image_size, visual_config.image_size), device=device, dtype=dtype
                )
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    output_ids = hf_model.generate(
                        input_ids=inputs["input_ids"],
                        images=[(images_crop, images_ori)],
                        images_seq_mask=inputs["images_seq_mask"],
                        images_spatial_crop=inputs["images_spatial_crop"],
                        max_new_tokens=decode_steps,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated = output_ids[0, inputs["input_ids"].shape[1] :]
                for step, token_id in enumerate(generated.tolist()):
                    past_seq_length = int(inputs["input_ids"].shape[1] + step)
                    data = {
                        "input_ids": torch.tensor([[int(token_id)]], dtype=torch.long, device=device),
                        "past_seq_length": past_seq_length,
                    }
                    with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
                        calib = unfold_args(self.get_data_preprocessor()(data))
                    batches.append(calib)
        hf_model.config.sliding_window = original_sliding_window
        return batches

    def _build_calib_data(self):
        images = self._collect_calib_images()
        if images and self._llm_prefill:
            logger = get_xhquant_logger()
            logger.info(
                f"Unlimited-OCR PTQ using {len(images)} real calibration image(s) "
                f"and {len(self._get_calib_prompts())} prompt(s) for prefill."
            )
            return self._build_real_prefill_calib(images)
        if images and not self._llm_prefill:
            logger = get_xhquant_logger()
            logger.info(
                f"Unlimited-OCR PTQ using real decode calibration from {len(images)} image(s), "
                f"{len(self._get_calib_prompts())} prompt(s)."
            )
            return self._build_real_decode_calib(images)
        return []

    def forward(self, *args: Any, **kwargs: Any):
        if kwargs.get("images") is not None:
            raise NotImplementedError("Pass visual outputs as image_embeds; raw image forward is handled by the visual submodel.")
        return super().forward(*args, **kwargs)

    def _get_hidden_size_for_dummy(self) -> int:
        if self.embed_tokens is not None and hasattr(self.embed_tokens, "embedding_dim"):
            return int(self.embed_tokens.embedding_dim)
        if hasattr(self.config, "hidden_size"):
            return int(self.config.hidden_size)
        if self.hf_model_dir is not None:
            try:
                return int(UnlimitedOCRConfig.from_pretrained(self.hf_model_dir).hidden_size)
            except Exception:
                pass
        return 2048

    def get_text_prefill_dummy_inputs(self):
        return {
            "input_ids": torch.randint(0, 100, (1, self.wrap_cfg.prefill_chunk_length), dtype=torch.long),
            "past_seq_length": 0,
        }

    def get_image_prefill_dummy_inputs(self):
        visual_config = self.config.visual_config
        image_grid_size = visual_config.image_size // visual_config.patch_size // visual_config.downsample_ratio
        image_token_count = (image_grid_size + 1) * image_grid_size + 1
        text_prefix_len = 1
        text_suffix_len = 1
        seq_length = text_prefix_len + image_token_count + text_suffix_len
        if seq_length > self.wrap_cfg.prefill_chunk_length:
            raise ValueError(
                f"Unlimited-OCR image prefill dummy length {seq_length} exceeds prefill_chunk_length "
                f"{self.wrap_cfg.prefill_chunk_length}."
            )

        input_ids = torch.full((1, seq_length), self.config.image_token_id, dtype=torch.long)
        input_ids[0, 0] = 1
        input_ids[0, -1] = 2
        images_seq_mask = torch.zeros((1, seq_length), dtype=torch.bool)
        images_seq_mask[:, text_prefix_len : text_prefix_len + image_token_count] = True
        image_embeds = torch.zeros((image_token_count, self._get_hidden_size_for_dummy()), dtype=torch.float32)
        images_spatial_crop = torch.ones((1, 2), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "past_seq_length": 0,
        }

    def get_prefill_dummy_inputs(self):
        if bool(self.wrap_cfg.get("image_prefill_dummy", False)):
            return self.get_image_prefill_dummy_inputs()
        return self.get_text_prefill_dummy_inputs()

    def get_decode_dummy_inputs(self):
        return {
            "input_ids": torch.randint(0, 100, (1, 1), dtype=torch.long),
            "past_seq_length": self.config.prefill_chunk_length,
        }

    def get_dummy_inputs(self, *, image_prefill: bool = False):
        if image_prefill:
            return self.get_image_prefill_dummy_inputs()
        return super().get_dummy_inputs()

    def get_export_cfg(self) -> dict[str, list[str]]:
        return super().get_export_cfg()

    def export_llm_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._fix_quanted_model_for_export()

        exported_info = self.get_export_info(output_dir)
        self._export_hmonnx(exported_info)
        meta_info = cast(VLLMModelMeta, exported_info.meta)
        meta_path = Path(exported_info.exported_dir) / "golden_meta_info.json"
        with open(meta_path, "w") as f:
            json.dump(meta_info.to_dict(), f, indent=4)
        logger.info(f"Exporting LLM completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info

    def export_visual_hmonnx(self, output_dir: str):
        return self.visual.export_hmonnx(output_dir)

    def _ensure_hmonnx_export_supported(self):
        visual_config = self.config.visual_config
        if not visual_config.hmonnx_export:
            raise NotImplementedError(
                "Unlimited-OCR gundam/crop configs are not HMONNX-exportable because dynamic crop counts "
                "break static graph shape assumptions; "
                f"export_mode={visual_config.export_mode!r}, crop_mode={visual_config.crop_mode}, "
                f"hmonnx_export={visual_config.hmonnx_export}. "
                "Use the base/no-crop config for HMONNX export."
            )
        if self.hf_model_dir is None or not Path(self.hf_model_dir).exists():
            raise FileNotFoundError(
                f"Unlimited-OCR HF checkpoint path does not exist: {self.hf_model_dir}. "
                "Set UNLIMITED_OCR_HF_MODEL or update cfg.model.hf_model before export."
            )

    def _fix_quanted_model_for_export(self):
        if isinstance(self._quanted_model, ModelSwitcher):
            self._quanted_model.prefill.fixed()
            self._quanted_model.decode.fixed()
            return
        self._quanted_model.fixed()

    def _extra_export_metadata(self, output_dir: str, meta_info):
        visual_config = self.config.visual_config
        meta_info.image_token_id = self.config.image_token_id
        meta_info.visual_export_mode = visual_config.export_mode
        meta_info.visual_crop_mode = visual_config.crop_mode
        meta_info.visual_image_size = visual_config.image_size
        meta_info.visual_base_size = visual_config.base_size
        meta_info.visual_patch_size = visual_config.patch_size
        meta_info.visual_downsample_ratio = visual_config.downsample_ratio
        return meta_info

    def _make_visual_hmonnx_relative(self, meta_info: VLLMModelMeta, exported_dir: str) -> None:
        visual_meta = meta_info.visual_config
        if visual_meta is None or getattr(visual_meta, "hmonnx", None) is None:
            return
        try:
            visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_dir).as_posix())
        except ValueError:
            visual_meta.hmonnx = str(Path(visual_meta.hmonnx).as_posix())

    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        self._ensure_hmonnx_export_supported()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()

        self._fix_quanted_model_for_export()
        self.visual.quanted_model.fixed()

        exported_info = self.get_export_info(output_dir)
        self.config.model_name = exported_info.model_name
        visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
        self.visual.config.model_name = f"{exported_info.model_name}_visual"

        try:
            visual_meta = self.visual.export_hmonnx(visual_output_dir)
            meta_info = cast(VLLMModelMeta, exported_info.meta)
            assert isinstance(meta_info, VLLMModelMeta), f"meta_info expected VLLMModelMeta, but got {type(meta_info)}"
            meta_info.visual_config = visual_meta
            self._make_visual_hmonnx_relative(meta_info, exported_info.exported_dir)
            self._export_hmonnx(exported_info)

            meta_path = Path(exported_info.exported_dir) / "golden_meta_info.json"
            with open(meta_path, "w") as f:
                json.dump(meta_info.to_dict(), f, indent=4)
            logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
            return meta_info
        except Exception as exc:
            raise RuntimeError(f"Unlimited-OCR export_hmonnx failed: {exc}") from exc

    @property
    def state(self) -> LLMModelState:
        return self._state
