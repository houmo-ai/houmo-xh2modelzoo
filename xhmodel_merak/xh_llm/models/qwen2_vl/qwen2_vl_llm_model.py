import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, Cache
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
)
from xhquant.utils import get_xhquant_logger, log_function_call
from xhquant.utils.registry import _DMRegistryCls

from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor

from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, LLMModelState, VLLMModelMeta
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import Qwen2VLDataPreprocess
from .qwen2_vl_hmonnx_inference import XHQwen2VLHMONNXModel
from .qwen2_vl_visual_model import XHQwen2VLVisualModel
from .xh_qwen2_vl_config import XHQwen2VLModelConfig


class _Qwen2VLHFCompatible(TextLLMHFCompatible):
    def _setup(self: Qwen2VLForConditionalGeneration, xh_model: "XHQwen2VLModel"):
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
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        hm_pixel_values=None,
        **kwargs,
    ):
        # HF calls this once for the full prompt and then once per decoded token.
        # Only the decoded-token steps may drop image tensors and keep the last id.
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
            pixel_values = None
            pixel_values_videos = None
            image_grid_thw = None
            video_grid_thw = None
            hm_pixel_values = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
            "use_cache": use_cache,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "hm_pixel_values": hm_pixel_values,
        }

    def forward(
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
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        # Merak keeps the visual branch as a sibling sub-model. Convert the raw
        # image patches to image embeddings here, then scatter them into the LLM
        # prompt at image-token positions in Qwen2VLDataPreprocess.
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
        data_input = data_processor(
            {
                "input_ids": input_ids,
                "image_embeds": image_embeds,
                "past_seq_length": self._past_seq_length,
                "image_grid_thw": image_grid_thw,
            }
        )
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            past_key_caches,
            past_value_caches,
        ) = data_input

        logits = self._llm_model.forward(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            past_key_caches,
            past_value_caches,
        )

        return Qwen2VLCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_caches,
            rope_deltas=data_processor.rope_deltas,
        )


def build_qwen2_vl_hf_compatible_model(hf_model: Qwen2VLForConditionalGeneration, xh_model: "XHQwen2VLModel"):
    compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen2VLHFCompatible)
    compatible_modules.convert(hf_model, xh_model=xh_model)
    return hf_model


@register_llm_model("Qwen2VLForConditionalGeneration")
class XHQwen2VLModel(VisionLLMModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Qwen2VLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HMONNXINFERENCE_CLS = XHQwen2VLHMONNXModel
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen2_vl_hf_compatible_model)
    CONFIG_CLS = XHQwen2VLModelConfig

    def __init__(self, config: XHQwen2VLModelConfig):
        super().__init__(config)
        self.visual = XHQwen2VLVisualModel(config.visual_config)
        self.visual.config.model_name = (
            f"{self.config.model_name}_{self.visual.config.max_size_w}x{self.visual.config.max_size_h}"
        )

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        self.visual.config.work_dir = str(Path(work_dir) / "visual")

    def _wraped_post(self, hf_model):
        super()._wraped_post(hf_model)
        self.config.image_token_id = hf_model.config.image_token_id
        self.config.video_token_id = hf_model.config.video_token_id
        self.config.vision_start_token_id = hf_model.config.vision_start_token_id
        self.config.vision_end_token_id = hf_model.config.vision_end_token_id
        self.config.spatial_merge_size = hf_model.config.vision_config.spatial_merge_size

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._llm_model_impl import register_wrap_cls as qwen2_vl_register_wrap_modules

        qwen2_vl_register_wrap_modules(hf_model)
        super().init_wrap_model(hf_model)

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model.language_model

    def get_tf_processor(self):
        return self.visual.get_tf_processor()

    def to_wrap(self):
        if self._state == LLMModelState.WRAP:
            return
        if self._state != LLMModelState.NONE:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.WRAP}")
        # The visual sub-model must enter the same Merak state before the LLM
        # wrapper is built; forward() asserts this invariant for all modes.
        hf_model = self.get_native_model()
        self.visual.to_wrap(hf_model)
        self._to_wrap(hf_model)
        self._state = LLMModelState.WRAP

    def release_inference_model(self):
        super().release_inference_model()
        self.visual.release_inference_model()

    def _set_device(self, device):
        super()._set_device(device)
        self.visual._set_device(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        return Qwen2VLDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.spatial_merge_size,
        )

    def get_dummy_inputs(self):
        data_batch = super().get_dummy_inputs()
        visual_dummy_inputs = self.visual._get_dummy_inputs()
        data_batch["image_grid_thw"] = visual_dummy_inputs["image_grid_thw"]
        return data_batch

    @classmethod
    def _load_quant_weight(cls, quant_weight_path: str, native_hf_model: nn.Module, strict: bool = True) -> bool:
        return super()._load_quant_weight(quant_weight_path, native_hf_model, strict=False)

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        if model_name is None or len(model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_size_h = self.visual.config.max_size_h
        image_size_w = self.visual.config.max_size_w
        model_name = f"hmquant_{model_name}_{image_size_w}x{image_size_h}_{str_datetime}"
        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = model_name
        export_data.str_datetime = str_datetime
        return export_data

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = {
            "input_names": [
                "inputs_embeds",
                "time_position_ids",
                "height_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
            ],
            "output_names": ["logits"],
        }
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info: VLLMModelMeta) -> VLLMModelMeta:
        meta_info.image_token_id = self.config.image_token_id
        meta_info.video_token_id = self.config.video_token_id
        meta_info.vision_start_token_id = self.config.vision_start_token_id
        meta_info.vision_end_token_id = self.config.vision_end_token_id
        meta_info.spatial_merge_size = self.config.spatial_merge_size
        return meta_info

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
        self.visual.quanted_model.fixed()
        # Export visual first so the top-level VLLM metadata can embed the
        # relative visual HMONNX path consumed by XHQwen2VLHMONNXModel.
        exported_info = self.get_export_info(output_dir)
        self.config.model_name = exported_info.model_name
        visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
        self.visual.config.model_name = (
            f"{exported_info.model_name}_{self.visual.config.max_size_w}x{self.visual.config.max_size_h}"
        )
        visual_meta = self.visual.export_hmonnx(visual_output_dir)
        visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
        meta_info = cast(VLLMModelMeta, exported_info.meta)
        meta_info.visual_config = visual_meta
        self._export_hmonnx(exported_info)
        json.dump(
            meta_info.to_dict(),
            open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"),
            indent=4,
        )
        logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info

    def forward(self, *args, **kwargs):
        assert self.visual._state == self._state, (
            f"Sub model state {self.visual._state} is different from main model state {self._state}"
        )
        return super().forward(*args, **kwargs)
