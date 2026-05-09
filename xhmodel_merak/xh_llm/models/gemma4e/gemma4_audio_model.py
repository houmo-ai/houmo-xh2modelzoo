import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import torch
from torch import nn
from transformers import AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from .gemma4_processor import XHGemma4Processor
from .xh_gemma4_config import Gemma4AudioModelMeta, XHGemma4AudioConfig


class _Gemma4AudioExportBridge(nn.Module):
    def __init__(self, hf_model: Gemma4ForConditionalGeneration):
        super().__init__()
        self.audio_tower = hf_model.model.audio_tower
        self.embed_audio = hf_model.model.embed_audio

    def forward(self, input_features, input_features_mask):
        audio_outputs = self.audio_tower(
            input_features=input_features,
            attention_mask=input_features_mask,
        )
        output_mask = None
        if hasattr(audio_outputs, "last_hidden_state"):
            hidden_states = audio_outputs.last_hidden_state
            output_mask = getattr(audio_outputs, "output_mask", None)
        elif isinstance(audio_outputs, (tuple, list)):
            hidden_states = audio_outputs[0]
            if len(audio_outputs) > 1:
                output_mask = audio_outputs[1]
        else:
            hidden_states = audio_outputs

        audio_embeds = self.embed_audio(inputs_embeds=hidden_states)
        if isinstance(audio_embeds, (tuple, list)):
            audio_embeds = audio_embeds[0]
        if output_mask is None:
            return audio_embeds
        return audio_embeds, output_mask


class _Gemma4AudioProcessor(BaseVisualProcessor):
    def forward(self, data: dict) -> list[torch.Tensor]:
        assert isinstance(data, dict), "Input data should be a dictionary."
        input_features = data.get("input_features")
        input_features_mask = data.get("input_features_mask")
        assert input_features is not None and input_features_mask is not None, (
            "Gemma4 audio export requires both `input_features` and `input_features_mask`."
        )
        return (input_features, input_features_mask)


@register_llm_model("Gemma4ForConditionalGeneration_audio", master=False)
class XHGemma4AudioModel(BaseVisionModel):
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Gemma4AudioModelMeta
    CONFIG_CLS = XHGemma4AudioConfig

    def __init__(self, config: XHGemma4AudioConfig):
        super().__init__(config)
        self.config = cast(XHGemma4AudioConfig, self.config)

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        from ._audio_model_impl import register_wrap_modules

        register_wrap_modules(hf_model)
        if hf_model is not None and hasattr(hf_model, "model") and hasattr(hf_model.model, "audio_tower"):
            hf_model = _Gemma4AudioExportBridge(hf_model)
        return super().init_wrap_model(hf_model)

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(self.hf_model_dir)
        processor.config.sampling_rate = self.config.sampling_rate
        processor.config.audio_feature_length = self.config.input_feature_length
        return processor

    def _get_data_preprocessor(self) -> BaseVisualProcessor:
        return _Gemma4AudioProcessor()

    def get_dummy_inputs(self) -> Any:
        processor = self.get_tf_processor()
        model_inputs = processor(
            text="<|audio|>Transcribe this audio.",
            audio=np.zeros(16000, dtype=np.float32),
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
        )
        return {
            "input_features": model_inputs["input_features"],
            "input_features_mask": model_inputs["input_features_mask"],
        }

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self.get_dummy_inputs()
        input_names = list(dummy_inputs.keys())
        dummy_values = tuple(value.float().cpu() if value.is_floating_point() else value.cpu() for value in dummy_inputs.values())

        work_dir = self.config.work_dir
        tmp_dir_ctx = None
        if not work_dir:
            tmp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = tmp_dir_ctx.name

        try:
            onnx_file = str(Path(work_dir) / "onnx" / "gemma4_audio.onnx")
            Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(onnx_file).exists():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                    torch.onnx.export(
                        wrap_model.float().cpu(),
                        dummy_values,
                        tmp_onnx_file,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=["audio_embeds", "audio_embeds_mask"],
                        verbose=False,
                    )
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
                    from xhquant.utils.onnx_simplify import onnx_simplify

                    onnx_model_simp, check = onnx_simplify(onnx_model)
                    if check:
                        onnx_model = onnx_model_simp
                    onnx.save(
                        onnx_model,
                        onnx_file,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=f"{Path(onnx_file).stem}_external_data",
                    )
                self._wrap_model.to(self.device, self.dtype)
            else:
                logger.info(f"from cached onnx: {onnx_file}")

            onnx_model = onnx.load(onnx_file)
            return to_frontend_graph(onnx_model, FrontendType.ONNX, list(dummy_values))
        finally:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["input_features", "input_features_mask"],
            "output_names": ["audio_embeds", "audio_embeds_mask"],
        }

    def export_hmonnx(self, output_dir: str) -> Gemma4AudioModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        source_onnx_dir = Path(self.config.work_dir) / "onnx"
        target_onnx_dir = Path(output_dir) / "onnx"
        if source_onnx_dir.exists():
            if source_onnx_dir.resolve() != target_onnx_dir.resolve():
                shutil.rmtree(target_onnx_dir, ignore_errors=True)
                shutil.copytree(source_onnx_dir, target_onnx_dir)
            meta_info.onnx = str(target_onnx_dir / "gemma4_audio.onnx")
        return meta_info

    def create_export_metadata(self, output_dir: str) -> Gemma4AudioModelMeta:
        meta_info = cast(Gemma4AudioModelMeta, self.get_export_metadata_cls()())
        meta_info.sampling_rate = self.config.sampling_rate
        meta_info.feature_size = self.config.feature_size
        meta_info.input_feature_length = self.config.input_feature_length
        return meta_info
