from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from xhquant.api import FrontendType, to_frontend_graph

from ....utils import calculate_file_md5
from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from .audio_utils import emotion2vec_frame_count, normalize_padded_waveform
from .configuration_emotion2vec import EMOTION2VEC_LABELS, Emotion2vecModelMeta, XHEmotion2vecConfig
from .emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel
from .modeling_emotion2vec import load_funasr_emotion2vec_model
from .xhquant_graph import XHEmotion2vecGraphModel


class _Emotion2vecProcessor(BaseVisualProcessor):
    def __init__(self, config: XHEmotion2vecConfig):
        self.config = config

    def forward(self, data: dict) -> list[torch.Tensor]:
        waveform = data.get("waveform")
        valid_samples = data.get("valid_samples")
        if waveform is None or valid_samples is None:
            raise ValueError("emotion2vec export requires waveform and valid_samples")
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("emotion2vec preprocessing expects one padded waveform")
        valid_count = int(valid_samples.reshape(-1)[0].item())
        normalized = normalize_padded_waveform(waveform.detach().cpu().numpy()[0], valid_count)
        valid_frames = torch.tensor([emotion2vec_frame_count(valid_count)], dtype=torch.int32)
        return [torch.from_numpy(normalized).unsqueeze(0), valid_frames]


@register_llm_model("Emotion2vecForEmotionRecognition", master=True, force=True)
class XHEmotion2vecModel(BaseVisionModel):
    HF_MODEL_CLS = None
    HF_AUTO_MODEL_CLS = None
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.emotion2vec.workflow:Emotion2vecWorkflow"
    META_CLS = Emotion2vecModelMeta
    HMONNXINFERENCE_CLS = Emotion2vecHMONNXModel
    CONFIG_CLS = XHEmotion2vecConfig

    def __init__(self, config: XHEmotion2vecConfig):
        super().__init__(config)
        self.config = cast(XHEmotion2vecConfig, self.config)
        self._native_model = None

    def get_native_model(self):
        if self._native_model is None:
            if self.hf_model_dir is None:
                raise ValueError("emotion2vec requires hf_model to load the official checkpoint")
            self._native_model = load_funasr_emotion2vec_model(self.hf_model_dir)
        return self._native_model

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        native_model = hf_model if hf_model is not None else self.get_native_model()
        if hasattr(native_model, "model"):
            native_model = native_model.model
        self._wrap_model = XHEmotion2vecGraphModel.from_funasr(
            native_model,
            window_samples=self.config.window_samples,
        )
        return self._wrap_model

    def _get_data_preprocessor(self) -> BaseVisualProcessor:
        return _Emotion2vecProcessor(self.config)

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        waveform = torch.zeros(1, self.config.window_samples, dtype=torch.float32)
        valid_sample_count = self.config.window_samples
        if self.hf_model_dir is not None:
            calibration_audio = Path(self.hf_model_dir) / "example" / "test.wav"
            if calibration_audio.exists():
                import soundfile as sf

                audio, sampling_rate = sf.read(str(calibration_audio), always_2d=False)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                if sampling_rate != self.config.sampling_rate:
                    raise ValueError(
                        f"emotion2vec calibration audio must be {self.config.sampling_rate} Hz, got {sampling_rate}"
                    )
                valid_sample_count = min(audio.size, self.config.window_samples)
                waveform[0, :valid_sample_count] = torch.from_numpy(audio[:valid_sample_count])
        valid_samples = torch.tensor([valid_sample_count], dtype=torch.int32)
        return {"waveform": waveform, "valid_samples": valid_samples}

    def _to_fronted(self, wrap_model):
        export_inputs = {
            "waveform": torch.zeros(1, self.config.window_samples, dtype=torch.float32),
            "valid_frames": torch.tensor([emotion2vec_frame_count(self.config.window_samples)], dtype=torch.int32),
        }
        values = tuple(value.cpu() for value in export_inputs.values())
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_file = Path(tmp_dir) / "emotion2vec.onnx"
            torch.onnx.export(
                wrap_model.float().cpu(),
                values,
                str(onnx_file),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["waveform", "valid_frames"],
                output_names=[
                    "frame_features",
                    "frame_padding_mask",
                    "utterance_feature",
                    "probabilities",
                ],
                verbose=False,
            )
            import onnx

            onnx_model = onnx.load(str(onnx_file), load_external_data=True)
            return to_frontend_graph(onnx_model, FrontendType.ONNX, list(values))

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["waveform", "valid_frames"],
            "output_names": [
                "frame_features",
                "frame_padding_mask",
                "utterance_feature",
                "probabilities",
            ],
        }

    def create_export_metadata(self, output_dir: str) -> Emotion2vecModelMeta:
        meta_info = cast(Emotion2vecModelMeta, self.get_export_metadata_cls()())
        meta_info.sampling_rate = self.config.sampling_rate
        meta_info.window_samples = self.config.window_samples
        meta_info.feature_dim = self.config.feature_dim
        meta_info.num_labels = self.config.num_labels
        meta_info.labels = list(EMOTION2VEC_LABELS)
        meta_info.model_id = self.config.model_id
        meta_info.model_config = self.config.to_dict()
        native_model = self.get_native_model()
        if native_model.proj is None:
            raise ValueError("emotion2vec emotion-recognition export requires the official classification head")
        classification_head_file = Path(output_dir) / "quant_embedding.pt"
        classification_head_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(native_model.proj.state_dict(), classification_head_file)
        meta_info.quant_embedding = str(classification_head_file.relative_to(output_dir))
        meta_info.quant_embedding_md5 = calculate_file_md5(classification_head_file)
        if self.hf_model_dir is not None:
            calibration_audio = Path(self.hf_model_dir) / "example" / "test.wav"
            if calibration_audio.exists():
                meta_info.calibration_audio = str(calibration_audio)
        return meta_info

    def export_hmonnx(self, output_dir: str) -> Emotion2vecModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(Path(exported_hmonnx_file).relative_to(Path(output_dir)))
        return meta_info


def build_emotion2vec_model(config: XHEmotion2vecConfig) -> XHEmotion2vecModel:
    return XHEmotion2vecModel(config)
