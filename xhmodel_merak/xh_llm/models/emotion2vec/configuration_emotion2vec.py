from __future__ import annotations

import json
from pathlib import Path

from xhmodel_merak.configuration_utils import BaseAttrDict, BaseConfig
from xhmodel_merak.xh_llm.types import BaseLLMModelConfig


EMOTION2VEC_LABELS = (
    "生气/angry",
    "unuse_0",
    "unuse_1",
    "开心/happy",
    "中立/neutral",
    "unuse_2",
    "难过/sad",
    "unuse_3",
    "<unk>",
)


class XHEmotion2vecConfig(BaseLLMModelConfig):
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str = "Emotion2vecForEmotionRecognition",
        hf_model: str | None = None,
        model_id: str = "iic/emotion2vec_plus_large",
        sampling_rate: int = 16000,
        window_samples: int = 256000,
        feature_dim: int = 1024,
        num_labels: int = 9,
        quant_scheme: dict | None = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int = 1,
        mix_search: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if quant_scheme is None:
            quant_scheme = {"quant_type": "w8a8h1_sefp"}
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            hf_model=hf_model,
            quant_scheme=quant_scheme,
            batch_size=batch_size,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            num_logits_to_keep=num_logits_to_keep,
            mix_search=mix_search,
            use_cache=use_cache,
            **kwargs,
        )
        self.model_id = model_id
        self.sampling_rate = int(sampling_rate)
        self.window_samples = int(window_samples)
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)


class Emotion2vecModelMeta(BaseConfig):
    def __init__(
        self,
        *,
        hmonnx: str | None = None,
        onnx: str | None = None,
        sampling_rate: int = 16000,
        window_samples: int = 256000,
        feature_dim: int = 1024,
        num_labels: int = 9,
        labels: list[str] | tuple[str, ...] | None = None,
        quant_embedding: str | None = None,
        quant_embedding_md5: str = "",
        golden_dir: str | None = None,
        calibration_audio: str | None = None,
        validation_status: str = "not_run",
        model_id: str = "iic/emotion2vec_plus_large",
        model_config: dict | BaseAttrDict | None = None,
        meta: dict | None = None,
        **kwargs,
    ):
        self._meta_path_ = kwargs.pop("_meta_path_", None)
        super().__init__(**kwargs)
        self.hmonnx = hmonnx
        self.onnx = onnx
        self.sampling_rate = int(sampling_rate)
        self.window_samples = int(window_samples)
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        self.labels = list(labels or EMOTION2VEC_LABELS)
        self.quant_embedding = quant_embedding
        self.quant_embedding_md5 = quant_embedding_md5
        self.golden_dir = golden_dir
        self.calibration_audio = calibration_audio
        self.validation_status = validation_status
        self.model_id = model_id
        self.model_config = BaseAttrDict(model_config or {})
        self.meta = meta or {"class_name": type(self).__name__}
        if self._meta_path_:
            base_dir = Path(self._meta_path_).parent
            if self.hmonnx is not None:
                self.hmonnx = str((base_dir / self.hmonnx).resolve())
            if self.onnx is not None:
                self.onnx = str((base_dir / self.onnx).resolve())
            if self.golden_dir is not None:
                self.golden_dir = str((base_dir / self.golden_dir).resolve())
            if self.quant_embedding is not None:
                self.quant_embedding = str((base_dir / self.quant_embedding).resolve())

    @classmethod
    def from_json_file(cls, meta_file: str | Path):
        meta_file = Path(meta_file)
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        payload["_meta_path_"] = str(meta_file)
        return cls.from_dict(payload)
