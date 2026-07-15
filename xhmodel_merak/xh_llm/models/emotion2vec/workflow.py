from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...workflows import BaseLLMWorkflow, ExportResult, QuantResult
from .configuration_emotion2vec import Emotion2vecModelMeta
from .modeling_emotion2vec import Emotion2vecExportBridge, load_funasr_emotion2vec_model


class Emotion2vecWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHEmotion2vecConfig"
    expected_model_cls_name = "XHEmotion2vecModel"

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides=None,
    ) -> ExportResult:
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
        if export_result.meta is None or not hasattr(export_result.meta, "to_dict"):
            raise TypeError("emotion2vec export must return serializable model metadata")
        meta_path = Path(export_result.work_dir) / "emotion2vec_meta.json"
        meta_path.write_text(
            json.dumps(export_result.meta.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return export_result

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        audio_file = self._resolve_golden_audio(input_messages)
        meta = self._resolve_export_meta(export_result)

        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError("soundfile is required to dump emotion2vec golden data") from exc

        waveform, sampling_rate = sf.read(audio_file, always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)
        if sampling_rate != meta.sampling_rate:
            raise ValueError(f"emotion2vec golden audio must be {meta.sampling_rate} Hz, got {sampling_rate}")
        if waveform.size == 0 or waveform.size > meta.window_samples:
            raise ValueError(
                f"emotion2vec golden audio length must be in [1, {meta.window_samples}], got {waveform.size}"
            )

        padded = np.zeros(meta.window_samples, dtype=np.float32)
        padded[: waveform.size] = waveform
        bridge = Emotion2vecExportBridge(
            load_funasr_emotion2vec_model(self.model_dir),
            window_samples=meta.window_samples,
        ).eval()
        bridge = bridge.to(device)
        with torch.no_grad():
            features, padding_mask = bridge(
                torch.from_numpy(padded).unsqueeze(0).to(device),
                torch.tensor([waveform.size], dtype=torch.int32, device=device),
            )
        valid_features = features[0, ~padding_mask[0]].float().cpu().numpy()

        golden_dir = Path(export_result.work_dir) / "golden"
        golden_dir.mkdir(parents=True, exist_ok=True)
        np.save(golden_dir / "frame_features.npy", valid_features)
        np.save(golden_dir / "utterance_feature.npy", valid_features.mean(axis=0))
        (golden_dir / "golden_meta.json").write_text(
            json.dumps(
                {
                    "audio": str(Path(audio_file).resolve()),
                    "sampling_rate": int(sampling_rate),
                    "valid_samples": int(waveform.size),
                    "frame_count": int(valid_features.shape[0]),
                    "feature_dim": int(valid_features.shape[1]),
                    "dtype": str(valid_features.dtype),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return str(golden_dir)

    @staticmethod
    def _resolve_golden_audio(input_messages: Any) -> str:
        if isinstance(input_messages, (str, Path)):
            audio_file = str(input_messages)
        elif isinstance(input_messages, Mapping):
            audio_file = input_messages.get("audio")
        else:
            audio_file = None
        if not isinstance(audio_file, str) or not audio_file:
            raise ValueError("emotion2vec input_messages must contain a non-empty 'audio' path")
        if not Path(audio_file).is_file():
            raise FileNotFoundError(audio_file)
        return audio_file

    @staticmethod
    def _resolve_export_meta(export_result: ExportResult) -> Emotion2vecModelMeta:
        if isinstance(export_result.meta, Emotion2vecModelMeta):
            return export_result.meta
        meta_path = Path(export_result.work_dir) / "emotion2vec_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"emotion2vec export metadata not found: {meta_path}")
        return Emotion2vecModelMeta.from_json_file(meta_path)


__all__ = ["Emotion2vecWorkflow"]