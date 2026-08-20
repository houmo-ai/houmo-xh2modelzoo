from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult
from .export_audio import export_minicpm_o_4_5_audio
from .export_common import normalize_component_artifacts
from .export_llm import export_minicpm_o_4_5_llm
from .export_speaker import export_minicpm_o_4_5_campplus, export_minicpm_o_4_5_speech_tokenizer
from .export_token2wav import (
    export_minicpm_o_4_5_token2wav_flow_decoder,
    export_minicpm_o_4_5_token2wav_flow_frontend,
    export_minicpm_o_4_5_token2wav_hift,
)
from .export_tts import export_minicpm_o_4_5_tts
from .export_vision import export_minicpm_o_4_5_vision
from .golden import dump_real_golden
from .synthetic_golden import dump_synthetic_golden


# The w4a8 export path is limited by the xhquant W4 CUDA kernel to
# prefill=256; w4a8h0_ssfp is the only quant_type whose bits(4) agree with
# quant.bits=4 (w4a8_sefp would silently fall back to w8a8). Enforcing both
# here keeps export.llm.quant_type consistent with the optional GPTQ block.
_W4A8_LLM_QUANT_TYPE = "w4a8h0_ssfp"
_W4A8_PREFILL_CHUNK_LENGTH = 256


class MiniCPMO45Workflow(BaseLLMWorkflow):
    SUPPORTED_COMPONENTS = (
        "vision",
        "audio",
        "llm",
        "tts",
        "speaker",
        "token2wav_flow_frontend",
        "token2wav_flow_decoder",
        "token2wav_hift",
    )

    @staticmethod
    def _parse_quant_type_bits(quant_type: str) -> tuple[int, int]:
        """Return ``(weight_bits, activation_bits)`` from a quant_type token."""
        prefix = quant_type.split("_", 1)[0]
        try:
            weight_part, activation_part = prefix[1:].split("a", 1)
            weight_bits = int(weight_part)
            activation_bits = int(activation_part.split("h", 1)[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "MiniCPM-o-4.5 export.model.quant_scheme.quant_type must contain "
                f"a w{{bits}}a{{bits}} token, got {quant_type!r}"
            ) from exc
        return weight_bits, activation_bits

    @classmethod
    def _format_export_basename(
        cls,
        model_cfg: Mapping[str, Any],
        quant_cfg: Mapping[str, Any] | None,
    ) -> tuple[str, str]:
        """Build the LLM export basename from existing workflow fields.

        Qwen-family exports name the LLM artifact from the workflow config.  MiniCPM
        follows that shape using only fields that are already present in this
        model's YAML. Returns ``(directory_name, file_stem)`` where the date is
        appended only to the directory name and the file stem stays stable.
        """
        chip_arch = str(model_cfg["chip_arch"]).lower()
        chip_token = "xh2" if chip_arch == "xh2a" else chip_arch
        model_name_token = str(model_cfg["model_name"]).lower()

        quant_type = str(model_cfg["quant_scheme"]["quant_type"])
        weight_bits, activation_bits = cls._parse_quant_type_bits(quant_type)
        if quant_cfg is not None and quant_cfg.get("bits") is not None:
            weight_bits = int(quant_cfg["bits"])
        quant_token = f"w{weight_bits}a{activation_bits}"

        prefill_chunk_length = int(model_cfg["prefill_chunk_length"])
        context_max_length = int(model_cfg["context_max_length"])

        def length_token(length: int) -> str:
            return f"{length // 1024}k" if length % 1024 == 0 else str(length)

        stem = (
            f"{chip_token}_{model_name_token}_{quant_token}_"
            f"{length_token(prefill_chunk_length)}_{length_token(context_max_length)}"
        )
        export_file_stem = f"hmquant_{stem}"
        return f"{export_file_stem}_{time.strftime('%Y%m%d')}", export_file_stem

    @classmethod
    def _validate_export_config(
        cls,
        export_cfg: Mapping[str, Any],
        quant_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        model_cfg = export_cfg.get("model")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("MiniCPM-o-4.5 export.model must be a mapping")
        for field in ("chip_arch", "model_type", "model_name"):
            if not isinstance(model_cfg.get(field), str) or not model_cfg[field]:
                raise ValueError(f"MiniCPM-o-4.5 export.model.{field} must be a non-empty string")
        if str(model_cfg["model_type"]) != "MiniCPMO45Model":
            raise ValueError(
                "MiniCPM-o-4.5 export.model.model_type must be 'MiniCPMO45Model', "
                f"got {model_cfg['model_type']!r}"
            )
        if str(model_cfg["chip_arch"]).lower() != "xh2a":
            raise ValueError(
                "MiniCPM-o-4.5 export.model.chip_arch must be 'XH2a', "
                f"got {model_cfg['chip_arch']!r}"
            )
        components = export_cfg.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("MiniCPM-o-4.5 export.components must be a mapping")
        missing = [name for name in cls.SUPPORTED_COMPONENTS if name not in components]
        if missing:
            raise ValueError(f"MiniCPM-o-4.5 export is missing components: {', '.join(missing)}")
        invalid_components = [name for name in cls.SUPPORTED_COMPONENTS if not isinstance(components[name], Mapping)]
        if invalid_components:
            raise ValueError("MiniCPM-o-4.5 component configs must be mappings: " + ", ".join(invalid_components))
        explicit_enabled = [name for name in cls.SUPPORTED_COMPONENTS if "enabled" in components[name]]
        if explicit_enabled:
            raise ValueError(
                "MiniCPM-o-4.5 runtime requires all components; omit redundant enabled fields: "
                + ", ".join(explicit_enabled)
            )
        llm_cfg = components["llm"]
        llm_quant_type = llm_cfg.get("quant_type")
        if quant_cfg is not None and int(quant_cfg.get("bits", 4)) == 4:
            if llm_quant_type != _W4A8_LLM_QUANT_TYPE:
                raise ValueError(
                    "MiniCPM-o-4.5 GPTQ bits=4 requires components.llm.quant_type="
                    f"{_W4A8_LLM_QUANT_TYPE}, got {llm_quant_type!r}"
                )
        model_quant_scheme = model_cfg.get("quant_scheme")
        if isinstance(model_quant_scheme, Mapping) and model_quant_scheme.get("quant_type"):
            model_quant_type = str(model_quant_scheme["quant_type"])
            if llm_quant_type != model_quant_type:
                raise ValueError(
                    "MiniCPM-o-4.5 export.model.quant_scheme.quant_type must match "
                    f"components.llm.quant_type ({model_quant_type!r} != {llm_quant_type!r})"
                )
        llm_wrap_cfg = llm_cfg.get("wrap_cfg")
        llm_length = llm_wrap_cfg.get("max_sequence_length") if isinstance(llm_wrap_cfg, Mapping) else None
        if llm_length is None or int(llm_length) <= 0:
            raise ValueError("MiniCPM-o-4.5 components.llm.wrap_cfg.max_sequence_length must be positive")
        llm_prefill = llm_wrap_cfg.get("input_sequence_length") if isinstance(llm_wrap_cfg, Mapping) else None
        if llm_prefill is None or int(llm_prefill) <= 0:
            raise ValueError("MiniCPM-o-4.5 components.llm.wrap_cfg.input_sequence_length must be positive")
        if quant_cfg is not None and int(quant_cfg.get("bits", 4)) == 4:
            if int(llm_prefill) != _W4A8_PREFILL_CHUNK_LENGTH:
                raise ValueError(
                    "MiniCPM-o-4.5 W4A8 export requires components.llm.wrap_cfg.input_sequence_length="
                    f"{_W4A8_PREFILL_CHUNK_LENGTH}, got {llm_prefill}"
                )

    def quant(self, output_dir: str, device: str, config_overrides: Mapping[str, Any] | None = None) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        self._validate_export_config(workflow_config.build_export_dict(), quant_cfg)
        from .quant_llm import quantize_minicpm_llm_gptq

        quanted_model_dir = quantize_minicpm_llm_gptq(
            model_dir=self.model_dir,
            output_dir=output_dir,
            quant_cfg=quant_cfg,
            device=device,
        )
        return QuantResult(raw_model_dir=self.model_dir, quanted_model_dir=quanted_model_dir)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = workflow_config.build_export_dict()
        quant_cfg = workflow_config.quant
        self._validate_export_config(export_cfg, quant_cfg)
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))
        model_dir = self._resolve_export_model_dir(quant_result)
        gptq_llm_model_dir = None
        if not quant_result.skipped:
            # Unlike a standalone LLM, MiniCPM's GPTQ artifact contains only
            # host.llm. Keep loading the original multimodal host, then let
            # the LLM exporter load and dequantize this packed submodel.
            gptq_llm_model_dir = model_dir
            model_dir = self.model_dir

        model_cfg = export_cfg["model"]
        target_device = str(model_cfg["chip_arch"])
        model_name = str(model_cfg["model_name"])
        components = export_cfg["components"]

        export_basename, export_file_stem = self._format_export_basename(model_cfg, quant_cfg)
        meta: dict[str, Any] = {
            "schema_version": 1,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_type": "MiniCPM-o-4.5",
            "model_name": model_name,
            "target_device": target_device,
            "chip_arch": target_device,
            "export_basename": export_basename,
            "hf_model": model_dir,
            "config": str(Path(config_file).relative_to(work_dir)),
            "components": {},
        }
        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        exporters = {
            "vision": export_minicpm_o_4_5_vision,
            "audio": export_minicpm_o_4_5_audio,
            "llm": export_minicpm_o_4_5_llm,
            "tts": export_minicpm_o_4_5_tts,
            "speaker": lambda **kw: {
                "quant_type": str(components["speaker"].get("quant_type", "w8a16_sefp")),
                "graphs": {
                    "campplus": export_minicpm_o_4_5_campplus(**kw)["graphs"]["campplus"],
                    "speech_tokenizer": export_minicpm_o_4_5_speech_tokenizer(**kw)["graphs"]["speech_tokenizer"],
                },
                "sequence_length": int(components["speaker"].get("sequence_length", 1000)),
                "feats_length": int(components["speaker"].get("feats_length", 3000)),
            },
            "token2wav_flow_frontend": export_minicpm_o_4_5_token2wav_flow_frontend,
            "token2wav_flow_decoder": export_minicpm_o_4_5_token2wav_flow_decoder,
            "token2wav_hift": export_minicpm_o_4_5_token2wav_hift,
        }
        for name in self.SUPPORTED_COMPONENTS:
            component = dict(components[name])
            # The GPTQ corpus is declared once under ``quant``.  Reuse it for
            # the LLM activation calibration export instead of maintaining a
            # second, potentially stale path in ``export.components.llm``.
            if name == "llm" and quant_cfg is not None:
                if "calibration_jsonl" not in component and quant_cfg.get("calibration_jsonl"):
                    component["calibration_jsonl"] = quant_cfg["calibration_jsonl"]
                component.setdefault("calibration_samples", min(5, int(quant_cfg.get("nsamples", 5))))
            component_model_dir = model_dir
            component_kwargs: dict[str, Any] = {}
            if name == "llm":
                component_kwargs["export_file_stem"] = export_file_stem
                if gptq_llm_model_dir is not None:
                    component_model_dir = self.model_dir
                    component_kwargs["gptq_llm_model_dir"] = gptq_llm_model_dir
            result = exporters[name](
                work_dir=work_dir,
                model_dir=component_model_dir,
                component_cfg=component,
                target_device=target_device,
                device=device,
                model_name=model_name,
                export_basename=export_basename,
                **component_kwargs,
            )
            result = normalize_component_artifacts(work_dir, result)
            meta["components"][name] = result
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any = None) -> str:
        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        mode = (
            "real"
            if isinstance(input_messages, Mapping)
            and (input_messages.get("mode") == "real" or "streaming_case" in input_messages)
            else "synthetic"
        )
        if mode == "real":
            golden_meta = dump_real_golden(work_dir, meta, device, input_messages)
        else:
            golden_meta = dump_synthetic_golden(work_dir, meta, device)
        golden_file = work_dir / "golden_meta_info.json"
        golden_file.write_text(json.dumps(golden_meta, indent=2), encoding="utf-8")
        return str(golden_file)


__all__ = ["MiniCPMO45Workflow"]
