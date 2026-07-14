from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...workflows.result import QuantResult
from ..qwen3_5.quant_adapter import _quant_output_path, _resolve_calibration_value
from ..qwen3_5.workflow import Qwen35Workflow


class Qwen3NextWorkflow(Qwen35Workflow):
    """Text-only Qwen3-Next specialization of the shared Qwen3.5 workflow."""

    expected_model_config_cls_names = {"XHQwen3NextModelConfig"}
    expected_model_cls_names = {"XHQwen3NextModel"}
    family_label = "Qwen3-Next text-only"

    _COMMON_QUANT_KEYS = {
        "algorithm",
        "method",
        "output_format",
        "artifact_format",
        "bits",
        "group_size",
        "rotation",
        "calibration",
        "runtime",
        "moe",
        "save_path",
        "output_dir",
        "batch_size",
        "nsamples",
        "seqlen",
    }
    _METHOD_QUANT_KEYS = {
        "gptq": {"max_quant_layers", "validation"},
        "autoround": {"sym", "iters", "seed", "deterministic", "format"},
    }
    _COMMON_SECTION_KEYS = {
        "moe": {
            "attn_bits",
            "self_attn_bits",
            "linear_attn_bits",
            "dense_mlp_bits",
            "expert_bits",
            "shared_expert_bits",
        },
    }
    _METHOD_SECTION_KEYS = {
        "gptq": {
            "calibration": {"jsonl", "text_key", "nsamples", "seqlen"},
            "runtime": {
                "batch_size",
                "device_map",
                "infer_device_map",
                "auto_forward_data_parallel",
                "trust_remote_code",
                "dry_run",
                "extra_args",
            },
            "validation": {
                "max_quant_layers",
                "check_saved_precision",
                "min_logit_cosine",
                "require_top1_match",
                "skip_ppl",
                "max_new_tokens",
                "prompt",
            },
        },
        "autoround": {
            "calibration": {"dataset", "jsonl", "nsamples", "seqlen"},
            "runtime": {
                "batch_size",
                "device_map",
                "gradient_accumulate_steps",
                "low_gpu_mem_usage",
                "trust_remote_code",
                "dry_run",
            },
        },
    }

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        messages = super().build_input_message(input_messages)
        if self._messages_have_image(messages):
            raise ValueError("Qwen3-Next workflow accepts text-only input_messages")
        return messages

    def _quant_autoround_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        return self._quant_qwen3_next_api(
            output_dir,
            device,
            quant_cfg,
            export_model_cfg,
            expected_method="autoround",
        )

    def _quant_gptqmodel_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        return self._quant_qwen3_next_api(
            output_dir,
            device,
            quant_cfg,
            export_model_cfg,
            expected_method="gptq",
        )

    def _quant_qwen3_next_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
        *,
        expected_method: str,
    ) -> QuantResult:
        del export_model_cfg
        method = str(quant_cfg.get("method") or expected_method).lower()
        if method != expected_method:
            raise ValueError(f"Qwen3-Next quant adapter expected method={expected_method!r}, got {method!r}")
        self._validate_qwen3_next_quant_config(quant_cfg)
        try:
            from gptqmodel.recipes.qwen3_next import quantize_qwen3_next
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Qwen3-Next quantization requires GPTQModel's gptqmodel.recipes.qwen3_next.quantize_qwen3_next API"
            ) from exc

        calibration = dict(quant_cfg.get("calibration") or {})
        runtime = dict(quant_cfg.get("runtime") or {})
        validation = dict(quant_cfg.get("validation") or {})
        moe = dict(quant_cfg.get("moe") or {})
        artifact_format = self._resolve_artifact_format(quant_cfg)
        common_kwargs = dict(
            model_dir=self.model_dir,
            output_dir=_quant_output_path(output_dir, quant_cfg),
            method=method,
            artifact_format=artifact_format,
            rotation=quant_cfg.get("rotation", False),
            bits=int(quant_cfg.get("bits", 4)),
            self_attn_bits=int(moe.get("self_attn_bits", moe.get("attn_bits", 8))),
            linear_attn_bits=int(moe.get("linear_attn_bits", 8)),
            dense_mlp_bits=int(moe.get("dense_mlp_bits", 4)),
            expert_bits=int(moe.get("expert_bits", 4)),
            shared_expert_bits=int(moe.get("shared_expert_bits", 8)),
            group_size=int(quant_cfg.get("group_size", 64)),
            dry_run=bool(runtime.get("dry_run", False)),
        )
        if method == "autoround":
            result = quantize_qwen3_next(
                **common_kwargs,
                sym=bool(quant_cfg.get("sym", True)),
                iters=int(quant_cfg.get("iters", 200)),
                dataset=str(calibration.get("dataset", "NeelNanda/pile-10k")),
                calibration_jsonl=calibration.get("jsonl"),
                batch_size=int(runtime.get("batch_size", quant_cfg.get("batch_size", 8))),
                gradient_accumulate_steps=int(runtime.get("gradient_accumulate_steps", 1)),
                low_gpu_mem_usage=bool(runtime.get("low_gpu_mem_usage", True)),
                nsamples=int(calibration.get("nsamples", quant_cfg.get("nsamples", 128))),
                seqlen=int(calibration.get("seqlen", quant_cfg.get("seqlen", 2048))),
                seed=int(quant_cfg.get("seed", 42)),
                deterministic=bool(quant_cfg.get("deterministic", False)),
                format=str(quant_cfg.get("format", "auto_round:gptqmodel")),
                device_map=runtime.get("device_map", "0"),
                trust_remote_code=bool(runtime.get("trust_remote_code", True)),
            )
        else:
            result = quantize_qwen3_next(
                **common_kwargs,
                max_quant_layers=validation.get("max_quant_layers", quant_cfg.get("max_quant_layers")),
                batch_size=int(runtime.get("batch_size", quant_cfg.get("batch_size", 1))),
                nsamples=int(calibration.get("nsamples", quant_cfg.get("nsamples", 256))),
                seqlen=int(calibration.get("seqlen", quant_cfg.get("seqlen", 1024))),
                calibration_jsonl=_resolve_calibration_value(
                    calibration.get(
                        "jsonl",
                        "gptqmodel://quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl",
                    )
                ),
                calibration_text_key=str(calibration.get("text_key", "text")),
                device=device,
                device_map=runtime.get("device_map", "auto"),
                infer_device_map=runtime.get("infer_device_map"),
                auto_forward_data_parallel=bool(runtime.get("auto_forward_data_parallel", False)),
                trust_remote_code=bool(runtime.get("trust_remote_code", True)),
                check_saved_precision=bool(validation.get("check_saved_precision", False)),
                min_logit_cosine=float(validation.get("min_logit_cosine", 0.99)),
                require_top1_match=bool(validation.get("require_top1_match", True)),
                skip_ppl=bool(validation.get("skip_ppl", False)),
                max_new_tokens=int(validation.get("max_new_tokens", 128)),
                prompt=validation.get("prompt"),
                extra_args=runtime.get("extra_args"),
            )
        return QuantResult(
            raw_model_dir=self.model_dir,
            quanted_model_dir=str(result.output_dir),
            meta=result,
        )

    @classmethod
    def _validate_qwen3_next_quant_config(cls, quant_cfg: Mapping[str, Any]) -> None:
        """Reject Qwen3.5-only/dead keys before invoking the Next recipe."""

        method = str(quant_cfg.get("method") or "gptq").lower()
        if method not in cls._METHOD_QUANT_KEYS:
            raise ValueError(f"Unsupported Qwen3-Next quant method: {method!r}")
        allowed_keys = cls._COMMON_QUANT_KEYS | cls._METHOD_QUANT_KEYS[method]
        unknown = set(quant_cfg) - allowed_keys
        if unknown:
            raise ValueError(
                f"Qwen3-Next {method} config contains keys not consumed by the qwen3_next recipe: {sorted(unknown)}"
            )
        section_keys = {
            **cls._COMMON_SECTION_KEYS,
            **cls._METHOD_SECTION_KEYS[method],
        }
        for section, allowed in section_keys.items():
            value = quant_cfg.get(section)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError(f"Qwen3-Next quant.{section} must be a mapping")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(
                    f"Qwen3-Next quant.{section} contains keys not consumed by the qwen3_next recipe: {sorted(unknown)}"
                )

    @classmethod
    def _validate_qwen3_next_gptq_config(cls, quant_cfg: Mapping[str, Any]) -> None:
        """Compatibility alias for callers validating the GPTQ profile."""

        cls._validate_qwen3_next_quant_config(quant_cfg)


__all__ = ["Qwen3NextWorkflow"]
