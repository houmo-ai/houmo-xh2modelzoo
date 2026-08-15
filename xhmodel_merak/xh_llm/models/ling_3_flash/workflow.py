"""End-to-end Merak workflow for Ling-3-Flash."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult
from ..qwen3_5.workflow import Qwen35Workflow
from .quant_adapter import (
    Ling3FlashQuantSpec,
    quantization_plan,
    quantize_ling3_flash,
)


class Ling3FlashWorkflow(Qwen35Workflow):
    expected_model_config_cls_names = {"XHLing3FlashModelConfig"}
    expected_model_cls_names = {"XHLing3FlashModel"}
    family_label = "Ling-3-Flash text-only"

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        messages = super().build_input_message(input_messages)
        if self._messages_have_image(messages):
            raise ValueError("Ling-3-Flash workflow accepts text-only input_messages")
        return messages

    def _quant_autoround_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        del export_model_cfg
        return self._quant_ling(
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            method="autoround",
        )

    def _quant_gptqmodel_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        del export_model_cfg
        return self._quant_ling(
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            method="gptq",
        )

    def _quant_ling(
        self,
        *,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        method: str,
    ) -> QuantResult:
        requested_method = str(quant_cfg.get("method") or method).lower().replace("-", "_")
        if requested_method == "auto_round":
            requested_method = "autoround"
        if requested_method != method:
            raise ValueError(
                f"Ling quant adapter expected method={method!r}, got {requested_method!r}"
            )

        calibration = dict(quant_cfg.get("calibration") or {})
        runtime = dict(quant_cfg.get("runtime") or {})
        moe = dict(quant_cfg.get("moe") or {})
        base_bits = int(quant_cfg.get("bits", 8))
        expert_bits = int(moe.get("expert_bits", 4))
        shared_bits = int(moe.get("shared_expert_bits", base_bits))
        dense_bits = int(moe.get("dense_mlp_bits", base_bits))
        attention_bits = int(moe.get("attention_bits", base_bits))
        symmetric = quant_cfg.get("sym", True)
        if (base_bits, expert_bits, shared_bits, dense_bits, attention_bits) != (
            8,
            4,
            8,
            8,
            8,
        ):
            raise ValueError(
                "Ling quant profile requires base/attention/dense/shared W8 and routed experts W4; "
                f"got base={base_bits}, attention={attention_bits}, dense={dense_bits}, "
                f"shared={shared_bits}, expert={expert_bits}"
            )
        if symmetric is not True:
            raise ValueError("Ling quant profile requires symmetric quantization (sym=true)")

        calibration_jsonl = calibration.get("jsonl")
        moe_batch_size = runtime.get("moe_batch_size")
        offload_to_disk = runtime.get(
            "offload_to_disk",
            quant_cfg.get("offload_to_disk", False),
        )
        wait_for_submodule_finalizers = runtime.get(
            "wait_for_submodule_finalizers",
            quant_cfg.get("wait_for_submodule_finalizers", True),
        )
        if not isinstance(offload_to_disk, bool):
            raise TypeError("runtime.offload_to_disk must be a boolean")
        if not isinstance(wait_for_submodule_finalizers, bool):
            raise TypeError(
                "runtime.wait_for_submodule_finalizers must be a boolean"
            )
        result_dir = str(Path(output_dir).expanduser().resolve())
        spec = Ling3FlashQuantSpec(
            model_dir=self.model_dir,
            output_dir=result_dir,
            method=method,
            dataset=str(calibration.get("dataset", "NeelNanda/pile-10k")),
            calibration_jsonl=(
                None if calibration_jsonl in (None, "") else str(calibration_jsonl)
            ),
            text_key=str(calibration.get("text_key", "text")),
            nsamples=int(calibration.get("nsamples", quant_cfg.get("nsamples", 128))),
            seqlen=int(calibration.get("seqlen", quant_cfg.get("seqlen", 2048))),
            batch_size=int(runtime.get("batch_size", quant_cfg.get("batch_size", 1))),
            moe_batch_size=(
                None if moe_batch_size is None else int(moe_batch_size)
            ),
            iters=int(quant_cfg.get("iters", 200)),
            group_size=int(quant_cfg.get("group_size", 64)),
            base_bits=base_bits,
            expert_bits=expert_bits,
            device=str(runtime.get("device", device)),
            device_map=str(runtime.get("device_map", "0")),
            offload_to_disk=offload_to_disk,
            offload_dir=str(
                runtime.get(
                    "offload_dir",
                    Path(result_dir).parent / "ling_3_flash_offload",
                )
            ),
            wait_for_submodule_finalizers=wait_for_submodule_finalizers,
            hessian_mse=bool(runtime.get("hessian_mse", True)),
            max_quant_layers=quant_cfg.get("max_quant_layers"),
            seed=int(quant_cfg.get("seed", self.seed)),
            dry_run=bool(runtime.get("dry_run", False)),
        )
        quantize_ling3_flash(spec)
        return QuantResult(
            raw_model_dir=self.model_dir,
            quanted_model_dir=result_dir,
            meta=quantization_plan(spec),
        )


__all__ = ["Ling3FlashWorkflow"]
