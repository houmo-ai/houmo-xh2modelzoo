"""GPTQModel quantization adapter for DeepSeek-V4-Flash-0731."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class DeepSeekV4FlashQuantSpec:
    """One source of truth for the XH2 W8/W4-G64 weight artifact."""

    model_dir: str
    prepared_model_dir: str
    output_dir: str
    method: str = "gptq"
    num_layers: int | None = None
    dataset: str | None = "NeelNanda/pile-10k"
    calibration_jsonl: str | None = None
    text_key: str = "text"
    nsamples: int = 128
    seqlen: int = 2048
    batch_size: int = 1
    base_bits: int = 8
    expert_bits: int = 4
    group_size: int = 64
    device: str = "cuda:0"
    preparation_device: str | None = "cuda:0"
    offload_to_disk: bool = False
    offload_dir: str | None = None
    wait_for_submodule_finalizers: bool = True
    hessian_mse: bool = True
    moe_batch_size: int | None = 128
    auto_forward_data_parallel: bool = False
    max_quant_layers: int | None = None
    auto_round_version: str = "v2"
    auto_round_iters: int = 200
    auto_round_lr: float | None = None
    auto_round_minmax_lr: float | None = None
    seed: int = 42
    dry_run: bool = False

    def validate(self) -> None:
        method = str(self.method).strip().lower().replace("-", "_")
        if method == "autoround":
            method = "auto_round"
        if method not in {"gptq", "auto_round"}:
            raise ValueError("method must be 'gptq' or 'autoround'")
        self.method = method
        if (self.base_bits, self.expert_bits, self.group_size) != (8, 4, 64):
            raise ValueError("DeepSeek-V4 profile requires base W8, routed experts W4, and group_size=64")
        if self.num_layers is not None and self.num_layers <= 0:
            raise ValueError("num_layers must be positive when specified")
        if min(self.nsamples, self.seqlen, self.batch_size) <= 0:
            raise ValueError("nsamples, seqlen, and batch_size must be positive")
        if self.moe_batch_size is not None and self.moe_batch_size <= 0:
            raise ValueError("moe_batch_size must be positive when specified")
        if self.max_quant_layers is not None and self.max_quant_layers < 0:
            raise ValueError("max_quant_layers must be non-negative")
        if self.auto_round_version not in {"v1", "v2"}:
            raise ValueError("auto_round_version must be 'v1' or 'v2'")
        if self.auto_round_iters < 0:
            raise ValueError("auto_round_iters must be non-negative")


def quantization_plan(spec: DeepSeekV4FlashQuantSpec) -> dict[str, Any]:
    spec.validate()
    return {
        "backend": "gptqmodel.recipes.deepseek_v4_flash",
        "method": spec.method,
        "model": str(Path(spec.model_dir).expanduser().resolve()),
        "prepared_model": str(Path(spec.prepared_model_dir).expanduser().resolve()),
        "output": str(Path(spec.output_dir).expanduser().resolve()),
        "num_layers": spec.num_layers,
        "precision": {
            "routed_non_shared_experts": "W4",
            "shared_expert_and_other_quantized_linears": "W8",
            "group_size": 64,
            "symmetric": True,
            "control_paths": "FP16",
        },
        "calibration": {
            "dataset": spec.dataset,
            "jsonl": spec.calibration_jsonl,
            "nsamples": spec.nsamples,
            "seqlen": spec.seqlen,
        },
        "offload_to_disk": spec.offload_to_disk,
        "offload_dir": spec.offload_dir if spec.offload_to_disk else None,
        "auto_forward_data_parallel": spec.auto_forward_data_parallel,
        "auto_round": {
            "version": spec.auto_round_version,
            "iters": spec.auto_round_iters,
        },
    }


def quantize_deepseek_v4_flash(
    spec: DeepSeekV4FlashQuantSpec,
    *,
    calibration_data: Iterable[str | Mapping[str, Any]] | None = None,
):
    """Run either algorithm through GPTQModel's maintained in-tree API."""

    spec.validate()
    from gptqmodel.recipes.deepseek_v4_flash import (
        quantize_deepseek_v4_flash as gptqmodel_quantize,
    )

    return gptqmodel_quantize(
        model_dir=spec.model_dir,
        prepared_model_dir=spec.prepared_model_dir,
        output_dir=spec.output_dir,
        num_layers=spec.num_layers,
        method=spec.method,
        bits=spec.base_bits,
        expert_bits=spec.expert_bits,
        group_size=spec.group_size,
        sym=True,
        batch_size=spec.batch_size,
        nsamples=spec.nsamples,
        seqlen=spec.seqlen,
        calibration_jsonl=spec.calibration_jsonl,
        calibration_text_key=spec.text_key,
        calibration_dataset=spec.dataset,
        calibration_data=calibration_data,
        device=spec.device,
        preparation_device=spec.preparation_device,
        trust_remote_code=True,
        offload_to_disk=spec.offload_to_disk,
        offload_path=spec.offload_dir if spec.offload_to_disk else None,
        hessian_mse=spec.hessian_mse,
        wait_for_submodule_finalizers=spec.wait_for_submodule_finalizers,
        moe_routing_batch_size=spec.moe_batch_size,
        auto_forward_data_parallel=spec.auto_forward_data_parallel,
        max_quant_layers=spec.max_quant_layers,
        auto_round_version=spec.auto_round_version,
        auto_round_iters=spec.auto_round_iters,
        auto_round_lr=spec.auto_round_lr,
        auto_round_minmax_lr=spec.auto_round_minmax_lr,
        seed=spec.seed,
        dry_run=spec.dry_run,
    )


__all__ = [
    "DeepSeekV4FlashQuantSpec",
    "quantization_plan",
    "quantize_deepseek_v4_flash",
]
