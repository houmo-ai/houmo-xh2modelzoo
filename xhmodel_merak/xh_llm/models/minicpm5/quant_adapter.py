"""GPTQModel quantization adapter for MiniCPM5-MoE."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult


@dataclass(frozen=True, slots=True)
class MiniCPM5QuantConfigError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def _resolve_calibration_jsonl_path(calibration_cfg: Mapping[str, Any]) -> str | None:
    jsonl_path = calibration_cfg.get("jsonl") or calibration_cfg.get("calibration_jsonl")
    if not jsonl_path:
        return None
    if str(jsonl_path).strip().startswith("path-to-"):
        raise MiniCPM5QuantConfigError(
            "MiniCPM5 GPTQ calibration.jsonl still uses a placeholder path; "
            "replace it with the generated calibration JSONL file path"
        )
    path = _resolve_calibration_jsonl(str(jsonl_path))
    if not path.is_file():
        raise FileNotFoundError(f"MiniCPM5 GPTQ calibration JSONL does not exist: {path}")
    return str(path)


def _resolve_calibration_jsonl(value: str) -> Path:
    return Path(value.strip()).expanduser()


def _normalize_moe_routing(moe_cfg: Mapping[str, Any]) -> tuple[str, int | None, int | str | None]:
    routing = str(moe_cfg.get("routing", moe_cfg.get("moe_routing", "bypass"))).strip().lower()
    if routing not in {"none", "bypass", "override"}:
        raise MiniCPM5QuantConfigError("MiniCPM5 GPTQ quant.moe.routing must be one of: none, bypass, override")
    batch_size = moe_cfg.get("routing_batch_size", moe_cfg.get("moe_routing_batch_size"))
    batch_size = int(batch_size) if batch_size is not None else None
    if batch_size is not None and batch_size < 1:
        raise MiniCPM5QuantConfigError("MiniCPM5 GPTQ MoE routing batch size must be positive")
    experts_per_token = moe_cfg.get("num_experts_per_tok", moe_cfg.get("moe_num_experts_per_tok"))
    if experts_per_token is not None and experts_per_token != "all":
        experts_per_token = int(experts_per_token)
        if experts_per_token < 1:
            raise MiniCPM5QuantConfigError("MiniCPM5 GPTQ MoE experts per token must be positive or 'all'")
    return routing, batch_size, experts_per_token


def quantize_with_gptqmodel_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
) -> QuantResult:
    """Translate workflow YAML into the stable GPTQModel MiniCPM5 recipe."""

    group_size = int(quant_cfg.get("group_size", 64))
    if group_size != 64:
        raise MiniCPM5QuantConfigError("MiniCPM5 GPTQModel group_size must be 64")

    bits = int(quant_cfg.get("bits", 4))
    method = str(quant_cfg.get("method", "gptq")).strip().lower().replace("-", "_")
    if method not in {"gptq", "autoround", "auto_round"}:
        raise MiniCPM5QuantConfigError("MiniCPM5 GPTQModel method must be 'gptq' or 'autoround'")

    calibration_cfg = quant_cfg.get("calibration")
    if not isinstance(calibration_cfg, Mapping):
        raise MiniCPM5QuantConfigError("MiniCPM5 GPTQ requires a quant.calibration mapping")
    runtime_cfg = quant_cfg.get("runtime")
    runtime_cfg = runtime_cfg if isinstance(runtime_cfg, Mapping) else {}
    moe_cfg = quant_cfg.get("moe")
    moe_cfg = moe_cfg if isinstance(moe_cfg, Mapping) else {}

    calibration_jsonl = _resolve_calibration_jsonl_path(calibration_cfg)
    calibration_dataset = None
    calibration_dataset_config = None
    calibration_split = str(calibration_cfg.get("split", "train"))
    if calibration_jsonl is None:
        dataset_name = str(calibration_cfg.get("dataset", "wikitext"))
        dataset_config = str(calibration_cfg.get("name", "wikitext-2-raw-v1"))
        if dataset_name != "wikitext" or dataset_config != "wikitext-2-raw-v1":
            raise MiniCPM5QuantConfigError(
                "MiniCPM5 GPTQ calibration must use wikitext/wikitext-2-raw-v1 or quant.calibration.jsonl"
            )
        calibration_dataset = dataset_name
        calibration_dataset_config = dataset_config

    self_attn_bits = int(moe_cfg.get("attn_bits", moe_cfg.get("self_attn_bits", 8)))
    dense_mlp_bits = int(moe_cfg.get("dense_mlp_bits", 8))
    shared_expert_bits = int(moe_cfg.get("shared_expert_bits", 8))
    expert_bits = int(moe_cfg.get("expert_bits", 4))
    routing, routing_batch_size, experts_per_token = _normalize_moe_routing(moe_cfg)

    save_path = Path(output_dir) / str(
        quant_cfg.get("output_name", f"{Path(model_dir).name}-gptqmodel-{bits}bit-{group_size}g")
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    from gptqmodel.recipes import quantize_minicpm5

    result = quantize_minicpm5(
        model_dir=model_dir,
        output_dir=str(save_path),
        method=method,
        bits=bits,
        self_attn_bits=self_attn_bits,
        dense_mlp_bits=dense_mlp_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        group_size=group_size,
        sym=_as_bool(quant_cfg.get("sym", True)),
        batch_size=int(runtime_cfg.get("batch_size", 1)),
        nsamples=int(calibration_cfg.get("nsamples", 128)),
        seqlen=int(calibration_cfg.get("seqlen", 1024)),
        calibration_jsonl=calibration_jsonl,
        calibration_text_key=str(calibration_cfg.get("text_key", calibration_cfg.get("calibration_text_key", "text"))),
        calibration_dataset=calibration_dataset,
        calibration_dataset_config=calibration_dataset_config,
        calibration_split=calibration_split,
        calibration_cache_dir=calibration_cfg.get("cache_dir"),
        device=device,
        trust_remote_code=_as_bool(runtime_cfg.get("trust_remote_code", True)),
        offload_to_disk=_as_bool(quant_cfg.get("offload_to_disk", True)),
        offload_to_disk_path=quant_cfg.get("offload_path"),
        hessian_mse=_as_bool(quant_cfg.get("hessian_mse", True)),
        damp_percent=float(quant_cfg.get("damp_percent", 0.01)),
        moe_routing=routing,
        moe_routing_batch_size=routing_batch_size,
        moe_num_experts_per_tok=experts_per_token,
        auto_round_version=str(quant_cfg.get("auto_round_version", "v1")).lower(),
        auto_round_iters=int(quant_cfg.get("iters", 200)),
        auto_round_lr=(float(quant_cfg["lr"]) if quant_cfg.get("lr") is not None else None),
        auto_round_minmax_lr=(float(quant_cfg["minmax_lr"]) if quant_cfg.get("minmax_lr") is not None else None),
        max_shard_size=str(quant_cfg.get("max_shard_size", "4GB")),
    )
    return QuantResult(
        raw_model_dir=model_dir,
        quanted_model_dir=str(Path(result.output_dir).resolve()),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["quantize_with_gptqmodel_api"]
