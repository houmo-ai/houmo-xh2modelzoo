from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...workflows.result import QuantResult
from .gptqmodel_compat import laguna_gptqmodel_loader


def quantize_with_gptqmodel_api(
    *,
    model_dir: str,
    output_dir: str,
    device: str,
    quant_cfg: Mapping[str, Any],
) -> QuantResult:
    """Optionally create a GPTQModel-compatible HF checkpoint for Laguna."""

    from gptqmodel import GPTQModel, QuantizeConfig

    bits = int(quant_cfg.get("bits", 4))
    group_size = int(quant_cfg.get("group_size", 64))
    save_path = Path(output_dir) / str(
        quant_cfg.get("output_name", f"{Path(model_dir).name}-gptqmodel-{bits}bit")
    )
    calibration = _load_calibration(quant_cfg)
    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=_as_bool(quant_cfg.get("sym", True)),
        desc_act=_as_bool(quant_cfg.get("desc_act", False)),
        hessian_mse=_as_bool(quant_cfg.get("hessian_mse", True)),
        offload_to_disk=_as_bool(quant_cfg.get("offload_to_disk", True)),
        offload_to_disk_path=quant_cfg.get("offload_path"),
    )

    with laguna_gptqmodel_loader(model_dir):
        model = GPTQModel.load(
            model_dir,
            quant_config,
            device=device,
            device_map=quant_cfg.get("device_map"),
            trust_remote_code=True,
        )
        model.quantize(
            calibration,
            batch_size=int(quant_cfg.get("batch_size", 1)),
            calibration_concat_size=quant_cfg.get("calibration_concat_size"),
        )
        model.save(str(save_path), max_shard_size=quant_cfg.get("max_shard_size", "4GB"))

    return QuantResult(raw_model_dir=model_dir, quanted_model_dir=str(save_path.resolve()))


def _load_calibration(quant_cfg: Mapping[str, Any]) -> list[str]:
    calibration = quant_cfg.get("calibration")
    if isinstance(calibration, Mapping):
        texts = calibration.get("texts")
        jsonl = calibration.get("jsonl") or calibration.get("dataset")
        text_key = str(calibration.get("text_key", "text"))
        nsamples = int(calibration.get("nsamples", 128))
    else:
        texts = None
        jsonl = None
        text_key = "text"
        nsamples = 128

    if isinstance(texts, Sequence) and not isinstance(texts, (str, bytes)):
        result = [str(text) for text in texts if str(text).strip()]
        if result:
            return result[:nsamples]

    if jsonl:
        import json

        path = Path(str(jsonl)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Laguna GPTQ calibration JSONL does not exist: {path}")
        result = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                text = record.get(text_key) if isinstance(record, Mapping) else None
                if isinstance(text, str) and text.strip():
                    result.append(text)
                if len(result) >= nsamples:
                    break
        if result:
            return result

    raise ValueError(
        "Laguna GPTQModel quantization requires quant.calibration.texts or "
        "quant.calibration.jsonl/dataset; the default HF floating-point workflow does not require this."
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = ["quantize_with_gptqmodel_api"]