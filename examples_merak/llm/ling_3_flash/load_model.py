#!/usr/bin/env python3
"""Load Ling-3-Flash through xh2modelzoo from original or GPTQModel weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xhmodel_merak.xh_llm.models.ling_3_flash._compat import (
    load_ling_config,
    patch_ling_remote_code_compatibility,
)
from xhmodel_merak.xh_llm.models.ling_3_flash.ling_3_flash_model import (
    XHLing3FlashModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--device-map",
        default="balanced",
        help="Accelerate/GPTQModel device map; use balanced with three visible GPUs",
    )
    parser.add_argument(
        "--max-memory",
        nargs="*",
        default=None,
        metavar="DEVICE=LIMIT",
        help="Example: 0=76GiB 1=76GiB 2=76GiB cpu=512GiB",
    )
    parser.add_argument("--config-only", action="store_true")
    return parser.parse_args()


def _parse_max_memory(values: list[str] | None) -> dict | None:
    if not values:
        return None
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --max-memory entry: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        result[int(key) if key.isdigit() else key] = value.strip()
    return result


def main() -> None:
    args = parse_args()
    patch_ling_remote_code_compatibility()
    config = load_ling_config(args.model)
    quant_config = getattr(config, "quantization_config", None)
    info = {
        "path": str(Path(args.model).expanduser().resolve()),
        "architecture": list(getattr(config, "architectures", [])),
        "model_type": config.model_type,
        "layers": int(config.num_hidden_layers),
        "experts": int(config.num_experts),
        "quantized": quant_config is not None,
        "quant_method": (
            quant_config.get("quant_method")
            if isinstance(quant_config, dict)
            else None
        ),
    }
    print(json.dumps(info, indent=2))
    if args.config_only:
        return

    kwargs = {
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    max_memory = _parse_max_memory(args.max_memory)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = XHLing3FlashModel.get_hf_model(args.model, **kwargs)
    num_decoder_layers = int(model.config.num_hidden_layers)
    all_layers = model.model.layers
    decoder_layers = all_layers[:num_decoder_layers]
    layer_types = [layer.attention_layer_type for layer in decoder_layers]
    print(
        json.dumps(
            {
                "loaded_class": type(model).__name__,
                "dtype": str(next(model.parameters()).dtype),
                "device_count": len({str(parameter.device) for parameter in model.parameters()}),
                "kda_layers": layer_types.count("linear_attention"),
                "mla_layers": layer_types.count("attention"),
                "mtp_layers": len(all_layers) - num_decoder_layers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
