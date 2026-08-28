from __future__ import annotations

import argparse
import json
from pathlib import Path

from xhmodel_merak.xh_other_model.models.kokoro.analysis import analyze_onnx
from xhmodel_merak.xh_other_model.models.kokoro.assets import (
    KOKORO_REFERENCE_ONNX_SHA256,
    resolve_model_assets,
    sha256,
    verify_release_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the pinned Kokoro release ONNX")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()
    assets = resolve_model_assets(args.model_dir)
    reference_onnx = args.onnx.expanduser().resolve()
    reference_sha256 = sha256(reference_onnx)
    if reference_sha256 != KOKORO_REFERENCE_ONNX_SHA256:
        raise ValueError(
            f"Unexpected reference ONNX SHA256 for {reference_onnx}: "
            f"expected {KOKORO_REFERENCE_ONNX_SHA256}, got {reference_sha256}"
        )
    result = {
        "asset_identity": {
            **verify_release_assets(assets),
            "reference_onnx_sha256": reference_sha256,
        },
        "onnx": analyze_onnx(reference_onnx),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
