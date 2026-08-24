from __future__ import annotations

import argparse
import json

from xhmodel_merak.xh_other_model.models.kokoro.analysis import analyze_onnx
from xhmodel_merak.xh_other_model.models.kokoro.assets import (
    resolve_model_assets,
    verify_release_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the pinned Kokoro release ONNX")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    assets = resolve_model_assets(args.model_dir)
    result = {
        "asset_identity": verify_release_assets(assets),
        "onnx": analyze_onnx(assets.reference_onnx),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
