from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow


DEFAULT_CONFIG_PATH = (
    "configs_merak/workflows/xh2a/audio_models/emotion2vec/"
    "emotion2vec_plus_large_xh2a_w8a8_16s.yaml"
)


def export_hmonnx(
    model_dir: str,
    config_path: str,
    output_dir: str,
    device: str = "cuda",
    overwrite: bool = False,
    golden_audio: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)

    workflow = AutoLLMWorkflow.from_config(
        model_dir=model_dir,
        config_path=config_path,
    )
    quant_result = workflow.quant(
        output_dir=f"{output_dir}_quant",
        device=device,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=output_dir,
        device=device,
    )
    if golden_audio is not None:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=device,
            input_messages={"audio": golden_audio},
        )
        print(f"golden_dir: {golden_dir}")
    return Path(export_result.work_dir) / "emotion2vec_meta.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the emotion2vec Merak W8A8 export workflow.")
    parser.add_argument("--model-dir", default="data/models/emotion2vec_plus_large")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default="work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump per-operator HMONNX golden data and official PyTorch reference outputs.",
    )
    parser.add_argument(
        "--golden-audio",
        default="data/models/emotion2vec_plus_large/example/test.wav",
        help="Audio used by workflow.dump_golden().",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        export_hmonnx(
            model_dir=args.model_dir,
            config_path=args.config_path,
            output_dir=args.output_dir,
            device=args.device,
            overwrite=args.overwrite,
            golden_audio=args.golden_audio if args.dump_golden else None,
        )
    )


if __name__ == "__main__":
    main()
