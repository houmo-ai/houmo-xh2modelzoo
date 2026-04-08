import argparse
import json
from pathlib import Path


def main(args):
    base_meta_path = Path(args.base_golden_meta).resolve()
    legacy_export_dir = Path(args.legacy_export_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else legacy_export_dir / "golden_meta_info.merak.json"

    base_meta = json.load(open(base_meta_path))
    legacy_meta = json.load(open(legacy_export_dir / "export_meta_info.json"))

    base_meta["create_time"] = legacy_meta.get("create_time", base_meta.get("create_time", ""))
    base_meta["hf_config"] = "hf_config"
    base_meta["quant_embedding"] = "token_embedding.pt"
    base_meta["quant_embedding_md5"] = ""
    base_meta["prefill_hmonnx"] = legacy_meta["prefill_onnx_file"]
    base_meta["decode_hmonnx"] = legacy_meta["decode_onnx_file"]
    base_meta["prefill_hmonnx_md5"] = ""
    base_meta["decode_hmonnx_md5"] = ""

    visual_hmonnx = base_meta_path.parent / base_meta["visual_config"]["hmonnx"]
    base_meta["visual_config"]["hmonnx"] = str(visual_hmonnx.resolve())

    model_config = base_meta["model_config"]
    model_config["hf_model"] = args.quantized_model_dir
    model_config["quant_weight"] = None
    model_config["model_name"] = args.model_name
    model_config["quant_scheme"]["quant_type"] = args.quant_type
    model_config["visual_config"]["hf_model"] = args.quantized_model_dir
    model_config["visual_config"]["model_name"] = f"{args.model_name}_visual"
    model_config["visual_config"]["quant_scheme"]["quant_type"] = args.quant_type

    output_path.write_text(json.dumps(base_meta, indent=2))
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt legacy qwen3_5_moe export_meta_info.json to Merak HMONNX meta")
    parser.add_argument("--legacy-export-dir", type=str, required=True, help="Legacy export directory containing export_meta_info.json")
    parser.add_argument("--base-golden-meta", type=str, required=True, help="Merak golden_meta_info.json to use as template")
    parser.add_argument("--quantized-model-dir", type=str, required=True, help="Underlying GPTQModel HF directory")
    parser.add_argument("--model-name", type=str, default="xh2_qwen35_35b_a3b_attn4_e4_se4_0324_w4a8_256_2k")
    parser.add_argument("--quant-type", type=str, default="w4a8h1_sefp")
    parser.add_argument("--output", type=str, default="", help="Optional output meta path")
    args = parser.parse_args()
    main(args)
