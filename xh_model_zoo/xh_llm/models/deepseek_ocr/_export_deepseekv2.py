import argparse,os.path as osp,torch
from pathlib import Path
from xhquant import export
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, Config, release_quanted_model_unused_parameters  # isort:skip
from xh_model_zoo.utils import MemoryTracker, TimeProfiler
from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.deepseek_ocr.deepseekv2_converter import DeepseekV2ConverterConfig



def main(args):
    export_deepseekv2(args)
    
def export_deepseekv2(args): 
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    # quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
    config = DeepseekV2ConverterConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=args.num_logits_to_keep,
    )

    prefix = f"{model_name}-{target_device}/llm-{args.context_length//1024}k-{quant_type}"
    work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, "DeepSeekV2", config, str(work_dir))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")  # /data02/datasets/qwen3-8B-AWQ
    parser.add_argument(
        "--model",
        type=str,
        default="/data02/datasets/DeepSeek-OCR/",  # /data02/datasets/Qwen2.5-1.5B-Instruct-int4-sym-inc
    )
    parser.add_argument("--export-mode", type=str, default="Tiny", choices=["Tiny", "Small", "Base", "Large", "Gundam"],help="export mode, default is hmonnx")
    parser.add_argument("--context-length", type=int, default=8192, help="max sequence length")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="input sequence length")
    parser.add_argument("--num_logits_to_keep", type=int, default=1, help="not for test ppl")
    parser.add_argument("--quant-type", default="w8a8h0_ssfp", help="quant type, default is w8a8")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    parser.add_argument("--device",type=str,default="xh2a",help="device, default is xh2a")
    parser.add_argument("--generate-golden", default=True, help="generate golden")
    parser.add_argument("--extra_config_file", type=str, default=None, help="extra config file")
    parser.add_argument("--demo", default=False, action="store_true", help="demo mode")
    args = parser.parse_args()
    main(args)
