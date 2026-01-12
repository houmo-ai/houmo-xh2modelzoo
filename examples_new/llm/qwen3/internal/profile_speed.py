import gc
import argparse, json, torch
from typing import Any, Optional, List
from pathlib import Path

from xhquant.api import ConfigDict, get_root_logger, xhquant_init

from xh_model_zoo_new.xh_llm.models.qwen3.qwen3_converter import Qwen3Converter, Qwen3ConverterConfig
from xh_model_zoo_new.xh_llm.base_llm_infer_adapter import BaseLLMHFCompatible
from xh_model_zoo_new.utils import auto_offload, xh_infer_auto_device_map

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-8B")
    parser.add_argument("--tasks", nargs="+", type=str, default=["arc_challenge"])
    parser.add_argument("--offload", action="store_true", help="offload mode")
    parser.add_argument("--demo_prompt", type=str, default="你多大了？用中文回答。", help="demo prompt")
    parser.add_argument("--eval_ppl", type=bool, default=True, help="eval ppl")
    parser.add_argument("--fast", action="store_true", help="fast mode")
    return parser.parse_args()


def main(args):
    from xhquant.api import QuantScheme

    quant_scheme = QuantScheme(target_device="XH2A", quant_type="w8a8h1_sefp")
    config = Qwen3ConverterConfig(num_logits_to_keep=0, quant_scheme=quant_scheme)
    C = Qwen3Converter(args.model, config)

    qmodel = C.quanted_model
    if args.fast:
        qmodel.enable_fast_precision_mode()
        # qmodel.enable_aligned_precision_mode()

    model = BaseLLMHFCompatible.from_qmodel(
        qmodel, C.hf_config, C.wrap_cfg, C.token_embedding, C.hf_model_path, C.tokenizer
    )

    if args.offload:
        device_map = xh_infer_auto_device_map(C.wraped_model, "XHTrace_Qwen3DecoderLayer")
        auto_offload(qmodel, device_map=device_map)
    else:
        model.cuda().half()

    if True:
        quanted_model = qmodel

        calib_data = [
            torch.randn(1, 2048, 4096, dtype=torch.float16, device="cuda"),
            torch.tensor([0], dtype=torch.int32, device="cuda"),
            torch.tensor([2048], dtype=torch.int32, device="cuda"),*model.past_key_caches,*model.past_value_caches
        ]
        calib_data = [t.cuda() for t in calib_data]
        from xhquant.debug.profiler_tool import GraphModuleProfiler

        profiler = GraphModuleProfiler(quanted_model, profile_precision=False)

        with torch.inference_mode():
            torch.cuda.empty_cache()
            quanted_model.disable_quant()
            C.wrap_cfg.input_sequence_length = 2048
            quanted_model.update_cfg(C.wrap_cfg)

            for _ in range(2):
                quanted_model(*calib_data)
            torch.cuda.synchronize()

            # time.sleep(1)
            gc.collect()
            profiler.profile(quanted_model, "PREFILL_DISABLED", *calib_data)

            torch.cuda.empty_cache()
            quanted_model.enable_fast_precision_mode()
            for _ in range(2):
                quanted_model(*calib_data)

            torch.cuda.synchronize()
            gc.collect()
            profiler.profile(quanted_model, "PREFILL_FAST", *calib_data)
            gc.collect()
        profiler.print_format_results()

        
        decode_profiler = GraphModuleProfiler(qmodel, profile_precision=False)
        decode_calib_data = calib_data.copy()
        decode_calib_data[0] = torch.randn(1, 1, 4096, dtype=torch.float16, device="cuda")
        decode_calib_data[1] = torch.tensor([1024], dtype=torch.int32, device="cuda")
        decode_calib_data[2] = torch.tensor([1], dtype=torch.int32, device="cuda")
        with torch.inference_mode():
            torch.cuda.empty_cache()
            quanted_model.disable_quant()
            C.wrap_cfg.input_sequence_length = 1
            quanted_model.update_cfg(C.wrap_cfg)

            for _ in range(2):
                quanted_model(*decode_calib_data)
            torch.cuda.synchronize()

            # time.sleep(1)
            gc.collect()
            decode_profiler.profile(quanted_model, "DECODE_DISABLED", *decode_calib_data)

            torch.cuda.empty_cache()
            quanted_model.enable_fast_precision_mode()
            for _ in range(2):
                quanted_model(*decode_calib_data)

            torch.cuda.synchronize()
            gc.collect()
            decode_profiler.profile(quanted_model, "DECODE_FAST", *decode_calib_data)
            gc.collect()
        decode_profiler.print_format_results()


    if args.eval_ppl:
        from xh_model_zoo_new.evaluation.wikippl_eval import evaluate_wikitext

        evaluate_wikitext(model, C.tokenizer)

    if args.demo_prompt is not None:
        print(model.demo(args.demo_prompt))

    # lm_eval_engine(model, C.tokenizer,tasks=args.tasks,export_dir=args.export_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
