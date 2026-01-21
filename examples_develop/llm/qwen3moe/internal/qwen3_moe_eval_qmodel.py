import argparse, json,torch
from typing import Any, Optional
from pathlib import Path
from xhquant.api import ConfigDict, get_root_logger, xhquant_init,auto_release

from xh_model_zoo_develop.xh_llm.models.qwen3moe import Qwen3MoeConverterConfig,Qwen3MoeConverter
from xh_model_zoo_develop.xh_llm.base_llm_infer_adapter import BaseLLMHFCompatible
from xh_model_zoo_develop.utils import auto_offload, xh_infer_auto_device_map
torch.set_grad_enabled(False)

def lm_eval_engine(hf_model: Any, tokenizer: Any, meta_info: Optional[dict] = None, cfg: Optional[ConfigDict] = None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable, make_table
    from lm_eval.models.huggingface import HFLM

    logger = get_root_logger()
    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, max_length=2048)
    # lm = XH2LLM(pretrained=hf_model, tokenizer=tokenizer, max_length=2048)  # 默认max_length=40960
    lm.model.eval()
    task_manager = TaskManager()
    tasks = [
        # "cmmlu",
        # "gsm8k",
        # "mathqa",
        # "openbookqa",
        # "winogrande",
        "arc_challenge",
        # "hellaswag",
        # "wikitext",
    ]
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, task_manager=task_manager, batch_size=1, device="cuda")
    # json.dump(result, open("result.json", "w", encoding="utf-8"), indent=4)
    if results is not None:
        # if "samples" in results:
        #     samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        result_json_file = Path(cfg.work_dir) / "eval_results_minmax_w4.json"
        with open(result_json_file, "w", encoding="utf-8") as f:
            f.write(dumped)

        logger.info(f"Results:\n{dumped}")
        logger.info(make_table(results))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-30B-A3B", help="model path")
    parser.add_argument("--offload", action="store_true", help="offload mode")
    parser.add_argument("--demo_prompt", type=str, default="你多大了？用中文回答。", help="demo prompt")
    parser.add_argument("--eval_ppl", type=bool, default=True, help="eval ppl")
    parser.add_argument("--fast", action="store_true", help="fast mode")
    return parser.parse_args()


def main(args):
    from xhquant.api import QuantScheme

    quant_scheme = QuantScheme(target_device="XH2A", quant_type="w8a8h1_sefp")
    config = Qwen3MoeConverterConfig(num_logits_to_keep=0, quant_scheme=quant_scheme)
    C = Qwen3MoeConverter(args.model, config)

    qmodel = C.quanted_model
    if args.fast:
        qmodel.enable_fast_precision_mode()

    model = BaseLLMHFCompatible.from_qmodel(
        qmodel, C.hf_config, C.wrap_cfg, C.token_embedding, C.hf_model_path, C.tokenizer
    )

    if args.offload:
        device_map = xh_infer_auto_device_map(C.wraped_model, "XHTrace_Qwen3MoeDecoderLayer")
        auto_offload(qmodel, device_map=device_map)
    else:
        model.cuda().half()

    if args.eval_ppl:
        from xh_model_zoo_develop.evaluation.wikippl_eval import evaluate_wikitext

        evaluate_wikitext(model, C.tokenizer)

    if args.demo_prompt is not None:
        print(model.demo(args.demo_prompt))

    lm_eval_engine(model, C.tokenizer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
