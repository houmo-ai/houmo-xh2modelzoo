import argparse
from pathlib import Path

import torch

from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler
from xhquant_llm.evaluation import evaluate_wikitext


from xhquant.api import xhquant_init, get_xhquant_logger  # isort:skip


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type
    debug = args.debug
    if debug:
        cfg_name += "_debug"

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / f"convert_{eval_type}.log")
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xhquant_init(log_file, debug)
    logger = get_xhquant_logger()
    # 使用config文件,代替命令行参数,方便调试不同的配置
    cfg = Config.fromfile(args.config)

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)

    model_cfg.num_logits_to_keep = 0  # 输出所有logits
    model_cfg.use_cache = False  # 关闭缓存
    enable_prefill_chunk = args.enable_prefill_chunk
    if enable_prefill_chunk:
        model_cfg.enable_prefill_chunk = True  # 开启prefill chunk后，设置chunk长度为512
        model_cfg.use_cache = True
        model_cfg.enable_auto_offload = args.auto_offload  # 是否启用自动显存卸载

    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.eval()
    tokenizer = xh_model.get_tokenizer()
    eval_input_seq_length = 2048
    cache_dir = "./data/cache"
    contexts = [
        TimeProfiler("wikitext", logger),
        MemoryTracker(device=device, name="ppl", logger=logger),
        LLMInferenceContextManager(xh_model),
    ]
    with ContextManagers(contexts):
        metrics = evaluate_wikitext(xh_model, tokenizer, "test", eval_input_seq_length, -1, cache_dir=cache_dir)
        for k, v in metrics.items():
            logger.info(f"{eval_type} {k} : {v:.4f}")


if __name__ == "__main__":
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_2k.py"
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=eval_types)
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Whether to enable prefill chunk, only for debug and development",
    )
    args = parser.parse_args()
    main(args)
