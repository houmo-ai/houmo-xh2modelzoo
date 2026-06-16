import argparse
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import lm_eval

# from examples._init_path import _init_path  # pylint: disable=unused-import # isort:skip
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table
from loguru import logger

from xhmodel_merak.cmmlu_dataset import ensure_cmmlu_task_overrides
from xhmodel_merak.xh_llm import (
    AutoLLMConfig,
    AutoLLMModel,
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.api import Config, get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler
from xhquant.utils.time_profiler import time_profiler


class XH2LLM(HFLM):
    def _model_call(self, inps, attn_mask=None, labels=None):
        # self.model.use_cache = False
        return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # self.model.use_cache = True
        return super()._model_generate(context, max_length, stop, **generation_kwargs)


class _CMMLUCustomDatasetWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Custom kwargs can be passed to `--metadata` in console" not in message


@contextmanager
def suppress_cmmlu_custom_dataset_warning(enabled: bool):
    if not enabled:
        yield
        return

    eval_logger = logging.getLogger("lm_eval.api.task")
    warning_filter = _CMMLUCustomDatasetWarningFilter()
    eval_logger.addFilter(warning_filter)
    try:
        yield
    finally:
        eval_logger.removeFilter(warning_filter)


def lm_eval_engine(
    hf_model: Any, tokenizer: Any, meta_info: dict, cfg: Config, device: str, include_path: str | None = None
):
    logger = get_xhquant_logger()
    lm = XH2LLM(pretrained=hf_model, tokenizer=tokenizer, max_length=2048, device=device)  # 默认max_length=40960
    lm.model.eval()
    task_manager = TaskManager(include_path=include_path)
    task = cfg.task if hasattr(cfg, "task") else "wikitext"
    tasks = [task]

    # tasks = [
    #     "wikitext",
    #     "winogrande",
    #     "arc_challenge",
    #     "hellaswag",
    #     "openbookqa",
    #     "cmmlu",
    #     "gsm8k",
    #     "mathqa",
    #     "mmlu",
    # ]  # ["cmmlu", "gsm8k", "mathqa", "openbookqa", "winogrande", "arc_challenge", "hellaswag"]

    for task in tasks:
        task_dir = Path(cfg.work_dir) / task
        task_dir.mkdir(parents=True, exist_ok=True)
        task_out_file = task_dir / f"{task}_results_minmax_w4.json"
        if task_out_file.exists():
            logger.info(f"Skip task {task}")
            results = json.load(open(task_out_file, "r", encoding="utf-8"))
            dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        else:
            warning_context = suppress_cmmlu_custom_dataset_warning(task == "cmmlu")
            with warning_context:
                results = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=[task],
                    task_manager=task_manager,
                    batch_size=1,
                    device=device,
                    use_cache=str(task_dir),
                )
            if "samples" in results:
                results.pop("samples")
            dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
            with open(task_out_file, "w", encoding="utf-8") as f:
                f.write(dumped)

        logger.info(f"{task} Results:\n{dumped}")
        logger.info(make_table(results))


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    xhquant_init(log_file, debug)

    # 使用config文件,代替命令行参数,方便调试不同的配置
    cfg = Config.fromfile(args.config)

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = work_dir / Path(args.config).name
    cfg.dump(config_file)

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)

    model_cfg.num_logits_to_keep = 0  # 输出所有logits
    model_cfg.use_cache = True  # 关闭缓存

    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    tokenizer = xh_model.get_tokenizer()
    eval_type = LLMModelState.from_string(eval_type)
    xh_model.set_state(eval_type)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.eval()
    contexts = [TimeProfiler(args.task, logger), LLMInferenceContextManager(xh_model)]
    if device == "cuda":
        contexts.insert(1, MemoryTracker(device=device, name="lm_eval", logger=logger))

    eval_cfg = Config()
    eval_cfg.work_dir = str(work_dir)
    eval_cfg.task = args.task
    include_path = ensure_cmmlu_task_overrides(Path("data")) if args.task == "cmmlu" else None

    with ContextManagers(contexts), time_profiler() as t:
        lm_eval_engine(xh_model, tokenizer, {}, eval_cfg, device, include_path=include_path)
        logger.info(f"lm eval on task {eval_cfg.task} time: {t():.04f} s")


if __name__ == "__main__":
    valid_tasks = [
        "wikitext",
        "winogrande",
        "arc_challenge",
        "hellaswag",
        "openbookqa",
        "cmmlu",
        "gsm8k",
        "mathqa",
        "mmlu",
    ]
    eval_types = LLMModelState.get_all_values()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_2k.py"
    )
    parser.add_argument("--eval-type", type=str, default="wrap", choices=eval_types)
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--task", type=str, default="wikitext", help="lm eval task name", choices=valid_tasks)
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Whether to enable prefill chunk, only for debug and development",
    )
    args = parser.parse_args()
    task = args.task
    if task not in valid_tasks:
        raise ValueError(f"Invalid task name: {task}. Valid tasks are: {valid_tasks}")
    main(args)
