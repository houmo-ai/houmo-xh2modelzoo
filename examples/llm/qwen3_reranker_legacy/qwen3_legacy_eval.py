import argparse
import json
from pathlib import Path
from typing import Any, Optional

import lm_eval
import torch
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table
from xhquant.api import ConfigDict, get_root_logger, xhquant_init
from xhquant.xhonnxruntime import config as xh_xhonnxruntime_config

from xh_model_zoo.xh_llm.models.qwen3_legacy import Qwen3LegacyHFCompatible, Qwen3LegacyInference

"""
需要导出的HMONNX支持logits任务,默认只支持generate任务,即只输出下一个token
"""

try:
    from lm_eval.models.huggingface import HFLM
    class XH2LLM(HFLM):
        def _model_call(self, inps, attn_mask=None, labels=None):
            # self.model.use_cache = False
            self.model.prefill = True
            return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            # self.model.use_cache = True
            return super()._model_generate(context, max_length, stop, **generation_kwargs)
except:
    pass


def lm_eval_engine(hf_model: Any, tokenizer: Any, meta_info: Optional[dict] = None, cfg: Optional[ConfigDict] = None):
    logger = get_root_logger()
    lm = XH2LLM(pretrained=hf_model, tokenizer=tokenizer, max_length=2048)  # 默认max_length=40960
    lm.model.eval()
    task_manager = TaskManager()
    # tasks = ["cmmlu", "gsm8k", "mathqa", "openbookqa", "winogrande", "arc_challenge", "hellaswag"]
    tasks = [
        "wikitext",
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


def main(args):
    xhquant_init(None, args.debug)
    inference_engine = Qwen3LegacyInference(args.config)
    hf_model_path = inference_engine.meta_info.get("hf_model_path", None)
    if hf_model_path is None:
        hf_model_path = args.hf_model
    assert Path(hf_model_path).exists(), f"HF model path {hf_model_path} does not exist."
    batch_size = inference_engine.batch_size
    logger = get_root_logger()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你多大了？用中文回答。"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "中国首都是哪里？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "香港特别行政区是哪一年成立的？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "神舟五号是哪年发射的？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "中国第一个进入太空的是谁？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "人类第一个进入太空的是谁？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "人类是哪年实现载人登月的？"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "中国第一颗卫星的名字是什么？"},
        ],
    ]
    assert len(messages) >= batch_size
    messages = messages[:batch_size]
    # ids, text = inference_engine._forward(messages)
    # logger.info(f"{ids}, {text}")
    device = inference_engine.device
    tokenizer = inference_engine.tokenizer
    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)

    wraped_hf_model = Qwen3LegacyHFCompatible.to_hf_compatible(hf_model_path, inference_engine)
    wraped_hf_model.eval()
    wraped_hf_model.to(device)
    xh_xhonnxruntime_config.disable_progress = True
    if args.eval_ppl:
        from xh_model_zoo_new.evaluation.wikippl_eval import evaluate_wikitext
        wiki_ppl = evaluate_wikitext(wraped_hf_model, tokenizer, seqlen=256)
        with open(f"{args.eval_ppl}", "w", encoding="utf-8") as f:
            f.write(str(wiki_ppl))

    else:
        with torch.no_grad():
            lm_eval_engine(wraped_hf_model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/Qwen3-8B-XH2a-2k-w4a8h0_ssfp/meta.json",
    )
    parser.add_argument("--hf-model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execution_device", type=str, default="cuda:0", help="execution device, default is cuda:0")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--eval_ppl", type=str, help="only eval ppl and save path")
    args = parser.parse_args()
    main(args)
