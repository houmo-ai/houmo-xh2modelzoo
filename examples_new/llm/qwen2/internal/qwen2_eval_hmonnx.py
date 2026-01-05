import argparse
import json
import os.path as osp
from pathlib import Path
from typing import Any, Optional, List

import torch, torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xhquant.api import ConfigDict, get_root_logger, xhquant_init
from xh_model_zoo_new.xh_llm.base_llm_infer_adapter import BaseLLMHFCompatible

torch.set_grad_enabled(False)


def lm_eval_engine(hf_model: Any, tokenizer: Any, tasks: List[str], results_path: str = None):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable, make_table
    from lm_eval.models.huggingface import HFLM

    logger = get_root_logger()
    lm = HFLM(pretrained=hf_model, tokenizer=tokenizer, max_length=2048)
    lm.model.eval()
    task_manager = TaskManager()
    # tasks = ["cmmlu", "gsm8k", "mathqa", "openbookqa", "winogrande", "arc_challenge", "hellaswag", "wikitext"]
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, task_manager=task_manager, batch_size=1, device="cuda")
    if results is not None:
        # if "samples" in results:
        #     samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)

        if results_path is not None:
            result_json_file = Path(results_path)
            with open(result_json_file, "w", encoding="utf-8") as f:
                f.write(dumped)

        logger.info(f"Results:\n{dumped}")
        logger.info(make_table(results))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=str, required=True, help="work directory containing exported onnx files")
    parser.add_argument("--demo_prompt", type=str, default="你多大了？用中文回答。", help="demo prompt")
    parser.add_argument("--tasks", nargs="+", type=str, default=None, help="List of tasks for evaluation")
    parser.add_argument("--offload", action="store_true", help="offload mode")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


@torch.no_grad()
def main(args):
    xhquant_init(None, args.debug)
    export_dir = Path(args.export_dir)

    # Load meta_info from work_dir/meta.json
    meta_info_path = export_dir / "meta.json"
    if not meta_info_path.exists():
        raise FileNotFoundError(f"meta.json not found in {export_dir}")

    with open(meta_info_path, "r", encoding="utf-8") as f:
        meta_info = json.load(f)

    # Get onnx paths from meta_info
    prefill_path = osp.join(export_dir, meta_info["prefill_onnx"])
    decoder_path = osp.join(export_dir, meta_info["decode_onnx"])
    tokenizer = AutoTokenizer.from_pretrained(osp.join(export_dir, meta_info["hf_config"]))
    hf_config = AutoConfig.from_pretrained(osp.join(export_dir, meta_info["hf_config"]))
    wrap_cfg = meta_info["wrap_cfg"]
    token_embedding_state_dict = torch.load(osp.join(export_dir, meta_info["token_embedding_file"]))
    token_embedding = nn.Embedding(
        token_embedding_state_dict["weight"].shape[0], token_embedding_state_dict["weight"].shape[1]
    )
    token_embedding.load_state_dict(token_embedding_state_dict)

    # Create model from HMONNX
    model = BaseLLMHFCompatible.from_hmonnx(
        prefill_path,
        decoder_path,
        hf_config,
        wrap_cfg,
        token_embedding,
        native_model_or_path=None,
        tokenizer=tokenizer,
    )
    if not args.offload:
        model.cuda().half()
    else:
        raise NotImplementedError("Offload mode is not implemented")

    if args.demo_prompt is not None:
        print(model.demo(args.demo_prompt))

    # Run lm_eval
    if args.tasks is not None:
        lm_eval_engine(model, tokenizer, args.tasks, results_path=Path(export_dir) / "eval_results_hmonnx.json")


if __name__ == "__main__":
    args = parse_args()
    main(args)
