import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from xhquant.api import set_random_seed

# Ensure local repo package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.api import Config, decode_next_token, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.llm_onnx_model import LLMONNXModel

try:
    from xh_model_zoo.xh_llm.models.llm_onnx_model import LLMLoRAONNXModel
except ImportError:
    LLMLoRAONNXModel = LLMONNXModel


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model-config",
        type=str,
        default="work_dirs/fireredasr_llm_merged/export_meta_info.json",
    )
    parser.add_argument("--prompt", type=str, default="请转写音频为文字")
    parser.add_argument("--max_decode_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--exec_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    return parser


def build_hmonnx_model(model_dir_path: Path, meta_info: dict):
    lora_mode = meta_info.get("lora_mode", "merge_lora")
    model_cls = LLMLoRAONNXModel if lora_mode == "keep_lora" else LLMONNXModel
    prefill_seq_len = int(meta_info.get("wrap_cfg", {}).get("input_sequence_length", 256))
    model_kwargs = dict(
        prefill=SimpleNamespace(
            onnx=str(model_dir_path / meta_info["prefill_onnx_file"]),
            input_sequence_length=prefill_seq_len,
        ),
        decode=SimpleNamespace(
            onnx=str(model_dir_path / meta_info["decode_onnx_file"]),
        ),
        kv_cache=SimpleNamespace(
            num_hidden_layers=meta_info["num_hidden_layers"],
            shape=meta_info["kv_cache_shape"],
        ),
    )
    if lora_mode == "keep_lora":
        model_kwargs["use_lora_mask"] = bool(meta_info.get("keep_lora_use_mask", True))
    model = model_cls(**model_kwargs)
    return model, lora_mode


def main():
    args = parse_arguments().parse_args()
    model_meta_file_path = Path(args.model_config)
    meta_info = json.load(open(model_meta_file_path, "r"))
    model_dir_path = model_meta_file_path.parent
    model_name = meta_info["model_name"]

    cfg = Config(
        dict(
            model_config=args.model_config,
            exec_device=args.exec_device,
        )
    )
    cfg_name = f"{model_name}_hmonnx_golden"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"

    set_random_seed(args.seed)
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")

    hf_cfg_dir = str(model_dir_path / meta_info["hf_config"])
    tokenizer = AutoTokenizer.from_pretrained(hf_cfg_dir)

    token_embedding_state_dict = torch.load(
        str(model_dir_path / meta_info["token_embedding_file"]),
        map_location="cpu",
        weights_only=True,
    )
    token_embedding = nn.Embedding(
        token_embedding_state_dict["weight"].shape[0],
        token_embedding_state_dict["weight"].shape[1],
    )
    token_embedding.load_state_dict(token_embedding_state_dict)

    model, lora_mode = build_hmonnx_model(model_dir_path, meta_info)
    model.set_input_embeddings(token_embedding)
    model.set_exec_device(args.exec_device)
    model.to(args.exec_device)
    model.to(torch.float16)

    golden_output_dir = Path(cfg.work_dir) / "golden"
    prefill_golden_dir = golden_output_dir / "prefill"
    decode_golden_dir = golden_output_dir / "decode"
    prefill_golden_dir.mkdir(exist_ok=True, parents=True)
    decode_golden_dir.mkdir(exist_ok=True, parents=True)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs.input_ids

    golden_meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "input_text": text,
        "input_ids": input_ids.tolist(),
        "source_export_meta_info": str(model_meta_file_path),
        "lora_mode": lora_mode,
    }
    json.dump(golden_meta, open(str(golden_output_dir / "golden_meta_info.json"), "w"), indent=4, ensure_ascii=False)

    prefill_data = {
        "input_ids": input_ids.to(args.exec_device),
        "past_seq_length": 0,
    }
    model.init_prefill()
    model.save_prefill_golden(str(prefill_golden_dir))
    with torch.no_grad():
        prefill_logits = model.prefill(prefill_data)
        next_token_id, next_token_text = decode_next_token(tokenizer, prefill_logits)
    logger.info(f"[prefill] next token: {next_token_id} {next_token_text}")
    model.release_prefill_session()

    model.init_decode()
    model.save_decode_golden(str(decode_golden_dir))
    decode_step = 0
    past_seq_length = input_ids.shape[-1]
    decode_data = {
        "input_ids": next_token_id.to(args.exec_device),
        "past_seq_length": past_seq_length,
    }
    eos_token_id = tokenizer.eos_token_id
    decode_tokens = []

    while decode_step < args.max_decode_steps:
        with torch.no_grad():
            decode_logits = model.decode(decode_data)
            next_token_id, next_token_text = decode_next_token(tokenizer, decode_logits)

        token_id = int(next_token_id[0][0].item())
        logger.info(f"[decode:{decode_step}] {next_token_id} {next_token_text}")
        if token_id == eos_token_id:
            break
        decode_tokens.extend(next_token_text)
        past_seq_length += 1
        decode_step += 1
        decode_data = {
            "input_ids": next_token_id.to(args.exec_device),
            "past_seq_length": past_seq_length,
        }

    logger.info(f"Golden export done. lora_mode={lora_mode}")
    logger.info(f"Prefill golden: {prefill_golden_dir}")
    logger.info(f"Decode golden: {decode_golden_dir}")


if __name__ == "__main__":
    main()
