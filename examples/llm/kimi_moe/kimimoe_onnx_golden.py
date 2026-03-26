import argparse
import json
import time
from pathlib import Path

# from examples._init_path import _init_path  # pylint: disable=unused-import # isort:skip
import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from modelscope import AutoModelForCausalLM, AutoTokenizer
from xhquant.api import Hook, set_random_seed

from xh_model_zoo.api import Config, get_root_logger, xhquant_llm_init, decode_next_token
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.kimi_moe import KimiMoeHMONNXModel
from xh_model_zoo.utils.time_profiler import TimeProfiler


def main(args):
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    # cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(Path(cfg.work_dir) / Path(args.config).name)

    golden_output_dir = Path(cfg.work_dir) / "golden"
    golden_output_dir.mkdir(exist_ok=True, parents=True)

    prefill_golden_dir = golden_output_dir / "prefill"
    if prefill_golden_dir.exists():
        logger.error(f"{prefill_golden_dir} already exists, please remove it first")
        exit(1)

    decode_golden_dir = golden_output_dir / "decode"
    # if decode_golden_dir.exists():
    #     logger.error(f"{decode_golden_dir} already exists, please remove it first")
    #     exit(1)

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息

    exec_device = cfg.exec_device
    device = cfg.device
    dtype = getattr(torch, cfg.dtype)
    batch_size = cfg.batch_size

    hf_model_config_dir = cfg.hf_model_config_dir

    tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir, trust_remote_code=True)
    token_embedding_state_dict = torch.load(cfg.embed_tokens, map_location="cpu", weights_only=True)
    token_embedding = nn.Embedding(
        token_embedding_state_dict["weight"].shape[0],
        token_embedding_state_dict["weight"].shape[1],
    )
    token_embedding.load_state_dict(token_embedding_state_dict)

    # hf_model_config = AutoConfig.from_pretrained(hf_model_dir)

    # hf_model = AutoModelForCausalLM.from_pretrained(hf_model_dir)
    # token_embedding = hf_model.model.get_input_embeddings()
    model: KimiMoeHMONNXModel = MODELS.build(cfg.model)
    model.set_input_embeddings(token_embedding)
    # hf_model = None

    torch.cuda.empty_cache()

    model.init_prefill()

    # prompt = "Give me a short introduction to large language model."
    messages = [
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language model.",
            }
        ],
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
    texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    batch_input_ids = []
    for text in texts:
        model_inputs = tokenizer([text], padding=False, return_tensors="pt")
        batch_input_ids.append(model_inputs.input_ids.cpu().numpy().tolist()[0])

    meta_info = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "input_text": text,
        "input_ids": batch_input_ids,
        "config": cfg.to_dict(),
    }
    json.dump(meta_info, open(str(golden_output_dir / "golden_meta_info.json"), "w", encoding="utf-8"), indent=4)

    data_batch = {
        "input_ids": batch_input_ids,
        "past_seq_length": [0] * batch_size,
    }

    model.eval()
    model.to(device)
    model.set_exec_device(exec_device)
    model.to(dtype)

    model.save_prefill_golden(prefill_golden_dir)
    with torch.no_grad(), TimeProfiler("prefill", logger):
        prefill_logits = model.prefill(data_batch)
        prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, prefill_logits)

    logger.info(f"Prefill next token: {prefill_next_token_id} {prefill_next_token_text}")
    model.release_prefill_session()  # release the session to save memory

    past_seq_len = [len(input_ids) for input_ids in batch_input_ids]
    batch_input_ids = prefill_next_token_id.cpu().tolist()

    model.init_decode()
    model.to(device)
    model.set_exec_device(exec_device)
    model.to(dtype)
    data_batch = {
        "input_ids": batch_input_ids,
        "past_seq_length": past_seq_len,
    }

    decode_step = 0
    while True:
        model.save_decode_golden(str(decode_golden_dir / f"step_{decode_step}"))
        with torch.no_grad(), TimeProfiler(f"decode_step_{decode_step}", logger):
            decode_logits = model.decode(data_batch)
            decode_next_token_id, decode_next_token_text = decode_next_token(tokenizer, decode_logits)
            logger.info(f"Decode next token: {decode_next_token_id} {decode_next_token_text}")
            break
    model.release_decode_session()  # release the session to save memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kimi/3b_30b/kimi_a3b_30b_instruct_lagacy_xh2a_2k_hmonnx.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)

    args = parser.parse_args()
    main(args)
