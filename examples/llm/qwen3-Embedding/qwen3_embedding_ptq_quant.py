import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoTokenizer

from xhquant.api import PrecisionMode, ptq_quantize, set_random_seed
from xhquant_llm.api import Config, EvalModelType, get_root_logger, xhquant_llm_init
from xhquant_llm.models import MODELS
from xhquant_llm.models.qwen3_legacy import XHQwen3EmbedingModel


def load_calib_texts(path: str, max_samples: int):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    texts = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                if "text" in obj and isinstance(obj["text"], str):
                    texts.append(obj["text"])
                else:
                    for v in obj.values():
                        if isinstance(v, str):
                            texts.append(v)
                            break
            if len(texts) >= max_samples:
                break
    return texts


def main():
    parser = argparse.ArgumentParser(description="PTQ quant for Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--config",
        type=str,
        default="/data01/home/feilong.kong/xhquant_llm/configs/qwen3_embeding/0.6B/qwen3_embeding_0.6b_legacy_xh2a_2k.py",
        help="xhquant_llm config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data01/home/feilong.kong/llm_models/Qwen/Qwen3-Embedding-0.6B",
        help="HF model path (override config)",
    )
    parser.add_argument("--calib-dataset", type=str, default="", help="jsonl with {text: ...}")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-seq-len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--out-dir", type=str, default="work_dirs/qwen3_embedding_0.6b_ptq")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "ptq_debug.log"
    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    cfg = Config.fromfile(args.config)
    if args.model:
        cfg.hf_model_dir = args.model
        if hasattr(cfg, "model") and isinstance(cfg.model, dict):
            cfg.model["hf_model"] = args.model
    if hasattr(cfg, "model") and isinstance(cfg.model, dict):
        wrap_cfg = cfg.model.get("wrap_cfg", {})
        wrap_cfg["max_sequence_length"] = args.context_length
        wrap_cfg["input_sequence_length"] = args.input_seq_len
        cfg.model["wrap_cfg"] = wrap_cfg

    cfg.device = args.device if torch.cuda.is_available() else "cpu"
    cfg.exec_device = cfg.device
    cfg.dtype = "float16"

    xh_model: XHQwen3EmbedingModel = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3EmbedingModel)

    tokenizer = xh_model.get_tokenizer()
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

    texts = load_calib_texts(args.calib_dataset, args.calib_samples)
    if not texts:
        texts = [
            "What year did humans first land on the Moon?",
            "Explain the difference between GPU and CPU.",
            "How much protein should a female eat per day?",
        ]

    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.input_seq_len,
        return_tensors="pt",
    )
    input_ids = batch.input_ids

    native_model = xh_model.get_hf_model("cpu")
    xh_model.init_wrap_model(native_model)

    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    xh_model.to(device)
    xh_model.to(dtype)

    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": 0,
    }

    xh_model.change_eval_type(EvalModelType.WRAPED)
    xh_model.interactive_mode = True
    xh_model.convert_to_fronted_graph(data_batch)
    xh_model.convert_to_quant_graph(cfg.target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()

    calib_data = xh_model.prepare_inputs(data_batch)
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (list, tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args

    logger.info("*************** Start PTQ Quantize ***************")
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [device])
    logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    # Save quantized state dict
    quant_state = xh_model.quanted_model.state_dict()
    out_path = out_dir / "ptq-state-dict.safetensors"
    save_safetensors_file(quant_state, str(out_path))
    logger.info(f"Saved PTQ state dict: {out_path}")


if __name__ == "__main__":
    main()
