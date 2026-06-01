import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModel
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.qwen3_embedding import (
    Qwen3EmbeddingHFCompatible,
    Qwen3EmbeddingInference,
    _linear_impl,  # noqa: F401
    _llm_model_impl,  # noqa: F401
)
from xhquant.api import get_root_logger, xhquant_init
from xhquant.utils.config import Config
from xhquant.xhonnxruntime import config as xhonnxruntime_config


def select_last_token_embedding(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    # [B, H]: already pooled by export config (num_logits_to_keep=1)
    if last_hidden_states.dim() == 2:
        return last_hidden_states

    # [B, 1, H]: keep the only token output
    if last_hidden_states.shape[1] == 1:
        return last_hidden_states[:, 0, :]

    if attention_mask.device != last_hidden_states.device:
        attention_mask = attention_mask.to(last_hidden_states.device)
    if attention_mask.shape[1] != last_hidden_states.shape[1]:
        attention_mask = attention_mask[:, : last_hidden_states.shape[1]]

    # EOS is expected to be the last valid token.
    last_index = attention_mask.sum(dim=1).to(torch.long) - 1
    last_index = torch.clamp(last_index, min=0, max=last_hidden_states.shape[1] - 1)
    batch_index = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_index, last_index, :]


def main(args):
    xhquant_init(None, args.debug)
    inference_engine = Qwen3EmbeddingInference(
        args.config,
        fast_mode=args.fast,
        device=args.device,
        execution_device=args.execution_device,
    )
    logger = get_root_logger()

    hf_model_path = inference_engine.meta_info.get("hf_model_path", None)
    if args.hf_model:
        hf_model_path = args.hf_model
    assert hf_model_path is not None and Path(hf_model_path).exists(), (
        f"HF model path {hf_model_path} does not exist."
    )

    if args.long_input:
        base = "这是一个用于测试长输入的句子。"
        long_text = base * (args.long_len // len(base) + 1)
        texts = [long_text[: args.long_len]]
    else:
        texts = [
            "What year did humans first land on the Moon?",
        ]

    tokenizer = inference_engine.tokenizer
    input_seq_len = inference_engine.meta_info.get("wrap_cfg", {}).get(
        "input_sequence_length", args.long_len
    )
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=input_seq_len,
        return_tensors="pt",
    )
    device = inference_engine.device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # HMONNX path via HF compatible wrapper
    hmonnx_model = Qwen3EmbeddingHFCompatible.to_hf_compatible(
        hf_model_path, inference_engine
    )
    hmonnx_model.eval()
    xhonnxruntime_config.disable_progress = True
    with torch.no_grad():
        hmonnx_outputs = hmonnx_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
    hmonnx_hidden = hmonnx_outputs.last_hidden_state
    hmonnx_emb = select_last_token_embedding(hmonnx_hidden, attention_mask)
    hmonnx_emb = F.normalize(hmonnx_emb, p=2, dim=1)

    logger.info(f"HMONNX embeddings shape: {tuple(hmonnx_emb.shape)}")

    if args.compare:
        # 0.6B uses PTQ only; compare against original FP weights
        model_name = inference_engine.meta_info.get("model_name", "")
        is_0_6b = "0.6b" in model_name.lower() or "0.6b" in str(hf_model_path).lower()
        quant_weight = None
        if not is_0_6b:
            quant_weight = args.quant_weight or inference_engine.meta_info.get(
                "quant_weight", None
            )
        wrap_cfg = inference_engine.meta_info.get("wrap_cfg", None)
        kv_cache_meta = inference_engine.meta_info.get("kv_cache", None)
        if wrap_cfg is None or kv_cache_meta is None:
            raise ValueError("Missing wrap_cfg/kv_cache in meta.json for HF compare.")
        hf_model = AutoModel.from_pretrained(
            hf_model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        if quant_weight:
            state_dict = load_safetensors_file(quant_weight)
            if "post_norm_linear.weight" in state_dict:
                hidden_size = hf_model.config.hidden_size
                hf_model.post_norm_linear = torch.nn.Linear(
                    hidden_size, hidden_size, bias=False
                )
            hf_model.load_state_dict(state_dict, strict=False)
        wrap_cfg = Config(wrap_cfg)
        hf_model = wrap_llm_model(hf_model, wrap_cfg)
        hf_model = hf_model.to(device)
        hf_model.eval()
        with torch.no_grad():
            if hasattr(hf_model, "model"):
                inputs_embeds = hf_model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = hf_model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(device=device, dtype=torch.float16)
            batch_size = input_ids.shape[0]
            past_seq_length_t = torch.zeros(
                (batch_size,), dtype=torch.int32, device=device
            )
            current_input_length_t = attention_mask.sum(dim=1).to(torch.int32)
            kv_shape = kv_cache_meta["shape"]
            kv_shape = [batch_size, kv_shape[1], kv_shape[2], kv_shape[3]]
            num_layers = kv_cache_meta["num_decoder_layers"]
            past_key_caches = [
                torch.zeros(kv_shape, dtype=torch.float16, device=device)
                for _ in range(num_layers)
            ]
            past_value_caches = [
                torch.zeros(kv_shape, dtype=torch.float16, device=device)
                for _ in range(num_layers)
            ]
            hf_outputs = hf_model(
                inputs_embeds,
                past_seq_length_t,
                current_input_length_t,
                past_key_caches,
                past_value_caches,
            )
        hf_hidden = hf_outputs.last_hidden_state
        hf_emb = select_last_token_embedding(hf_hidden, attention_mask)
        hf_emb = F.normalize(hf_emb, p=2, dim=1)

        if hmonnx_emb.device != hf_emb.device:
            hmonnx_emb = hmonnx_emb.to(hf_emb.device)
        diff = (hf_emb - hmonnx_emb).abs()
        logger.info(f"Max abs diff: {diff.max().item():.6e}")
        logger.info(f"Mean abs diff: {diff.mean().item():.6e}")
    else:
        logger.info(f"HMONNX embeddings[0][:8]: {hmonnx_emb[0][:8].tolist()}")


if __name__ == "__main__":
    # import debugpy

    # debugpy.listen(("0.0.0.0", 1160))
    # print("✅ debugpy listening on 0.0.0.0:5678, waiting for VSCode attach...")
    # debugpy.wait_for_client()
    # print("✅ VSCode attached, continue running.")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/Qwen3-Embedding-4B-XH2a-2k-w8a8h0_ssfp/meta.json",
    )
    parser.add_argument("--hf-model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--execution_device",
        type=str,
        default="cuda:0",
        help="execution device, default is cuda:0",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument(
        "--compare", action="store_true", help="compare with native HF model"
    )
    parser.add_argument(
        "--quant-weight", type=str, help="gptq+quarot safetensors for HF comparison"
    )
    parser.add_argument(
        "--long-input", action="store_true", help="use a long single input"
    )
    parser.add_argument("--long-len", type=int, default=1000, help="long input length")
    args = parser.parse_args()
    main(args)
