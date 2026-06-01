"""Verify text wrap decode: prefill + 1 decode step, compare with HF."""
import torch
from transformers import AutoModelForImageTextToText

from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhmodel_merak.xh_llm.models.gemma4.gemma4_processor import XHGemma4Processor


def main():
    device = "cuda:0"
    cfg = Config.fromfile(
        "configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py"
    )
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(model_cfg)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.hf_model,
        dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()
    processor = XHGemma4Processor.from_pretrained(model_cfg.hf_model)
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "Hello world, test."}]}]
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]
    print(f"seq_len={seq_len}")

    # --- HF: prefill + 1 decode step with use_cache=True ---
    with torch.no_grad():
        hf_out = hf_model(**inputs, use_cache=True)
    hf_prefill_logits = hf_out.logits  # (1, seq_len, vocab)
    hf_kv = hf_out.past_key_values
    next_token = hf_prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
    print(f"HF prefill next token: {next_token.item()}")

    # HF decode
    hf_decode_inputs = {
        "input_ids": next_token,
        "past_key_values": hf_kv,
        "use_cache": True,
    }
    with torch.no_grad():
        hf_decode_out = hf_model(**hf_decode_inputs)
    hf_decode_logits = hf_decode_out.logits  # (1, 1, vocab)
    print(f"HF decode logits shape: {hf_decode_logits.shape}")

    # --- Wrap: prefill + 1 decode step ---
    xh_model.init_wrap_model(hf_model)
    xh_model._device = device
    xh_model._dtype = torch.bfloat16
    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)

    image_embeds = torch.zeros(1, 0, 3072, dtype=torch.bfloat16, device=device)

    # Prefill
    prefill_data = {
        "input_ids": inputs["input_ids"],
        "mm_token_type_ids": inputs.get("mm_token_type_ids"),
        "image_embeds": image_embeds,
        "past_seq_length": 0,
    }
    wrap_inputs = xh_model.get_data_preprocessor()(prefill_data)
    with torch.no_grad():
        wrap_prefill = xh_model.wrap_model(*wrap_inputs)
    wrap_next_token = wrap_prefill[:, seq_len - 1, :].argmax(dim=-1, keepdim=True)
    print(f"Wrap prefill next token: {wrap_next_token.item()}")

    # Verify prefill logits match
    hf_last = hf_prefill_logits[:, -1, :].float()
    wrap_last = wrap_prefill[:, seq_len - 1, :].float()
    cos_prefill = torch.nn.functional.cosine_similarity(
        hf_last.flatten(), wrap_last.flatten(), dim=0
    ).item()
    print(f"Prefill last-token cosine: {cos_prefill:.8f}")

    # Decode — past_seq_length = actual seq_len (not padded chunk size),
    # so position_ids and cache indexing align with HF's KV layout.
    decode_data = {
        "input_ids": next_token,
        "mm_token_type_ids": torch.zeros_like(next_token),
        "image_embeds": torch.zeros(1, 0, 3072, dtype=torch.bfloat16, device=device),
        "past_seq_length": seq_len,
    }
    wrap_decode_inputs = xh_model.get_data_preprocessor()(decode_data)
    with torch.no_grad():
        wrap_decode = xh_model.wrap_model(*wrap_decode_inputs)
    # decode output: first token position
    wrap_decode_logits = wrap_decode[:, 0, :].float()
    hf_decode_last = hf_decode_logits[:, -1, :].float()
    cos_decode = torch.nn.functional.cosine_similarity(
        hf_decode_last.flatten(), wrap_decode_logits.flatten(), dim=0
    ).item()
    print(f"Decode cosine: {cos_decode:.8f}")

    hf_decode_token = hf_decode_logits[:, -1, :].argmax(dim=-1).item()
    wrap_decode_token = wrap_decode[:, 0, :].argmax(dim=-1).item()
    print(f"HF decode token: {hf_decode_token}, Wrap decode token: {wrap_decode_token}")
    print(f"Tokens match: {hf_decode_token == wrap_decode_token}")


if __name__ == "__main__":
    main()
