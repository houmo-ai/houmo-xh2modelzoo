import argparse
import gc
import json
from pathlib import Path

import torch


SAMPLE_RATE = 16000
MAX_GEN_LEN = 448


def _clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_meta(work_dir: Path) -> dict:
    meta_path = work_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _build_prompt_tokens(
    processor, model_config, meta=None,
    language="zh", task="transcribe", notimestamps=True,
) -> list[int]:
    """Build prompt tokens for the decoder prefix.

    Reads ``prefill.prompt_token_ids`` from ``export_meta_info.json`` when
    available.  If the meta tokens match the requested ``language`` / ``task`` /
    ``notimestamps`` they are returned as-is (trusting the export).  On mismatch
    a warning is printed and the requested tokens are used — the demo always
    honours the explicit prompt parameters so that callers get predictable
    output regardless of which export generated the meta file.
    """
    expected = _resolve_token_ids(processor, model_config, language, task, notimestamps)
    if meta and isinstance(meta.get("prefill"), dict) and "prompt_token_ids" in meta["prefill"]:
        meta_ids = list(meta["prefill"]["prompt_token_ids"])
        if meta_ids != expected:
            print(
                "WARNING: export_meta_info.json prefill.prompt_token_ids "
                f"{meta_ids} differ from requested prompt "
                f"({language=}, {task=}, {notimestamps=}) -> {expected}. "
                "Using requested tokens."
            )
            return expected
        return meta_ids
    return expected


def _resolve_token_ids(processor, model_config, language="zh", task="transcribe", notimestamps=True) -> list[int]:
    tokenizer = processor.tokenizer
    unk_id = tokenizer.unk_token_id

    def _resolve(token_str: str, param_name: str) -> int:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is None or tid == unk_id:
            raise ValueError(
                f"Cannot resolve {param_name} token '{token_str}' "
                f"(got id={tid}). Check --prompt-{param_name} value."
            )
        return tid

    sot_id = model_config.decoder_start_token_id
    lang_id = _resolve(f"<|{language}|>", "language")
    task_id = _resolve(f"<|{task}|>", "task")
    tokens = [sot_id, lang_id, task_id]
    if notimestamps:
        tokens.append(_resolve("<|notimestamps|>", "no-timestamps"))
    return tokens


_BASE_INPUT_NAMES = [
    "decoder_input_ids",
    "cache_position",
    "past_len",
    "current_len",
    "mask_atten",
    "encoder_attention_mask",
]


def _validate_dec_names(dec_names: list[str], num_decode_layers: int) -> None:
    import re
    expected = 6 + 4 * num_decode_layers
    if len(dec_names) != expected:
        raise ValueError(
            f"Expected {expected} decoder graph inputs "
            f"({6} base + {4 * num_decode_layers} cache/key/value), "
            f"got {len(dec_names)}.\n"
            f"Input names: {dec_names}"
        )
    for i, expected_name in enumerate(_BASE_INPUT_NAMES):
        if dec_names[i] != expected_name:
            raise ValueError(
                f"Unexpected base name at position {i}: "
                f"expected '{expected_name}', got '{dec_names[i]}'.\n"
                f"Full input names: {dec_names}"
            )
    pattern = re.compile(r"^(\w+)_(\d+)$")
    for g in range(4):
        base_offset = 6 + g * num_decode_layers
        for layer_idx in range(num_decode_layers):
            idx = base_offset + layer_idx
            name = dec_names[idx]
            m = pattern.match(name)
            if not m:
                raise ValueError(
                    f"Input at position {idx} ('{name}') does not follow "
                    f"the expected pattern '<prefix>_<layer>'.\n"
                    f"Full input names: {dec_names}"
                )
            parsed_layer = int(m.group(2))
            if parsed_layer != layer_idx:
                raise ValueError(
                    f"Expected layer {layer_idx} at position {idx}, "
                    f"got layer {parsed_layer} from name '{name}'.\n"
                    f"Full input names: {dec_names}"
                )


def _build_dec_inputs(
    *,
    dec_names,
    input_ids,
    position_ids,
    past_len,
    current_len,
    mask_atten,
    encoder_attention_mask,
    k_cache,
    v_cache,
    k_list,
    v_list,
    num_decode_layers,
):
    """Assemble the decoder/prefill graph inputs by name.

    Graph input order (non-interleaved, matches the exported HMONNX):
      6 base + k_cache[0..N] + v_cache[0..N] + key_state[0..N] + value_state[0..N]
    """
    _validate_dec_names(dec_names, num_decode_layers)
    inputs = {
        dec_names[0]: input_ids,
        dec_names[1]: position_ids,
        dec_names[2]: past_len,
        dec_names[3]: current_len,
        dec_names[4]: mask_atten,
        dec_names[5]: encoder_attention_mask,
    }
    base_idx = 6
    for layer_idx in range(num_decode_layers):
        inputs[dec_names[base_idx + layer_idx]] = k_cache[layer_idx]
        inputs[dec_names[base_idx + num_decode_layers + layer_idx]] = v_cache[layer_idx]
        inputs[dec_names[base_idx + num_decode_layers * 2 + layer_idx]] = k_list[layer_idx]
        inputs[dec_names[base_idx + num_decode_layers * 3 + layer_idx]] = v_list[layer_idx]
    return inputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Whisper HMONNX transcription.")
    parser.add_argument("--hf-model", dest="hf_model", required=True, help="Whisper HF model directory.")
    parser.add_argument(
        "--work-dir",
        dest="work_dir",
        required=True,
        help="Whisper export output directory (contains export_meta_info.json).",
    )
    parser.add_argument("--audio", required=True, help="Path to an audio file (wav/mp3).")
    parser.add_argument("--device", default="cuda", help="Device label. Default: cuda")
    parser.add_argument(
        "--prompt-language",
        default="zh",
        help="Language token for the decoder prompt prefix (default: zh). "
             "Example: en, zh, ja. Only used when export_meta_info.json lacks "
             "prefill.prompt_token_ids.",
    )
    parser.add_argument(
        "--prompt-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task token for the decoder prompt prefix (default: transcribe). "
             "Only used as fallback when meta has no prompt_token_ids.",
    )
    parser.add_argument(
        "--prompt-no-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include <|notimestamps|> in the prompt prefix (default: true). "
             "Only used as fallback when meta has no prompt_token_ids.",
    )
    return parser.parse_args(argv)


def main(args=None) -> None:
    args = parse_args(args) if args is None else args
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from xhquant.api import HMONNXInference
    from xhquant.core.cache_tensor import CacheTensor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    work_dir = Path(args.work_dir)
    meta = _load_meta(work_dir)
    model_cfg = meta["model_cfg"]
    num_heads = int(model_cfg["num_heads"])
    head_dim = int(model_cfg["head_dim"])
    num_decode_layers = int(model_cfg["num_decode_layers"])
    max_source_positions = int(model_cfg["max_source_positions"])

    processor = WhisperProcessor.from_pretrained(args.hf_model)
    model_config = WhisperForConditionalGeneration.from_pretrained(args.hf_model).config
    prompt_tokens = _build_prompt_tokens(
        processor, model_config, meta,
        language=args.prompt_language,
        task=args.prompt_task,
        notimestamps=args.prompt_no_timestamps,
    )
    prefill_cfg = meta.get("prefill", {})
    expected_prompt_len = len(prefill_cfg.get("cache_position", prompt_tokens))
    if len(prompt_tokens) != expected_prompt_len:
        raise ValueError(
            f"Prompt length {len(prompt_tokens)} does not match the exported "
            f"prefill graph input size {expected_prompt_len}. "
            f"Adjust --prompt-language/--prompt-task/--prompt-no-timestamps "
            f"or re-export with matching prompt_token_ids."
        )
    eos_id = model_config.eos_token_id

    print(f">>> 加载音频文件: {args.audio}")
    audio_array, sr = sf.read(args.audio)
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)

    # --- Encoder ---
    print(">>> 运行 Encoder...")
    encoder = HMONNXInference(str(work_dir / meta["encoder"]["hmonnx_file"]))
    # to_fast_mode() must stay off: the current xhquant runtime raises
    # "non-positive groups is not supported" in its fast-mode conv2d path on the
    # encoder Conv1d->Conv2d layers. Default path is correct (matches golden).
    encoder.to(device)
    input_features = (
        processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        .input_features.half()
        .to(device)
    )
    enc_out = encoder(input_features)
    k_list = enc_out[:num_decode_layers]
    v_list = enc_out[num_decode_layers:]
    del encoder
    _clean_memory()

    # --- Decoder (prefill + decode) ---
    print(">>> 准备 Decoder...")
    prefill = HMONNXInference(str(work_dir / meta["prefill"]["hmonnx_file"]))
    prefill.to(device)
    decoder = HMONNXInference(str(work_dir / meta["decoder"]["hmonnx_file"]))
    decoder.to(device)

    dec_names = decoder.get_input_names()
    cache_max_len = decoder.get_input(dec_names[6]).shape[2]
    k_cache = [
        CacheTensor(torch.zeros([1, num_heads, cache_max_len, head_dim], device=device, dtype=torch.float16))
        for _ in range(num_decode_layers)
    ]
    v_cache = [
        CacheTensor(torch.zeros([1, num_heads, cache_max_len, head_dim], device=device, dtype=torch.float16))
        for _ in range(num_decode_layers)
    ]
    encoder_attention_mask = torch.zeros((1, 1, 1, max_source_positions), device=device, dtype=torch.float16)

    # === Prefill ===
    print(">>> 开始 Prefill ...")
    input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.int32)
    prompt_len = len(prompt_tokens)
    position_ids = torch.arange(prompt_len, device=device, dtype=torch.int32).unsqueeze(0)
    mask_atten = torch.full(
        (1, num_heads, prompt_len, cache_max_len), -65504.0, device=device, dtype=torch.float16
    )
    for i in range(prompt_len):
        mask_atten[:, :, i, : i + 1] = 0.0

    inputs = _build_dec_inputs(
        dec_names=dec_names,
        input_ids=input_ids,
        position_ids=position_ids,
        past_len=torch.tensor([0], device=device, dtype=torch.int32),
        current_len=torch.tensor([prompt_len], device=device, dtype=torch.int32),
        mask_atten=mask_atten,
        encoder_attention_mask=encoder_attention_mask,
        k_cache=k_cache,
        v_cache=v_cache,
        k_list=k_list,
        v_list=v_list,
        num_decode_layers=num_decode_layers,
    )
    out = prefill.run(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logits = out[0]
    k_cache = [CacheTensor(t) for t in out[1 : 1 + num_decode_layers]]
    v_cache = [CacheTensor(t) for t in out[1 + num_decode_layers : 1 + 2 * num_decode_layers]]

    step = prompt_len
    print(">>> Prefill 完成，开始 Generate...")
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    all_tokens_list: list[int] = []
    decoded_text = ""

    # === Decode loop ===
    while step < MAX_GEN_LEN:
        if next_token == eos_id:
            suffix = f"Step {step}: {decoded_text} [EOS]" if decoded_text else f"Step {step}: [EOS]"
            print(f"\n{suffix}")
            break

        all_tokens_list.append(next_token)
        all_tokens = torch.tensor([all_tokens_list], device=device)
        decoded_text = processor.batch_decode(all_tokens, skip_special_tokens=True)[0]
        print(f"\r\033[KStep {step}: {decoded_text}", end="", flush=True)

        input_ids = torch.tensor([[next_token]], device=device, dtype=torch.int32)
        mask_atten = torch.zeros((1, num_heads, 1, cache_max_len), device=device, dtype=torch.float16)
        if step + 1 < cache_max_len:
            mask_atten[:, :, :, step + 1 :] = -65504

        inputs = _build_dec_inputs(
            dec_names=dec_names,
            input_ids=input_ids,
            position_ids=torch.tensor([[step]], device=device, dtype=torch.int32),
            past_len=torch.tensor([step], device=device, dtype=torch.int32),
            current_len=torch.tensor([1], device=device, dtype=torch.int32),
            mask_atten=mask_atten,
            encoder_attention_mask=encoder_attention_mask,
            k_cache=k_cache,
            v_cache=v_cache,
            k_list=k_list,
            v_list=v_list,
            num_decode_layers=num_decode_layers,
        )
        out = decoder.run(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logits = out[0]
        k_cache = [CacheTensor(t) for t in out[1 : 1 + num_decode_layers]]
        v_cache = [CacheTensor(t) for t in out[1 + num_decode_layers : 1 + 2 * num_decode_layers]]
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        step += 1

    print("\n推理完成！")
    print(f"最终结果: {decoded_text}")
    del decoder
    _clean_memory()


if __name__ == "__main__":
    main()
