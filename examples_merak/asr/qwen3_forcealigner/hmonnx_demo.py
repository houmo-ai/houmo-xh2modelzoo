import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-ForceAligner HMONNX inference.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate_audio_feature_length(
    feature_length: int,
    max_audio_length: int,
) -> None:
    if feature_length > max_audio_length:
        raise ValueError(
            "Audio input is too long for the exported encoder: "
            f"feature_length={feature_length}, "
            f"max_audio_length={max_audio_length}. "
            "Use a shorter audio file or re-export with a larger "
            "--max-audio-length."
        )


def _validate_prefill_length(
    input_length: int,
    max_prefill_length: int,
) -> None:
    if input_length > max_prefill_length:
        raise ValueError(
            "Forced-alignment input is too long for the exported prefill "
            f"graph: input_length={input_length}, "
            f"prefill.sequence_length={max_prefill_length}. "
            "Shorten the audio/text input or re-export with a larger "
            "--sequence-length."
        )


def main():
    import librosa
    import numpy as np
    import torch
    import torch.nn as nn
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
    from transformers import AutoConfig
    from xhquant.api import HMONNXInference
    from xhquant.core import CacheTensor

    args = parse_args()
    work_dir = Path(args.work_dir)
    meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    max_audio_length = int(meta["encoder"]["model_cfg"]["fixed_max_audio_length"])
    max_prefill = int(meta["prefill_input_sequence_length"])

    processor = Qwen3ASRProcessor.from_pretrained(
        str(work_dir / meta["hf_config"]), fix_mistral_regex=True
    )
    config = AutoConfig.from_pretrained(str(work_dir / meta["hf_config"]), trust_remote_code=True)
    aligner = Qwen3ForceAlignProcessor()
    embedding_state = torch.load(work_dir / meta["token_embedding_file"], map_location="cpu")
    weight = embedding_state["weight"]
    embedding = nn.Embedding(*weight.shape).to(device, dtype=torch.float16).eval()
    embedding.weight.data.copy_(weight.to(device=device, dtype=torch.float16))

    encoder = HMONNXInference(str(work_dir / meta["encoder"]["hmonnx_file"]))
    encoder.to(str(device))
    prefill = HMONNXInference(str(work_dir / meta["prefill_onnx_file"]))
    prefill.to(str(device))

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    words, aligner_text = aligner.encode_timestamp(args.text, args.language)
    inputs = processor(
        text=[aligner_text], audio=[audio.astype(np.float32)], return_tensors="pt", padding=True
    ).to(device)
    feature_lens = inputs["feature_attention_mask"].sum(
        dim=-1
    ).to(torch.int32)
    feature_length = int(feature_lens.max().item())
    _validate_audio_feature_length(feature_length, max_audio_length)
    features = inputs["input_features"]
    if features.shape[2] < max_audio_length:
        features = torch.nn.functional.pad(
            features,
            (0, max_audio_length - features.shape[2]),
        )
    audio_embeds = encoder.run(
        {"input_features": features.to(torch.float16), "feature_lens": feature_lens}
    )
    if isinstance(audio_embeds, (list, tuple)):
        audio_embeds = audio_embeds[0]
    if isinstance(audio_embeds, np.ndarray):
        audio_embeds = torch.from_numpy(audio_embeds)
    audio_embeds = audio_embeds.to(device=device, dtype=torch.float16)

    input_ids = inputs["input_ids"]
    tokenizer = processor.tokenizer
    audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    pad_indices = (input_ids == audio_pad_id).nonzero(as_tuple=True)[1]
    start, end = pad_indices[0].item(), pad_indices[-1].item()
    audio_token_count = end - start + 1
    if audio_token_count > audio_embeds.shape[1]:
        raise RuntimeError(
            "Encoder returned fewer audio embeddings than the processor "
            f"requested: embeddings={audio_embeds.shape[1]}, "
            f"audio_pad_tokens={audio_token_count}."
        )
    audio_embeds = audio_embeds[:, :audio_token_count]
    text_embeds = embedding(input_ids)
    merged = torch.cat(
        [
            text_embeds[:, :start],
            audio_embeds,
            text_embeds[:, end + 1 :],
        ],
        dim=1,
    )
    valid_length = int(merged.shape[1])
    _validate_prefill_length(valid_length, max_prefill)
    prefill_embeds = torch.zeros(
        (1, max_prefill, weight.shape[1]),
        dtype=torch.float16,
        device=device,
    )
    prefill_embeds[:, :valid_length] = merged[:, :valid_length]

    cache_shape = tuple(int(dim) for dim in meta["kv_cache_shape"])
    input_names = prefill.get_input_names()
    embed_name = "input_embeds" if "input_embeds" in input_names else "inputs_embeds"
    past_length_name = "valid_length" if "valid_length" in input_names else "past_seq_length"
    current_length_name = (
        "current_length" if "current_length" in input_names else "current_input_length"
    )
    prefill_inputs = {
        embed_name: prefill_embeds,
        past_length_name: torch.tensor([0], dtype=torch.int32, device=device),
        current_length_name: torch.tensor([valid_length], dtype=torch.int32, device=device),
    }
    for index in range(int(meta["num_hidden_layers"])):
        key_name = f"model_layers_{index}_self_attn_kcache_input"
        value_name = f"model_layers_{index}_self_attn_vcache_input"
        if key_name in input_names:
            prefill_inputs[key_name] = CacheTensor(
                torch.zeros(cache_shape, dtype=torch.float16, device=device)
            )
        if value_name in input_names:
            prefill_inputs[value_name] = CacheTensor(
                torch.zeros(cache_shape, dtype=torch.float16, device=device)
            )
    logits = prefill.run(prefill_inputs)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).to(device)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)

    output_ids = logits.argmax(dim=-1)
    if output_ids.shape[1] < input_ids.shape[1]:
        raise RuntimeError(
            "Prefill output is shorter than the processor token sequence: "
            f"output_length={output_ids.shape[1]}, "
            f"input_ids_length={input_ids.shape[1]}."
        )
    timestamp_ids = output_ids[:, : input_ids.shape[1]][
        input_ids == config.timestamp_token_id
    ]
    timestamps = (timestamp_ids * config.timestamp_segment_time).cpu().numpy()
    result = aligner.parse_timestamp(words, timestamps)
    for item in result:
        item["start_time"] = round(item["start_time"] / 1000.0, 3)
        item["end_time"] = round(item["end_time"] / 1000.0, 3)
        print(f"{item['text']}\t{item['start_time']:.3f}\t{item['end_time']:.3f}")


if __name__ == "__main__":
    main()
