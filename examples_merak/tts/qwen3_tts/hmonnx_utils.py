import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def load_export_meta(work_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    meta_file = work_dir / "export_meta_info.json"
    if not meta_file.is_file():
        raise FileNotFoundError(meta_file)
    return json.loads(meta_file.read_text(encoding="utf-8"))


def load_component_meta(work_dir: str | Path, export_meta: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    work_dir = Path(work_dir)
    component = export_meta["components"].get(name)
    if not component:
        raise KeyError(f"component {name!r} is missing in export_meta_info.json")
    meta_file = work_dir / component["meta_file"]
    if not meta_file.is_file():
        raise FileNotFoundError(meta_file)
    return meta_file, json.loads(meta_file.read_text(encoding="utf-8"))


def build_hmonnx_model_cfg(work_dir: str | Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    export_meta = load_export_meta(work_dir)
    text_meta_file, text_meta = load_component_meta(work_dir, export_meta, "text_projection")
    talker_meta_file, _ = load_component_meta(work_dir, export_meta, "talker")
    predictor_meta_file, _ = load_component_meta(work_dir, export_meta, "code_predictor")
    tokenizer_meta_file, _ = load_component_meta(work_dir, export_meta, "speech_tokenizer")
    return {
        "type": "Qwen3TTSHMONNXInference",
        "hf_model": export_meta["hf_model"],
        "text_projection": {
            "type": "Qwen3TTSTextProjectionInference",
            "onnx_file": str(text_meta_file.parent / text_meta["hmonnx"]),
        },
        "code_predictor": {
            "type": "Qwen3TTSCodePredictorInference",
            "model_cfg": str(predictor_meta_file),
        },
        "talker": {
            "type": "Qwen3TTSTalkerInference",
            "model_cfg": str(talker_meta_file),
        },
        "speech_tokenizer": {
            "type": "Qwen3TTSSpeechTokenizerInference",
            "model_cfg": str(tokenizer_meta_file),
        },
    }


def build_hmonnx_model(work_dir: str | Path, device: str):
    from xhmodel_merak.xh_other_model.builder import MODELS
    from xhmodel_merak.xh_other_model.models.qwen3_tts import Qwen3TTSHMONNXInference

    model = MODELS.build(build_hmonnx_model_cfg(work_dir))
    if not isinstance(model, Qwen3TTSHMONNXInference):
        raise TypeError(f"expected Qwen3TTSHMONNXInference, got {type(model)!r}")
    return model.to(device)


def infer_request(export_meta: dict[str, Any], args) -> dict[str, Any]:
    mode = args.mode or export_meta.get("tts_mode", "custom_voice")
    return {
        "mode": mode,
        "text": args.text or export_meta.get("tts_text") or DEFAULT_TEXT,
        "language": args.language,
        "speaker": args.speaker or export_meta.get("tts_speaker") or "vivian",
        "instruct": args.instruct or export_meta.get("tts_instruct") or "",
        "ref_audio": args.ref_audio or export_meta.get("ref_audio"),
        "ref_text": args.ref_text or export_meta.get("ref_text") or "",
    }


def build_voice_clone_prompt(work_dir: str | Path, export_meta: dict[str, Any], request: dict[str, Any], device: str, xvec_only: bool):
    mode = str(request["mode"]).replace("-", "_").lower()
    if mode not in {"voice_clone", "voiceclone", "base"}:
        return None
    if "base_frontend" not in export_meta.get("components", {}):
        return None

    import torchaudio
    from qwen_tts import VoiceClonePromptItem
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

    work_dir = Path(work_dir)
    meta_file, frontend_meta = load_component_meta(work_dir, export_meta, "base_frontend")
    encode_path = meta_file.parent / frontend_meta["speech_tokenizer_encode_hmonnx"]
    speaker_path = meta_file.parent / frontend_meta["speaker_encoder_hmonnx"]
    if not encode_path.exists():
        raise FileNotFoundError(encode_path)
    if not speaker_path.exists():
        raise FileNotFoundError(speaker_path)
    if not request["ref_audio"]:
        raise ValueError("voice_clone mode requires --ref-audio or export.ref_audio")

    runtime_device = torch.device(device)
    encode_session = HMONNXInference(str(encode_path))
    encode_session.to(runtime_device)
    speaker_session = HMONNXInference(str(speaker_path))
    speaker_session.to(runtime_device)
    encode_input_dtype = getattr(encode_session.inputs[0], "dtype", torch.float16)
    speaker_input_dtype = getattr(speaker_session.inputs[0], "dtype", torch.float16)

    wav, sr = torchaudio.load(request["ref_audio"])
    wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    sample_rate = int(frontend_meta.get("input_sample_rate", 24000))
    if int(sr) != sample_rate:
        wav = torchaudio.functional.resample(wav, int(sr), sample_rate)
    wav = wav.squeeze(0).contiguous()

    audio_samples = int(frontend_meta["audio_samples"])
    valid_len = min(int(wav.numel()), audio_samples)
    input_values = torch.zeros((1, audio_samples), dtype=encode_input_dtype)
    padding_mask = torch.zeros((1, audio_samples), dtype=torch.int32)
    input_values[0, :valid_len] = wav[:valid_len].to(encode_input_dtype)
    padding_mask[0, :valid_len] = 1
    encode_out = encode_session(input_values.to(runtime_device), padding_mask.to(runtime_device))
    audio_codes, valid_frames = encode_out if isinstance(encode_out, (tuple, list)) else (encode_out, None)
    code_len = audio_codes.shape[1] if valid_frames is None else int(valid_frames.detach().cpu().reshape(-1)[0])
    ref_code = None if xvec_only else audio_codes[0, :code_len, :].detach().cpu().to(torch.long)

    mels = mel_spectrogram(
        wav.unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=sample_rate,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    mel_frames = int(frontend_meta["mel_frames"])
    mel_dim = int(frontend_meta["mel_dim"])
    padded_mels = torch.zeros((mels.shape[0], mel_frames, mel_dim), dtype=speaker_input_dtype)
    valid_mel_frames = min(int(mels.shape[1]), mel_frames)
    padded_mels[:, :valid_mel_frames, :] = mels[:, :valid_mel_frames, :].to(speaker_input_dtype)
    speaker_out = speaker_session(padded_mels.to(runtime_device))
    if isinstance(speaker_out, (tuple, list)):
        speaker_out = speaker_out[0]
    ref_spk_embedding = speaker_out[0].detach().cpu().to(torch.float32)

    return [
        VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=bool(xvec_only),
            icl_mode=bool(not xvec_only),
            ref_text=request["ref_text"],
        )
    ]


def resolve_stateful_decoder_meta(work_dir: str | Path, export_meta: dict[str, Any]) -> Path:
    meta_file, _ = load_component_meta(work_dir, export_meta, "stateful_decoder")
    return meta_file
