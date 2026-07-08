# Shared component definitions (plain Python module).
# Each variant's component config imports e.g. `from config.llm._components import TALKER`
# and deep-merges it with the model-level config brought in via `_base_`
# (which provides hf_model / tts_mode). The three variants share identical
# component structure and differ only at the model-level config, so the
# definitions live here to remove the previously triplicated wrap_cfg / export_cfg.

# Talker sub-model
TALKER = dict(
    type="XHQwen3TTSTalker",
    wrap_cfg=dict(
        max_sequence_length=2048,
        input_sequence_length=256,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(cache_axis=2),
        enable_rope=True,
    ),
    quant_config=dict(),
    frontend_type="TorchFX",
    export_cfg=dict(
        input_names=["inputs_embeds", "past_seq_length", "current_input_length"],
        output_names=["logits", "past_hidden"],
    ),
)

# CodePredictor sub-model
CODE_PREDICTOR = dict(
    type="XHQwen3TTSCodePredictor",
    wrap_cfg=dict(
        max_sequence_length=18,
        input_sequence_length=2,
        use_cache=True,
        num_logits_to_keep=1,
        kv_cache=dict(cache_axis=2),
        enable_rope=True,
    ),
    quant_config=dict(),
    frontend_type="TorchFX",
    export_cfg=dict(
        input_names=["inputs_embeds", "past_seq_length", "current_input_length"],
        output_names=["logits"],
    ),
)


# ---------------------------------------------------------------------------
# Variant registry: parametrized model-level / hmonnx configs pick an entry by
# variant, replacing the original 3 model-level + 3 hmonnx duplicated files.
# ---------------------------------------------------------------------------
import os

# variant -> model-level fields (mirrors the original 3 model-level configs)
VARIANTS = {
    "0_6B_base": dict(
        hf_model_dir="./data/models/Qwen3-TTS-12Hz-0.6B-Base",
        tts_mode="voice_clone",
        # voice-clone needs a reference audio + text; download before export:
        #   curl -sSL -o /tmp/clone_1.wav \
        #       https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav
        ref_audio="/tmp/clone_1.wav",
        ref_text="甚至出现交易几乎停滞的情况。",
    ),
    "0_6B_customvoice": dict(
        hf_model_dir="./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        tts_mode="custom_voice",
        tts_speaker="vivian",
    ),
    "1_7B_customvoice": dict(
        hf_model_dir="./data/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        tts_mode="custom_voice",
        tts_speaker="vivian",
    ),
    "1_7B_voicedesign": dict(
        hf_model_dir="./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        tts_mode="voice_design",
        tts_instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    ),
}
VARIANT_CHOICES = tuple(VARIANTS.keys())

# Shared target text (identical across variants)
TTS_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def current_variant():
    """Variant selected via environment variable, default 0_6B_customvoice."""
    return os.environ.get("QWEN3TTS_VARIANT", "0_6B_customvoice")


# variant x component -> existing work_dir name (single source of truth;
# centralizes the 1.7B naming asymmetry)
WORKNAME = {
    "0_6B_base": {
        "talker": "qwen3_tts_12hz_0_6B_base_talker_2k_xh2a",
        "code_predictor": "qwen3_tts_12hz_0_6B_base_code_predictor_2k_xh2a",
        "text_projection": "qwen3_tts_12hz_0_6B_base_text_projection_xh2a",
        "speech_tokenizer": "qwen3_tts_12hz_0_6B_base_speech_tokenizer_xh2a",
    },
    "0_6B_customvoice": {
        "talker": "qwen3_tts_12hz_0_6B_customvoice_talker_2k_xh2a",
        "code_predictor": "qwen3_tts_12hz_0_6B_customvoice_code_predictor_2k_xh2a",
        "text_projection": "qwen3_tts_12hz_0_6B_customvoice_text_projection_xh2a",
        "speech_tokenizer": "qwen3_tts_12hz_0_6B_customvoice_speech_tokenizer_xh2a",
    },
    "1_7B_customvoice": {
        "talker": "qwen3_tts_12hz_1_7B_customvoice_talker_2k_xh2a",
        "code_predictor": "qwen3_tts_12hz_1_7B_customvoice_code_predictor_2k_xh2a",
        "text_projection": "qwen3_tts_12hz_1_7B_customvoice_text_projection_xh2a",
        "speech_tokenizer": "qwen3_tts_12hz_1_7B_customvoice_speech_tokenizer_xh2a",
    },
    "1_7B_voicedesign": {
        # Note: 1.7B talker/code_predictor carry "voicedesign", while
        # text_projection/speech_tokenizer do not (matching existing dirs).
        "talker": "qwen3_tts_12hz_1_7B_voicedesign_talker_2k_xh2a",
        "code_predictor": "qwen3_tts_12hz_1_7B_voicedesign_code_predictor_2k_xh2a",
        "text_projection": "qwen3_tts_12hz_1_7B_text_projection_xh2a",
        "speech_tokenizer": "qwen3_tts_12hz_1_7B_speech_tokenizer_xh2a",
    },
}


# ---------------------------------------------------------------------------
# Injection helpers: scripts call these after Config.fromfile to write variant
# differences into the parsed cfg. Config files stay pure (non-lazy) and do not
# read env vars, avoiding lazy/non-lazy inheritance-chain conflicts.
# ---------------------------------------------------------------------------
def apply_variant(cfg, variant):
    """Inject model-level fields: hf_model_dir / model.hf_model / tts_mode /
    tts_text and variant-specific fields (ref_audio/ref_text/tts_speaker/tts_instruct)."""
    v = VARIANTS[variant]
    cfg.hf_model_dir = v["hf_model_dir"]
    cfg.tts_mode = v["tts_mode"]
    cfg.tts_text = TTS_TEXT
    if "model" in cfg and cfg.model is not None and "hf_model" in cfg.model:
        cfg.model.hf_model = v["hf_model_dir"]
    for k, val in v.items():
        if k not in ("hf_model_dir", "tts_mode"):
            cfg[k] = val
    return cfg


def apply_hf_model_dir_override(cfg, hf_model_dir):
    """Override the HF model path after applying a variant."""
    if not hf_model_dir:
        return cfg
    cfg.hf_model_dir = hf_model_dir
    if "model" in cfg and cfg.model is not None and "hf_model" in cfg.model:
        cfg.model.hf_model = hf_model_dir
    return cfg


def apply_variant_hmonnx(cfg, variant):
    """Inject the variant's work_dirs product paths into the four sub-models of
    an hmonnx cfg.model, taken from WORKNAME (handles 1.7B asymmetry). Also
    injects model-level fields via apply_variant."""
    apply_variant(cfg, variant)
    w = WORKNAME[variant]
    root = "work_dirs"
    cfg.model.text_projection.onnx_file = f"{root}/{w['text_projection']}/hmonnx/text_projection_XH2a.onnx"
    cfg.model.code_predictor.model_cfg = f"{root}/{w['code_predictor']}/meta.json"
    cfg.model.talker.model_cfg = f"{root}/{w['talker']}/meta.json"
    cfg.model.speech_tokenizer.model_cfg = f"{root}/{w['speech_tokenizer']}/meta.json"
    return cfg
