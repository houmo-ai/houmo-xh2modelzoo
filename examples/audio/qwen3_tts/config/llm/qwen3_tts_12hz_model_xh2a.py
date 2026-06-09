# Parametrized model-level config (replaces the base/customvoice/voicedesign trio).
# Kept pure (no imports, non-lazy). Variant differences are injected at runtime by
# the export/demo scripts after reading --variant, via
# config.llm._components.apply_variant(cfg, variant), which sets
# hf_model_dir / tts_mode / ref_audio / ref_text / tts_speaker / tts_instruct.
_base_ = [
    "../../../../../configs/_base_/xh2a_base.py",
]
# placeholder; overridden by the script according to --variant
hf_model_dir = "./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"

model = dict(
    hf_model=hf_model_dir,
)
