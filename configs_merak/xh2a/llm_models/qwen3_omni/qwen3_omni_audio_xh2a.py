_base_ = [
    "../../_base_/xh2a_base.py",
]

hf_model_dir = "./data/models/Qwen3-Omni"

model = dict(
    model_type="Qwen3OmniMoeForConditionalGeneration_audio",
    hf_model=hf_model_dir,
    model_name="xh2_qwen3_omni_audio_w8a8",
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        ops={},
    ),
)
