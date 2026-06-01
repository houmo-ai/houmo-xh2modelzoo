_base_ = [
    "./_qwen3_omni_text_xh2a_2k.py",
]

hf_model_dir = "./data/models/Qwen3-Omni"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_qwen3_omni_text_w8a8_256_2k",
)
