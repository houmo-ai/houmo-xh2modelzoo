_base_ = [
    "../_qwen3_xh2a_2k.py",
]
hf_model_dir = "./data/models/Qwen3-8B"  # 模型路径

model = dict(
    model_type="Qwen3ForCausalLM",
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3-8B_w8a8_256_2k",
)
