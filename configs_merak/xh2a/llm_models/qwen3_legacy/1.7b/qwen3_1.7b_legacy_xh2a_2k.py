_base_ = [
    "../_qwen3_legacy_xh2a_2k.py",
]
hf_model_dir = "./data/Qwen3-1.7B"  # 模型路径

model = dict(
    model_type="Qwen3ForCausalLM_legacy",
    hf_model=hf_model_dir,
)
