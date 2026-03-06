_base_ = [
    "../qwen2/7b/qwen2_7b_instruct_xh2a_2k_gptq_quarot_4bit_ssfp.py",
]

hf_model_dir = "./weights/Qwen2-7B-Instruct"

model = dict(
    hf_model=hf_model_dir,
)
