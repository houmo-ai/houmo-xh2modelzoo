_base_ = [
    "./qwen3_5_moe_35b_a3b_instruct_spec_mtp_hf_autoround_xh2a_2k.py",
]

model = dict(
    model_name="qwen3_6_35b_a3b_spec_mtp_head_k81920_test",
    mtp_head_k=81920,
    force_rerank=True,
)
