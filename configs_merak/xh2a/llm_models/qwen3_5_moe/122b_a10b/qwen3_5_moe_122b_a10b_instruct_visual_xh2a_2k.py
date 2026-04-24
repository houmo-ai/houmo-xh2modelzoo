_base_ = [
    "../../../_base_/xh2a_base.py",
]

hf_model_dir = "data/models/Qwen3.5-122B-A10B/"

model = dict(
    model_type="Qwen3_5MoeForConditionalGeneration_visual",
    hf_model=hf_model_dir,
    model_name="xh2_qwen3_5_moe_vision_122b_a10b_w8a8",
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        ops={},
    ),
    max_size_w=448,
    max_size_h=448,
)
