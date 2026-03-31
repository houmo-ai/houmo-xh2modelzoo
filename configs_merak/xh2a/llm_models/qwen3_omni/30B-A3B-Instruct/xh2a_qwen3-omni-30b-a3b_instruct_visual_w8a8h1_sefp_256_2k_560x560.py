_base_ = [
    "./_30b_a3b_instruct_common.py",
]

model = dict(
    model_type="Qwen3OmniMoeForConditionalGeneration_visual",
    model_name="xh2a_qwen3-omni-30b-a3b-instruct_visual_w8a8h1_sefp_256_2k_560x560",
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        ops={},
    ),
    max_size_w=576,
    max_size_h=576,
)
