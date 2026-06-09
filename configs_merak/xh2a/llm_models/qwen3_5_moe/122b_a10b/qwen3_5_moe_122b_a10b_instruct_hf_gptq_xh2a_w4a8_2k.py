_base_ = [
    "./qwen3_5_moe_122b_a10b_instruct_xh2a_2k.py",
]

hf_model_dir = "./data/models/Qwen3.5-122B-A10B-gptq-attn8-expert4-expertDown5-shared8-base4-64g_060920"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-122B-A10B-GPTQ_w4a8_256_2k",
    quant_scheme=dict(
        quant_type="w4a8h0_ssfp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
)
