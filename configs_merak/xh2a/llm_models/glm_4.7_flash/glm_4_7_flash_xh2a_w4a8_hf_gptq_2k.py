_base_ = [
    "./glm_4_7_flash_xh2a_2k.py",
]

# This GPTQ checkpoint keeps attention/shared experts at W8 and routed experts
# at W4 (group size 64). The export path preserves available GPTQ weights and
# automatically falls back to W8 for modules without 4-bit quant_weight.
hf_model_dir = "/data01/home/xuzk/datas/gptqmodel/gptqmodel/output/GLM-4.7-Flash-4bit-64g"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_GLM-4.7-Flash-attn8-expert4-64g_w4a8_256_2k",
    quant_scheme=dict(
        _delete_=True,
        quant_type="w4a8h1_ssfp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    quant_weight=None,
)
