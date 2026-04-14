_base_ = [
    "./glm_4_7_flash_xh2a_2k.py",
]

model = dict(
    model_name="xh2_GLM-4.7-Flash_w4a8_256_2k",
    quant_scheme=dict(
        _delete_=True,
        w_scheme=dict(
            bits=4,
            fp_mode="ssfp",
        ),
        act_scheme=dict(
            bits=8,
            fp_mode="sefp",
        ),
        ops={},
    ),
    quant_weight="work_dirs/glm_4_7_flash_xh2a_2k_gptq_4bit/gptq-state-dict.safetensors",
)
