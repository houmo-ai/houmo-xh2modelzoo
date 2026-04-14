_base_ = [
    "./glm_4_7_flash_xh2a_2k.py",
]

hf_model_dir = "/data02/datasets/chuyuan.wei/GLM-4.7-Flash_gptq_4bit/"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_GLM-4.7-Flash-GPTQ-Int4_w4a8_256_2k",
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
)
