_base_ = [
    "./qwen2_7b_instruct_xh2a_2k_gptq_quarot_4bit.py",
]

quant_config = dict(
    w_schema=dict(
        fp_mode="ssfp",  # sefp or ssfp
        hidden_bit=False,
        man_bit=4,  # 量化参数的mantissa位数
    ),
)
model = dict(
    quant_config=quant_config,
)
resume_from = "work_dirs/qwen2_7b_instruct_xh2a_2k_gptq_quarot_4bit/quarot_gptq-state-dict.safetensors"
