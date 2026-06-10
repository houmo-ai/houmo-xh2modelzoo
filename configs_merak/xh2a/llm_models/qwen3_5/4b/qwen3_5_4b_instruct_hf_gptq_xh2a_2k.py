_base_ = [
    "./qwen3_5_4b_instruct_xh2a_2k.py",
]

# Quantized HF/GPTQModel repository.  Keep model.quant_weight unset so Merak
# loads the packed GPTQModel/AutoRound checkpoint directly.
hf_model_dir = "/data01/home/yujy/work/gptqmodel/output/Qwen3.5-4B-mode1-llm-only"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-4B_w4a8_256_2k",
    quant_scheme=dict(
        quant_type="w4a8h0_ssfp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    quant_weight=None,
)
