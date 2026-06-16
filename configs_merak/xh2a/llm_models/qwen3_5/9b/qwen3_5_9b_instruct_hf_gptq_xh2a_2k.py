_base_ = [
    "./qwen3_5_9b_instruct_xh2a_2k.py",
]

# Quantized HF/GPTQModel repository.  For this format the standard loading
# contract is: set model.hf_model to the quantized directory and keep
# model.quant_weight unset.
hf_model_dir = "weights/Qwen3.5-9B-mode1-llm-only"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.5-9B_w4a8_256_2k",
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
