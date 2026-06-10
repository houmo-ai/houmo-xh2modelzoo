_base_ = [
    "./qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py",
]

# Quantized HF/GPTQModel repository.  For this format the standard loading
# contract is: set model.hf_model to the quantized directory and keep
# model.quant_weight unset.
hf_model_dir = "/data01/home/yujy/work/xh2modelzoo/weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_Qwen3.6-35B-A3B_w4a8_256_2k",
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
