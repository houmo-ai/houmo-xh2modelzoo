_base_ = [
    "./gemma4_31b_it_xh2a_2k.py",
]

model = dict(
    model_type="Gemma4ForConditionalGeneration_visual",
    model_name="xh2_gemma4_31b_it_visual_w8a8_2k",
)
