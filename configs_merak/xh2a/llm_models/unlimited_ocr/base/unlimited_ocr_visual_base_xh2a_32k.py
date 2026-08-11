_base_ = [
    "../../../_base_/xh2a_base.py",
]

import os

hf_model_dir = os.environ.get("UNLIMITED_OCR_HF_MODEL", "./data/models/Unlimited-OCR")
image_token_id = 128815

model = dict(
    model_type="UnlimitedOCRForCausalLM_visual",
    hf_model=hf_model_dir,
    model_name="xh2_unlimited_ocr_base_visual_w8a8_1024",
    export_mode="base",
    hmonnx_export=True,
    image_size=1024,
    base_size=1024,
    crop_mode=False,
    patch_size=16,
    downsample_ratio=4,
    image_token_id=image_token_id,
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        ops={},
    ),
)
