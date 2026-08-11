_base_ = [
    "../_unlimited_ocr_xh2a.py",
]

import os

hf_model_dir = os.environ.get("UNLIMITED_OCR_HF_MODEL", "./data/models/Unlimited-OCR")

model = dict(
    model_name="xh2_unlimited_ocr_gundam_w8a8_256_32k",
    visual_config=dict(
        model_type="UnlimitedOCRForCausalLM_visual",
        hf_model=hf_model_dir,
        model_name="xh2_unlimited_ocr_gundam_visual_w8a8_640",
        export_mode="gundam",
        hmonnx_export=False,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        patch_size=16,
        downsample_ratio=4,
        image_token_id=128815,
        max_crop_num=32,
        quant_scheme=dict(
            quant_type="w8a8h1_sefp",
            ops={},
        ),
    ),
)
