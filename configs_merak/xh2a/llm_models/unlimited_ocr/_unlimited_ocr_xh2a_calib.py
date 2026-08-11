_base_ = [
    "./_unlimited_ocr_xh2a.py",
]

import os

calib_image_dir = os.environ.get("UNLIMITED_OCR_CALIB_IMAGE_DIR", "./data/calib_data/unlimited_ocr_mix")

model = dict(
    model_name="xh2_unlimited_ocr_calib_w16a16_256_32k",
    quant_scheme=dict(
        quant_type="w16a16h0_sefp",
        nodes=dict(
            lm_head=dict(
                quant_type="w16a16h0_sefp",
            ),
        ),
    ),
    visual_config=dict(
        quant_scheme=dict(
            quant_type="w16a16h0_sefp",
            ops={},
        ),
    ),
    calib_config=dict(
        enable=True,
        # Mixed real-document calibration set: 7 screenshots + 14 rendered PDF
        # pages (Unlimited-OCR.pdf). Dense document pages better match the OCR
        # activation distribution than the previous 6-image set.
        image_dir=calib_image_dir,
        prompts=[
            "<image>\\nFree OCR. ",
            "<image>\\n<|grounding|>Convert the document to markdown. ",
        ],
        num_samples=21,
        decode_num_samples=4,
        decode_steps=8,
    ),
)
