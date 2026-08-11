_base_ = [
    "../../_base_/xh2a_base.py"
]

import os

# Override with UNLIMITED_OCR_HF_MODEL when the checkpoint is stored outside
# the repository's default data layout.
hf_model_dir = os.environ.get("UNLIMITED_OCR_HF_MODEL", "./data/models/Unlimited-OCR")
calib_image_dir = os.environ.get("UNLIMITED_OCR_CALIB_IMAGE_DIR", "./data/calib_data/sampledata")
image_token_id = 128815

model = dict(
    model_type="UnlimitedOCRForCausalLM",
    hf_model=hf_model_dir,
    model_name="xh2_unlimited_ocr_w8a8_256_32k",
    context_max_length=32768,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
    sliding_window_size=128,
    image_token_id=image_token_id,
    visual_config=dict(
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
    ),
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    only_first_block=False,
    # Real-image PTQ calibration. Set enable=True to calibrate the prefill graph
    # on real document images instead of the text-only dummy. base image prompt
    # (~278 tokens) is chunked to prefill_chunk_length during calibration.
    calib_config=dict(
        enable=False,
        # Override with UNLIMITED_OCR_CALIB_IMAGE_DIR for local datasets.
        image_dir=calib_image_dir,
        prompt="<image>\\nFree OCR. ",
        num_samples=8,
    ),
)
