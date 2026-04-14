_base_ = [
    "./_glm_4_7_flash_xh2a_2k.py",
]

hf_model_dir = "/data02/datasets/chuyuan.wei/GLM-4.7-Flash"

model = dict(
    hf_model=hf_model_dir,
    model_name="xh2_GLM-4.7-Flash_w8a8_256_2k",
)
