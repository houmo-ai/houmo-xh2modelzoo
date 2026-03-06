_base_ = [
    "../qwen2/7b/qwen2_7b_instruct_xh2a_4k.py",
]

# FireRedASR export script will resolve hf_model_dir automatically from:
#   1) --hf_model_dir
#   2) <fireredasr_model_dir>/Qwen2-7B-Instruct  (preferred default)
#   3) this config value
# Keep this as placeholder for local override when needed.
hf_model_dir = "./weights/Qwen2-7B-Instruct"

model = dict(
    hf_model=hf_model_dir,
)
