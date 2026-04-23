_base_ = [
    "./qwen3_5_35b_a3b_xh2a.py",
]

# Qwen3.6-35B-A3B uses the same qwen3_5_moe architecture as Qwen3.5-35B-A3B;
# only weights differ. This config inherits everything from the 35B_A3B config.

release = dict(
    xh_version="xh2",
    modelscope_name="qwen3_6_moe",
)
