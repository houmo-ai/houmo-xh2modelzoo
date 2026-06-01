_base_ = [
    "./qwen2_7b_instruct_xh2a_4k.py",
]
model = dict(
    wrap_cfg=dict(  # 改写模型时，需要传入的配置参数
        max_sequence_length=2048,
    ),
)
