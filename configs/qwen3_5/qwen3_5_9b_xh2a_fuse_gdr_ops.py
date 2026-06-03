_base_ = [
    "qwen3_5_9b_xh2a.py",
]

model = dict(
    wrap_cfg=dict(
        fuse_gdr_ops=True,
    ),
)
