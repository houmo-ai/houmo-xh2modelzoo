_base_ = [
    "./qwen2_7b_instruct_xh2a_2k.py",
]
gptq = True

gptq_config = dict(
    calib_dataset="wikitext2",
    calib_samples=128,
    seqlen=2048,
    w_clip=True,
    w_bits=4,
    w_asym=False,
    w_groupsize=64,
    percdamp=0.01,
    act_order=False,
    int8_down_proj=False,
    heading_gptq=True,
)
