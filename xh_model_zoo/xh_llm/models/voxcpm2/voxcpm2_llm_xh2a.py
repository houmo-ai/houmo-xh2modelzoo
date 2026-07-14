# Copyright 2025 HOUMO AI
#
# File: voxcpm2_lm_xh2a.py
# Description:
#   Export config for VoxCPM2 base_lm and residual_lm on XH2a.
#
#   Unlike qwen3_asr where `model` is fully specified in the config, VoxCPM2's
#   LM wrap classes take the MiniCPMModel **instance** at runtime (not a path),
#   so `model.hf_model` is left as a placeholder and filled in by the export
#   script via `cfg.model.hf_model = <module>`.
#
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# 运行时通过 CLI 覆盖的参数(默认值仅作为样板):
# -----------------------------------------------------------------------------
# - model.wrap_cfg.input_sequence_length  <= prefill 长度(base_lm prefill 用)
# - model.wrap_cfg.max_sequence_length    <= KV cache 总长度
# - model.wrap_cfg.kv_cache.cache_axis    <= KV cache 时间维轴,base_lm 用 2
# - model.cache_length                    <= decode 时 KV cache 可容纳长度
# -----------------------------------------------------------------------------

work_dir = "work_dirs/voxcpm2_xh2a"
device = "cuda"
exec_device = "cuda"
dtype = "float16"

# 预留给脚本用,导出 base_lm 时填 base_lm 实例,导出 residual_lm 时填 residual_lm 实例
hf_model = None
config_dir = None


# -----------------------------------------------------------------------------
# wrap config 模板
#
# VoxCPM2 里 base_lm 和 residual_lm 都是 MiniCPMModel,导出配置结构相同,
# 只是几个开关不同:
#   - use_rope: base_lm=True, residual_lm=False
#   - use_embed_tokens: base_lm=True, residual_lm=False
#
# 导出脚本会 deepcopy 这个 dict 后按需覆盖。
# -----------------------------------------------------------------------------

_base_wrap_cfg = dict(
    use_cache=True,
    # input_sequence_length 默认等于 prefill_length(216),decode 图会覆盖为 1
    input_sequence_length=216,
    # max_sequence_length 直接等于 KV cache 总长度
    max_sequence_length=1024,
    # kv cache shape 在基类 prepare_kv_cache 里构造,这里只给 axis
    kv_cache=dict(
        cache_axis=2,  # [1, kv_heads, cache_len, head_dim] 的 cache_len 维
    ),
    # prefill 输出全部 hidden(给 residual_lm 用);decode 只留最后一个
    num_logits_to_keep=0,
    # base_lm 启用 rope;residual_lm 在导出脚本里把它关掉
    enable_rope=True,
    # only_first_block 不启用
    only_first_block=False,
    # batch_size 固定 1
    batch_size=1,
)


# -----------------------------------------------------------------------------
# 量化配置
# -----------------------------------------------------------------------------

# VoxCPM2 的输出质量对量化敏感,建议 w8a8 + sefp(和 Qwen3-ASR 对齐)
quant_type = "w8a8_sefp"


# -----------------------------------------------------------------------------
# 模型配置(LLMBaseModel 的构造参数)
#
# hf_model 必须由导出脚本在运行时注入。这里留占位。
# -----------------------------------------------------------------------------

model = dict(
    # 导出脚本根据要导 base_lm 还是 residual_lm 选择:
    #   type="XHVoxCPM2BaseLMModel" 或 "XHVoxCPM2ResidualLMModel"
    type="XHVoxCPM2BaseLMModel",
    hf_model=None,       # 运行时注入 base_lm / residual_lm 实例
    wrap_cfg=_base_wrap_cfg,
    quant_config=dict(),  # 导出脚本会注入 xhquant.QuantScheme
    frontend_type="TorchFX",
    allow_quant=True,
    export_cfg=dict(),   # 导出脚本会填 input_names / output_names
)
