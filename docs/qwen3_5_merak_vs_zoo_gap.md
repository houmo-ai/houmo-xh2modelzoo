# Qwen3.5 Merak vs xh_model_zoo 差距分析

## 架构定位

| 层次 | xh_model_zoo | xhmodel_merak |
|------|-------------|---------------|
| 导出 | `Qwen3_5ConverterXH2a` (手动组装图) | `XHQwen3_5Model.export_hmonnx()` (框架自动 trace+quant) |
| 运行时 | `Qwen3_5ONNXModel` / `Qwen3_5SpecDecodeONNXModel` | **共用 zoo 的运行时** (导出格式兼容) |
| 配置 | `Qwen3_5ConvertConfig` (dataclass) | `XHQwen3_5ModelConfig` (继承 VisionLLMModelConfig) |

关键认知：merak 只负责**导出**，运行时推理复用 xh_model_zoo 的 `Qwen3_5ONNXModel` / `Qwen3_5SpecDecodeONNXModel`。

---

## 已完成 (merak)

| 功能 | 状态 | 验证 |
|------|------|------|
| 基础 prefill/decode 导出 | done | ONNX 输出正确 |
| Spec decode MTP 导出 | done | 242 outputs, runtime generate 通过 |
| Spec decode DFlash 导出 | done | 代码完成，待整网验证 |
| MTP draft model 导出 | done | 独立 ONNX 导出成功 |
| DFlash draft model (context/decode) | done | 代码完成 |
| Visual model 导出 | done | ONNX 导出成功 |
| golden_meta_info.json 兼容 | done | spec_decode section 格式正确 |
| split_conv_cache 支持 | done | _llm_model_impl 已实现 |
| alpha_scaling_layers | done | _llm_model_impl L1364 |
| chunk_inverse_alpha | done | _llm_model_impl L1365 |
| enable_rope | done | _llm_model_impl L547 |

---

## 差距清单

### P0 - 功能缺失 (影响正确性)

| # | 功能 | zoo 实现 | merak 现状 | 影响 |
|---|------|---------|-----------|------|
| 1 | `normalize_force_fp32` 配置透传 | `quant_config["ops_cfg"]["Normalize"] = dict(force_fp32=True)` | 未暴露到 config，依赖 quant_scheme 手动配置 | 精度问题 |
| 2 | `cumsum_matmul_quant_config` | zoo converter 注入到 quant_cfg.nodes_cfg | merak 无此配置入口 | 量化精度 |
| 3 | `only_first_block=False` 整网导出验证 | zoo 有完整 32 层导出+推理验证 | merak 仅验证了 4 层 (3 linear + 1 full) | 整网可能有隐藏 bug |

### P1 - 配置暴露 (影响易用性)

| # | 功能 | 说明 |
|---|------|------|
| 4 | `linear_chunk_size` | merak 硬编码 64，未暴露到 XHQwen3_5ModelConfig |
| 5 | `split_conv_cache` | merak wrap_cfg 支持但未暴露到 config |
| 6 | `alpha_scaling_layers` | 硬编码 [8, 20]，未暴露到 config |
| 7 | `chunk_inverse_alpha` | 硬编码 0.5，未暴露到 config |

### P2 - 工具链 (影响效率)

| # | 功能 | 说明 |
|---|------|------|
| 8 | 整网 spec decode benchmark | zoo 有 `qwen3_5_xh2a_spec_decode_bench.py`，merak 无 |
| 9 | MTP hot vocab 构建 | zoo 有 `build_mtp_hot_vocab.py`，merak 无需求 (共用) |
| 10 | 导出验证脚本 | zoo 有 `_export_validation.py`，merak 无 |
| 11 | batch export 自动化 | zoo 有 shell 脚本，merak 无 |

### P3 - 文档 (影响可维护性)

| # | 功能 | 说明 |
|---|------|------|
| 12 | MTP benchmark guide | zoo 有 15KB 文档 |
| 13 | 完整 README | zoo 有 39KB README，merak 仅 1.5KB |

---

## 不需要迁移的部分

| 功能 | 原因 |
|------|------|
| `Qwen3_5ONNXModel` | 运行时类，merak 导出后直接复用 |
| `Qwen3_5SpecDecodeONNXModel` | 运行时类，merak 导出后直接复用 |
| `Qwen3_5ConverterXH2a` | merak 框架自动完成 trace+quant，不需要手动 converter |
| `qwen3_5_spec_decode_metrics.py` | 评测脚本，与导出框架无关 |
| `build_spec_decode_dataset.py` | 数据集工具，与导出框架无关 |
| benchmark 对比脚本 | 通用工具，不绑定导出框架 |

---

## 建议优先级

1. **整网导出验证** (P0#3) — 用 32 层模型跑 export + spec decode test
2. **normalize_force_fp32 透传** (P0#1) — 加到 XHQwen3_5ModelConfig
3. **cumsum_matmul_quant_config** (P0#2) — 加到 config 并注入 quant_cfg
4. **配置暴露** (P1) — 将硬编码参数提升到 config
5. **导出验证脚本** (P2#10) — 对比 merak 和 zoo 导出的 ONNX 数值一致性
