# Qwen3.5-27B XH2a 量化导出与推理

Qwen3.5 是阿里巴巴通义千问系列的混合线性注意力大语言模型，本目录提供完整的 XH2a 量化导出、精度验证、Golden 生成及 HMONNX 推理脚本。

## 模型架构

| 属性 | 值 |
|------|-----|
| Architecture | `Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM` |
| 参数量 | 27B |
| 层数 | 64（16 full_attention + 48 linear_attention） |
| hidden_size | 5120 |
| num_heads / num_kv_heads | 24 / 4 |
| head_dim | 256 |
| intermediate_size | 17408 |
| 位置编码 | M-RoPE（time/height/width），partial_rotary_factor=0.25 |
| 全注意力层 | Gated Full Attention（每 4 层一个） |
| 线性注意力层 | GatedDeltaNet（独立 qkv/z/b/a 投影） |

## 支持的权重格式

| 来源 | 权重路径 | 量化方案 | 说明 |
|------|---------|---------|------|
| 官方 BF16 | `weights/Qwen3.5-27B` | w8a8h1_sefp | 权重 8-bit，激活 8-bit |
| GPTQ 4-bit | `weights/Qwen3.5-27B-GPTQModel-self-generated-4bit-hessian-mse` | w4a8h0_ssfp | 权重 4-bit，激活 8-bit |

## 环境依赖

```bash
conda activate xhquant
# transformers >= 5.2.0, torch >= 2.8.0
```

## 目录结构

```
examples/llm/qwen3_5/
├── README.md                            # 本文档
├── qwen3_5_xh2a_export_hmonnx.py       # 量化导出主脚本（验证 + ONNX 导出 + Golden）
├── qwen3_5_xh2a_demo.py                # HMONNX 推理 Demo（单轮/多轮对话）
├── qwen3_5_xh2a_hmonnx_test.py         # HMONNX 推理 Benchmark
├── _export_validation.py                # 多阶段精度验证工具
├── _runtime.py                          # HMONNX 运行时加载工具
├── common.py                            # 公共工具函数
├── demo.py                              # HF 原生推理 Demo（基线对比用）
└── 模型迁移适配.md                       # 迁移需求文档

configs/qwen3_5/
└── qwen3_5_27b_xh2a.py                 # XH2a 导出配置

xh_model_zoo/xh_llm/models/qwen3_5/
├── __init__.py                          # 模块导出
├── _model.py                            # Wrap 模块注册（从 xhquant_llm 移植）
├── qwen3_5_convert_config.py            # 转换配置数据类
├── qwen3_5_converter.py                 # LLMConverter 实现
├── qwen3_5_llm_model.py                 # XHQwen3_5Model 模型包装
└── qwen3_5_onnx_model.py               # HMONNX 推理包装
```

## 导出命令

### BF16 → w8a8（含验证 + Golden）

```bash
CUDA_VISIBLE_DEVICES=0,6,7 conda run -n xhquant python \
  examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_27b_xh2a.py \
  --hf_model_dir weights/Qwen3.5-27B \
  --dtype fp16 \
  --valid \
  --golden \
  --work_dir work_dirs/qwen3_5_27b_bf16_export
```

### GPTQ → w4a8（含验证 + Golden）

```bash
CUDA_VISIBLE_DEVICES=0,6,7 conda run -n xhquant python \
  examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_27b_xh2a.py \
  --hf_model_dir weights/Qwen3.5-27B-GPTQModel-self-generated-4bit-hessian-mse \
  --dtype fp16 \
  --valid \
  --golden \
  --work_dir work_dirs/qwen3_5_27b_gptq_export
```

### 仅生成 Golden（复用已有 ONNX）

```bash
CUDA_VISIBLE_DEVICES=0,6,7 conda run -n xhquant python \
  examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
  --config configs/qwen3_5/qwen3_5_27b_xh2a.py \
  --hf_model_dir weights/Qwen3.5-27B \
  --dtype fp16 \
  --golden_only \
  --existing_work_dir work_dirs/qwen3_5_27b_bf16_export \
  --work_dir work_dirs/qwen3_5_27b_bf16_export
```

## 精度验证流程

`--valid` 启用多阶段逐层对比（prefill 首字 + 64-token decode）：

```
HF 原始模型
  ↓ 对比
Wrap 模型（注册 XHTrace 模块后的等价模型）
  ↓ 对比
Frontend 图（TorchFX trace 后的计算图）
  ↓ 对比
Quant 图（PTQ 量化后的计算图）
  ↓ 导出
ONNX 模型（prefill + decode 两张图）
```

每个阶段对比：
- **Prefill 首字 logits**：max abs error
- **64-token decode 全输出**：token-level exact_match + text exact_match

## HMONNX 推理

### Demo（单轮对话）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xhquant python \
  examples/llm/qwen3_5/qwen3_5_xh2a_demo.py \
  --config work_dirs/qwen3_5_27b_bf16_export/meta.json \
  --prompt "你好呀，你是谁，中文回答" \
  --max-new-tokens 256 \
  --dtype fp16
```

### Demo（交互多轮对话）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xhquant python \
  examples/llm/qwen3_5/qwen3_5_xh2a_demo.py \
  --config work_dirs/qwen3_5_27b_bf16_export/meta.json \
  --interactive \
  --dtype fp16
```

### Benchmark 测试

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xhquant python \
  examples/llm/qwen3_5/qwen3_5_xh2a_hmonnx_test.py \
  --config work_dirs/qwen3_5_27b_bf16_export/meta.json \
  --prompt "请用中文简要介绍一下混合线性注意力模型。" \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 3 \
  --dtype fp16
```

## 导出产物

```
work_dirs/qwen3_5_27b_bf16_export/
├── meta.json                            # 模型元信息（HMONNX 加载入口）
├── export_meta_info.json                # 导出过程元信息
├── token_embedding.pt                   # Token Embedding 权重
├── hf_config/                           # HF 模型配置文件
├── prefill_onnx/                        # Prefill ONNX 模型
│   └── qwen3_5_27b_xh2a_prefill.onnx
├── decode_onnx/                         # Decode ONNX 模型
│   └── qwen3_5_27b_xh2a_decode.onnx
└── hmquant_xh2_qwen3_5_27b_w8_a8_256_2k_YYYYMMDD/  # Golden 发布目录
    ├── golden_meta_info.json
    ├── quant_embedding.pt
    ├── prefill/                         # Prefill ONNX + Golden（233 outputs/step）
    │   ├── *_prefill_with_act.onnx
    │   ├── *_prefill_external_data/
    │   └── step_0/
    └── decode/                          # Decode ONNX + Golden（234 outputs/step）
        ├── *_decode_with_act.onnx
        ├── *_decode_external_data/
        └── step_0/
```

## Cache 结构

| Cache 类型 | 层数 | Shape | 说明 |
|-----------|------|-------|------|
| KV Cache（全注意力层） | 16 | `[1, 4, 2048, 256]` | batch × kv_heads × context_len × head_dim |
| Conv Cache（线性注意力层） | 48 | `[1, 10240, 4]` | 短卷积状态 |
| Recurrent State（线性注意力层） | 48 | `[1, 48, 128, 128]` | GatedDeltaNet 循环状态 |

## 注意事项

1. **M-RoPE**：Qwen3.5 使用 Multi-modal RoPE（time/height/width 三组位置 ID），纯文本推理时三组相同（均为 `arange(0, seq_len)`）。
2. **RoPE 在线/离线计算**：通过 `support_long_context_over_fp16_limit` 参数控制：
   - **`False`（默认，在线）**：每次 forward 从 position_ids + inv_freq 实时计算 cos/sin，无需预分配缓存，支持任意序列长度。
   - **`True`（离线/预缓存）**：启动时预计算 `max_pe_length` 长度的 cos/sin cache，用索引查表，适用于 position_id 可能超过 FP16 上限（65504）的场景。
   - 导出时可通过 `--support_long_context_over_fp16_limit` 切换到离线模式。
3. **Vision 部分**：当前仅支持 LLM 部分，Vision 编码器迁移后续补充。
4. **GPTQ 自动检测**：导出脚本自动检测 GPTQ 权重并将 `w_schema.bits` 从 8 覆盖为 4。
5. **GPU 需求**：27B 模型导出需要 3 张 A100 80GB（通过 `device_map='auto'` 自动分配）。验证阶段自动选择空闲最大的 GPU。
6. **Prefill/Decode 分离**：混合注意力模型的 Prefill（chunk 模式）和 Decode（recurrent 模式）使用不同的计算图，分别导出。
