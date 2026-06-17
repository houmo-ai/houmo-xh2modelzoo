# DeepSeek-V4 Export & Inference

DeepSeek-V4 (Flash-slim5l, 5 层精简版) → HMONNX 导出与推理。

## 架构概览

```
5 层 Transformer, 256 专家 MoE, bf16, ~46GB
├── Layer 0,1: sliding attention (window=128)
├── Layer 2,4:  CSA compressor (compress_rate=4, 64 compressed entries)
└── Layer 3:    HCA compressor (compress_rate=128, 2 compressed entries)
```

MLA (Multi-head Latent Attention) 特性:
- LoRA-style Q 投影 (q_a → q_b)
- Shared KV 投影 (K==V, single head)
- Grouped output 投影
- Per-head attention sinks
- Interleaved RoPE (qk_rope_head_dim=64) + Conjugate RoPE on output
- CSA/HCA KV cache compressor

## 环境要求

- Python: xhquant conda 环境
- GPU: 导出需 1 张 ≥40GB **空闲**显存（MoE 256 expert sefp 量化峰值 ~20GB，显存被占会 OOM）；验证 wrapped/HF 需多张 (accelerate dispatch ~46GB)
- 选卡：跑前用 `nvidia-smi` 选一张 `free` 显存 ≥40GB 的卡，避免与其他进程冲突导致量化阶段 OOM
- HF 模型路径: `/data01/nfs_shared/llm/DeepSeek-V4-Flash-slim5l`

## Step 1: 量化导出

```bash
CUDA_VISIBLE_DEVICES=0 python \
    examples/llm/deepseek_v4/deepseek_v4_xh2a_export_hmonnx.py \
    --model /data01/nfs_shared/llm/DeepSeek-V4-Flash-slim5l \
    --context-length 2048 \
    --input-sequence-length 256 \
    --quant-type w8a8h1_sefp
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (内置路径) | HF 模型目录 |
| `--context-length` | 2048 | 最大序列长度 (KV cache 容量) |
| `--input-sequence-length` | 256 | Prefill 输入长度 (必须整除 context-length) |
| `--quant-type` | w8a8h1_sefp | 量化类型 (w8a8h1_sefp / w16a16) |
| `--num-logits-to-keep` | 1 | Prefill 只输出最后一个位置的 logits |
| `--work-dir` | 自动生成 | 输出目录 |

**输出产物：**

```
work_dir/
├── meta.json                  # 模型元信息 (推理引擎读取)
├── hf_config/                 # HF tokenizer + config 副本
├── token_embedding.pt         # token embedding 权重
├── hmonnx/
│   ├── prefill/*.onnx         # Prefill 模型
│   └── decode/*.onnx          # Decode 模型
└── convert.log                # 导出日志
```

`meta.json` 关键字段:
- `compressed_kv_cache`: compressor 层的 compressed KV 形状
  - Layer 2: `[1,1,64,512]` (CSA, compress_rate=4)
  - Layer 3: `[1,1,2,512]` (HCA, compress_rate=128)
  - Layer 4: `[1,1,64,512]` (CSA, compress_rate=4)

## Step 2: 量化精度验证

> W8A8 量化精度已修复（历史根因：interleaved RoPE 的 `cos/sin.repeat_interleave(2)`
> 在 xh2a 量化 runtime 下致 rope 维度失真，已改用 rope_dim/2 配对广播修复）。
> 下面两步验证应全部达标。

### Step 2a: 验证 wrapping 正确 (wrapped PT vs HF)

确认 xh2modelzoo 的 wrapping 逻辑无误（排除适配层误差，应 >0.999）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python \
    examples/llm/deepseek_v4/verify_wrapped_pt.py \
    --model /data01/nfs_shared/llm/DeepSeek-V4-Flash-slim5l
```

**预期:** cosine > 0.999。若不达标，问题在 wrapping 适配层，与量化无关。

### Step 2b: 验证 ONNX 量化精度 (prefill + decode)

ONNX 量化模型 vs HF float 基线。采用 **teacher forcing**（ONNX decode 用 HF 的
token 输入），排除 argmax 翻转导致的输入分叉干扰，反映真实量化误差：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python \
    examples/llm/deepseek_v4/verify_prefill_decode_cosine.py \
    --config work_dirs/DeepSeek-V4-Flash-slim5l-XH2a-2k-w8a8h1_sefp/meta.json \
    --steps 8 --device cuda:7
```

**预期:**
- prefill cosine > 0.99
- decode dec 1-8 cosine > 0.95（W8A8 正常水平，多数步 >0.99）
- argmax 大部分匹配（个别 token top1 接近可能翻转，属 W8A8 固有量化误差）

**通过判定:** prefill > 0.99 且 decode 每步 > 0.95 即达标。

> **若不达标如何排查:** 精度问题最常见根因是 RoPE（interleaved）——
> `_apply_interleaved_rope` 的 `repeat_interleave` 在 xh2a 量化下曾致 rope 维度
> (qk_rope_head_dim=64) cosine≈0，已改 rope_dim/2 配对广播修复。若复现，重点查该处。
> `trash/` 下有历史诊断脚本（依赖已移除的 DSV4_DEBUG_LAYERS dump 机制，仅供参考）。

## Compressed KV Cache 机制

Prefill 时 compressor 层生成 compressed KV (对长序列的压缩表示), decode 时复用:

```
Prefill:
  CSA: 256 tokens → 64 compressed entries (compress_rate=4)
  HCA: 256 tokens → 2 compressed entries  (compress_rate=128)

Decode:
  compressed KV 拼在 KV cache 前面:
  [compressed(0..63) | regular(64..319) | zero_padding(320..2111)]
  hybrid mask: j > (i + compressed_len) 精确遮住 padding
```

Prefill 额外输出 compressed_kv, decode 接收为额外输入。Non-compressor 层用 `[1,1,1,512]`
零张量占位, 保持输入对齐。

## 精度基准 (w8a8h1_sefp，修复 RoPE 后)

| 指标 | Cosine | 说明 |
|------|--------|------|
| Wrapped PT vs HF (逐层) | > 0.999 | 浮点模型精度验证 |
| ONNX prefill vs HF | 0.9995 | Prefill 量化精度 |
| ONNX decode vs HF (teacher forcing, dec 1-8) | 0.97~0.99 | W8A8 正常水平 |

> **修复记录:** 修复前 prefill cosine 仅 0.882（RoPE rope 维度 cosine≈0）。
> 根因：`_apply_interleaved_rope` 的 `cos/sin.repeat_interleave(2)` 在 xh2a 量化
> runtime 下数值失真。修复：去掉 repeat_interleave，cos/sin 用 rope_dim/2 配对广播
> （`ox = x1*cos - x2*sin`、`oy = x2*cos + x1*sin`）。诊断教训：含 -inf 的
> attn_weights 上 cos_finite 会产生假象，应用不含 -inf 的中间量定位。

## 文件说明

```
examples/llm/deepseek_v4/
├── README.md                              # 本文档
├── deepseek_v4_xh2a_export_hmonnx.py     # Step 1: 量化导出
├── verify_wrapped_pt.py                   # Step 2a: wrapped PT vs HF 验证
└── verify_prefill_decode_cosine.py        # Step 2b: ONNX prefill+decode cosine 验证（唯一量化精度验证脚本）

xh_model_zoo/xh_llm/models/deepseek_v4/
├── __init__.py                            # 公开 API (Inference, ConvertConfig)
├── _model.py                              # Attention / RotaryEmb / Norm wrappers
├── _layers.py                             # DecoderLayer / Model / ForCausalLM
├── _moe.py                                # SparseMoeBlock wrapper (256 experts)
├── _hc.py                                 # HyperConnection / HyperHead
├── _compressor_nl.py                      # CSA/HCA 非线性 + interleaved RoPE
├── inference.py                           # ONNX 推理引擎 (prefill + decode)
├── deepseek_v4_converter.py              # HF → wrapped → quantized → ONNX converter
├── deepseek_v4_convert_config.py         # 导出配置
└── deepseek_v4_hf_compatible.py          # HF GenerationMixin 兼容层 (lm_eval 用)
```
