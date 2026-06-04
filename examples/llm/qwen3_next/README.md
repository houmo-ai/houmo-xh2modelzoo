# Qwen3-Next XH2a 导出与推理

Qwen3-Next（80B-A3B MoE）的 XH2a 量化导出、多阶段验证、Golden 生成与 HMONNX 推理。

Qwen3-Next 是混合架构模型，同时使用标准注意力（Attention）和线性注意力（GatedDeltaNet），
包含 48 层 Transformer、512 个 MoE 专家（每次激活 4 个），总参数 80B、激活参数约 3B。

## 环境依赖

推荐使用 `xhquant` 环境，并在仓库根目录执行：

```bash
conda activate xhquant
export PYTHONPATH=./
```

> **GPU 需求**：80B MoE 模型需要多张 GPU。推荐 5×A100 80GB（400GB 显存）。
> 导出脚本通过 `AutoOffloadGraphModel` 自动将模型分布到多张 GPU 上。

## 支持情况

| 能力 | 状态 | 量化方案 |
|------|------|----------|
| BF16 LLM 导出 | ✅ 已支持 | w8a8h1_sefp |
| GPTQ LLM 导出 | ✅ 已支持 | w4a8h0_ssfp |
| 多阶段验证（`--valid`） | ✅ 已支持 | GPU 推理验证 |
| Golden 生成（`--golden`） | ✅ 已支持 | prefill + decode |
| HMONNX 推理 Demo | ✅ 已支持 | 单轮文本 |


## 推荐完整流程（4-layer 预检 → 80B 整网导出 → Demo）

Qwen3-Next 当前默认兼容参数为：

- `split_conv_cache=True`：默认 q/k/v 三路 conv cache；需要旧 merged 格式时传 `--no_split_conv_cache`。
- `normalize_force_fp32=False`：也可显式传 `--normalize-force-fp32=False`。
- `use_manual_depthwise_conv1d=False`：也可显式传 `--use_manual_depthwise_conv1d=False`。
- `fuse_gdr_ops=False`：也可显式传 `--fuse_gdr_ops=False`。

### 1. 4-layer 裁剪模型快速预检

```bash
conda activate xhquant
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1
OUT=work_dirs/qwen3_next_verify_$(date +%Y%m%d_%H%M%S)

python examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py \
  --hf_model_dir weights/Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse-4layers \
  --work_dir "$OUT/4layers" \
  --max_sequence_length 512 \
  --normalize-force-fp32=False \
  --use_manual_depthwise_conv1d=False \
  --fuse_gdr_ops=False \
  --golden

python examples/llm/qwen3_next/qwen3_next_xh2a_hmonnx_test.py \
  --config "$OUT/4layers/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max-new-tokens 128 \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --resource-tight-mode \
  --device cuda \
  --exec-device cuda
```

预检重点：golden 目录中应能看到 split cache 输出，例如 `conv_cache_out_q_0`、`conv_cache_out_k_0`、`conv_cache_out_v_0`。

### 2. 80B A3B 整网导出 + Golden

80B A3B 整网导出会先把 GPTQ 权重反量化到导出图里，显存峰值明显高于运行时 demo。建议优先使用 8 张 80G GPU；如果机器显存不足，可以按实际资源收缩 `CUDA_VISIBLE_DEVICES` 和 `--golden_max_memory`，但双卡 80G 在反量化阶段可能 OOM。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
OUT=work_dirs/qwen3_next_verify_$(date +%Y%m%d_%H%M%S)
python examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py \
  --hf_model_dir weights/Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse \
  --work_dir "$OUT/80b" \
  --max_sequence_length 8192 \
  --normalize-force-fp32=False \
  --use_manual_depthwise_conv1d=False \
  --fuse_gdr_ops=False \
  --golden \
  --golden_multi_gpu \
  --golden_max_memory '{"0":"70GiB","1":"70GiB","2":"70GiB","3":"70GiB","4":"70GiB","5":"70GiB", "cpu":"320GiB"}'
```

### 3. 80B A3B 整网 Demo / Benchmark

```bash
export CUDA_VISIBLE_DEVICES=0,1,3,4
python examples/llm/qwen3_next/qwen3_next_xh2a_hmonnx_test.py \
  --config "$OUT/80b/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max-new-tokens 128 \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --resource-tight-mode \
  --device cuda \
  --exec-device cuda
```

## QTL-357 实测记录（2026-06-04）

本次 Qwen3-Next 80B-A3B 已按默认兼容参数验证：

- `split_conv_cache=True`
- `normalize_force_fp32=False`
- `use_manual_depthwise_conv1d=False`
- `fuse_gdr_ops=False`

```bash
BASE=/data01/home/yujy/work/xh2modelzoo/work_dirs/full_goal_20260604_0824
```

### 本次产物

| 类型 | `meta.json` | 说明 |
|------|-------------|------|
| 4-layer 预检 | `$BASE/qwen3_next_4layer_split_default_metafix/meta.json` | `architecture=Qwen3NextForCausalLM`，`kv=1`，`linear=3`，`max_context_tokens=512` |
| 80B 整网 | `$BASE/qwen3_next_80b_split_default_full_metafix_8gpu/meta.json` | `architecture=Qwen3NextForCausalLM`，`kv=12`，`linear=36`，`max_context_tokens=512` |

Golden 目录中已出现 split cache 命名，例如 `past_conv_cache_q_0/k_0/v_0` 与 `conv_cache_out_q_0/k_0/v_0`。

### 本次测试结果

| 类型 | 输出 token | latency(s) | tok/s | 结果说明 |
|------|------------|------------|-------|----------|
| 4-layer 预检 Demo | 30 | 31.3795 | 0.9560 | 4-layer 裁剪模型可跑通；输出语义不作为整网质量判断 |
| 80B 整网 Demo | 32 | 563.6241 | 0.0568 | 输出正常中文自我介绍 |

日志：

```bash
$BASE/logs/qwen3_next_4layer_split_default_metafix_export_golden.log
$BASE/logs/qwen3_next_4layer_split_default_metafix_long_demo.log
$BASE/logs/qwen3_next_80b_split_default_full_metafix_8gpu_export_golden.log
$BASE/logs/qwen3_next_80b_split_default_full_metafix_8gpu_long_demo.log
```

## Split Conv Cache 导出

`--split_conv_cache` 将线性注意力的 conv_cache 从单个合并 tensor 拆分为 3 个独立 tensor（q, k, v）。当前默认开启；可显式传 `--no_split_conv_cache` 使用合并 tensor。当前导出与 runtime 需要同时兼容 split/merged 两种输入输出命名。`--normalize-force-fp32=False`、`--use_manual_depthwise_conv1d=False`、`--fuse_gdr_ops=False` 是默认兼容配置。

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python \
  examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py \
  --hf_model_dir weights/Qwen3-Next-80B-A3B-Instruct \
  --split_conv_cache \
  --valid \
  --golden \
  --golden_multi_gpu
```

开启后 meta.json 中 `linear_cache.layers` 的每层会输出 `conv_shapes`（3 个 shape 的列表）而非 `conv_shape`（单个 shape）。ONNX 模型的输入/输出命名变化：

| 模式 | 输入名 | 输出名 |
|------|--------|--------|
| merged (`--no_split_conv_cache`) | `past_conv_cache_0` | `conv_cache_out_0` |
| 默认 split | `past_conv_cache_q_0` / `past_conv_cache_k_0` / `past_conv_cache_v_0` | `conv_cache_out_q_0` / `conv_cache_out_k_0` / `conv_cache_out_v_0` |

## LLM 导出

### BF16 → XH2a（w8a8）

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python \
  examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py \
  --hf_model_dir weights/Qwen3-Next-80B-A3B-Instruct \
  --valid \
  --golden \
  --golden_multi_gpu
```

### GPTQ → XH2a（w4a8）

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python \
  examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py \
  --hf_model_dir weights/Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse \
  --work_dir work_dirs/qwen3_next_80b_a3b_instruct/qwen3_next_80b_a3b_instruct_xh2a_Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse-20260330 \
  --valid \
  --golden \
  --golden_multi_gpu
```

### 仅重新生成 Golden（跳过导出）

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python \
  examples/llm/qwen3_next/qwen3_next_xh2a_export_hmonnx.py \
  --hf_model_dir weights/Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse \
  --work_dir work_dirs/qwen3_next_80b_a3b_instruct/qwen3_next_80b_a3b_instruct_xh2a_Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse-20260327 \
  --golden \
  --golden_multi_gpu \
  --package_release
```

### 关键参数说明

| 参数 | 说明 |
|------|------|
| `--hf_model_dir` | HuggingFace 模型目录（BF16 或 GPTQ） |
| `--valid` | 开启多阶段验证：Wrap vs HF、Wrap vs Frontend、Frontend vs Quant |
| `--golden` | 生成 prefill + decode 的 golden 数据 |
| `--golden_multi_gpu` | Golden 生成使用多 GPU（80B 必须开启） |

## 验证流程

开启 `--valid` 后，导出脚本会执行以下多阶段验证（全部在 GPU 上运行）：

1. **Wrap vs HF**：比较包装模型与原始 HuggingFace 模型的 logits 差异
2. **Wrap vs Frontend**：比较包装模型与 TorchFX 前端图的 logits 差异与全量 token 输出
3. **Frontend vs Quant**：比较前端图与量化图的 logits 差异与全量 token 输出

验证通过标准：
- Frontend 与 Wrap 的全量 token 输出 `exact_match=True`
- Quant 与 Frontend 的全量 token 输出 `exact_match=True`

### 已验证结果

#### BF16（Qwen3-Next-80B-A3B-Instruct）

| 阶段 | 指标 | 值 |
|------|------|------|
| Wrap vs HF (logits) | max abs error | 3.55e-01 |
| Wrap vs Frontend (logits) | max abs error | 0.0 |
| Frontend vs Quant (logits) | max abs error | 1.45 |
| Wrap vs Frontend (tokens) | exact_match | ✅ True (23/23) |
| Frontend vs Quant (tokens) | exact_match | ✅ True (23/23) |

#### GPTQ（Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse）

| 阶段 | 指标 | 值 |
|------|------|------|
| Frontend vs Quant (logits) | max abs error | 1.03 |
| Wrap vs Frontend (tokens) | exact_match | ✅ True (23/23) |
| Frontend vs Quant (tokens) | exact_match | ✅ True (23/23) |

## HMONNX 推理

### 单轮文本 Demo

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python \
  examples/llm/qwen3_next/qwen3_next_xh2a_demo.py \
  --config work_dirs/qwen3_next_80b_a3b_instruct/qwen3_next_80b_a3b_instruct_xh2a_Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse-20260330/export_meta_info.json \
  --prompt "你好，请介绍一下你自己" \
  --max-new-tokens 256 \
  --resource-tight-mode
```

### Benchmark 测试

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python \
  examples/llm/qwen3_next/qwen3_next_xh2a_hmonnx_test.py \
  --config work_dirs/qwen3_next_80b_a3b_instruct/qwen3_next_80b_a3b_instruct_xh2a_Qwen3-Next-80B-A3B-Instruct-gptqmodel-attn8-moe4-hs-mse-20260327/export_meta_info.json \
  --prompt "你好，请介绍一下你自己" \
  --max-new-tokens 32 \
  --resource-tight-mode
```

> **提示**：80B 模型推荐使用 `--resource-tight-mode`，避免 prefill 和 decode 模型同时驻留 GPU 导致 OOM。

### 示例输出

```
model_name: Qwen3-Next-80B-A3B-Instruct
prompt: 你好，请介绍一下你自己
output: 你好！我是Qwen，是阿里巴巴集团旗下的通义实验室自主研发的超大规模语言模型...
```

## 导出产物

导出完成后，产物目录结构如下：

```
work_dirs/qwen3_next_80b_a3b_instruct/
  qwen3_next_80b_a3b_instruct_xh2a_<model_name>/
    meta.json                          # 推理入口配置
    prefill_onnx/                      # Prefill 图（chunk 模式）
      qwen3_next_..._prefill.onnx
    decode_onnx/                       # Decode 图（recurrent 模式）
      qwen3_next_..._decode.onnx
    hmquant_xh2_qwen3next_80b_.../     # Golden 数据
      prefill/
      decode/
```

## 注意事项

1. Qwen3-Next 的 Prefill 和 Decode 是两张独立的图：Prefill 使用 chunk 模式处理长输入，Decode 使用 recurrent 模式逐 token 生成。
2. 80B 模型导出全流程（含验证和 Golden）约需 2.5 小时（BF16）或 2.5 小时（GPTQ）。
3. `--valid` 验证阶段使用 `AutoOffloadGraphModel` 将模型自动分布到多张 GPU，验证完毕后自动清理 GPU 显存。
4. GPTQ 模型权重命名规则：`attn8` 表示注意力层 8-bit，`moe4` 表示 MoE 专家层 4-bit，`hs` 表示 hessian，`mse` 表示 MSE 优化。
5. 运行前务必设置 `PYTHONPATH=./`，否则 `xh_model_zoo` 包导入可能失败。
