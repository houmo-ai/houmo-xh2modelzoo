# Qwen3.5 MTP 投机解码 Benchmark 使用指南

## 1. 概述

`qwen3_5_mtp_benchmark.py` 是 Qwen3.5 系列模型（Dense/MoE）MTP（Multi-Token Prediction）投机解码的一站式基准测试脚本。它从 safetensors 中手动加载 HF Transformers 忽略的 `mtp.*` 权重，构建独立的 MTP Head，并实现了 **forced-decode 批量验证** 算法（v2），每轮只需 1 次主模型前向即可验证 K 个草稿 token。

### 1.1 支持的模型

| 模型 | 路径示例 | 类型 | 显存需求（fp16） |
|------|---------|------|----------------|
| Qwen3.5-9B | `weights/Qwen3.5-9B` | Dense | ~19 GB |
| Qwen3.5-27B | `weights/Qwen3.5-27B` | Dense | ~56 GB |
| Qwen3.5-35B-A3B | `weights/Qwen3.5-35B-A3B` | MoE | ~72 GB |

### 1.2 核心功能

| 模式 | 说明 | CLI 参数 |
|------|------|---------|
| 接受率测量 | 默认模式，batch 计算每个 prompt 的 top-1 接受率 | （默认） |
| 时延对比 | 逐 step 对比主模型 vs MTP Head 前向耗时 | `--timing` |
| 投机解码（顺序） | 逐 token 验证，每次两次前向 | `--spec-decode` |
| **投机解码（forced-decode）** | ★ 最优方案，K 个草稿批量验证，每轮 1 次前向 | `--forced-decode` |
| Multi-K 实验 | 一次跑完 K=2,3,4 的 forced-decode | `--multi-k` |

---

## 2. 环境准备

```bash
conda activate xhquant

# 设定使用的 GPU（例如 GPU 1）
export CUDA_VISIBLE_DEVICES=1
```

依赖：`torch`, `transformers`, `safetensors`（xhquant 环境已包含）。

---

## 3. 脚本架构详解

### 3.1 文件位置

```
examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py   # 主脚本（~1760 行）
```

### 3.2 核心模块划分

```
┌──────────────────────────────────────────────────────┐
│  MTP Head 构建层                                      │
│  ├─ MTPAttention / MTPMLP / MTPMoEExpert             │
│  ├─ MTPDecoderLayer / MTPMoEDecoderLayer             │
│  ├─ Qwen3_5MTPHead                                    │
│  │   ├─ forward_batch()  — 批量前向（prefill 阶段）     │
│  │   └─ forward_step()   — 单步前向（decode 阶段）      │
│  ├─ load_mtp_weights()   — 从 safetensors 加载 mtp.*  │
│  └─ build_mtp_head()     — 组装完整 MTP Head           │
├──────────────────────────────────────────────────────┤
│  DeltaNet Forced-Decode Patch                         │
│  ├─ _patched_deltanet_forward()  — 猴子补丁           │
│  │   seq_len>1 时逐 token 走 recurrent 路径            │
│  │   保存中间 conv+recurrent state 供回滚              │
│  ├─ ForcedDecodeContext           — 上下文管理器        │
│  ├─ rollback_deltanet_states()   — 状态回滚（含conv）   │
│  └─ trim_full_attention_kv()     — 裁剪 KV cache      │
├──────────────────────────────────────────────────────┤
│  投机解码核心算法                                       │
│  ├─ generate_drafts()            — MTP 自回归生成 K 草稿│
│  └─ speculative_decode_forced()  — ★ 完整 forced-decode│
├──────────────────────────────────────────────────────┤
│  Benchmark 包装层                                      │
│  ├─ benchmark_speculative_decode_forced()             │
│  └─ benchmark_multi_k()                               │
└──────────────────────────────────────────────────────┘
```

### 3.3 PreNormCapture — 获取主模型 pre-norm 隐状态

MTP Head 需要主模型最后一层 RMSNorm **之前** 的隐状态。通过 `register_forward_pre_hook` 挂在 `TextModel.norm` 上：

```python
class PreNormCapture:
    def __init__(self, model):
        self._hook = _get_text_model(model).norm.register_forward_pre_hook(self._fn)
    def _fn(self, _module, args):
        self.hidden_states.append(args[0].detach())
```

每次 `model()` 调用后，`cap.hidden_states[0]` 即为 `[B, L, H]` 的 pre-norm hidden。

---

## 4. 核心算法：Forced-Decode 投机解码（v2）

### 4.1 为什么需要 Forced-Decode？

Qwen3.5 的 DeltaNet 层有两条路径：
- **chunk 路径**（seq_len > 1，prefill 用）：并行处理，但忽略已有 recurrent state
- **recurrent 路径**（seq_len = 1，decode 用）：逐 token 更新 conv+recurrent state

投机解码需要一次前向验证 K+1 个 token（seq_len = K+1），但直接走 chunk 路径会导致结果不一致。Forced-decode 补丁强制将多 token 前向拆成逐 token recurrent 处理，同时保存中间状态供回滚。

### 4.2 算法流程

```
一轮投机解码：

1. MTP Head 自回归生成 K 个草稿 token: [d₀, d₁, ..., d_{K-1}]
   └─ step 0 用真实 main hidden，step 1..K-1 用 MTP 自己的 hidden（推测性 KV）

2. 拼接 [next_tok, d₀, d₁, ..., d_{K-1}]，送入主模型做 1 次前向
   └─ ForcedDecodeContext 激活 → DeltaNet 逐 token 走 recurrent
   └─ 中间状态保存：token 0 的 state, token 1 的 state, ...

3. 逐位验证：logits[j] 的 top-1 == d_j ?

4a. 全部接受（accepted_count == K）：
    ├─ 提交 K+1 个 token（next_tok + K 个草稿）
    ├─ bonus token = argmax(logits[K])
    ├─ 更新 MTP KV（用真实 main hidden）
    └─ 继续下一轮

4b. 位置 j 拒绝（accepted_count < K）：
    ├─ 提交 j+1 个 token（next_tok + j 个接受的草稿）
    ├─ rollback DeltaNet state → 恢复到 token j 后的 conv + recurrent
    ├─ trim KV cache 删除后 K-j 个多余条目
    ├─ replacement = argmax(logits[j])
    └─ 继续下一轮
```

### 4.3 DeltaNet 状态回滚（关键）

DeltaNet 有 **两种** 需要回滚的状态：

| 状态类型 | 形状 | 说明 |
|---------|------|------|
| `conv_state` | `[B, conv_dim, kernel_size=4]` | causal conv1d 滑动窗口 |
| `recurrent_state` | `[B, num_heads, head_dim, head_dim]` | gated delta rule 循环状态 |

```python
# 保存中间状态（_patched_deltanet_forward 中）：
if t < seq_len - 1:
    _intermediate_deltanet_states[layer_idx].append({
        "conv": cache_params.conv_states[layer_idx].clone(),
        "recurrent": cache_params.recurrent_states[layer_idx].clone(),
    })

# 回滚（拒绝时）：
def rollback_deltanet_states(past_kv, rollback_idx):
    for layer_idx, states_list in _intermediate_deltanet_states.items():
        past_kv.conv_states[layer_idx] = states_list[rollback_idx]["conv"]
        past_kv.recurrent_states[layer_idx] = states_list[rollback_idx]["recurrent"]
```

**消融实验证明**：只回滚 recurrent 不回滚 conv 会导致接受率下降 ~12%，文本完全失配。

### 4.4 MTP 草稿生成

```python
generate_drafts(mtp_head, main_hidden, next_tok, K, position, mtp_kv):
    # Step 0: 用真实 main hidden + 真实 mtp_kv
    logits_0, h0 = mtp_head.forward_step(main_hidden, next_tok, ...)
    draft_0 = argmax(logits_0)

    # Step 1..K-1: 用 MTP 自身 hidden + 推测性 KV (clone)
    spec_kv = {"key": mtp_kv["key"].clone(), "value": mtp_kv["value"].clone()}
    for j in 1..K-1:
        logits_j, h = mtp_head.forward_step(h, drafts[-1], kv=spec_kv)
        draft_j = argmax(logits_j)
```

关键点：step 0 直接写入真实 `mtp_kv`，step 1+ 用 clone 的 `spec_kv` 以防污染。

### 4.5 MTP 计时与 mtp_ratio

脚本在主模型前向和 MTP 操作两端放置了 `torch.cuda.synchronize()` 时间栅栏：

```python
# 主模型计时
torch.cuda.synchronize()
_main_t0 = time.perf_counter()
with ForcedDecodeContext():
    out = model(input_ids=all_toks, ...)
torch.cuda.synchronize()
main_model_time_acc += time.perf_counter() - _main_t0

# MTP 计时（接受/拒绝路径均有）
torch.cuda.synchronize()
_mtp_t0 = time.perf_counter()
# ... MTP KV update + generate_drafts ...
torch.cuda.synchronize()
mtp_time_acc += time.perf_counter() - _mtp_t0

# 最终指标
mtp_ratio = mtp_time / (mtp_time + main_model_time)
```

---

## 5. CLI 用法

### 5.1 最简调用 — 单 prompt 接受率

```bash
CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-9B
```

### 5.2 Forced-decode 单个 prompt

```bash
CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-9B \
    --forced-decode \
    --num-draft-tokens 2 \
    --prompt "请解释量子纠缠的原理"
```

### 5.3 全 prompt × 单 K

```bash
CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-9B \
    --forced-decode \
    --num-draft-tokens 3 \
    --run-all \
    --max-new-tokens 128
```

### 5.4 ★ Multi-K 全量实验（推荐）

一次跑完 K=2,3,4 × 全部 10 个 prompt：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-9B \
    --multi-k \
    --run-all \
    --max-new-tokens 128
```

### 5.5 Timing benchmark

测量单 step 主模型与 MTP Head 的时延对比：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-9B \
    --timing \
    --timing-steps 50
```

### 5.6 完整参数列表

```
--model             模型路径 (default: weights/Qwen3.5-27B)
--prompt            自定义 prompt（覆盖 --run-all）
--run-all           跑全部 10 个内置 TEST_PROMPTS
--timing            时延 benchmark
--spec-decode       顺序投机解码（v1，已废弃）
--forced-decode     forced-decode 投机解码（v2，推荐）
--num-draft-tokens  每轮草稿 token 数 K (default: 1)
--multi-k           自动跑 K=2,3,4 的 forced-decode
--max-new-tokens    最大生成 token 数 (default: 4096)
--timing-steps      timing 模式步数 (default: 50)
--dtype             数据类型 (default: bf16)
--system-prompt     系统提示词
```

---

## 6. 综合实验脚本

对于需要精确控制输出（JSON + timing + text 对比）的场景，使用独立实验脚本：

```bash
# 保存为 /tmp/mtp_full_experiment.py 或项目中任意路径
CUDA_VISIBLE_DEVICES=1 python /tmp/mtp_full_experiment.py \
    --model weights/Qwen3.5-9B \
    --tag 9b
```

该脚本：
1. 对 10 个 prompt 分别跑 baseline（标准自回归）
2. 对每个 prompt 跑 K=2, 3, 4 的 forced-decode
3. 记录接受率、文本匹配率、MTP 计算占比、完整文本
4. 输出到 `/tmp/mtp_full_{tag}.json`

可并行跑三个模型：
```bash
CUDA_VISIBLE_DEVICES=1 python /tmp/mtp_full_experiment.py --model weights/Qwen3.5-9B --tag 9b &
CUDA_VISIBLE_DEVICES=2 python /tmp/mtp_full_experiment.py --model weights/Qwen3.5-27B --tag 27b &
CUDA_VISIBLE_DEVICES=7 python /tmp/mtp_full_experiment.py --model weights/Qwen3.5-35B-A3B --tag 35b &
wait
```

---

## 7. 输出指标说明

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| `acceptance_rate` | 草稿 token 接受率 | accepted_drafts / (num_rounds × K) |
| `text_match` | 生成文本是否与 baseline 完全一致 | `baseline_text == mtp_text` |
| `mtp_ratio` | MTP 计算占 decode 总时间的比例 | mtp_time / (mtp_time + main_model_time) |
| `tok_per_s` | 总吞吐 | num_tokens / elapsed_s |
| `tok_per_fwd` | 每次主模型前向产出 token 数 | num_tokens / num_main_fwd |
| `avg_tok_per_round` | 每轮平均产出 token 数 | num_tokens / num_rounds |

---

## 8. 调试技巧

### 8.1 减小 max-new-tokens 快速验证

```bash
CUDA_VISIBLE_DEVICES=1 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-9B \
    --forced-decode --num-draft-tokens 2 \
    --max-new-tokens 32 \
    --prompt "Hello"
```

### 8.2 关键断点位置

| 功能 | 函数 | 大致行号 |
|------|------|---------|
| MTP 权重加载 | `build_mtp_head()` | L473 |
| DeltaNet 补丁逐 token 处理 | `_patched_deltanet_forward()` | L967 |
| 状态保存（conv + recurrent） | 同上，`if t < seq_len - 1` | L1003 |
| 状态回滚 | `rollback_deltanet_states()` | L1058 |
| 草稿生成 | `generate_drafts()` | L1085 |
| 投机解码主循环 | `speculative_decode_forced()` | L1132 |
| 接受路径 | 同上，`if accepted_count == K` | L1237 |
| 拒绝路径 + 回滚 | 同上，`else` | L1274 |

### 8.3 VS Code launch.json

见项目 `.vscode/launch.json` 中的 debug 配置，以 9B 模型为例提供了多种调试场景。

---

## 9. MTP Head Trimming Workflow

本节介绍将 MTP Head 裁剪到 K 个热门 token 后的完整工作流：**Rerank → Benchmark → Export NPU**。

### 9.1 三步流程

```
Step 1  Rerank — 构建 K-trimmed 仓库
Step 2  Benchmark — 测量裁剪后接收率损失
Step 3  Export — 导出 NPU 可用的 HMONNX
```

### 9.2 Step 1：Rerank（生成 reranked 仓库）

```bash
cd analysis/mtp_head_longtail/v2_reranked
MODEL=/path/to/weights/Qwen3.5-4B K=82000 python pipeline/04_merge_select.py
MODEL=/path/to/weights/Qwen3.5-4B K_LIST=82000 python pipeline/05_export_reranked.py
MODEL=/path/to/weights/Qwen3.5-4B K=82000 \
  DST=/path/to/weights/Qwen3.5-4B-reranked-K82000 python pipeline/09_rerank_full_model.py
```

产物：`weights/Qwen3.5-4B-reranked-K82000/`，含 `mtp_lm_head.pt` shape `[82000, 2560]`。

### 9.3 Step 2：Benchmark（测量接收率）

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-4B-reranked-K82000 \
    --mtp-head-pt weights/Qwen3.5-4B-reranked-K82000/mtp_lm_head.pt \
    --forced-decode --num-draft-tokens 4 --max-new-tokens 128 \
    --save-json results/k82000_acceptance.json
```

**实测结果（K=82000，Qwen3.5-4B，Sprint 2）**：接收率 delta = **−1.73 pt**（原始 vs K=82000）。
可接受范围参考：delta < 3 pt 视为合格。如需更低损失可尝试更大 K（如 100000）。

### 9.4 Step 3：Export NPU HMONNX（`--mtp-head-k`）

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
    --hf_model_dir weights/Qwen3.5-4B \
    --config configs/qwen3_5/qwen3_5_4b_xh2a.py \
    --mtp-head-k 82000 \
    --spec_decode_mode mtp \
    --work_dir work_dirs/qwen3_5_4b_mtpk82000 \
    --max_sequence_length 4096
```

若 `weights/Qwen3.5-4B-reranked-K82000/mtp_lm_head.pt` 已存在，则跳过生成直接导出。
强制重新生成：加 `--force-rerank`。自定义输出路径：加 `--reranked-repo-dir <path>`。

### 9.5 调整 K 的指引

| K | 显存节省（lm_head） | 接收率 delta | 推荐场景 |
|---|---------|---------|---------|
| 32000 | ~58% | TBD | 极限省显存 |
| 48000 | ~68% | TBD | 平衡 |
| 82000 | ~46% | −1.73 pt | 当前推荐 |
| 100000 | ~34% | TBD | 最小损失 |

工具链不绑死 K，任意值均可一键导出。

对裁剪后模型的 benchmark 用法 → 参见 [MTP_HEAD_PRUNING.md §5](./MTP_HEAD_PRUNING.md#5-benchmark接收率)
