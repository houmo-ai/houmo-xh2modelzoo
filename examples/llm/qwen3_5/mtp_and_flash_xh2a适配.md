# Qwen3.5 Spec Decode（MTP / DFlash）xh2a 改造说明

> **状态**：4B / 9B / 27B dense 与 35B-A3B / 3.6-35B-A3B MoE 的 MTP / DFlash 改造已全部闭环  
> **目标**：按新的 speculative decoding ABI 重做 target / draft / export / runtime，去掉旧的“逐 token verify + draft 侧外部 cos/sin + 无 draft cache”实现。

---

## 1. 改造目标

这次改造不是在旧实现上打补丁，而是把 Qwen3.5 的 spec decode 路径改成下面这套契约：

1. **Target prefill** 输出 spec decode 需要的 hidden，并初始化 target 自身 cache。
2. **Target decode / verify** 一次接收 `1 + K` 个 token 做 batch verify，而不是 Python 侧逐 token 调 decoder。
3. **MTP draft** 具备显式 KV cache，分成 prefill / decode 两张图。
4. **DFlash draft** 先把 target hidden 预计算成每层 KV cache，再用 decode 图消费 noise embedding。
5. **draft 侧不再输入 cos/sin**，统一改成连续 position + 图内预计算 RoPE cache + `DynamicSlice`。
6. **verify 输出可回滚的线性注意力状态**，包括 compact conv cache 和 stacked recurrent state。

默认 `K = 4`。

---

## 2. 新 ABI 总览

| 图 | 作用 | 关键输入 | 关键输出 |
| --- | --- | --- | --- |
| target prefill | prompt 预填充 | `inputs_embeds`, `past_seq_length`, `current_input_length`, caches | `logits`, `target_hidden` 或 `post_norm_hidden`, target caches |
| target decode | verify | 当前 token + `K` 个 draft token，一次性输入 | `logits[1+K]`, hidden 序列, verify-time conv/recurrent outputs |
| MTP prefill | 用 target prefill hidden 建 MTP cache | `post_norm_hidden`, `next_token_embedding`, `past_seq_length`, `current_input_length`, MTP KV cache | `logits`, `post_norm_out`（KV cache 输入端原地更新，不再作为输出） |
| MTP decode | 单步起草 | 单 token embedding + 单步 hidden + MTP KV cache | `logits`, `post_norm_out`（KV cache 原地更新） |
| DFlash context | 把 target hidden 预计算成 DFlash 各层 KV cache | `target_hidden`, `past_seq_length`, `current_input_length`, per-layer KV cache | per-layer `present_key_cache_i`, `present_value_cache_i` |
| DFlash decode | 一次生成 K 个 draft 候选 | `noise_embedding[1+K]`, `past_seq_length`, `current_input_length`, `attn_mask`, per-layer KV cache | `logits[1+K]` |

### 2.1 Position / RoPE

- 不再从 runtime 往 draft 图喂 `rope_cos` / `rope_sin`。
- 统一使用连续 position。
- 图内预计算长 RoPE cache。
- prefill / decode 通过 `past_seq_length` 和 `current_input_length` 用 `DynamicSlice` 取对应位置。

### 2.2 Verify 语义

verify 输入为：

```text
[current_token] + [draft_0, draft_1, draft_2, draft_3]
```

一次 target decode 输出：

- 5 个位置的 logits
- 5 个位置对应的 spec hidden
- verify 所需的 compact conv cache 输出
- verify 所需的 stacked recurrent state 输出

这样 acceptance / rejection 不再需要 Python 侧把 decoder 一步一步跑 5 次。

---

## 3. MTP 改造

### 3.1 原则

MTP 的 block 与主模型 block 同类，都是 causal self attention，因此必须带 KV cache，不能再做成“无 cache 的一步 head”。

### 3.2 现实现状

已落地：

- `_mtp_model.py`
  - 图内 RoPE cache
  - 显式 `past_key_cache` / `past_value_cache`（输入端原地更新，不再作为输出）
  - 输出 `logits`, `post_norm_out`
- `qwen3_5_mtp_model.py`
  - wrapper 改成新输入输出名
  - 真实 4B 权重加载已打通

### 3.3 Prefill 语义

prompt prefill 阶段，MTP cache 不是空转，而是消费：

```text
(target hidden[t], token[t+1] embedding)
```

这样 draft cache 与 target prefill 后的上下文保持一致。

---

## 4. DFlash 改造

### 4.1 原则

DFlash 是非因果交叉注意力。target 多层 hidden 不应该每轮重复重算 K/V，而应该在 prefill / verify 后直接转成 DFlash 各层 KV cache。

### 4.2 现实现状

已落地：

- `_dflash_model.py`
  - `context` 模式：`target_hidden -> per-layer KV cache`
  - `decode` 模式：`noise_embedding + cached target KV -> logits`
  - 图内 RoPE cache
- `qwen3_5_dflash_model.py`
  - context / decode 双模式 wrapper
  - 真实 4B DFlash 权重加载与前向已打通

### 4.3 Mask 语义

DFlash decode 使用 additive mask：

- valid 位置为 `0`
- invalid 位置为大负数

旧 runtime 里把有效位置写成 `1` 的错误已经修正。

---

## 5. Target 图改造

### 5.1 Prefill

target prefill 在 spec decode 模式下会额外导出：

- `post_norm_hidden`（MTP）
- `target_hidden`（DFlash，多层 hidden concat）

并将 `num_logits_to_keep=0`，避免只保留最后一个 logits 导致 hidden 序列不完整。

### 5.2 Decode / Verify

target decode 在 spec decode 模式下会：

- 固定导出 `input_sequence_length = 1 + K`
- 打开 `verify_output_intermediates`
- 输出 verify 所需的线性状态序列

线性状态语义：

- `conv_cache_out`：输出可裁剪的 compact 滑窗序列
- `recurrent_state_out`：输出每个 verify step 的状态栈，供 acceptance 后精确选择

---

## 6. Runtime 改造

核心 runtime 在：

- `xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_spec_decode_onnx_model.py`

当前设计：

1. **prefill**
   - target prefill 跑完整个 prompt
   - 同步累计 MTP cache 或 DFlash context cache
2. **draft**
   - MTP：逐步用 `post_norm_out` 链式起草 K 个 token
   - DFlash：一次 decode 输出 `1 + K` 个位置 logits，从位置 1..K 读 draft token
3. **verify**
   - target decode 一次验证 `[current] + drafts`
4. **rollback / commit**
   - 从 verify 输出里截取 acceptance 对应的 conv / recurrent state
   - DFlash 追加 accepted hidden 到 context cache
   - MTP 从 accepted point 重建 draft KV cache

这一步的目标就是彻底去掉“为了省事直接一步一步 verify”的旧逻辑。

---

## 7. Export 改造

导出脚本：

- `examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py`

已改成多图导出：

### 7.1 MTP

- `prefill_onnx`
- `decode_onnx`
- `draft_prefill_onnx`
- `draft_decode_onnx`

### 7.2 DFlash

- `prefill_onnx`
- `decode_onnx`
- `draft_context_onnx`
- `draft_decode_onnx`

### 7.3 Meta

`meta.json` / 标准化 meta 中已增加：

- `draft_prefill_onnx`
- `draft_context_onnx`
- `draft_decode_onnx`
- `block_size`
- `verify_length`
- `hidden_output_name`

---

## 8. 关键文件

| 文件 | 说明 |
| --- | --- |
| `xh_model_zoo/xh_llm/models/qwen3_5/_mtp_model.py` | MTP draft 核心实现 |
| `xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_mtp_model.py` | MTP wrapper |
| `xh_model_zoo/xh_llm/models/qwen3_5/_dflash_model.py` | DFlash context/decode 核心实现 |
| `xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_dflash_model.py` | DFlash wrapper |
| `xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_spec_decode_onnx_model.py` | spec decode runtime |
| `xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_onnx_model.py` | target ONNX runtime helper |
| `examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py` | 多图导出入口 |
| `examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py` | HMONNX runtime loader / benchmark |

---

## 9. 当前验证进度

### 9.1 已完成

- Python 语法检查通过
- 新 MTP draft 模型张量级 smoke test 通过
- 新 DFlash context/decode 模型张量级 smoke test 通过
- 真实 `weights/Qwen3.5-4B` MTP draft 权重加载与前向通过
- 真实 `weights/Qwen3.5-4B-DFlash` context/decode 权重加载与前向通过
- 真实 `weights/Qwen3.5-4B` **MTP 四图导出完成**
- 真实 `weights/Qwen3.5-4B` **DFlash 四图导出完成**
- 真实 4B **MTP HMONNX spec decode 闭环通过**
- 真实 4B **DFlash HMONNX spec decode 闭环通过**

### 9.2 4B 实测结果

使用命令：

```bash
source env.sh
python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py \
  --config work_dirs/qwen3_5_4b_mtp_k4_export/meta.json \
  --device cuda:0 --exec_device cuda:0 --dtype fp16 \
  --prompt 'Count from 1 to 5.' --max_new_tokens 16 --warmup_runs 0 --benchmark_runs 1

python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py \
  --config work_dirs/qwen3_5_4b_dflash_k4_export/meta.json \
  --device cuda:0 --exec_device cuda:0 --dtype fp16 \
  --prompt 'Count from 1 to 5.' --max_new_tokens 16 --warmup_runs 0 --benchmark_runs 1
```

结果：

| 模式 | 输出 | rounds | draft_tokens | accepted | avg_accepted_per_round | avg_latency_s | tokens_per_second |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MTP | `1, 2, 3, 4, 5.` | 5 | 20 | 9 | 1.80 | 12.7996 | 1.0938 |
| DFlash | `1, 2, 3, 4, 5.` | 10 | 40 | 4 | 0.40 | 19.6854 | 0.7112 |

说明：

- **MTP**：4B 路径已拿到稳定非零 acceptance。修复点有两个：一是 MTP draft 的 Qwen3.5 RMSNorm 需要按 `(1 + weight)` 语义加载；二是 runtime 里 MTP 的 cache / position 语义要比 target `past_seq_len` 少 1，不能直接复用 target 的位置计数。
- **DFlash**：4B 路径已经拿到非零接受率，说明新的 `target hidden -> context cache -> batch verify` 链路是有效的。

### 9.3 多模型 rollout 最终结果

| 模型 | 模式 | 产物位置 | ABI 结果 | 真实验证 |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B | MTP | `work_dirs/qwen3_5_4b_mtp_k4_export/` | prefill `logits/hidden = [1,256,*]`，decode verify `logits/hidden = [1,5,*]` | `Count from 1 to 5.` → `1, 2, 3, 4, 5.` |
| Qwen3.5-4B | DFlash | `work_dirs/qwen3_5_4b_dflash_k4_export/` | prefill `target_hidden = [1,256,*]`，decode verify `target_hidden = [1,5,*]` | `Count from 1 to 5.` → `1, 2, 3, 4, 5.` |
| Qwen3.5-9B | MTP | `work_dirs/qwen3_5_9b_mtp_k4_export/` | 同新 ABI | 已闭环通过 |
| Qwen3.5-9B | DFlash | `work_dirs/qwen3_5_9b_dflash_k4_export/` | 同新 ABI | 已闭环通过 |
| Qwen3.5-27B | MTP | `work_dirs/qwen3_5_27b_mtp_k4_export/` | 同新 ABI | `Count from 1 to 5.` → `1 2 3 4 5`，`accepted=5`，`avg_accepted_per_round=1.00` |
| Qwen3.5-27B | DFlash | `work_dirs/qwen3_5_27b/qwen3_5_27b_xh2a_Qwen3.5-27B/` | prefill `target_hidden=[1,256,25600]`，decode `target_hidden=[1,5,25600]` | `中国的首都是哪里？请只回答城市名。` → `北京`，`accepted=1` |
| Qwen3.5-35B-A3B | MTP | `work_dirs/Qwen3.5-35B-A3B-XH2a-2k-w8a8h0_sefp-spec_mtp/` | prefill `post_norm_hidden=[1,256,2048]`，decode `post_norm_hidden=[1,5,2048]` | `Count from 1 to 5.` → `1, 2, 3, 4, 5.` |
| Qwen3.5-35B-A3B | DFlash | `work_dirs/Qwen3.5-35B-A3B-XH2a-2k-w8a8h0_sefp-spec_dflash/` | prefill `target_hidden=[1,256,10240]`，decode `target_hidden=[1,5,10240]` | `中国的首都是哪里？请只回答城市名。` → `北京` |
| Qwen3.6-35B-A3B | MTP | `work_dirs/Qwen3.6-35B-A3B-XH2a-2k-w8a8h0_sefp-spec_mtp/` | target prefill/decode 同新 ABI；draft prefill 有 KV cache，draft decode 输出单步 logits + cache | `Count from 1 to 5.` → `1, 2, 3, 4, 5\n` |
| Qwen3.6-35B-A3B | DFlash | `work_dirs/Qwen3.6-35B-A3B-XH2a-2k-w8a8h0_sefp-spec_dflash/` | prefill `target_hidden=[1,256,10240]`，decode `target_hidden=[1,5,10240]` | `中国的首都是哪里？` → `中国的首都是北京。` |

### 9.4 MoE MTP draft 权重兼容修复

35B-A3B 与 3.6-35B-A3B 的 MTP draft 都是 sparse MoE，但权重命名不一致：

- **Qwen3.5-35B-A3B**
  - `mtp.layers.0.mlp.experts.<idx>.gate_proj.weight`
  - `mtp.layers.0.mlp.experts.<idx>.up_proj.weight`
  - `mtp.layers.0.mlp.experts.<idx>.down_proj.weight`
- **Qwen3.6-35B-A3B**
  - `mtp.layers.0.mlp.experts.gate_up_proj`
  - `mtp.layers.0.mlp.experts.down_proj`

因此 `_mtp_model.py` 里的 MoE loader 需要同时支持：

1. 逐 expert 的旧格式；
2. `gate_up_proj` / `down_proj` 的打包格式。

修复后：

- 3.6 MTP draft 导出不再在 `expert_idx = int(parts[3])` 处崩溃；
- `draft_onnx/` 已正确导出；
- runtime 已能用新 draft 图完成真实 speculative decode。

### 9.5 forced 模式 float suite（标准 benchmark 口径）

对比脚本：

- float MTP：`examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py`
- float DFlash：`/data01/home/yujy/work/dflash/dflash/qwen3_5_transformers_benchmark.py`
- 统一采集器：`examples/llm/qwen3_5/qwen3_5_spec_decode_metrics.py`
- 结果目录：`output/spec_decode_metrics/`

评测口径：

1. float 推理全部使用 **forced verify**。
2. MTP 使用 `TEST_PROMPTS` 10 条样例；DFlash 使用 `mt-bench` 10 条样例。
3. 全部使用 `max_new_tokens=4096`，并按 **thinking / non-thinking** 分开统计。
4. `text_match_cases` 表示 speculative 输出与 baseline **完全一致** 的 case 数。
5. **全量平均** 反映真实运行观察；**matched-only 平均** 只统计完全对齐 baseline 的 case，更适合回答“产生同样输出时需要多少 decoder 次数”。
6. `Qwen3.6-35B-A3B / MTP` 的 float suite 因 bf16 mixed-dtype Triton 问题改为 **fp16** 重跑。

实际测试入口：

- MTP 标准脚本：`examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py`
- DFlash 标准脚本：`/data01/home/yujy/work/dflash/dflash/qwen3_5_transformers_benchmark.py`
- 本次批量重跑命令统一经由：`examples/llm/qwen3_5/qwen3_5_spec_decode_metrics.py`

命令示例：

```bash
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate xhquant
cd /data01/home/yujy/work/xh2modelzoo
export PYTHONPATH=/data01/home/yujy/work/dflash:/data01/home/yujy/work/xh2modelzoo

# MTP
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_spec_decode_metrics.py \
  --output-json output/spec_decode_metrics/qwen3_5_27b_float_mtp_suite_thinking.json \
  float-mtp-suite \
  --model weights/Qwen3.5-27B \
  --max-new-tokens 4096 \
  --num-draft-tokens 4 \
  --dtype bf16 \
  --system-prompt 'You are a helpful assistant.' \
  --enable-thinking

# DFlash（上面这批 suite 历史结果使用 block size = 5）
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_spec_decode_metrics.py \
  --output-json output/spec_decode_metrics/qwen3_5_27b_float_dflash_suite_mtbench_thinking.json \
  float-dflash-suite \
  --model weights/Qwen3.5-27B \
  --draft-model weights/Qwen3.5-27B-DFlash \
  --dataset mt-bench \
  --max-samples 10 \
  --max-new-tokens 4096 \
  --dtype bf16 \
  --block-size 5 \
  --enable-thinking
```

说明：

- **上面这批 DFlash suite 历史结果的 block size = 5**，即运行命令里的 `--block-size 5`。
- 对应 DFlash decode 每轮会输出 `1 + K = 5` 个位置，其中 1 个是当前位置、4 个是 draft token。
- 后续已按 parity audit 修正统一采集器默认行为：**未显式传 `--block-size` 时，改为读取 draft model config（当前 DFlash 模型均为 16）**，不再硬编码 5。

#### 9.5.1 MTP（全量平均）

| 模型 | 模式 | cases | text_match_cases | baseline decoder | spec decoder | MTP prefill | MTP decode | 总接受率 | matched-only 总接受率 | 结果文件 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B | non-thinking | 10 | 0 | 1261.3 | 429 | 1 | 2557.7 | 0.4817 | 0.0000 | `qwen3_5_4b_float_mtp_suite_non_thinking.json` |
| Qwen3.5-4B | thinking | 10 | 0 | 2543.6 | 825.5 | 1 | 4960.4 | 0.4899 | 0.0000 | `qwen3_5_4b_float_mtp_suite_thinking.json` |
| Qwen3.5-9B | non-thinking | 10 | 0 | 995.2 | 319 | 1 | 1970.5 | 0.5394 | 0.0000 | `qwen3_5_9b_float_mtp_suite_non_thinking.json` |
| Qwen3.5-9B | thinking | 10 | 0 | 2266.8 | 622.7 | 1 | 4103.5 | 0.6310 | 0.0000 | `qwen3_5_9b_float_mtp_suite_thinking.json` |
| Qwen3.5-27B | non-thinking | 10 | 0 | 1208 | 346.2 | 1 | 2246.4 | 0.6031 | 0.0000 | `qwen3_5_27b_float_mtp_suite_non_thinking.json` |
| Qwen3.5-27B | thinking | 10 | 0 | 2427.5 | 632.6 | 1 | 4219.3 | 0.6664 | 0.0000 | `qwen3_5_27b_float_mtp_suite_thinking.json` |
| Qwen3.5-35B-A3B | non-thinking | 10 | 1 | 1048.2 | 339.3 | 1 | 2142.6 | 0.5560 | 0.5556 | `qwen3_5_35a3b_float_mtp_suite_non_thinking.json` |
| Qwen3.5-35B-A3B | thinking | 10 | 0 | 2450.7 | 669.3 | 1 | 4350.7 | 0.6239 | 0.0000 | `qwen3_5_35a3b_float_mtp_suite_thinking.json` |
| Qwen3.6-35B-A3B | non-thinking | 10 | 0 | 1227.3 | 581.2 | 1 | 3048.7 | 0.3236 | 0.0000 | `qwen3_6_35a3b_float_mtp_suite_non_thinking_fp16.json` |
| Qwen3.6-35B-A3B | thinking | 10 | 0 | 3065.4 | 1249.4 | 1 | 6340.9 | 0.2774 | 0.0000 | `qwen3_6_35a3b_float_mtp_suite_thinking_fp16.json` |

#### 9.5.2 MTP（仅 text-match case）

| 模型 | 模式 | matched cases | baseline decoder | spec decoder | MTP prefill | MTP decode | matched-only 总接受率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | non-thinking | 1 | 28 | 9 | 1 | 52 | 0.5556 |

#### 9.5.3 DFlash（全量平均）

| 模型 | 模式 | cases | text_match_cases | baseline decoder | spec decoder | DFlash prefill | DFlash decode | 总接受率 | matched-only 总接受率 | 结果文件 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B | non-thinking | 10 | 1 | 379.5 | 165.1 | 0 | 165.1 | 0.3376 | 0.4904 | `qwen3_5_4b_float_dflash_suite_mtbench_non_thinking.json` |
| Qwen3.5-4B | thinking | 10 | 0 | 2661.4 | 895.9 | 0 | 895.9 | 0.6385 | 0.0000 | `qwen3_5_4b_float_dflash_suite_mtbench_thinking.json` |
| Qwen3.5-9B | non-thinking | 10 | 1 | 418.5 | 187.6 | 0 | 187.6 | 0.3629 | 0.6037 | `qwen3_5_9b_float_dflash_suite_mtbench_non_thinking.json` |
| Qwen3.5-9B | thinking | 10 | 0 | 2545.3 | 802.3 | 0 | 802.3 | 0.6089 | 0.0000 | `qwen3_5_9b_float_dflash_suite_mtbench_thinking.json` |
| Qwen3.5-27B | non-thinking | 10 | 4 | 403.2 | 182.3 | 0 | 182.3 | 0.3947 | 0.5116 | `qwen3_5_27b_float_dflash_suite_mtbench_non_thinking.json` |
| Qwen3.5-27B | thinking | 10 | 0 | 2274.9 | 745.9 | 0 | 745.9 | 0.5810 | 0.0000 | `qwen3_5_27b_float_dflash_suite_mtbench_thinking.json` |
| Qwen3.5-35B-A3B | non-thinking | 10 | 0 | 428.4 | 198.4 | 0 | 198.4 | 0.3454 | 0.0000 | `qwen3_5_35a3b_float_dflash_suite_mtbench_non_thinking.json` |
| Qwen3.5-35B-A3B | thinking | 10 | 0 | 2431.7 | 748 | 0 | 748 | 0.5696 | 0.0000 | `qwen3_5_35a3b_float_dflash_suite_mtbench_thinking.json` |
| Qwen3.6-35B-A3B | non-thinking | 10 | 0 | 459.3 | 531.6 | 0 | 531.6 | 0.6158 | 0.0000 | `qwen3_6_35a3b_float_dflash_suite_mtbench_non_thinking.json` |
| Qwen3.6-35B-A3B | thinking | 10 | 0 | 2324 | 964.7 | 0 | 964.7 | 0.5300 | 0.0000 | `qwen3_6_35a3b_float_dflash_suite_mtbench_thinking.json` |

#### 9.5.4 DFlash（仅 text-match case）

| 模型 | 模式 | matched cases | baseline decoder | spec decoder | DFlash prefill | DFlash decode | matched-only 总接受率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | non-thinking | 1 | 153 | 52 | 0 | 52 | 0.4904 |
| Qwen3.5-9B | non-thinking | 1 | 139 | 41 | 0 | 41 | 0.6037 |
| Qwen3.5-27B | non-thinking | 4 | 265.5 | 90 | 0 | 90 | 0.5116 |

#### 9.5.5 DFlash / MTP parity 与 block-size 审计

审计目标：

1. DFlash 先回到旧脚本口径 `block_size=16`，MTP 固定 `K=4`。
2. 对比旧脚本与当前统一采集器在 **fp16 / 同一用例 / non-thinking** 下的输出、decoder 次数和接受率是否完全一致。
3. parity 成立后，再比较 DFlash `block_size=16` 与 `block_size=5` 的差异。

parity 结果：

| 路线 | 模型 | 配置 | 结论 |
| --- | --- | --- | --- |
| DFlash | Qwen3.5-4B | fp16, block_size=16 | 完全一致 |
| DFlash | Qwen3.5-9B | fp16, block_size=16 | 完全一致 |
| DFlash | Qwen3.5-27B | fp16, block_size=16 | 完全一致 |
| DFlash | Qwen3.5-35B-A3B | fp16, block_size=16 | 完全一致 |
| DFlash | Qwen3.6-35B-A3B | fp16, block_size=16 | 完全一致 |
| MTP | Qwen3.5-4B | fp16, K=4 | 完全一致 |
| MTP | Qwen3.5-9B | fp16, K=4 | 完全一致 |
| MTP | Qwen3.5-27B | fp16, K=4 | 完全一致 |
| MTP | Qwen3.5-35B-A3B | fp16, K=4 | 完全一致 |
| MTP | Qwen3.6-35B-A3B | fp16, K=4 | 完全一致 |

结论：**当前 collector 与旧 float benchmark 的实现没有漂移；此前 DFlash acceptance 偏低的直接原因是评测口径把 block size 改成了 5，而不是 collector 算法写错。**

DFlash `block_size=16` vs `block_size=5`（同一条 `mt-bench` prompt，fp16，non-thinking）：

| 模型 | acc@16 | acc@5 | spec decoder@16 | spec decoder@5 | 与 baseline 对齐@16 | 与 baseline 对齐@5 | 16/5 输出是否相同 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Qwen3.5-4B | 0.0780 | 0.2833 | 118 | 120 | 是 | 否 | 否 |
| Qwen3.5-9B | 0.0710 | 0.2703 | 124 | 123 | 是 | 是 | 是 |
| Qwen3.5-27B | 0.0959 | 0.3164 | 105 | 113 | 是 | 是 | 是 |
| Qwen3.5-35B-A3B | 0.0830 | 0.3065 | 114 | 115 | 否 | 否 | 是 |
| Qwen3.6-35B-A3B | 0.0844 | 0.2620 | 113 | 125 | 否 | 否 | 否 |

这说明：

- `block_size=5` **确实会显著抬高 acceptance rate**。
- 但 `block_size=5` **不保证保持与 `block_size=16` 或 baseline 相同的输出**；4B 和 Qwen3.6-35B-A3B 在该用例上已经出现文本差异。
- 因此若要求与旧 DFlash float benchmark 口径一致，应默认使用 **model config / 16**；`5` 只能作为显式实验参数，不能继续作为默认值。
- 需要特别区分两层语义：
  1. **训练 / checkpoint 默认值**：当前 DFlash draft config 的 `block_size` 全部是 **16**。
  2. **推理 / verify 运行值**：如果运行时显式传 `--block-size 5`，当前实现不是“先生成 16 再只看前 5 个”，而是**直接按 5 个位置构造整轮 verify**：
     - 第 1 个位置是当前已确定 token
     - 后 4 个位置是 draft token
     - verify 也只检查这 4 个 draft token 的连续命中数
- 所以用户说“训练默认 16，但验证时看前 5 个（第 1 个确定，看后 4 个接收率）”在**效果上接近**，但从代码实现上更准确的描述是：**训练默认 16，推理时可以 override 成 5；override 后整轮就是 1+4，而不是从 16 里截前 5。**

如果采用用户补充的另一种统计口径：

- **推理仍然使用 `block_size=16`**
- **只是在算接受率时，只看前 5 个位置（1 个 current + 4 个 draft）**

那么当前环境下（bf16, full10, `max_new_tokens=20`）的派生结果是：

| 模型 | 推理 block_size | text-match | baseline decoder | DFlash decoder | 按完整 15 个 draft 算的接受率 | 按前 4 个 draft 派生的接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 16 | 100.0% | 19.0 | 7.1 | 0.1474 | 0.4167 |
| Qwen3.5-9B | 16 | 90.0% | 19.0 | 6.8 | 0.1586 | 0.4654 |

这说明：

- **bs16 推理不变，只改统计口径到“前 5 位”后，接受率会明显上升**
- 但这个值依然**不是**“直接把推理改成 `block_size=5`”得到的结果；它只是从 **bs16 的真实 verify 结果**里，派生出“前 4 个 draft token 的命中率”
- 因此三种口径要严格分开：
  1. **bs16 推理 + 完整 15 个 draft 统计**
  2. **bs16 推理 + 只看前 4 个 draft 的派生统计**
  3. **bs5 推理 + 真实 4 个 draft 统计**

`Qwen3.5-4B`, `python_tutorial`, `bf16`, `block_size=16`, `max_new_tokens=20` 的前 5 位 trace 示例：

表头含义先说明清楚：

- `本轮 current`：这一轮已经确定、并放在 verify 第 0 个位置的 token
- `draft #1~#4`：DFlash 在这一轮提出的前 4 个 draft token（真实 bs16 一共会提 15 个，这里只展示前 4 个）
- `target verify 位置0~4`：target 在 verify 时，对位置 0~4 给出的贪心结果
  - 位置 0 对应 `current`
  - 位置 1~4 对应 `draft #1~#4`
- `full15 接受数`：按完整 bs16 语义，这一轮一共连续接受了多少个 draft token（最多 15）
- `prefix4 接受数`：如果只看前 4 个 draft token，这一轮接受了多少个；它等于 `min(full15 接受数, 4)`
- `committed 前缀`：这一轮真正提交到输出里的前几个 token 文本

| round | 本轮 current | draft #1~#4 | target verify 位置0~4 | full15 接受数 | prefix4 接受数 | committed 前缀 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 0 | `#` | `[' Binary', ' Search', ' Tree', ' (']` | `[' ', ' Search', ' Tree', ' (', 'BST']` | 0 | 0 | `#` |
| 1 | `' '` | `['1', '.', ' A', ' complete']` | `['user', '.', ' ', '.', ' guide']` | 0 | 0 | `' '` |
| 2 | `'user'` | `[':', ' A', ' Python', ' tutorial']` | `['user', '\\n', '\\n', ' script', ' for']` | 0 | 0 | `'user'` |
| 12 | `' '` | `['1', '.', ' A', ' Python']` | `['1', '.', '1', '.', ' program']` | 2 | 2 | `' 1.'` |

可以看到：

- bs16 推理时，verify 实际会看到 **1 + 15** 个位置
- 这个表里只把最前面的 **5 个位置**摘出来给人看
- 例如第 12 轮：
  - `current = ' '`
  - DFlash 提的前 4 个 draft 是 `['1', '.', ' A', ' Python']`
  - target verify 在位置 0~4 的结果是 `['1', '.', '1', '.', ' program']`
  - 位置 1 和位置 2 连续命中，所以 `prefix4 接受数 = 2`
  - 到后面位置开始不连续命中，就停止继续接受
- 这就是“**推理用 16，但统计时只看前 5 位**”的精确定义

#### 9.5.6 结果解读

- **标准 suite 口径下，acceptance rate 和 text-match 不再等价**：很多 thinking 档次都有 0.53~0.67 的平均接受率，但 `text_match_cases = 0/10`。
- **Qwen3.5 MTP** 在全量平均上整体比 DFlash 更强，27B/35B-A3B thinking 分别到 **0.6664 / 0.6239**。
- **Qwen3.6 MTP** 明显弱于 Qwen3.5，对应 non-thinking / thinking 只有 **0.3236 / 0.2774**，且必须切到 fp16 才能完成 float suite。
- **Qwen3.6 DFlash non-thinking** 虽然平均接受率有 **0.6158**，但 target spec decoder 平均次数 **531.6 > 459.3**，说明“接受率高”并不自动代表 decoder 次数减少。
- 上面这个 **Qwen3.6 DFlash non-thinking 的“spec decoder > baseline decoder”**，从原始 case 看**不是统计 bug**，而是输出路径已经明显跑偏：
  - `text_match_cases = 0/10`
  - 多个 case 的 speculative 输出 token 数远大于 baseline，甚至直接跑到 `max_new_tokens=4096`
  - 例如：
    - `mt-bench:0`：baseline `1180` token / `1179` decode，spec `4096` token / `1204` decode
    - `mt-bench:4`：baseline `272` token / `271` decode，spec `1942` token / `519` decode
- 计数本身是自洽的：当前配置 `block_size=5`，每轮最多接受 4 个 draft token，因此理论上
  - `spec_tokens ≈ spec_decode_rounds * (1 + 4 * acceptance_rate)`
  - `mt-bench:0` 代入后正好约等于 `4096`
- 所以这里更接近的问题是：**Qwen3.6 DFlash 在这组口径下虽然局部 token 接受率不低，但全局文本/EOS 路径严重偏离 baseline**。这会导致：
  1. speculative 输出比 baseline 更长
  2. target verify 轮数也随之变多
  3. 最终出现 `spec decoder > baseline decoder`
- 进一步按用户要求抽查 **dtype 敏感性**（`block_size=5`, non-thinking, `max_new_tokens=128`）后，结论更清楚：
  - `mt-bench:4`：bf16 在第 **9** 个 token 首次分叉；fp16 在 **128 token 内未分叉**
  - `mt-bench:7`：bf16 在第 **11** 个 token 分叉；fp16 推迟到第 **59** 个 token
  - `mt-bench:8`：bf16 在第 **14** 个 token 分叉；fp16 推迟到第 **25** 个 token
- 这说明 **fp16 明显比 bf16 稳定**；Qwen3.6 DFlash 的坏结果里，至少有一部分确实是 **bf16 数值误差放大**，不是单纯统计口径问题。
- 但 **fp16 也还没有做到“基本完全等于 baseline”**：case 7 / 8 仍会在 128 token 内分叉，只是比 bf16 晚很多。
- 这个 **fp32 多卡 benchmark 崩溃** 已经在 `examples/llm/qwen3_5/qwen3_5_spec_decode_metrics.py` 里修掉：当前脚本会在检测到 `hf_device_map` 跨多张 GPU 时，自动切到 **multi-GPU safe 的 DFlash forced 路径**，把 `target_hidden` / `draft_tokens` 归并到 target 输入卡后再做 verify。
- 修复后的实测（`CUDA_VISIBLE_DEVICES=1,2,3`, `float-dflash`, `Qwen3.6-35B-A3B`, `block_size=5`, `max_new_tokens=32`, `mt-bench:4`）已经可以正常输出 JSON，不再报跨卡 `torch.cat`：
  - fp32：`baseline decode=31`, `spec decode=31`, `text_match=false`, `overall_acceptance_rate=0.0081`
  - 同一脚本下的 fp16 回归检查：`baseline decode=31`, `spec decode=13`, `text_match=true`
- 所以现在新的结论是：**fp32 路径已经能跑，不再被 benchmark 框架本身挡住；而且至少在这个 case 上，fp32 结果依然明显偏离 baseline。** 这说明 Qwen3.6 DFlash 的问题并不只是 “bf16 精度太差”，还存在更深的 draft / verify 一致性问题。
- **DFlash / MTP parity 已确认**：当前 collector 与旧 benchmark 在用户要求口径下完全一致。
- **DFlash 默认 block size 已修正**：未显式传 `--block-size` 时跟随 draft model config（当前模型为 16），不再默认 5。
- 当前 20 组 suite 里，真正拿到 text-match 的只有 4 组：
  - DFlash non-thinking：Qwen3.5-4B（1/10）、Qwen3.5-9B（1/10）、Qwen3.5-27B（4/10）
  - MTP non-thinking：Qwen3.5-35B-A3B（1/10）
- 每个 case 的 `round_acceptance_rates`、`accepted_drafts_per_round`、baseline/spec decoder 次数都保存在对应 JSON；如果要逐 case 对表或看“每轮接受率”，直接打开 `output/spec_decode_metrics/*.json` 即可。

#### 9.5.7 参考飞书 DFlash 报告的复现说明

用户引用的飞书文档：`https://houmo.feishu.cn/wiki/LxrywqxSSi1h6EkBvyUcnTXtn2g`

这份文档里“DFlash 接受率很高”的结论，和前面 parity 审计里看到的低接受率，**不是同一组实验**。当前已确认的差异有：

1. 飞书文档的高接受率主结论来自 **小 block 的 acceptance sweep**，重点是 `block_size=2~4`，而不是 `16`。
2. 飞书文档的 correctness 主表是 **DFlash forced + block_size=4 + full10**。
3. 这次 parity 审计是按用户要求固定到 **fp16**，并且先看 **block_size=16**，随后再对比 `16 vs 5`；这和飞书文档不是同一口径。
4. 飞书附录 I 的 acceptance 指标打印的是 `benchmark.py` 里的 **Average Acceptance length** / `估计 draft 接受率` 体系；它和我们后来在统一采集器里看的 case-by-case 平均接受率不是同一个统计视角。

当前环境下，按最接近飞书文档的方式复现（**bf16 + DFlash forced + block_size=4 + full10**）得到：

| 模型 | max_new_tokens | text-match | baseline decoder | DFlash decoder | 平均 acceptance length | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-4B | 20 | 90.0% | 19.0 | 8.5 | 2.5231 | 最接近文档里的 `4B full10 = 90%` |
| Qwen3.5-9B | 20 | 100.0% | 19.0 | 7.9 | 2.7198 | 当前环境比文档里的 `90%` 更高 |
| Qwen3.5-4B | 32 | 80.0% | 30.7 | 14.1 | 2.3404 | 输出更长后开始偏离 |
| Qwen3.5-9B | 32 | 100.0% | 30.8 | 13.2 | 2.5218 | 当前环境仍保持全匹配 |

额外长度 sweep（bf16, bs=4, full10）：

- **4B**：`max_new_tokens=8/12/16` 都是 **100%**，`20` 变成 **90%**，`24/32` 变成 **80%**。
- **9B**：`max_new_tokens=8/12/16/20/24/32` 当前环境里都是 **100%**。

这说明两件事：

1. 飞书文档里的“高接受率/高匹配率”主要对应 **bf16 + 小 block + 较短输出**。
2. 当前环境虽然能复现出 **4B 在 20 token 附近掉到 90%** 这一趋势，但不能逐字复刻飞书文档里的所有数字，说明**文档产出时使用的权重/运行入口/统计口径与当前环境存在漂移**。

逐轮 trace 示例（`Qwen3.5-4B`, `python_tutorial`, `bf16`, `block_size=4`, `max_new_tokens=32`）：

- baseline decoder 次数：**32**
- DFlash decoder 轮数：**25**
- 文本是否一致：**否**

节选若干轮：

| round | current | draft | verify argmax | accepted drafts | committed | replacement |
| ---: | --- | --- | --- | ---: | --- | --- |
| 10 | `\\n` | `['#', ' Python', ' Search']` | `['#', ' ', ' ', ' Algorithm']` | 1 | `\\n#` | `' '` |
| 11 | `' '` | `['1', '.', ' Binary']` | `['1', '.', '1', ' Search']` | 2 | `' 1.'` | `'1'` |
| 12 | `'1'` | `['.', ' ', ' **']` | `['user', ' **', '1', 'The']` | 0 | `'1'` | `'user'` |
| 17 | `' '` | `['1', ' Python', ' Binary']` | `['1', '.', ' ', ' Search']` | 1 | `' 1'` | `'.'` |
| 21 | `'1'` | `['.', ' Introduction', ' Binary']` | `['.', ' ', '\\n', ' Search']` | 1 | `'1.'` | `' '` |

说明：

- 这里的 `draft` 是 DFlash 当前轮提议的 3 个 draft token。
- `verify argmax` 是 target 在 verify 阶段逐位置给出的贪心结果。
- `accepted drafts` 表示从左到右连续命中的 draft token 个数；一旦某一位不一致，就停止接受并走 replacement。
- trace 里保留的是 **原始 token 级文本**，前几轮会看到 `user`、换行等 chat template 边界 token，这是正常现象。
- 持久化复现产物已保存到：`/data01/home/yujy/.copilot/session-state/e3e48b87-4e5e-4d72-a5fd-e34c13cadf20/files/dflash_doc_repro_summary.md`

---

## 10. 当前结论

这次改造的重点不是“把 draft 模型能跑起来”，而是把 spec decode 的**状态契约**做对：

- MTP 必须有 draft KV cache
- DFlash 必须复用由 target hidden 预计算出来的 per-layer KV cache
- verify 必须是 target batch verify，而不是 Python 侧单步循环
- draft 侧不再接收外部 cos/sin
- 线性注意力状态必须能在 acceptance / rejection 后正确回滚

当前所有目标模型已经证明：

- target prefill / decode 与 draft 多图 ABI 能在真实 HMONNX 上闭环
- runtime 已不再依赖逐 token forced verify
- MTP / DFlash 两条 draft 路线都已经在 dense / MoE 上跑出正向 acceptance 或正确文本
- DFlash 的 `target hidden -> per-layer KV cache -> append accepted hidden back to cache` 语义已在 dense / MoE 路线上统一落地
- MTP 的 draft KV cache、prefill/decode 拆图和 batched verify 语义已在 dense / MoE 路线上统一落地


总结：
1. 当前的实现不符合我的需求，没有意义，verify一个一个验证，是不是有病，draft的意义何在
2. 模型上的问题：

    1. MTP 
        - 模型架构上，MTP与主模型的block是一样的，都是causal self attention，那应该也是有kv cache的，现在导出的hmonnx是没有的，这个是有问题的，需要修改（https://houmo.feishu.cn/wiki/FF5dw4zMqipWuEkMmQlclylcnWc 的第 8 节MTP 投机解码流程图与详解）
        - cos和 sin作为模型的输入，这个也是不对的，position id是连续的，与普通的大模型一样，如qwen3嘛，直接预计算一个256k的 position，模型的输入增加一个 current_length和valid length，然后用 dynamic slice（start valid lenght, prefill长度固定 256, decode长度固定1）（使用xhquant/nn/modules里面的算子）的方式输入到模型里就好了，
        - 那么从输入来看, mtp导出hmonnx的时候也需要prefill和 decode，因为prefill阶段和decode生成draft token的输入不一样，prefill阶段是target模型prefill的 hidden state和token id embedding，用来产生mtp的kv cache。prefill的输入长度与 target模型的 prefill的token长度一样，kv cache的长度与target模型的kv cache长度一致。decode阶段的输入长度是 1。
        - targe verify（decode) 模型，输入（1 + K（生成的draft token数量）），与原始decode的不一样的地方：conv state的输出原来是1*8192*(3+1)，现在是1*8192*(3+1+k)，因为conv 滑窗重叠所以可以直接往后加，recurrent state的原来是输出1个recurrent state，现在是输出 1+k 个recurrent state（状态会覆盖，为了回滚需要都保存），因为每生成一个token就要验证一次，所以需要输出当前token的conv/recurrent state，供验证失败时回滚使用，验证时假设成功 2 个，那么conv cache就取1*8192*(3+1+2 - 4, 3+1+2)，recurrent state就取第3个（0是prefill的状态，1是第一个token的状态，2是第二个token的状态）.
        - K设 4吧
    2. DFlash
        - 模型架构上，DFlash的输入由noise embedding和target模型的hidden state，组成，注意力是非因果交叉注意力，target模型多层的 hidden state需要cat起来作为输入,经过fc和 norm后送到每一层的交叉注意力中，经过k_proj, v_proj, k_norm和rope，那么其实是可以这样的，这一部分target hidden states转成kv cache是可以从target 模型prefill阶段的hidden state把dflash的每一个 layer的k_proj, v_proj, k_norm和rope接上，不就成了kv cache了么，这样就不需要每轮都重算了，直接预计算好了。这样的话，Dflash也需要每个Layer一个kv cache，长度与target模型的kv cache长度一致，由于他是全局注意力，需要增加mask屏蔽kv cache多余的部分， mask的大小就可以直接是kv cache的长度。
        - 这个kv cache，target hidden state prefill的时候是生成好的，直接在dflash 运行前放入，使用valid length和 block size来填充当前 draft cache进去，然后再计算attention。 verify阶段模型也要跟prefill一样将dflash的每一个 layer的k_proj, v_proj, k_norm和rope接上，验证完后，如果accept了2个token，那么就把这2个token的hidden state转成dflash的kv cache追加到原来预计算的kv cache后面，valid length增加2，下一轮dflash时就可以直接使用了，这样就不需要每轮都重算了,  变成了mask有效部分是valid length + 2 + block size。
        - DFlash的输入noise embedding是1+k个token的embedding，k个draft token（有一个占位token）, 这个大小 1+k = 16是模型默认的，但是从计算上来说好像没有必要吧，1+k = 5，对于精度而言，感觉也不影响。
        - cos和 sin作为模型的输入，这个也是不对的，position id是连续的，与普通的大模型一样，如qwen3嘛，直接预计算一个256k的 position，模型的输入增加一个 current_length和valid length，然后用 dynamic slice（start valid lenght, prefill长度固定 256, decode长度固定1）（使用xhquant/nn/modules里面的算子）的方式输入到模型里就好了，
        - targe verify（decode) 模型，输入（1 + K（生成的draft token数量）），与原始decode的不一样的地方：conv state的输出原来是1*8192*(3+1)，现在是1*8192*(3+1+k)，因为conv 滑窗重叠所以可以直接往后加，recurrent state的原来是输出1个recurrent state，现在是输出 1+k 个recurrent state（状态会覆盖，为了回滚需要都保存），因为每生成一个token就要验证一次，所以需要输出当前token的conv/recurrent state，供验证失败时回滚使用，验证时假设成功 2 个，那么conv cache就取1*8192*(3+1+2 - 4, 3+1+2)，recurrent state就取第3个（0是prefill的状态，1是第一个token的状态，2是第二个token的状态）.
    3. 通病
        - Dflash和mtp的verify是一样的， 为了省事， verify直接用decoder直接一步一步验证，坚决不应该这么做
3. 模型的路径
    - 浮点模型
        1. weights/Qwen3.5-4B
        2. weights/Qwen3.5-9B
        3. weights/Qwen3.5-27B
        4. weights/Qwen3.5-35B-A3B
        5. weights/Qwen3.6-35B-A3B
    - DFlash模型
        1. weights/Qwen3.5-4B-DFlash
        2. weights/Qwen3.5-9B-DFlash
        3. weights/Qwen3.5-27B-DFlash
        4. weights/Qwen3.5-35B-A3B-DFlash
        5. weights/Qwen3.6-35B-A3B-DFlash
4. 当前的实现都是不对的，需要推倒重来，按照上面说的方式来实现，MTP和DFlash的draft模型都需要修改，target模型的prefill和decoder也需要修改，导出脚本也需要修改，推理脚本也需要修改，工作量比较大。
5. 拆解任务，完成需求，修改脚本，验证hmonnx推理的正确性


浮点的mtp是这个：examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py 
浮点的dflash是这个：/data01/home/yujy/work/dflash/dflash/qwen3_5_transformers_benchmark.py
浮点推理都是使用的 force模式

量化的话使用杠杠导出的每个模型的mtp和dflash的hmonnx来测试
conda activate xhquant
卡随便用
最后输出MarkDown，和飞书文档，我想要这些数据
  1. 标准推理：非draft模型的decoder次数
  2. draft模型推理
    1. mtp
        1. 产生同样的输出的他的decoder次数, 以及mtp prefill和decode的次数
        2. 接收率，每轮的接收率以及总的接收率
    2. dflash
        1. 产生同样的输出的他的decoder次数, 以及dflash prefill和decode的次数
        2. 接收率，每轮的接收率以及总的接收率

我觉得是有问题的，浮点的接收率很低呢，你的脚本是不是有问题，之前
浮点的mtp是这个：examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py 
浮点的dflash是这个：/data01/home/yujy/work/dflash/dflash/qwen3_5_transformers_benchmark.py
的测试是很高的呀

然后呢，测试数据也太少了，按照浮点的标准测试脚本，里面有很多测试用例的，其次输出token也要设很多才行，4096这样，think和非think要单独测试，对吧


那我觉得脚本实现是有问题的，dflash的接收率太低了，之前在浮点的dflash测试脚本里，接收率是很高的呀，你看看是不是脚本实现有问题了，之前的脚本是这个：/data01/home/yujy/work/dflash/dflash/qwen3_5_transformers_benchmark.py
他当时固定 block size = 16，
我想你做dflash实验：
1. 先把 block size 固定为 16, 与之前的测试口径保持一致，
然后呢，对比之前的脚本与现在的脚本，都固定float16，block size 固定为 16，测试同样的用例，dflash的输出和结果是不是一模一样，分别对比所有模型
2. 如果结果不一样了，说明现在的脚本实现有问题了，需要修复，直到结果和之前的脚本完全一致了，再进行下一步测试。 
3. 结果一致了之后，再把 block size 从 16 改成 5，看看结果会有什么变化，是否必须设 16

对于mtp也是如此，k设 4，先对比之前的脚本与现在的脚本，测试同样的用例，mtp的输出和结果是不是一模一样，分别对比所有模型，如果结果不一样了，说明现在的脚本实现有问题了，需要修复，直到结果和之前的脚本完全一致了，再进行下一步测试。

对于结果还是有疑惑，https://houmo.feishu.cn/wiki/LxrywqxSSi1h6EkBvyUcnTXtn2g 这个文档中测试的浮点的dflash（/data01/home/yujy/work/dflash/dflash/qwen3_5_transformers_benchmark.py）很高呀，你刚刚对比跑出来的结果看着接收率不高，是什么问题呢，我想你复现了下这个文档中的结果，并补充下标准推理：非draft模型的decoder次数和draft模型的decoder次数. 例子也贴上，浮点推理的结果，dflash推理，每步的结果，verify之后接收的结果，我想看看每步的结果是否符合预期



- 浮点模型
    1. weights/Qwen3.5-4B
    2. weights/Qwen3.5-9B
    3. weights/Qwen3.5-27B
    4. weights/Qwen3.5-35B-A3B
    5. weights/Qwen3.6-35B-A3B
- DFlash模型
    1. weights/Qwen3.5-4B-DFlash
    2. weights/Qwen3.5-9B-DFlash
    3. weights/Qwen3.5-27B-DFlash
    4. weights/Qwen3.5-35B-A3B-DFlash
    5. weights/Qwen3.6-35B-A3B-DFlash
- auto round w4量化模型
  1. 4B: /data01/home/yujy/work/auto-round/output/Qwen3.5-4B-mode1-llm-only
  2. 9B: /data01/home/yujy/work/auto-round/output/Qwen3.5-9B-mode1-llm-only
  2. 27B: /data01/home/yujy/work/auto-round/output/sym-mode1
  3. 35B: /data01/home/yujy/work/auto-round/output/Qwen3.5-35B-A3B-mode1-llm-only
  4. 36B: /data01/home/yujy/work/auto-round/output/Qwen3.6-35B-A3B-mode1-llm-only
分析了examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py代码基本没有问题，
下面我要做大规模测试，测试用例，构造256条，涵盖人文、社科、科技、数学、tool calls、Coding等多个领域，prompt长度在 100 到 1000 之间均匀分布，测试输出token最大数量设置为8192，分别测试think和non-think两种情况。
首先：
  - 任务 1：
    - 模型需要重新导出，之前设的kv cache上长度是 2048，改成 8192。
      导出模型时dflash的block size设为16，mtp的k设为4
    - 现在导出的都是w8a8的模型，我想同时导出w4a8，auto round量化的模型。
    - 然后呢，模型需要重新导出，之前设的kv cache上长度是 2048，改成 8192。
    - verify那边拆掉concat，按 recurrent_state0_1这种方式输出，concat在芯片上不友好，qwen3_5_xh2a_spec_decode_test.py里也要改一下，按照新的输出方式来回滚。
    - 导出模型时dflash的block size设为16，mtp的k设为4
  - 任务 2：
    - 基于上面构造的测试用例，在examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py上进行大规模测试，
    - 需要获得的结果，在当前的脚本上，
        - 增加标准推理：非draft模式的decoder次数，产生同样的输出的他的decoder次数, 以及mtp prefill和decode的次数，  dflash 产生同样的输出的他的decoder次数, 以及dflash prefill和decode的次数（这个理论上用prefill和verify的模型就可以完成，需要你verify的时候构造的输入有效的只有一个 1）
  - 结果展示：
    - 最终输出MarkDown，和飞书文档，在上面这些结果的基础上，增加每种类别数据一个case 非draft和 draft模型完整结果的对比。

conda activate xhquant
GPU空着的都可以用

你当前修改存在的问题：
1. 模型修改上，conv cache不需要修改，因为之前实现里，conv cache并不是concat的，利用conv step的连续性，拼在了一起
2. 结果上，你只构建了脚本，没有跑全量的测试，不满足要求
3. 作为我花钱请你干活，你需要完整的跑完导出，测试的所有流程，并把结果整理成文档，提供到飞书文档，而不是只提供了一个脚本，我自己去跑，去看结果，这样的工作方式是非常不负责任的。




遇到xhquant(/data01/home/yujy/work/xhquanttool)的一个 bug
在：git checkout e4868bca运行
python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py --config work_dirs/qwen3_5_9b_mtp_k4_export/meta.json
python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py --config work_dirs/qwen3_5_9b_dflash_k4_export/meta.json
没问题
在：develop分支上运行就挂了，怀疑是xhquant(/data01/home/yujy/work/xhquanttool)的develop分支上改了什么东西导致的，正在排查中，你帮我修复下
conda activate xhquant
选一张空闲的卡，完成
