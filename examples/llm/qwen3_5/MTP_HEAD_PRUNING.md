# MTP Head 裁剪（K-trim + rerank）使用指南

## TL;DR

MTP（Multi-Token Prediction）输出 head 有 V=248,320 维，但推理时高频 token 只占少数。  
本工具把输出 head **裁到 K 维**，并把 hot token 重排到 `[0, K)`，减少 draft step 的 matmul IO。  
一行命令导出 4B 裁剪版：

```bash
python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
    --config configs/qwen3_5/qwen3_5_4b_xh2a.py \
    --hf_model_dir weights/Qwen3.5-4B \
    --dtype fp16 --max_sequence_length 8192 \
    --spec_decode_mode mtp --num_draft_tokens 4 \
    --spec-draft-head-weight-bits 4 \
    --work_dir work_dirs/qwen3_5_4b_xh2a_mtp_k81920 \
    --mtp-head-k 81920
```

---

## 1. 背景与原理

### MTP head 长尾问题

Qwen3.5 词表 V=248,320，但日常中英文对话 + 代码场景里，真正高频的 token 只有 3~8 万个左右。  
MTP draft step 每次都要做 `hidden @ lm_head.T`（形状 `[B, H] × [V, H]^T`），其中绝大多数行永远不会被采样到，白白占用 IO 和算力。

详细语料分析见 `analysis/mtp_head_longtail/`。

### 为什么要 rerank

裁剪不等于简单截断。原词表里 hot token 的 id 分布散乱（中文字符 id 多集中在 15 万+），直接截到前 K 行会漏掉大量高频 token。

**rerank 做的事**：
1. 统计语料中各 token 出现频次，选出 top-K 热词（`hot_ids`）
2. 构造一个置换映射 `id_map`：让 hot token 的新 id 落在 `[0, K)`
3. 把模型的 embed / lm_head 按此置换重排行

这样 MTP draft step 只需要计算前 K 行，完全不需要 scatter-back。

### 不变性

| 张量 | 处理方式 | 原因 |
|------|----------|------|
| `embed_tokens.weight` [V, H] | **保留全 V 维**，仅重排行顺序 | 任何 token 都可能出现在 prefill/decode 输入中 |
| `lm_head.weight` [V, H] | 重排后**裁到 [K, H]** | 只预测 hot token |
| `mtp_lm_head.pt` | 导出 [K, H] bf16，供推理侧直接加载 | — |

### tie_word_embeddings 处理

- **4B**（`tie_word_embeddings=True`）：embed 和 lm_head 共享权重。rerank 只重排 embed，`mtp_lm_head.pt` 取 embed 前 K 行。
- **9B**（`tie_word_embeddings=False`）：有独立 lm_head。rerank 同时重排 embed 和 lm_head，`mtp_lm_head.pt` 取 lm_head 前 K 行。

---

## 2. K 挡位建议

| K | 词表覆盖率（加权） | lm_head IO 节省 | 推荐场景 |
|---|---|---|---|
| 32,000 | 96.3% | ~87% | 极限省显存，接收率可能小幅下降 |
| 48,000 | 98.4% | ~81% | 推荐拐点，覆盖率高、IO 收益大 |
| **81,920** | **~99.7%** | **~67%** | **当前默认推荐，接收率几乎无损** |
| 82,000 | ~99.7% | ~67% | 与 81920 接近 |

**当前推荐：K=81920**
- 4B forced-decode 接收率与全词表差距 < 2 pt
- 9B K=81920 实测平均接收率 **82.59%**（math_proof 89.8%、english_explanation 87.5%、chinese_science 79.7%）

**现成数据**：内部 artifactory 已提供打包好的 `freqs/` + `selection/`（约 13 MB，包含 K ∈ {32000, 48000, 81172, 81920, 82000} 的 hot_ids / id_map / 频次统计），无需重新跑语料统计。

下载与解压：

```bash
# 1. 拉取打包好的数据集
mkdir -p analysis/mtp_head_longtail/v2_reranked
curl -L -o /tmp/qwen3_5_mtp_head_pruning_dataset_v1.tar.gz \
    http://10.10.1.53:8081/artifactory/model_zoo2/qwen3_5_mtp_head_pruning/qwen3_5_mtp_head_pruning_dataset_v1.tar.gz

# 2. 解压到本仓库默认期望路径
tar -xzf /tmp/qwen3_5_mtp_head_pruning_dataset_v1.tar.gz \
    -C analysis/mtp_head_longtail/v2_reranked/
# 解压后得到 analysis/mtp_head_longtail/v2_reranked/{freqs,selection}/
```

> 数据集打包脚本：`tools/package_mtp_dataset.sh`（默认打包 freqs+selection；
> `--upload` 触发再次上传）。reranked HF 仓库（约 1.6 GB / K）未打包，可由
> `examples/llm/qwen3_5/utils/model_utils.py` 中的 `rerank_model_for_mtp()` 按需重建。

> 注：4B 和 9B 词表 bit-exact 完全相同，hot_ids / id_map 可跨尺寸复用，无需为每个模型尺寸单独统计。

### 2.1 完整端到端流程（从原始语料重建 freqs / selection）

如果需要在新语料上重跑统计，或换 K 挡位，可走完整流程。整体链路：

```
原始语料 (corpus/)                       ── moss-003-sft-data + HF 流式公共语料
       │  build_mtp_hot_vocab.py
       ▼
词频统计 (freqs/*.counter.pt)            ── per-corpus token counter
       │  build_mtp_hot_vocab.py (--select)
       ▼
hot 词表选择 (selection/{hot_ids,id_map}_K.pt)
       │  examples/llm/qwen3_5/utils/model_utils.py::rerank_model_for_mtp()
       ▼
reranked HF 仓库 (weights/<model>-reranked-K<K>/)  + mtp_lm_head.pt
       │  qwen3_5_xh2a_export_hmonnx.py --mtp-head-k K
       ▼
HMONNX 产物 (prefill / decode / mtp_draft .hmonnx + meta.json)
```

**第一步：下载原始语料**

```bash
mkdir -p analysis/mtp_head_longtail/v2_reranked
curl -L -o /tmp/qwen3_5_mtp_head_pruning_corpus_v1.tar.gz \
    http://10.10.1.53:8081/artifactory/model_zoo2/qwen3_5_mtp_head_pruning/qwen3_5_mtp_head_pruning_corpus_v1.tar.gz
# 校验：sha256 = cd07ea6ccbbb125f86f63cbbf851133b03cb9a4cf381263501c03991d8bb5479

tar -xzf /tmp/qwen3_5_mtp_head_pruning_corpus_v1.tar.gz \
    -C analysis/mtp_head_longtail/v2_reranked/
# 得到 analysis/mtp_head_longtail/v2_reranked/corpus/moss/moss-003-sft-data.jsonl (~8.2 GB)
```

> 包内只含开源 `moss-003-sft-data.jsonl`（8.2 GB JSONL）。公共语料（BELLE、C4-en、
> CMMLU、COIG-PC、ShareGPT-Zh、wudao 等）在 `build_mtp_hot_vocab.py` 中由
> `datasets.load_dataset(..., streaming=True)` 在线拉取，不打包。`openclaw_raw.tsv`
> 为内部数据，未对外提供——`--rebuild-openclaw` 默认关闭，跳过即可。

**第二步：重建 freqs/**

```bash
python examples/llm/qwen3_5/build_mtp_hot_vocab.py \
    --hf_model_dir weights/Qwen3.5-4B \
    --corpus-dir analysis/mtp_head_longtail/v2_reranked/corpus \
    --output-dir analysis/mtp_head_longtail/v2_reranked \
    --num-proc 16
```

每个公共语料和 moss 子集会在 `freqs/` 下写一个 `{name}.counter.pt`。可用
`--skip-public` 重用已有的 `freqs/public_*.counter.pt`，只重统 moss。

**第三步：生成 selection/**

```bash
python examples/llm/qwen3_5/build_mtp_hot_vocab.py \
    --select-only \
    --output-dir analysis/mtp_head_longtail/v2_reranked \
    --k-list 32000 48000 81920 82000
```

写出 `selection/hot_ids_{K}.pt`、`selection/id_map_{K}.pt`、`selection/stats_{K}.json`。

**第四步：导出（自动调用 rerank_model_for_mtp）**

直接走 §3 的 `--mtp-head-k K`；脚本会在 `weights/<model>-reranked-K<K>/` 下生成
重排仓库（若不存在），并把 `--hf_model_dir` 重定向过去。也可显式调用：

```python
from examples.llm.qwen3_5.utils.model_utils import rerank_model_for_mtp
rerank_model_for_mtp(
    src_repo="weights/Qwen3.5-4B",
    dst_repo="weights/Qwen3.5-4B-reranked-K81920",
    hot_ids_path="analysis/mtp_head_longtail/v2_reranked/selection/hot_ids_81920.pt",
    id_map_path="analysis/mtp_head_longtail/v2_reranked/selection/id_map_81920.pt",
    k=81920,
)
```

---

## 3. 一键导出（最常用路径）

### 3.1 导出 4B（K=81920）

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
    --config configs/qwen3_5/qwen3_5_4b_xh2a.py \
    --hf_model_dir weights/Qwen3.5-4B \
    --dtype fp16 \
    --max_sequence_length 8192 \
    --spec_decode_mode mtp \
    --num_draft_tokens 4 \
    --spec-draft-head-weight-bits 4 \
    --work_dir work_dirs/qwen3_5_4b_xh2a_mtp_k81920 \
    --mtp-head-k 81920
```

### 3.2 导出 9B（K=81920）

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py \
    --config configs/qwen3_5/qwen3_5_9b_xh2a.py \
    --hf_model_dir weights/Qwen3.5-9B \
    --dtype fp16 \
    --max_sequence_length 8192 \
    --spec_decode_mode mtp \
    --num_draft_tokens 4 \
    --spec-draft-head-weight-bits 4 \
    --work_dir work_dirs/qwen3_5_9b_xh2a_mtp_k81920 \
    --mtp-head-k 81920
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--mtp-head-k K` | 触发 MTP head 裁剪，K 为保留的热词数 |
| `--reranked-repo-dir DIR` | reranked 仓库目标路径（默认：`weights/{model_name}-reranked-K{K}/`） |
| `--force-rerank` | 强制重新生成 reranked 仓库，即使已存在 |
| `--spec_decode_mode mtp` | 开启 MTP speculative decode 模式 |
| `--num_draft_tokens 4` | 每轮 draft 步数 |
| `--spec-draft-head-weight-bits 4` | MTP head 量化位宽 |

**自动行为**：传 `--mtp-head-k` 时，脚本自动在 `weights/{model_name}-reranked-K{K}/` 下生成重排仓库，再将 `--hf_model_dir` 重定向到该目录进行导出。若目录已存在且含 `mtp_lm_head.pt`，则跳过生成。

**期望输出**（关键日志行）：

```
[mtp-head-k] Using reranked hf_model_dir: weights/Qwen3.5-4B-reranked-K81920
```

**产物目录**：

```
work_dirs/qwen3_5_4b_xh2a_mtp_k81920/
├── prefill.hmonnx
├── decode.hmonnx
├── mtp_draft.hmonnx        # MTP draft head，输出维度 K
└── meta.json
```

---

## 4. 浮点验证

> **必跑！** 量化前先确认 reranked 仓库浮点 bit-exact，确保重排无误。

```bash
CUDA_VISIBLE_DEVICES=0 \
    MODEL_OLD=weights/Qwen3.5-4B \
    MODEL_NEW=weights/Qwen3.5-4B-reranked-K81920 \
    K=81920 \
    python analysis/mtp_head_longtail/v2_reranked/float_verify_dialog.py
```

**期望输出**：

```
match: 5/5, all_pass: True
```

9B 同理，将环境变量替换为对应路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
    MODEL_OLD=weights/Qwen3.5-9B \
    MODEL_NEW=weights/Qwen3.5-9B-reranked-K81920 \
    K=81920 \
    python analysis/mtp_head_longtail/v2_reranked/float_verify_dialog.py
```

---

## 5. Benchmark（接收率）

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py \
    --model weights/Qwen3.5-4B-reranked-K81920 \
    --forced-decode \
    --num-draft-tokens 4 \
    --run-all \
    --max-new-tokens 128
```

可选：显式指定 mtp_head 路径（默认自动检测模型目录下的 `mtp_lm_head.pt`）：

```bash
--mtp-head-pt weights/Qwen3.5-4B-reranked-K81920/mtp_lm_head.pt
```

**样例数字（9B K=81920，forced-decode）**：

| 场景 | 接收率 |
|------|--------|
| math_proof | 89.8% |
| english_explanation | 87.5% |
| chinese_science | 79.7% |
| **平均** | **82.6%** |

---

## 6. Spec decode 端到端测试

导出后，用 `meta.json` 做 HMONNX 推理验证：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py \
    --config work_dirs/qwen3_5_4b_xh2a_mtp_k81920/meta.json \
    --prompt "请介绍一下量子计算的基本原理"
```

---

## 7. GPTQ / autoround 量化模型支持

量化模型（如 GPTQ、autoround）通常只有 `tokenizer.json`，没有独立的 `vocab.json`。

`rerank_model_for_mtp` 自动处理此情况：若找不到 `vocab.json`，则 fallback 从 `tokenizer.json["model"]["vocab"]` 读取词表。用法与 FP16 模型完全一致，把 `--hf_model_dir` 换成量化模型目录即可。

**已验证**：`weights/Qwen3.5-9B-mode1-llm-only/` smoke 通过。

---

## 8. 高级用法

### 8.1 重新统计语料生成 hot_ids

**何时需要**：
- 业务语料分布与现有 hot_ids 偏差较大
- 使用了全新的 tokenizer（注意：4B/9B 共用同一词表，无需分别统计）

```bash
python examples/llm/qwen3_5/build_mtp_hot_vocab.py \
    --model weights/Qwen3.5-4B \
    --corpus-dir analysis/mtp_head_longtail/v2_reranked/corpus \
    --freqs-dir analysis/mtp_head_longtail/v2_reranked/freqs \
    --output-dir analysis/mtp_head_longtail/v2_reranked/selection \
    --k-list 32000,48000,81920
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--model` | 模型目录（用于加载 tokenizer） |
| `--corpus-dir` | openclaw_raw.tsv 所在目录 |
| `--freqs-dir` | token 频次文件读写目录（.counter.pt） |
| `--output-dir` | 输出 hot_ids / id_map / stats 的目录 |
| `--k-list` | 逗号分隔的 K 值列表（默认：`32000,48000,81920,82000`） |
| `--skip-public` | 跳过从 HF datasets 下载公共语料，假设 freqs/public_*.counter.pt 已存在 |
| `--rebuild-counters` | 重新从 corpus/openclaw_raw.tsv 生成 counter 文件（默认已存在则跳过） |

**输出格式**：

| 文件 | 类型 | 说明 |
|------|------|------|
| `hot_ids_{K}.pt` | `torch.int64 Tensor[K]` | 升序，存原始 old_id |
| `id_map_{K}.pt` | `dict` | `{old_to_new: int64[V]`（-1=cold token），`new_to_old: int64[K]}` |
| `selection_stats.json` | JSON | 各 K 值的覆盖率、各来源占比 |

### 8.2 直接调用 Python API

```python
from examples.llm.qwen3_5.utils.model_utils import rerank_model_for_mtp

rerank_model_for_mtp(
    original_model_dir="weights/Qwen3.5-4B",
    K=81920,
    dst_dir="weights/Qwen3.5-4B-reranked-K81920",
    force=False,
)
```

**完整函数签名**：

```python
def rerank_model_for_mtp(
    original_model_dir: str,
    K: int,
    dst_dir: str,
    hot_ids_path: Optional[str] = None,
    id_map_path: Optional[str] = None,
    force: bool = False,
) -> str:
    ...
```

参数说明：
- `original_model_dir`：原始 HF 模型目录（FP16 或量化模型均可）
- `K`：保留的热词数
- `dst_dir`：重排后模型的输出目录
- `hot_ids_path`：手动指定 hot_ids_{K}.pt 路径（默认：`analysis/mtp_head_longtail/v2_reranked/selection/hot_ids_{K}.pt`）
- `id_map_path`：手动指定 id_map_{K}.pt 路径（同上默认）
- `force`：即使目标目录已存在也强制重新生成
- 返回值：重排仓库的绝对路径

### 8.3 自定义 hot_ids 来源

若有自己统计的热词，可通过 `hot_ids_path` / `id_map_path` 显式传入，覆盖默认查找位置：

```python
rerank_model_for_mtp(
    original_model_dir="weights/Qwen3.5-4B",
    K=81920,
    dst_dir="weights/Qwen3.5-4B-reranked-custom",
    hot_ids_path="my_selection/hot_ids_81920.pt",
    id_map_path="my_selection/id_map_81920.pt",
)
```

---

## 9. 产物 layout

```
weights/{name}-reranked-K{K}/
├── config.json                  vocab_size 保持 V（embed 保留全维）
├── model-*.safetensors          embed 已 rerank；有独立 lm_head 时也 rerank
├── model.safetensors.index.json 更新 shard 映射
├── tokenizer.json               token id 已 remap
├── vocab.json                   仅当原模型有此文件时存在
├── merges.txt                   仅当原模型有此文件时存在
├── tokenizer_config.json        added_tokens_decoder 已 remap
├── mtp_lm_head.pt               [K, H] bf16，MTP 输出 head 实际使用的权重
├── id_map.pt                    {old_to_new: int64[V], new_to_old: int64[K]}
└── hot_ids.pt                   int64[K] 升序，原始 old_id
```

---

## 10. 常见问题

**Q: rerank 后推理乱码？**  
A: 一般是 tie_word_embeddings 误判或 lm_head 未重排。确认日志中 `mtp_lm_head.pt` 来源（应显示来自 `lm_head` 还是 `embed`）。

**Q: 4B 跑通，9B 乱码？**  
A: 早期版本存在 tie 硬编码 bug（误将 9B 也当作 tie=True 处理），已修复。如重现请确认本仓库版本。

**Q: 量化模型报缺少 vocab.json？**  
A: 已自动 fallback 从 tokenizer.json 内嵌 vocab 读取，无需手动转换。

**Q: K 怎么选？**  
A: 参考第 2 节，多数场景 K=81920 即可。对显存极度敏感可尝试 K=48000（覆盖率 98.4%）。

**Q: 不同尺寸模型共用同一 hot_ids 吗？**  
A: 是的，4B 和 9B 词表 bit-exact 完全相同，`analysis/.../selection/` 下的 hot_ids / id_map 直接复用。

**Q: 导出时提示 `mtp_lm_head.pt` 已存在，是否会重新生成？**  
A: 默认跳过（幂等），加 `--force-rerank` 强制重新生成。

---

## 11. 设计参考与历史

| 组件 | 路径 |
|------|------|
| 语料统计工具 | `examples/llm/qwen3_5/build_mtp_hot_vocab.py` |
| 模型重排实现 | `examples/llm/qwen3_5/utils/model_utils.py` — `rerank_model_for_mtp()` |
| 浮点验证脚本 | `analysis/mtp_head_longtail/v2_reranked/float_verify_dialog.py` |
| 详细分析报告 | `analysis/mtp_head_longtail/v2_reranked/README.md`（数据来源、K 选择依据、覆盖率分析） |
| 已有 hot_ids 数据 | `analysis/mtp_head_longtail/v2_reranked/selection/` |
| 已生成 reranked 仓库 | `analysis/mtp_head_longtail/v2_reranked/output/*-reranked-K*/` |
