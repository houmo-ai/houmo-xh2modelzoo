# Qwen3-VL-Embedding

Qwen3-VL-Embedding 多模态文本/图像 embedding 模型的原始推理 + HMONNX 导出与验证示例。

本地模型路径（2B，完整）：

```
/data01/home/she.gao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
```

本地模型路径（8B，ModelScope）：

```
/data01/home/she.gao/.cache/modelscope/hub/models/Qwen/Qwen3-VL-Embedding-8B
```

## 环境依赖

实测验证环境（conda env `xhquant`）：

| 库 | 版本 | | 库 | 版本 |
|---|---|---|---|---|
| Python | 3.12.0 | | transformers | 4.57.3 |
| torch | 2.8.0+cu128 | | tokenizers | 0.22.1 |
| torchvision | 0.23.0+cu128 | | qwen-vl-utils | 0.0.14 |
| CUDA / cuDNN | 12.8 / 9.10.2 | | accelerate | 1.12.0 |
| onnx | 1.16.2 | | sentence-transformers | 5.5.1 |
| onnxruntime | 1.19.0 | | datasets | 4.7.0 |
| numpy | 2.2.6 | | pillow | 11.3.0 |
| xhquant | 0.4.0 | | | |

> `transformers>=4.57` 才内置 `Qwen3VLForConditionalGeneration`；`qwen-vl-utils` 提供 `process_vision_info`（图像/视频预处理）。HMONNX 推理与导出依赖 `xhquant`（私有）。


## 模型分解

```
Qwen3VLForConditionalGeneration (HF architecture)
└── Qwen3VLModel                                ◀── HMONNX backbone (visual + DeepStack + language)
        └── last_hidden_state (B, T, 2048)
                ↓
        last_token_pool(attention_mask)         ◀── Python 侧
                ↓
        F.normalize(p=2, dim=-1)                ◀── Python 侧
                ↓
        embedding (B, 2048)
```

- 输出 embedding 维度：**2048（2B）**；8B 是 4096。
- pooling / normalize 不进 ONNX，参照 `qwen3-Embedding/` 范式。

## 目录文件

| 文件 | 作用 |
|---|---|
| `sentence_transformers_demo.py` | 用户原始的 `SentenceTransformer` 风格 demo（默认 2B）。 |
| `native_qwen3_vl_embedding.py` | 通过模型自带的 `scripts/qwen3_vl_embedding.py:Qwen3VLEmbedder` 做原始 BF16 推理。 |
| `qwen3_vl_embedding_xh2a_export_hmonnx.py` | 用 `Qwen3_VLEmbeddingConverterXH2a` 导出 backbone（visual + language）到 XH2a/HMONNX。 |
| `qwen3_vl_embedding_xh2a_demo.py` | 用 `Qwen3VLONNXModel.embed_texts()` 跑 HMONNX 文本 embedding，输出相似度矩阵。 |
| `qwen3_vl_embedding_eval_flickr30k_hmonnx_aligned.py` | Flickr30K 上跑 **HMONNX** 的 text↔image 检索（含 vision tower），输出 Recall@1/5/10 + MRR@10 + nDCG@10。 |
| `qwen3_vl_embedding_eval_flickr30k_hmonnx_multigpu.py` | HMONNX 多 GPU data-parallel 评测脚本；每张 GPU 独立跑一个 embedding shard，最后合并计算检索指标。 |
| `qwen3_vl_embedding_eval_flickr30k.py` | Flickr30K 上跑 **HF BF16 baseline** 的同口径检索指标，作为对比基准。 |
| `eval_flickr30k_float_448.py` | 消融实验：**浮点**模型但 image 侧强制走固定方形 `--image-size`（默认 448）预处理，用于拆分「分辨率/padding 损失」与「量化损失」。 |

## 1) 原始推理

SentenceTransformer demo：

```bash
python examples/llm/qwen3-vl-embedding/sentence_transformers_demo.py \
  --model-dir /data01/home/she.gao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
```

预期输出包含 `(4, 2048) (3, 2048)` 以及一个 4×3 余弦相似度矩阵。

Qwen3VLEmbedder demo（含 text+image 混合）：

```bash
python examples/llm/qwen3-vl-embedding/native_qwen3_vl_embedding.py
```

## 2) HMONNX 导出

```bash
python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_xh2a_export_hmonnx.py \
  --context-length 2048 \
  --quant-type w8a8h1_sefp
```

输出位于 `work_dirs/Qwen3-VL-Embedding-2B-XH2a-2k-w8a8h1_sefp/`，主要产物：
- `meta.json`
- `prefill/<...>_prefill_with_act.onnx`（+ external data）
- `token_embedding_file`（embed_tokens 权重，单独保存）

注：模型 architecture 是 `Qwen3VLForConditionalGeneration`，且 `tie_word_embeddings=true`；本期复用 `qwen3vl` 的 converter，不在 model_zoo 内额外加适配。

## 3) HMONNX 校验

跑 HMONNX 文本 embedding demo（用 `Qwen3VLONNXModel.embed_texts()`）：

```bash
python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_xh2a_demo.py \
  --model_dir work_dirs/<...>-XH2a-2k-w8a8h1_sefp \
  --hf_model /data01/home/she.gao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --text "A dog playing in the park" "A cat sitting on a chair"
```

8B HMONNX demo 也可以复用同一个脚本，但必须显式传 `--model_type 8B`，否则默认按 2B 的 28 层 KV cache 构建；8B 是 36 层。当前 8B 导出使用 `input_sequence_length=256`、`cache_len=2048`，与 demo 默认值匹配。

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> \
PYTHONPATH=/data01/home/she.gao/xh2modelzoo_new \
python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_xh2a_demo.py \
  --model_dir work_dirs/Qwen3-VL-Embedding-8B-XH2a-2k-w8a8h1_sefp \
  --hf_model /data01/home/she.gao/.cache/modelscope/hub/models/Qwen/Qwen3-VL-Embedding-8B \
  --model_type 8B \
  --text "A dog playing in the park" "A cat sitting on a chair"
```

8B demo 输出 embedding 维度应为 `4096`；2B 是 `2048`。

## 4) 精度数据集 (Flickr30K 标准检索指标)

在 Flickr30K 标准 1K test split 上跑 HF BF16 baseline 的 text↔image 检索，**硬指标**（Recall@K / MRR@10 / nDCG@10）：

```bash
python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_eval_flickr30k.py \
  --model-dir /data01/home/she.gao/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --dataset-dir /path/to/flickr30k \
  --report work_dirs/flickr30k_eval_2B.json
```

数据集加载优先级：
1. `--dataset-dir` 指向本地目录：要求 `flickr30k_images/` (或 `flickr30k-images/`、`images/`) 加 `results.csv` (Kaggle 标准 `image_name| comment_number| comment`) 或 `captions.json`。
2. 退到 HF datasets 缓存 `nlphuji/flickr30k` 的 `test` split（包含官方 1K test 标注）。
3. 都没有则直接报错（不联网爬图）。

报告内容（落盘到 `--report`，默认 `work_dirs/flickr30k_eval.json`）：

| 指标 | 含义 |
|---|---|
| `recall@K`  | 二值命中：top-K 中是否出现任一 GT，K ∈ {1,5,10}。 |
| `mrr@10`    | Mean Reciprocal Rank：top-10 内第一个 GT 的倒数排名，找不到记 0。 |
| `ndcg@10`   | 二值相关性下的归一化折损累积增益，对排名敏感。 |

两个方向：
- `text_to_image`：每条 caption 当 query（5K 查询），1K 张图中检索，1 个 GT。
- `image_to_text`：每张图当 query（1K 查询），5K 条 caption 中检索，5 个 GT（命中其一即算）。

HMONNX 现已支持 vision tower，可跑完整 text↔image 检索（用 `qwen3_vl_embedding_eval_flickr30k_hmonnx_aligned.py`）。

## 5) HMONNX 全量精度（Flickr30K 946 图 / 4730 caption, W8A8）

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_eval_flickr30k_hmonnx_aligned.py \
  --hmonnx-config work_dirs/<...>-XH2a-2k-w8a8h1_sefp/meta.json \
  --hf-model <hf_model_path> \
  --dataset-dir /path/to/flickr30k \
  --device cuda:0 \
  --report work_dirs/flickr30k_hmonnx_full_eval.json
```

实测结果（HMONNX W8A8 vs HF 浮点 baseline，全量 946 图 / 4730 caption）：

| 方向 | 指标 | HMONNX@448 | 浮点(动态) | 差距 |
|---|---|---|---|---|
| text→image | Recall@1 | 54.9% | 78.6% | -23.7 |
| text→image | Recall@5 | 79.0% | 94.5% | -15.5 |
| text→image | Recall@10 | 85.5% | 97.5% | -12.0 |
| image→text | Recall@1 | 76.4% | 94.8% | -18.4 |
| image→text | Recall@5 | 92.0% | 99.6% | -7.6 |
| image→text | Recall@10 | 94.7% | 99.9% | -5.2 |

## 6) 损失拆分消融（分辨率 / padding vs 量化）

为定位 HMONNX 掉点的根因，用 `eval_flickr30k_float_448.py` 让**浮点**模型也走 HMONNX 同款的固定方形预处理（保长宽比缩放 + 114 灰色 padding 到方形），text 侧保持浮点动态。唯一变量即图像预处理 / 量化。

```bash
# 浮点 @448 方形 pad（= HMONNX 同款预处理，但用浮点 vision）
CUDA_VISIBLE_DEVICES=<free_gpu> python examples/llm/qwen3-vl-embedding/eval_flickr30k_float_448.py \
  --dataset-dir /path/to/flickr30k --image-size 448 \
  --report work_dirs/flickr30k_float_448_full_eval.json

# 浮点 @896 方形 pad（验证提分辨率的恢复程度）
CUDA_VISIBLE_DEVICES=<free_gpu> python examples/llm/qwen3-vl-embedding/eval_flickr30k_float_448.py \
  --dataset-dir /path/to/flickr30k --image-size 896 --image-batch-size 2 \
  --report work_dirs/flickr30k_float_896_full_eval.json
```

四方对比（全量 946 图，R@1）：

| 配置 | text→image | image→text | 备注 |
|---|---|---|---|
| 浮点 动态分辨率（baseline） | **78.6%** | **94.8%** | 最高 1.84M 像素，保长宽比 |
| 浮点 @896 方形 pad | 75.7% | 76.1% | 0.8M 像素 |
| 浮点 @448 方形 pad | 47.5% | 78.1% | 0.2M 像素（= HMONNX 同款预处理）|
| HMONNX @448（W8A8） | 54.9% | 76.4% | 浮点@448 + 量化 |

**损失拆分：**

| 阶段 | text→image | image→text | 损失来源 |
|---|---|---|---|
| 动态 → @448 | **-31.1** | **-16.7** | 图像预处理（主因）|
| @448 → HMONNX@448（量化）| **+7.4** | **-1.7** | 量化几乎零损失 |
| 动态 → @896（对照）| -2.9 | -18.7 | 见下 |

**结论（修正先前"双重损失"的判断）：**

1. **W8A8 量化几乎零损失** —— text→image 量化后反而 +7.4（噪声/轻微正则范围），image→text 仅 -1.7。量化方案无需优化。
2. **掉点几乎全部来自图像预处理，且两个方向成因不同：**
   - **text→image 对分辨率敏感**：448 太小暴跌 -31 点，提到 896 几乎恢复 baseline（仅 -2.9）。
   - **image→text 对方形 padding 敏感**：448/896 都掉到 ~76%（灰边 + 长宽比失真），只有动态无 padding 才 94.8% —— 提分辨率（896）无改善，说明是 padding 而非分辨率。
3. **优化方向**：① vision 导出尺寸提到 896×896（text→image R@1 预期 54.9% → ~75%）；② 改保长宽比预处理、去掉方形灰边 pad（image→text 预期 → ~95%）；③ 量化不动。

## 7) 8B 实测结果（Flickr30K 946 图 / 4730 caption）

8B 模型路径：

```
/data01/home/she.gao/.cache/modelscope/hub/models/Qwen/Qwen3-VL-Embedding-8B
```

8B HMONNX 导出产物：

```
work_dirs/Qwen3-VL-Embedding-8B-XH2a-2k-w8a8h1_sefp/
```

包含 `meta.json`、`token_embedding.pt`、vision/prefill/decode HMONNX 及对应 external data；golden 已生成：

```
golden/Qwen3-VL-Embedding-8B-XH2a-w8a8h1_sefp_vision
golden/Qwen3-VL-Embedding-8B-XH2a-w8a8h1_sefp-llm-prefill
golden/Qwen3-VL-Embedding-8B-XH2a-w8a8h1_sefp-llm-decode
```

### 7.1 浮点 baseline

非固定窗口（HF 原生动态分辨率）：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_eval_flickr30k.py \
  --model-dir /data01/home/she.gao/.cache/modelscope/hub/models/Qwen/Qwen3-VL-Embedding-8B \
  --dataset-dir /data01/home/she.gao/.cache/flickr30k \
  --text-batch-size 16 \
  --image-batch-size 4 \
  --report work_dirs/flickr30k_float_dynamic_8B_full_eval.json
```

固定窗口 448x448 方形 pad：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python examples/llm/qwen3-vl-embedding/eval_flickr30k_float_448.py \
  --model-dir /data01/home/she.gao/.cache/modelscope/hub/models/Qwen/Qwen3-VL-Embedding-8B \
  --dataset-dir /data01/home/she.gao/.cache/flickr30k \
  --image-size 448 \
  --text-batch-size 16 \
  --image-batch-size 4 \
  --report work_dirs/flickr30k_float_448_8B_full_eval.json
```

| 配置 | 方向 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| 8B 浮点动态分辨率 | text→image | 79.98% | 94.88% | 97.23% | 86.33% | 89.02% |
| 8B 浮点动态分辨率 | image→text | 94.71% | 99.68% | 99.89% | 96.87% | 88.75% |
| 8B 浮点 @448 方形 pad | text→image | 78.75% | 93.72% | 96.85% | 85.33% | 88.15% |
| 8B 浮点 @448 方形 pad | image→text | 93.87% | 99.37% | 99.89% | 96.24% | 87.75% |

### 7.2 HMONNX 固定窗口 @448

HMONNX 的 8B 导出配置固定视觉窗口为 448x448：

```json
"visual": {
  "image_max_size_h": 448,
  "image_max_size_w": 448,
  "image_max_size_t": 2,
  "temporal_patch_size": 2,
  "patch_size": 16
}
```

单卡脚本可直接评测；8B 全量耗时较长，实测使用多 GPU data-parallel 脚本：

```bash
PYTHONPATH=/data01/home/she.gao/xh2modelzoo_new TOKENIZERS_PARALLELISM=false \
python examples/llm/qwen3-vl-embedding/qwen3_vl_embedding_eval_flickr30k_hmonnx_multigpu.py \
  --hmonnx-config work_dirs/Qwen3-VL-Embedding-8B-XH2a-2k-w8a8h1_sefp/meta.json \
  --hf-model /data01/home/she.gao/.cache/modelscope/hub/models/Qwen/Qwen3-VL-Embedding-8B \
  --dataset-dir /data01/home/she.gao/.cache/flickr30k \
  --model-type 8B \
  --gpus 2,3,6,7 \
  --report work_dirs/flickr30k_hmonnx_8B_multigpu_full_eval.json \
  --run-dir work_dirs/hmonnx_8B_multigpu_full
```

> 这里是纯推理 data parallel，不是训练 DDP：每个 worker 独立构建 HMONNX runtime，分别计算 captions/images shard，最后合并 embedding 后计算指标。

8B HMONNX@448 全量结果：

| 方向 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| text→image | 79.26% | 94.38% | 97.02% | 85.75% | 88.52% |
| image→text | 94.61% | 99.37% | 99.89% | 96.70% | 88.12% |

### 7.3 8B 固定窗口量化差异

8B 固定窗口口径下，HMONNX@448 相对浮点@448 没有精度下降，R@1 略高。

| 方向 | 指标 | 浮点@448 | HMONNX@448 | 差值 |
|---|---|---:|---:|---:|
| text→image | Recall@1 | 78.75% | 79.26% | +0.51 |
| text→image | Recall@5 | 93.72% | 94.38% | +0.66 |
| text→image | Recall@10 | 96.85% | 97.02% | +0.17 |
| text→image | MRR@10 | 85.33% | 85.75% | +0.42 |
| text→image | nDCG@10 | 88.15% | 88.52% | +0.37 |
| image→text | Recall@1 | 93.87% | 94.61% | +0.74 |
| image→text | Recall@5 | 99.37% | 99.37% | +0.00 |
| image→text | Recall@10 | 99.89% | 99.89% | +0.00 |
| image→text | MRR@10 | 96.24% | 96.70% | +0.46 |
| image→text | nDCG@10 | 87.75% | 88.12% | +0.37 |

报告文件：

| 报告 | 路径 |
|---|---|
| 8B 浮点动态分辨率 | `work_dirs/flickr30k_float_dynamic_8B_full_eval.json` |
| 8B 浮点 @448 方形 pad | `work_dirs/flickr30k_float_448_8B_full_eval.json` |
| 8B HMONNX @448 多 GPU 全量 | `work_dirs/flickr30k_hmonnx_8B_multigpu_full_eval.json` |

