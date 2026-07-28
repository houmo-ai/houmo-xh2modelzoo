# Qwen3-VL-Embedding

该目录提供 Qwen3-VL-Embedding 的 Merak Workflow、HMONNX Demo 和
原生模型 Demo。模型同时支持文本、图片以及图文组合 embedding。

## 模型结构

导出包含两张 HMONNX 图：

- `vision`：将图片转换为 visual embeddings 和三层 DeepStack 特征。
- `prefill`：输出完整序列的最后一层 hidden states。

Pooling 和 L2 normalize 保持在 Python 侧：

```text
last_hidden_state -> last valid token -> L2 normalize -> embedding
```

2B 模型的 embedding 维度为 2048，8B 模型为 4096。

## 环境

在可用的 Python 或 Conda 环境中安装本仓库及通用依赖：

```bash
pip install -v -e . --no-build-isolation
pip install "transformers>=4.57.3" qwen-vl-utils pillow
```

模型导出和 HMONNX 推理还需要匹配版本的 `xhquant`。
Qwen3-VL-Embedding 依赖 Transformers 4.57.3 或兼容版本。

## 导出与 Golden

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/qwen3_vl_embedding/qwen3_vl_embedding_workflow.py \
  --model-dir <model_dir> \
  --export-output-dir work_dirs/qwen3_vl_embedding_export \
  --device cuda:0 \
  --dump-golden \
  --prompt "A dog playing in the park" \
  --image <image_file> \
  --overwrite
```

默认配置为 2B、XH2a、W8A8、2048 context、256 prefill 和
448×448 vision 输入。可以通过 `--context-length`、
`--prefill-length`、`--image-size` 和 `--quant-type` 覆盖。
8B 使用：

```text
configs_merak/workflows/xh2a/llm_models/qwen3_vl_embedding/8b/
qwen3_vl_embedding_8b_xh2a_w8a8.yaml
```

导出根目录的 `export_meta_info.json` 是唯一产物入口。它同时包含
模型 metadata，并索引 Vision 和 Prefill HMONNX 文件。导出完成后
不会保留单独的 `golden_meta_info.json` 和 Vision 中间 ONNX。

指定 `--dump-golden` 后，会分别生成
`visual/golden/step_0/` 和 `prefill/golden/step_0/`。每次生成前会
清理已有 golden；不会额外生成 golden metadata JSON。Golden 包含
图输入、输出和中间节点数据，占用空间可能明显大于 HMONNX 本身。
`--prompt` 和 `--image` 可以指定生成 golden 使用的文本和图片；
未指定时使用默认 prompt 和生成的灰色图片。

## HMONNX Demo

文本 embedding：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/qwen3_vl_embedding/hmonnx_demo.py \
  --export-dir work_dirs/qwen3_vl_embedding_export \
  --text "A dog playing in the park" \
         "A cat sitting on a chair" \
  --device cuda:0
```

增加图片 embedding：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/qwen3_vl_embedding/hmonnx_demo.py \
  --export-dir work_dirs/qwen3_vl_embedding_export \
  --image <image_file> \
  --device cuda:0
```

输入经过 Processor 后不能超过导出的 `prefill_chunk_length`。
超长输入会明确报错，不会静默截断。

## 原生模型 Demo

```bash
python examples_merak/llm/qwen3_vl_embedding/native_demo.py \
  --model-dir <model_dir>
```

原生和 HMONNX Demo 都输出归一化 embedding 及余弦相似度矩阵，
便于在相同输入下比较迁移前后的结果。
原生 Demo 使用模型目录自带的
`scripts/qwen3_vl_embedding.py`，确保 checkpoint 按 embedding
模型结构正确加载。

直接比较同一批文本的原生与 HMONNX embedding：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/qwen3_vl_embedding/compare_native_hmonnx.py \
  --model-dir <model_dir> \
  --export-dir work_dirs/qwen3_vl_embedding_export \
  --image <image_file> \
  --device cuda:0
```

`--image` 可选；提供后会按导出 vision 尺寸统一缩放，再一起比较。
脚本输出逐条 embedding 的余弦相似度，以及两侧的相似度矩阵。
W8A8 HMONNX 与原生浮点模型存在量化误差，不要求逐元素完全相等。

## Flickr30K 评测

原生模型：

```bash
python examples_merak/llm/qwen3_vl_embedding/eval_flickr30k.py \
  --backend native \
  --model-dir <model_dir> \
  --dataset-dir <flickr30k_dir>
```

HMONNX：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/qwen3_vl_embedding/eval_flickr30k.py \
  --backend hmonnx \
  --export-dir work_dirs/qwen3_vl_embedding_export \
  --dataset-dir <flickr30k_dir> \
  --device cuda:0
```

评测同时输出 text-to-image 和 image-to-text 的 Recall@1/5/10、
MRR@10 与 nDCG@10。可以用 `--max-images` 做小样本验证。

原生固定方形 448 消融：

```bash
python examples_merak/llm/qwen3_vl_embedding/eval_flickr30k.py \
  --backend native \
  --model-dir <model_dir> \
  --dataset-dir <flickr30k_dir> \
  --native-image-size 448
```

多 GPU HMONNX data parallel 评测：

```bash
PYTHONPATH=$PWD python \
  examples_merak/llm/qwen3_vl_embedding/eval_flickr30k_multigpu.py \
  --export-dir work_dirs/qwen3_vl_embedding_export \
  --dataset-dir <flickr30k_dir> \
  --gpus 0,1
```
