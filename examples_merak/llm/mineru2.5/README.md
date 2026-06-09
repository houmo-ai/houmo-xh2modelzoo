# MinerU2.5 XH/HMONNX 示例说明

本目录包含 MinerU2.5-Pro 在 XH/HMONNX 路径上的调试、导出、推理和评估脚本。

默认使用的原始 HF 模型路径：

```bash
/data02/datasets/MinerU2.5-Pro-2604-1.2B
```

建议在仓库根目录执行命令：

```bash
cd /data01/home/chuyuan.wei/code/xh2modelzoo
python ...
```

## 脚本说明

| 文件 | 作用 |
| --- | --- |
| `debug_scripts/mineru2_5_native_generate.py` | 最小原始 HF 冒烟测试，直接用 `MinerUClient.two_step_extract` 识别 `houmo_logo.jpg`。 |
| `debug_scripts/mineru2_5_xh_generate.py` | 构建 XH PyTorch 模型和多个静态 `XHQwen2VLVisualModel`，通过静态 ViT bucket 路由接入 `MinerUClient`。 |
| `mineru2_5_xh_export_hmonnx.py` | 导出共享 LLM HMONNX、默认 visual HMONNX，以及多个静态 visual bucket HMONNX，并写出 `mineru_visual_buckets.json`。 |
| `mineru2_5_xh_hmonnx_generate.py` | 使用导出的 HMONNX 模型对单张图片进行 MinerU 推理，运行时根据 `mineru_visual_buckets.json` 切换静态 ViT。 |
| `mineru2_5_omnidocbench_hf_eval.py` | 评估原始 HF MinerU2.5-Pro 在 OmniDocBench 上的抽样结果，也支持 static ViT 模拟模式。 |
| `mineru2_5_omnidocbench_hmonnx_eval.py` | 评估 HMONNX 模型在同一组 OmniDocBench 样本上的结果，核心推理逻辑复用 `mineru2_5_xh_hmonnx_generate.py`。 |

相关配置：

```bash
configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py
configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py
```

## 静态 ViT Bucket 设计

MinerU 的 layout 阶段会把整页 resize 到固定大小，默认是 `1036x1036`。

但 content 阶段不是从这张 `1036x1036` 图上裁剪，而是把 layout 输出的归一化 bbox 映射回原始页面图片，再从原图裁剪 block。因此 content crop 的宽度可能明显超过 1036。例如 PPT 页面中常见：

```text
112x1736
168x1624
252x1680
364x2044
```

为了满足 NPU 静态图部署，当前方案使用有限个静态 visual bucket。默认 `1036x1036` visual 来自完整 VLM 导出，额外 bucket 定义在：

```bash
configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py
```

当前额外 bucket：

```python
140x392, 196x560, 280x784, 392x1036,
112x1792, 168x1792, 252x1792, 392x2044,
560x560, 1036x392
```

所有尺寸都是 28 的倍数，以满足 Qwen2-VL patch/merge 对齐。

### 路由和预处理策略

当前脚本不再使用简单的“能 fit 就 fit，否则 fallback 到 1036x1036”的策略，而是对所有 bucket 打分：

```text
score = 2.0 * aspect_cost + 0.8 * downscale_cost + 0.2 * padding_cost
```

推理时的预处理流程：

```text
PIL crop
-> Qwen smart_resize 得到 native 尺寸
-> 根据比例/缩放/padding score 选择 bucket
-> 等比缩放到 bucket 内，默认最多放大 2 倍
-> 白底居中补齐到 bucket 尺寸
-> processor
-> 校验 image_grid_thw 是否等于 bucket / patch_size
```

该逻辑不拉伸、不裁剪，只做等比缩放和白底补齐。

相关参数：

```bash
--static-vit-max-upscale 2.0
```

## 原始 HF 冒烟测试

运行：

```bash
python \
  examples_merak/llm/mineru2.5/debug_scripts/mineru2_5_native_generate.py
```

默认输入：

```bash
data/images/houmo_logo.jpg
```

该脚本直接加载：

```python
Qwen2VLForConditionalGeneration.from_pretrained(...)
AutoProcessor.from_pretrained(...)
MinerUClient(...).two_step_extract(...)
```

## XH PyTorch 静态 ViT 推理

运行：

```bash
python \
  examples_merak/llm/mineru2.5/debug_scripts/mineru2_5_xh_generate.py \
  --config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py \
  --visual-buckets-config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py \
  --image-path data/images/houmo_logo.jpg \
  --max-new-tokens 128 \
  --no-tqdm
```

该脚本会：

1. 根据 LLM config 构建一份共享 XH LLM/VLM。
2. 根据 visual bucket config 构建多个 `XHQwen2VLVisualModel`。
3. 包装成 `MinerUClient` 需要的 HF 风格 `generate()` 接口。
4. 对 layout/content 阶段的图片按 ratio score 路由到静态 ViT bucket。

常用参数：

```bash
--eval-type wrap
--layout-image-size 1036
--static-vit-max-upscale 2.0
--image-analysis
```

## HMONNX 导出

导出共享 LLM HMONNX 和所有静态 visual bucket：

```bash
python \
  examples_merak/llm/mineru2.5/mineru2_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py \
  --visual-buckets-config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py \
  --force
```

导出目录位于：

```bash
work_dirs/mineru2_5_llm_1_2b_xh2a_4k/
```

当前已经验证过的最新导出目录：

```bash
work_dirs/mineru2_5_llm_1_2b_xh2a_4k/hmquant_xh2_mineru2_5_pro_1_2b_w8a8_256_4k_1036x1036_20260604
```

关键文件：

```bash
golden_meta_info.json
mineru_visual_buckets.json
prefill/*.onnx
decode/*.onnx
visual/*.onnx
visual_buckets/<HxW>/*.onnx
```

`mineru_visual_buckets.json` 是 MinerU 专用 manifest，用于记录每个静态 visual bucket 的尺寸和 HMONNX 路径，不修改通用 HMONNX metadata schema。

常用参数：

```bash
--skip-existing-visual-buckets
--context-length 4096
--prefill-chunk-length 256
--quant-type w8a8h1_sefp
```

如果修改了 `mineru2_5_visual_buckets_1_2b_xh2a.py`，需要重新导出 HMONNX 和 `mineru_visual_buckets.json`。

## HMONNX 单图推理

运行：

```bash
EXPORT_DIR=work_dirs/mineru2_5_llm_1_2b_xh2a_4k/hmquant_xh2_mineru2_5_pro_1_2b_w8a8_256_4k_1036x1036_20260604

python \
  examples_merak/llm/mineru2.5/mineru2_5_xh_hmonnx_generate.py \
  --config ${EXPORT_DIR}/golden_meta_info.json \
  --visual-buckets-manifest ${EXPORT_DIR}/mineru_visual_buckets.json \
  --image-path data/images/houmo_logo.jpg \
  --max-new-tokens 128 \
  --no-tqdm
```

该脚本会：

1. 加载共享 HMONNX LLM、默认 visual、额外 bucket visual。
2. 使用和 HF static 模拟一致的 ratio bucket 预处理。
3. 每次 `generate()` 前根据 `image_grid_thw` 切换 `hmonnx_model.visual`。
4. 使用 chunk prefill 支持超过单个 prefill chunk 的输入，只要总长度不超过 context length。

## OmniDocBench 数据集

评估脚本使用 OmniDocBench：

```bash
opendatalab/OmniDocBench
```

本地目录：

```bash
work_dirs/mineru2_5_omnidocbench_data
```

标注文件：

```bash
work_dirs/mineru2_5_omnidocbench_data/OmniDocBench.json
```

当前工作区中的 `OmniDocBench.json` 约 41 MB。

下载标注文件：

```bash
huggingface-cli download \
  opendatalab/OmniDocBench \
  OmniDocBench.json \
  --repo-type dataset \
  --local-dir work_dirs/mineru2_5_omnidocbench_data
```

图片下载说明：

- `mineru2_5_omnidocbench_hf_eval.py` 默认会通过 `hf_hub_download` 下载缺失的抽样图片。
- `mineru2_5_omnidocbench_hmonnx_eval.py` 不负责下载图片，因此建议先跑 HF eval，或确保样本图片已经存在。
- 如果图片已存在，可以给 HF eval 加 `--no-download-missing`。

## OmniDocBench 评估方法

这里的评估是 MinerU 抽取链路的回归测试，不是 OmniDocBench 官方指标。

流程：

```text
image
-> MinerUClient.two_step_extract
-> 递归收集 result 中的 text/html/latex/content
-> 拼成页面级 prediction
-> 和 OmniDocBench 标注中的 text/html/latex 字段比较
```

指标：

- `edit_similarity`：归一化后用 `difflib.SequenceMatcher` 计算相似度。
- `char_precision`、`char_recall`、`char_f1`：归一化后的字符 multiset overlap。
- `avg_time`：每页 `two_step_extract` 的平均耗时。

默认参与评估的类别包括：

```text
title, text_block, list_group, reference, figure_caption,
table, table_caption, equation_isolated, equation_semantic,
header, footer, page_number, code_txt, ...
```

## 固定 10 条回归样本

当前 HMONNX 回归使用以下 manifest 中的前 10 条：

```bash
work_dirs/mineru2_5_omnidocbench_hmonnx_extract/omnidocbench_hmonnx_sample_manifest.json
```

具体样本：

| rank | json_index | image | page | size | source | layout |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | 1286 | `PPT_1001115_eng_page_003.png` | 3 | 1500x2000 | PPT2PDF | single_column |
| 1 | 1357 | `PPT_8076_MEYER_Chapter_2_-_Language_Change_page_021.png` | 21 | 1500x2000 | PPT2PDF | single_column |
| 2 | 1342 | `PPT_Catalysis.ppt_page_016.png` | 16 | 1500x2000 | PPT2PDF | single_column |
| 3 | 1353 | `PPT_MMAT5390Lecture1_page_024.png` | 24 | 1500x2000 | PPT2PDF | single_column |
| 4 | 1338 | `PPT_all655920_page_003.png` | 3 | 1500x2000 | PPT2PDF | single_column |
| 5 | 1292 | `PPT_english-studies-s6-on-the-road-resource-3_page_002.png` | 2 | 1500x2000 | PPT2PDF | other_layout |
| 6 | 1319 | `PPT_lecture1_page_022.png` | 22 | 1500x2667 | PPT2PDF | single_column |
| 7 | 1436 | `book_en_A.Course.in.Abstract.Harmonic.Analysis.-.Gerald.B.Folland.0849384907_page_119.png` | 119 | 1731x1039 | book | single_column |
| 8 | 1444 | `book_en_国外数学教材-数论-Melvyn B. Nathanson—Elementary Methods in Number Theory_0092.png` | 92 | 1734x1253 | book | single_column |
| 9 | 1456 | `book_en_搬书匠-3246-Electronics Cookbook-2017-英文版_page_193.png` | 193 | 1838x1400 | book | single_column |

## 原始 HF 评估

评估原始 HF dynamic ViT：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hf_eval.py \
  --sample-manifest work_dirs/mineru2_5_omnidocbench_hmonnx_extract/omnidocbench_hmonnx_sample_manifest.json \
  --sample-size 10 \
  --output-dir work_dirs/mineru2_5_omnidocbench_hf_hmonnx10_float16 \
  --dtype float16 \
  --no-tqdm
```

评估原始 HF 模型 + static ViT 模拟：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hf_eval.py \
  --sample-manifest work_dirs/mineru2_5_omnidocbench_hmonnx_extract/omnidocbench_hmonnx_sample_manifest.json \
  --sample-size 10 \
  --output-dir work_dirs/mineru2_5_omnidocbench_hf_staticvit_ratio_hmonnx10_float16_gpu0 \
  --dtype float16 \
  --simulate-static-vit \
  --static-vit-max-upscale 2.0 \
  --no-download-missing \
  --no-tqdm
```

## HMONNX 评估

评估最新 HMONNX：

```bash
EXPORT_DIR=work_dirs/mineru2_5_llm_1_2b_xh2a_4k/hmquant_xh2_mineru2_5_pro_1_2b_w8a8_256_4k_1036x1036_20260604

CUDA_VISIBLE_DEVICES=0 python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hmonnx_eval.py \
  --config ${EXPORT_DIR}/golden_meta_info.json \
  --visual-buckets-manifest ${EXPORT_DIR}/mineru_visual_buckets.json \
  --hf-sample-manifest work_dirs/mineru2_5_omnidocbench_hmonnx_extract/omnidocbench_hmonnx_sample_manifest.json \
  --sample-count 10 \
  --output-dir work_dirs/mineru2_5_omnidocbench_hmonnx_ratio_20260604_extract \
  --no-tqdm
```

输出文件：

```bash
omnidocbench_hmonnx_sample_manifest.json
omnidocbench_hmonnx_results.json
omnidocbench_hmonnx_scores.json
```

## 最新精度结果

最新已验证的 HMONNX 导出目录：

```bash
work_dirs/mineru2_5_llm_1_2b_xh2a_4k/hmquant_xh2_mineru2_5_pro_1_2b_w8a8_256_4k_1036x1036_20260604
```

固定 10 条样本上的结果：

| run | edit_similarity | char_precision | char_recall | char_f1 | avg_time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 HF dynamic fp16 | 0.6272 | 0.9383 | 0.9351 | 0.9344 | 13.91s |
| 原始 HF + static ViT ratio 模拟 | 0.6340 | 0.9390 | 0.9230 | 0.9267 | 14.64s |
| 最新 HMONNX ratio buckets | 0.6354 | 0.9401 | 0.9230 | 0.9272 | 183.57s |

逐条 `char_f1`：

| rank | HF static ratio | HMONNX ratio |
| ---: | ---: | ---: |
| 0 | 0.9695 | 0.9695 |
| 1 | 0.9688 | 0.9688 |
| 2 | 0.7328 | 0.7328 |
| 3 | 0.9631 | 0.9631 |
| 4 | 0.9961 | 0.9961 |
| 5 | 1.0000 | 1.0000 |
| 6 | 0.8039 | 0.8039 |
| 7 | 0.9126 | 0.9126 |
| 8 | 0.9251 | 0.9306 |
| 9 | 0.9949 | 0.9949 |

当前 HMONNX 结果已经基本对齐 HF static ratio 模拟。HMONNX 和 HF dynamic 之间剩余差异主要来自静态 ViT bucket 化和量化/运行时差异，而不是长横条 fallback 问题。

## 复现注意事项

- 建议显式设置 `CUDA_VISIBLE_DEVICES=0` 或其他空闲 GPU，避免 `device_map=auto` 把模型放到繁忙 GPU。
- 当前 MinerU 推理强制/默认 batch size 为 1，避免一次 `generate()` 中混入不同 visual bucket。
- `hmonnx_generate` 和 `hmonnx_eval` 只能路由到 `mineru_visual_buckets.json` 中已经导出的 bucket。
- 修改 visual bucket config 后，需要重新运行 `mineru2_5_xh_export_hmonnx.py`。
- 静态 bucket 预处理逻辑只在 MinerU2.5 示例脚本层实现，不修改通用 Qwen2-VL 模型代码。
