# MinerU2.5 XH/HMONNX 使用说明

本目录提供 MinerU2.5-Pro 的原始 HF 推理、XH/HMONNX 导出、HMONNX 推理，以及 OmniDocBench 回归评估脚本。

推荐优先使用新的 workflow 接口：

```bash
python examples_merak/llm/mineru2.5/mineru2_5_workflow.py
```

该入口封装了量化和 HMONNX 导出流程，适合作为后续部署流程的主入口。其它脚本主要用于单步调试、对齐验证和评估。

默认原始 HF 模型路径：

```bash
/data02/datasets/MinerU2.5-Pro-2604-1.2B
```

建议在仓库根目录执行命令：

```bash
cd /data01/home/chuyuan.wei/code/xh2modelzoo
conda run -n xh2modelzoo python ...
```

## 脚本总览

| 文件 | 作用 | 典型用法 |
| --- | --- | --- |
| `mineru2_5_workflow.py` | 推荐入口；调用 `XHMinerU25HMONNXWorkflow` 完成量化和 HMONNX 导出。 | 端到端生成可部署 HMONNX 产物。 |
| `mineru2_5_xh_export_hmonnx.py` | 手动导出共享 LLM HMONNX、默认 visual HMONNX、额外静态 ViT bucket，并生成 `mineru_visual_buckets.json`。 | 需要细粒度控制导出参数时使用。 |
| `mineru2_5_xh_hmonnx_generate.py` | 使用 HMONNX 模型对单张图片执行 MinerU 两阶段识别。 | HMONNX 单图冒烟测试。 |
| `mineru2_5_omnidocbench_hf_eval.py` | 原始 HF 模型的 OmniDocBench 抽样评估；也支持 HF + static ViT 模拟。 | 建立 HF dynamic/static 精度基线。 |
| `mineru2_5_omnidocbench_hmonnx_eval.py` | HMONNX 模型的 OmniDocBench 抽样评估，推理逻辑复用 `mineru2_5_xh_hmonnx_generate.py`。 | 验证 HMONNX 与 HF/static baseline 的一致性。 |
| `mineru2_5_hf_static_vit_generate.py` | 原始 HF 模型单图推理，但在 processor 前模拟静态 ViT bucket。 | 快速验证静态 ViT 路由和预处理。 |
| `static_vit_utils.py` | 公共静态 ViT bucket 路由和图片预处理逻辑。 | 被 HF/XH/HMONNX 脚本复用，不建议单独运行。 |
| `debug_scripts/mineru2_5_native_generate.py` | 最小原始 HF 单图冒烟测试，不启用静态 ViT。 | 检查原生 MinerUClient 是否可用。 |
| `debug_scripts/mineru2_5_xh_generate.py` | XH PyTorch 模型 + 多静态 `XHQwen2VLVisualModel` 的单图调试入口。 | HMONNX 导出前验证 XH PyTorch 路径。 |

相关配置：

```bash
configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py
configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py
configs_merak/workflows/xh2a/llm_models/mineru2_5/mineru2_5_pro_xh2a_4k.yaml
```

## 静态 ViT 策略

MinerU2.5 的识别流程分两阶段：

1. layout 阶段：整页图像固定输入，当前使用 `1036x1036`。
2. content 阶段：根据 layout bbox 回到原图裁剪 block，crop 尺寸会随文本块、表格、公式、标题变化。

为了满足 NPU 静态图部署，content 阶段不再使用任意动态 ViT 尺寸，而是在有限个静态 bucket 中路由。

当前推荐组合：

```text
layout bucket: 1036x1036
content buckets: 140x392, 168x1792, 392x2044, 560x560, 1036x392
routing: fit_padding
content fallback to 1036x1036: disabled by default
batch size: 1
```

`ratio` 策略已经禁用。传入 `--static-vit-score-mode ratio` 会直接报错，因为 100 条回归显示该策略在缩减 bucket 后精度不稳定。

### fit_padding 路由

对原始 crop `h x w` 和候选 bucket `H x W`：

```text
s_fit = min(W / w, H / h)
s = min(s_fit, s_up_max)
new_h = round(h * s)
new_w = round(w * s)
pad_pixels = H * W - new_h * new_w
```

默认参数：

```text
s_up_max = 2.5
alpha_down = 10.0
beta_up = 1.0
gamma_pad = 3.0
ref_area = 448 * 448
```

评分函数：

```text
downscale_penalty = max(0.0, 1.0 / s - 1.0) ** 2
upscale_penalty = max(0.0, s - 1.0) ** 2
padding_penalty = pad_pixels / ref_area

score = (
    alpha_down * downscale_penalty
    + beta_up * upscale_penalty
    + gamma_pad * padding_penalty
)
```

预处理流程：

```text
原始 PIL crop
-> 直接基于原始 h/w 对 bucket 打分
-> 选择 score 最小的 bucket
-> 等比 resize 到 bucket 内
-> 白底居中补齐到 bucket 尺寸
-> processor
-> 校验 image_grid_thw == bucket / patch_size
```

该逻辑不做非等比拉伸，不裁剪内容。

## 推荐入口：workflow

运行：

```bash
conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_workflow.py
```

该脚本会：

1. 加载 `XHMinerU25HMONNXWorkflow`。
2. 基于 HF 模型和 workflow yaml 执行量化。
3. 基于量化结果导出 HMONNX。
4. 生成 LLM prefill/decode、visual HMONNX 和 MinerU 静态 ViT manifest。

默认配置在脚本顶部常量中：

```python
HF_MODEL_DIR = "/data02/datasets/MinerU2.5-Pro-2604-1.2B"
CONFIG_PATH = "./configs_merak/workflows/xh2a/llm_models/mineru2_5/mineru2_5_pro_xh2a_4k.yaml"
QUANT_OUTPUT_DIR = "..."
EXPORT_OUTPUT_DIR = "..."
```

按需修改 `QUANT_OUTPUT_DIR`、`EXPORT_OUTPUT_DIR` 或 `CONFIG_OVERRIDES` 后再运行。

## 手动导出 HMONNX

当需要绕开 workflow、单独调试导出参数时使用：

```bash
conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py \
  --visual-buckets-config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py \
  --force
```

导出产物应至少包含：

```text
golden_meta_info.json
mineru_visual_buckets.json
prefill/*.onnx
decode/*.onnx
visual 或 visual_1036x1036/*.onnx
额外 visual bucket *.onnx
```

`mineru_visual_buckets.json` 是 MinerU 专用 manifest，记录每个静态 visual bucket 的尺寸和 HMONNX 路径，不修改通用 HMONNX metadata schema。

## 单图推理

### 原始 HF

```bash
conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/debug_scripts/mineru2_5_native_generate.py
```

用途：确认原始 HF 模型和 `MinerUClient.two_step_extract` 可用。

### 原始 HF + 静态 ViT 模拟

```bash
conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_hf_static_vit_generate.py \
  --image-path data/images/houmo_logo.jpg \
  --max-new-tokens 128 \
  --no-tqdm
```

用途：在不使用 XH/HMONNX 的情况下，验证静态 ViT bucket 路由和图片预处理是否符合预期。

### XH PyTorch 静态 ViT

```bash
conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/debug_scripts/mineru2_5_xh_generate.py \
  --config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_llm_1_2b_xh2a_4k.py \
  --visual-buckets-config configs_merak/xh2a/llm_models/mineru2.5/1_2b/mineru2_5_visual_buckets_1_2b_xh2a.py \
  --image-path data/images/houmo_logo.jpg \
  --max-new-tokens 128 \
  --no-tqdm
```

用途：构建 XH PyTorch 模型和多个静态 `XHQwen2VLVisualModel`，确认导出 HMONNX 前的 XH PyTorch 路径可用。

### HMONNX

```bash
EXPORT_DIR=/path/to/exported_mineru2_5_hmonnx

conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_xh_hmonnx_generate.py \
  --config ${EXPORT_DIR}/golden_meta_info.json \
  --visual-buckets-manifest ${EXPORT_DIR}/mineru_visual_buckets.json \
  --image-path data/images/houmo_logo.jpg \
  --max-new-tokens 128 \
  --no-tqdm
```

用途：加载 HMONNX LLM 和所有 manifest 中声明的 visual bucket，按 `image_grid_thw` 在推理时切换静态 ViT。

`houmo_logo.jpg` 期望输出：

```python
[
  {"type": "image", "bbox": [0.152, 0.121, 0.862, 0.533], "content": None},
  {"type": "title", "bbox": [0.156, 0.573, 0.801, 0.81], "content": "后摩智能\nHOU MO.AI"}
]
```

## OmniDocBench 数据下载和复现

评估脚本使用 Hugging Face 数据集：

```text
opendatalab/OmniDocBench
```

为了复现同一批 100 条测试，固定数据集 revision 和确定性抽样策略：

```text
dataset_revision = d386947f7fc3bafdcd756c8485845a2f43a19875
sample_strategy = image_path_evenly_spaced
sample_size = 100
```

只要数据集 revision、抽样策略、样本数量一致，就会得到相同的 100 条样本 manifest。建议固定一个本地数据目录和输出目录：

```bash
DATA_DIR=/path/to/OmniDocBench
OUT_DIR=/path/to/mineru2_5_hf_dynamic_100
DATASET_REVISION=d386947f7fc3bafdcd756c8485845a2f43a19875
```

下载标注文件：

```bash
hf download \
  opendatalab/OmniDocBench \
  OmniDocBench.json \
  --repo-type dataset \
  --revision ${DATASET_REVISION} \
  --local-dir ${DATA_DIR}
```

如果没有 `hf` 命令，可以使用：

```bash
huggingface-cli download \
  opendatalab/OmniDocBench \
  OmniDocBench.json \
  --repo-type dataset \
  --revision ${DATASET_REVISION} \
  --local-dir ${DATA_DIR}
```

生成固定 100 条 manifest，并按同一 revision 下载这 100 条图片：

```bash
python - <<'PY'
import json
import math
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

data_dir = Path(os.environ["DATA_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
revision = os.environ["DATASET_REVISION"]
out_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / "OmniDocBench.json", "r", encoding="utf-8") as f:
    data = json.load(f)

indexed = list(enumerate(data))
indexed.sort(key=lambda item: item[1].get("page_info", {}).get("image_path", ""))
sample_size = 100
positions = [math.floor(i * (len(indexed) - 1) / (sample_size - 1)) for i in range(sample_size)]
selected = [indexed[pos] for pos in positions]

manifest = []
for rank, (json_index, record) in enumerate(selected):
    image_path = record["page_info"]["image_path"]
    if "/" not in image_path:
        image_path = f"images/{image_path}"
    page_info = record.get("page_info", {})
    manifest.append(
        {
            "sample_rank": rank,
            "json_index": json_index,
            "image_path": image_path,
            "page_number": page_info.get("page_number"),
            "width": page_info.get("width"),
            "height": page_info.get("height"),
            "page_attribute": page_info.get("page_attribute", {}),
        }
    )
    hf_hub_download(
        "opendatalab/OmniDocBench",
        repo_type="dataset",
        filename=image_path,
        revision=revision,
        local_dir=str(data_dir),
    )

manifest_path = out_dir / "omnidocbench_sample_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"saved sample manifest: {manifest_path}")
PY
```

后续所有 100 条评估都复用同一个 manifest：

```bash
SAMPLE_MANIFEST=${OUT_DIR}/omnidocbench_sample_manifest.json
```

运行 HF dynamic baseline：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hf_eval.py \
  --dataset-dir ${DATA_DIR} \
  --sample-manifest ${SAMPLE_MANIFEST} \
  --output-dir ${OUT_DIR} \
  --sample-size 100 \
  --dtype float16 \
  --no-download-missing \
  --no-tqdm
```

评估脚本本身也支持缺图时自动下载，但严格复现时建议使用上面的固定 revision 下载步骤，然后评估时加 `--no-download-missing`。

## OmniDocBench 评估

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

| 指标 | 含义 |
| --- | --- |
| `edit_similarity` | 归一化文本后，用 `difflib.SequenceMatcher` 计算相似度。 |
| `char_precision` | 字符 multiset overlap precision。 |
| `char_recall` | 字符 multiset overlap recall。 |
| `char_f1` | 字符级 F1。 |
| `avg_time` | 每页 `two_step_extract` 平均耗时。 |

### HF dynamic 评估

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hf_eval.py \
  --dataset-dir ${DATA_DIR} \
  --sample-manifest ${SAMPLE_MANIFEST} \
  --sample-size 100 \
  --output-dir /path/to/hf_dynamic_100 \
  --dtype float16 \
  --no-download-missing \
  --no-tqdm
```

### HF static ViT 模拟评估

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hf_eval.py \
  --dataset-dir ${DATA_DIR} \
  --sample-manifest ${SAMPLE_MANIFEST} \
  --sample-size 100 \
  --output-dir /path/to/hf_static_fitpadding_100 \
  --dtype float16 \
  --simulate-static-vit \
  --no-download-missing \
  --no-tqdm
```

### HMONNX 评估

```bash
EXPORT_DIR=/path/to/exported_mineru2_5_hmonnx

CUDA_VISIBLE_DEVICES=0 conda run -n xh2modelzoo python \
  examples_merak/llm/mineru2.5/mineru2_5_omnidocbench_hmonnx_eval.py \
  --config ${EXPORT_DIR}/golden_meta_info.json \
  --visual-buckets-manifest ${EXPORT_DIR}/mineru_visual_buckets.json \
  --dataset-dir ${DATA_DIR} \
  --hf-sample-manifest ${SAMPLE_MANIFEST} \
  --sample-count 100 \
  --output-dir /path/to/hmonnx_100 \
  --no-tqdm
```

## 当前 100 条评估结果

以下结果均使用同一份 `image_path_evenly_spaced` 100 条样本。

| 配置 | edit_similarity | char_precision | char_recall | char_f1 | avg_time |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF dynamic ViT fp16 | 0.7420 | 0.9307 | 0.9424 | 0.9256 | 45.55s |
| HF dynamic ViT bf16 | 0.7422 | 0.9309 | 0.9420 | 0.9255 | 37.24s |
| 10 bucket fit_padding | 0.7429 | 0.9307 | 0.9425 | 0.9266 | 39.19s |
| 5 bucket fit_padding | 0.7387 | 0.9311 | 0.9408 | 0.9256 | 37.52s |

结论：

- fp16 和 bf16 dynamic 精度基本一致。
- `fit_padding` 静态 ViT 在 5 个 content bucket 下仍能保持接近 HF dynamic 的 `char_f1`。
- `ratio` 策略已禁用，不再作为推荐或可选评估方案。

## 已验证的 HMONNX 单图结果

使用 workflow export 产物测试 `data/images/houmo_logo.jpg`：

```text
layout bucket: 1036x1036
title crop bucket: 140x392
output: 后摩智能\nHOU MO.AI
```

该测试说明当前 HMONNX generate 脚本可以正确读取 workflow 导出的 `mineru_visual_buckets.json`，并按最新版 `fit_padding` 路由执行两阶段识别。

## 注意事项

- HMONNX 推理只能路由到 `mineru_visual_buckets.json` 中已经导出的 bucket。
- 修改 visual bucket config 后，需要重新导出 HMONNX。
- 当前 MinerU 推理默认 batch size 为 1，避免一次 `generate()` 中混入不同 visual bucket。
- `content` 阶段默认禁止路由到 `1036x1036`；不要打开 `--allow-content-fallback-bucket`，除非是在做对照实验。
- 静态 ViT 预处理逻辑只在 MinerU2.5 示例脚本层实现，不修改通用 Qwen2-VL 模型代码。
