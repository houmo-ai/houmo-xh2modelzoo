# UI-TARS 1.5（Qwen2.5-VL）示例脚本说明

本目录只需要关注以下 8 个脚本：

- `demo.py`
- `eval.py`
- `gen_calib_data.py`
- `ui_tars_utils.py`
- `ui_tars1_5_common_quant.py`
- `ui_tars1_5_xh2a_demo.py`
- `ui_tars1_5_xh2a_eval.py`
- `ui_tars1_5_xh2a_export_hmonnx.py`

其余文件（`*.json`、`*.png`、`*.txt`、`run.log` 等）通常是运行产物/样例产物，不在本文档范围内。

## 约定

UI-TARS 1.5 的提示词把任务约束成 “GUI agent”，模型输出采用类似下面的结构：

```
Thought: ...
Action: click(point='<point>x1 y1</point>')
```

其中 `Action:` 行是机器可解析的动作调用（`click/drag/type/scroll/...`），坐标通常是 `<point>x y</point>` 或 `"(x,y)"`。

`ui_tars_utils.py` 负责两件关键事情：

- 生成提示词模板（`MOBILE_USE_DOUBAO` / `COMPUTER_USE_DOUBAO` / `GROUNDING_DOUBAO`）以及把 `PIL.Image` 封装成 Qwen2.5-VL 的输入消息结构（`construct_prompt`）。
- 把模型输出解析成结构化字典（`parse_output`），并把 `<point>x y</point>` 转成 `"(x,y)"`，再用 AST 解析动作函数名与参数。

## 脚本：`demo.py`（HF 直接推理 + 可视化点击点）

用途：

- 用 HuggingFace `Qwen2_5_VLForConditionalGeneration` 直接跑一张图 + 一条指令。
- 把模型输出原文保存成 `txt`，并在原图上标注点击点（如果解析到了 `click/point`）。

输入/输出：

- 输入：`--model_path`（默认 `weights/UI-TARS-1.5-7B`）、`--image_path`、`--prompt`
- 输出：`--output_dir` 下的
  - `<image_stem>.txt`（模型原始输出）
  - `<image_stem>_annotated.png`（标注后的图）
  - `log.txt`（loguru 日志）

运行示例：

```bash
python demo.py \
  --model_path weights/UI-TARS-1.5-7B \
  --image_path data/test/example.png \
  --prompt "Could you help me set the Number Threads to Use to 4?" \
  --output_dir data/output
```

说明要点：

- 脚本内部用 `smart_resize()` 计算 “模型侧分辨率”（宽高是 28 的倍数、像素数在指定范围内），用于把模型坐标映射回原图像素坐标后做可视化。

## 脚本：`eval.py`（HF 模型在 ScreenSpot 上做 grounding 评测）

用途：

- 读取 ScreenSpot 的测试集 JSON（每条样本包含 `bbox/img_size/instruction/img_filename`）。
- 对每个样本执行 grounding 推理，只取 “点击点”。
- 判断预测点是否落在 GT `bbox` 内，输出整体准确率与分类准确率。

输入/输出：

- 必填参数：
  - `--model_path`：HF 模型目录（例如 `weights/UI-TARS-1.5-7B`）
  - `--screenspot_imgs`：图片目录（例如 `data/screenspot/images`）
  - `--screenspot_test`：任务 JSON 目录（例如 `data/screenspot/annotations`）
  - `--log_path`：评测结果输出路径（JSON）
- 可选参数：
  - `--task`：`all` 或逗号分隔任务名
  - `--device`：默认 `cuda`
  - `--num_gpu`：>1 时用多进程多卡

运行示例（单卡）：

```bash
python eval.py \
  --model_path weights/UI-TARS-1.5-7B \
  --screenspot_imgs /path/to/screenspot/images \
  --screenspot_test /path/to/screenspot/test \
  --log_path data/output/screenspot_report.json \
  --device cuda
```

运行示例（多卡，多进程 spawn）：

```bash
python eval.py \
  --model_path weights/UI-TARS-1.5-7B \
  --screenspot_imgs /path/to/screenspot/images \
  --screenspot_test /path/to/screenspot/test \
  --log_path data/output/screenspot_report.json \
  --num_gpu 8
```

说明要点：

- 推理使用 `mode="grounding"` 的提示词（见 `ui_tars_utils.construct_prompt(..., mode="grounding")`），输出只需要 `click(point=...)`。
- 脚本会先对图做 `smart_resize()`，再把预测坐标归一化到 \([0,1]\) 并映射回原始 `img_size` 做评测。

## 脚本：`gen_calib_data.py`（生成量化校准数据）

用途：

- 从 ScreenSpot 的任务 JSON 中读取样本（指令 + 图片文件名），取前 `--num_samples` 条有效样本。
- 为每条样本生成一条 Qwen2.5-VL 模型结构期望的 `messages`（image + text prompt，指令来自该样本的 `instruction`）。
- 以 JSON Lines 形式写入校准文件（每行一条 `messages`）。

输入/输出：

- 输入：`--screenspot_imgs`（必填）、`--screenspot_test`（必填）、`--task`、`--num_samples`
- 输出：`--output_file`（默认 `data/calib_data.json`，实际格式是 JSONL）

运行示例：

```bash
python gen_calib_data.py \
  --screenspot_imgs /path/to/screenspot/images \
  --screenspot_test /path/to/screenspot/annotation \
  --task all \
  --num_samples 128 \
  --output_file data/calib_data.json
```

## 脚本：`ui_tars1_5_common_quant.py`（通用量化：Quarot + GPTQ）

用途：

- 读取 HF 模型（先放 CPU 以省显存），可选执行：
  - Quarot（旋转/变换类量化前处理）
  - GPTQ（基于校准集的权重量化）
- 把中间/最终 `state_dict` 以 `safetensors` 保存到 `work_dirs/<cfg_name>/`。

输入/输出：

- 输入：
  - `--model`：HF 模型目录（默认 `weights/UI-TARS-1.5-7B`）
  - `--data_files`：校准数据文件列表（通常把 `gen_calib_data.py` 的输出传进来）
  - `--skip-quarot` / `--skip-gptq`：跳过对应步骤
  - `--w-bits` / `--w-head-bits` / `--calib-samples` 等：GPTQ 量化配置
- 输出：
  - `work_dirs/<cfg_name>/*-state-dict.safetensors`
  - 以及可能的 `layers_cache/`（GPTQ 缓存）

运行示例（典型流程）：

```bash
python ui_tars1_5_common_quant.py \
  --model weights/UI-TARS-1.5-7B \
  --data_files data/calib_data.json \
  --calib-samples 128
```

说明要点：

- GPTQ 路径依赖 `xh_model_zoo.xh_llm.quarot.quantizer_utils.gptq`，并通过 `--data_files` 传入校准数据。
- 最终保存前会把 `quant_weight` 张量尝试压成 `int8/int16`，其他权重保存为 `float16`。

## 脚本：`ui_tars1_5_xh2a_export_hmonnx.py`（导出到 XH2a HMONNX）

用途：

- 把 HF 的 `Qwen2_5_VLForConditionalGeneration` 转换/导出成 XH2a 侧可用的 HMONNX/ONNX 产物。
- 配置量化方案（`QuantScheme`），并支持挂载外部量化权重（`--quant_weight`）。

输入/输出：

- 输入：
  - `--model`：HF 模型目录
  - `--quant-type`：默认 `w4a8h0_ssfp`
  - `--quant_weight`：可选，指向 GPTQ/Quarot 产生的量化权重
  - `--context-length` / `--max_pe_length` / 视觉相关参数等
  - `--sample_image_path`：默认 `data/test/example.png`，用于生成导出阶段所需的 demo 图片输入
- 输出：
  - `work_dirs/<prefix>/`（转换产物与日志），其中 `<prefix>` 来自脚本内部的 `f"{model_name}-{target_device}"`

运行示例：

```bash
python ui_tars1_5_xh2a_export_hmonnx.py \
  --model weights/UI-TARS-1.5-7B \
  --quant-type w4a8h0_ssfp \
  --context-length 4096
```

## 脚本：`ui_tars1_5_xh2a_demo.py`（XH2a 导出模型推理 + 可视化）

用途：

- 加载 `ui_tars1_5_xh2a_export_hmonnx.py` 导出的模型目录（读 `meta.json`）。
- 构造 `Qwen2_5_VLONNXModel` 并加载 `token_embedding`。
- 执行 `model.chat(...)` 得到输出文本，并把点击点标到图上。

输入/输出：

- 输入：`--model_dir`（必填，导出目录）、`--image`、`--instruction`
- 输出：`--output_image`（默认 `demo_result.png`）以及同名 `txt`

运行示例：

```bash
python ui_tars1_5_xh2a_demo.py \
  --model_dir work_dirs/<prefix> \
  --image data/test/example.png \
  --instruction "Could you help me set the Number Threads to Use to 4?" \
  --output_image data/output/xh2a_demo.png
```

## 脚本：`ui_tars1_5_xh2a_eval.py`（XH2a 导出模型在 ScreenSpot 上评测）

用途：

- 与 `eval.py` 类似，但推理后端替换成导出的 `Qwen2_5_VLONNXModel`。
- 额外处理坐标映射：根据 `model.resize_v1` 决定使用等比缩放映射或分别按宽高缩放。
- 支持 `--num_gpu > 1` 的多进程多卡评测。
- 支持 `--max_samples` 限制评测样本数量。

输入/输出：

- 必填参数：
  - `--model_dir`：导出模型目录
  - `--screenspot_imgs` / `--screenspot_test` / `--log_path`
- 可选参数：
  - `--task`：筛选任务（默认 `all`）
  - `--num_gpu`：并行评测（默认 `1`）
  - `--max_samples`：只评测前 N 条样本（默认 `None`）

运行示例：

```bash
python ui_tars1_5_xh2a_eval.py \
  --model_dir work_dirs/<prefix> \
  --screenspot_imgs /path/to/screenspot/images \
  --screenspot_test /path/to/screenspot/test \
  --log_path data/output/xh2a_screenspot_report.json \
  --num_gpu 8
```
