# MiniCPM-V-4.5 Merak Workflow

本目录提供 MiniCPM-V-4.5 的 HMONNX 导出、Golden 生成和推理示例。模型由
SigLIP2 Vision Encoder、2D/3D Resampler 和 Qwen3-8B LLM 组成，支持图片、视频、
图文混合及纯文本输入。

## 环境

MiniCPM-V-4.5 remote code 需要 `transformers==4.57.1`：

```bash
pip install "transformers==4.57.1"
pip install -v -e . --no-build-isolation
export TOKENIZERS_PARALLELISM=false
```

## 配置

正式配置仅保留两种：

| 配置 | Vision / VisionVideo | LLM |
| --- | --- | --- |
| `minicpm_v_4_5_xh2a_w8a8.yaml` | W8A8 | W8A8，不使用 GPTQ |
| `minicpm_v_4_5_xh2a_w4a8_gptq.yaml` | W8A8 | W4A8，GPTQ 4-bit，group size 64 |

配置目录：

```text
configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b
```

视觉图使用 `patch_capacity=1600`，同时用于图片和视频帧。GPTQ 只处理 LLM
骨干；W4A8 配置从仓库内 QwenVL 风格混合校准池固定抽取 64 条样本。

## 导出

W8A8 LLM：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_5/minicpm_v_4_5_workflow.py \
  --model-dir <minicpm_v_4_5_model_dir> \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/minicpm_v_4_5_w8a8_export \
  --device cuda:0 --overwrite
```

W4A8 LLM + GPTQ：

```bash
GPTQMODEL_SOURCE=<gptqmodel_dir> CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_5/minicpm_v_4_5_workflow.py \
  --model-dir <minicpm_v_4_5_model_dir> \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w4a8_gptq.yaml \
  --quant-output-dir work_dirs/minicpm_v_4_5_w4a8_quant \
  --export-output-dir work_dirs/minicpm_v_4_5_w4a8_export \
  --device cuda:0 --overwrite
```

增加 `--dump-golden` 可在导出后生成 Golden。主产物入口为导出目录中的
`golden_meta_info.json`，其中包含 Vision、VisionVideo、Prefill 和 Decode 图信息。

## 推理

浮点基线：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_5/float_demo.py \
  --model-dir <minicpm_v_4_5_model_dir> \
  --image <image_file> \
  --device cuda:0
```

HMONNX：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python examples_merak/llm/minicpm_v_4_5/hmonnx_demo.py \
  --export-meta <export_dir>/hmquant_*/golden_meta_info.json \
  --image <image_file> \
  --prompt "请描述这张图片。"
```

可重复传入 `--image`，或使用 `--video` 执行视频推理。运行时仍需通过 metadata
中的 `model_config.hf_model` 访问原始模型目录，以加载 processor 和 tokenizer。
