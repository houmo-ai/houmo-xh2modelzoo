# Gemma4 Series unified workflow demo

本目录只推荐新的统一 workflow API。E2B、E4B、31B dense、26B-A4B MoE 都从
同一个入口进入：

```text
Gemma4ForConditionalGeneration
  -> AutoLLMWorkflow.from_config(...)
  -> xhmodel_merak.xh_llm.models.gemma4_series.workflow.Gemma4SeriesWorkflow
  -> xhmodel_merak.xh_llm.models.gemma4_series.XHGemma4SeriesModel
```

旧 `gemma4/`、`gemma4e/`、`gemma4_moe/` 目录只作为历史兼容面；新的
Gemma4 Series 导出/生成不要再把它们当实现入口。

## 详细文档

完整结构差异、输入输出 shape、HMONNX 产物说明、量化/导出/demo/e2e 命令见：

- [Gemma4 Series Merak 统一 Workflow 使用与模型说明](../../../docs/gemma4_series_merak_workflow_guide_20260621.md)
- [Gemma4 ViT padded 输入定版方案](../../../docs/gemma4_vit_padded_input_design_20260616.md)

README 只保留最小入口和常用命令，避免和长期文档重复。

## 推荐配置

| preset | HF checkpoint | workflow YAML |
| --- | --- | --- |
| `e2b` | `/data01/datasets/gemma-4-E2B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml` |
| `e4b` | `/data01/datasets/gemma-4-E4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml` |
| `31b` | `/data01/datasets/gemma-4-31B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml` |
| `26b-a4b` | `/data01/datasets/gemma-4-26B-A4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml` |

四份 YAML 都使用同一个 public model type：

```yaml
model_type: Gemma4ForConditionalGeneration
prefill_chunk_length: 256
sliding_kv_cache_input_mode: slice_window
```

正式 QTL-384 集成导出请用 `--context-max-length 8192` 覆盖 YAML 默认值。

## 当前推荐最优权重

| preset | 推荐量化 | 环境变量 |
| --- | --- | --- |
| `e2b` | AutoRound | `GEMMA4_E2B_AUTOROUND_HF` |
| `e4b` | GPTQModel | `GEMMA4_E4B_GPTQMODEL_HF` |
| `26b-a4b` | AutoRound | `GEMMA4_26B_A4B_AUTOROUND_HF` |
| `31b` | AutoRound | `GEMMA4_31B_AUTOROUND_HF` |

## 轻量解析验证

不会加载大模型权重，不会量化/导出：

```bash
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset e2b --dry-run
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset e4b --dry-run
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset 31b --dry-run
python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset 26b-a4b --dry-run
```

## 复用最优量化 HF 目录导出 HMONNX

示例：E4B GPTQModel，8192 context，256 prefill，并为所有模块 dump golden。

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/gemma4_workflow_demo.py \
  --preset e4b \
  --action existing-hf \
  --existing-hf-model-dir "$GEMMA4_E4B_GPTQMODEL_HF" \
  --work-dir ./work_dirs/qtl384_gemma4_best_exports/e4b_gptqmodel \
  --context-max-length 8192 \
  --prefill-chunk-length 256 \
  --sliding-kv-cache-input-mode slice_window \
  --golden \
  --force
```

完整四模型命令见详细文档。

## 真实 generate demo

Text：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --prompt "请阅读下面长资料卡并回答最后的问题：..." \
  --max-decode-steps 128
```

Image：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --image-path /path/to/real_image.png \
  --prompt "请用中文描述图片中的文字、颜色、形状和布局。" \
  --max-decode-steps 128
```

Video / Audio：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --video-path /path/to/video_or_frames \
  --prompt "请总结视频中多帧画面的变化。" \
  --max-decode-steps 128

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/generate.py \
  --backend hmonnx \
  --model-config /path/to/golden_meta_info.json \
  --audio-path /path/to/audio.wav \
  --prompt "请转写并概括音频内容。" \
  --max-decode-steps 128
```

Audio 只适用于 E2B/E4B。

## 严格验收规则摘要

- text prompt token 必须大于 1024，且 `prompt_tokens + max_new_tokens <= context`。
- 每个模型至少覆盖 text 和 image generate；E2B/E4B 还要覆盖 audio/video。
- image/video/audio 必须是真实多模态问答，不能用无关问题替代。
- video 必须经过独立 `video_visual` HMONNX。
- audio 必须经过 `audio` HMONNX。
- `py_compile`、dry-run、processor-only validation、meta 文件存在都不能替代 e2e generate。
- 单 GPU 同时只跑一个 Gemma4 重型导出/生成任务。
