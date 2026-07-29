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

Gemma4 Series Merak 路径已验证使用 Transformers 5.13。不要据此直接修改整个
仓库的 4.57 依赖约束；专用环境、Safetensors 下限和验证矩阵见
[Merak Transformers 5.13 兼容性结论](../../../docs/merak_transformers_5_13_compatibility_20260727.md)。

## 详细文档

完整结构差异、输入输出 shape、HMONNX 产物说明、量化/导出/demo/e2e 命令见：

- [Gemma4 Series Merak 统一 Workflow 使用与模型说明](../../../docs/gemma4_series_merak_workflow_guide_20260621.md)
- [Gemma4 ViT padded 输入定版方案](../../../docs/gemma4_vit_padded_input_design_20260616.md)

README 只保留最小入口和常用命令，避免和长期文档重复。

## 推荐配置

| preset | HF checkpoint | workflow YAML |
| --- | --- | --- |
| `e2b` | `weights/gemma-4-E2B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml` |
| `e4b` | `weights/gemma-4-E4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml` |
| `31b` | `weights/gemma-4-31B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml` |
| `26b-a4b` | `weights/gemma-4-26B-A4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml` |

四份 YAML 都使用同一个 public model type：

```yaml
model_type: Gemma4ForConditionalGeneration
prefill_chunk_length: 320
sliding_kv_cache_input_mode: slice_window
```

## Prefill chunk selection

Gemma4 Series now exports exactly one text prefill graph: `prefill` with
`prefill_chunk_length: 320`. The 320 width is configurable but must stay at
least 280 so one image feature block is never split. Runtime chunking packs text
freely and treats each contiguous visual/audio range as atomic: text can fill
remaining space after a complete range, image ranges normally occupy 280 tokens,
and video frames are about 70 tokens so multiple complete frames can share one
320-token prefill chunk. If the next multimodal range would cross the chunk
boundary, the planner starts a new `prefill` call instead of slicing that range.

All prefill/decode graphs share one physical sliding-cache width:

```text
aligned(sliding_window + prefill_chunk_length, 16)
```

MTP target verify keeps that physical cache width and controls the visible range
with masks. `accepted_count` is a verify-only input for the MTP slice-window
cache rollback path; it is not a general prefill/decode user control.

正式 QTL-384 集成导出请用 `--context-max-length 8192` 覆盖 YAML 默认值。

## model_name 命名规范

Gemma4 Series workflow YAML 使用显式 `model_name`：

```yaml
export:
  model:
    model_name: gemma_4_e4b
```

命名格式：

```text
gemma_4_<variant>
```

示例：

```text
gemma_4_e2b
gemma_4_e4b
gemma_4_31b
gemma_4_26b_a4b
```

说明：

- context、prefill、量化方法和 MPE 由 YAML/metadata 表达，不再塞进目录名。
- `h1_sefp` 只保留在 `quant_scheme.quant_type`，不进入目录名。

## 当前推荐最优权重

当前按最新 CEval 结果和 MoE 加载稳定性选择：E2B 用 AutoRound，其余模型
优先用 GPTQModel。

| preset | 推荐量化 | 环境变量 |
| --- | --- | --- |
| `e2b` | AutoRound | `GEMMA4_E2B_AUTOROUND_HF` |
| `e4b` | GPTQModel | `GEMMA4_E4B_GPTQMODEL_HF` |
| `26b-a4b` | GPTQModel | `GEMMA4_26B_A4B_GPTQMODEL_HF` |
| `31b` | GPTQModel | `GEMMA4_31B_GPTQMODEL_HF` |

## CEval 精度摘要

数据源：`work_dirs/qtl384_gemma4_ceval_maxtok512_live_summary.md`，更新时间
`2026-06-23 00:09:49 CST`。CEval full 使用 `--limit 0`，共 1346 samples；
`--max-tokens 512`，避免长答案被截断。

| preset | FP | GPTQ weight-only | AutoRound weight-only | GPTQ HMONNX | AutoRound HMONNX | 当前推荐 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `e2b` | 0.3143 | 0.2860 | 0.3247 | 0.2897 (390/1346) | 0.3276 (441/1346) | AutoRound |
| `e4b` | 0.5743 | 0.5468 | 0.5520 | 0.5505 (741/1346) | 0.5364 (722/1346) | GPTQModel |
| `26b-a4b` | 0.7325 | 0.7229 | 0.7162 | 0.7214 (971/1346) | 0.7273 (979/1346) | GPTQModel |
| `31b` | 0.8024 | 0.7883 | 0.7734 | 0.8118 (full-only, RUN) | 0.8023 (full-only, RUN) | GPTQModel |

说明：31B 的 full-only 与 split-merged 评测会有重复/去重口径差异；README
只保留用于选型的汇总数，详细路径和中间状态见上面的 work_dirs summary。

## MTP target + draft export

MTP uses the same Gemma4 Series model family.  Use the `full_mtp` YAMLs with
`gemma4_series_quant_export.py`; pass the assistant/draft HF directory from the
CLI.  `Gemma4SeriesWorkflow.export()` owns both target HMONNX export and
assistant draft ONNX export; the CLI only injects paths/config overrides.  The
draft is written under the same exported model directory and recorded in
`golden_meta_info.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir weights/gemma-4-E2B-it \
  --config configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml \
  --existing-hf-model-dir /path/to/quant_hf \
  --export-output-dir work_dirs/gemma4_series_export/e2b_mtp \
  --mtp-assistant-model-dir weights/gemma-4-E2B-it-assistant \
  --device cuda:0 \
  --force
```

MTP YAMLs use `spec_decode_mode: mtp` as the high-level switch.  The target
`target_hidden_state` output is derived from that mode, so YAML does not need a
separate `enable_mtp_outputs` field.  Draft quantization has one source of
truth under `mtp_config`:

```yaml
mtp_config:
  body_quant_type: w8a8h1_sefp
  lm_head_quant_type: w4a8h0_ssfp
```

`lm_head_quant_type` controls the assistant logits head quantization; manifest
`draft_head_weight_bits` is derived from it.  Runtime verify/generate tools
should prefer `--meta /path/to/golden_meta_info.json` so target and draft paths
come from the single manifest.

Non-MTP export uses the same CLI without `--mtp-assistant-model-dir` and with a
normal `full.yaml` config:

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir weights/gemma-4-E2B-it \
  --config configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml \
  --existing-hf-model-dir /path/to/quant_hf \
  --export-output-dir work_dirs/gemma4_series_export/e2b_non_mtp \
  --device cuda:0 \
  --force
```

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

## 一键量化/导出：`gemma4_series_quant_export.py`

重型量化导出统一走 `gemma4_series_quant_export.py`，不要再通过
`gemma4_workflow_demo.py` 做正式产物。脚本只负责：

1. 读取 workflow YAML；
2. 注入 HF / existing quant / MTP assistant 路径；
3. 调用 `workflow.quant()`、`workflow.export()`、可选 `workflow.dump_golden()`；
4. 打印 JSON summary，最终运行以 `golden_meta_info.json` 为单一 manifest。

单机多任务时遵守“单 GPU 单任务”，例如：

```bash
export CUDA_VISIBLE_DEVICES=0
PYTHON=${PYTHON:-python}
```

### 复用已有量化 HF 目录：非 MTP

```bash
$PYTHON examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir weights/gemma-4-E2B-it \
  --config configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml \
  --existing-hf-model-dir "$GEMMA4_E2B_AUTOROUND_HF" \
  --export-output-dir work_dirs/gemma4_series_export/e2b_non_mtp \
  --device cuda:0 \
  --dump-golden \
  --prompt "请用中文解释量子力学。" \
  --force
```

### 复用已有量化 HF 目录：MTP target + draft

必须使用 `gemma4_*_full_mtp.yaml`，或确保 YAML 中有完整
`export.model.mtp_config`。传入 `--mtp-assistant-model-dir` 时，CLI 会在真正
运行前校验 effective config；如果误用了普通 `full.yaml`，会直接报出缺失的
MTP 字段。

```bash
$PYTHON examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir weights/gemma-4-E2B-it \
  --config configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml \
  --existing-hf-model-dir "$GEMMA4_E2B_AUTOROUND_HF" \
  --export-output-dir work_dirs/gemma4_series_export/e2b_mtp \
  --mtp-assistant-model-dir weights/gemma-4-E2B-it-assistant \
  --device cuda:0 \
  --dump-golden \
  --prompt "请用中文解释量子力学。" \
  --force
```

导出完成后用 manifest 验证 target/draft 契约，并跑 512-token MTP 生成：

```bash
META=$(find work_dirs/gemma4_series_export/e2b_mtp -name golden_meta_info.json | sort | tail -1)

$PYTHON examples_merak/llm/gemma4_series/mtp_hmonnx_inference.py verify \
  --preset e2b \
  --meta "$META" \
  --json

$PYTHON examples_merak/llm/gemma4_series/mtp_hmonnx_inference.py generate \
  --preset e2b \
  --meta "$META" \
  --prompt "请用中文解释量子力学的核心思想，并说明它和人工智能推理的关系。" \
  --max-new-tokens 512 \
  --num-draft-tokens 4 \
  --device cuda:0
```

### 从原始 HF 从头量化再导出

不传 `--existing-hf-model-dir` 且不传 `--base` 时会执行真实 quant，此时必须
给 `--quant-output-dir`：

```bash
$PYTHON examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
  --hf-model-dir weights/gemma-4-E4B-it \
  --config configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml \
  --quant-output-dir work_dirs/gemma4_series_export/e4b_quant \
  --export-output-dir work_dirs/gemma4_series_export/e4b_hmonnx \
  --device cuda:0 \
  --dump-golden \
  --prompt "请用中文概括 Gemma4 的能力。" \
  --force
```

### 四模型推荐组合批量导出模板

下面模板只覆盖当前推荐组合；按机器空闲情况给每个任务分配一张卡即可。
`GPU` 是物理卡号，命令内部仍用 `--device cuda:0`。

```bash
set -euo pipefail
PYTHON=${PYTHON:-python}
ROOT=work_dirs/gemma4_series_quant_export_$(date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/logs"

run_one() {
  local name=$1 gpu=$2 hf=$3 cfg=$4 quant_hf=$5
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" \
    examples_merak/llm/gemma4_series/gemma4_series_quant_export.py \
      --hf-model-dir "$hf" \
      --config "$cfg" \
      --existing-hf-model-dir "$quant_hf" \
      --export-output-dir "$ROOT/$name/export" \
      --device cuda:0 \
      --dump-golden \
      --prompt "请用中文解释量子力学、人工智能和科学推理之间的关系。" \
      --force > "$ROOT/logs/${name}.log" 2>&1
  find "$ROOT/$name/export" -name golden_meta_info.json | sort | tail -1 \
    > "$ROOT/${name}_meta.txt"
}

run_one e2b_autoround 0 weights/gemma-4-E2B-it \
  configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml \
  "$GEMMA4_E2B_AUTOROUND_HF" &
run_one e4b_gptq 1 weights/gemma-4-E4B-it \
  configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml \
  "$GEMMA4_E4B_GPTQMODEL_HF" &
run_one 26b_a4b_gptq 2 weights/gemma-4-26B-A4B-it \
  configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml \
  "$GEMMA4_26B_A4B_GPTQMODEL_HF" &
run_one 31b_gptq 3 weights/gemma-4-31B-it \
  configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml \
  "$GEMMA4_31B_GPTQMODEL_HF" &
wait

find "$ROOT" -name golden_meta_info.json -print | sort
```

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
