# Qwen3.5 / Qwen3.6 Merak 模型与 HMONNX 图规格

本文是 Qwen3.5 / Qwen3.6 Merak workflow 的模型文档。它补充 README 中的快速入口，说明默认配置、推荐 YAML、量化/导出字段、MTP/DFlash/Visual 支持范围，以及导出后 HMONNX 图在推理中的数据流。

## 1. 支持范围

| 模型 | family | model_size | HF 默认路径 | 已验证外部量化路径 | 推荐 YAML 变体 |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | `qwen3_5` | `9b` | `weights/Qwen3.5-9B` | `weights/Qwen3.5-9B-mode1-llm-only` | full / mtp / dflash / visual_only_448 / visual_only_896 |
| Qwen3.6-27B | `qwen3_5` | `27b` | `weights/Qwen3.6-27B` | 暂无本轮验证 | full / mtp / dflash / visual_only_448 / visual_only_896 |
| Qwen3.6-35B-A3B | `qwen3_5_moe` | `35b_a3b` | `weights/Qwen3.6-35B-A3B` | `weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400` | full / mtp / dflash / visual_only_448 / visual_only_896 |

本轮运行验证重点：9B 和 35B-A3B。MTP/DFlash 只验证外部已量化模型（`existing_hf` / quant 产物），不把 base 模型纳入 spec-decode runtime 验证矩阵。

## 2. 公共 API

```python
from xhmodel_merak.xh_llm.models.qwen3_5.workflow_api import (
    export,
    get_default_export_config,
    get_default_quant_config,
    get_default_workflow_config,
    get_export_config_help,
    get_model_docs,
    get_quant_config_help,
    list_recommended_configs,
    quant,
)
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
```

推荐集成方式：

```python
workflow = AutoLLMWorkflow.from_config(
    hf_model_dir="weights/Qwen3.5-9B",
    config_path="configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml",
)
quant_result = workflow.quant(output_dir="work_dirs/qwen35_9b_quant", device="cuda:0")

export_result = workflow.export(
    quant_result=quant_result,
    output_dir="work_dirs/qwen35_9b_export",
    device="cuda:0",
)
```

上游不需要重新维护 AutoRound、GPTQModel、`quant_scheme`、视觉尺寸、MTP/DFlash 细节。普通用户选择推荐 YAML；进阶用户通过结构化 help 与 config override 修改字段。

## 3. 默认配置获取

### 3.1 默认量化配置

```python
quant_cfg = get_default_quant_config()
```

默认值：

```yaml
algorithm: gptqmodel
method: autoround
output_format: gptqmodel_hf
artifact_format: gptqmodel_hf
bits: 4
group_size: 64
sym: true
iters: 200
format: auto_gptq
calibration:
  dataset: NeelNanda/pile-10k
  nsamples: 128
  seqlen: 2048
runtime:
  batch_size: 8
  trust_remote_code: true
  low_gpu_mem_usage: true
```

约束：`group_size` 必须为 64；默认 AutoRound workflow YAML 必须包含量化配置。GPTQ 伴随配置使用 `_gptq.yaml` 后缀并设置 `method: gptq`。只有显式 base 验证时才使用 `config_overrides={"quant": None}`。

### 3.2 默认 workflow / export 配置

```python
# 按唯一 YAML 名称取完整 workflow 配置
workflow_cfg = get_default_workflow_config(name="qwen3_5_9b_full_mtp")

# 按 family/model_size/variant 选择
export_cfg = get_default_export_config(family="dense", model_size="9b", variant="dflash")

# visual_only 需要指定 visual_size 才唯一
visual_cfg = get_default_export_config(
    family="moe",
    model_size="35b_a3b",
    variant="visual_only",
    visual_size=896,
)
```

`list_recommended_configs()` 返回所有推荐 YAML 的结构化索引，字段包括 `name`、`family`、`model_size`、`variant`、`visual_size`、`config_path`、`model_type`、`model_name`。

## 4. 配置说明获取

```python
quant_help = get_quant_config_help()
export_help = get_export_config_help()
model_docs = get_model_docs()
```

`quant_help["fields"]` / `export_help["fields"]` 是字段级说明，格式为：

```python
{
    "export.model.context_max_length": {
        "type": "int",
        "default": 2048,
        "description": "KV cache 最大长度，影响 prefill/decode HMONNX cache shape。",
    }
}
```

上游 CLI 可以直接把这些结构化说明打印为帮助文本，不需要复制每个模型的参数表。

## 5. 推荐 YAML 命名

YAML 名称只表达拓扑/导出形态，不表达量化格式。不要把 `w4`、`w8`、`gptq`、`autoround` 写进 workflow YAML 文件名。

推荐路径：

```text
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_visual_only_448.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_visual_only_896.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/*.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/*.yaml
```

## 6. Quant 流程

### 6.1 默认 AutoRound → GPTQModel HF

默认 `quant()` 运行 AutoRound，并保存 GPTQModel 兼容 HF 目录。`QuantResult` 会记录：

| 字段 | 说明 |
| --- | --- |
| `hf_model_dir` | 原始 HF 模型路径 |
| `quanted_model_dir` | 量化后 HF/GPTQModel 目录 |
| `skipped` | base 验证时为 True |

### 6.2 使用外部已量化模型

```python
config_overrides = {
    "quant": {
        "algorithm": "existing_hf",
        "artifact_format": "gptqmodel_hf",
        "source_algorithm": "autoround",
        "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
    }
}
```

35B-A3B：

```python
config_overrides = {
    "quant": {
        "algorithm": "existing_hf",
        "artifact_format": "gptqmodel_hf",
        "source_algorithm": "autoround",
        "existing_hf_model_dir": "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400",
    }
}
```

### 6.3 base 验证

```python
quant_result = quant(..., config_overrides={"quant": None})
```

这只用于显式验证 base HF 模型；默认配置仍然是量化，不是 `null`。

## 7. Export 变体

| variant | 说明 | 关键字段 |
| --- | --- | --- |
| `full` | LLM prefill/decode；YAML 带 `visual_config` 时包含视觉分支配置 | `context_max_length=2048`, `prefill_chunk_length=256` |
| `mtp` | full + MTP draft 模型 | `spec_decode_mode=mtp`, `num_draft_tokens=4`, `output_post_norm_hidden=true`, `mtp_config` |
| `dflash` | full + DFlash draft 模型 | `spec_decode_mode=dflash`, `num_draft_tokens=9`, `output_hidden_state_indices`, `dflash_config` |
| `visual_only` | 只导出视觉塔 | `model_type=*_visual`, `max_size_w/h=448 or 896` |

`fuse_gdr_ops` 默认 `False`。编译器支持成熟后，可以通过 override 或 YAML 改成 `True`：

```python
config_overrides={"export.model.fuse_gdr_ops": True}
```

## 8. Visual 支持

默认 full YAML 里的 `visual_config` 使用 448×448：

```yaml
visual_config:
  max_size_w: 448
  max_size_h: 448
  quant_scheme:
    quant_type: w8a8h1_sefp
    ops: {}
```

推荐 visual-only YAML 提供 448×448 和 896×896 两种尺寸。full 导出若要切换视觉尺寸，用：

```python
config_overrides={
    "export.model.visual_config.max_size_w": 896,
    "export.model.visual_config.max_size_h": 896,
}
```

visual-only 导出若要切换尺寸，用对应 `*_visual_only_448.yaml` 或 `*_visual_only_896.yaml`，或覆盖 `export.model.max_size_w/h`。

## 9. MTP 数据流

MTP（Multi-Token Prediction）是单层 draft transformer。target prefill/decode 输出 `post_norm_hidden`，MTP draft 使用 token embedding + hidden 预测多个 draft token，然后 target decode 负责 verify。

```text
Target Prefill/Decode
  -> logits
  -> post_norm_hidden
       -> MTP Draft Prefill/Decode
            -> draft logits -> draft tokens
                 -> Target Decode verify
```

9B 默认 MTP 配置：

```yaml
spec_decode_mode: mtp
num_draft_tokens: 4
output_post_norm_hidden: true
mtp_config:
  hidden_size: 4096
  num_key_value_heads: 4
  head_dim: 256
  input_sequence_length: 1
  context_max_length: 2048
```

35B-A3B 默认 MTP 配置：`hidden_size=2048`、`num_key_value_heads=2`、`head_dim=256`。

## 10. DFlash 数据流

DFlash 使用 target 指定层 hidden states 构造 draft 上下文，再由 DFlash draft decode 生成 draft logits。

```text
Target Prefill/Decode
  -> target_hidden (output_hidden_state_indices)
       -> DFlash Context: build draft KV cache
       -> DFlash Decode: produce draft logits
            -> Target Decode verify
```

DFlash YAML 中 `dflash_config.target_model_dir` 必须保持 `null`：

```yaml
dflash_config:
  hf_model: weights/Qwen3.5-9B-DFlash
  target_model_dir: null
```

运行时 `XHQwen3_5ModelConfig` 会把 `target_model_dir=None` 解析成当前 `export.model.hf_model`。这样用户覆盖 HF/base/quant 路径时，DFlash 不会继续指向旧模型目录。

## 11. HMONNX 图 IO 概览

以下以 Qwen3.5-9B 典型配置为例：`hidden_size=4096`、`context_max_length=2048`、`prefill_chunk_length=256`。

### 11.1 Target Prefill / Decode

输入名来自 `XHQwen3_5Model.get_export_cfg()`：

| 输入 | 说明 |
| --- | --- |
| `inputs_embeds` | token/visual embedding，prefill 为 `[1, 256, H]`，decode 通常为 `[1, 1, H]` |
| `time_position_ids`, `hight_position_ids`, `width_position_ids` | 文本/视觉统一 3D RoPE 位置 |
| `past_seq_length` | cache 已占用长度 |
| `current_input_length` | 当前输入长度 |
| `linear_attn_mask` | linear attention mask |
| `past_key_cache_*`, `past_value_cache_*` | full-attention KV cache |
| `past_conv_cache_q/k/v_*` | split linear-attention conv cache |
| `past_recurrent_state_*` | linear-attention recurrent state |

输出：

| 输出 | 说明 |
| --- | --- |
| `logits` | 当前步 logits |
| `conv_cache_out_*` | 更新后的 linear conv cache |
| `recurrent_state_out_*` | 更新后的 recurrent state |
| `post_norm_hidden` | MTP 模式输出给 draft |
| `target_hidden` | DFlash 模式输出给 draft |

### 11.2 MTP Draft Prefill / Decode

MTP draft 输入名：

| 输入 | 说明 |
| --- | --- |
| `next_token_embedding` | 当前 token embedding |
| `post_norm_hidden` | target 或上一步 draft hidden |
| `past_seq_length` | draft cache 偏移 |
| `current_input_length` | draft 当前长度 |
| `past_key_cache`, `past_value_cache` | draft 单层 KV cache |

输出：`logits`、`post_norm_out`。

### 11.3 DFlash Context / Decode

DFlash context 输入：`target_hidden`、`past_seq_length`、`current_input_length`、`past_key_cache_*`、`past_value_cache_*`。输出为更新后的 draft KV cache。

DFlash decode 输入：`noise_embedding`、`past_seq_length`、`current_input_length`、`attn_mask`、`past_key_cache_*`、`past_value_cache_*`。输出为 `logits`。

DFlash 图目前以代码实现和推荐 YAML 为依据；若后续实际导出产物的 tensor 名称或 shape 变化，应同步更新本文和 `get_model_docs()`。

## 12. 推理流水线

### 常规 full

```text
prompt/image -> processor -> inputs_embeds
  -> Target Prefill -> logits + caches
  -> Target Decode loop -> logits + updated caches
```

### MTP

```text
Target Prefill -> logits + caches + post_norm_hidden
MTP Draft Prefill -> draft cache
loop:
  MTP Draft Decode x K -> draft tokens
  Target Decode verify -> accepted tokens + updated target cache
```

### DFlash

```text
Target Prefill -> logits + caches + target_hidden
loop:
  DFlash Context -> draft cache from target_hidden
  DFlash Decode -> draft tokens
  Target Decode verify -> accepted tokens + updated target cache + next target_hidden
```

## 13. 配置维护原则

1. 默认配置必须能量化；`quant: null` 只允许通过显式 override 表达 base 验证。
2. YAML 名称只描述拓扑，不描述 w4/w8/gptq/autoround。
3. full、MTP、DFlash、visual-only 尽量复用同一套 quant/export 字段，差异只落在对应 `mtp_config` / `dflash_config` / visual 尺寸。
4. DFlash `target_model_dir` 不写死，运行时跟随当前 `hf_model`。
5. 上游 imodelzoo/customized_models 展示字段说明时，应调用 `get_quant_config_help()`、`get_export_config_help()`、`get_model_docs()`，不要复制一份易过期文档。
