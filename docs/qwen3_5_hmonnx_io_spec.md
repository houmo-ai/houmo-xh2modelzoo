# Qwen3.5 / Qwen3.6 Merak 模型与 HMONNX 图规格

本文是 Qwen3.5 / Qwen3.6 Merak workflow 的模型文档。它补充 README 中的快速入口，说明推荐 YAML、量化/导出字段、MTP/DFlash/Visual 支持范围，以及导出后 HMONNX 图在推理中的数据流。

## 1. 支持范围

| 模型 | family | model_size | HF 默认路径 | 推荐 YAML 变体 |
| --- | --- | --- | --- | --- |
| Qwen3.5-9B | `qwen3_5` | `9b` | `weights/Qwen3.5-9B` | full / mtp / dflash / visual_only_token_gears |
| Qwen3.6-27B | `qwen3_5` | `27b` | `weights/Qwen3.6-27B` | full / mtp / dflash / visual_only_token_gears |
| Qwen3.6-35B-A3B | `qwen3_5_moe` | `35b_a3b` | `weights/Qwen3.6-35B-A3B` | full / mtp / dflash / visual_only_token_gears |

如果从已经量化好的 HF 模型目录导出，应直接把 `hf_model_dir` 指向该目录，并将 `quant` 设置为 `null`，让 workflow 跳过量化阶段。

## 2. 推荐调用方式

```python
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

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

上游不需要重新维护 AutoRound、GPTQModel、`quant_scheme`、视觉尺寸、MTP/DFlash 细节。普通用户选择推荐 YAML；进阶用户通过 `config_overrides` 修改字段。

## 3. 推荐量化配置

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

约束：`group_size` 必须为 64；GPTQ 伴随配置使用 `_gptq.yaml` 后缀并设置 `method: gptq`。`quant: null` 表示跳过量化，可直接写在 YAML 中，也可通过 `config_overrides={"quant": None}` 覆盖。

## 4. 推荐 YAML 命名

YAML 名称只表达拓扑/导出形态，不表达量化格式。不要把 `w4`、`w8`、`gptq`、`autoround` 写进 workflow YAML 文件名。

推荐路径：

```text
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_visual_only_token_gears_99p_candidate.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/*.yaml
configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/*.yaml
```

## 5. Quant 流程

### 5.1 默认 AutoRound → GPTQModel HF

默认 `quant()` 运行 AutoRound，并保存 GPTQModel 兼容 HF 目录。`QuantResult` 会记录：

| 字段 | 说明 |
| --- | --- |
| `hf_model_dir` | 原始 HF 模型路径 |
| `quanted_model_dir` | 量化后 HF/GPTQModel 目录 |
| `skipped` | base 验证时为 True |

### 5.2 使用已量化 HF 模型目录

```python
workflow = AutoLLMWorkflow.from_config(
    hf_model_dir="/path/to/quanted_hf_model",
    config_path="configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml",
)
quant_result = workflow.quant(..., config_overrides={"quant": None})
```

该方式同样适用于 MTP 和 DFlash 导出。导出形态仍然由 `config_path` 指向的 YAML 决定。

### 5.3 base 验证

```python
quant_result = workflow.quant(..., config_overrides={"quant": None})
```

这会跳过量化并直接使用 base HF 模型。等价地，也可以在 workflow YAML 中写 `quant: null`。

## 6. Export 变体

| variant | 说明 | 关键字段 |
| --- | --- | --- |
| `full` | LLM prefill/decode；YAML 带 `visual_config` 时包含视觉分支配置 | `context_max_length=2048`, `prefill_chunk_length=256` |
| `mtp` | full + MTP draft 模型 | `spec_decode_mode=mtp`, `num_draft_tokens=4`, `output_post_norm_hidden=true`, `mtp_config` |
| `dflash` | full + DFlash draft 模型 | `spec_decode_mode=dflash`, `num_draft_tokens=9`, `output_hidden_state_indices`, `dflash_config` |
| `visual_only` | 只导出视觉塔 | `model_type=*_visual`, `visual_input_mode=patches`, `image_token_gears` |

所有 Qwen3.5/Qwen3.5-MoE workflow YAML 默认同时启用两项 GDR fuse：

```python
config_overrides={
    "export.model.fuse_gdr_ops": True,
    "export.model.fuse_gdr_block_recurrent_ops": True,
}
```

## 7. Visual 支持

full YAML 和 visual-only YAML 统一使用多档 patch-token 输入：

```yaml
visual_config:
  visual_input_mode: patches
  image_token_gears: [96, 196, 384, 704, 1536]
  image_token_capacity: 1536
  spatial_merge_size: 2
  quant_scheme:
    quant_type: w8a8h1_sefp
    ops: {}
```

不再支持固定图像宽高或 `visual_input_mode: image`。Host 侧把图像转换为 flattened
patches，根据 post-merge token 数按 `smallest_fit` 路由到最小可容纳档位；超过
1536 image tokens 的输入会明确拒绝。

完整导出包名使用 `visualm96_196_384_704_1536` 标记视觉档位，例如
`hmquant_xh2_qwen3_8_27b_w4a8_256_256k_mpe256k_visualm96_196_384_704_1536_20260825`。
五档图位于包根目录的 `visual_m96`、`visual_m196`、`visual_m384`、`visual_m704`、
`visual_m1536`，与 `prefill`、`decode` 同级，不再嵌套在 `visual/` 下。根目录的
`visual_gears.json` 和 `golden_meta_info.json` 记录运行时路由信息。
visual-only 导出使用相同的五档目录（根目录为 `visual_m*`），并额外写出
`visual_meta_info.json` 作为独立加载入口以及 `visual_gears.json` 作为路由清单。
发布布局校验与 step 链接修复只支持根目录 `visual_m*`；不兼容旧 `visual/m*`
或 `m*` 布局，旧格式产物需重新导出。

golden 生成固定覆盖全部五档，每档执行满 capacity 的 deterministic patches，产物写入
对应 `step_0`。完整模型随后还会用真实图片验证实际选档、visual 输出截断、prefill 和
decode，命中档位的真实图片 golden 单独写入 `step_1`，不会覆盖逐档基准；visual-only 的 quick test 则逐档执行并报告
`m96=(1,96,H)` ... `m1536=(1,1536,H)`，不依赖文本模型 metadata。

## 8. MTP 数据流

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

## 9. DFlash 数据流

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

## 10. HMONNX 图 IO 概览

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

DFlash 图目前以代码实现和推荐 YAML 为依据；若后续实际导出产物的 tensor 名称或 shape 变化，应同步更新本文。

## 11. 推理流水线

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

## 12. 配置维护原则

1. `quant: null` 表示跳过量化；推荐量化配置仍应显式写出 quant 字段，base 验证可以在 YAML 中写 `quant: null` 或通过 `config_overrides={"quant": None}` 覆盖。
2. YAML 名称只描述拓扑，不描述 w4/w8/gptq/autoround。
3. full、MTP、DFlash、visual-only 尽量复用同一套 quant/export 字段，差异只落在对应 `mtp_config` / `dflash_config` / visual 尺寸。
4. DFlash `target_model_dir` 不写死，运行时跟随当前 `hf_model`。
5. 上游 imodelzoo/customized_models 展示字段说明时，应以本文件和推荐 YAML 为准，避免重新维护一套易过期字段表。
