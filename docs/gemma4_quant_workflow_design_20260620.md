# Gemma4 Series Quant Workflow 设计（QTL-384）

## 目标

Gemma4 E4B、31B、26B-A4B 继续走统一 `gemma4_series` workflow，quant 阶段只暴露稳定的宏观配置：

- `algorithm: gptqmodel`：默认主路径，输出 GPTQModel HF artifact。
- `algorithm: autoround, preset: mode1`：薄封装 AutoRound 官方维护脚本，按模型 topology 自动选择 dense/MoE 脚本。
- workflow 不复刻量化内部细节，只负责：topology 识别、GPTQModel 包内默认校准集、MoE bypass、CLI/recipe 参数翻译、artifact contract 校验。

## GPTQModel 默认策略

| 模型 | topology | 默认校准集 | MoE routing |
| --- | --- | --- | --- |
| E4B | dense | `gptqmodel://quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl` | 不启用 |
| 31B | dense | `gptqmodel://quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl` | 不启用 |
| 26B-A4B | moe | `gptqmodel://quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl` | 默认 `bypass` |

兼容规则：

- 如果用户显式配置 `calibration.jsonl` / `calibration.calibration_jsonl`，优先使用显式路径。
- 如果历史配置还写着 `dataset: wikitext`，workflow 会按 topology 自动替换成上述 GPTQModel package resource JSONL；preflight 会在 GPTQModel 不可 import、资源不存在或显式路径不可读时给出明确错误，避免量化中途崩溃。
- 如果用户显式配置非 wikitext 的 `calibration.dataset`，保持用户配置，不强行替换。
- 26B-A4B 未显式配置 `quant.moe.routing` 时默认补 `bypass`，保证专家能收到校准激活。

## AutoRound mode1 封装

同一个 workflow 配置入口：

```yaml
quant:
  algorithm: autoround
  preset: mode1
  artifact_format: gptqmodel_hf
  bits: 4
  group_size: 64
```

内部按 HF config/topology 分派：

| topology | script | 关键参数 |
| --- | --- | --- |
| dense | `third_party/auto-round/scripts_gemma4/quantize.py` | `--mode llm-only --llm_bits 4 --llm_group_size 64 --format auto_gptq` |
| moe | `third_party/auto-round/scripts_gemma4_moe/quantize_moe.py` | `--llm_bits 4 --llm_group_size 64 --format auto_gptq`，默认使用 EBSS JSONL，按 512 seqlen 校准，不做拼接 |

提供两个模板函数：

- `dump_autoround_mode1_quant_config_template()`：dense mode1 推荐模板。
- `dump_autoround_moe_mode1_quant_config_template()`：26B-A4B MoE mode1 推荐模板。

## GPTQModel 侧薄封装要求

`gptqmodel.recipes.gemma4.quantize_gemma4(..., moe={"routing": "bypass"})` 会转成：

```bash
python examples/quantization/examples/example_gemma4.py \
  ... \
  --calibration-jsonl <EBSS_JSONL> \
  --moe-routing bypass
```

`example_gemma4.py` 中再创建 `MoEConfig(routing=ExpertsRoutingBypass(...))`，不在 xh2modelzoo 内部直接依赖 GPTQModel 的 MoE 生命周期实现。

## 保持简练的边界

- xh2modelzoo 只做配置归一化与脚本/recipe 分派。
- GPTQModel 继续拥有 GPTQ load/quantize/save、MoE bypass 生命周期、artifact patch。
- AutoRound 脚本继续拥有 dense/MoE 量化实现与保存后的 reload/generate 校验。
- 新增配置不拆出 `gemma4_moe` workflow；26B-A4B 仍由 `gemma4_series` 统一 API 根据 config 自动识别。
