# Merak Model Delivery

`merak_delivery` 提供从 Merak 模型配置到可追踪交付版本的统一流转能力，覆盖：

- 模型卡校验
- 模型量化和 HMONNX 导出
- Golden 数据生成
- `hm_eval` 精度评测
- 交付清单生成
- 导出产物检查
- Release 状态登记
- 模型目录构建
- Release README 生成

该目录只负责编排和记录交付过程。实际量化、导出和 Golden 生成由 `xhmodel_merak` 的 `AutoLLMWorkflow` 执行，精度评测由仓库中的 `hm_eval` 执行。

## 目录结构

```text
merak_delivery/
├── schemas/                     # 模型卡及交付产物 JSON Schema
├── tools/
│   ├── merak_model_flow.py      # 统一 CLI 入口及兼容命令
│   ├── merak_version_flow.py    # run-workflow 独立入口
│   ├── collect_merak_delivery_manifest.py
│   ├── check_merak_artifact.py
│   ├── register_merak_release.py
│   ├── build_model_catalog.py
│   ├── render_merak_readme.py
│   └── core/
│       ├── model_flow.py        # 完整模型流转编排
│       ├── evaluator.py         # hm_eval 接入和报告转换
│       ├── delivery_store.py    # Manifest、Release、Catalog 等交付数据
│       ├── cli.py               # CLI 构建与分发
│       └── bindings.py          # 兼容函数绑定
├── releases/                    # Release 状态示例或登记结果
├── model_catalog/               # 模型目录数据示例或生成结果
├── tests/                       # merak_delivery 内部回归测试
└── docs/                        # 设计和实施文档
```

默认运行产物写入仓库根目录下的 `work_dirs/merak_delivery/`，而不是写入源码目录。
模型卡随模型示例放在 `examples_merak/<类别>/<模型目录>/model_cards/`，
`merak_delivery` 工具默认递归扫描这些 `model_cards/*.yaml`。

## 完整流程

```text
Model Card
    │
    ├── validate
    │
    └── run-workflow
          ├── quant
          ├── export
          ├── dump_golden（可选）
          ├── hm_eval（可选）
          ├── collect-manifest
          ├── check-artifact
          ├── register-release（可选）
          ├── build-catalog（可选）
          └── render-readme（可选）
```

`run-workflow` 的主编排代码位于 `tools/core/model_flow.py` 的 `MerakModelFlow.run()`。

## 快速开始

以下命令均在仓库根目录执行。

### 1. 校验模型卡

校验全部 Merak 模型卡：

```bash
python merak_delivery/tools/merak_model_flow.py validate
```

校验指定模型卡：

```bash
python merak_delivery/tools/merak_model_flow.py validate \
  --root examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml
```

### 2. 查看流转计划

`--dry-run` 不加载模型，也不执行量化和导出，只打印将要执行的流程和输出路径：

```bash
python merak_delivery/tools/merak_model_flow.py run-workflow \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --dry-run
```

建议新增模型后先运行 `validate` 和 `--dry-run`。

### 3. 执行量化和导出

```bash
python merak_delivery/tools/merak_model_flow.py run-workflow \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --dump-golden
```

常用参数：

- `--model-dir`：覆盖模型卡中的 `internal.workflow.model_dir`
- `--work-dir`：覆盖本次流转工作目录
- `--device`：覆盖模型卡中的运行设备
- `--bits`：覆盖量化 bit 数
- `--skip-quant`：将 workflow quant stage 设为 `None`
- `--quant-output-dir`：指定量化输出目录
- `--export-output-dir`：指定导出目录
- `--dump-golden`：导出后执行 Golden Dump
- `--golden-prompt`：设置文本 Golden Prompt
- `--debug`：启用 workflow debug 模式

### 4. 在完整流程中执行精度评测

```bash
python merak_delivery/tools/merak_model_flow.py run-workflow \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --dump-golden \
  --run-eval \
  --eval-backend hmonnx \
  --eval-datasets ceval mmlu_pro
```

默认行为：

- `eval_model` 默认使用模型卡中的 `external.model.id`
- 数据集默认仅使用 `ceval`
- 数据集 Hub 默认使用 `modelscope`，可通过 `--eval-dataset-hub huggingface` 覆盖
- HMONNX meta 默认使用本次导出生成的 `golden_meta_info.json`
- HMONNX 标准报告默认写入 `<work_dir>/eval_report.json`
- Float 标准报告默认写入 `<work_dir>/eval_report_float.json`，不会覆盖 HMONNX 报告
- `hm_eval` 原始结果默认写入 `<work_dir>/hm_eval/`

快速抽样验证可增加：

```bash
--eval-limit 10 --eval-max-tokens 32
```

`--eval-limit 0` 表示使用完整数据集。正式精度验收不应使用抽样结果代替全量结果。

### 5. 执行完整交付登记

```bash
python merak_delivery/tools/merak_model_flow.py run-workflow \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --dump-golden \
  --run-eval \
  --eval-backend hmonnx \
  --eval-datasets ceval mmlu_pro \
  --register-release \
  --build-catalog \
  --render-readme
```

只有产物检查通过后，流程才会继续执行 Release、Catalog 和 README 阶段。

## 独立运行精度评测

如果 HMONNX 已经导出，可以不重复执行 quant/export，单独运行评测：

```bash
python merak_delivery/tools/merak_model_flow.py run-eval \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --eval-backend hmonnx \
  --eval-datasets ceval mmlu_pro \
  --hmonnx-meta work_dirs/path/to/golden_meta_info.json
```

也可以直接通过 `hm_eval` 的 Float backend 评估原始浮点模型。模型权重目录复用
`hm_eval/model_configs/<model_id>.yaml` 中的 `hf_model_dir`，无需额外传入模型目录：

```bash
python merak_delivery/tools/merak_model_flow.py run-eval \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --eval-backend float
```

未显式传入 `--eval-datasets` 时默认执行完整 `ceval`；Float 标准报告默认写入
`<work_dir>/eval_report_float.json`。可通过 `--output` 指定其他报告路径。

多模态模型可以额外传入视觉子图 meta：

```bash
--vision-hmonnx-meta work_dirs/path/to/vision/export_meta_info.json
```

评测输出遵循 `schemas/merak_eval_report.schema.json`：

```json
{
  "schema_version": 1,
  "model_id": "qwen3_0_6b",
  "version_id": "qwen3_0_6b_xh2a_w8a16_256_2k_20260708",
  "status": "passed",
  "tasks": [],
  "logs": {}
}
```

当所有数据集的 `hm_eval` 状态均为 `completed` 时，报告状态为 `passed`；任一数据集失败或没有有效任务时，报告状态为 `failed`。

## 分阶段工具

统一入口的每个交付阶段也可以独立执行。

### 收集 Manifest

```bash
python merak_delivery/tools/collect_merak_delivery_manifest.py \
  --model-card examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml \
  --work-dir work_dirs/merak_delivery/qwen3_0_6b/example \
  --export-dir work_dirs/path/to/export \
  --golden-meta work_dirs/path/to/golden_meta_info.json \
  --eval-report work_dirs/path/to/eval_report.json
```

### 检查导出产物

```bash
python merak_delivery/tools/check_merak_artifact.py \
  --manifest work_dirs/merak_delivery/qwen3_0_6b/example/delivery_manifest.json
```

### 注册 Release

```bash
python merak_delivery/tools/register_merak_release.py \
  --manifest work_dirs/path/to/delivery_manifest.json \
  --artifact-check work_dirs/path/to/artifact_check.json \
  --eval-report work_dirs/path/to/eval_report.json
```

### 构建 Catalog

```bash
python merak_delivery/tools/build_model_catalog.py
```

### 渲染模型 Release README

```bash
python merak_delivery/tools/render_merak_readme.py \
  --model-id qwen3_0_6b
```

## 模型卡

模型卡是完整流转的唯一输入定义。当前示例：

- `examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml`
- `examples_merak/llm/qwen3_5/model_cards/qwen3_5_9b.yaml`
- `examples_merak/llm/gemma4_series/model_cards/gemma4_e4b.yaml`

模型卡统一放在对应示例目录下的 `model_cards/` 子目录。一个基础模型只对应一张卡片；
同一模型尺寸存在多个 workflow 时优先选择标准 `full` 配置作为主配置，GPTQ、MTP、
DFlash、no-quant、visual-only 等技术变体不会生成重复的前端模型卡。

可以从 `examples_merak` 中实际引用的 workflow YAML 自动生成卡片草稿：

```bash
python merak_delivery/tools/generate_merak_model_cards.py --dry-run
python merak_delivery/tools/generate_merak_model_cards.py
```

生成器只扫描源码、README 和 YAML，不导入模型代码，也不运行 quant/export。
默认不覆盖已有人工卡片；需要重建自动卡片时显式使用 `--overwrite`。扫描报告写入
`work_dirs/merak_delivery/card_generation_report.json`。

自动生成适合填充 config 路径、模型族、目标设备、量化格式、组件列表和 YAML 中
显式声明的 IO。模型参数量、计算量和正式精度没有可靠来源时必须保留
`status: missing` 或空的对比列表，不能依靠文件名猜测：

Catalog 会区分两层量化口径：顶层 `quant.bits` 表示 weight-only 权重位宽，
`export.*.quant_type` 表示 HMONNX 导出模板。列表中的“交付量化”会将前者的权重位宽
与后者的激活位宽组合显示（例如 `quant.bits: 4` + `w8a8h1_sefp` 显示为
`W4A8`）；详情仍保留完整的 HMONNX 导出模板，避免把 W4A8 链路误标为 W8A8。

- 原始浮点模型来源、URL 和 license
- 实际可用的模型目录
- Merak 量化模型发版路径/下载链接
- 正式的量化前后精度对比
- 业务级输入输出含义，以及 YAML 未声明的子模型 IO

schema v2 除 `schema_version` 外只有两个业务顶层模块：

- `external`（对外信息）：模型公共标识、参数量/激活参数量、计算量、浮点与量化精度对比、负责人。
- `internal`（对内信息）：模型类型和模态、来源、组件、输入输出语义与类型、workflow、运行和发版配置。

前端列表与详情页均按这两个模块展示；子模型/计算图属于 `internal.components`，
整体业务输入输出属于 `internal.io`。

主要字段：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | 模型卡 schema 版本，当前为 `2` |
| `external.model` | 模型 ID、家族和显示名 |
| `external.parameters` | 总参数量、激活参数量、单位和确认状态 |
| `external.compute` | 输入维度、Prefill/Decode 计算量、统一单位、确认状态和可选计算口径备注 |
| `external.precision` | 浮点格式、量化格式及浮点/量化精度对比证据 |
| `external.owner` | 对外负责人 |
| `internal.model` | 模型类型、模态、任务和标签 |
| `internal.source` | 原始模型来源及本地权重路径 |
| `internal.components` | 主模型及子模型/计算图的类型、精度和 IO |
| `internal.io` | 整体输入输出名称、含义、类型、dtype 和 shape |
| `internal.workflow` | Merak workflow 配置、模型目录、执行类和阶段 |
| `internal.runtime` | 设备、随机种子和默认工作目录 |
| `internal.release` | 版本、目标状态、Reviewer 和量化模型发版信息 |
| `internal.benchmark` | benchmark 脚本覆盖、运行方式、指标门限、结果文件和版本匹配状态 |

参数量与计算量按以下统一口径维护：

- `parameters.total` 记录卡片主模型的逻辑参数个数；共享/绑定权重只计一次，优化器状态、重复保存的替代 checkpoint 和不属于主模型的辅助资产不计入。
- dense 模型的 `parameters.active` 等于 `total`；MoE 优先采用官方 active 口径，没有官方值时才按 checkpoint expert 张量与 top-k 配置推导。多运行分支无法用一个 active 数表达时保留 `null`。
- checkpoint 或官方上游模型卡可直接确认的值标为 `verified`；架构静态计算、同族模型继承或近似口径标为 `inferred`；没有可靠证据继续标为 `missing`，不只按模型名猜数。
- `compute` 保留 `input_shape`、`prefill`、`decode`、`unit`、`status` 和可选 `note`。当前文本模型统一使用 `[batch, sequence] = [1, 2048]` 的输入维度和 `MAC=2 FLOPs` 静态 profiler 口径。
- 文本自回归模型中，`prefill` 是整段 2048-token 输入的计算量，`decode` 是复用同一 2048-token KV Cache 生成 1 个新 token 的计算量；两者统一使用 `TFLOPs`。非自回归模型则将绑定输入场景的一次完整前向或端到端推理计算量写入 `prefill`。
- 没有自回归 Decoder 阶段的模型，将绑定输入场景的一次完整前向或端到端推理计算量记录在 `prefill`，并令 `decode: null`；多组件模型必须用 `note` 说明包含的组件、分辨率/帧数/步数和未计入项。
- 这里记录的是理论计算量，不表示芯片峰值算力。真机时延、利用率和吞吐应进入 benchmark 或 profiler 证据。

默认量化格式按参数量分档：有明确特殊配置的模型保留原值；其余总参数量小于 7B 的模型使用 `W8A8`，7B 及以上使用 `W4A8`。模型卡中的 `quantized_format` 继续携带 H0/H1、SEFP/SSFP 等实际导出后缀。

`sync_merak_benchmarks.py` 只读取 benchmark 源码和已存在的结果文件，不导入或执行
测试模块。脚本存在、导出成功和测试门限都不会自动写成正式精度；只有模型版本、
量化配置和结果证据均匹配时才标记为实测。可单独刷新 benchmark 元数据：

```bash
python merak_delivery/tools/sync_merak_benchmarks.py
```

当前校验要求 `internal.workflow.class` 为 `auto`，且阶段顺序为：

```yaml
actions: [quant, export, dump_golden, eval]
```

新增模型时应复制同系列模型卡并修改字段，不要直接修改 JSON Schema 来绕过校验。

## 交付产物

一次完整流转通常生成：

```text
<work_dir>/
├── quant/                       # 量化模型
├── export/                      # HMONNX 导出结果
├── hm_eval/                     # hm_eval 原始结果
├── eval_report.json             # 标准精度评测报告
├── delivery_manifest.json       # 交付清单
└── artifact_check.json          # 产物检查报告

work_dirs/merak_delivery/
├── releases/merak/<model_id>/   # Release 状态
└── model_catalog/data/          # Catalog JSON 数据
```

主要数据契约：

| Schema | 产物 |
| --- | --- |
| `schemas/merak_model_card.schema.json` | 模型卡 |
| `schemas/merak_delivery_manifest.schema.json` | `delivery_manifest.json` |
| `schemas/merak_artifact_check.schema.json` | `artifact_check.json` |
| `schemas/merak_eval_report.schema.json` | `eval_report.json` |
| `schemas/merak_release_state.schema.json` | Release YAML |
| `schemas/merak_catalog.schema.json` | Catalog JSON |

## Release 门禁

Release 状态由以下门禁共同决定：

| Gate | 说明 |
| --- | --- |
| `metadata_valid` | 模型卡信息是否存在 |
| `workflow_exported` | 是否生成交付 Manifest |
| `golden_valid` | Golden meta 是否存在 |
| `artifact_valid` | HMONNX 产物检查是否通过 |
| `accuracy_valid` | `eval_report.status` 是否为 `passed` |
| `compiler_valid` | 编译器验收状态，当前默认 `pending` |

状态优先级：

```text
failed gate -> blocked
accuracy_valid passed -> accuracy_passed
artifact_valid passed -> artifact_passed
golden_valid passed -> golden_ready
workflow_exported passed -> exported
otherwise -> ready
```

## 开发与测试

运行 `merak_delivery` 内部测试：

```bash
python -m pytest merak_delivery/tests -q
```

运行交付全流程兼容测试：

```bash
python -m pytest tests/test_merak_model_delivery_full_flow.py -q
```

检查 CLI：

```bash
python merak_delivery/tools/merak_model_flow.py --help
python merak_delivery/tools/merak_model_flow.py run-workflow --help
python merak_delivery/tools/merak_model_flow.py run-eval --help
```

修改代码时应保持：

1. 原有子命令、参数名和退出码兼容。
2. 核心业务类位于 `tools/core/`。
3. `tools/merak_model_flow.py` 保留统一入口和阶段函数兼容层。
4. 新测试优先放在 `merak_delivery/tests/`。
5. 不在源码目录写入大型模型、量化产物或 HMONNX 文件。
