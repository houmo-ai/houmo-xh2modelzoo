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
├── model_cards/                 # 每个模型的交付定义
│   └── merak/
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
  --root merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml
```

### 2. 查看流转计划

`--dry-run` 不加载模型，也不执行量化和导出，只打印将要执行的流程和输出路径：

```bash
python merak_delivery/tools/merak_model_flow.py run-workflow \
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
  --dry-run
```

建议新增模型后先运行 `validate` 和 `--dry-run`。

### 3. 执行量化和导出

```bash
python merak_delivery/tools/merak_model_flow.py run-workflow \
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
  --dump-golden
```

常用参数：

- `--model-dir`：覆盖模型卡中的 `workflow.model_dir`
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
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
  --dump-golden \
  --run-eval \
  --eval-backend hmonnx \
  --eval-datasets ceval mmlu_pro
```

默认行为：

- `eval_model` 默认使用模型卡中的 `model.id`
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
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
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
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
  --eval-backend hmonnx \
  --eval-datasets ceval mmlu_pro \
  --hmonnx-meta work_dirs/path/to/golden_meta_info.json
```

也可以直接通过 `hm_eval` 的 Float backend 评估原始浮点模型。模型权重目录复用
`hm_eval/model_configs/<model_id>.yaml` 中的 `hf_model_dir`，无需额外传入模型目录：

```bash
python merak_delivery/tools/merak_model_flow.py run-eval \
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
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
  --model-card merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml \
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

- `model_cards/merak/qwen3/qwen3_0_6b.yaml`
- `model_cards/merak/qwen3_5/qwen3_5_9b.yaml`
- `model_cards/merak/gemma4_series/gemma4_e4b.yaml`

主要字段：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | 模型卡 schema 版本，当前为 `1` |
| `model` | 模型 ID、家族、显示名、模态、任务和标签 |
| `source` | 原始模型来源及本地权重路径 |
| `workflow` | Merak workflow 配置、模型目录、执行类和阶段 |
| `runtime` | 设备、随机种子和默认工作目录 |
| `frontend` | 输入输出、演示信息和限制说明 |
| `release` | 版本 ID、目标状态、Owner 和 Reviewer |

当前校验要求 `workflow.class` 为 `auto`，且阶段顺序为：

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
