# HMONNX Workflow 使用与开发指南

`xhmodel_merak.xh_llm.workflows` 是 `xh_llm` SDK 的用户侧编排层，用于把量化、HMONNX 导出和 golden 生成流程封装成稳定 API，方便 examples 和外部仓库复用。

Workflow 不替代底层模型实现。模型注册、模型加载、HMONNX 导出、HMONNX 推理仍然由 `AutoLLMConfig`、`AutoLLMModel`、`AutoLLMHONNXModel` 和 `xhmodel_merak.xh_llm.models` 下的具体模型类完成。

## 当前能力边界

- 当前统一入口是 `AutoLLMWorkflow.from_config()`。
- 当前必须显式传入 `config_path`，尚未实现 `from_pretrained(hf_model_dir)` 自动选择 YAML 配置。
- 当前没有 workflow CLI；后续 CLI 可以直接复用 `AutoLLMWorkflow.from_config()`。
- 旧的 `xhmodel_merak.xh_llm.workflows.models.*` 路径已移除，不再兼容旧 import。
- workflow YAML 当前仍放在仓库顶层 `configs_merak/workflows/...` 下。

## 设计原则

- 统一入口：调用方优先使用 `AutoLLMWorkflow.from_config()`，不直接 import 具体 workflow 类。
- 复用模型注册：根据 YAML 中的 `export.model.model_type` 复用 `xh_llm` 现有 lazy import 和 registry 机制。
- 模型目录收敛：具体 workflow 子类放在对应 `xhmodel_merak/xh_llm/models/<model_name>/workflow.py` 中。
- 避免重依赖 eager import：`workflows/__init__.py` 不导入具体模型 workflow；只有解析到具体模型后才按需导入 workflow 模块。
- 配置字段驱动扩展：模型结构相同但 workflow 行为不同的场景，通过 `quant` 或 `export` 下的字段做简单分发，不新增顶层 `workflow` 字段。
- `BaseHMONNXWorkflow` 仍然是唯一 workflow 基类，不新增 `GenericHMONNXWorkflow`。

## 目录结构

公共 workflow 模块：

```text
xhmodel_merak/xh_llm/workflows/
  __init__.py
  auto.py
  base.py
  config.py
  result.py
  utils.py
```

具体模型 workflow：

```text
xhmodel_merak/xh_llm/models/qwen2_vl/
  workflow.py

xhmodel_merak/xh_llm/models/qwen3_legacy/
  workflow.py
```

YAML 配置：

```text
configs_merak/workflows/<chip_arch>/llm_models/<model_name>/...
```

示例脚本：

```text
examples_merak/llm/<model_name>/*_workflow.py
```

## AutoLLMWorkflow

推荐使用 `AutoLLMWorkflow.from_config()` 创建 workflow：

```python
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow


workflow = AutoLLMWorkflow.from_config(
    hf_model_dir="/path/to/hf_model",
    config_path="./configs_merak/workflows/xh2a/llm_models/qwen2_vl/2b/qwen2_vl_2b_xh2a_4k.yaml",
    seed=1024,
    debug=False,
)
```

解析流程：

1. 读取 `WorkflowConfig`。
2. 使用传入的 `hf_model_dir` 临时填充 `export.model.hf_model`。
3. 根据 `export.model.model_type` 调用 `get_model_class()`，复用现有模型 registry 和 lazy import。
4. 读取模型类上的 `WORKFLOW_CLS` 字符串。
5. 懒加载并实例化对应 workflow 类。
6. 如果模型类没有 `WORKFLOW_CLS`，回退到 `BaseHMONNXWorkflow`。

`WORKFLOW_CLS` 写在模型类上，格式为 `"module.path:ClassName"`：

```python
@register_llm_model("Qwen2VLForConditionalGeneration")
class XHQwen2VLModel(VisionLLMModel):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen2_vl.workflow:XHQwen2VLHMONNXWorkflow"
```

使用字符串而不是直接引用类，是为了避免普通 `AutoLLMModel` 路径 import 模型类时顺带加载 workflow 及其可选依赖。

## BaseHMONNXWorkflow

`BaseHMONNXWorkflow` 是所有 workflow 的基类，同时保留默认 `quant()` 和 `export()` 实现。

构造参数：

- `hf_model_dir`：原始 HF 模型目录，会被规范化成绝对路径字符串。
- `config_path`：workflow YAML 配置路径。
- `seed`：导出时传给 `set_random_seed()`。
- `debug`：传给 `xhquant_init()`。

主要方法：

- `quant(output_dir, device, config_overrides=None) -> QuantResult`
- `export(quant_result, output_dir, device, config_overrides=None) -> ExportResult`
- `dump_golden(export_result, device, input_messages) -> str`

基类默认 `quant()` 只支持：

```yaml
quant: null
```

这种情况下返回 `skipped=True` 的 `QuantResult`。如果 YAML 中存在非空 `quant` 配置，具体 workflow 子类必须实现自己的量化逻辑。

基类 `export()` 适用于普通单模型导出流程：

1. 应用本次调用的 `config_overrides`。
2. 根据 `QuantResult` 决定导出使用原始 HF 模型还是量化 HF 模型。
3. 构造兼容旧导出链路的 export config。
4. 创建输出目录，初始化日志和 seed。
5. 通过 `AutoLLMConfig` 和 `AutoLLMModel` 创建模型。
6. 校验解析出的模型 config 类名和模型类名。
7. 调用 `xh_model.export_hmonnx(output_dir)`。
8. 返回 `ExportResult`。

基类还提供 `_find_golden_meta_file(export_result)`，供具体 workflow 的 `dump_golden()` 复用。

## WorkflowConfig

Workflow YAML 必须包含两个顶层字段：

```yaml
quant: null

export:
  model:
    chip_arch: XH2a
    model_type: Qwen2VLForConditionalGeneration
    hf_model: null
    model_name: xh2_qwen2-vl-2b_w8a8_256_4k
    context_max_length: 4096
    prefill_chunk_length: 256
    use_cache: true
    num_logits_to_keep: 1
    quant_scheme:
      quant_type: w8a8h0_sefp
      ops: {}
```

`export.model` 至少需要：

- `chip_arch`
- `model_type`
- `hf_model`

`export.model.hf_model` 在 YAML 中通常写为 `null`，真实路径由 `AutoLLMWorkflow.from_config(hf_model_dir=...)` 和 `QuantResult` 在运行时注入。

`WorkflowConfig.build_export_dict(export_hf_model_dir)` 会深拷贝 `export`，并把 `export.model.hf_model` 覆盖为真实路径。`export` 下的其他字段会保持原样传给旧导出链路。

## config_overrides

`config_overrides` 只对当前 `quant()` 或 `export()` 调用生效，不修改 workflow 对象中保存的原始配置。

示例：

```python
CONFIG_OVERRIDES = {
    "export.model.context_max_length": 8192,
    "export.model.prefill_chunk_length": 512,
    "export.model.chip_arch": "XH2a",
}
```

覆盖规则：

- 路径从 YAML 顶层开始写。
- 只能覆盖已存在字段，不能新增字段。
- mapping 覆盖 mapping 时，会递归检查子字段是否已存在。
- 应用覆盖后会重新校验 workflow YAML 的基础结构。

## QuantResult 和 ExportResult

`QuantResult` 描述量化阶段产物：

```python
QuantResult(
    hf_model_dir="/path/to/original_hf_model",
    skipped=False,
    quanted_model_dir="/path/to/quanted_hf_model",
    is_quant_weight_format=False,
)
```

字段含义：

- `hf_model_dir`：原始 HF 模型路径，必须和 workflow 初始化时的 `hf_model_dir` 指向同一路径。
- `skipped`：量化是否跳过。跳过时导出使用原始 HF 模型路径。
- `quanted_model_dir`：量化后 HF 模型目录。未跳过量化时必须提供。
- `is_quant_weight_format`：预留字段，当前 workflow 通常不使用。

`ExportResult` 描述导出阶段产物：

```python
ExportResult(
    work_dir="./work_dirs/qwen2_vl_workflow_export",
    config_file="./work_dirs/qwen2_vl_workflow_export/qwen2_vl_2b_xh2a_4k.yaml",
    meta=meta,
)
```

字段含义：

- `work_dir`：导出工作目录，包含 log、实际使用的 workflow config 副本和 HMONNX 产物目录。
- `config_file`：导出时 dump 的 workflow YAML 文件。
- `meta`：底层 `export_hmonnx()` 返回的元信息对象。

## 新增模型 Workflow

新增模型 workflow 时，把文件放在模型目录：

```text
xhmodel_merak/xh_llm/models/<model_name>/workflow.py
```

最小实现：

```python
from ...workflows.base import BaseHMONNXWorkflow


class XHNewModelHMONNXWorkflow(BaseHMONNXWorkflow):
    expected_model_config_cls_name = "XHNewModelConfig"
    expected_model_cls_name = "XHNewModel"
```

然后在对应模型类上声明：

```python
class XHNewModel(...):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.new_model.workflow:XHNewModelHMONNXWorkflow"
```

只有在需要模型专属量化、导出后处理或 golden 输入构造时才覆盖方法。

注意：

- 不要在 `xhmodel_merak/xh_llm/models/__init__.py` 中导入具体 workflow。
- 不要在单模型 `__init__.py` 中 eager import workflow。
- 量化、processor、datasets、gptqmodel 等重依赖应放在方法内部 import。

## Qwen2-VL Workflow

`XHQwen2VLHMONNXWorkflow` 位于：

```text
xhmodel_merak/xh_llm/models/qwen2_vl/workflow.py
```

它在基类 `export()` 之后检查 `export.visual_buckets`：

```python
visual_buckets_cfg = workflow_config.export.get("visual_buckets")
if visual_buckets_cfg is not None:
    ...
```

没有 `visual_buckets` 时就是普通 Qwen2-VL 导出。

`dump_golden()` 的 `input_messages` 约定：

```python
{
    "image": "./data/images/qwen2_vl_demo.jpeg",
    "text": "描述这张图片",
}
```

## MinerU / visual_buckets

MinerU 2.5 与 Qwen2-VL 模型结构相同，因此仍然使用：

```yaml
export:
  model:
    model_type: Qwen2VLForConditionalGeneration
    ...
  visual_buckets:
    model:
      chip_arch: XH2a
      model_type: Qwen2VLForConditionalGeneration_visual
      patch_size: 14
      quant_scheme:
        quant_type: w8a16h0_ssfp
        ops: {}
    buckets:
      - max_size_h: 140
        max_size_w: 392
```

存在 `export.visual_buckets` 时，Qwen2-VL workflow 会在主模型导出完成后额外导出静态 visual bucket，并写出：

```text
mineru_visual_buckets.json
```

这类行为通过配置字段触发，不需要新增 MinerU 模型类，也不需要新增顶层 `workflow` 字段。

## Qwen3 Legacy Workflow

`XHQwen3LegacyHMONNXWorkflow` 位于：

```text
xhmodel_merak/xh_llm/models/qwen3_legacy/workflow.py
```

量化逻辑：

- `quant: null`：跳过量化，直接导出原始 HF 模型。
- `quant` 非空：读取 `quant.bits`，使用 `GPTQModel` 量化，并返回 `quanted_model_dir`。

示例：

```yaml
quant:
  bits: 4
```

`dump_golden()` 的 `input_messages` 支持字符串或 `{"text": ...}`：

```python
"你多大了？用中文回答。"
```

或：

```python
{"text": "你多大了？用中文回答。"}
```

## 示例

Qwen2-VL：

```bash
python examples_merak/llm/qwen2_vl/qwen2_vl_workflow.py
```

MinerU 2.5：

```bash
python examples_merak/llm/mineru2.5/mineru2_5_workflow.py
```

Qwen3 Legacy：

```bash
python examples_merak/llm/qwen3_legacy/qwen3_legacy_workflow.py
```

这些示例均通过 `AutoLLMWorkflow.from_config()` 创建 workflow。

## 后续可扩展方向

以下能力当前尚未实现：

- `AutoLLMWorkflow.from_pretrained(hf_model_dir)` 自动选择 workflow YAML。
- 将内置 workflow YAML 移入 `xhmodel_merak` 包内，供外部安装包稳定访问。
- workflow CLI。
- 结构化配置说明接口。

如果实现自动选择 YAML，建议不要依赖当前仓库顶层 `configs_merak/workflows` 路径，而是把可自动选择的 YAML 放到 `xhmodel_merak` 包内部，并使用 `importlib.resources` 查找。
