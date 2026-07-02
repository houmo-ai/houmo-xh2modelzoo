# HMONNX Workflow 使用与开发指南

本文面向新增或维护 LLM workflow 的开发者，说明 `xhmodel_merak.xh_llm.workflows` 的设计、核心类、子类开发规范和典型实现方式。

Workflow 是 `xh_llm` 的流程编排层，负责把量化、HMONNX 导出、golden 生成组织成稳定的 Python API。底层模型注册、模型配置解析、HMONNX 导出和 HMONNX 推理仍由 `AutoLLMConfig`、`AutoLLMModel`、`AutoLLMHONNXModel` 以及各模型目录下的具体模型类负责。

## 1. Workflow 设计理念

### 目标

Workflow 的目标是让模型导出流程具备统一入口、统一配置和可扩展的子类定制能力。

统一入口指调用方优先使用 `AutoLLMWorkflow.from_config()` 创建 workflow，然后按固定顺序调用：

```python
workflow = AutoLLMWorkflow.from_config(
    model_dir="/path/to/hf_model",
    config_path="configs_merak/workflows/xh2a/llm_models/qwen3/0_6b/qwen3_0_6b_xh2a_w8a16.yaml",
)

quant_result = workflow.quant(output_dir="./work_dirs/quant", device="cuda:0")
export_result = workflow.export(quant_result, output_dir="./work_dirs/export", device="cuda:0")
meta_file = workflow.dump_golden(export_result, device="cuda:0", input_messages="你是谁？")
```

统一配置指 workflow YAML 只保留两个顶层概念：

- `quant`：描述量化阶段需要什么。
- `export`：描述 HMONNX 导出阶段需要什么。

可扩展指基类负责稳定主流程，子类只覆盖模型差异。普通单模型导出复用 `BaseLLMWorkflow.export()`；有专属量化、额外导出产物、特殊 golden 输入或模型族校验时，在子类中覆盖对应方法。

### 设计原则

- 基类提供主流程，子类覆盖差异点。
- workflow 不替代模型类，不在 workflow 中重新实现模型结构或 HMONNX 导出细节。
- 配置优先放在已有结构下：量化行为放在 `quant`，导出模型参数放在 `export.model`，导出附加能力放在 `export.<feature>`。
- 具体 workflow 放在对应模型目录，例如 `xhmodel_merak/xh_llm/models/qwen3/workflow.py`。
- 具体 workflow 通过模型类上的 `WORKFLOW_CLS` 字符串懒加载，避免普通模型 import 路径引入额外依赖。
- 重依赖放在方法内部 import，例如 `gptqmodel`、`datasets`、processor 相关依赖。

### 设计边界

Workflow 只编排流程，不直接管理模型 registry，不直接实现 HMONNX 图导出，不直接封装推理后端。它负责连接这些已有能力：

```text
WorkflowConfig
    -> AutoLLMWorkflow
        -> BaseLLMWorkflow / model-specific workflow
            -> AutoLLMConfig
            -> AutoLLMModel
            -> AutoLLMHONNXModel
```

这个边界能让简单模型的 workflow 很薄，也允许复杂模型逐步扩展自己的量化、导出后处理和 golden 逻辑。

## 2. Workflow 具体实现方案

公共 workflow 代码位于：

```text
xhmodel_merak/xh_llm/workflows/
  __init__.py
  auto.py
  base.py
  config.py
  result.py
  utils.py
```

模型子类 workflow 通常位于：

```text
xhmodel_merak/xh_llm/models/<model_name>/workflow.py
```

YAML 配置通常位于：

```text
configs_merak/workflows/<chip_arch>/llm_models/<model_family>/...
```

### AutoLLMWorkflow

`AutoLLMWorkflow` 是 workflow 的统一工厂类。新调用方通常只需要关心：

```python
AutoLLMWorkflow.from_config(model_dir, config_path, seed=1024, debug=False)
```

职责：

- 读取 workflow YAML。
- 从 `export.model` 中解析 `model_type`。
- 通过 `get_model_class()` 复用现有模型 registry。
- 读取模型类上的 `WORKFLOW_CLS`。
- 懒加载具体 workflow 子类。
- 如果模型类没有声明 `WORKFLOW_CLS`，回退到 `BaseLLMWorkflow`。

关键方法：

- `from_config()`：创建 workflow 实例的统一入口。
- `_get_workflow_class()`：解析模型类上的 `WORKFLOW_CLS`，并校验目标类必须继承 `BaseLLMWorkflow`。

`WORKFLOW_CLS` 的格式是 `"module.path:ClassName"`：

```python
class XHNewModel(...):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.new_model.workflow:XHNewModelWorkflow"
```

### WorkflowConfig

`WorkflowConfig` 是 workflow YAML 的结构化包装。它不负责业务逻辑，只负责配置读取、基础校验、覆盖和复制。

重要属性：

- `data`：原始 YAML 数据。
- `source`：YAML 文件路径。
- `name`：由 `source` 文件名推导出的配置名。
- `quant`：返回 `data["quant"]`，值必须是 mapping 或 `None`。
- `export`：返回 `data["export"]`，值必须是非空 mapping。

重要方法：

- `from_file(config_file)`：读取 YAML，校验基础结构，返回 `WorkflowConfig`。
- `dump(config_file)`：把当前 workflow 配置写出到 YAML。
- `with_overrides(overrides)`：返回应用本次覆盖后的新 `WorkflowConfig`，不修改原对象。
- `build_export_dict()`：深拷贝 `export`，供后续构造 `xhquant.Config`。

YAML 最小结构：

```yaml
quant: null

export:
  model:
    chip_arch: XH2a
    model_type: Qwen3ForCausalLM
    # 该字段仅用作占位，实际会在导出前实时覆盖
    hf_model: null
    # 按照 HuggingFace 上的模型名称填写，全部小写，'.' 和 '-' 改成 '_'
    model_name: qwen3_0_6b
```

`export.model.chip_arch` 和 `export.model.model_type` 是必填字段。`export.model.hf_model` 在 YAML 中写 `null` 占位，基类 export 时会覆盖成真实导出模型目录。

`model_name` 写模型基础名，按 HuggingFace 上的模型名称规范化：全部小写，`.` 和 `-` 都改成 `_`。

```yaml
model_name: qwen3_5_9b
```

不要在 YAML 中写完整导出名：

```yaml
# 不推荐
model_name: xh2_qwen3_5_9b_dflash_w4a8_256_2k_mpe256k
```

完整导出名由 `BaseLLMWorkflow._format_model_name()` 在导出前补齐。

`with_overrides()` 的覆盖规则：

- `quant` 可以整体替换，例如 `{"quant": None}`。
- `export` 不能整体替换，应写成 `export.model.context_max_length` 这样的 dotted path。
- 除 `quant` 外，只能覆盖已存在字段，不能新增字段。
- mapping 覆盖 mapping 时，会递归检查子字段是否存在。

`build_export_dict()` 只复制 `export`。它不注入 `hf_model`，也不格式化 `model_name`。新代码应优先让 `BaseLLMWorkflow._build_export_config()` 完成导出配置最终化。

### QuantResult

`QuantResult` 是量化阶段和导出阶段之间的协议对象。

字段：

- `raw_model_dir`：原始模型目录。必须和 workflow 初始化时传入的 `model_dir` 指向同一个目录。
- `skipped`：是否跳过量化。
- `quanted_model_dir`：量化后的 HF 模型目录。`skipped=False` 时必须提供。
- `is_quant_weight_format`：预留字段，表示产物是否是量化权重文件格式；当前大多数 workflow 不需要使用。

跳过量化时返回：

```python
QuantResult(
    raw_model_dir=self.model_dir,
    skipped=True,
)
```

完成量化并保存为 HF 目录时返回：

```python
QuantResult(
    raw_model_dir=self.model_dir,
    skipped=False,
    quanted_model_dir="/path/to/quanted_hf_model",
)
```

### ExportResult

`ExportResult` 是导出阶段产物描述。

字段：

- `work_dir`：本次导出的工作目录。
- `config_file`：导出时写出的 workflow YAML 副本。
- `meta`：底层 `export_hmonnx()` 返回的元信息对象。

子类的 `dump_golden()`、导出后处理、manifest 生成通常都从 `ExportResult` 中继续定位 HMONNX 产物。

### BaseLLMWorkflow

`BaseLLMWorkflow` 是唯一基类，提供通用量化跳过逻辑、通用单模型导出流程和一些 helper。

重要属性：

- `workflow_config`：由 `config_path` 读取出的 `WorkflowConfig`。
- `model_dir`：规范化后的原始模型目录。
- `seed`：导出时使用的随机种子。
- `debug`：传给 `xhquant_init()`。
- `expected_model_config_cls_name`：可选，校验 `AutoLLMConfig` 解析出的 config 类型。
- `expected_model_cls_name`：可选，校验 `AutoLLMModel` 创建出的模型类型。

重要方法：

- `quant(output_dir, device, config_overrides=None)`：
  - 基类只支持 `quant: null`。
  - `quant` 非空时抛 `NotImplementedError`，要求子类实现模型专属量化。

- `export(quant_result, output_dir, device, config_overrides=None)`：
  - 应用本次 `config_overrides`。
  - 调用 `_build_export_config()` 构造传给 `xhquant.Config` 的配置。
  - 初始化日志、随机种子和输出目录。
  - 通过 `AutoLLMConfig` 和 `AutoLLMModel` 创建导出模型。
  - 校验期望的 config/model 类名。
  - 调用底层 `export_hmonnx()`。
  - 返回 `ExportResult`。

- `dump_golden(export_result, device, input_messages)`：
  - 基类只定义接口。
  - 子类必须根据模型输入格式实现。

- `_resolve_export_model_dir(quant_result)`：
  - 校验 `quant_result.raw_model_dir` 和 workflow 的 `model_dir` 一致。
  - `skipped=True` 时返回原始 HF 模型目录。
  - `skipped=False` 时返回 `quanted_model_dir`。

- `_build_export_config(quant_result, workflow_config)`：
  - 调用 `_resolve_export_model_dir()` 得到真实导出模型目录。
  - 深拷贝 `workflow_config.export`。
  - 覆盖 `export.model.hf_model`。
  - 调用 `_format_model_name()` 覆盖 `export.model.model_name`。

- `_format_model_name(workflow_config, export_model_dir)`：
  - 默认生成 `{chip_arch}_{model_name}_{spec_decode_mode}_{quant_scheme}_{prefill}_{context}_mpe{max_pe}`。
  - `spec_decode_mode` 取 `export.model.spec_decode_mode`；字段缺失或为空时不拼接这一段。
  - `XH2a` 会映射成 `xh2`。
  - `quant_scheme.quant_type` 只取 `w{数字}a{数字}`。
  - `quant.bits` 表示量化权重 bit 数；如果存在，最终命名里的 `w` 位宽优先使用 `quant.bits`。
  - `max_pe_length` 优先取 `export.model.max_pe_length`；缺失时尝试在 `export_model_dir` 下递归查找 `config.json` 的 `max_position_embeddings`。
  - 如果格式化所需字段不足，返回 YAML 中原始 `model_name`。
  - 如果 `export.model.model_name` 本身缺失，直接报错。
  - 如果 `export.model.quant_scheme.quant_type` 存在但不包含 `w{数字}a{数字}`，直接报错。

- `_find_golden_meta_file(export_result)`：
  - 在导出目录下查找唯一的 `hmquant*/golden_meta_info.json`。
  - 常被子类 `dump_golden()` 复用。

### utils.py

`utils.py` 当前主要提供路径比较 helper，例如 `same_abs_path()`。基类用它判断 `QuantResult.raw_model_dir` 是否和 workflow 初始化时的 `model_dir` 指向同一目录，避免把不匹配的量化产物传给导出阶段。

## 3. Workflow 子类开发规范

### 子类放置和注册

新增 workflow 子类时，把文件放在对应模型目录：

```text
xhmodel_merak/xh_llm/models/<model_name>/workflow.py
```

最小子类：

```python
from ...workflows.base import BaseLLMWorkflow


class XHNewModelWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHNewModelConfig"
    expected_model_cls_name = "XHNewModel"
```

在模型类上声明：

```python
class XHNewModel(...):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.new_model.workflow:XHNewModelWorkflow"
```

不要在 `models/__init__.py` 或单模型 `__init__.py` 中 eager import workflow。`WORKFLOW_CLS` 的字符串路径会在 `AutoLLMWorkflow.from_config()` 中按需加载。

### 必须实现和可选实现

必须实现的情况：

- YAML 中 `quant` 非空：必须覆盖 `quant()`。
- 需要生成 golden：必须覆盖 `dump_golden()`。
- 导出流程不是普通单模型导出：必须覆盖 `export()`。

可选实现的情况：

- 只需要校验模型类型：可以在 `export()` 中先调用私有校验方法，再 `super().export()`。
- 只需要导出后写 manifest：覆盖 `export()`，先 `super().export()`，再基于 `ExportResult` 写文件。
- 只需要支持多种输入格式：实现 `build_input_message()`，在 `dump_golden()` 中复用。
- 需要不同命名规则：覆盖 `_format_model_name()`。

推荐结构：

```python
class XHNewModelWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHNewModelConfig"
    expected_model_cls_name = "XHNewModel"

    def quant(self, output_dir, device, config_overrides=None):
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        ...

    def export(self, quant_result, output_dir, device, config_overrides=None):
        self._validate_export_model(config_overrides)
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
        ...
        return export_result

    def dump_golden(self, export_result, device, input_messages):
        meta_file = self._find_golden_meta_file(export_result)
        ...
        return meta_file
```

### 量化实现规范

`quant()` 的返回值必须是 `QuantResult`。

如果支持跳过量化，应优先保留这个分支：

```python
if workflow_config.quant is None:
    return QuantResult(raw_model_dir=self.model_dir, skipped=True)
```

如果量化后保存成 HF 模型目录，应返回：

```python
return QuantResult(
    raw_model_dir=self.model_dir,
    quanted_model_dir=save_path,
)
```

如果从已经量化好的 HF 模型目录导出，应跳过量化阶段，方式是将
`quant` 设置为 `null`。代码里对应 `workflow_config.quant is None`，
此时 `QuantResult.raw_model_dir` 指向传入的原始模型目录，`export()` 会直接
使用该目录构造导出配置。

### 导出实现规范

普通模型不要重写完整 `export()`。优先：

```python
export_result = super().export(...)
```

然后在导出后做模型专属处理，例如写 manifest、额外导出 visual bucket、复制额外文件。

如果子类需要在调用 `get_model_class()` 或构造 export plan 前拿到 `hf_model`，可以临时构造 export dict：

```python
export_cfg = workflow_config.build_export_dict()
# New export config finalization should go through BaseLLMWorkflow._build_export_config().
export_cfg["model"]["hf_model"] = self.model_dir
```

这类代码只应出现在子类内部校验或规划路径中，不应替代基类 `_build_export_config()`。

### Golden 实现规范

`dump_golden()` 应完成三件事：

1. 定位导出的 `golden_meta_info.json`。
2. 根据模型输入格式构造 tokenizer/processor 输入。
3. 跑一次真实 HMONNX generate，并返回 meta 文件路径。

常见起点：

```python
meta_file = self._find_golden_meta_file(export_result)
hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_file)
```

`input_messages` 的格式由子类定义。文本模型可以支持字符串或 `{"text": ...}`；多模态模型可以支持 `{"image": ..., "text": ...}`；复杂模型可以允许直接传 `messages`。

### 配置字段设计规范

新增字段时遵循以下约定：

- 量化阶段字段放在 `quant`。
- 普通导出模型字段放在 `export.model`。
- 导出附加能力放在 `export.<feature>`，例如 `export.visual_buckets`。
- 不要新增无必要的顶层字段。

`model_name` 写模型基础名，不写完整导出名。命名应按 HuggingFace 上的模型名称规范化：全部小写，`.` 和 `-` 都改成 `_`。

```yaml
model_name: qwen3_6_27b
```

基类会在导出前补齐芯片、投机解码模式、量化位宽、prefill/context 长度和 `mpe`。开发子类模型 YAML 时，尽量遵守基类字段规约：

- `quant.bits` 表示量化权重 bit 数。
- `export.model.spec_decode_mode` 表示投机解码方式，例如 `mtp` 或 `dflash`。

如果某个子图或 visual-only 配置缺少格式化所需字段，基类会保留原始 `model_name`，子类可以自行决定是否额外命名。如果子类模型确实有特殊含义或不同规则，也可以自定义字段和命名逻辑，但这类配置无法直接复用基类 `_format_model_name()`，需要在子类中覆盖该方法。

### 定制化能力边界

当前基类设计足够灵活，常见定制需求基本都可以通过子类局部覆盖完成：

- 量化方式不同：覆盖 `quant()`。
- 需要导出多个 HMONNX：覆盖 `export()`，复用基类导出主模型后继续导出子图。
- 导出后需要操作hmonnx产物：覆盖 `export()`，用 `ExportResult` 定位导出目录并进行相应操作。
- 需要不同输入模态：覆盖 `dump_golden()` 和 `build_input_message()`。
- 需要导出前校验：在 `export()` 中先调用私有校验方法。
- 需要完全不同的主流程：覆盖 `export()`，但尽量复用 `WorkflowConfig`、`QuantResult`、`ExportResult` 和 `_resolve_export_model_dir()`。

## 4. Workflow 子类开发示例讲解

### Qwen3：简单文本模型

文件：

```text
xhmodel_merak/xh_llm/models/qwen3/workflow.py
```

`XHQwen3HMONNXWorkflow` 展示了简单文本模型的典型写法。

它设置了两个校验属性：

```python
expected_model_config_cls_name = "XHQwen3ModelConfig"
expected_model_cls_name = "XHQwen3Model"
```

`quant()` 支持：

- `quant: null`：跳过量化。
- `quant.bits`：权重量化 bit 数，由子类量化实现解释并执行。

`export()` 只是调用 `super().export()`，说明普通 LLM 不需要重写导出主流程。

`dump_golden()` 使用 tokenizer 处理纯文本输入。`input_messages` 支持：

```python
"你是谁？用中文回答。"
```

或：

```python
{"text": "你是谁？用中文回答。"}
```

开发新文本模型时，可以先按 Qwen3 的结构实现：量化逻辑按需覆盖，导出复用基类，golden 只处理文本输入。

### Qwen2-VL：带导出后处理的多模态模型

文件：

```text
xhmodel_merak/xh_llm/models/qwen2_vl/workflow.py
```

`XHQwen2VLHMONNXWorkflow` 展示了“先复用基类导出，再做导出后处理”的写法。

主流程是：

```python
workflow_config = self.workflow_config.with_overrides(config_overrides)
export_result = super().export(...)

visual_buckets_cfg = workflow_config.export.get("visual_buckets")
if visual_buckets_cfg is not None:
    self._write_visual_bucket_manifest(...)
return export_result
```

普通 Qwen2-VL 配置没有 `export.visual_buckets`，此时行为和基类导出一致。

MinerU 这类多 visual bucket 场景通过 `export.visual_buckets` 扩展：

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

存在 `export.visual_buckets` 时，Qwen2-VL workflow 会：

1. 复用基类导出主模型。
2. 从 `ExportResult.meta` 中读取默认 visual HMONNX 信息。
3. 解析多个静态 bucket。
4. 对非默认 bucket 额外导出 visual HMONNX。
5. 写出 `mineru_visual_buckets.json`。

`dump_golden()` 使用 processor 处理图文输入：

```python
{
    "image": "./data/images/qwen2_vl_demo.jpeg",
    "text": "描述这张图片",
}
```

这个例子说明：复杂导出产物不一定需要重写完整 workflow。只要主模型导出仍符合基类流程，就可以在 `super().export()` 后追加处理。

### Qwen3.5：复杂模型族 workflow

文件：

```text
xhmodel_merak/xh_llm/models/qwen3_5/workflow.py
```

`Qwen35Workflow` 展示了一个模型族 workflow 如何承载更多分发逻辑。它同时支持 dense、MoE、visual-only 等导出形态，因此复杂度明显高于 Qwen3 和 Qwen2-VL。

`quant()` 的分发逻辑包括：

- `quant: null`：跳过量化，直接从传入的 HF 模型目录导出；如果该目录本身
  已经是量化后的 HF 模型目录，也使用这个方式。
- `gptqmodel + method: gptq`：调用 GPTQModel adapter。
- `gptqmodel + method: autoround` 或 legacy `autoround`：调用 AutoRound adapter。

辅助方法的职责：

- `_resolve_artifact_format()`：统一解析 `artifact_format` 和 `output_format`。
- `_validate_group_size()`：集中校验 group size 约束。
- `_normalize_path()`：规范化外部量化目录路径。
- `_validate_export_model()`：导出前校验 YAML 解析出的模型类和 config 类是否属于 Qwen3.5/Qwen3.6 支持范围。
- `_messages_have_image()`：判断 golden 输入是否包含 image，决定使用 tokenizer 还是 processor。

`dump_golden()` 同时支持文本和图文输入：

```python
"解释一下 speculative decoding。"
```

```python
{"text": "解释一下 speculative decoding。"}
```

```python
{
    "image": "./data/images/demo.jpeg",
    "text": "描述这张图片。",
}
```

```python
{
    "messages": [
        {"role": "user", "content": "你好"}
    ]
}
```

这个例子说明：当模型家族有多种量化来源、多种导出形态、多种输入模态时，workflow 子类可以集中管理分发和校验逻辑，同时继续复用基类的导出主流程。

### 从哪个示例开始

新增模型时按复杂度选择参考对象：

- 普通文本 LLM：优先参考 Qwen3。
- 多模态模型，导出后还要写额外文件或导出额外子图：参考 Qwen2-VL。
- 一个模型族需要支持多量化来源、多导出形态、多输入模态：参考 Qwen3.5，但只复制自己真正需要的部分。

不要一开始复制最复杂的 workflow。先继承 `BaseLLMWorkflow`，只在确实需要时逐步覆盖方法。基类的目标就是让简单模型保持简单，同时允许复杂模型按需扩展。

### 开发注意事项

开发新 workflow 时应尽量遵守本文档约定的设计规范。优先通过子类覆盖方法、
增加模型目录内的辅助函数、扩展 YAML 中已有的 `quant` 和 `export` 字段来表达
模型差异。

修改 `BaseLLMWorkflow`、`WorkflowConfig`、`AutoLLMWorkflow` 等公共基类和
公共入口时务必慎重。这些代码会影响所有 workflow 子类，不应为了单个模型的
特殊需求引入破坏性改动。如果确实需要调整公共接口、修改基类行为，或者进行
影响面较大的重构，应先与相关同事讨论清楚设计方案和迁移范围，再开始实现。
