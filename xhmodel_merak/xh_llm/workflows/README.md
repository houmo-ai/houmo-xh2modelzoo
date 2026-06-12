# HMONNX Workflow 使用与开发指南

本文档说明 `xhmodel_merak.xh_llm.workflows` 的设计理念、配置格式、子类开发方式和示例使用方法。Workflow 的目标是把一个模型从 HF 原始模型或量化模型到 HMONNX 产物的流程封装成稳定 API，方便 examples 和外部仓库复用。

## 设计理念

Workflow 层负责串联量化、导出和 golden 生成流程，但不替代底层模型导出实现。底层模型仍然由 `xhmodel_merak.xh_llm` 中已有的 `AutoLLMConfig`、`AutoLLMModel`、`AutoLLMHONNXModel` 和具体模型类完成。

核心设计约束如下：

- 统一入口：每个模型提供一个 workflow 类，调用方只需要构造 workflow 对象，然后依次调用 `quant()`、`export()`，必要时调用 `dump_golden()`。
- 配置外置：量化和导出配置统一写在 YAML 中，避免继续扩展旧式 Python config。
- 兼容旧导出：YAML 的 `export` 部分尽量保持旧式 config 的结构，导出前会构造旧式 config 字典交给现有导出链路。
- 模型路径运行时注入：`export.model.hf_model` 在 YAML 中只作为占位字段，真实模型路径由 workflow 初始化传入的 `hf_model_dir` 和 `QuantResult` 决定。
- 单次调用覆盖：`config_overrides` 只对当前 `quant()` 或 `export()` 调用生效，不修改 workflow 对象保存的原始配置。
- 避免重依赖 eager import：`workflows/__init__.py` 只导出基础类、配置类和结果对象，不主动 import 各模型 workflow，避免外部仓库仅导入基础 API 时拉起 `transformers`、`qwen_vl_utils`、`datasets`、`gptqmodel` 等重依赖。
- 路径类型收敛：跨对象保存和返回的路径使用 `str`。只有做局部文件系统操作时临时使用 `Path`。

## 目录结构

```text
xhmodel_merak/xh_llm/workflows/
  __init__.py
  base.py
  config.py
  result.py
  utils.py
  models/
    __init__.py
    qwen2_vl.py
    mineru2_5.py
    qwen3_legacy.py
```

对应的 YAML 配置放在：

```text
configs_merak/workflows/<chip_arch>/llm_models/<model_name>/...
```

示例脚本放在：

```text
examples_merak/llm/<model_name>/*_workflow.py
```

## 公共接口

### BaseHMONNXWorkflow

`BaseHMONNXWorkflow` 是所有模型 workflow 的基类。

```python
workflow = SomeModelWorkflow(
    hf_model_dir="/path/to/hf_model",
    config_path="/path/to/workflow.yaml",
    seed=1024,
    debug=False,
)
```

构造参数：

- `hf_model_dir`：原始 HF 模型目录。传入后会被规范化成绝对路径字符串。
- `config_path`：workflow YAML 配置路径。
- `seed`：导出时调用 `set_random_seed()` 的 seed。
- `debug`：传给 `xhquant_init()` 的 debug 开关。

主要方法：

- `quant(output_dir, device, config_overrides=None) -> QuantResult`
- `export(quant_result, output_dir, device, config_overrides=None) -> ExportResult`
- `dump_golden(export_result, device, input_messages) -> str`

基类默认 `quant()` 只支持 `quant: null` 的配置，这种情况下返回 `skipped=True` 的 `QuantResult`。如果 YAML 中存在非空 `quant` 配置，子类必须实现自己的量化逻辑。

基类 `export()` 适用于绝大多数单模型导出流程：

1. 基于 `config_overrides` 得到本次调用使用的 `WorkflowConfig`。
2. 根据 `QuantResult` 决定导出使用的 HF 模型路径。
3. 通过 `WorkflowConfig.build_export_dict()` 构造旧式导出 config。
4. 创建 `output_dir`，初始化日志和 seed。
5. dump 本次实际使用的 workflow YAML 到输出目录。
6. 使用 `AutoLLMConfig.from_pretrained(cfg.model)` 和 `AutoLLMModel.from_pretrained()` 构造模型。
7. 校验解析出来的模型 config 类名和模型类名。
8. 调用 `xh_model.export_hmonnx(output_dir)`。
9. 返回 `ExportResult`。

基类 `dump_golden()` 当前是抽象接口。不同模型的 tokenizer、processor、chat template 和输入结构可能不同，因此由子类实现。

### WorkflowConfig

`WorkflowConfig` 是 YAML 的内存表示，核心字段：

- `data: dict[str, Any]`：完整 YAML 数据。
- `source: str`：配置来源文件路径。

常用方法：

- `WorkflowConfig.from_file(config_file)`：读取 YAML，校验基础结构。
- `dump(config_file)`：将当前配置 dump 到 YAML 文件。
- `with_overrides(overrides)`：返回应用覆盖后的新配置对象。
- `build_export_dict(export_hf_model_dir)`：构造旧式导出 config，并把 `export.model.hf_model` 覆盖为真实导出模型路径。

`with_overrides()` 的路径从最外层写起，例如：

```python
CONFIG_OVERRIDES = {
    "export.model.context_max_length": 8192,
    "export.model.prefill_chunk_length": 512,
    "export.model.chip_arch": "XH2a",
}
```

覆盖规则：

- 只能覆盖已存在字段，不能新增字段。
- 覆盖前会检查路径是否合法。
- 如果用 mapping 覆盖 mapping，也会递归检查子字段是否已存在。
- 覆盖只对当前调用生效，不会修改 workflow 对象中的原始配置。

### QuantResult

`QuantResult` 描述量化阶段产物。

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
- `is_quant_weight_format`：预留字段，用于描述量化产物是否是单独权重文件；当前 workflow 设计中通常不用关心。

`export()` 会通过 `_resolve_export_hf_model_dir()` 选择导出路径：

- `skipped=True`：使用 `self.hf_model_dir`。
- `skipped=False`：使用 `quant_result.quanted_model_dir`。

### ExportResult

`ExportResult` 描述导出阶段产物。

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

## YAML 配置规范

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
      nodes:
        lm_head:
          quant_type: w8a8h0_sefp
      ops: {}
    only_first_block: false
```

### quant

`quant` 描述量化阶段配置。

- `quant: null` 表示不执行量化，直接从 `hf_model_dir` 导出。
- 非空 mapping 表示模型子类需要实现量化逻辑。

Qwen3 legacy GPTQ 示例：

```yaml
quant:
  bits: 4
```

对应子类会读取 `workflow_config.quant["bits"]`，使用 `GPTQModel` 执行量化，并返回包含 `quanted_model_dir` 的 `QuantResult`。

### export

`export` 描述导出阶段配置。基类会把 `export` 的内容作为旧式导出 config 的主体。

`export.model` 至少必须包含：

- `chip_arch`
- `model_type`
- `hf_model`

其中 `hf_model` 在 YAML 中保留为 `null`，仅用于占位。导出前会被覆盖为真实路径：

- 如果量化跳过，覆盖为 workflow 初始化传入的 `hf_model_dir`。
- 如果量化执行成功，覆盖为 `QuantResult.quanted_model_dir`。

`export.model` 下的其他字段保持和旧式 Python config 中的 `model = dict(...)` 尽量一致，例如：

- `model_name`
- `context_max_length`
- `prefill_chunk_length`
- `use_cache`
- `num_logits_to_keep`
- `quant_scheme`
- `visual_config`
- `only_first_block`

### 扩展配置

模型子类可以在 `export` 下增加自己的扩展字段。MinerU 示例中增加了 `visual_buckets`：

```yaml
export:
  model:
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

这类字段不会被基类直接消费，由对应子类在 override 的 `export()` 中读取并处理。

## 子类开发指南

新增模型 workflow 时，推荐按以下步骤实现。

### 1. 新增模型 workflow 文件

文件放在：

```text
xhmodel_merak/xh_llm/workflows/models/<model_name>.py
```

不要在 `workflows/__init__.py` 或 `workflows/models/__init__.py` 中 eager import 模型 workflow。调用方需要显式导入：

```python
from xhmodel_merak.xh_llm.workflows.models.qwen2_vl import XHQwen2VLHMONNXWorkflow
```

这样可以避免外部仓库只使用基础 API 时拉起重依赖。

### 2. 继承 BaseHMONNXWorkflow

最小子类示例：

```python
from collections.abc import Mapping
from typing import Any

from ..base import BaseHMONNXWorkflow
from ..result import ExportResult, QuantResult


class XHNewModelHMONNXWorkflow(BaseHMONNXWorkflow):
    expected_model_config_cls_name = "XHNewModelConfig"
    expected_model_cls_name = "XHNewModel"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        return super().quant(
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        return super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
```

`expected_model_config_cls_name` 和 `expected_model_cls_name` 用于在导出时校验自动解析出的模型类型，避免 YAML 写错 `model_type` 后静默导出成另一个模型。

### 3. 实现量化逻辑

如果模型不需要 workflow 负责量化，直接复用基类 `quant()` 即可，并在 YAML 中写：

```yaml
quant: null
```

如果模型需要量化，则子类读取 `workflow_config.quant` 并返回正确的 `QuantResult`。

Qwen3 legacy 的模式：

```python
workflow_config = self.workflow_config.with_overrides(config_overrides)
if workflow_config.quant is None:
    return QuantResult(hf_model_dir=self.hf_model_dir, skipped=True)

bits = workflow_config.quant["bits"]
save_path = ...
...
return QuantResult(
    hf_model_dir=self.hf_model_dir,
    quanted_model_dir=save_path,
)
```

注意事项：

- `output_dir` 是量化产物根目录，由调用方传入。
- `device` 是量化执行设备，由调用方传入。
- 量化依赖建议放在方法内部导入，避免模块 import 时拉起重依赖。
- 返回的 `hf_model_dir` 必须指向原始模型路径，不能写成量化模型路径。
- 未跳过量化时必须设置 `quanted_model_dir`。

### 4. 复用或扩展导出逻辑

普通模型直接调用 `super().export()`。

如果模型导出有额外步骤，可以 override `export()`，先调用基类导出主模型，再处理附加产物。

MinerU 的模式：

```python
workflow_config = self.workflow_config.with_overrides(config_overrides)
export_result = super().export(
    quant_result=quant_result,
    output_dir=output_dir,
    device=device,
    config_overrides=config_overrides,
)

visual_buckets_cfg = workflow_config.export.get("visual_buckets")
if visual_buckets_cfg is not None:
    self._write_visual_bucket_manifest(...)

return export_result
```

适合在子类中扩展的场景：

- 同一个模型要导出多个静态 shape。
- 主模型导出后还要额外导出 visual/audio/encoder 子图。
- 需要写额外 manifest 文件。
- 需要根据 `export_result.meta` 找到底层导出目录。

### 5. 实现 dump_golden

基类不默认实现 `dump_golden()`，因为不同模型的输入构造方式不一致。

子类通常需要完成：

1. 从 `ExportResult.work_dir` 下找到唯一的 `hmquant*` 目录。
2. 在该目录下找到 `golden_meta_info.json`。
3. 使用 `AutoLLMHONNXModel.from_pretrained(meta_file)` 加载模型。
4. 按模型需要构造 tokenizer 或 processor 输入。
5. 设置 `hmonnx_model.enable_golden = True`。
6. 用较短生成触发 golden 保存，通常 `max_new_tokens=2`。
7. 返回 `golden_meta_info.json` 路径字符串。

Qwen2-VL 使用 processor：

```python
messages = self.build_input_message(input_messages)
model_inputs = processor.apply_chat_template(messages).to(device)
```

Qwen3 legacy 使用 tokenizer：

```python
messages = self.build_input_message(input_messages)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
```

### 6. 定义 input_messages 约定

`input_messages` 是 workflow 层暴露给调用方的输入容器，具体结构由模型子类定义。

Qwen2-VL：

```python
input_messages = {
    "image": "./data/images/qwen2_vl_demo.jpeg",
    "text": "描述这张图片",
}
```

转换成 Qwen2-VL chat message：

```python
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "..."},
            {"type": "text", "text": "..."},
        ],
    }
]
```

Qwen3 legacy：

```python
input_messages = {"text": "你多大了？用中文回答。"}
```

也支持直接传字符串：

```python
input_messages = "你多大了？用中文回答。"
```

## 样例使用指南

### Qwen2-VL

示例脚本：

```bash
python examples_merak/llm/qwen2_vl/qwen2_vl_workflow.py
```

核心流程：

```python
from xhmodel_merak.xh_llm.workflows.models.qwen2_vl import XHQwen2VLHMONNXWorkflow

workflow = XHQwen2VLHMONNXWorkflow(
    hf_model_dir="/data02/datasets/Qwen2-VL-2B-Instruct",
    config_path="./configs_merak/workflows/xh2a/llm_models/qwen2_vl/2b/qwen2_vl_2b_xh2a_4k.yaml",
    seed=1024,
    debug=False,
)

quant_result = workflow.quant(
    output_dir="./work_dirs/qwen2_vl_workflow_quant",
    device="cuda",
    config_overrides=None,
)
export_result = workflow.export(
    quant_result=quant_result,
    output_dir="./work_dirs/qwen2_vl_workflow_export",
    device="cuda",
    config_overrides=None,
)
workflow.dump_golden(
    export_result=export_result,
    device="cuda",
    input_messages={
        "image": "./data/images/qwen2_vl_demo.jpeg",
        "text": "描述这张图片",
    },
)
```

### MinerU 2.5

示例脚本：

```bash
python examples_merak/llm/mineru2.5/mineru2_5_workflow.py
```

MinerU 复用 Qwen2-VL 的模型结构，但导出后会额外处理 `export.visual_buckets`：

- 主 VLM 仍由基类 `export()` 导出。
- 子类根据 YAML 中的 `visual_buckets.buckets` 导出额外 visual 静态图。
- 每个额外 visual bucket 输出到主导出目录下的 `visual_<WxH>` 目录。
- 子类写出 `mineru_visual_buckets.json`，记录各 bucket 的 HMONNX 路径和 fallback bucket。

适合类似需求的模型参考这个子类：同一个模型结构需要为不同 shape 导出多个静态图时，不需要改基类，优先在子类扩展 `export()`。

### Qwen3 Legacy

示例脚本：

```bash
python examples_merak/llm/qwen3_legacy/qwen3_legacy_workflow.py
```

对应 YAML：

```text
configs_merak/workflows/xh2a/llm_models/qwen3_legacy/1_7b/qwen3_1_7b_legacy_xh2a_w4a8_gptq_2k.yaml
```

Qwen3 legacy 的 workflow 展示了非空 `quant` 的处理方式：

- `quant.bits` 控制 GPTQ bits。
- 子类内部导入 `datasets`、`gptqmodel`。
- 量化产物保存到 `output_dir` 下。
- `export()` 根据 `QuantResult.quanted_model_dir` 自动使用量化后的模型目录。

## 外部仓库调用建议

外部仓库调用时建议只 import 需要的模型 workflow：

```python
from xhmodel_merak.xh_llm.workflows.models.qwen3_legacy import XHQwen3LegacyHMONNXWorkflow
```

不要依赖 `workflows.models.__init__` 聚合导出模型类。这样可以避免不必要的依赖加载，也能让不同模型的可选依赖保持隔离。

推荐调用顺序：

```python
workflow = XHQwen3LegacyHMONNXWorkflow(
    hf_model_dir="/path/to/hf_model",
    config_path="/path/to/config.yaml",
)

quant_result = workflow.quant(
    output_dir="/path/to/quant_output",
    device="cuda",
)

export_result = workflow.export(
    quant_result=quant_result,
    output_dir="/path/to/export_output",
    device="cuda",
)
```

如果调用方已经有量化后的 HF 模型目录，也可以直接构造 `QuantResult`：

```python
from xhmodel_merak.xh_llm.workflows import QuantResult

quant_result = QuantResult(
    hf_model_dir="/path/to/original_hf_model",
    skipped=False,
    quanted_model_dir="/path/to/quanted_hf_model",
)
```

但要注意 `hf_model_dir` 必须和 workflow 初始化传入的原始路径一致，否则 `export()` 会报错。

## 输出目录约定

`export(output_dir=...)` 要求 `output_dir` 不存在。如果目录已经存在，会抛出 `FileExistsError`。示例脚本通过 `FORCE_OVERWRITE` 主动删除旧目录，便于本地调试。

导出目录通常包含：

```text
work_dirs/<name>/
  export_hmonnx.log
  <workflow_config_name>.yaml
  hmquant*/
    golden_meta_info.json
    ...
```

不同模型可能有额外产物。例如 MinerU 会在 `hmquant*` 目录下写：

```text
visual_<WxH>/
mineru_visual_buckets.json
```

## 路径处理约定

Workflow public API 中路径统一使用字符串：

- `hf_model_dir`
- `config_path`
- `output_dir`
- `QuantResult.hf_model_dir`
- `QuantResult.quanted_model_dir`
- `ExportResult.work_dir`
- `ExportResult.config_file`

内部只有在需要文件系统操作时使用 `Path`，例如：

- 检查目录是否存在。
- 创建目录。
- 拼接文件名。
- 读取或写入 YAML、JSON。
- 扫描 `hmquant*` 目录。

路径比较使用 `same_abs_path(path1, path2)`，它会先做 `abspath(normpath(str(path)))` 再比较。

## 常见问题

### 为什么 YAML 里 `export.model.hf_model` 是 null？

因为原始模型路径通常由调用方决定，不应该写死在配置里。同时导出阶段可能使用量化后的模型目录，而不是原始模型目录。

Workflow 的处理方式是：

- 初始化时保存原始 `hf_model_dir`。
- 量化跳过时，导出使用原始 `hf_model_dir`。
- 量化执行后，导出使用 `QuantResult.quanted_model_dir`。
- `WorkflowConfig.build_export_dict()` 在导出前实时覆盖 `export.model.hf_model`。

### 为什么 config_overrides 不能新增字段？

`config_overrides` 是单次调用的安全覆盖机制，不是动态配置生成器。只允许覆盖已有字段，可以减少拼错路径、漏写 YAML 字段或意外改变配置结构带来的风险。

### 为什么不在 __init__.py 中导出所有模型 workflow？

不同模型的量化、processor、视觉处理依赖差异很大。`__init__.py` eager import 所有模型会让外部仓库只导入基础 workflow 时也加载大量可选依赖。当前约定是：

```python
from xhmodel_merak.xh_llm.workflows import WorkflowConfig
from xhmodel_merak.xh_llm.workflows.models.qwen2_vl import XHQwen2VLHMONNXWorkflow
```

基础对象从 `workflows` import，具体模型类从 `workflows.models.<model>` 显式 import。

### 什么时候应该改基类？

只有当逻辑对所有模型都成立时才改基类，例如：

- `QuantResult` 到导出路径的统一解析。
- YAML 读取、校验、覆盖。
- 输出目录、日志、seed、模型类型校验。
- 旧式 export config 的统一构造。

如果只是某个模型有额外 shape、额外子图、特殊量化或特殊 golden 输入，优先在模型子类中实现。

### 新增模型最小检查命令是什么？

```bash
python -m compileall -q xhmodel_merak/xh_llm/workflows examples_merak/llm/<model>/<model>_workflow.py
```

如果环境中有完整 `xhquant`、模型权重和数据依赖，再运行对应示例脚本做真实导出。
