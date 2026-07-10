# xh_other_model 迁移指南

本文档用于指导将旧 `xh_model_zoo`、旧 `xh2modelzoo/examples` 脚本，或其他历史导出流程迁移到 `xhmodel_merak/xh_other_model`。目标是让模型通过统一 workflow 接口完成量化、导出和 golden 生成，并逐步移除对旧目录和旧示例脚本的运行期依赖。

## 迁移目标

迁移完成后，一个模型应满足以下要求：

- 模型代码位于 `xhmodel_merak/xh_other_model/models/<model_name>/`。
- workflow YAML 位于 `configs_merak/workflows/xh2a/other_models/<model_name>/...`。
- 示例、demo、评估和分析脚本位于 `examples_merak/...`。
- 统一使用 `AutoWorkflow.from_config()` 启动 workflow。
- YAML 中 `quant` 管理量化配置，`export` 管理导出配置。
- 主模型类型固定在 `export.model.type`，用于自动绑定模型类和 workflow。
- `export()` 只负责导出，不生成 golden。
- `dump_golden()` 只负责 golden，不混入 quant/export。
- 不再依赖 `./examples`、`./xh_model_zoo`、`./configs` 中的代码或配置，也不动态加载这些目录下的旧脚本。

原则上，在相同配置下，迁移后的 HMONNX 应尽量与迁移前一致。若为规避导出或 runtime bug 必须改变图结构或模型代码，应保持改动最小，并在代码或文档中说明原因。

## 当前架构

`xh_other_model` 主要由以下部分组成：

- `builder.py`
  - 维护 `MODELS` registry。
  - 通过 `@register_other_model("<type>")` 注册模型类。
  - 通过 `get_model_class(cfg)` 按 `cfg["type"]` 找到模型类。

- `scan_model_types.py`
  - 扫描 `xh_other_model/models/**.py` 中的 `@register_other_model(...)`。
  - 自动建立 `model_type -> model package` 映射。

- `workflows/config.py`
  - 读取和校验 workflow YAML。
  - 要求顶层存在 `quant` 和 `export`。
  - 要求 `export.target_device` 非空。
  - 要求 `export.model.type` 非空。
  - 支持 `with_overrides()`，但只允许覆盖已存在的路径，避免命令行 typo 静默生效。

- `workflows/auto.py`
  - `AutoOtherModelWorkflow.from_config()` 读取 `export.model`，找到模型类，再通过模型类的 `WORKFLOW_CLS` 绑定具体 workflow。

- `xhmodel_merak/workflows/auto.py`
  - 顶层入口 `AutoWorkflow`。
  - 自动分发到 `xh_other_model` 或 `xh_llm` 子包。
  - 新示例脚本应优先使用该入口。

## 推荐目录布局

以 `<model_name>` 为例：

```text
xhmodel_merak/xh_other_model/models/<model_name>/
  __init__.py
  workflow.py
  model.py 或 *_model.py
  *_inference.py
  _export_utils.py
  其他从旧实现迁移来的私有 helper

configs_merak/workflows/xh2a/other_models/<model_name>/
  <variant>.yaml

examples_merak/<domain>/<model_name>/
  <model_name>_workflow.py
  hmonnx_demo.py
  hmonnx_streaming_demo.py        # 如有流式图
  eval/...                        # 如有评估
  analysis/...                    # 如有分析
  README.md
```

迁移后的模型包不应从 `./examples`、`./xh_model_zoo`、`./configs` 动态导入 helper、读取配置或复用旧脚本。旧流程中仍需要的函数应搬到模型包内部，例如 `_export_utils.py`。

当前 `xh_other_model` 只迁移了旧 `xh_model_zoo` 中一部分公共代码，例如 `base_llm_model.py`。迁移模型时，如果遇到老模型依赖旧 `xh_model_zoo` 的公共代码，但该公共代码还没有进入 `xh_other_model`，优先在 `xhmodel_merak/xh_other_model/models/<model_name>/` 这个模型子路径中实现所需功能。除非多个迁移模型确实共享同一行为且边界已经明确，否则不要在 `xh_other_model` 公共目录增加新代码，以免扩大耦合。

## 模型注册和 Auto 绑定

主模型类需要通过 `register_other_model` 注册，并设置 `WORKFLOW_CLS`：

```python
from ...base_llm_model import LLMBaseModel
from ...builder import register_other_model


@register_other_model("XHExampleModel")
class XHExampleModel(LLMBaseModel):
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.example.workflow:ExampleWorkflow"
```

要点：

- `export.model.type` 必须与 `@register_other_model(...)` 的字符串一致。
- 多子模型工作流只需要选择一个主模型放在 `export.model`，用于 Auto 绑定。
- 其他子模型配置可放在 `export.<descriptive_name>`，例如 `export.expert_model`、`export.components`、`export.speech_tokenizer`。
- 不要增加手工维护的 model type 映射表；扫描器会自动从装饰器建立映射。

示例脚本中统一使用：

```python
from xhmodel_merak.workflows import AutoWorkflow

workflow = AutoWorkflow.from_config(
    model_dir=args.model_dir,
    config_path=config_path,
    debug=args.debug,
)
```

## YAML 迁移规则

workflow YAML 必须是完整配置，不依赖旧 Python config 继承链。旧 py config 中的 `_base_`、变量拼接和默认值都要在 YAML 中展开。

基本结构：

```yaml
quant: null

export:
  target_device: XH2a
  model:
    type: XHExampleModel
    hf_model: null
    wrap_cfg:
      input_sequence_length: 1024
    quant_config: {}
    frontend_type: TorchFX
    export_cfg:
      input_names: []
      output_names: []
```

规则：

- `quant` 放量化阶段配置；没有独立量化阶段时可为 `null`。
- `export` 放导出阶段配置。
- `export.target_device` 写芯片架构，例如 `XH2a`。
- `export.model` 保留旧 `MODELS.build()` 所需的完整配置，不要拆碎。
- `export.model.type` 位置固定，供 Auto 接口查找模型类。
- 多子模型时，主模型仍放 `export.model`，其他子模型可放 `export.modelA`、`export.modelB` 或更清晰的描述性字段。
- 每个子模型导出精度必须能通过 YAML 控制。
- 导出的 HMONNX 文件名应体现 `target_device` 和 `quant_type`。

常见字段布局示例：

```yaml
export:
  target_device: XH2a
  components:
    encoder:
      quant_type: w8a8_sefp
      max_audio_length: 1500
    prefill_decode:
      quant_type: w8a8_sefp
      prefix_token_budget: 512
  model:
    type: XHExampleLLMModel
    ...
```

非 `MODELS.build()` 参数，例如音频长度、prefix token 预算、组件列表、流式图开关，可以放在 `export.*` 下的描述性字段中，但要保持结构清晰，且命令行 override 路径必须对应 YAML 中已存在的字段。

## Workflow 实现规范

具体 workflow 继承 `BaseOtherModelWorkflow`，至少实现：

```python
class ExampleWorkflow(BaseOtherModelWorkflow):
    def quant(self, output_dir: str, device: str, config_overrides=None) -> QuantResult:
        ...

    def export(self, quant_result: QuantResult, output_dir: str, device: str, config_overrides=None) -> ExportResult:
        ...

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any) -> str:
        ...
```

### 配置读取

不要写回 `workflow_config.data`。推荐做法：

```python
workflow_config = self.workflow_config.with_overrides(config_overrides)
export_cfg = workflow_config.build_export_dict()
```

如果需要派生字段，例如把量化后的模型目录写入 `export.model.hf_model`，只修改局部 copy：

```python
model_cfg = copy.deepcopy(export_cfg["model"])
model_cfg["hf_model"] = self._resolve_export_model_dir(quant_result)
```

### quant()

`quant()` 只处理量化。

- 如果模型没有独立量化阶段，返回 skipped `QuantResult`。
- 不要在 `quant()` 中导出 HMONNX。
- 不要在 `quant()` 中生成 golden。

### export()

`export()` 只处理导出。

要求：

- 调用 `workflow_config.dump()` 将本次实际使用的 YAML dump 到 `output_dir`。
- dump 文件名建议使用 `workflow_config.name`，不要另起一套 `effective` 命名。
- 写出顶层 `export_meta_info.json`。
- 不生成 golden。
- 输出目录层级不要随意新增；除非迁移方案明确要求，否则保持旧布局兼容。
- 子模型原来有 `meta.json` 就保留；原来没有则不新增为强制要求。

示例：

```python
work_dir = Path(output_dir)
work_dir.mkdir(parents=True, exist_ok=True)
config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))
...
meta_file = work_dir / "export_meta_info.json"
meta_file.write_text(json.dumps(meta, indent=4, ensure_ascii=False), encoding="utf-8")
return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)
```

### dump_golden()

`dump_golden()` 只处理 golden。

要求：

- 通过 `export_result.work_dir` 找到导出产物。
- 读取 `export_meta_info.json` 和必要的子模型 metadata。
- 覆盖所有已导出的图，不只覆盖主图。
- 对每个图清理或覆盖原 golden 目录，保证可重复生成。
- 输入构造方式尽量对齐迁移前的 golden 或导出脚本。
- 如果旧流程使用全 0 或随机 dummy 输入，迁移后使用同样构造方式即可，不要求随机值逐 bit 一致。
- 对不需要用户输入的模型，子类可以把 `input_messages` 设为默认 `None`；基类协议不需要因此改变。

示例：

```python
def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any = None) -> str:
    work_dir = Path(export_result.work_dir)
    meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    ...
    return str(work_dir)
```

不要新增 `golden_meta_info.json` 作为通用要求。只有旧实现已经有类似 metadata，或某个模型明确需要，才保留对应行为。

## 示例脚本规范

迁移后的 workflow 示例建议满足：

- `--model-dir` 必填，不写本机默认模型路径。
- `--config-path` 必填且建议提供默认值，例如 `configs_merak/workflows/xh2a/other_models/<model_name>/<config_name>.yaml`。
- `--overwrite` 只删除导出目录，统一使用 `_remove_output_dir_if_needed()`。
- 提供 `--dump-golden`，在 `export()` 后调用 `workflow.dump_golden(...)`。
- 使用 `AutoWorkflow`，不要直接绑定子包 Auto。

示例：

```python
def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


...
workflow = AutoWorkflow.from_config(...)
quant_result = workflow.quant(...)
export_result = workflow.export(...)
if args.dump_golden:
    workflow.dump_golden(export_result=export_result, device=args.device)
```

README 中不要出现本机 conda 环境名或本机绝对路径。使用 `<env_name>`、`<gpu_id>`、`<model_dir>`、`<audio_file>` 这类占位符，并说明如何安装环境、如何启动脚本。

## Demo、评估和多图调度

如果模型导出多个 HMONNX 图，图之间的调度通常在 demo 或 inference adapter 中实现。HMONNX runtime 不会自动理解业务级调度关系。

迁移 demo 时需要确认：

- 是否需要 encoder -> prefill -> decode 调度。
- KV cache 是否需要包装成 runtime 要求的类型。
- prefill/decode 是否使用不同输入形状。
- 流式模型是否需要额外 stateful 图。
- demo 是否还硬编码旧产物目录、旧模型名或旧 metadata 路径。
- eval 和 analysis 是否读取新的 `export_meta_info.json`。

Qwen3-ASR 的两种 HMONNX 推理方式就是脚本侧自行调度多图：一种分段离线推理，一种累计音频加文本 prefix 推理。两者通常不需要导出不同主模型，但导出配置中的音频长度和 prefix 预算要覆盖对应场景。

Qwen3-TTS 的流式 demo 需要额外 stateful decoder 图。默认 YAML 可直接包含该组件，避免用户为流式场景修改其他无关配置。

## 依赖迁移规则

迁移完成后，模型包运行期不应依赖 `./examples`、`./xh_model_zoo`、`./configs`。

必须检查：

```bash
rg -n "xh_model_zoo|(^|[\"'/])examples([\"'/]|$)|(^|[\"'/])configs([\"'/]|$)|spec_from_file_location|importlib.util" \
  xhmodel_merak/xh_other_model/models/<model_name>
```

如有命中：

- 旧 helper 代码应搬到当前模型包，例如 `_export_utils.py`。
- 不要用 `spec_from_file_location` 或 `importlib.util` 动态加载 `./examples`、`./xh_model_zoo`、`./configs` 下的文件。
- 不要修改仓库外部依赖源码来临时跑通迁移。
- 如果需要第三方包，例如 `qwen_asr`、`qwen_tts`、`lerobot`，应在环境安装说明中写明，而不是依赖本机 editable 路径。

## 产物和 metadata

推荐导出目录结构：

```text
<output_dir>/
  <workflow_config_name>.yaml
  export_meta_info.json
  <component_a>/
    ...
  <component_b>/
    ...
```

规则：

- 顶层 `export_meta_info.json` 是统一入口。
- 子模型 `meta.json` 或 `export_meta_info.json` 只在兼容旧实现或子模型工具链需要时保留。
- 导出文件名包含 `target_device` 和 `quant_type`。

## 常见经验和风险

### YAML 精度字段没有真正生效

不要只在 YAML 中添加 `quant_type`。必须确认该字段传到每个子模型的导出函数，并体现在 HMONNX 文件名中。Qwen3-ASR 的 prefill 和 decode 都要体现 `prefill_decode.quant_type`。

### golden 覆盖不完整

多子模型迁移时，golden 要覆盖所有导出的 HMONNX：

- LLM prefill/decode。
- encoder/frontend/tokenizer。
- projection。
- stateful decoder。
- 其他辅助图。

Qwen3-TTS 需要覆盖 Talker、CodePredictor、TextProjection、SpeechTokenizer、Base frontend、StatefulDecoder。

### runtime 或 converter 限制

如果 HMONNX runtime 暴露算子或 metadata 问题，优先判断是否影响模型语义。

- 如果是 runtime metadata-only 问题，可在 golden 生成路径做局部 workaround，但不要改写已导出的 HMONNX。
- 如果是模型导出会产生 runtime 不支持的算子，优先做最小模型代码修复，并说明与旧导出的差异。
- 不要为了跑通测试大范围重构模型实现。

### traceable module 重复注册

同进程导出多个相近模型时，可能出现 traceable module 重复注册或错误复用。PI05 的 Gemma2B 和 GemmaExpert 属于这类情况。可在模型适配层做局部清理或隔离，避免修改全局依赖包源码。

### 本机环境污染

不要修改仓库外部依赖源码来跑通测试。例如不应改写某个本机 `customized_models/.../lerobot`。如果环境里 editable 安装了错误来源，应修正环境安装来源，而不是把外部改动作为迁移的一部分。

## 迁移流程清单

### 1. 盘点旧实现

- 找到旧模型类、注册方式、builder 用法。
- 找到旧 py config 和完整继承链。
- 找到导出脚本、golden 脚本、demo、streaming demo、eval、analysis、README。
- 列出全部子模型和 HMONNX 图。
- 列出需要搬迁的 helper。

### 2. 建立模型包

- 创建 `xhmodel_merak/xh_other_model/models/<model_name>/`。
- 迁移模型适配类、HMONNX inference adapter、导出 helper。
- 如果旧模型依赖尚未迁移的 `xh_model_zoo` 公共 helper，优先在当前模型子目录实现所需子集，不要默认扩展 `xh_other_model` 公共目录。
- 注册主模型类，并设置 `WORKFLOW_CLS`。
- 确认 `scan_model_types.py` 能扫描到注册类型。

### 3. 转 YAML

- 完整展开旧 py config。
- 固定 `quant`、`export`、`export.target_device`、`export.model.type`。
- 保留 `export.model` 作为旧 `MODELS.build()` 的完整 config。
- 为每个子模型提供 YAML 可控精度。

### 4. 实现 workflow

- `quant()`、`export()`、`dump_golden()` 职责分离。
- 不写回 `workflow_config.data`。
- 导出时 dump workflow YAML。
- 写顶层 `export_meta_info.json`。
- 文件名包含芯片架构和精度。

### 5. 迁移示例和 README

- 示例脚本使用 `AutoWorkflow`。
- `--model-dir` 必填。
- 默认输出目录使用 `work_dirs/...`。
- 支持 `--dump-golden`。
- README 使用通用环境安装说明，不写本机路径。

### 6. 验证

至少执行：

```bash
python -m py_compile xhmodel_merak/xh_other_model/models/<model_name>/*.py
python -m py_compile examples_merak/<domain>/<model_name>/*.py
```

运行导出：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/<domain>/<model_name>/<model_name>_workflow.py \
  --model-dir <model_dir> \
  --device cuda:0 \
  --overwrite
```

运行 golden：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD \
python examples_merak/<domain>/<model_name>/<model_name>_workflow.py \
  --model-dir <model_dir> \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

再按模型情况运行：

- HMONNX demo。
- streaming demo。
- eval 小样本。
- analysis 脚本。
- 产物目录检查。

最终检查旧依赖：

```bash
rg -n "xh_model_zoo|(^|[\"'/])examples([\"'/]|$)|(^|[\"'/])configs([\"'/]|$)|spec_from_file_location|importlib.util" \
  xhmodel_merak/xh_other_model/models/<model_name>
```

如果同配置下能比较旧产物和新产物，应比较 HMONNX 文件名、图输入输出、metadata 和 demo 输出。无法做到 bitwise 一致时，应记录差异原因。
