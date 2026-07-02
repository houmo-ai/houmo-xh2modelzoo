# Gemma4 Series model implementation 大一统设计与迁移边界

日期：2026-06-18  
JIRA：QTL-384  
范围：`/data01/datasets/gemma-4-E4B-it`、`/data01/datasets/gemma-4-31B-it`、`/data01/datasets/gemma-4-26B-A4B-it`，以及 `xhmodel_merak/xh_llm/models/gemma4_series` 目标实现。


## 0.1 2026-06-18 当前落地状态

本设计已按 QTL-384 迁入 `gemma4_series`，当前实现不再是单纯
workflow facade。`xhmodel_merak/xh_llm/models/gemma4_series` 已自拥有：

- `xh_gemma4_series_config.py`：统一 config/meta/variant 注入；
- `variants.py`：E4B、31B、26B-A4B 差异识别；
- `gemma4_series_processor.py`：本地 processor；
- `data_preprocess.py`：本地 text/image/video/audio scatter 与 E4B PLE；
- `gemma4_series_vision_model.py`：image/video padded ViT；
- `gemma4_series_audio_model.py` + `_audio_model_impl.py`：E4B audio；
- `llm_text.py` + `_llm_model_impl.py`：text bridge、KV cache、wrap modules；
- `gemma4_series_llm_model.py`：直接继承 `VisionLLMModel` 的唯一 public 主类；
- `gemma4_series_hmonnx_inference.py`：本地 HMONNX runtime；
- `workflow.py`：统一 quant/export/golden workflow。

当前 `gemma4_series` 目录内不再依赖 `gemma4` / `gemma4e` /
`gemma4_moe` / `_legacy_adapters` 作为实现来源；旧目录只作为历史
legacy 代码保留，不再是 Gemma4 Series 新功能落点。public master
registration 由 `Gemma4ForConditionalGeneration -> gemma4_series` 负责。

已验证：

```bash
python -m py_compile xhmodel_merak/xh_llm/models/gemma4_series/*.py \
  tests/gemma4_merak_llm_wrapper_test.py
pytest -q tests/gemma4_merak_llm_wrapper_test.py -k gemma4_series --tb=short
pytest -q tests/gemma4_merak_llm_wrapper_test.py --tb=short
```

当前 wrapper 回归为 `46 passed`。重型三 checkpoint HMONNX export/generate
仍需在 GPU 资源和量化产物齐备后作为 e2e 验收单独执行。

> 说明：下文第 0-12 节保留迁移设计的历史脉络；其中
> “当前/现有”若与本节冲突，均指 2026-06-18 迁移启动前状态。
> 已落地状态以本节为准。

## 0. 结论先行（迁移设计起点）

迁移前仓库已经做了一层 `examples_merak/llm/gemma4_series` +
`Gemma4Workflow` 的 **workflow facade 统一**，但还没有完成用户要求的
**model implementation 大一统**。

迁移前实际链路是：

```text
examples_merak/llm/gemma4_series/gemma4_workflow_demo.py
  -> AutoLLMWorkflow.from_config(...)
  -> configs_merak/workflows/xh2a/llm_models/gemma4/{e4b,31b,26b_a4b}/*.yaml
  -> xhmodel_merak/xh_llm/models/gemma4/workflow.py
  -> xhmodel_merak/xh_llm/models/{gemma4,gemma4e,gemma4_moe} legacy 实现
```

这只能降低调用侧复杂度，不能解决底层维护问题。真正目标应改为：

```text
examples_merak/llm/gemma4_series/*                  # 薄 demo / 验收入口
  -> AutoLLMWorkflow.from_config(...)
  -> configs_merak/workflows/xh2a/llm_models/gemma4_series/{e4b,31b,26b_a4b}/*.yaml
  -> xhmodel_merak/xh_llm/models/gemma4_series/workflow.py
  -> xhmodel_merak/xh_llm/models/gemma4_series/*    # 唯一 Gemma4 series 实现
```

`gemma4/gemma4e/gemma4_moe` 后续只允许作为过渡期代码来源或兼容 shim，不能继续作为新功能落点。

---

## 1. 迁移前实现到底做到了什么

### 1.1 examples facade

`examples_merak/llm/gemma4_series/gemma4_workflow_demo.py` 维护了统一 preset：

| preset | HF checkpoint | 迁移前 YAML |
|---|---|---|
| `e4b` | `/data01/datasets/gemma-4-E4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4/e4b/gemma4_e4b_full.yaml` |
| `31b` | `/data01/datasets/gemma-4-31B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4/31b/gemma4_31b_full.yaml` |
| `26b-a4b` | `/data01/datasets/gemma-4-26B-A4B-it` | `configs_merak/workflows/xh2a/llm_models/gemma4/26b_a4b/gemma4_26b_a4b_full.yaml` |

表面 API 是统一的：

```python
workflow = AutoLLMWorkflow.from_config(...)
quant_result = workflow.quant(...)
export_result = workflow.export(...)
workflow.dump_golden(...)
```

`base-export` 不是绕开 workflow，而是标准 `quant -> export` 两阶段，其中 quant 阶段用：

```python
config_overrides={"quant": None}
```

显式跳过重型 quant，仅把原始 HF checkpoint 作为 export 输入。

### 1.2 workflow facade

迁移前 `xhmodel_merak/xh_llm/models/gemma4/workflow.py` 已经把三种 preset 纳入同一个 `Gemma4Workflow`：

- `quant=None` 只允许显式 base validation；
- 默认 quant 仍要求 `group_size=64`；
- export 模型默认 `context_max_length=2048`、`prefill_chunk_length=256`；
- visual 遵循 padded ViT：image `[1,2520,768] -> [1,280,*]`，video `[1,630,768] -> [1,70,*]`；
- `dump_golden` 支持 text/image/video/audio message 构造。

### 1.3 generate 验收层

`examples_merak/llm/gemma4_series/generate.py` 迁移前已经作为统一 HMONNX demo 支持：

- text：主 prefill/decode；
- image：`visual` + 主 prefill/decode；
- video：`video_visual` + 主 prefill/decode；
- audio：`audio` + 主 prefill/decode，仅 E4B 支持。

但在迁移前这仍是 demo/runtime 层统一，不是模型目录统一。

---

## 2. 迁移前实现为什么不满足“大一统”

### 2.1 底层仍是三套实现

迁移前实际模型实现仍分散在：

```text
xhmodel_merak/xh_llm/models/gemma4/      # 31B dense 相关 + 现有 workflow facade
xhmodel_merak/xh_llm/models/gemma4e/     # E4B PLE/audio/KV-share 相关
xhmodel_merak/xh_llm/models/gemma4_moe/  # 26B-A4B MoE with-mask 相关
```

这会导致：

1. 代码复用靠 import 顺序和交叉引用，不是清晰边界；
2. 同一 public model type 存在重复注册风险；
3. E4B/31B/26B 的差异没有被显式建模；
4. 后续改 mask、KV cache、ViT、audio、PLE 时仍可能三处同步修改；
5. examples facade 越写越厚，违背“上游 imodelzoo 容易集成”的目标。

### 2.2 注册冲突是必须优先拆掉的问题

迁移前注册映射由 `xhmodel_merak/xh_llm/scan_model_types.py` 扫描所有 `@register_llm_model(...)` 得到：

```python
model_types = {result["model_type"]: result["module_name"] for result in results}
```

重复 key 会被后扫描模块覆盖。历史上已经出现：

```text
gemma4/gemma4_llm_model.py   -> Gemma4ForConditionalGeneration
gemma4e/gemma4_llm_model.py  -> Gemma4ForConditionalGeneration
```

这不是 checkpoint 选择问题，而是注册 key 冲突。新增 `gemma4_series` 后不能再依赖扫描顺序；必须保证：

```text
Gemma4ForConditionalGeneration 只有 gemma4_series 一个 master 注册来源
```

旧目录若需要保留兼容，只能：

- 删除同名 public master decorator；或
- 改成 legacy 私有 key；或
- 由 `gemma4_series` 统一注册旧 alias 并内部转发。

不能让旧目录继续注册同一个 public key。

---

## 3. 目标目录结构

目标新增目录：

```text
xhmodel_merak/xh_llm/models/gemma4_series/
  __init__.py
  variants.py
  xh_gemma4_series_config.py
  data_preprocess.py
  gemma4_series_llm_model.py
  gemma4_series_vision_model.py
  gemma4_series_audio_model.py
  gemma4_series_processor.py
  gemma4_series_hmonnx_inference.py
  workflow.py
  _text_dense_impl.py
  _text_ple_impl.py
  _text_moe_impl.py
  _vision_impl.py
  _audio_impl.py
  _hf_compatible.py
  _legacy_adapters.py        # 过渡期文件，迁移完成后应尽量删除
```

设计原则：

- `__init__.py` 只导出 public API，不 eager import workflow 重依赖；
- `workflow.py` 只属于 `gemma4_series`，不再挂在 `gemma4`；
- `variants.py` 是三模型差异的唯一事实源；
- `xh_gemma4_series_config.py` 是唯一 config/meta 定义；
- text/vision/audio 实现可拆文件，但 public model class 只能有一个；
- `_legacy_adapters.py` 只能在迁移期复用旧类，不能成为长期核心。

---

## 4. VariantSpec：先把差异显式化

新增：

```python
@dataclass(frozen=True)
class Gemma4SeriesVariantSpec:
    name: Literal["e4b", "31b", "26b_a4b", "unknown"]
    topology: Literal["dense", "moe"]
    has_audio: bool
    has_image: bool
    has_video: bool
    has_per_layer_input: bool
    has_shared_kv_layers: bool
    attention_k_eq_v: bool
    visual_hidden_size: int | None
    audio_feature_size: int | None
    sliding_window: int | None
    local_attention_window_size: int | None
    global_attention_window_size: int | None
```

解析函数：

```python
def resolve_gemma4_series_variant(hf_config: Mapping[str, Any]) -> Gemma4SeriesVariantSpec:
    ...
```

判定依据：

| 差异点 | E4B | 31B | 26B-A4B |
|---|---:|---:|---:|
| `text_config.enable_moe_block` | false | false | true |
| `audio_config` | 有 | 无 | 无 |
| `hidden_size_per_layer_input` | 256 | 0 | 0 |
| `num_kv_shared_layers` | 18 | 0 | 0 |
| `attention_k_eq_v` | false | true | true |
| video | 支持 | 可导出 visual graph，但验收按模型能力决定 | 可导出 visual graph，但验收按模型能力决定 |

注意：variant 不能靠路径名猜；路径名只可作为日志/debug 辅助。权威来源必须是 HF `config.json`。

---

## 5. 统一 config/meta 边界

### 5.1 Public config

新增：

```python
class XHGemma4SeriesModelConfig(VisionLLMModelConfig): ...
class XHGemma4SeriesVisualConfig(HFModelConfig): ...
class XHGemma4SeriesAudioConfig(HFModelConfig): ...
```

外部 YAML 只使用：

```yaml
export:
  model:
    model_type: Gemma4ForConditionalGeneration
    model_cls: XHGemma4SeriesModelConfig  # 如果当前框架需要显式 config class
```

原则：

- 31B/E4B/26B-A4B 不再暴露不同 config class；
- MoE 的 router/expert/sliding-window 参数由 `variant` + HF config 自动填充；
- E4B 的 audio/PLE/shared-KV 参数由 `variant` + HF config 自动填充；
- visual/video_visual config 遵守 `docs/gemma4_vit_padded_input_design_20260616.md`，不回退 448 固定方图；
- video 必须单独 `video_visual_config`，不得 pad 到 image ViT。

### 5.2 Public meta

新增：

```python
class Gemma4SeriesModelMeta(VLLMModelMeta):
    variant: str
    capabilities: dict[str, bool]
    visual_config: Gemma4SeriesVisualModelMeta | None
    video_visual_config: Gemma4SeriesVisualModelMeta | None
    audio_config: Gemma4SeriesAudioModelMeta | None
    per_layer_input_embedding: str | None
```

meta 必须让 runtime 不再猜旧模型目录：

```json
{
  "variant": "e4b",
  "capabilities": {
    "text": true,
    "image": true,
    "video": true,
    "audio": true
  }
}
```

---

## 6. 统一 model class 边界

### 6.1 唯一 public master model

新增唯一主注册：

```python
@register_llm_model("Gemma4ForConditionalGeneration")
class XHGemma4SeriesModel(VisionLLMModel):
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.gemma4_series.workflow:XHGemma4SeriesHMONNXWorkflow"
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    CONFIG_CLS = XHGemma4SeriesModelConfig
    META_CLS = Gemma4SeriesModelMeta
    HMONNXINFERENCE_CLS = XHGemma4SeriesHMONNXModel
```

禁止新代码直接注册：

```text
Gemma4ForConditionalGeneration_with_mask
Gemma4ForConditionalGeneration_visual
Gemma4EForConditionalGeneration_visual
Gemma4ForConditionalGeneration_audio
```

这些只能作为内部 submodel / 兼容 alias，不得成为 imodelzoo 面向的 public API。

### 6.2 内部组合

`XHGemma4SeriesModel.__init__` 只负责组合，不塞满分支逻辑：

```python
self.variant = resolve_gemma4_series_variant(...)
self.text = build_text_adapter(self.variant, config)
self.visual = XHGemma4SeriesVisionModel(config.visual_config)
self.video_visual = XHGemma4SeriesVisionModel(config.video_visual_config)
self.audio = XHGemma4SeriesAudioModel(config.audio_config) if variant.has_audio else None
```

文本差异放入 adapter：

| adapter | 使用模型 | 责任 |
|---|---|---|
| `DenseTextAdapter` | 31B | dense MLP、K=V、full/sliding mask |
| `PLETextAdapter` | E4B | per-layer input、shared KV skip、audio/mm token type |
| `MoeTextAdapter` | 26B-A4B | router/experts、MoE with-mask、local/global mask |

### 6.3 E4B PLE 原则

E4B 不能把 per-layer embedding 后的大量 per-layer input 留在 host 侧长期拼装。

目标：

- token embedding 可以作为输入准备的一部分；
- `embed_tokens_per_layer` 产出的 per-layer input 以及每层 gate/projection 应尽量合入主 ONNX；
- runtime meta 中允许保留 `per_layer_input_embedding` 权重/路径作为过渡，但最终 forward 边界必须以“主 ONNX 内部完成 per-layer input 消费”为目标；
- 不允许为了省事把 E4B 退回成外部每层输入列表长期协议。

---

## 7. 统一 vision/audio/processor 边界

### 7.1 Vision

新增：

```python
class XHGemma4SeriesVisionModel(VisionLLMModel): ...
```

必须遵守定版 ViT 方案：

| 子图 | 输入 shape | 输出 token |
|---|---|---|
| `visual` | `[1,2520,768]` | 280 |
| `video_visual` | `[1,630,768]` | 70 |

要求：

- image/video 两套静态图；
- 可以共享 HF 权重来源；
- 不允许 video pad 到 image 图；
- 不再引入 448 固定尺寸配置；
- Host 侧 processor 生成 `pixel_position_ids`、`pool_indices`、`attention_mask`。

### 7.2 Audio

新增：

```python
class XHGemma4SeriesAudioModel(VisionLLMModel): ...
```

只对 `variant.has_audio=True` 启用。默认音频窗口遵循 HF processor / 当前 legacy 默认：

```text
sampling_rate = 16000
feature_size = 128
input_feature_length 默认约 2999，对应 30s window
```

如需更长音频，通过 `audio_config.input_feature_length` 显式配置，不隐式动态化。固定长度 audio 子图需要同时输入：

```text
input_features
input_features_mask
```

mask 不能省略；静音/padding 由 mask 区分。

### 7.3 Processor

新增：

```python
class XHGemma4SeriesProcessor(...): ...
```

职责：

- 统一 text/image/video/audio message；
- image/video 输出遵守 padded ViT 协议；
- audio 输出固定长度 feature + mask；
- 对外不暴露旧 `gemma4e/gemma4_moe` processor。

---

## 8. 统一 HMONNX runtime 边界

新增：

```python
class XHGemma4SeriesHMONNXModel(VisonLLMHMONNXModel): ...
class Gemma4SeriesKVCacheMixinHMONNX(KVCacheMixin): ...
```

runtime 初始化只看 `Gemma4SeriesModelMeta`：

```python
self.variant = meta.variant
self.visual = load_if_present(meta.visual_config)
self.video_visual = load_if_present(meta.video_visual_config)
self.audio = load_if_present(meta.audio_config)
```

KV cache：

- 31B：`attention_k_eq_v=True`，K=V 权重/投影语义保持一致，cache 输出仍按接口显式给 key/value；
- E4B：`num_kv_shared_layers>0`，shared-KV 层不单独分配/输出独立 KV cache；
- 26B-A4B：MoE local/global attention mask 与 slice-window cache 逻辑保留，但通过 `MoeTextAdapter` 管理；
- 所有模型导出规则固定：`context_max_length=2048`，`prefill_chunk_length/input_sequence_length=256`；
- slice-window 输出必须按 `slice_window + sequence_length` 规则覆盖，不能只看输入全量长度。

---

## 9. workflow 新流程目标

### 9.1 目标 workflow 入口

新增：

```python
class Gemma4SeriesWorkflow(BaseLLMWorkflow): ...
class XHGemma4SeriesHMONNXWorkflow(Gemma4SeriesWorkflow): ...
```

`XHGemma4SeriesModel.WORKFLOW_CLS` 指向它。

迁移前 `Gemma4Workflow` 中有价值的逻辑迁移到 `gemma4_series/workflow.py`：

- recommended configs；
- AutoRound / existing_hf / base-export quant 策略；
- export config validation；
- `dump_golden` message 构造；
- strict e2e 约束说明。

迁移后旧 `gemma4/workflow.py` 只能是兼容 wrapper：

```python
from xhmodel_merak.xh_llm.models.gemma4_series.workflow import *
```

且不得再作为新配置的目标。

### 9.2 YAML 目标路径

新增配置目录：

```text
configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml
configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml
configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml
```

旧路径可保留一段时间，但 examples 和文档必须切到新路径。

---

## 10. 迁移阶段

### Phase 0：设计冻结与验收标准

产物：本文档。  
目标：明确不再在旧三目录做新功能，只允许为迁移拆注册/搬代码。

验收：

- 文档明确迁移前 facade 与目标 implementation 的区别；
- 文档列出目录、类边界、注册策略、workflow 迁移策略；
- 文档纳入 ViT/audio/PLE/KV/MoE 关键约束。

### Phase 1：建立 `gemma4_series` skeleton

产物：

```text
xhmodel_merak/xh_llm/models/gemma4_series/__init__.py
xhmodel_merak/xh_llm/models/gemma4_series/variants.py
xhmodel_merak/xh_llm/models/gemma4_series/xh_gemma4_series_config.py
xhmodel_merak/xh_llm/models/gemma4_series/workflow.py
```

要求：

- `Gemma4ForConditionalGeneration` 唯一 public master 注册指向 `gemma4_series`；
- 旧 `gemma4/gemma4e/gemma4_moe` 去掉同名 public master 注册或改为私有 legacy key；
- `AutoLLMWorkflow.from_config()` 对三套新 YAML 都解析到 `Gemma4SeriesWorkflow`；
- 轻量测试覆盖注册映射不冲突。

### Phase 2：迁移 config/meta/workflow

要求：

- `XHGemma4SeriesModelConfig` 吸收当前 `gemma4/xh_gemma4_config.py` 与 `gemma4_moe/xh_gemma4_moe_config.py` 的差异；
- `Gemma4SeriesModelMeta` 写出 `variant` 和 `capabilities`；
- workflow recommended configs 切到 `gemma4_series` 路径；
- examples preset 切到新 YAML；
- 旧 workflow 只做兼容 re-export。

### Phase 3：迁移 model/processor/runtime

要求：

- `XHGemma4SeriesModel` 组合 text/visual/video_visual/audio；
- 31B/E4B/26B-A4B 通过 `VariantSpec` 分发到 adapter；
- E4B PLE 只把 embedding lookup 保存为 artifact，`per_layer_inputs`
  作为主 ONNX 输入，embedding 之后的 per-layer 计算留在主图内部；
- image/video padded ViT 从 series vision 走；
- audio 从 series audio 走；
- HMONNX runtime 从 `Gemma4SeriesModelMeta` 加载，不再识别旧目录类名。

### Phase 4：清理旧入口与全量验收

要求：

- `gemma4/gemma4e/gemma4_moe` 不再有同名 public master 注册；
- examples 不再 import legacy model 包；
- 三模型按规则重导出：`context=2048`、`input_sequence_length=256`；
- `to_quanted_aligned` 必跑；
- E4B 跑 text/image/video/audio generate；
- 31B/26B 跑 text/image generate；
- 所有 e2e prompt token > 1024，且为真实 QA。

---

## 11. P8 任务 Prompt（六要素）

### Task A：注册与 skeleton

- 背景：当前 `Gemma4ForConditionalGeneration` 被旧目录重复注册，导致扫描顺序覆盖。
- 目标：新增 `gemma4_series` skeleton，并让 public master 注册唯一指向 series。
- 输入：本文档、`scan_model_types.py`、旧三目录 `__init__`/decorator。
- 约束：不实现重型导出；不得靠 import 顺序解决冲突。
- 验收：轻量脚本打印 `MODEL_TYPE_MAPPING_MODULES["Gemma4ForConditionalGeneration"] == "gemma4_series"`；旧同名注册不存在。
- 风险：改注册可能影响旧配置；需要在 series 内提供兼容 alias 或迁移 YAML。

### Task B：config/meta/workflow 迁移

- 背景：当前 workflow facade 放在 `gemma4/workflow.py`，不是 series 目录。
- 目标：把 workflow 与 config/meta 迁到 `gemma4_series`，新 YAML 指向 series。
- 输入：`gemma4/workflow.py`、`gemma4/xh_gemma4_config.py`、`gemma4_moe/xh_gemma4_moe_config.py`。
- 约束：保留 `quant=None` 仅 base validation；默认 `quant_scheme=w8a8h1_sefp`；context/input 固定 2048/256。
- 验收：`AutoLLMWorkflow.from_config()` 三 preset 均返回 `Gemma4SeriesWorkflow`。
- 风险：WorkflowConfig override 对新增 key 严格，需要避免无意放宽全局校验。

### Task C：text adapter 迁移

- 背景：Dense、PLE、MoE 当前分散在三套 text 实现。
- 目标：通过 `VariantSpec` 建立统一 text 分发边界；当前落地先用
  `XHGemma4SeriesModel` 本地化 dense/PLE/MoE helper，后续如继续
  拆 adapter，必须保持 public API 不变。
- 输入：`gemma4/gemma4_llm_model.py`、`gemma4e/gemma4_llm_model.py`、`gemma4_moe/gemma4_moe_with_mask_model.py`。
- 约束：E4B PLE 不能把 embedding 后的 per-layer 计算退回 host；
  MoE router/expert 不能塞进 dense 分支。
- 验收：三种 tiny config 单测能构建正确 adapter；KV shape 单测覆盖 shared-KV 与 MoE local/global。
- 风险：一次搬完容易破坏已验证路径；允许第一步 adapter 包旧实现，但必须保留删除计划。

### Task D：vision/audio/processor/runtime 迁移

- 背景：padded ViT 方案已定版，audio 当前只在 E4B legacy 中完整。
- 目标：series vision/audio/processor/runtime 统一加载 image/video/audio 子图。
- 输入：`docs/gemma4_vit_padded_input_design_20260616.md`、`gemma4e/gemma4_audio_model.py`、`gemma4e/gemma4_hmonnx_inference.py`。
- 约束：video 单独 630 patch 图；audio 固定长度必须带 mask；31B/26B 不声明 audio capability。
- 验收：E4B text/image/video/audio generate 真实 QA 通过；31B/26B text/image 通过。
- 风险：audio 质量测试不能用单一合成音频代表完整 ASR，但链路必须正确。

---

## 12. 不做什么

- 不继续在 `gemma4/gemma4e/gemma4_moe` 上加新 public API。
- 不靠 import 顺序覆盖注册 key。
- 不把 video pad 到 image ViT。
- 不用短 prompt / smoke test 当 e2e。
- 不把 `export.model.quant_scheme` 改成 `None` 来假装 base export。
- 不把 26B-A4B 作为单独 public workflow 暴露给上游。
