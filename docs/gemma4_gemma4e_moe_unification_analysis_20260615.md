# Gemma4 / Gemma4e / Gemma4-MoE 结构与实现大一统分析

> 日期：2026-06-15  
> 范围：`gemma-4-31B-it`、`gemma-4-E4B-it`、`gemma-4-26B-A4B-it`，以及 `xhmodel_merak.xh_llm.models.{gemma4,gemma4e,gemma4_moe}` 当前实现。  
> 结论先行：**31B 被解析成 gemma4e 的直接原因是注册 key 冲突，不是 checkpoint 本身变成了 E4B；三者可以做“统一入口 + config-driven 分支”，但不建议继续让 31B 与 E4B 共用同一个裸 `Gemma4ForConditionalGeneration` 注册 key。**

<callout emoji="⚠️" background-color="light-yellow" border-color="yellow">
本次分析只读代码与 checkpoint 配置/权重索引，不运行重型导出，不加载完整权重到 GPU。验证命令使用 `conda activate gemma4` 后的轻量 Python 检查。
</callout>

---

## 1. 现象与根因：为什么 31B 走到了 gemma4e

### 1.1 用户看到的现象

导出入口：

```python
# examples_merak/llm/gemma4/gemma4_xh_export_hmonnx.py:118-119
model_cfg: XHGemma4ModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
xh_model: XHGemma4Model = AutoLLMModel.from_pretrained(config=model_cfg)
```

直觉上，脚本在 `TYPE_CHECKING` 中标注的是：

```python
from xhmodel_merak.xh_llm.models.gemma4 import XHGemma4Model, XHGemma4ModelConfig
```

但运行时结果落到：

```text
xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config.XHGemma4ModelConfig
xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model.XHGemma4Model
```

### 1.2 实际加载链路

1. `gemma4_xh_export_hmonnx.py` 读取 mmengine/xhquant config。
2. `AutoLLMConfig.from_pretrained(cfg.model)` 优先使用 `cfg.model.model_type`；若为空才读 HF `config.json` 的 `architectures[0]`。
3. `AutoLLMConfig` 调 `get_config_class(base_config)`。
4. `get_config_class` 先调 `get_model_class`，`get_model_class` 通过 `MODEL_TYPE_MAPPING_MODULES` 决定要动态导入哪个 `models/<module>`。
5. `MODEL_TYPE_MAPPING_MODULES` 来自 `scan_model_types.get_support_all_model_types()`，对所有 `@register_llm_model(...)` 扫描后构建字典。
6. **同一个 key 重复出现时，后扫描到的模块覆盖前面的模块。**

关键代码证据：

| 文件 | 证据 |
|---|---|
| `xhmodel_merak/xh_llm/auto_llm_config.py:29-40` | `model_type` 为空时才由 `AutoConfig.architectures[0]` 推断，随后 `get_config_class(base_config)` |
| `xhmodel_merak/xh_llm/builder.py:25-54` | `MODEL_TYPE_MAPPING_MODULES[model_type]` 决定 `importlib.import_module(f".models.{module_name}")` |
| `xhmodel_merak/xh_llm/scan_model_types.py:67-74` | `model_types = {result["model_type"]: result["module_name"] for result in results}`；重复 key 字典覆盖 |
| `xhmodel_merak/xh_llm/models/gemma4/gemma4_llm_model.py:281` | `@register_llm_model("Gemma4ForConditionalGeneration")` |
| `xhmodel_merak/xh_llm/models/gemma4e/gemma4_llm_model.py:495` | 也注册 `@register_llm_model("Gemma4ForConditionalGeneration")` |

### 1.3 轻量实验证据

执行环境：

```bash
conda activate gemma4
PYTHONPATH=$PWD python <轻量脚本>
```

输出摘录：

```text
cfg.model.model_type = Gemma4ForConditionalGeneration
cfg.model.hf_model = /path/to/gemma4-mode1
mapping Gemma4ForConditionalGeneration = gemma4e
AutoLLMConfig class = xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config XHGemma4ModelConfig
model class = xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model XHGemma4Model
```

扫描注册项得到：

```text
Gemma4ForConditionalGeneration => gemma4 (.../models/gemma4/gemma4_llm_model.py:281)
Gemma4ForConditionalGeneration => gemma4e (.../models/gemma4e/gemma4_llm_model.py:495)

Final mapping:
Gemma4ForConditionalGeneration => gemma4e
Gemma4ForConditionalGeneration_with_mask => gemma4_moe
```

### 1.4 根因判定

**根因是代码注册冲突。**

- 不是 31B checkpoint 的 `config.json` 写成了 Gemma4e。
- 不是 `hf_model_dir` 目录名触发了 Gemma4e。
- 不是 `TYPE_CHECKING` import 影响运行时类型。
- 是 `Gemma4ForConditionalGeneration` 同时被 `gemma4` 与 `gemma4e` 注册，且扫描映射最终指向 `gemma4e`。


### 1.5 旁证：`gemma4_series/export_hmonnx.py` 已经在用 import 顺序规避冲突

仓库里另一个 series 导出脚本也印证了这不是孤例：

- `examples_merak/llm/gemma4_series/export_hmonnx.py:87-92` 注释说明：导入 `gemma4_moe` 会传递加载 legacy `gemma4`，先注册 `Gemma4ForConditionalGeneration`；随后显式导入 `gemma4e`，让同名注册最终指向 gemma4e runtime，以便和 `generate.py` 的 runtime 对齐。
- 同文件定义：`MODEL_TYPE_MOE = "Gemma4ForConditionalGeneration_with_mask"`、`MODEL_TYPE_DENSE = "Gemma4ForConditionalGeneration"`、`MODEL_TYPE_VISUAL = "Gemma4ForConditionalGeneration_visual"`。

这说明当前代码已经存在“靠 import 顺序控制最终实现”的 workaround。它可以作为临时兼容手段，但不适合作为大一统后的长期注册策略。

---

## 2. 三个 checkpoint 的结构对比

### 2.1 总览表

| 项 | Gemma4-31B | Gemma4e-E4B | Gemma4-MoE 26B-A4B |
|---|---:|---:|---:|
| 路径 | `/data01/datasets/gemma-4-31B-it` | `/data01/datasets/gemma-4-E4B-it` | `/data01/datasets/gemma-4-26B-A4B-it` |
| 顶层 `model_type` | `gemma4` | `gemma4` | `gemma4` |
| `architectures` | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
| text hidden size | 5376 | 2560 | 2816 |
| text layers | 60 | 42 | 30 |
| attention heads | 32 | 8 | 16 |
| KV heads | 16 | 2 | 8 |
| head dim | 256 | 256 | 256 |
| dense intermediate | 21504 | 10240 | 2112 |
| max position | 262144 | 131072 | 262144 |
| sliding window | 1024 | 512 | 1024 |
| full/sliding 层数 | 10 full + 50 sliding | 7 full + 35 sliding | 5 full + 25 sliding |
| `attention_k_eq_v` | True | False | True |
| `num_kv_shared_layers` | 0 | 18 | 0 |
| `hidden_size_per_layer_input` | 0 | 256 | 0 |
| `enable_moe_block` | False | False | True |
| `num_experts` | None | None | 128 |
| `top_k_experts` | None | None | 8 |
| `moe_intermediate_size` | None | None | 704 |
| vision tower | 有，1152 hidden / 27 layers | 有，768 hidden / 16 layers | 有，1152 hidden / 27 layers |
| audio tower | 无 | 有，1024 hidden / 12 layers | 无 |
| processor | `Gemma4Processor` | `Gemma4Processor` | `Gemma4Processor` |
| 权重组织 | 2 分片 + index | 单个 `model.safetensors`，无 index | 2 分片 + index |

### 2.2 共同点

三者共享以下基础事实：

- 顶层 `model_type` 都是 `gemma4`。
- `architectures` 都是 `Gemma4ForConditionalGeneration`。
- tokenizer / processor 都走 `Gemma4Processor` / `GemmaTokenizer`。
- 都有 `image_token_id=258880`、`vision_soft_tokens_per_image=280`。
- text config 都有 full attention + sliding attention 的 `layer_types` 混合布局。
- rope 参数都区分：
  - full attention：`partial_rotary_factor=0.25`、`rope_theta=1000000.0`、`rope_type=proportional`
  - sliding attention：`rope_theta=10000.0`、`rope_type=default`

这说明三者适合共享：

1. 顶层 processor/tokenizer 入口；
2. 基础 text/vision/audio 子配置解析；
3. full/sliding attention mask 与 rope cache 框架；
4. HMONNX export 的通用工作流。

### 2.3 必须分叉的结构点

#### 2.3.1 31B：dense VLM，旧 Gemma4 代码原本覆盖得更贴近

31B 特征：

- dense MLP：`enable_moe_block=false`，无 experts/router。
- 无 audio tower。
- `attention_k_eq_v=True`，全局层可能 `v_proj=None`，K=V。
- `num_kv_shared_layers=0`，KV cache 可按普通逐层准备。
- 视觉塔尺寸较大：hidden 1152 / 27 layers。

权重索引证据：

```text
model.language_model.layers.0.mlp.down_proj.weight
model.language_model.layers.0.mlp.gate_proj.weight
model.language_model.layers.0.mlp.up_proj.weight
model.vision_tower.encoder.layers.0...
model.embed_vision.embedding_projection.weight
```

没有 `.experts` / `.router` / `audio_tower`。

#### 2.3.2 E4B：不是简单“小 31B”，而是 PLE + audio + KV shared 的 Gemma4e 分支

E4B 特征：

- dense MLP：无 MoE experts/router。
- 有 `audio_config`，权重有 `model.audio_tower.*`。
- `hidden_size_per_layer_input=256`，存在 per-layer input 机制。
- `num_kv_shared_layers=18`，实现上需要跳过 shared KV 层的独立 cache 分配。
- `attention_k_eq_v=False`，V 与 K 不能按 31B 的 K=V 简化处理。
- 视觉塔尺寸较小：hidden 768 / 16 layers。
- 单文件 `model.safetensors`，无 index，需要 loader 支持 safetensors header fallback。

权重名证据：

```text
model.audio_tower.layers.0.feed_forward1...
model.language_model.embed_tokens_per_layer.weight
model.language_model.layers.0.per_layer_input_gate.weight
model.language_model.layers.0.per_layer_projection.weight
model.vision_tower.encoder.layers.0...
```

仓库实现证据：

- `gemma4e/data_preprocess.py` 定义 `Gemma4PerLayerInputBuilder`，并在 forward 中插入 `per_layer_inputs`。
- `gemma4e/gemma4_llm_model.py:125-129` 按 `hidden_size_per_layer_input` 选择 PLE bridge 或 dense bridge。
- `gemma4e/gemma4_llm_model.py:508-510` 支持 `visual` 和 `audio` 都可选。
- `gemma4e/gemma4_llm_model.py:534-546` 遇到 `is_kv_shared_layer` 会跳过独立 KV cache shape。

因此 Gemma4e 当前代码方向是合理的，但它被注册成和 31B 同一个裸 key 后，会把 31B 也带进 E4B 的实现体系，这是混乱来源。

#### 2.3.3 26B-A4B：MoE VLM，必须走 MoE 专属实现

26B-A4B 特征：

- `enable_moe_block=true`
- `num_experts=128`
- `top_k_experts=8`
- `moe_intermediate_size=704`
- 无 audio tower。
- text 只有 30 层，但权重总量来自 experts。

权重索引证据：

```text
model.language_model.layers.0.experts.down_proj
model.language_model.layers.0.experts.gate_up_proj
model.language_model.layers.0.router.proj
model.language_model.layers.0.router.scale
model.language_model.layers.0.router.per_expert_scale
```

仓库实现证据：

- `gemma4_moe/gemma4_moe_with_mask_model.py:132` 注册为 `Gemma4ForConditionalGeneration_with_mask`。
- `gemma4_moe/_llm_model_impl.py:198-234` 将 HF router / experts 转换成 `MoeBlock` 所需权重。
- `gemma4_moe/_llm_model_impl.py:284-304` forward 中 dense MLP 与 MoE branch 并行组合。
- `gemma4_moe/gemma4_moe_hmonnx_inference.py` 有 local/global attention mask 组合逻辑。

MoE 不应与 dense 31B/E4B 共用同一 text layer forward 分支；最多共享 outer shell、processor、rope/mask 工具函数。

---

## 3. 当前三套实现方案对比

| 维度 | `models/gemma4` | `models/gemma4e` | `models/gemma4_moe` |
|---|---|---|---|
| 当前注册主 key | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration_with_mask` |
| 目标模型 | 31B dense VLM | E4B dense + audio/PLE/KV-share | 26B-A4B MoE VLM |
| text wrapper | 旧 dense text graph，full + sliding masks | dense / PLE bridge 自动分支 | MoE with-mask graph |
| visual | 强依赖 visual_config | visual 可选，支持 compact/offline full export modes | MoE visual model |
| audio | 无 | 有 `XHGemma4AudioModel` | 无 |
| per-layer input | 有少量 decoder 支持，但旧 31B 实际为 0 | 完整 `Gemma4PerLayerInputBuilder` 与 artifact | 无 |
| KV shared | 无专门 shared KV skip | 有 `is_kv_shared_layer` skip | 无 |
| MoE | 明确无 MoE，兼容 generate no-op | 明确 dense，避免 GPTQ dense 被误走 MoE GPTQ | 专用 router + experts + MoeBlock |
| HMONNX class | `XHGemma4HMONNXModel` | `XHGemma4_HMONNXModel` | `XHGemma4MoeWithMaskHMONNXModel` |
| 主要问题 | 与 gemma4e 重名注册，被覆盖 | 适配 E4B 合理，但不该吞掉 31B | 主 key 已分开，但 visual key 仍和其他重复 |

---

## 4. “大一统”推荐方案

### 4.1 不推荐方案：继续让多个模块注册同一个裸 key

不推荐继续让 `gemma4` 与 `gemma4e` 都注册：

```python
@register_llm_model("Gemma4ForConditionalGeneration")
```

原因：

1. HF 的 `architectures` 对三者都是 `Gemma4ForConditionalGeneration`，这个字段不能区分 31B / E4B / MoE。
2. 仓库扫描注册时重复 key 会发生覆盖，最终行为取决于文件排序，不稳定且不可读。
3. 运行时类名都叫 `XHGemma4Model` / `XHGemma4ModelConfig`，只看类名看不出来自哪个 module。
4. 一旦加更多变体，冲突会指数级恶化。

### 4.2 推荐方案：统一 family resolver + 变体 key 分流

建议将 Gemma4 family 做成两层：

```text
HF checkpoint config
        │
        ▼
Gemma4 family resolver
        │
        ├── dense_31b / legacy_dense  → xhmodel_merak.xh_llm.models.gemma4
        ├── e4b / ple_audio_dense     → xhmodel_merak.xh_llm.models.gemma4e
        └── moe_a4b                   → xhmodel_merak.xh_llm.models.gemma4_moe
```

#### 变体判定规则

| 判定字段 | 31B | E4B | 26B-A4B |
|---|---:|---:|---:|
| `text_config.enable_moe_block` | false | false | true |
| `text_config.num_experts` | None | None | 128 |
| `audio_config` | None | dict | None |
| `text_config.hidden_size_per_layer_input` | 0 | 256 | 0 |
| `text_config.num_kv_shared_layers` | 0 | 18 | 0 |

建议优先级：

1. `enable_moe_block` 或 `num_experts` 非空 → `Gemma4ForConditionalGeneration_with_mask` / MoE。
2. `audio_config` 非空，或 `hidden_size_per_layer_input > 0`，或 `num_kv_shared_layers > 0` → `Gemma4EForConditionalGeneration`（内部 key 名，可自定义）。
3. 否则 → `Gemma4ForConditionalGeneration_dense` 或保留 legacy `Gemma4ForConditionalGeneration` 指向 31B。

### 4.3 注册策略建议

#### 短期止血

- 让 `Gemma4ForConditionalGeneration` 只指向当前“历史正确”的 31B legacy dense 实现。
- Gemma4e 改用独立内部 key，例如：
  - `Gemma4EForConditionalGeneration`
  - 或 `Gemma4ForConditionalGeneration_e4b`
- MoE 保持 `Gemma4ForConditionalGeneration_with_mask`。
- visual/audio 辅助模型也不要共用同一个 `Gemma4ForConditionalGeneration_visual`，至少区分：
  - `Gemma4ForConditionalGeneration_visual`
  - `Gemma4EForConditionalGeneration_visual`
  - `Gemma4MoeForConditionalGeneration_visual`

优点：改动少、能立即消除 31B 误入 gemma4e。

#### 中期统一

新增 family resolver，例如：

```python
def resolve_gemma4_variant(hf_model_dir: str) -> str:
    cfg = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    text = cfg.text_config
    if getattr(text, "enable_moe_block", False) or getattr(text, "num_experts", None):
        return "Gemma4ForConditionalGeneration_with_mask"
    if getattr(cfg, "audio_config", None) is not None:
        return "Gemma4EForConditionalGeneration"
    if getattr(text, "hidden_size_per_layer_input", 0):
        return "Gemma4EForConditionalGeneration"
    if getattr(text, "num_kv_shared_layers", 0):
        return "Gemma4EForConditionalGeneration"
    return "Gemma4ForConditionalGeneration"
```

然后 `AutoLLMConfig` 在遇到 `Gemma4ForConditionalGeneration` 时可二次 refine 为内部 key。

#### 长期治理

抽公共模块：

```text
models/gemma4_common/
  config_resolver.py
  tokenizer_processor.py
  rope.py
  masks.py
  kv_cache.py
  visual_common.py
models/gemma4_dense31/
models/gemma4e/
models/gemma4_moe/
```

不要强行把三者所有 forward 合到一个巨型类；统一的是 family contract，不是把 dense/audio/PLE/MoE 写成一坨 if-else。

---

## 5. 迁移执行顺序

### Phase 0：只读验证与断言

- 增加一个轻量测试：给 31B/E4B/MoE 三个 hf dir，断言 resolver 输出目标 key。
- 增加 registry duplicate 检查：同一个 master key 被多个模块注册时失败，而不是悄悄覆盖。

### Phase 1：注册 key 止血

- `Gemma4ForConditionalGeneration` 保留给 31B dense legacy。
- `gemma4e` 改独立 key。
- configs 中 E4B 显式使用 gemma4e key，或通过 resolver 自动转。
- MoE 继续用 with-mask key。

### Phase 2：公共能力下沉

优先抽：

1. config resolver；
2. token id / processor 公共处理；
3. full/sliding attention mask；
4. rope cache；
5. safetensors index/header 读取工具。

### Phase 3：导出验证矩阵

| 模型 | 目标验证 | 关键断言 |
|---|---|---|
| 31B dense | `AutoLLMConfig` 返回 `models.gemma4` config | 无 audio、无 MoE、`hidden_size_per_layer_input=0` |
| E4B | 返回 `models.gemma4e` config | 有 audio、PLE、KV shared，单文件 safetensors |
| 26B-A4B | 返回 `models.gemma4_moe` config | 有 experts/router，with-mask HMONNX |

---

## 6. 当前风险清单

| 风险 | 等级 | 说明 | 建议 |
|---|---|---|---|
| master key 重复注册 | 高 | 已导致 31B 解析到 gemma4e | 立即治理 key / resolver |
| visual key 重复注册 | 高 | `Gemma4ForConditionalGeneration_visual` 在 gemma4、gemma4e、gemma4_moe 中都出现 | 辅助模型也要 variant key |
| E4B 单 safetensors 无 index | 中 | loader 若默认 index 会失败 | 支持 header fallback |
| E4B PLE | 高 | `embed_tokens_per_layer` 与 per-layer projection 是额外输入链路 | 不可按普通 dense 31B 简化 |
| E4B KV shared | 高 | 18 个 shared KV 层不能按每层 cache 分配 | 必须按 `is_kv_shared_layer` 跳过 |
| MoE experts/router | 高 | 26B-A4B forward 与 dense 完全不同 | MoE 分支必须独立 |
| 类名相同 | 中 | 都叫 `XHGemma4Model`，日志不看 `__module__` 容易误判 | 日志打印 module + class |

---

## 7. 可直接下发的 P8 Task Prompt

### Task A：Registry/Resolver 止血

#### WHY
31B 被解析到 gemma4e，根因是 `Gemma4ForConditionalGeneration` 重复注册并由扫描顺序覆盖。继续保留会导致导出链路不可预测。

#### WHAT
- [ ] 增加 Gemma4 family resolver，按 checkpoint `text_config` / `audio_config` / MoE 字段转成内部 key。
- [ ] 消除 `Gemma4ForConditionalGeneration` master 重复注册。
- [ ] 为 gemma4e 和 gemma4_moe visual/audio 辅助模型使用变体 key。
- [ ] 增加 duplicate master key 检查或测试。

#### WHERE
- `xhmodel_merak/xh_llm/scan_model_types.py`
- `xhmodel_merak/xh_llm/builder.py` 或新建 `models/gemma4_common/resolver.py`
- `xhmodel_merak/xh_llm/models/gemma4*/...`
- 对应 `configs_merak/.../gemma4*` 测试配置

#### DONE
- 轻量脚本断言：31B → `models.gemma4`，E4B → `models.gemma4e`，26B-A4B → `models.gemma4_moe`。
- 不跑重型导出也能通过 registry 测试。

#### DON'T
- 不要把 MoE forward 合进 dense forward。
- 不要靠目录名字符串判断变体，优先用 config 结构字段。

### Task B：三模型导出验证矩阵

#### WHY
大一统后必须证明不是“能 import”，而是每个模型的关键结构路径都走对。

#### WHAT
- [ ] 为 31B/E4B/MoE 分别补轻量构造测试。
- [ ] 检查 config class、model class、HMONNX class、visual/audio/MoE 子模块是否匹配。
- [ ] 对 safetensors index/header 做只读兼容测试。

#### WHERE
- `tests/` 中新增 Gemma4 family 测试。
- 只读访问 `/data01/datasets/gemma-4-*`。

#### DONE
- pytest 轻量测试通过。
- 每个模型打印 module path，而不是只打印 class name。

#### DON'T
- 不要加载完整 31B/26B 权重。
- 不要占用 GPU。

---

## 8. 结论

1. **31B 变成 gemma4e 是注册映射问题。** 31B config 仍然是 `Gemma4ForConditionalGeneration` / `model_type=gemma4`，但这个 key 在仓库里被 `gemma4e` 最终接管。
2. **Gemma4 和 Gemma4e 不能仅凭 `architectures` 区分。** 三个 checkpoint 都是 `Gemma4ForConditionalGeneration`，必须读 `text_config` / `audio_config` / MoE 字段。
3. **Gemma4e 代码方向有合理性。** 它覆盖了 E4B 的 audio、PLE、KV shared 特征；问题是注册策略让它吞掉 31B。
4. **MoE 应保持独立分支。** 26B-A4B 有 experts/router/top-k，与 dense 的 text layer forward 差异大，只能共享外围公共工具。
5. **推荐统一 family contract，而不是一个巨型类。** 最稳妥是 resolver 分发 + common utilities + variant modules。

---

## 9. 证据命令摘要

```bash
# 代码图优先检查：当前 graph 为空，因此回退 shell 搜索
# Graph statistics: Files=0, Total nodes=0, Last updated=never

# 注册扫描
conda activate gemma4
PYTHONPATH=$PWD python - <<'PY'
from xhmodel_merak.xh_llm.scan_model_types import parse_register_llm_models
from pathlib import Path
models_dir=Path('xhmodel_merak/xh_llm/models')
for py_file in sorted(models_dir.rglob('*.py')):
    for r in parse_register_llm_models(py_file.resolve()):
        if 'Gemma4' in r['model_type']:
            print(r)
PY

# AutoLLMConfig 轻量复现
conda activate gemma4
PYTHONPATH=$PWD python - <<'PY'
from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig
from xhmodel_merak.xh_llm.configuration_auto import MODEL_TYPE_MAPPING_MODULES
cfg=Config.fromfile('configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_w4a8_autoround_2k.py')
print(cfg.model.model_type, cfg.model.hf_model)
print(MODEL_TYPE_MAPPING_MODULES.get('Gemma4ForConditionalGeneration'))
mcfg=AutoLLMConfig.from_pretrained(cfg.model)
print(type(mcfg).__module__, type(mcfg).__name__)
PY

# checkpoint config 与 safetensors index/header 只读检查
conda activate gemma4
python - <<'PY'
# 读取 config.json/tokenizer_config.json/processor_config.json/generation_config.json
# 对 index.json 只读 weight_map；E4B 用 safetensors safe_open 只读 keys/header
PY
```
