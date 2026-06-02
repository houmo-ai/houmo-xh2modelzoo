# Qwen3.5 迁移方案 v2：按 Merak 原生设计重做

## 1. 设计原则

**不搬代码，按 merak 的框架契约重新实现。**

merak 的核心设计：
- `XHBaseModel` 状态机 = Converter（`to_wrap → to_fronted → to_quanted → export_hmonnx`）
- `XHSubModel` = 子模型（Vision、Draft 等独立导出单元）
- `BaseLLMHMONNXModel` = 推理引擎（prefill/decode 分发）
- `@register_llm_model` = 注册即可用

**zoo 的 `Qwen3_5ConverterXH2a` 是冗余的** — merak 的 `XHQwen3_5Model.export_hmonnx()` 已经是 Converter。
**zoo 的 `Qwen3_5ONNXModel` 是冗余的** — merak 的 `XHQwen3_5_HMONNXModel` 已经是推理引擎。

## 2. 当前 Merak Qwen3.5 能力现状

### 已具备 ✅

| 能力 | 实现 |
|------|------|
| LLM 导出（prefill + decode） | `XHQwen3_5Model.export_hmonnx()` |
| Vision 导出 | `XHQwen3_5VisionModel.export_hmonnx()` |
| LLM 推理 | `XHQwen3_5_HMONNXModel.forward()` |
| VL 推理 | `XHQwen3_5_HMONNXModel` + `VisualHMONNXModel` |
| Linear Attention (chunk/recurrent) | `_llm_model_impl.py` + `_delta_rule.py` + `_gdr_ops.py` |
| KV + Linear Cache 管理 | `KVCacheWithLinearMixin` |
| ModelSwitcher (prefill/decode 分离) | `qwen3_5_llm_model.py` |
| HF-Compatible generate() | `_Qwen3_5HFCompatible` |

### 缺失 🔴

| 能力 | 说明 |
|------|------|
| MTP Draft 模型导出 | 需要独立的 MTP 子模型，走 `XHSubModel` 模式 |
| DFlash Draft 模型导出 | 需要独立的 DFlash 子模型，走 `XHSubModel` 模式 |
| SpecDecode 推理 | 需要扩展 `XHQwen3_5_HMONNXModel` 支持 draft→verify→accept 循环 |
| `verify_output_intermediates` | 主模型 forward 需输出中间 hidden states 供 spec decode 验证 |
| `split_conv_cache` | 导出时分离 conv cache 为独立输入（spec decode rollback 需要） |

## 3. 迁移设计

### 3.1 MTP Draft 模型 — 新建 `XHQwen3_5MTPDraftModel(XHSubModel)`

**设计思路**：MTP 是独立的子模型，和 Vision 一样走 `XHSubModel` 模式。

```
文件：_mtp_model_impl.py     — DynamicModule 层定义 + register_wrap_cls()
文件：qwen3_5_mtp_model.py   — XHQwen3_5MTPDraftModel(XHSubModel)
```

**契约实现**：
- `HF_MODEL_CLS` = None（MTP 没有独立 HF 模型，从主模型权重加载）
- `_to_wrap()` — 加载 MTP 权重，构建 wrap model
- `get_dummy_inputs()` — `[next_token_embedding, post_norm_hidden, past_seq_length, ...]`
- `get_export_cfg()` — 定义 MTP 的 input/output names
- `export_hmonnx()` — 导出单个 MTP ONNX

**与主模型的关系**：
- 由 `XHQwen3_5Model` 组合持有（类似 `self.visual`）
- 主模型 `export_hmonnx()` 中调用 `self.mtp_draft.export_hmonnx()`

### 3.2 DFlash Draft 模型 — 新建 `XHQwen3_5DFlashDraftModel(XHSubModel)`

**设计思路**：DFlash 有两个模式（context + decode），导出为两个 ONNX。

```
文件：_dflash_model_impl.py    — DynamicModule 层定义 + register_wrap_cls()
文件：qwen3_5_dflash_model.py  — XHQwen3_5DFlashDraftModel(XHSubModel)
```

**契约实现**：
- `_to_wrap()` — 加载 DFlash 权重
- `export_hmonnx()` — 导出 context.onnx + decode.onnx（两个模式各一个）
- `get_export_cfg()` — 分 context/decode 两套 input/output names

**与主模型的关系**：
- 由 `XHQwen3_5Model` 组合持有（`self.dflash_draft`）
- 主模型 `export_hmonnx()` 中调用 `self.dflash_draft.export_hmonnx()`

### 3.3 主模型增强 — 修改 `_llm_model_impl.py` + `qwen3_5_llm_model.py`

**需要增加的能力**：

| 功能 | 改动位置 | 说明 |
|------|---------|------|
| `verify_output_intermediates` | `_Qwen3_5TextModel.forward()` | 可选输出每层 post_norm hidden state |
| `split_conv_cache` | `_Qwen3_5GatedDeltaNet` | 导出时 conv cache 拆为独立 tensor（rollback 需要） |
| Draft 模型组合 | `XHQwen3_5Model` | 新增 `self.mtp_draft` / `self.dflash_draft` 可选属性 |
| 扩展 `export_hmonnx()` | `XHQwen3_5Model` | 主模型导出后，按配置导出 draft 模型 |

### 3.4 SpecDecode 推理 — 扩展 `XHQwen3_5_HMONNXModel`

**设计思路**：不新建类，在现有 `XHQwen3_5_HMONNXModel` 中增加 spec decode 能力。

```
文件：qwen3_5_hmonnx_inference.py  — 扩展现有类
```

**新增方法**：
- `load_draft_model(meta_info)` — 从 meta.json 加载 draft ONNX session
- `generate_with_spec_decode(input_ids, ...)` — spec decode 生成循环
- `_run_draft_mtp()` / `_run_draft_dflash()` — draft token 生成
- `_verify_and_accept()` — 验证 + 接受/拒绝
- `_snapshot_linear_cache()` / `_restore_linear_cache()` — DeltaNet 状态快照/回滚

**不新建子类的原因**：merak 的 `HMONNXINFERENCE_CLS` 是类级别绑定的，一个模型只有一个推理类。spec decode 是同一个推理引擎的增强模式，不是另一个引擎。

### 3.5 Config 扩展 — 修改 `xh_qwen3_5_config.py`

```python
class XHQwen3_5ModelConfig(VisionLLMModelConfig):
    # 现有字段...

    # 新增 spec decode 配置
    spec_decode_mode: Optional[str] = None  # "mtp" | "dflash" | None
    mtp_model_path: Optional[str] = None
    dflash_model_path: Optional[str] = None
    num_draft_tokens: int = 4
    mtp_hot_vocab_size: Optional[int] = None
```

## 4. 文件变更清单

### 新建文件（4 个）

| 文件 | 用途 | 行数估算 |
|------|------|---------|
| `_mtp_model_impl.py` | MTP DynamicModule 层 | ~300 行 |
| `qwen3_5_mtp_model.py` | MTP SubModel 封装 | ~150 行 |
| `_dflash_model_impl.py` | DFlash DynamicModule 层 | ~250 行 |
| `qwen3_5_dflash_model.py` | DFlash SubModel 封装 | ~200 行 |

### 修改文件（5 个）

| 文件 | 改动 | 影响范围 |
|------|------|---------|
| `_llm_model_impl.py` | +`verify_output_intermediates` +`split_conv_cache` | ~80 行增量 |
| `qwen3_5_llm_model.py` | +draft 模型组合 +扩展 export_hmonnx | ~60 行增量 |
| `qwen3_5_hmonnx_inference.py` | +spec decode 推理循环 | ~200 行增量 |
| `xh_qwen3_5_config.py` | +spec decode 配置字段 | ~15 行增量 |
| `__init__.py` | +导出新符号 | ~5 行 |

### 不动的文件（12 个）

`_delta_rule.py`, `_gdr_ops.py`, `_compat.py`, `configuration_qwen3_5.py`, `modeling_qwen3_5.py`, `modeling_qwen3_5_patch.py`, `data_preprocess.py`, `qwen3_5_processor.py`, `qwen3_5_vision_model.py`, `_vision_model_impl.py`, `processing_qwen3_5.py`, `image_processing_qwen3_5.py`

### 同步差异（2 个，低优先级）

| 文件 | 差异 | 处理 |
|------|------|------|
| `_vision_model_impl.py` | 135 行 diff | 评估后按需合入 |
| `processing_qwen3_5.py` | 61 行 diff | 评估后按需合入 |

## 5. Examples 脚本策略

**不再"重写"老版脚本，而是新建 merak 原生 examples。**

```
examples/merak/qwen3_5/
  export.py              — 调用 XHQwen3_5Model 状态机一把导出
  inference.py           — 调用 XHQwen3_5_HMONNXModel 推理
  spec_decode_bench.py   — spec decode benchmark
  vl_demo.py             — VL 推理 demo
```

老版 `examples/llm/qwen3_5/` 标记 deprecated，不再维护。

## 6. 执行计划

### Sprint 1：Draft 模型实现（2 天）

| # | 任务 | 产出 |
|---|------|------|
| 1.1 | 实现 `_mtp_model_impl.py` | MTP DynamicModule 层 |
| 1.2 | 实现 `qwen3_5_mtp_model.py` | MTP SubModel，可独立 export |
| 1.3 | 实现 `_dflash_model_impl.py` | DFlash DynamicModule 层 |
| 1.4 | 实现 `qwen3_5_dflash_model.py` | DFlash SubModel，可独立 export |
| 1.5 | 验证：draft 模型可走通 `to_wrap → to_fronted → export_hmonnx` | ONNX 文件产出 |

### Sprint 2：主模型增强 + 集成导出（1.5 天）

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | `_llm_model_impl.py` 增加 `verify_output_intermediates` | 主模型可输出中间 hidden |
| 2.2 | `_llm_model_impl.py` 增加 `split_conv_cache` | conv cache 可独立导出 |
| 2.3 | `qwen3_5_llm_model.py` 组合 draft 模型 | `export_hmonnx()` 一把导出主模型 + draft |
| 2.4 | `xh_qwen3_5_config.py` 扩展配置 | spec decode 配置可用 |
| 2.5 | 验证：`XHQwen3_5Model.export_hmonnx()` 一次调用产出全部 artifacts | meta.json 包含 draft 信息 |

### Sprint 3：SpecDecode 推理（1.5 天）

| # | 任务 | 产出 |
|---|------|------|
| 3.1 | `qwen3_5_hmonnx_inference.py` 增加 draft 加载 | 可加载 draft ONNX |
| 3.2 | 实现 `_snapshot_linear_cache` / `_restore_linear_cache` | DeltaNet 状态可回滚 |
| 3.3 | 实现 `generate_with_spec_decode()` | spec decode 生成循环 |
| 3.4 | 验证：spec decode 推理端到端跑通 | 输出正确，acceptance rate 合理 |

### Sprint 4：Examples + 收尾（1 天）

| # | 任务 | 产出 |
|---|------|------|
| 4.1 | 新建 `examples/merak/qwen3_5/export.py` | 一键导出脚本 |
| 4.2 | 新建 `examples/merak/qwen3_5/inference.py` | 推理 demo |
| 4.3 | 新建 `examples/merak/qwen3_5/spec_decode_bench.py` | benchmark |
| 4.4 | 更新 `__init__.py` 导出符号 | 公开 API 完整 |
| 4.5 | 老版 examples 标记 deprecated | 清理 |

## 7. 风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| MTP/DFlash 权重加载路径与 HF 不一致 | 中 | 参考 zoo 的 `from_pretrained` 逻辑，适配到 `_to_wrap()` |
| DeltaNet 状态 rollback 精度问题 | 高 | 用 zoo 的 golden data 做 bit-level 对比验证 |
| `XHSubModel` 没有 `ModelSwitcher` | 低 | Draft 模型不需要 prefill/decode 分离（只有 decode） |
| DFlash 双模式导出（context + decode）| 中 | 参考 Vision 的独立导出模式，两次 trace 两次 export |

## 8. 验收标准

1. `XHQwen3_5Model(config).export_hmonnx(output_dir)` 一次调用产出：
   - `prefill/*.onnx` + `decode/*.onnx`（主模型）
   - `visual/*.onnx`（Vision，如配置）
   - `draft_mtp/*.onnx`（MTP，如配置）
   - `draft_dflash_context/*.onnx` + `draft_dflash_decode/*.onnx`（DFlash，如配置）
   - `meta.json`（包含所有模型路径和 spec decode 配置）

2. `XHQwen3_5_HMONNXModel.from_hmonnx_meta(meta)` 加载后：
   - `.forward()` — 标准推理
   - `.generate_with_spec_decode()` — spec decode 推理
   - 两种模式输出一致（spec decode 是加速，不改变结果分布）

3. 零 zoo 依赖 — merak 的 qwen3_5 模块不 import 任何 `xh_model_zoo` 代码

## 9. 总工时

| Sprint | 工时 | 产出 |
|--------|------|------|
| Sprint 1 | 2 天 | Draft 模型可独立导出 |
| Sprint 2 | 1.5 天 | 一把导出全部 artifacts |
| Sprint 3 | 1.5 天 | SpecDecode 推理跑通 |
| Sprint 4 | 1 天 | Examples + 收尾 |
| **总计** | **6 天** | |
