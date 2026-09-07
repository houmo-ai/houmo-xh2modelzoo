# Qwen3.5 / Qwen3.5-MoE Merak export

`debug_scripts/qwen3_5_xh_export_hmonnx.py` 是 dense 与 MoE 的调试/兼容导出入口。导出形态只由 `configs_merak/...` 配置文件决定；脚本不再接受模型结构、量化、MTP/DFlash 等模型参数。

## 环境

```bash
conda activate xhquant_55
# 单任务使用一张空闲 GPU，例如：CUDA_VISIBLE_DEVICES=0 <command>
```

## 当前 workflow 默认配置与精度

新的量化/导出 workflow 统一使用
`examples_merak/llm/qwen3_5/README_workflow.md` 中的 YAML 配置。默认选择
未带 `_gptq` 后缀的 `full.yaml`，即 AutoRound weight-only -> GPTQModel HF
artifact -> HMONNX export；只有显式需要直接 GPTQModel GPTQ 时才使用
`*_gptq.yaml`。4B 当前仍是旧式 Merak Python config，精度表中一起列出，
方便和 9B / 27B / 35B-A3B 对齐查看。

| 模型 | 默认配置 | CEval float | CEval AutoRound weight-only | CEval HMONNX |
|---|---|---:|---:|---:|
| Qwen3.5-4B | `qwen3_5/4b/qwen3_5_4b_instruct_xh2a_2k.py` | 82.39% | 79.13% | 79.94% |
| Qwen3.5-9B | `qwen3_5/9b/qwen3_5_9b_full.yaml` | 84.92% | 84.40% | 83.58% |
| Qwen3.6-27B | `qwen3_5/27b/qwen3_6_27b_full.yaml` | 90.64% | 90.27% | 89.90% |
| Qwen3.6-35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml` | 89.60% | 89.90% | 88.71% |

CEval 口径：EvalScope / 官方 prompt，5-shot，1346 题，生成长度 4096。
数据来自飞书评测表
`https://houmo.feishu.cn/docx/RBLhdZ3EHoQBiGx4aT9c671Gn7f`
（更新时间 2026-06-22 11:00:36）。

## 配置入口

- 9B 浮点：`configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py`
- 9B 量化 HF/GPTQModel：`configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_hf_gptq_xh2a_2k.py`
- 9B MTP：`configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_spec_mtp_xh2a_2k.py`
- 9B DFlash：`configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_spec_dflash_xh2a_2k.py`
- Qwen3.6 35B-A3B 浮点：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py`
- Qwen3.6 35B-A3B 量化 HF/GPTQModel：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_hf_autoround_xh2a_2k.py`
- Qwen3.6 35B-A3B MTP：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_spec_mtp_xh2a_2k.py`
- Qwen3.6 35B-A3B DFlash：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_spec_dflash_xh2a_2k.py`

量化 HF/GPTQModel 目录的统一约定：把 `model.hf_model` 指向量化后的 HF repo，`model.quant_weight` 保持为空。`quant_weight` 只保留给“浮点 HF 结构 + 独立 torch checkpoint 权重文件/目录”的旧式检查点恢复。

模型配置支持 `enable_layer_tag`。开启后，导出 HMONNX 时会在每个 layer 结束处插入 Tag，方便 PP 并行分配 GPU 以及按 layer 切分 HMONNX。

也可以通过环境变量临时开启，无需修改模型配置：

```bash
export LAYER_TAG_ENABLE=1
```

环境变量支持的真值为 `1`、`true`、`yes`、`on`；开启后会在 wrap 配置中注入
`enable_layer_tag=True`。

## 导出 HMONNX

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py \
  --force

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_spec_mtp_xh2a_2k.py \
  --force
```

导出 **Qwen3.5-122B-A10B** 时必须先开启超大模型分层导出：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
# 可选真值：1 / true / yes / on

# 可选：并行导出 placeholder 子图（默认 1）。多卡时可按可见 GPU 数设置。
export XH2MODELZOO_EXPORT_WORKERS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_moe_122b_a10b_instruct_xh2a_2k.py \
  --force
```

更换模型时只替换 `--config`。输出默认在 `work_dirs/<config_stem>/`；如需指定输出目录，只能使用运行参数 `--work-dir`，不要通过命令行覆盖模型配置。

## Demo / Golden / PPL

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config work_dirs/<config_stem>/<export_dir>/golden_meta_info.json \
  --prompt "Describe this image." \
  --image-path data/images/demo_qwen3_vl.jpeg
```

- `--golden` 会在 demo 推理时保存 golden 输出。
- dense 与 MoE 都使用同一个 demo 入口；如果 meta 支持视觉分支且图片存在，则走图文输入，否则回退到纯文本输入。
- PPL 烟测使用 `examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_ppl_eval.py`。
- workflow 的 `--dump-golden` 会先为 m96 / m196 / m384 / m704 / m1536 五档
  visual 图各执行一次并写入 `visual_m*/step_0`，再用真实图片完成 visual →
  prefill → decode 链路；真实图片命中的 visual 档另写 `step_1`。visual-only 导出则在根目录写 `visual_meta_info.json`，
  golden 位于 `visual_m*/step_0`。
- workflow 的 `--quick-test` 可直接识别 full-model 与 visual-only 产物；对
  visual-only 会加载 `visual_meta_info.json` 并逐档校验输出 shape，不再误走文本生成。

## Spec decode / GDR fuse 验证矩阵

Merak 导出验证固定以下基础开关：

- `normalize_force_fp32=False`（即旧口径里的 `force_norm_fp32=False`）
- `use_manual_depthwise_conv1d=False`
- 视觉验证必须使用真实图片输入，不能跳过 `visual/` 分支。

当前需要覆盖的参数组合：

| 组合名 | `fuse_gdr_ops` | `fuse_gdr_block_recurrent_ops` | `split_conv_cache` | 说明 |
|---|---:|---:|---:|---|
| `fuse0_split1` | `False` | `False` | `True` | 显式关闭两项 GDR fuse 的对照路径 |
| `fuse0_split0` | `False` | `False` | `False` | merged conv cache 导出路径 |
| `gdr_block_recurrent_split1` | `False` | `True` | `True` | 仅启用不改变 I/O 契约的 GDRBlockTriInverse + GDRRecurrentScan |
| `fuse1_split1` | `True` | `True` | `True` | 所有 checked-in YAML 的默认路径 |

> `fuse_gdr_ops=True` 现在仅表示启用 GDRChunkScan；`split1` 表示 `split_conv_cache=True`。

`num_draft_tokens` 可以在 Merak config 的 `model` 顶层配置，例如：

```python
model = dict(
    spec_decode_mode="mtp",      # 或 "dflash"
    num_draft_tokens=4,          # MTP 当前默认配置
    mtp_config=dict(...),
)

model = dict(
    spec_decode_mode="dflash",
    num_draft_tokens=9,          # DFlash 当前默认配置
    dflash_config=dict(...),
)
```

只需要写在 `model` 顶层，不需要在 `mtp_config` / `dflash_config` 里重复写；导出主流程读取的是 `self.config.num_draft_tokens`。修改后需要重新导出，因为 target decode / verify ONNX 的输入长度会随之变化：

- MTP：`spec_decode.block_size = num_draft_tokens`
- DFlash：verify 长度和 `spec_decode.block_size = num_draft_tokens + 1`
- 导出的 `golden_meta_info.json` 会记录 `spec_decode.num_draft_tokens`，HMONNX runtime 再从 meta 中读取该值。

最近一次完整验证覆盖 3 类模型 × 3 种模式 × 3 组参数，共 27 个 canonical case：

| 模型 / 模式 | `fuse0_split1` | `fuse0_split0` | `fuse1_split1` |
|---|---:|---:|---:|
| Qwen3.5 9B Base | passed | passed | passed |
| Qwen3.5 9B MTP | passed | passed | passed |
| Qwen3.5 9B DFlash | passed | passed | passed |
| Qwen3.6 27B Base | passed | passed | passed |
| Qwen3.6 27B MTP | passed | passed | passed |
| Qwen3.6 27B DFlash | passed | passed | passed |
| Qwen3.6 35B-A3B MoE Base | passed | passed | passed |
| Qwen3.6 35B-A3B MoE MTP | passed | passed | passed |
| Qwen3.6 35B-A3B MoE DFlash | passed | passed | passed |

验证内容：

- Base：export、release layout、visual meta、真实图片 generate、PPL。
- MTP / DFlash：export、release layout、visual meta、真实图片 generate。
- DFlash 还需确认 `dflash_draft_context/`、`dflash_draft_context_decode/`、`dflash_draft_decode/` 三类 draft HMONNX 均已导出并写入 `golden_meta_info.json` 的 `spec_decode` section。

注意事项：

- `--golden` demo 为了保存 golden artifact，会把生成长度压到很短；它只能证明 HMONNX 推理链路可执行，不适合判断回答语义质量。
- MTP / DFlash 的接收率不由 `qwen3_5_xh_hmonnx_generate.py` 默认输出；需要使用 `examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_spec_decode_test.py` 读取导出后的 `golden_meta_info.json` 单独评估。
- 代表性 9B `fuse1_split1` 文本 spec decode smoke：
  - MTP：`accepted=10 / draft_tokens=36`，接收率约 `27.78%`，`avg_accepted_per_round=1.11`。
  - DFlash：`accepted=5 / draft_tokens=117`，接收率约 `4.27%`，`avg_accepted_per_round=0.38`。
  - 二者均能生成正常中文句子。

## Golden 规范

完整导出产物保留 release-style layout：`prefill/`、`decode/`、可选
`visual_m*`、可选 `mtp_draft_*` / `dflash_draft_*`，这些目录均位于包根目录，并在
`golden_meta_info.json` 和根目录的 `visual_gears.json` 中记录相对路径。visual-only 产物以
`visual_meta_info.json` 为入口，五档 HMONNX 和 golden 分别放在
`visual_m96`、`visual_m196`、`visual_m384`、`visual_m704`、`visual_m1536` 下。每次 golden 生成必须覆盖
全部五档；真实图片命中的档位是端到端补充验证，不能代替逐档 golden。
