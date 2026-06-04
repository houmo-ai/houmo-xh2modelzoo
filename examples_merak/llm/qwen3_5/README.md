# Qwen3.5 / Qwen3.5-MoE Merak export

`qwen3_5_xh_export_hmonnx.py` 是 dense 与 MoE 的统一 Merak HMONNX 导出入口。导出形态只由 `configs_merak/...` 配置文件决定；脚本不再接受模型结构、量化、MTP/DFlash 等模型参数。

## 环境

```bash
conda activate xhquant_55
# 单任务使用一张空闲 GPU，例如：CUDA_VISIBLE_DEVICES=0 <command>
```

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

## 导出 HMONNX

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py \
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
- PPL 烟测使用 `examples_merak/llm/qwen3_5/qwen3_5_xh_ppl_eval.py`。

## Golden 规范

导出产物需保留 release-style layout：`prefill/`、`decode/`、可选 `visual/`、可选 `mtp_draft_*` / `dflash_draft_*`，并在 `golden_meta_info.json` 中记录相对路径，避免旧式 `decoder/`、`mtp/`、`dflash/` 顶层目录。
