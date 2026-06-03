# Qwen3.5-MoE

Qwen3.5-MoE 在 `xhmodel_merak` 中的迁移示例，覆盖配置、HMONNX 导出、导出后生成验证与 PPL 评估。

## 环境

所有命令默认在仓库根目录执行：

```bash
source env.sh
```

## 配置

- 基础配置：`configs_merak/xh2a/llm_models/qwen3_5_moe/_qwen3_5_moe_xh2a_2k.py`
- 35B-A3B 配置：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py`
- 视觉分支配置：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_visual_xh2a_2k.py`
- 默认模型目录：`/data01/nfs_shared/Qwen3.5-35B-A3B`

## 导出 HMONNX

使用配置文件导出：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py
```

或者直接指定模型目录：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py --model /data01/nfs_shared/Qwen3.5-35B-A3B --context-length 2048 --prefill-chunk-length 256
```

如果要验证 GPTQModel 权重流程，可额外传入量化权重目录：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py \
  --model /data01/nfs_shared/Qwen3.5-35B-A3B \
  --context-length 2048 \
  --prefill-chunk-length 256 \
  --quant-type w4a8h1_sefp \
  --quant-weight /data01/home/huxing/gptqmodel/work_dirs/Qwen35_35B_A3B_attn4_e4_se4_0324
```

如果你已经有旧版 `xhquant_llm` 导出的 GPTQ HMONNX 目录（例如 `/data01/home/huxing/xhquant_llm/work_dirs/qwen35moe_w4a8_gptqmodel_0324_v2`），可以先生成一个 Merak 兼容的 meta 文件，再直接复用旧导出做生成/PPL 验证：

```bash
python examples_merak/llm/qwen3_5_moe/qwen3_5_moe_xh_adapt_legacy_meta.py \
  --legacy-export-dir /data01/home/huxing/xhquant_llm/work_dirs/qwen35moe_w4a8_gptqmodel_0324_v2 \
  --base-golden-meta work_dirs/qwen3_5_moe_35b_a3b_instruct_xh2a_2k/<export_dir>/golden_meta_info.json \
  --quantized-model-dir /data01/home/huxing/gptqmodel/work_dirs/Qwen35_35B_A3B_attn4_e4_se4_0324
```

## HMONNX 生成验证

```bash
python examples_merak/llm/qwen3_5_moe/qwen3_5_moe_xh_hmonnx_generate.py --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json --auto-offload
```

如果导出的 HMONNX 模型实现了视觉处理分支，脚本会优先走图文输入；否则自动回退为纯文本生成。

常用附加参数：

- `--fast`：切换到 fast 模式
- `--golden`：导出 Golden 输出
- `--auto-offload`：开发阶段调试显存卸载
- `--think`：开启 think 模式

## HMONNX PPL 评估

```bash
python examples_merak/llm/qwen3_5_moe/qwen3_5_moe_xh_ppl_eval.py \
  --config work_dirs/<cfg_name>/<export_dir>/golden_meta_info.json \
  --max-samples 4096
```

## 122B模型导出问题分析：

对于122B的模型， 每个Linear的weight是fp16，quant_weight是int8，所以它的参数量应该是360Gb这个量级，wrap只是对nn.Module的**class**进行替换，本质上不会增加内存占用，wrap阶段会对Moe模块做一些内存复制的转移操作，这里可能会导致内存占用增大，但应该只是临时的内存占用。
需要注意量化阶段有哪些地方导致了内存持续增长的？
