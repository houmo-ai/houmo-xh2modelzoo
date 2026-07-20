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
- 122B-A10B 配置：`configs_merak/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_moe_122b_a10b_instruct_xh2a_2k.py`
- 视觉分支配置：`configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_visual_xh2a_2k.py`
- 默认模型目录：`/data01/nfs_shared/Qwen3.5-35B-A3B`

### Layer Tag

如果需要在每个 layer 结束处插入 Tag，便于 PP 并行分配 GPU 或按 layer 切分 HMONNX，
可以在导出前设置：

```bash
export LAYER_TAG_ENABLE=1
```

环境变量支持的真值为 `1`、`true`、`yes`、`on`；开启后会在 wrap 配置中注入
`enable_layer_tag=True`，无需修改模型配置。

## 导出 HMONNX

使用配置文件导出：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py --config configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py
```

导出 **122B-A10B** 时必须启用超大模型分层导出（placeholder 路径），否则峰值内存会过高：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
# 可选真值：1 / true / yes / on

# 可选：并行导出 placeholder 子图（默认 1）。多卡时可按可见 GPU 数设置。
export XH2MODELZOO_EXPORT_WORKERS=4

python examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_moe_122b_a10b_instruct_xh2a_2k.py
```

workflow 方式同样需要该环境变量：

```bash
export HUGE_MODEL_EXPORT_ENABLED=1
export XH2MODELZOO_EXPORT_WORKERS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-122B-A10B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml \
  --quant-output-dir work_dirs/qwen3_5_122B_quant \
  --export-output-dir work_dirs/qwen3_5_122B_export \
  --overwrite
```

### Placeholder 输出契约

Qwen3.5-MoE 的超大模型分层导出会将以下 Hugging Face 模块作为独立 placeholder 子图展开：

- `Qwen3_5MoeAttention`
- `Qwen3_5MoeGatedDeltaNet`
- `Qwen3_5MoeSparseMoeBlock`

这些模块的 `forward` 返回值中不允许包含 `None`，包括 tuple、list 或 mapping 中的嵌套成员。
主图中的 `PlaceHolderModule` 与独立 HMONNX 子图通过固定数量、固定顺序的 tensor 端口连接，
而 `None` 无法表示为 HMONNX tensor 输出。适配或升级 Transformers 实现时，必须保证上述模块
在 prefill 和 decode 路径下均只返回 tensor，或者返回仅由 tensor 组成的扁平容器。

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
