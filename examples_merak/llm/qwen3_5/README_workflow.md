# Qwen3.5 / Qwen3.6 Merak Workflow 使用说明

本文说明 Qwen3.5、Qwen3.6 dense、Qwen3.6 MoE 在 Merak workflow 下的推荐用法。统一入口是：

```python
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

workflow = AutoLLMWorkflow.from_config(
    model_dir="/path/to/hf_model",
    config_path="configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml",
)

quant_result = workflow.quant(output_dir="./work_dirs/qwen3_5_quant", device="cuda")
export_result = workflow.export(
    quant_result=quant_result,
    output_dir="./work_dirs/qwen3_5_export",
    device="cuda",
)
```

可直接参考脚本：

```text
examples_merak/llm/qwen3_5/qwen3_5_workflow.py
```

该脚本覆盖当前推荐链路：

```text
quant -> export -> dump_golden -> quick_test_hmonnx
```

## 环境

```bash
conda activate xh2modelzoo
```

Qwen3.5/Qwen3.6 Merak 路径已验证可使用 Transformers 5.13；共享仓库仍保留
4.57 全局约束，推荐用独立环境。版本矩阵和升级边界见
[Merak Transformers 5.13 兼容性结论](../../../docs/merak_transformers_5_13_compatibility_20260727.md)。

导出和验证建议单任务使用一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py ...
```

## 推荐配置

workflow YAML 位于：

```text
configs_merak/workflows/xh2a/llm_models/qwen3_5/
configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/
```

文件命名按模型规模和导出形态组织：

- `full.yaml`：完整 LLM/VLM 导出，默认 AutoRound weight-only 量化。
- `full_gptq.yaml`：完整导出，使用 GPTQModel GPTQ 量化。
- `full_mtp.yaml` / `full_mtp_gptq.yaml`：MTP speculative decoding 导出。
- `full_dflash.yaml` / `full_dflash_gptq.yaml`：DFlash speculative decoding 导出。
- `visual_only_448.yaml` / `visual_only_896.yaml`：只导出 visual tower。
- 带 `_gptq` 后缀的 YAML 使用 `quant.method: gptq`。
- 不带 `_gptq` 后缀的 YAML 使用 `quant.method: autoround`。

常用配置：

| 模型 | 默认 full 配置 | 其他导出形态 |
| --- | --- | --- |
| Qwen3.5-9B | `qwen3_5/9b/qwen3_5_9b_full.yaml` | `full_mtp`、`full_dflash`、`visual_only_448`、`visual_only_896` |
| Qwen3.6-27B | `qwen3_5/27b/qwen3_6_27b_full.yaml` | `full_mtp`、`full_dflash`、`visual_only_448`、`visual_only_896` |
| Qwen3.6-35B-A3B | `qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml` | `full_mtp`、`full_dflash`、`visual_only_448`、`visual_only_896` |

完整 HMONNX 输入输出说明见：

```text
docs/qwen3_5_hmonnx_io_spec.md
```

## 量化

`Qwen35Workflow.quant()` 当前支持三类量化入口：

- `quant: null`：跳过量化，直接使用传入的 `model_dir`。
- `quant.algorithm: gptqmodel` + `quant.method: autoround`：调用 AutoRound adapter，生成 GPTQModel HF 量化目录。
- `quant.algorithm: gptqmodel` + `quant.method: gptq`：调用 GPTQModel GPTQ adapter，生成 GPTQModel HF 量化目录。

默认 AutoRound 配置示例：

```yaml
quant:
  algorithm: gptqmodel
  method: autoround
  output_format: gptqmodel_hf
  artifact_format: gptqmodel_hf
  bits: 4
  group_size: 64
  sym: true
  iters: 200
  seed: 42
  quant_nontext_module: false
  calibration:
    dataset: NeelNanda/pile-10k
    nsamples: 128
    seqlen: 2048
  runtime:
    batch_size: 8
    trust_remote_code: true
    low_gpu_mem_usage: true
```

Qwen3.5/Qwen3.6 workflow 量化要求 `group_size: 64`。

如果已经有量化后的 HF 模型目录，不需要再走量化分支。直接把
`--model-dir` 指向该量化 HF 目录，并设置 `--export-from-quanted-model`：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir /path/to/quanted_hf_model \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_5_export \
  --overwrite

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/SGGM-VL-27B-R3.6-mode1-llm-only-W4G64 \
  --model-name SGGM-VL-27B-R3.6 \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/SGGM-VL-27B-R3.6_export \
  --overwrite

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B-mode1-llm-only \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_5_9b_flashattention_fuse_gdr_export \
  --context-max-length 262144 \
  --enable-flash-attention \
  --enable-fuse-gdr-block-recurrent-ops \
  --overwrite

CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B-mode1-llm-only \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_5_9b_mtp_flashattention_fuse_gdr_export \
  --context-max-length 262144 \
  --enable-flash-attention \
  --enable-fuse-gdr-block-recurrent-ops \
  --overwrite

CUDA_VISIBLE_DEVICES=1 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B-mode1-llm-only \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_5_9b_dflash_flashattention_fuse_gdr_export \
  --context-max-length 262144 \
  --enable-flash-attention \
  --enable-fuse-gdr-block-recurrent-ops \
  --overwrite

CUDA_VISIBLE_DEVICES=1 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B-mode1-llm-only \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_5_9b_fuse_3gdr_export \
  --context-max-length 2048 \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --golden-device-map cuda:0 cuda:1 \
  --overwrite

CUDA_VISIBLE_DEVICES=6,7 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir work_dirs/qwen3_5_122B_quant \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_5_122b_a10b_fuse_3gdr_export \
  --context-max-length 2048 \
  --enable-fuse-gdr-ops \
  --enable-fuse-gdr-block-recurrent-ops \
  --golden-device-map cuda:0 cuda:1 \
  --dump-golden \
  --overwrite

CUDA_VISIBLE_DEVICES=4 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml \
  --export-from-quanted-model \
  --export-output-dir work_dirs/qwen3_6_35b_flashattention_fuse_gdr_export \
  --context-max-length 262144 \
  --enable-flash-attention \
  --enable-fuse-gdr-block-recurrent-ops \
  --dump-golden \
  --overwrite
```

代码里等价写法是：

```python
quant_result = workflow.quant(
    output_dir="./work_dirs/qwen3_5_quant",
    device="cuda",
    config_overrides={"quant": None},
)
```

此时 `QuantResult.skipped` 为 `True`，`export()` 会直接使用 workflow 初始化时传入的 `model_dir`。

## 导出

导出阶段只消费 `quant()` 返回的 `QuantResult`。如果要使用已经量化好的
HF/GPTQModel 目录，不要在 workflow 里新增 `existing_hf` algorithm；使用
`--export-from-quanted-model`，或者在代码中传 `config_overrides={"quant": None}`。

### Workflow CLI

`qwen3_5_workflow.py` 是 Qwen3.5 量化和导出的唯一示例入口。
模型结构和量化参数放在 YAML；FlashAttention/GDR 默认值也放在 YAML，
必要时可用 workflow CLI 做本次运行的显式开关覆盖。

#### Layer Tag

如果需要在每个 layer 结束处插入 Tag，便于 PP 并行分配 GPU 或按 layer 切分 HMONNX，
可以在运行 workflow 前设置：

```bash
export LAYER_TAG_ENABLE=1
```

环境变量支持的真值为 `1`、`true`、`yes`、`on`；开启后会在 wrap 配置中注入
`enable_layer_tag=True`，无需修改模型配置。

真实量化后导出：

```bash
CUDA_VISIBLE_DEVICES=2 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --quant-output-dir work_dirs/qwen3_5_9b_quant \
  --export-output-dir work_dirs/qwen3_5_9b_export \
  --device cuda:0 \
  --overwrite
```

使用已经量化好的 HF/GPTQModel 目录导出时，传入量化模型目录并使用
`--export-from-quanted-model`。该模式会通过 `quant=None` 跳过量化，
不会引入额外的 workflow quant algorithm。

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-9B-mode1-llm-only \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml \
  --quant-output-dir work_dirs/qwen3_5_9b_flash_quant \
  --export-output-dir work_dirs/qwen3_5_9b_flash_export \
  --device cuda:0 \
  --export-from-quanted-model \
  --overwrite
```

FlashAttention 默认由 YAML 控制。例如各 full/gptq YAML 中默认关闭：

```yaml
export:
  model:
    flash_attention:
      enable: false
      q_bits: 8
      s_bits: 8
```

`qwen3_5_workflow.py` 不提供 `flash-q-bits` / `flash-s-bits` 参数，
q/s bits 保持 YAML 配置。运行时如需临时开关 FlashAttention，可传
`--enable-flash-attention` 或 `--disable-flash-attention`，只覆盖
`export.model.flash_attention.enable`。当前本地 HMONNX runtime 只验收 q/s=8。

GDR 同样默认由 YAML 控制。运行时需要临时覆盖时，可以使用
`--enable-fuse-gdr-ops` / `--disable-fuse-gdr-ops`，以及
`--enable-fuse-gdr-block-recurrent-ops` /
`--disable-fuse-gdr-block-recurrent-ops`。

## Export overrides

`export()` consumes the `QuantResult` from `quant()`.  Do not expose variant/profile/mode/base/quant as public parameters; select a YAML and use explicit overrides only when needed.

GDR fuse switches remain YAML-owned by default.  `fuse_gdr_ops` only controls
GDRChunkScan because it can change the prefill recurrent-state HMONNX I/O
contract; `fuse_gdr_block_recurrent_ops` controls GDRBlockTriInverse and
GDRRecurrentScan without changing model inputs/outputs:

```python
export_result = workflow.export(
    quant_result=quant_result,
    output_dir="./work_dirs/qwen3_5_export",
    device="cuda",
    config_overrides={
        "export.model.visual_config.max_size_h": 896,
        "export.model.visual_config.max_size_w": 896,
    },
)
```

代码里也可以通过 export config_overrides 临时覆盖 GDR fuse：

```python
config_overrides = {
    "export.model.fuse_gdr_ops": True,
    "export.model.fuse_gdr_block_recurrent_ops": True,
}
```

普通 full / visual-only / MTP / DFlash 不通过脚本参数选择模型结构，而是通过选择不同 YAML 文件决定。

## Golden 和 Quick Test

导出完成后可以生成 golden：

```python
workflow.dump_golden(
    export_result=export_result,
    device="cuda",
    input_messages={"text": "用中文简单介绍 Qwen3.5。"},
)
```

也可以直接使用示例脚本参数：

```bash
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir /path/to/hf_model \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --quant-output-dir work_dirs/qwen3_5_quant \
  --export-output-dir work_dirs/qwen3_5_export \
  --dump-golden \
  --quick-test \
  --overwrite

# 122B 导出必须启用 HUGE_MODEL_EXPORT_ENABLED，走 placeholder 分层导出以降峰值内存。
export HUGE_MODEL_EXPORT_ENABLED=1
# 可选：并行导出 placeholder 子图（默认 1）。多卡时可按可见 GPU 数设置。
export XH2MODELZOO_EXPORT_WORKERS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-122B-A10B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml \
  --quant-output-dir work_dirs/qwen3_5_122B_quant \
  --export-output-dir work_dirs/qwen3_5_122B_export \
  --quick-test \
  --overwrite

CUDA_VISIBLE_DEVICES=0,1,2,3 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3.5-122B-A10B \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full_gptq.yaml \
  --quant-output-dir work_dirs/qwen3_5_122B_gptq_quant \
  --export-output-dir work_dirs/qwen3_5_122B_gptq_export \
  --quick-test \
  --overwrite
```

quick test 使用：

```python
from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import quick_test_hmonnx

quick_result = quick_test_hmonnx(
    export_result,
    prompt="用中文介绍一下 Qwen3.5。",
    device="cuda",
    max_new_tokens=64,
    do_sample=False,
)
```

`quick_test_hmonnx()` 会根据导出的 meta 自动选择普通 generate 或 speculative decoding。MTP/DFlash 导出的接受率统计来自 `quick_result.accept_rate` 和 `quick_result.stats`。

命令行生成可以使用：

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config  work_dirs/qwen3_5_122B_export/hmquant_xh2_qwen3_5_122b_a10b_w4a8_256_2k_mpe256k_448x448_20260702/golden_meta_info.json \
  --prompt "用中文介绍一下 Qwen3.5" \
  --no-sample \
  --max-new-tokens 128 \
  --auto-offload \
  --cuda-graph \
  --use-v2

python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config  work_dirs/qwen3_5_9b_flashattention_fuse_gdr_export/hmquant_xh2_qwen3_5_9b_w4a8_256_256k_mpe256k_448x448_20260707/golden_meta_info.json \
  --prompt "你好，用一句话介绍北京。" \
  --think \
  --no-sample \
  --max-new-tokens 20480 \
  --auto-offload \
  --cuda-graph \
  --use-v2
```

MTP/DFlash 的专项 speculative decoding 验证可以使用：

```bash
python examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_spec_decode_test.py \
  --config work_dirs/qwen3_5_export/hmquant*/golden_meta_info.json \
  --prompt "写一首关于 AI 的短诗" \
  --max-new-tokens 128 \
  --benchmark-runs 1
```

## 示例脚本参数

`qwen3_5_workflow.py` 的常用参数：

- `--model-dir`：HF 模型目录；如果要从量化后的 HF 模型导出，也传该目录。
- `--config-path`：workflow YAML 路径。
- `--quant-output-dir`：量化输出目录。
- `--export-output-dir`：HMONNX 导出输出目录。
- `--device`：量化、导出、golden、quick test 使用的设备。
- `--overwrite`：删除已有导出目录后重跑。
- `--dump-golden`：导出后生成 golden。
- `--quick-test`：导出后运行 HMONNX quick test。
- `--export-from-quanted-model`：跳过量化，直接从 `--model-dir` 指定的 HF 目录导出。
- `--model-name`：临时覆盖 `export.model.model_name`，用于 YAML 架构相同但导出命名不同的 checkpoint；`.` 和 `-` 会归一化成 `_`。
- `--bits`：临时覆盖 `quant.bits`。
- `--max-size-h` / `--max-size-w`：临时覆盖 visual tower 输入尺寸。
- `--context-max-length` / `--context-length`：临时覆盖导出最大上下文长度（`export.model.context_max_length`），并在 MTP/DFlash 配置存在时同步覆盖 draft cache 长度。
- `--enable-flash-attention` / `--disable-flash-attention`：临时覆盖 FlashAttention 开关。
- `--enable-fuse-gdr-ops` / `--disable-fuse-gdr-ops`：临时覆盖 GDRChunkScan fuse。
- `--enable-fuse-gdr-block-recurrent-ops` / `--disable-fuse-gdr-block-recurrent-ops`：临时覆盖 GDRBlockTriInverse/GDRRecurrentScan fuse。

新增 workflow 集成优先复用 `AutoLLMWorkflow.from_config()` 和 checked-in YAML，不要新增一套模型结构参数。
