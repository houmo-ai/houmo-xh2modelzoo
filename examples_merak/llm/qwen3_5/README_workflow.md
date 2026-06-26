# Qwen3.5 / Qwen3.6 Merak Workflow 使用说明

本文说明 Qwen3.5、Qwen3.6 dense、Qwen3.6 MoE 在 Merak workflow 下的推荐用法。统一入口是：

```python
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

workflow = AutoLLMWorkflow.from_config(
    hf_model_dir="/path/to/hf_model",
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

- `quant: null`：跳过量化，直接使用传入的 `hf_model_dir`。
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
```

代码里等价写法是：

```python
quant_result = workflow.quant(
    output_dir="./work_dirs/qwen3_5_quant",
    device="cuda",
    config_overrides={"quant": None},
)
```

此时 `QuantResult.skipped` 为 `True`，`export()` 会直接使用 workflow 初始化时传入的 `hf_model_dir`。

## 导出

导出阶段消费 `quant()` 返回的 `QuantResult`：

```python
export_result = workflow.export(
    quant_result=quant_result,
    output_dir="./work_dirs/qwen3_5_export",
    device="cuda",
)
```

如需临时调整 visual tower 尺寸，可以通过 override 修改 YAML 字段：

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

GDR fuse 也通过配置或 override 控制：

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
  --config work_dirs/qwen3_5_export/hmquant*/golden_meta_info.json \
  --prompt "用中文介绍一下 Qwen3.5" \
  --no-sample \
  --max-new-tokens 128
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
- `--bits`：临时覆盖 `quant.bits`。
- `--max-size-h` / `--max-size-w`：临时覆盖 visual tower 输入尺寸。

新增 workflow 集成优先复用 `AutoLLMWorkflow.from_config()` 和 checked-in YAML，不要新增一套模型结构参数。
