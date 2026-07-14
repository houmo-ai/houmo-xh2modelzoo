# Qwen3-Next Merak

Text-only Merak export for `Qwen3NextForCausalLM`, with optional
`Qwen3NextMTP` draft graphs. The target graph is always exported with ordinary
attention. Page attention is derived from that graph at runtime.

Qwen3-Next reuses the Qwen3.5-MoE workflow implementation and cache ABI. Each
Qwen3.5-MoE and Qwen3-Next workflow YAML is nevertheless a standalone, complete
configuration with its own quantization and export fields:

```text
configs_merak/workflows/xh2a/llm_models/qwen3_next/80b_a3b/
  qwen3_next_80b_a3b_full.yaml
  qwen3_next_80b_a3b_full_gptq.yaml
  qwen3_next_80b_a3b_full_fa_all16.yaml
  qwen3_next_80b_a3b_full_mtp_gptq.yaml
```

The shared workflow runner can be used directly (the Qwen3-Next workflow is a
text-only specialization of `Qwen35Workflow`):

```bash
conda activate xhquant_55
CUDA_VISIBLE_DEVICES=0,1 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py \
  --model-dir weights/Qwen3-Next-80B-A3B-Instruct \
  --config-path configs_merak/workflows/xh2a/llm_models/qwen3_next/80b_a3b/qwen3_next_80b_a3b_full_mtp_gptq.yaml \
  --quant-output-dir work_dirs/qwen3_next_quant \
  --export-output-dir work_dirs/qwen3_next_export
```

Reduced validation export using the GPTQModel artifact with MTP:

```bash
conda activate xhquant_55
CUDA_VISIBLE_DEVICES=0,1 python examples_merak/llm/qwen3_next/qwen3_next_xh_export_hmonnx.py \
  --model work_dirs/qwen3_next_gptq_first4_mtp_defused \
  --output-dir work_dirs/qwen3_next_merak_first4 \
  --max-layers 4 --mtp --split-conv-cache
```

For focused exporter experiments, the direct script also exposes
`--fuse-gdr-ops`, `--fuse-gdr-block-recurrent-ops`,
`--manual-depthwise-conv1d`, `--[no-]split-conv-cache`, and
`--flash-attention-bits {8,16}`. For checked-in workflows, update the complete
YAML for each affected Qwen3.5-MoE and Qwen3-Next profile so their optimization
settings remain explicitly synchronized.

The four target-runtime combinations use one exported target graph:

```bash
# ordinary attention + eager
python examples_merak/llm/qwen3_next/qwen3_next_xh_hmonnx_generate.py --config META
# ordinary attention + CUDA Graph
python examples_merak/llm/qwen3_next/qwen3_next_xh_hmonnx_generate.py --config META --cuda-graph
# page attention + eager
python examples_merak/llm/qwen3_next/qwen3_next_xh_hmonnx_generate.py --config META --page-attention
# page attention + CUDA Graph
python examples_merak/llm/qwen3_next/qwen3_next_xh_hmonnx_generate.py --config META --page-attention --cuda-graph
```

There is deliberately no separate page-attention export. Runtime conversion
fuses page attention from the same ordinary-attention HMONNX before eager or
CUDA Graph execution.
