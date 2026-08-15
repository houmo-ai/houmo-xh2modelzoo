# Ling 3 Flash Merak

Ling 3 Flash uses the standard Merak workflow and consumes either an original
HF checkpoint or an already quantized GPTQModel/AutoRound HF directory. The
checked-in workflow configurations live with the other Merak configurations:

```text
configs_merak/workflows/xh2a/llm_models/ling_3_flash/
  ling_3_flash_autoround.yaml
  ling_3_flash_gptq.yaml
```

Run commands from the repository root in the `xhquant_55` environment. The
examples below consume the existing quantized artifacts directly and do not
run or modify GPTQModel.

## Export existing quantized checkpoints

AutoRound checkpoint, streamed low-memory path:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n xhquant_55 \
  python examples_merak/llm/ling_3_flash/export.py \
  --model-dir work_dirs/ling_3_flash_autoround_w8w4_g64_sym \
  --config-path configs_merak/workflows/xh2a/llm_models/ling_3_flash/ling_3_flash_autoround.yaml \
  --export-output-dir work_dirs/ling_3_flash_autoround_hmonnx_lowmem \
  --export-from-quanted-model \
  --low-memory \
  --overwrite
```

GPTQ checkpoint, regular full-model path:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n xhquant_55 \
  python examples_merak/llm/ling_3_flash/export.py \
  --model-dir work_dirs/ling_3_flash_gptq_w8w4_g64_sym \
  --config-path configs_merak/workflows/xh2a/llm_models/ling_3_flash/ling_3_flash_gptq.yaml \
  --export-output-dir work_dirs/ling_3_flash_gptq_hmonnx_regular \
  --export-from-quanted-model \
  --no-low-memory \
  --overwrite
```

Use both export modes for each checkpoint in a release matrix. `--low-memory`
materializes one placeholder subtree at a time; `--no-low-memory` forces the
ordinary full-model path. Omitting both preserves `HUGE_MODEL_EXPORT_ENABLED`
from the parent process. The streamed implementation also supports original,
non-quantized safetensors checkpoints; `--export-from-quanted-model` is only
the input validation/quant-skip contract for existing quantized artifacts.

Both YAMLs enable `fuse_gdr_ops` and
`fuse_gdr_block_recurrent_ops`. FlashAttention remains disabled by default and
can be enabled for an explicit comparison with `--flash-attention`.

## HMONNX validation

The inference CLI defaults to HMONNX V2, packed W4 initializers, and CUDA
Graph. A short semantic smoke test is:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n xhquant_55 \
  python examples_merak/llm/ling_3_flash/infer.py \
  --meta work_dirs/ling_3_flash_autoround_hmonnx_lowmem/hmquant*/golden_meta_info.json \
  --device cuda:0 \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 32 \
  --print-token-ids
```

`--no-use-v2`, `--no-pack-w4`, and `--no-cuda-graph` are diagnostic
overrides. Large W4 models normally need the default contract.
