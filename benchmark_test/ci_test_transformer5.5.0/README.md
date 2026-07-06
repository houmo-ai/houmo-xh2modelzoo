# Transformer 5.5.0 CI lane

This lane is intentionally small: it downloads pre-trimmed model archives from
Artifactory with `wget`, runs workflow quantization, then exports HMONNX. It
reuses the existing full workflow YAMLs and applies runtime overrides for CI.
It does not generate CI YAMLs, upload artifacts, or run demo/generate.

The Qwen3.5 cases follow the public workflow pattern documented in
`examples_merak/llm/qwen3_5/README_workflow.md`: instantiate with
`AutoLLMWorkflow.from_config()`, then run `quant()` and `export()` against the
checked-in workflow YAMLs.  CI only adds small runtime overrides for trimmed
models, calibration size, naming, and quant bits.

## Coverage

Default CI runs every case below with both `gptq` and `autoround` backends:

- `Qwen3.5-0.8B`: first 4 blocks, W8
- `Qwen3.5-2B`: first 4 blocks, W8
- `Qwen3.5-4B`: first 4 blocks, W4
- `Qwen3.5-9B`: first 4 blocks, W4
- `Qwen3.5-27B`: first 4 blocks, W4
- `Qwen3.5-35B-A3B`: first 4 blocks, W4
- `gemma-4-E2B-it`: first 6 blocks, W4
- `gemma-4-E4B-it`: first 6 blocks, W4
- `gemma-4-31B-it`: first 6 blocks, W4
- `gemma-4-26B-A4B-it`: first 6 blocks, W4

`Qwen3.5-122B-A10B` is not part of CI.

## GPTQModel checkout

This lane imports GPTQModel recipe APIs from a Gerrit checkout without
installing GPTQModel in the test job. Configure `GPTQMODEL_SOURCE_DIR` and
`PYTHONPATH` before running CI; see `GPTQMODEL_ENV.md` for the exact commands
for IT/image setup. `run.sh` installs the lightweight packages that this
lane needs from GPTQModel requirements but xh2modelzoo does not declare, plus
Qwen/Gemma import-time packages. It generates temporary pip constraints from
the preinstalled `torch`, `torchvision`, `torchaudio`, `triton`, and
`nvidia-*` packages so missing Python dependencies are installed without
upgrading the CUDA stack. On the Ubuntu 24 CI image the script uses
`--break-system-packages` because the job runs against the image-owned system
Python.

## Calibration data

AutoRound uses a local JSONL at:

```text
data/calib_data/NeelNanda-pile-10k.jsonl
```

The JSONL is committed with the repository so CI does not need to download
calibration data or reach public dataset hosts.

## Run

```bash
./benchmark_test/ci_test_transformer5.5.0/run.sh
```

Run one model or one backend case:

```bash
TRANSFORMER55_CI_CASES=Qwen3.5-35B-A3B ./benchmark_test/ci_test_transformer5.5.0/run.sh
TRANSFORMER55_CI_CASES=Qwen3.5-35B-A3B-autoround ./benchmark_test/ci_test_transformer5.5.0/run.sh
```
