# Qwen3.5 / Qwen3.6 Merak workflow README

Use `Qwen35Workflow` for dense Qwen3.5/Qwen3.6 and Qwen3.6 MoE exports.  The public API is intentionally small: quantization only receives paths/device/overrides, and export consumes the returned `QuantResult`.  Model topology, AutoRound/GPTQModel settings, visual tower size, MTP/DFlash, and GDR fuse are YAML config or override fields.

```python
from xhmodel_merak.xh_llm.models.qwen3_5 import Qwen35Workflow

workflow = Qwen35Workflow.from_config(hf_model_dir, config_path, seed=1024, debug=False)
quant_result = workflow.quant(output_dir, device, config_overrides=None)
export_result = workflow.export(quant_result, output_dir, device, config_overrides=None)
```

See `examples_merak/llm/qwen3_5/qwen3_5_workflow.py` for a copyable standard example with simple constants.

Full model / HMONNX IO documentation: `docs/qwen3_5_hmonnx_io_spec.md`.

## Config defaults and structured help

Upstream integrations should read defaults and help from the model package instead of copying
per-model tables:

```python
from xhmodel_merak.xh_llm.models.qwen3_5 import (
    get_default_export_config,
    get_default_quant_config,
    get_default_workflow_config,
    get_export_config_help,
    get_model_docs,
    get_quant_config_help,
    list_recommended_configs,
)

quant_cfg = get_default_quant_config()
workflow_cfg = get_default_workflow_config(name="qwen3_5_9b_full_mtp")
export_cfg = get_default_export_config(family="dense", model_size="9b", variant="dflash")
model_docs = get_model_docs()
```

- `get_default_quant_config()` returns the default AutoRound -> GPTQModel HF quant section.
- `get_default_workflow_config(...)` returns one checked-in recommended YAML as a deep copy.
- `get_default_export_config(...)` returns only the `export` section from one recommended YAML.
- `get_quant_config_help()` / `get_export_config_help()` return field-level structured help.
- `get_model_docs()` returns supported model paths, validation scope, and the full doc link.
- `list_recommended_configs()` returns the stable YAML index (`name`, `family`, `model_size`,
  `variant`, `visual_size`, `config_path`, `model_type`, `model_name`).

Use `name=...` when the exact YAML is known.  Use `family/model_size/variant` for UI selection;
`visual_only` needs `visual_size=448` or `896` to be unique.

## Recommended YAML configs

YAML filenames are topology names only.  Do not put quant-format tokens such as `w4`, `w8`, `gptq`, or `autoround` in workflow YAML filenames.

- 9B full: `configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml`
- 9B MTP: `configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml`
- 9B DFlash: `configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml`
- 9B visual-only: `configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_visual_only_448.yaml` or `qwen3_5_9b_visual_only_896.yaml`
- 27B full/MTP/DFlash/visual-only: `configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/*.yaml`
- 35B-A3B MoE full/MTP/DFlash/visual-only: `configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/*.yaml`

Default full configs include the visual branch when `export.model.visual_config` exists.  Visual-only configs are separate full visual tower exports.  Default visual buckets are `448x448` and `896x896`; change them in YAML or with overrides.  Full configs use `{"export.model.visual_config.max_size_w": 896, "export.model.visual_config.max_size_h": 896}`; visual-only configs use `{"export.model.max_size_w": 896, "export.model.max_size_h": 896}`.

## Quantization contract

Default YAML quantization uses AutoRound and saves a GPTQModel-compatible HF artifact.
The workflow mirrors the existing mode1 LLM-only shell scripts without calling or editing
`third_party/auto-round` directly:

```yaml
quant:
  algorithm: autoround
  output_format: gptqmodel_hf
  artifact_format: gptqmodel_hf
  bits: 4
  group_size: 64
  sym: true
  iters: 200
  seed: 42
  quant_nontext_module: false
  autoround_format: auto_gptq        # dense script default
  calibration:
    dataset: NeelNanda/pile-10k
    nsamples: 128
    seqlen: 2048
  runtime:
    batch_size: 8
    trust_remote_code: true
```

MoE YAMLs add the options that were previously only in `scripts_qwen35moe/run_mode1_llm_only.sh`:

```yaml
quant:
  autoround_format: auto_round:gptqmodel
  runtime:
    batch_size: 8
    trust_remote_code: true
    device_map: balanced
    low_gpu_mem_usage: true
  moe:
    attn_bits: 8
    shared_expert_bits: 8
```

Script comparison summary:

| Field | Dense mode1 | MoE mode1 | Workflow YAML |
| --- | --- | --- | --- |
| LLM bits/group | `--llm_bits 4 --llm_group_size 64` | same | `bits: 4`, `group_size: 64` |
| LLM-only | `--mode llm-only` | same | `quant_nontext_module: false` |
| calibration | `pile-10k`, `nsamples=128`, `seqlen=2048`, `batch_size=8` | same defaults | `calibration.*`, `runtime.batch_size` |
| seed/sym/iters | `--seed 42 --sym --iters 200` | same defaults | `seed`, `sym`, `iters` |
| save format | `auto_gptq` | `auto_round:gptqmodel` | `autoround_format` |
| MoE-only knobs | n/a | `device_map`, `low_gpu_mem_usage`, `attn_bits`, `shared_expert_bits` | `runtime.*`, `moe.*` |

`group_size` must remain `64` for Qwen3.5/Qwen3.6 workflow quantization.
Use overrides for run-specific MoE experiments, e.g.
`{"quant.iters": 400, "quant.calibration.nsamples": 256}` to reproduce an `n256-iter400` run.

### Quant-only CLI

Use the quant-only entrypoint when upstream only needs to create or reuse the
GPTQModel-compatible HF artifact.  The CLI intentionally exposes only paths and
source selection; AutoRound details such as `group_size=64`, `bits`, `dataset`,
MoE `attn_bits`, and `shared_expert_bits` stay in YAML.

```bash
# Dense 9B AutoRound -> GPTQModel-compatible HF artifact.
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_quant.py \
  --hf-model-dir weights/Qwen3.5-9B \
  --config configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --output-dir work_dirs/qwen3_5_9b_quant \
  --device cuda:0 \
  --force

# MoE 35B-A3B AutoRound with YAML-configured balanced device_map,
# low_gpu_mem_usage, attn_bits=8, and shared_expert_bits=8.
CUDA_VISIBLE_DEVICES=0,1 python examples_merak/llm/qwen3_5/qwen3_5_quant.py \
  --hf-model-dir weights/Qwen3.6-35B-A3B \
  --config configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml \
  --output-dir work_dirs/qwen3_6_35b_a3b_quant \
  --device cuda \
  --force
```

For validation-only paths, the same CLI can return a `QuantResult` without
running AutoRound:

```bash
python examples_merak/llm/qwen3_5/qwen3_5_quant.py \
  --hf-model-dir weights/Qwen3.5-9B \
  --config configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml \
  --output-dir work_dirs/qwen3_5_9b_quant \
  --base
```

For base validation, override the top-level quant section to `None`:

```python
quant_result = workflow.quant(output_dir, device, config_overrides={"quant": None})
```

For an existing externally quantized HF/GPTQModel artifact, replace the whole quant section:

```python
# Qwen3.5 9B
config_overrides = {
    "quant": {
        "algorithm": "existing_hf",
        "artifact_format": "gptqmodel_hf",
        "source_algorithm": "autoround",
        "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
    }
}

# Qwen3.6 35B-A3B MoE
config_overrides = {
    "quant": {
        "algorithm": "existing_hf",
        "artifact_format": "gptqmodel_hf",
        "source_algorithm": "autoround",
        "existing_hf_model_dir": "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400",
    }
}
```

## Export overrides

`export()` always consumes the `QuantResult` from `quant()`.  Do not expose variant/profile/mode/base/quant as public parameters; select a YAML and use explicit overrides only when needed.

`fuse_gdr_ops` remains config/override-only:

```python
export_result = workflow.export(
    quant_result,
    output_dir,
    device,
    config_overrides={"export.model.fuse_gdr_ops": True},
)
```

## Running

```bash
conda activate xhquant_55
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_workflow.py
```

`qwen3_5_workflow.py` runs the full demo path:

```text
quant -> export -> dump_golden -> quick_test_hmonnx
```

## Quick HMONNX conversation and SpecDecode metrics

Use `quick_test_hmonnx()` after export to run a fast runtime check.  It finds the
exported `hmquant*/golden_meta_info.json`, runs normal HMONNX generate for full
or visual exports, and automatically switches to MTP/DFlash speculative decoding
when the meta contains `spec_decode.mode`.

```python
from xhmodel_merak.xh_llm.models.qwen3_5 import quick_test_hmonnx

quick_result = quick_test_hmonnx(
    export_result,
    prompt="用中文介绍一下 Qwen3.5",
    device="cuda:0",
    max_new_tokens=64,
    do_sample=False,
)

print(quick_result.output_text)
print(quick_result.tokens_per_second)
print(quick_result.spec_decode_mode)
print(quick_result.accept_rate)
```

The MTP/DFlash acceptance rate is computed as:

```text
accept_rate = accepted_drafts_total / draft_tokens_total
```

`accepted_drafts_total` and `draft_tokens_total` come from the Qwen3.5
spec-decode runtime with `return_stats=True`.  The same result also records
`avg_accepted_per_round` and `accepted_drafts_per_round` in `quick_result.stats`.

For CLI use, run the normal HMONNX generate wrapper:

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_hmonnx_generate.py \
  --config work_dirs/qwen3_5_9b_workflow_export/hmquant*/golden_meta_info.json \
  --prompt "用中文介绍一下 Qwen3.5" \
  --no-sample \
  --max-new-tokens 128
```

For MTP/DFlash acceptance-rate checks, run the spec-decode wrapper:

```bash
python examples_merak/llm/qwen3_5/qwen3_5_xh_spec_decode_test.py \
  --config work_dirs/qwen3_5_validation_matrix/qwen35_9b_mtp_existing_hf/export/hmquant*/golden_meta_info.json \
  --prompt "写一首关于 AI 的诗" \
  --max-new-tokens 128 \
  --benchmark-runs 1
```

For programmatic integrations, call the lower-level helpers directly:

- `find_hmonnx_meta_file(export_result_or_path)`
- `hmonnx_generate(meta_file=..., prompt=...)`
- `spec_decode_generate(meta_file=..., prompt=...)`
- `quick_test_hmonnx(export_result_or_meta_file, prompt=...)`

## Runtime validation matrix

Use `qwen3_5_validation_matrix.py` for the requested first-pass verification matrix.  It runs 9B and 35B-A3B with base HF weights and existing external HF/GPTQModel quant artifacts, each with `fuse_gdr_ops=false` and `fuse_gdr_ops=true`.  This does not add public workflow parameters; each case is just a YAML plus explicit `config_overrides`.

```bash
# Check dependencies, YAMLs, base weights, and external quant artifacts first.
conda activate xhquant_55
python examples_merak/llm/qwen3_5/qwen3_5_validation_matrix.py --preflight-only

# Run the full 8-case matrix.
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_validation_matrix.py --force

# Run one case without golden generation.
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_validation_matrix.py \
  --scenario qwen35_9b_existing_hf_fuse_false \
  --skip-golden \
  --force

# Run one case and write quick_test_result.json with output and accept_rate.
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_validation_matrix.py \
  --scenario qwen35_9b_mtp_existing_hf \
  --quick-test \
  --quick-test-max-new-tokens 64 \
  --force

# Reuse an already exported scenario and only run quick_test_hmonnx.
CUDA_VISIBLE_DEVICES=0 python examples_merak/llm/qwen3_5/qwen3_5_validation_matrix.py \
  --scenario qwen35_9b_mtp_existing_hf \
  --skip-export \
  --quick-test \
  --quick-test-max-new-tokens 256
```

Matrix:

| Scenario | HF model | Quant source | fuse_gdr_ops |
| --- | --- | --- | --- |
| `qwen35_9b_base_fuse_false` | `weights/Qwen3.5-9B` | base override `{"quant": None}` | `false` |
| `qwen35_9b_base_fuse_true` | `weights/Qwen3.5-9B` | base override `{"quant": None}` | `true` |
| `qwen35_9b_existing_hf_fuse_false` | `weights/Qwen3.5-9B` | `weights/Qwen3.5-9B-mode1-llm-only` | `false` |
| `qwen35_9b_existing_hf_fuse_true` | `weights/Qwen3.5-9B` | `weights/Qwen3.5-9B-mode1-llm-only` | `true` |
| `qwen36_35b_a3b_base_fuse_false` | `weights/Qwen3.6-35B-A3B` | base override `{"quant": None}` | `false` |
| `qwen36_35b_a3b_base_fuse_true` | `weights/Qwen3.6-35B-A3B` | base override `{"quant": None}` | `true` |
| `qwen36_35b_a3b_existing_hf_fuse_false` | `weights/Qwen3.6-35B-A3B` | `weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400` | `false` |
| `qwen36_35b_a3b_existing_hf_fuse_true` | `weights/Qwen3.6-35B-A3B` | `weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400` | `true` |

Legacy Python config entrypoints and generation helpers remain supported. New workflow integrations should use `Qwen35Workflow` with workflow YAMLs.
