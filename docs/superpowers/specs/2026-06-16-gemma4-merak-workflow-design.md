# Gemma4 Merak Workflow Migration Design

Date: 2026-06-16

## Background

Gemma4 in `xh2modelzoo` currently has several partially overlapping Merak surfaces:

- E4B, 31B, and 26B-A4B must now converge on the same public `gemma4_series` workflow/model registration surface.
- Legacy `gemma4e` and `gemma4_moe` packages may remain as internal compatibility code, but they must not override the unified public registry keys.
- `examples_merak/llm/gemma4_series` already attempts one script for Dense/MoE export and generation, but it is still a script-level orchestration surface rather than the public API that upstream `imodelzoo/customized_models` should call.

The newer workflow examples for MinerU2.5, Qwen2-VL, Qwen3 legacy, and the Qwen3.5/Qwen3.6 migration use a cleaner contract:

1. Create a model-family workflow from `model_dir` and a workflow YAML config.
2. Run `workflow.quant(output_dir=...)`.
3. Run `workflow.export(quant_result=..., output_dir=...)`.
4. Optionally run `workflow.dump_golden(...)`.

Gemma4 must follow that same standard. In addition, `docs/gemma4_vit_padded_input_design_20260616.md` is now the finalized ViT input contract, and every Gemma4 model variant must obey it.

## Goals

- Provide one stable Gemma4 model-family API for E4B, 31B, and 26B-A4B.
- Keep upstream integration simple: choose a recommended YAML, pass model/output paths, call workflow methods.
- Use the Qwen3.5 workflow migration as the implementation standard: model-package workflow class, topology-named YAML, default quant config, explicit base validation override, existing-HF override, golden dumping, and validation matrix.
- Consolidate E4B, 31B, and 26B-A4B around the existing `gemma4_series` public model/config class: `Gemma4ForConditionalGeneration` / `XHGemma4ModelConfig`.
- Reuse legacy MoE internals only as private compatibility code where necessary; do not expose `_with_mask` or `gemma4_moe` as the new public API.
- Make the finalized padded ViT contract mandatory for Dense and MoE visual paths.
- Preserve old scripts until the new workflow passes complete validation.

## Non-goals

- Do not expose public API parameters such as `variant`, `profile`, `mode`, `component`, `w4a8`, or `autoround`.
- Do not create separate YAMLs solely named by quantization source or precision, such as `*_w4a8.yaml`, `*_gptq.yaml`, or `*_autoround.yaml`.
- Do not make visual input shape policy optional per model; all visual paths use the padded ViT contract.
- Do not delete old Gemma4 scripts in the first implementation patch.
- Do not require upstream callers to know xhquant operator-level quantization behavior or whether the implementation internally dispatches Dense vs MoE.

## Model Comparison and Implementation Implications

| Model | Dataset path | Family | Text topology | Vision topology | Implementation implication |
|---|---|---|---|---|---|
| Gemma4 E4B IT | `/data01/datasets/gemma-4-E4B-it` | Dense | 42 layers, hidden 2560, sliding/full attention pattern, no experts | 16-layer ViT, hidden 768, patch 16, pooling kernel 3 | Must run through the unified `gemma4_series` Dense path. Any E-series-only attributes become optional/defaulted in `gemma4_series` instead of requiring a public `gemma4e` route. |
| Gemma4 31B IT | `/data01/datasets/gemma-4-31B-it` | Dense | 60 layers, hidden 5376, sliding/full attention pattern, no experts | 27-layer ViT, hidden 1152, patch 16, pooling kernel 3 | Existing `gemma4` is the anchor implementation. 31B must remain compatible with the new padded ViT contract. |
| Gemma4 26B-A4B IT | `/data01/datasets/gemma-4-26B-A4B-it` | MoE topology under same Gemma4 HF class | 30 layers, hidden 2816, 128 experts, active A4B-style sparse MLP | 27-layer ViT, hidden 1152, patch 16, pooling kernel 3 | New configs must still use `Gemma4ForConditionalGeneration` and `XHGemma4ModelConfig`; MoE differences are topology details, not a separate public registration. |

All three models share the same user contract, config class, tokenizer/processor expectations, golden API, and visual input protocol. Legacy Dense/MoE implementation branches may exist internally, but the public workflow/YAML layer must not require upstream to select a different registered model class.  In code, `XHGemma4ModelConfig` reads the HF `text_config.enable_moe_block` flag and `XHGemma4Model.from_pretrained()` dispatches 26B-A4B to an internal MoE adapter while preserving the public `Gemma4ForConditionalGeneration` model type and `XHGemma4ModelConfig` config class.

## Public API

The recommended API is a workflow object exported from the Gemma4 Series model package:

```python
from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

workflow = Gemma4SeriesWorkflow.from_config(
    model_dir="/data01/datasets/gemma-4-31B-it",
    config_path=(
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/"
        "gemma4_31b_full.yaml"
    ),
    seed=1024,
    debug=False,
)

quant_result = workflow.quant(
    output_dir="./work_dirs/gemma4_31b_quant",
    device="cuda",
)

export_result = workflow.export(
    quant_result=quant_result,
    output_dir="./work_dirs/gemma4_31b_export",
    device="cuda",
)

workflow.dump_golden(
    export_result=export_result,
    device="cuda",
    input_messages={"text": "用中文介绍一下你自己。"},
)
```

Thin convenience functions may exist, but they must wrap the same workflow flow:

```python
from xhmodel_merak.xh_llm.models.gemma4_series.workflow import export, quant

quant_result = quant(
    model_dir="/data01/datasets/gemma-4-E4B-it",
    config_path="configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml",
    output_dir="./work_dirs/gemma4_e4b_quant",
    device="cuda",
)

export_result = export(
    model_dir="/data01/datasets/gemma-4-E4B-it",
    config_path="configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml",
    quant_result=quant_result,
    output_dir="./work_dirs/gemma4_e4b_export",
    device="cuda",
)
```

The thin functions intentionally require the same `model_dir` and `config_path` as
`Gemma4SeriesWorkflow.from_config(...)`; they are convenience wrappers, not hidden global
state.  This keeps `imodelzoo` integration explicit and side-effect free.

The stable model-family surface should provide:

- `Gemma4SeriesWorkflow.from_config(...)`
- `Gemma4SeriesWorkflow.quant(...)`
- `Gemma4SeriesWorkflow.export(...)`
- `Gemma4SeriesWorkflow.dump_golden(...)`
- `list_recommended_configs()`
- `get_quant_config_help()`
- `get_export_config_help()`
- `dump_quant_config_template(path)`
- `dump_export_config_template(path)`

## Quant Semantics

The default YAML must express the recommended quantization process. It must not default to `quant: null`.

Default quant section:

```yaml
quant:
  algorithm: autoround
  output_format: gptqmodel_hf
  artifact_format: gptqmodel_hf
  bits: 4
  group_size: 64
  calibration:
    dataset: wikitext
    split: train
    nsamples: 128
    seqlen: 2048
  runtime:
    batch_size: 1
    trust_remote_code: true
```

Semantics:

- `algorithm: autoround` describes how the quantized weights are produced.
- `output_format/artifact_format: gptqmodel_hf` describes how the quantized artifact is saved and loaded.
- `bits: 4` and `group_size: 64` are the recommended initial defaults, matching Qwen3.5 workflow conventions.
- `quant()` returns `QuantResult(raw_model_dir=..., quanted_model_dir=..., skipped=False)` for produced or externally supplied quantized HF artifacts.

### Base Validation

Base model validation must explicitly disable quantization at call time:

```python
quant_result = workflow.quant(
    output_dir="./work_dirs/gemma4_31b_base_quant",
    device="cuda",
    config_overrides={"quant": None},
)
```

This returns a skipped `QuantResult`, and `export()` uses the original `model_dir`.

### Existing Quantized HF Artifacts

When validation or integration already has a quantized HF directory, callers override the quant section rather than selecting a different YAML:

```python
quant_result = workflow.quant(
    output_dir="./work_dirs/gemma4_31b_existing_quant",
    device="cuda",
    config_overrides={
        "quant": {
            "algorithm": "existing_hf",
            "artifact_format": "gptqmodel_hf",
            "source_algorithm": "autoround",
            "existing_hf_model_dir": "/path/to/gemma4-31b-autoround-gptqmodel",
        }
    },
)
```

For Gemma4 26B-A4B MoE, the same pattern applies. If MoE export needs `fallback_hf_model` for missing float weights, that path belongs in the export YAML or in `config_overrides`, not in a separate API parameter.

## Export Semantics

`export()` always consumes `QuantResult`.

- If `quant_result.skipped` is true, export uses the original HF model directory.
- If `quant_result.skipped` is false, export uses `quant_result.quanted_model_dir`.
- `export.model.hf_model` in YAML remains a runtime-filled placeholder.
- The default export quant scheme is `w8a8h1_sefp`; when GPTQModel-format HF weights are used, xhquant internals determine effective W4/W8 behavior from the weight format/range.
- Implementation-specific flags such as `fuse_gdr_ops`, visual-only export, or MoE fallback paths stay in YAML/config overrides, not stable function parameters.

The workflow validates that the selected YAML resolves to one of the accepted Gemma4 model classes/config classes before export starts.

For discovery and integration scaffolding, callers can inspect:

```python
from xhmodel_merak.xh_llm.models import gemma4

print(gemma4.list_recommended_configs())
print(gemma4.get_quant_config_help())
print(gemma4.get_export_config_help())
gemma4.dump_quant_config_template("./gemma4_quant_template.yaml")
gemma4.dump_export_config_template("./gemma4_export_template.yaml")
```

## Mandatory ViT Padded Input Contract

All Gemma4 visual models must follow `docs/gemma4_vit_padded_input_design_20260616.md`. The workflow, YAML defaults, visual processors, visual model wrappers, and golden paths all treat this as the single source of truth.

### Host/Processor Outputs

The Host side must produce:

```text
pixel_values:              [1, 2520, 768]
pixel_position_ids:        [1, 2520, 2]
pool_indices:              [1, 280, 9]
visual_attention_mask:     [1, 1, 1, 2520]
valid_soft_token_count:    scalar / host metadata
```

Rules:

- `pixel_values` are official padded patch inputs, not fixed-square image tensors.
- Real patch positions use `[x, y]`.
- Padding positions use `[0, 0]`, not HF's native `[-1, -1]`.
- `visual_attention_mask` is the Python-side name; the exported ViT ONNX input name remains `attention_mask`.
- `valid_soft_token_count` is passed through metadata needed by the LLM side to trim image embeddings.

### ViT Graph Inputs

The exported visual graph input names and shapes are:

```text
pixel_values:        [1, 2520, 768]   float
pixel_position_ids:  [1, 2520, 2]     int32
pool_indices:        [1, 280, 9]      int32
attention_mask:      [1, 1, 1, 2520]  float
```

The visual graph must implement:

- patch projection from padded patch vectors;
- position embedding via `pixel_position_ids` gather;
- RoPE via precomputed cos/sin tables and `pixel_position_ids` gather;
- attention key mask add using `attention_mask`;
- pooling via Host-generated `pool_indices` gather/reduce;
- fixed output `[1, 280, text_hidden_size]`.

The visual graph must not reintroduce:

- fixed-square-only real patch inputs;
- NPU graph-side pooling index generation;
- graph-side `Cos/Sin` for RoPE;
- `OneHot + MatMul` position embedding;
- clamping to handle `[-1, -1]` padding positions.

### LLM Side Use

The LLM side must trim visual output before replacing `<image>` token embeddings:

```python
image_embeds = image_embeds[:, :valid_soft_token_count, :]
```

The generated prompt must contain exactly `valid_soft_token_count` image tokens. If token count and visual feature count differ, generation/export validation should fail early with a clear error.

### Variant Requirements

- E4B Dense, 31B Dense, and 26B-A4B MoE all use the same visual input contract.
- Differences in visual hidden size or number of vision layers are internal model config facts; they must not change the public visual input protocol.
- MoE may keep its own internal visual export path temporarily, but its exported ONNX/HMONNX protocol must match the padded contract before the workflow is considered complete.

## YAML Naming Rules

YAML names describe export topology, not quantization format.

Use:

- `*_full.yaml`: full model export including text prefill/decode and configured visual branch.
- `*_visual_only_448.yaml`: visual tower only using the padded 2520-patch protocol; the suffix denotes recommended validation image bucket, not graph shape.
- Future topology-specific YAMLs may use names such as `*_full_mtp.yaml` only if Gemma4 gains such topology support.

Recommended files:

```text
configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/
  gemma4_e4b_full.yaml
  gemma4_e4b_visual_only_448.yaml

configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/
  gemma4_31b_full.yaml
  gemma4_31b_visual_only_448.yaml

configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/
  gemma4_26b_a4b_full.yaml
  gemma4_26b_a4b_visual_only_448.yaml
```

## Representative YAML Shapes

### Dense Full Export

```yaml
quant:
  algorithm: autoround
  output_format: gptqmodel_hf
  artifact_format: gptqmodel_hf
  bits: 4
  group_size: 64
  calibration:
    dataset: wikitext
    split: train
    nsamples: 128
    seqlen: 2048
  runtime:
    batch_size: 1
    trust_remote_code: true

export:
  model:
    chip_arch: XH2a
    model_type: Gemma4ForConditionalGeneration
    hf_model: null
    model_name: xh2_gemma4_31b_full_256_2k
    context_max_length: 2048
    prefill_chunk_length: 256
    max_pe_length: 32768
    use_cache: true
    num_logits_to_keep: 1
    quant_scheme:
      quant_type: w8a8h1_sefp
      nodes:
        lm_head:
          quant_type: w8a8h1_sefp
      ops: {}
    visual_config:
      export_mode: padded
      image_seq_length: 280
      max_patches: 2520
      patch_size: 16
      pooling_kernel_size: 3
      quant_scheme:
        quant_type: w8a8h1_sefp
        ops: {}
    only_first_block: false
```

### 26B-A4B Full Export

26B-A4B keeps the same public model/config entry as E4B and 31B. Do not use
`Gemma4ForConditionalGeneration_with_mask` in new workflow YAML.

```yaml
quant:
  algorithm: autoround
  output_format: gptqmodel_hf
  artifact_format: gptqmodel_hf
  bits: 4
  group_size: 64
  calibration:
    dataset: wikitext
    split: train
    nsamples: 128
    seqlen: 2048
  runtime:
    batch_size: 1
    trust_remote_code: true

export:
  model:
    chip_arch: XH2a
    model_type: Gemma4ForConditionalGeneration
    hf_model: null
    model_name: xh2_gemma4_26b_a4b_full_256_2k
    context_max_length: 2048
    prefill_chunk_length: 256
    use_cache: true
    num_logits_to_keep: 1
    quant_scheme:
      quant_type: w8a8h1_sefp
      ops: {}
    visual_config:
      export_mode: padded
      image_seq_length: 280
      max_patches: 2520
      patch_size: 16
      pooling_kernel_size: 3
      quant_scheme:
        quant_type: w8a8h1_sefp
        ops: {}
    only_first_block: false
```

Existing `_with_mask` configs are legacy compatibility artifacts only.

## Code Structure

Recommended implementation structure:

```text
xhmodel_merak/xh_llm/models/gemma4/
  workflow.py                  # Gemma4SeriesWorkflow and thin API helpers
  xh_gemma4_config.py           # Dense config extended for E4B/31B and padded visual protocol
  gemma4_processor.py           # Host-side padded ViT processor outputs
  gemma4_visual_model.py        # Padded visual wrapper/export protocol
  gemma4_llm_model.py           # Unified public entry; dense path plus internal 26B-A4B MoE adapter

xhmodel_merak/xh_llm/models/gemma4_moe/
  gemma4_moe_with_mask_model.py # Legacy/internal compatibility only; not a new public workflow entry
  gemma4_moe_visual_model.py    # Registers only a unique compatibility visual key, never overrides gemma4 visual
  xh_gemma4_moe_config.py       # Legacy compatibility config may preserve padded visual fields

configs_merak/workflows/xh2a/llm_models/gemma4_series/
  *.yaml                        # Topology-named workflow configs

examples_merak/llm/gemma4_series/
  gemma4_series_workflow.py      # Minimal workflow example
  gemma4_validation_matrix.py    # Runtime validation matrix
```

Registry invariant: `Gemma4ForConditionalGeneration` has one public master implementation in `xhmodel_merak.xh_llm.models.gemma4_series`; `Gemma4ForConditionalGeneration_visual` is not a separate visual-only Gemma4 Series export path.
Legacy `gemma4e`/`gemma4_moe` modules must use unique compatibility keys if they register submodels.

`Gemma4SeriesWorkflow` should mirror Qwen3.5 workflow patterns:

- `from_config()` classmethod returning the concrete workflow.
- `quant()` supporting explicit base validation, existing HF artifacts, and GPTQModel-HF output, with AutoRound mode1 available as an explicit alternative.
- `export()` validating Gemma4 model/config class names before running.
- `dump_golden()` accepting string, message list, `{"text": ...}`, and `{"image": ..., "text": ...}` inputs.
- helper APIs for config listing/help/templates.

## Example Script

The main example should stay intentionally small:

```python
from pathlib import Path
import shutil

from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

HF_MODEL_DIR = "/data01/datasets/gemma-4-31B-it"
CONFIG_PATH = "configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml"
QUANT_OUTPUT_DIR = "work_dirs/gemma4_31b_workflow_quant"
EXPORT_OUTPUT_DIR = "work_dirs/gemma4_31b_workflow_export"
DEVICE = "cuda"
CONFIG_OVERRIDES = None

for path in (QUANT_OUTPUT_DIR, EXPORT_OUTPUT_DIR):
    if Path(path).exists():
        shutil.rmtree(path)

workflow = Gemma4SeriesWorkflow.from_config(HF_MODEL_DIR, CONFIG_PATH)
quant_result = workflow.quant(QUANT_OUTPUT_DIR, DEVICE, config_overrides=CONFIG_OVERRIDES)
export_result = workflow.export(quant_result, EXPORT_OUTPUT_DIR, DEVICE, config_overrides=CONFIG_OVERRIDES)
workflow.dump_golden(export_result, DEVICE, {"text": "用中文介绍 Gemma4。"})
```

Base validation changes only `CONFIG_OVERRIDES`:

```python
CONFIG_OVERRIDES = {"quant": None}
```

Existing quantized HF validation changes only the quant override:

```python
CONFIG_OVERRIDES = {
    "quant": {
        "algorithm": "existing_hf",
        "artifact_format": "gptqmodel_hf",
        "source_algorithm": "autoround",
        "existing_hf_model_dir": "/path/to/existing/gemma4-gptqmodel-hf",
    }
}
```

## Migration Strategy

1. Add the workflow design doc and review it.
2. Add tests first for workflow dispatch, quant semantics, YAML validation, and visual protocol metadata.
3. Implement `Gemma4SeriesWorkflow` and thin package exports.
4. Add topology-named workflow YAMLs.
5. Update Dense `gemma4` to support E4B/31B differences behind one public model family.
6. Update Dense and MoE visual processors/wrappers to fully obey the padded ViT contract.
7. Add minimal example and validation matrix scripts.
8. Run static tests, then complete runtime validation for E4B, 31B, and 26B-A4B.
9. Keep old scripts until the validation matrix passes and downstream users confirm the new API.

## Verification Plan

Static/unit verification:

- `Gemma4SeriesWorkflow.from_config()` returns the concrete workflow.
- `AutoLLMWorkflow.from_config()` dispatches to `Gemma4SeriesWorkflow` for Gemma4 YAMLs.
- Default YAML quant is non-null and uses GPTQModel recipe semantics; AutoRound mode1 is an explicit alternative template/override.
- `config_overrides={"quant": None}` is the only accepted skipped-quant path.
- `quant.algorithm='existing_hf'` requires `existing_hf_model_dir` and returns a non-skipped `QuantResult`.
- Export model validation accepts Dense, MoE, and visual-only Gemma4 classes only.
- Visual config exposes padded protocol fields: `image_seq_length=280`, `max_patches=2520`, `patch_size=16`, `pooling_kernel_size=3`.
- Visual export config input names are exactly `pixel_values`, `pixel_position_ids`, `pool_indices`, `attention_mask`.
- `dump_golden()` builds correct messages for text and VLM inputs.

Runtime validation:

- E4B base validation: `quant: None` + full export + text golden + VLM golden.
- E4B default quant path: GPTQModel-HF + full export + golden.
- 31B base validation: `quant: None` + full export + text golden + VLM golden.
- 31B default quant path: GPTQModel-HF + full export + golden.
- 26B-A4B base validation: `quant: None` + full export + text golden + VLM golden.
- 26B-A4B default quant path: GPTQModel-HF + full export + golden.
- Existing quantized HF override for Dense and MoE when artifacts are available.

Regression verification:

- Existing workflow tests for MinerU2.5, Qwen2-VL, and Qwen3 legacy still pass.
- Existing Gemma4 debug scripts are not deleted or silently repointed during the first patch.
- `docs/gemma4_vit_padded_input_design_20260616.md` remains unchanged as the source of truth.

## Risks

- E4B currently has separate `gemma4e` code; consolidating into `gemma4_series` may expose optional attributes that 31B has but E4B lacks. The implementation should default missing attributes in config/model setup rather than branching in the public API.
- MoE has a separate with-mask path and historically separate visual export. The workflow may need internal special handling for vision export and meta merge, but that must not leak into API parameters.
- AutoRound/GPTQModel availability is environment-dependent. The validation matrix should preflight optional modules and paths before launching long jobs.
- The padded ViT protocol changes visual graph inputs and host preprocessing. All generation/golden paths must be migrated together to avoid token/feature count mismatches.

## Review Focus

Please review these decisions before implementation:

1. `Gemma4SeriesWorkflow` should live under `xhmodel_merak.xh_llm.models.gemma4_series.workflow` and become the stable Gemma4 family API.
2. E4B/31B Dense should unify on `gemma4_series`; `gemma4e` should not remain a public route for this workflow.
3. Default YAML quant should be GPTQModel-HF; AutoRound mode1 is selected explicitly via the AutoRound template/override, and skipped quant is only via explicit `config_overrides={"quant": None}`.
4. All Dense/MoE visual exports must obey the padded ViT input contract exactly.
5. YAML filenames should describe topology (`full`, `visual_only_448`) and not quantization precision/source.
