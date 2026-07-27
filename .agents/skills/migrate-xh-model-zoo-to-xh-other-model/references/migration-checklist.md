# Migration Checklist

Use this checklist when porting a model from legacy `xh_model_zoo` or old `examples` scripts into `xhmodel_merak/xh_other_model`.

## Contents

- [Source and Applicability](#source-and-applicability)
- [Inventory](#inventory)
- [Layout and Registration](#layout-and-registration)
- [YAML](#yaml)
- [Workflow](#workflow)
- [Metadata](#metadata)
- [Examples, README, and Demos](#examples-readme-and-demos)
- [Dependency Removal](#dependency-removal)
- [Validation Commands](#validation-commands)
- [Review Risks](#review-risks)

## Source and Applicability

- Read `xhmodel_merak/xh_other_model/MIGRATION_GUIDE.md` before implementation and treat it as the source of truth.
- Treat named models, fixed component lists, graph layouts, config fields, and workarounds as reference cases only.
- Derive required submodels, artifacts, scheduling, metadata, and compatibility handling from the target model's legacy behavior and actual export.
- Do not copy a model-specific workaround into another migration unless the same failure mode is confirmed.

## Inventory

- List old source packages and scripts:
  - `xh_model_zoo/...`
  - `examples/<domain>/<model>/...`
  - old py config files and their inheritance chain
  - old README commands
  - demos, streaming demos, eval, and analysis scripts
- List submodels and artifacts:
  - LLM prefill graph
  - LLM decode graph
  - encoder/frontend/tokenizer graph
  - projection graph
  - stateful or streaming graph
  - embeddings or extra `.pt` files
- Build the expected artifact and graph list from the target model. Do not copy a fixed list from another model's case notes.
- List old runtime helper dependencies. Move needed code into the model package.
- If the old model depends on `xh_model_zoo` shared code that is not already migrated into `xh_other_model`, prefer implementing the required subset under `xhmodel_merak/xh_other_model/models/<model_name>/`. Avoid adding new public shared code unless multiple migrated models genuinely need it and the user agrees.
- Keep repository-internal implementation imports within `xhmodel_merak.xh_other_model`. Do not depend on sibling or parent packages such as `xhmodel_merak.xh_llm`, `xhmodel_merak.workflows`, or `xhmodel_merak.utils`.

## Layout and Registration

- Put model implementation code under `xhmodel_merak/xh_other_model/models/<model_name>/`.
- Put workflow YAML under `configs_merak/workflows/xh2a/other_models/<model_name>/`.
- Put runnable examples, demos, eval, analysis, and README files under `examples_merak/...`.
- Make the concrete workflow inherit `BaseOtherModelWorkflow`.
- Register the primary model with `register_other_model` and set `WORKFLOW_CLS`.
- Make `export.model.type` exactly match the `register_other_model` decorator string.
- Confirm `scan_model_types.py` discovers the registered model type.
- Do not add a manually maintained model-type alias map when scanner-based registration already works.

## YAML

- `quant`: all quantization options; use `null` when there is no independent quantization stage.
- `export`: all export options.
- `export.model`: the full config that will be passed to `MODELS.build()`.
- `export.model.type`: fixed location for auto binding.
- Additional submodel configs: use `export.<descriptive_name>` or `export.modelA`, `export.modelB`.
- `export.target_device`: chip architecture, usually `XH2a`.
- Per-submodel quant type: every exported submodel must have a YAML-controlled precision.
- Flatten old Python config inheritance completely.
- Confirm each precision field reaches every affected submodel and graph export and appears in the HMONNX filename.
- Put non-`MODELS.build()` options under clear, descriptive `export.*` fields.
- Allow command-line overrides only for paths that already exist in YAML.

## Workflow

- Do not write derived values back into `workflow_config.data`.
- Keep helper functions small and only when reused or clarifying.
- Avoid abstractions that are only called once.
- Keep `quant`, `export`, and `dump_golden` responsibilities separate.
- Return a skipped `QuantResult` when the model has no independent quantization stage.
- Dump the effective workflow config into the export output directory using `workflow_config.name`; do not introduce an `effective` filename convention.
- Preserve output directory conventions unless the user asks to change them.
- Include target device and quant type in exported HMONNX file names.
- For models that need no user input, set `input_messages=None` only in the subclass override; do not change the base protocol.

## Metadata

- Keep top-level `export_meta_info.json`.
- Keep submodel `meta.json` only when the old implementation already had submodel meta files. If the old export did not create submodel meta files, do not add them as a migration requirement.
- Make paths relative where existing code expects relative paths; keep absolute paths only for external model/data roots.
- Make `dump_golden()` generate golden directories for every graph in the actual export result.
- Do not introduce `golden_meta_info.json` or another global golden index unless the legacy implementation or model explicitly requires it.
- Clear or overwrite each graph's golden directory before regenerating it.
- Recreate legacy prompts, preprocessing, masks, cache initialization, shapes, dtypes, and zero/random dummy-input construction where applicable.

## Examples, README, and Demos

- Use the top-level `xhmodel_merak.workflows.AutoWorkflow` entrypoint in workflow example scripts, not model-family-specific Auto classes such as `AutoOtherModelWorkflow` or `AutoLLMWorkflow`.
- Require `--model-dir` and do not provide a machine-local model path as its default.
- Expose `--config-path` with a default under `configs_merak/workflows/xh2a/other_models/<model_name>/`.
- Support `--dump-golden` after export.
- Make `--overwrite` remove only the export directory through `_remove_output_dir_if_needed()`.
- Use `work_dirs/...` as the default output convention.
- Use placeholders such as `<env_name>`, `<gpu_id>`, `<model_dir>`, and `<audio_file>` in README commands.
- Document formal environment and third-party installation plus script startup. Do not include machine-local conda names, absolute paths, or editable dependency paths.
- Remove hardcoded legacy artifact directories, model names, and metadata paths from demos.
- Make demos, eval, and analysis consume the new `export_meta_info.json` and artifact layout.
- For multi-graph, cached, or streaming models, conditionally check graph scheduling, runtime state conversion, per-graph shapes/dtypes/initialization, and extra stateful or auxiliary graphs based on the actual exported artifacts.

## Dependency Removal

Do not import helpers or runnable code from another `examples_merak` example directory. Put reusable behavior in the corresponding model package or an explicitly agreed shared module.

After migration, run a search in the migrated package for:

```bash
rg -n "xh_model_zoo|(^|[\"'/])examples([\"'/]|$)|(^|[\"'/])configs([\"'/]|$)|spec_from_file_location|importlib.util" xhmodel_merak/xh_other_model/models/<model_name>

rg -n --pcre2 "xhmodel_merak\.(?!xh_other_model(?:\.|$))|from\s+xhmodel_merak\s+import" \
  xhmodel_merak/xh_other_model/models/<model_name> --glob "*.py"
```

Any match must be removed unless it is harmless documentation text outside runtime code.
Also resolve relative imports and confirm that every repository-internal target remains under `xhmodel_merak.xh_other_model`.

- Do not modify source code in repository-external dependencies to make the migration pass.
- Document formal third-party installation instead of relying on machine-local editable installs.
- If an editable installation points to the wrong source, fix the environment rather than including external source changes in the migration.

## Validation Commands

Adjust environment, model path, GPU, variant, and script paths to the task:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD conda run -n <env> \
  python examples_merak/<domain>/<model>/<workflow_script>.py \
  --model-dir <model_dir> --device cuda:0 --overwrite

CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD conda run -n <env> \
  python -m py_compile xhmodel_merak/xh_other_model/models/<model_name>/*.py

CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD conda run -n <env> \
  python -m py_compile examples_merak/<domain>/<model_name>/*.py
```

Then run:

- The workflow example with `--dump-golden`.
- HMONNX demos.
- streaming demos if a stateful graph is exported.
- eval with a small sample.
- analysis scripts.
- artifact inspection: `export_meta_info.json`, old-compatible submodel metadata if present, and generated golden directories.
- comparison of old and new HMONNX filenames, graph inputs and outputs, metadata, and demo outputs under the same config when feasible.
- documentation of the reason for any unavoidable difference.

## Review Risks

- Hidden old dependencies from dynamic imports.
- Cross-example imports under `examples_merak`.
- Imports from `xh_other_model` into sibling or parent packages, especially `xhmodel_merak.xh_llm`.
- YAML missing inherited values from old py configs.
- `export.model.type` not fixed, breaking auto binding.
- `quant_type` present in YAML but not passed into all submodel exports.
- HMONNX names not reflecting quant/device.
- Golden generation only covering the main graph.
- Demos hardcoding old output directories or old model names.
- Runtime fixes that change exported graph semantics unnecessarily.
- Model-specific component lists, scheduling rules, metadata conventions, or workarounds applied as universal requirements.
- Traceable-module or global-registry cleanup applied without first confirming the conflict.
- Repository-external source edits or machine-local editable installs hidden in the migration environment.
