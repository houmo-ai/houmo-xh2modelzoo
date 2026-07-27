---
name: migrate-xh-model-zoo-to-xh-other-model
description: Migrate legacy models from xh_model_zoo, xh2modelzoo examples, or old Python export scripts into xhmodel_merak/xh_other_model. Use when Codex is asked to port a model or multi-submodel workflow; convert old py configs to workflow YAML; move builder/base/common model code; implement quant/export/dump_golden workflows; migrate demos, eval, analysis, and README files; or remove dependencies on xh_model_zoo and examples.
---

# Migrate XH Model Zoo To XH Other Model

## Core Rules

Before changing migration code in the xh2modelzoo repository, read `xhmodel_merak/xh_other_model/MIGRATION_GUIDE.md` and treat it as the source of truth. Use this skill to execute that guide, not to override it.

Treat named models, component lists, graph layouts, config fields, and workarounds as reference cases only. Derive requirements from the target model's legacy implementation and actual exported graph list. Do not generalize a model-specific subgraph, scheduling rule, metadata convention, or workaround to every migration.

Preserve behavior first. In the same configuration, migrated HMONNX should match the old export as closely as possible. Any intentional deviation must be minimal, documented, and justified by a runtime/export bug.

Keep `workflow.py` simple. Do not introduce broad abstractions for one-off code paths. Prefer explicit, readable steps over indirection.

Do not mutate `workflow_config.data`. Build derived dictionaries locally and dump the effective workflow config from the workflow object.

Do not leave runtime dependencies on `./examples`, `./xh_model_zoo`, or `./configs`. Move required helper code into the model package, usually a private module such as `_export_utils.py`.

Avoid imports between different migrated example directories under `examples_merak`. Move reusable runtime or export helpers into the corresponding model package or an explicitly agreed shared module instead of coupling examples to one another.

Keep implementation code under `xhmodel_merak/xh_other_model` within that package boundary. Do not import code from sibling or parent paths such as `xhmodel_merak.xh_llm`, `xhmodel_merak.workflows`, or `xhmodel_merak.utils`. The top-level `xhmodel_merak.workflows.AutoWorkflow` entrypoint is for migrated examples and docs only; do not create a reverse dependency on it from `xh_other_model`.

Use the top-level `xhmodel_merak.workflows.AutoWorkflow` interface in migrated examples and docs. Do not recommend model-family-specific Auto entrypoints such as `AutoOtherModelWorkflow` or `AutoLLMWorkflow` for new migrated workflow scripts unless there is a specific debugging reason.

`xhmodel_merak/xh_other_model` only contains part of the old `xh_model_zoo` shared code, such as `base_llm_model.py`. If a migrated model depends on old shared code that is not already present in `xh_other_model`, prefer implementing the needed behavior inside `xhmodel_merak/xh_other_model/models/<model_name>/` to avoid coupling. Do not add new code to the public `xh_other_model` shared directory unless the behavior is genuinely shared by multiple migrated models and the user has agreed to that boundary.

Do not modify source code in repository-external dependencies to make a migration pass. Document formal third-party installation steps and fix an incorrect editable installation at the environment level. Apply traceable-module or global-registry cleanup only after confirming the conflict, and keep the workaround local to the model adapter.

## Migration Workflow

1. Inspect the old implementation:
   - Find old model classes, registration decorators, builder usage, py configs, export scripts, golden paths, demos, eval scripts, analysis scripts, and README commands.
   - Identify all submodels and HMONNX graphs. For LLM-style graphs, distinguish prefill/decode and KV-cache conventions.
   - Identify old helper scripts that are dynamically imported or shared through `examples/`.

2. Create or update the model package under `xhmodel_merak/xh_other_model/models/<model_name>/`:
   - Move model implementations, wrappers, hmonnx inference adapters, and helper code needed at runtime or export time.
   - When old `xh_model_zoo` shared helpers are missing from `xh_other_model`, implement the required subset locally in this model package whenever practical instead of extending public shared modules.
   - Keep all repository-internal imports within `xhmodel_merak.xh_other_model`; do not reuse implementations from sibling or parent packages such as `xhmodel_merak.xh_llm`.
   - Keep old public semantics and config keys where practical.
   - Make the workflow inherit `BaseOtherModelWorkflow`.
   - Register the primary model with `register_other_model`, set `WORKFLOW_CLS`, and make `export.model.type` exactly match the decorator string.
   - Confirm `scan_model_types.py` discovers the registered type.
   - Avoid hand-maintained type alias maps when auto scanning/registration already supports `export.model.type`.

3. Convert old Python configs to workflow YAML:
   - Put workflow YAML under `configs_merak/workflows/xh2a/other_models/<model_name>/`.
   - Put quantization configuration under `quant`.
   - Set `quant: null` when there is no independent quantization stage.
   - Put export configuration under `export`.
   - Fix the primary model config at `export.model.type`.
   - Preserve the old `MODELS.build()` config as the whole `export.model` object. Do not split it into unrelated YAML fields.
   - For multi-submodel workflows, choose one main submodel as `export.model`; use names such as `export.modelA`, `export.modelB`, or descriptive fields for additional submodels.
   - Fully materialize inherited Python config values. Do not silently drop values from base configs.
   - Put chip architecture in `export.target_device` unless the repository already has a stronger convention.
   - Expose every submodel quantization/export precision through YAML.
   - Allow command-line overrides only for paths that already exist in YAML.

4. Implement workflow methods:
   - `quant()` should only handle quantization or explicitly return a skipped result when quantization is not supported.
   - `export()` should export models only. It must not generate golden data.
   - `export()` must dump the workflow config into the output directory using `workflow_config.name`; do not introduce a separate `effective` filename convention.
   - Export output names must include quant precision and target device.
   - Keep a top-level `export_meta_info.json`. Preserve submodel `meta.json` files only when the old implementation already had them; do not introduce submodel `meta.json` as a new requirement.
   - Do not add or remove output directory levels unless the user asks for that layout change.

5. Implement `dump_golden()`:
   - Generate all golden data here, never in `quant()` or `export()`.
   - Use `export_result.work_dir` to find `export_meta_info.json` and any old-compatible submodel metadata that exists.
   - Cover every exported submodel and every HMONNX graph, including prefill/decode pairs, frontend graphs, tokenizer graphs, stateful decoders, and projection graphs.
   - Construct golden inputs in the same way as the legacy export or golden scripts whenever possible. Reuse tokenizer prompts, preprocessing, `prepare_inputs()` helpers, decode input construction, masks, cache initialization, and dtype conventions from the old flow. If the old flow used zero or random dummy tensors, recreate the same construction pattern; exact random values do not need to match unless the legacy flow required them.
   - Make golden generation repeatable. Clear or overwrite per-graph golden directories before regenerating.
   - For a model that needs no user input, give only the subclass `input_messages` parameter a default of `None`; do not change the base protocol.
   - Do not introduce `golden_meta_info.json` or another global golden index unless the legacy implementation or model explicitly requires it.
   - If HMONNX golden runtime has metadata-only issues, prefer a local, documented workaround that does not alter model computation. Fix the converter when the bug belongs there.

6. Migrate examples and docs:
   - Put migrated runnable examples under `examples_merak/...`.
   - Use `from xhmodel_merak.workflows import AutoWorkflow` in workflow example scripts.
   - Require `--model-dir` without a machine-local default. Expose `--config-path` with a default under `configs_merak/workflows/xh2a/other_models/<model_name>/`, and support `--dump-golden`.
   - Make `--overwrite` remove only the export directory through `_remove_output_dir_if_needed()`. Use `work_dirs/...` as the default output convention.
   - Update demos, streaming demos, eval, analysis, and README to consume the new `export_meta_info.json` and the migrated artifact layout while preserving old submodel metadata conventions.
   - Use placeholders such as `<env_name>`, `<gpu_id>`, and `<model_dir>` in README commands. Document installation and startup without machine-local conda names, absolute paths, or editable dependency paths.
   - Do not import helpers or runnable code from another `examples_merak` example directory. Move shared behavior into the model package or an explicitly agreed shared module.
   - Do not keep dynamic imports from old example scripts.
   - For multi-graph, cached, or streaming models, inspect the actual artifacts and conditionally validate graph scheduling, runtime state conversion, per-graph shapes/dtypes/initialization, and extra stateful or auxiliary graphs.

## Validation

Run validation in the intended environment and device requested by the user. For GPU-specific work, set `CUDA_VISIBLE_DEVICES` and confirm the conda environment.

Minimum validation:
   - `python -m py_compile` for changed Python modules.
   - Search the migrated model package for forbidden dependencies on `./examples`, `./xh_model_zoo`, `./configs`, `spec_from_file_location`, and dynamic helper loading.
   - Search `xhmodel_merak/xh_other_model` implementation code for imports outside the package, including `xhmodel_merak.xh_llm`, `xhmodel_merak.workflows`, and `xhmodel_merak.utils`. Resolve relative imports and confirm that their targets remain under `xhmodel_merak.xh_other_model`.
   - Inspect migrated example scripts for imports from other `examples_merak` example directories.
   - Export every requested model variant.
   - Run `dump_golden()` and confirm golden data is generated for all exported submodels and graphs.
   - Run all migrated HMONNX demos, streaming demos when applicable, small-sample eval, and analysis scripts.
   - Inspect artifact directories and compare HMONNX filenames, graph inputs and outputs, metadata, and demo outputs under the same config when feasible. Record the reason for any unavoidable difference.

Always read [migration-checklist.md](references/migration-checklist.md) before implementation and final validation. Read [qwen3-case-notes.md](references/qwen3-case-notes.md) only for Qwen3-ASR/Qwen3-TTS migrations or when the target exhibits the same explicitly described failure mode; never treat those cases as universal requirements.
