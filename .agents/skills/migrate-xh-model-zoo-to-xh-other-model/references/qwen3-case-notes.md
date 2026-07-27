# Qwen3 Migration Notes

These notes capture lessons from migrating Qwen3-ASR and Qwen3-TTS into `xhmodel_merak/xh_other_model`.

Read this file only for Qwen3-ASR/Qwen3-TTS migrations or when another target exhibits the same explicitly described failure mode. Treat every model name, component list, graph layout, config field, metadata choice, and workaround below as a reference case rather than a universal migration requirement. Derive the actual requirements from the target model's legacy behavior and exported artifacts.

## Qwen3-ASR

- The HMONNX multi-graph scheduling is implemented in the demo scripts. The HMONNX runtime does not automatically schedule encoder, prefill, decode, and cache movement across graphs.
- Keep `export.model.type` fixed for the main model so auto workflow binding can resolve the model class and workflow.
- For prefill/decode export, ensure `prefill_decode.quant_type` or equivalent YAML precision actually reaches the `MODELS.build()` config and appears in both prefill and decode filenames.
- Put non-`MODELS.build()` export parameters under descriptive `export.*` fields, for example sequence/audio length budgets and component-specific precision.
- `dump_golden()` should read `export_result.work_dir`, open `export_meta_info.json`, and generate golden from exported HMONNX artifacts. `export()` should not mention golden generation.
- If runtime operator limitations appear, prefer the smallest model-code change that preserves semantics. Document any difference from old export behavior.

## Qwen3-TTS

- There are multiple variants, for example Base, CustomVoice, and VoiceDesign. Each variant should have its own workflow YAML.
- Default export can include the extra stateful decoder artifact required by streaming. Running streaming should not require changing unrelated export config.
- Keep a top-level `export_meta_info.json`. For Qwen3-TTS, keep submodel `meta.json` files because the migrated demos and eval use component-local metadata; this is a model-specific compatibility choice, not a universal migration requirement.
- Migrate all native demos, HMONNX demos, streaming demos, eval, analysis, and README commands to `examples_merak`.
- Do not dynamically load old helper scripts from `examples/audio/qwen3_tts`. Move wrappers and export helpers into the model package, such as `_export_utils.py`.
- Cover golden for all submodels:
  - Talker prefill and decode.
  - CodePredictor prefill and decode.
  - TextProjection.
  - SpeechTokenizer.
  - Base frontend graphs such as `speech_tokenizer.encode` and `speaker_encoder`.
  - StatefulDecoder when exported.
- HMONNX golden can expose metadata-only issues not seen in normal inference. In the Qwen3-TTS stateful decoder case, a converted HMONNX graph contained duplicate `value_info` for `expand`; normal inference worked, but golden saving failed. A temporary runtime-local tensor-info patch is acceptable if it does not alter the exported HMONNX or model computation. The long-term fix belongs in the converter/transform cleanup.

## Output Compatibility

- Preserve old artifact naming and layout unless the migration plan explicitly changes it.
- If layout changes are required, update every demo/eval/analysis script and README together.
- Avoid symlink or temporary runner dependencies as final migration results.

## Environment Lessons

- Confirm the requested conda environment before long exports.
- Use the user-requested GPU via `CUDA_VISIBLE_DEVICES`, and pass `cuda:0` inside the process when the visible GPU is remapped.
- Large exports should be rerun from a clean output directory only when the user asked for overwrite or the workflow supports it.
