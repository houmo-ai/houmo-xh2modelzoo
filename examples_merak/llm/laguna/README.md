# Laguna-S-2.1 Merak Workflow

The model snapshot is expected at `/data01/datasets/Laguna-S-2.1`. By default, the workflow converts the Laguna remote-code model to the Merak graph, runs the repository's aligned XH2a PTQ during HMONNX export, and optionally generates golden data from the exported prefill/decode graphs.

Low-memory export is enabled by default. The workflow builds the non-MoE graph on the meta device, materializes its weights once, and then loads, quantizes, exports, and releases each sparse MoE block independently. The temporary block graphs are inlined into the final prefill/decode HMONNX files. Use `--no-low-memory` only when comparing against the legacy whole-model export path.

GPTQModel compatibility is retained as an optional path. The default Laguna configuration keeps `quant: null`, so the standard route remains HF BF16 to native XH2a PTQ. The full AutoRound configuration instead calls GPTQModel's maintained `gptqmodel.recipes.laguna_autoround` recipe, which runs the Laguna QuaRot + AutoRound backend and produces a routed-expert-W4/rest-W8/G64 HF checkpoint before HMONNX export.

The AutoRound route requires an importable GPTQModel source checkout that exposes `gptqmodel.recipes.laguna_autoround` and contains `third_party/auto-round/scripts_laguna/quantize_laguna.py`. The selected quantization interpreter must provide the GPTQModel and AutoRound dependencies and `transformers>=5.12.0`; set `GPTQMODEL_PYTHON` or `quant.runtime.python` when it differs from the interpreter running this workflow.

XH2a export uses `w4a8h0_ssfp`; `lm_head` remains `w8a8h1_sefp`. Calibration is performed by the standard `export_hmonnx()` path with the model's prefill/decode dummy inputs. The tokenizer is loaded with `trust_remote_code=True` and Transformers' `fix_mistral_regex=True` compatibility fix.

Run the full workflow from the repository root:

```bash
python examples_merak/llm/laguna/laguna_workflow.py \
  --model-dir /data01/datasets/Laguna-S-2.1 \
  --config-path configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/laguna_s_2_1_xh2a_w4a8.yaml \
  --export-output-dir work_dirs/laguna_s_2_1_export \
  --device cuda:1 \
  --dump-golden \
  --overwrite
```

To quantize the BF16 checkpoint with Laguna's validated QuaRot + AutoRound profile and then export it, use separate quantization and export directories:

```bash
GPTQMODEL_PYTHON=/path/to/gptqmodel/python \
python examples_merak/llm/laguna/laguna_workflow.py \
  --model-dir /data01/datasets/Laguna-S-2.1 \
  --config-path configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/laguna_s_2_1_full_autoround_expert_w4_rest_w8_g64_xh2a_w4a8.yaml \
  --quant-output-dir work_dirs/laguna_s_2_1_autoround_quantized \
  --export-output-dir work_dirs/laguna_s_2_1_autoround_export \
  --device cuda:0 \
  --overwrite
```

The full configuration fixes the validated profile to symmetric rest-W8, routed-expert-W4, group size 64, and Hadamard rotation. Its default AutoRound device map is `0,1,2,3`; adjust `quant.runtime.device_map` and `quant.runtime.rotation_device` in a local workflow override when using a different GPU layout. The quantization directory contains the AutoRound HF checkpoint, the subprocess log, and `laguna_autoround_recipe_provenance.json`.

For the checked-in AutoRound expert-W4 checkpoint, select the matching model directory and workflow explicitly:

```bash
python examples_merak/llm/laguna/laguna_workflow.py \
  --model-dir /data01/home/xuzk/datasets/Laguna-S-2.1-autoround-expert-w4-rest-w8-g64 \
  --config-path configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/laguna_s_2_1_autoround_expert_w4_rest_w8_g64_xh2a_w4a8.yaml \
  --export-output-dir work_dirs/laguna_s_2_1_autoround_export \
  --device cuda:1
```

For minimal dense-layer debugging, add `--only-first-block --context-length 512 --prefill-chunk-length 64`. To smoke-test the first real fused-MoE layer as well, use `--max-layers 2`. Release artifacts must be generated without a layer limit.

The export directory contains the resolved workflow YAML, `hmquant_*` release directory, prefill/decode HMONNX files, `golden_meta_info.json`, copied HF configuration/tokenizer files, quantized embedding data, and golden step data produced by `--dump-golden`. Each verification run should use a distinct `--export-output-dir`; `--overwrite` intentionally removes that directory before rebuilding it.
