# Laguna-S-2.1 Merak Workflow

The model snapshot is expected at `/data01/datasets/Laguna-S-2.1`. By default, the workflow loads the official floating-point BF16 checkpoint directly, converts the Laguna remote-code model to the Merak graph, runs the repository's aligned XH2a PTQ during HMONNX export, and optionally generates golden data from the exported prefill/decode graphs.

GPTQModel compatibility is retained as an optional path. Set the workflow `quant` block to `algorithm: gptqmodel` and provide either `quant.calibration.texts` or a local `quant.calibration.jsonl` dataset to produce a GPTQModel-compatible HF checkpoint before HMONNX export. The checked-in Laguna configuration intentionally keeps `quant: null`, so the default and currently validated route remains HF BF16 to native XH2a PTQ.

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

For minimal dense-layer debugging, add `--only-first-block --context-length 512 --prefill-chunk-length 64`. To smoke-test the first real fused-MoE layer as well, use `--max-layers 2`. Release artifacts must be generated without a layer limit.

The export directory contains the resolved workflow YAML, `hmquant_*` release directory, prefill/decode HMONNX files, `golden_meta_info.json`, copied HF configuration/tokenizer files, quantized embedding data, and golden step data produced by `--dump-golden`. `--overwrite` removes the export directory before rebuilding it.
