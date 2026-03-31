# Spark XH (IPTForCausalLM) Examples

Model: Spark MoE with MLA (Multi-head Latent Attention), registered as `IPTForCausalLM`.

## Debug Scripts

### Native HF Generate

```bash
python examples_merak/llm/spark_xh/debug_scripts/native_spark_xh_generate.py \
    --model-dir ./data/models/ipt_30b \
    --prompt "你多大了？用中文回答。"
```

### XH Framework Generate

```bash
python examples_merak/llm/spark_xh/debug_scripts/spark_xh_xh_generate.py \
    --config configs_merak/xh2a/llm_models/spark_xh/30b/spark_xh_30b_xh2a_2k.py \
    --eval-type wrap
```

Supported `--eval-type` values: `wrap`, `prefill`, `decode`, `joint`, `joint_nostream`.

## HMONNX Export

### Using config file

```bash
python examples_merak/llm/spark_xh/spark_xh_xh_export_hmonnx.py \
    --config configs_merak/xh2a/llm_models/spark_xh/30b/spark_xh_30b_xh2a_2k.py \
    --force
```

### Using CLI arguments

```bash
python examples_merak/llm/spark_xh/spark_xh_xh_export_hmonnx.py \
    --model ./data/models/ipt_30b \
    --model-type IPTForCausalLM \
    --context-length 2048 \
    --prefill-chunk-length 256 \
    --quant-type w8a8_sefp
```

## HMONNX Inference

```bash
python examples_merak/llm/spark_xh/spark_xh_xh_hmonnx_generate.py \
    --config work_dirs/<exported_dir>/golden_meta_info.json \
    --prompt "你多大了？用中文回答。" \
    --max-new-tokens 10
```

Options: `--fast` (fast mode), `--golden` (save golden outputs), `--think` (enable thinking), `--auto-offload`.
