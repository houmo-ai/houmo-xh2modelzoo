# MiniCPM5-15B-A2.5B

This example supports the remote-code `MiniCPM5MoEForCausalLM` checkpoint with
the standard xh2 text-LLM prefill/decode KV-cache ABI.

```bash
cd /path/to/xh2modelzoo
export PYTHONPATH=$PWD
python examples_merak/llm/minicpm5/minicpm5_workflow.py \
  --model-dir $PATH/MiniCPM5-15B-A2.5B-0518-job_366159_step_12000_hybrid_thinking \
  --config-path configs_merak/workflows/xh2a/llm_models/minicpm5/15b_a2_5b/minicpm5_15b_a2_5b_xh2a_w8a8.yaml \
  --export-output-dir work_dirs/minicpm5_15b_a2_5b_export \
  --overwrite
```

Use `--dump-golden --prompt '...'` after a successful export to run the real
HMONNX generation path.

For GPTQModel native AutoRound W4A8, use
`configs_merak/workflows/xh2a/llm_models/minicpm5/15b_a2_5b/minicpm5_15b_a2_5b_xh2a_w4a8_autoround.yaml`.
Replace the portable `path-to-minicpm5-calibration-jsonl` placeholder with the
calibration JSONL path before quantization. The W8A8 floating-point export is
provided by `minicpm5_15b_a2_5b_xh2a_w8a8.yaml`.

The W4A8 GPTQModel checkpoint uses mixed-precision MoE quantization: routed
experts use W4 weights, while attention, shared experts, and dense MLP use W8
weights. The export keeps `lm_head` at W8A8, and GPTQModel uses group size 64
for the quantized linear layers. As in other MoE models, this profile is
expressed in `quant.moe`; `export.model.quant_scheme` is the separate HMONNX
target scheme.

### Calibration JSONL

The mixed calibration file contains one `{"text": "..."}` record per line.
The repository script combines WikiText-2 training text with rendered MMLU
multiple-choice text, then interleaves tokenized segments. The default recipe
uses 128 randomly selected MMLU records and writes eight 256-token segments.

Generate it with:

```bash
python examples_merak/llm/minicpm5/build_calibration.py \
  --model-dir $PATH/MiniCPM5-15B-A2.5B-0518-job_366159_step_12000_hybrid_thinking \
  --output path-to-minicpm5-calibration-jsonl \
  --mmlu-samples 128 \
  --segment-count 8 \
  --segment-length 256 \
  --seed 42
```

Replace `path-to-minicpm5-calibration-jsonl` in the AutoRound YAML with the
generated file. A GPTQModel-provided calibration JSONL can also be used when
it contains a `text` field in each record.

## Generation demos

MiniCPM5-15B-A2.5B is a text-only causal language model (MoE); it is not an
audio, vision, or omni-modal checkpoint. The checkpoint registers
`MiniCPM5MoEForCausalLM`, and its remote-code `forward()` accepts text LLM
inputs such as `input_ids`, `attention_mask`, `position_ids`, and KV cache. It
does not accept image, video, or audio inputs.

The native Transformers and exported HMONNX demos follow the same top-level
layout as `minicpm_v_4_6`:

```bash
python examples_merak/llm/minicpm5/float_demo.py \
  --model-dir $PATH/MiniCPM5-15B-A2.5B-0518-job_366159_step_12000_hybrid_thinking \
  --device cuda:0 --dtype fp16

python examples_merak/llm/minicpm5/hmonnx_demo.py \
  --config work_dirs/<export>/hmquant_*/golden_meta_info.json \
  --device cuda:0
```
