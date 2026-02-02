# FunASR Seaco Paraformer Export (Runtime Patch)

This folder exports a Seaco Paraformer model with an in-memory override of
FunASR `export_meta` so you do not need to edit `site-packages`.

## Protect your FunASR installation
- Use the conda env `xhquant` (or another isolated env).
- Do not edit `.../site-packages/funasr`; the patch is loaded from this repo at runtime.
- To recover a clean install, reinstall FunASR in the env.

## Setup (recommended)
```bash
conda activate xhquant
# Install FunASR and dependencies (e.g., torch) in this env.
pip install -U funasr
```

## Quick start
```bash
python examples/audio/funasr/funasr_export.py \
  --input examples/audio/funasr/asr_example_hotword.wav \
  --device cpu
```

## Verify export in `xhquant`
```bash
conda activate xhquant
python examples/audio/funasr/funasr_export.py --device cpu
```
Expected:
- `generate result (before rotation)` and `generate result (after rotation)` are printed
- `export result` is printed; it includes the export output location

## Inputs (encoder/decoder)
We keep a set of prepared inputs in:
- `examples/audio/funasr/inputs/`
  - `speech.npy`
  - `enc.npy`
  - `enc_mask.npy`
  - `pre_acoustic_embeds.npy` (padded to max length)
  - `pre_token_mask.npy` (padded to max length)

You can regenerate them from encoder + predictor ONNX:
```bash
python examples/audio/funasr/prepare_inputs.py \
  --inputs-dir examples/audio/funasr/inputs \
  --max-token-len 100
```

## End-to-end pipeline (export -> quant -> infer)
1) Export ONNX (writes a manifest used by later steps):
```bash
conda activate xhquant
python examples/audio/funasr/funasr_export.py --device cpu
```
If you want a specific export directory:
```bash
python examples/audio/funasr/funasr_export.py --device cpu --export-dir /path/to/export_dir
```

2) Prepare inputs (pad to fixed length 100 by default):
```bash
python examples/audio/funasr/prepare_inputs.py \
  --inputs-dir examples/audio/funasr/inputs \
  --max-token-len 100
```

3) Quantize encoder/decoder:
```bash
python examples/audio/funasr/encoder_quant.py \
  --inputs-dir examples/audio/funasr/inputs

python examples/audio/funasr/decoder_quant.py \
  --inputs-dir examples/audio/funasr/inputs \
  --max-token-len 100
```
Quantized outputs are written to:
- `work_dirs/<onnx_stem>/hmonnx/<onnx_stem>_<quant_type>_XH2a.onnx`

4) Run baseline ONNX inference:
```bash
python examples/audio/funasr/test_infer.py \
  --inputs-dir examples/audio/funasr/inputs
```

5) Run quantized inference + compare:
```bash
python examples/audio/funasr/test_hmonnx_infer.py \
  --inputs-dir examples/audio/funasr/inputs \
  --compare
```
If you used non-default quant types, add:
```bash
python examples/audio/funasr/test_hmonnx_infer.py \
  --encoder-quant-type w8a8_sefp \
  --decoder-quant-type w8a8
```

## Manifest & path overrides
`funasr_export.py` writes a manifest to:
- `examples/audio/funasr/exports/latest.json`

All scripts default to this manifest if it exists. You can override with explicit paths:
```bash
python examples/audio/funasr/prepare_inputs.py \
  --encoder /path/to/encoder.onnx \
  --predictor /path/to/predictor.onnx

python examples/audio/funasr/test_infer.py \
  --encoder /path/to/encoder.onnx \
  --predictor /path/to/predictor.onnx \
  --decoder /path/to/decoder.onnx
```
Quantized models can also be passed explicitly:
```bash
python examples/audio/funasr/test_hmonnx_infer.py \
  --encoder-hmonnx /path/to/encoder_hmonnx.onnx \
  --decoder-hmonnx /path/to/decoder_hmonnx.onnx
```

## Export-meta patch behavior
- Default patch path: `examples/audio/funasr/export_meta_seacoparaformer.py`
- Place your custom `export_meta_seacoparaformer.py` next to this script if you need an override.
- If the override file is missing, the script warns and falls back to upstream `export_meta`.
- To enforce the override:
```bash
python examples/audio/funasr/funasr_export.py --require-export-meta
```
- To skip patching:
```bash
python examples/audio/funasr/funasr_export.py --no-patch-export-meta
```

## Customize
- Use a different model:
```bash
python examples/audio/funasr/funasr_export.py --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```
- Adjust conda env checks:
```bash
python examples/audio/funasr/funasr_export.py --expected-conda-env myenv
python examples/audio/funasr/funasr_export.py --no-conda-check
```
- Disable generate sanity runs:
```bash
python examples/audio/funasr/funasr_export.py --skip-generate
```
- Skip rotation:
```bash
python examples/audio/funasr/funasr_export.py --skip-rotation
```
- Quantized export / export type:
```bash
python examples/audio/funasr/funasr_export.py --quantize --export-type onnx
```
- Tune hmonnx vs onnx input dtypes (defaults: hmonnx=fp16, onnx=fp32):
```bash
python examples/audio/funasr/test_hmonnx_infer.py \
  --hmonnx-float-dtype fp16 \
  --onnx-float-dtype fp32
```

## Notes
- The override is applied in memory and restored before exit.
- If FunASR changes its export API, update the override accordingly.
 - Decoder models may be fixed-length. Use `--max-token-len` to pad/trim inputs (default 100).
