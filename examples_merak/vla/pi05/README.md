# PI0.5 DROID XH2A Workflow

This directory provides a reproducible public PI0.5 DROID flow: download the
ModelScope checkpoint, verify its identity, run a LeRobot smoke test, export
XH2A HMONNX graphs, and compare one HMONNX sample with LeRobot. Run all
commands from the repository root.

## Scope

The documented public checkpoint is
[`lerobot/pi05_droid`](https://www.modelscope.cn/models/lerobot/pi05_droid).
The workflow exports the official H15 contract with two selected cameras.

| Field | Value |
| --- | --- |
| Workflow variant | `droid-openpi-h15` |
| Selected image indices | `[0, 1]` |
| Static prefix length | `712` |
| Action horizon | `15` |
| Cache capacity | `1024` |
| Target | XH2A HMONNX |

The public mirror is a standalone checkpoint. Its package does not include
the DROID normalizer state files required by the reference processor, so the
demo explicitly receives a paired DROID runtime bundle. This adapts only the
processor and feature schema; it does not modify the downloaded weights.

## Environment

```bash
conda create -n pi05 python=3.12
conda activate pi05

pip install -v -e . --no-build-isolation
pip install lerobot==0.5.1 modelscope
```

Set repository-relative asset locations before running the examples:

```bash
export PI05_PROJECT_ROOT=weights/pi05_result_check_droid
export PI05_MODELSCOPE_DIR="$PI05_PROJECT_ROOT/models/modelscope_lerobot_pi05_droid"
export PI05_RUNTIME_DIR="$PI05_PROJECT_ROOT/models/pi05_droid_openpi_to_lerobot"
export PALIGEMMA_CONFIG_DIR=<paligemma-config-dir>
```

The PaliGemma tokenizer and configuration files can be downloaded from
ModelScope:

```bash
modelscope download \
  --model AI-ModelScope/paligemma-3b-pt-224 \
  added_tokens.json special_tokens_map.json tokenizer_config.json \
  tokenizer.json tokenizer.model config.json preprocessor_config.json \
  generation_config.json \
  --local_dir "$PALIGEMMA_CONFIG_DIR"
```

## Download And Verify

Download the public snapshot and verify every runtime file in one pass. The
script checks the ModelScope SHA-256 and a local MD5 identity for each file.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_modelscope_droid.py \
  --download \
  --verify \
  --model-dir "$PI05_MODELSCOPE_DIR"
```

| File | SHA-256 | MD5 |
| --- | --- | --- |
| `model.safetensors` | `684c8b2033dcfacee9c6a83f38810ecc77df2c81aeb171bb87319da01fafcfaa` | `b653cdd10007defc65adea3b892ce741` |
| `config.json` | `bb8d260332ad7206ab80accb739da60309d2b8ea23120363c30a4dcd0c9a91e5` | `2025b6860656591d730b44a2bc0712c4` |
| `policy_preprocessor.json` | `be2cd1acc33229d42be0640cb55b5d5a4bb3e8d8e27ddfcdcf3f99f1fe3beaef` | `c42d985ffd0787681550c77c008b70af` |
| `policy_postprocessor.json` | `142c8b622aeac5c14631e057279df058d3005f616d909cfe30a04524de0b0891` | `c83b1d525df6c97f142f4c996b84d08e` |

ModelScope currently exposes this snapshot through `master`. The source file
revision recorded during verification is
`0fd787830f979307d217c9523eef28231e9d7e3e`; use the file hashes as the
reproducibility anchor.

## LeRobot Smoke Test

The following command uses fixed DROID `sample_0000` and fixed flow noise. It
writes the action shape, action SHA-256, latency, and optional comparison data
to JSON.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_modelscope_droid.py \
  --demo \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --runtime-config-dir "$PI05_RUNTIME_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --device cuda:0 \
  --output "$PI05_MODELSCOPE_DIR/demo_sample_0000.json"
```

Add `--compare-model-dir "$PI05_RUNTIME_DIR"` to compare the public checkpoint
with the paired runtime bundle on the same input. This is a numerical check,
not a robot task-success evaluation.

## Export HMONNX

Export the four PI0.5 components, generate golden data, and verify the eight
HMONNX artifacts and attention-mask graph structure:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_fae_pipeline.py \
  --variant droid-openpi-h15 \
  --project-root "$PI05_PROJECT_ROOT" \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir work_dirs/pi05_modelscope_droid_h15_XH2a \
  --quant-output-dir work_dirs/pi05_modelscope_droid_quant \
  --device cuda:0 \
  --skip-validation \
  --hash-model \
  --overwrite
```

The export manifest is written to
`work_dirs/pi05_modelscope_droid_h15_XH2a/fae_delivery_manifest.json`.
`--skip-validation` is required when the model directory does not carry the
paired normalizer state files; it does not skip export, golden generation, or
the HMONNX graph checks.

## HMONNX Smoke Test

Run one HMONNX software sample against the same public checkpoint's LeRobot
reference. The separate runtime bundle supplies the paired processor state.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_hmonnx_validation.py \
  --project-root "$PI05_PROJECT_ROOT" \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --runtime-config-dir "$PI05_RUNTIME_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir work_dirs/pi05_modelscope_droid_h15_XH2a \
  --results-dir work_dirs/pi05_modelscope_droid_h15_XH2a/modelscope_demo \
  --hmonnx-output modelscope_h15_hmonnx_droid.jsonl \
  --reference-output modelscope_h15_lerobot_droid.jsonl \
  --summary-output modelscope_h15_summary.json \
  --start-sample-id 0 \
  --num-samples 1 \
  --device cuda:0
```

The generated summary contains the action shape, cosine, MAE, RMSE, max-abs,
and software runtime latency. It verifies conversion consistency only; final
HMM compilation and board validation require the separate XH2 compiler
environment.
