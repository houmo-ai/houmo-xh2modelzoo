# PI0.5 DROID and LIBERO Deployment Workflows

This directory contains the reproducible PI0.5 workflows used to download
checkpoints, export XH2A HMONNX graphs, validate conversion accuracy, run
software-HMONNX inference, and evaluate closed-loop LIBERO task success. It
separates three contracts that must not be mixed:

1. the public ModelScope DROID checkpoint with the official H15 action horizon;
2. the customer DROID checkpoint with the deployed H50 action horizon; and
3. the public LeRobot LIBERO checkpoint with H50 closed-loop simulation.

Run every command from the repository root. Generated assets stay under
`work_dirs`; the workflows do not require `/tmp` for persistent output.

## Delivery Snapshot

| Track | Validation protocol | Result | What it proves |
| --- | --- | --- | --- |
| Official DROID H15 | Public ModelScope checkpoint, fixed sample 0, fixed flow noise | cosine `0.999705`, MAE `0.008121` | One-sample public-checkpoint conversion smoke test |
| Customer DROID H50 | 100 fixed DROID inputs and one fixed noise file | mean cosine `0.994272`, mean MAE `0.017014`, CUDA Graph `6/6` sessions | HMONNX software parity against the LeRobot reference |
| LIBERO H50 numerical | 10 initial observations from each of 10 tasks, fixed noise | postprocessed mean cosine `0.999426`, mean MAE `0.003602` | HMONNX conversion parity on simulator observations |
| LIBERO H50 closed loop | 10 tasks, 20 episodes per task, matching process partitions | FP `187/200`, HMONNX `185/200`, delta `-1.0 pp` | End-to-end simulator task success |

The complete LIBERO comparison video is stored at:

`$MODEL_ZOO2_BASE_URL/model_zoo2/pi05/libero/libero_all_tasks_all_persisted_fp_vs_hmonnx_native_rmsnorm.mp4`

The video contains every rollout persisted by the final matched-partition
evaluation: 100 FP videos and 100 HMONNX videos paired across all ten tasks.
The score covers 200 episodes per runtime; the evaluator persisted videos for
episodes 0 through 9 of each task.

## Supported Contracts

| Field | Official DROID | Customer DROID | LIBERO LeRobot |
| --- | --- | --- | --- |
| Workflow variant | `droid-openpi-h15` | `droid-customer-h50` | `libero-lerobot-h50` |
| Checkpoint source | ModelScope `lerobot/pi05_droid` | Customer runtime bundle | ModelScope `lerobot/pi05_libero_finetuned` |
| Selected cameras | `[0, 1]` | `[0, 1]` | `[0, 1]` |
| State dimension | `8` | `8` | `8` |
| Action shape | `[15, 8]` | `[50, 8]` | `[50, 7]` |
| Static prefix length | `712` | `712` | `712` |
| KV cache capacity | `1024` | `1024` | `1024` |
| Quantization | `w8a8h1_sefp` | `w8a8h1_sefp` | `w8a8h1_sefp` |
| Target | XH2A HMONNX | XH2A HMONNX | XH2A HMONNX |

## Entry Points

| Script | Purpose |
| --- | --- |
| `pi05_modelscope_droid.py` | Download, verify, and smoke-test the public DROID checkpoint |
| `pi05_workflow.py` | Export one PI0.5 workflow YAML through Merak |
| `pi05_droid_fae_pipeline.py` | Run DROID preflight, export, validation, and delivery-manifest checks |
| `pi05_droid_hmonnx_validation.py` | Compare DROID HMONNX action chunks with the LeRobot reference |
| `pi05_droid_standard_eval.py` | Stream real samples from the standard DROID dataset |
| `pi05_action_comparison_report.py` | Build the customer DROID dashboard, CSV, and JSON report |
| `pi05_libero_fp_eval.py` | Run the official LeRobot LIBERO evaluator with a local tokenizer |
| `pi05_libero_hmonnx_validation.py` | Run fixed-noise LIBERO conversion validation |
| `pi05_libero_hmonnx_eval.py` | Replace only `predict_action_chunk` with HMONNX for closed-loop evaluation |
| `pi05_libero_comparison_report.py` | Aggregate matched FP/HMONNX partitions into accuracy reports |
| `pi05_libero_video_report.py` | Build the complete persisted-rollout MP4 and file-level manifest |

## 1. Environment

The validated environment uses Python 3.12, PyTorch 2.8, LeRobot 0.5.1,
TorchCodec 0.7.0, and FFmpeg with `libx264` and `drawtext` support.

```bash
conda create -n pi05 python=3.12
conda activate pi05

pip install -v -e . --no-build-isolation
pip install lerobot==0.5.1 modelscope torchcodec==0.7.0 matplotlib
pip install "lerobot[libero]==0.5.1" cmake==3.31.6

ffmpeg -version
ffprobe -version
```

TorchCodec 0.7.0 is required for the PyTorch 2.8 environment used here. Newer
TorchCodec releases target newer PyTorch versions and do not decode the DROID
AV1 videos in this environment. CMake 3.31 avoids simulator-dependency build
failures seen with CMake 4.x.

Define repository-relative asset locations once:

```bash
export PI05_DROID_ROOT=weights/pi05_result_check_droid
export PI05_MODELSCOPE_DIR="$PI05_DROID_ROOT/models/modelscope_lerobot_pi05_droid"
export PI05_CUSTOMER_DIR="$PI05_DROID_ROOT/models/pi05_droid_openpi_to_lerobot"
export PI05_LIBERO_DIR=weights/pi05_libero_finetuned
export PALIGEMMA_CONFIG_DIR=/data01/datasets/paligemma-3b-pt-224

export PI05_CUSTOMER_EXPORT_DIR=work_dirs/pi05_droid_customer_h50_compact_maskadd2_XH2a
export PI05_LIBERO_EXPORT_DIR=work_dirs/pi05_libero_lerobot_h50_compact_maskadd2_XH2a
export PI05_LIBERO_RUN_ROOT="$PI05_LIBERO_EXPORT_DIR/validation_results/closed_loop_native_rmsnorm_20260810"
export MODEL_ZOO2_BASE_URL=<artifactory-base-url>
```

The checkpoint metadata names the gated Hugging Face tokenizer
`google/paligemma-3b-pt-224`. Download the equivalent files from ModelScope and
always pass the local directory explicitly:

```bash
modelscope download \
  --model AI-ModelScope/paligemma-3b-pt-224 \
  added_tokens.json special_tokens_map.json tokenizer_config.json \
  tokenizer.json tokenizer.model config.json preprocessor_config.json \
  generation_config.json \
  --local_dir "$PALIGEMMA_CONFIG_DIR"
```

For LIBERO, keep runtime configuration under `work_dirs`:

```bash
export HOME="$PWD/work_dirs/pi05_libero_runtime/home"
export LIBERO_CONFIG_PATH="$PWD/work_dirs/pi05_libero_runtime/config"
```

## 2. Official DROID H15

### 2.1 Checkpoint Identity

The public checkpoint is
[`lerobot/pi05_droid`](https://www.modelscope.cn/models/lerobot/pi05_droid).
ModelScope currently serves the snapshot through `master`; the recorded source
revision is `0fd787830f979307d217c9523eef28231e9d7e3e`. File hashes are the
reproducibility anchor.

| File | SHA-256 | MD5 |
| --- | --- | --- |
| `model.safetensors` | `684c8b2033dcfacee9c6a83f38810ecc77df2c81aeb171bb87319da01fafcfaa` | `b653cdd10007defc65adea3b892ce741` |
| `config.json` | `bb8d260332ad7206ab80accb739da60309d2b8ea23120363c30a4dcd0c9a91e5` | `2025b6860656591d730b44a2bc0712c4` |
| `policy_preprocessor.json` | `be2cd1acc33229d42be0640cb55b5d5a4bb3e8d8e27ddfcdcf3f99f1fe3beaef` | `c42d985ffd0787681550c77c008b70af` |
| `policy_postprocessor.json` | `142c8b622aeac5c14631e057279df058d3005f616d909cfe30a04524de0b0891` | `c83b1d525df6c97f142f4c996b84d08e` |

The public package does not contain the DROID normalizer state files required
by the reference processor. A paired DROID runtime bundle supplies processor
and feature-schema state only; it does not modify the public model weights.

### 2.2 Download and Verify

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_modelscope_droid.py \
  --download \
  --verify \
  --model-dir "$PI05_MODELSCOPE_DIR"
```

### 2.3 LeRobot Smoke Test

This command runs public-checkpoint inference on fixed DROID `sample_0000` and
fixed flow noise. It records the action shape, action hash, latency, and model
identity.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_modelscope_droid.py \
  --demo \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --runtime-config-dir "$PI05_CUSTOMER_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --inputs-dir "$PI05_DROID_ROOT/inputs" \
  --noise-file "$PI05_DROID_ROOT/noise/noise_droid_100_h50_a32_seed20260707.npz" \
  --device cuda:0 \
  --output "$PI05_MODELSCOPE_DIR/demo_sample_0000.json"
```

Add `--compare-model-dir "$PI05_CUSTOMER_DIR"` to compare the public weights
with the paired runtime bundle on the same input. This is a numerical smoke
test, not a robot task-success evaluation.

### 2.4 Export HMONNX

The FAE wrapper performs preflight checks, calls the Merak workflow, validates
all eight HMONNX artifacts, verifies attention-mask graph structure, and writes
an auditable delivery manifest.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_fae_pipeline.py \
  --variant droid-openpi-h15 \
  --project-root "$PI05_DROID_ROOT" \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir work_dirs/pi05_modelscope_droid_h15_XH2a \
  --quant-output-dir work_dirs/pi05_modelscope_droid_quant \
  --device cuda:0 \
  --skip-validation \
  --hash-model \
  --overwrite
```

`--skip-validation` is required for the standalone public package because its
normalizer state is absent. It does not skip export, golden-data generation,
HMONNX artifact validation, or graph-structure checks. The manifest is written
to `work_dirs/pi05_modelscope_droid_h15_XH2a/fae_delivery_manifest.json`.

### 2.5 HMONNX Software Smoke Test

Use the public model weights with the paired processor bundle:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_hmonnx_validation.py \
  --project-root "$PI05_DROID_ROOT" \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --runtime-config-dir "$PI05_CUSTOMER_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir work_dirs/pi05_modelscope_droid_h15_XH2a \
  --results-dir work_dirs/pi05_modelscope_droid_h15_XH2a/modelscope_demo \
  --hmonnx-output modelscope_h15_hmonnx_droid.jsonl \
  --reference-output modelscope_h15_lerobot_droid.jsonl \
  --summary-output modelscope_h15_summary.json \
  --start-sample-id 0 \
  --num-samples 1 \
  --hmonnx-runtime v2-cuda-graph \
  --device cuda:0
```

The reproduced public-checkpoint sample has action shape `[15, 8]`, cosine
`0.999704897`, MAE `0.008120899`, RMSE `0.009934093`, and maximum absolute
error `0.026313096`. The result is stored at
`work_dirs/pi05_modelscope_droid_h15_XH2a/modelscope_demo/modelscope_h15_summary.json`.

### 2.6 Standard DROID Dataset Evaluation

`pi05_droid_standard_eval.py` streams source Parquet rows and AV1 frames from
[`lerobot/droid_1.0.1`](https://huggingface.co/datasets/lerobot/droid_1.0.1)
instead of downloading the roughly 412 GB dataset snapshot. The deterministic
selection uses distinct language tasks from independent episodes. Each sample
contains both real camera streams, the real 8-dimensional state, and a
continuous 15-step dataset action window at the episode midpoint.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_standard_eval.py \
  --num-samples 8 \
  --seed 445 \
  --model-dir "$PI05_MODELSCOPE_DIR" \
  --runtime-config-dir "$PI05_CUSTOMER_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir work_dirs/pi05_modelscope_droid_h15_XH2a \
  --output work_dirs/pi05_modelscope_droid_h15_XH2a/standard_droid_seed445_8.json \
  --device cuda:0
```

The standard dataset stores absolute joint targets. The report also converts
them to the OpenPI delta-action convention while preserving absolute gripper
commands. HMONNX-versus-LeRobot is the conversion acceptance gate; comparison
against a stochastic policy's dataset action is an offline diagnostic only.

## 3. Customer DROID H50

### 3.1 Model and Input Identity

The customer deployment uses the paired DROID runtime checkpoint and a 50-step
action horizon.

| Item | Value |
| --- | --- |
| Model | `weights/pi05_result_check_droid/models/pi05_droid_openpi_to_lerobot/model.safetensors` |
| Model SHA-256 | `acc741031fd59a6ad39ec30c11e916d65ae88ffaf39b1938e918d528cb25f26c` |
| Tensor inventory | 813 tensors, all stored as F32 |
| Inputs | sample IDs 0 through 99 from the fixed DROID input bundle |
| Noise | `noise_droid_100_h50_a32_seed20260707.npz` |
| Noise SHA-256 | `29d1b136a58a1ee2a16609cbc3b03ed977f773e4fcf574b61e72f6f5e21a1ae4` |
| Output action | `[50, 8]` |

The fixed-noise hash and per-sample `input_sha256` values are carried through
both JSONL files. The customer report refuses to compare records when sample
ID, dataset index, input hash, noise key, or noise hash differs.

### 3.2 End-to-End FAE Export and Validation

This is the preferred fresh-run command. It exports Vision, Gemma, Expert, and
Other components, generates golden data, validates graph structure, runs 100
LeRobot/HMONNX samples, and writes `fae_delivery_manifest.json`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_fae_pipeline.py \
  --variant droid-customer-h50 \
  --project-root "$PI05_DROID_ROOT" \
  --model-dir "$PI05_CUSTOMER_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --inputs-dir "$PI05_DROID_ROOT/inputs" \
  --noise-file "$PI05_DROID_ROOT/noise/noise_droid_100_h50_a32_seed20260707.npz" \
  --export-dir "$PI05_CUSTOMER_EXPORT_DIR" \
  --quant-output-dir work_dirs/pi05_droid_customer_quant \
  --results-dir "$PI05_CUSTOMER_EXPORT_DIR/validation_results" \
  --start-sample-id 0 \
  --num-samples 100 \
  --device cuda:0 \
  --hash-model \
  --overwrite
```

Use `--check-only` for preflight without inference, `--dry-run` to print the
resolved child commands, or `--skip-export --reuse-validation` to verify an
existing result set and regenerate the handoff manifest.

### 3.3 Native-RMSNorm CUDA Graph Validation

The final customer-facing run uses the V2 CUDA Graph runtime and writes a
separate immutable result directory:

```bash
export PI05_CUSTOMER_RESULTS="$PI05_CUSTOMER_EXPORT_DIR/validation_results/customer_delivery_native_rmsnorm_20260810"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_droid_hmonnx_validation.py \
  --project-root "$PI05_DROID_ROOT" \
  --model-dir "$PI05_CUSTOMER_DIR" \
  --runtime-config-dir "$PI05_CUSTOMER_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --inputs-dir "$PI05_DROID_ROOT/inputs" \
  --noise-file "$PI05_DROID_ROOT/noise/noise_droid_100_h50_a32_seed20260707.npz" \
  --export-dir "$PI05_CUSTOMER_EXPORT_DIR" \
  --results-dir "$PI05_CUSTOMER_RESULTS" \
  --hmonnx-output droid_hmonnx_100.jsonl \
  --reference-output droid_gt_100.jsonl \
  --summary-output droid_hmonnx_vs_gt_summary.json \
  --start-sample-id 0 \
  --num-samples 100 \
  --hmonnx-runtime v2-cuda-graph \
  --device cuda:0
```

Generate the customer dashboard and auditable per-sample table:

```bash
PYTHONPATH=$PWD conda run -n pi05 \
python examples_merak/vla/pi05/pi05_action_comparison_report.py \
  --reference "$PI05_CUSTOMER_RESULTS/droid_gt_100.jsonl" \
  --candidate "$PI05_CUSTOMER_RESULTS/droid_hmonnx_100.jsonl" \
  --output-prefix "$PI05_CUSTOMER_RESULTS/droid_hmonnx_vs_gt_dashboard" \
  --cosine-threshold 0.99 \
  --mae-threshold 0.02
```

### 3.4 Customer DROID Result

| Metric | Result |
| --- | ---: |
| Mean per-sample cosine | `0.994272` |
| Minimum per-sample cosine | `0.932169` |
| Flattened cosine | `0.993527` |
| Mean per-sample MAE | `0.017014` |
| Flattened RMSE | `0.032478` |
| Maximum absolute error | `0.377285` |
| Samples passing both per-sample thresholds | `68/100` |
| Captured CUDA Graph sessions | `6/6` |
| Steady HMONNX software latency | `489.97 ms/action chunk` |

The aggregate gate is mean cosine at least `0.99` and mean MAE at most `0.02`;
the run passes both. The `68/100` value applies both thresholds independently
to every sample and is therefore a stricter diagnostic, not the aggregate
acceptance decision. Latency is from the CUDA software-HMONNX runtime on this
host and must not be presented as XH2A NPU latency.

Customer artifacts are under
`$PI05_CUSTOMER_RESULTS`:

| Artifact | Contents |
| --- | --- |
| `droid_gt_100.jsonl` | LeRobot reference actions and provenance |
| `droid_hmonnx_100.jsonl` | HMONNX actions, provenance, latency, and runtime status |
| `droid_hmonnx_vs_gt_summary.json` | Validation contract and per-sample metrics |
| `droid_hmonnx_vs_gt_dashboard.json` | Aggregate, per-dimension, and per-sample report |
| `droid_hmonnx_vs_gt_dashboard.csv` | Flat per-sample metrics |
| `droid_hmonnx_vs_gt_dashboard.png` | Customer-facing dashboard |

### 3.5 CUDA Graph Root Cause and Fix

The Expert graph receives an activation with shape `[1, 50, 1024]` and a
dynamic AdaRMSNorm weight with shape `[1, 1, 1024]`. The original XH2A dispatch
accepted only a one-dimensional weight, so the broadcast-equivalent weight
fell through to an implementation that synchronized the stream during CUDA
Graph capture. The native fix canonicalizes singleton-leading weights to
`[1024]` and uses the XH2A Triton RMSNorm path. It does not use a software FP32
RMSNorm fallback.

SeFP first-use autotuning was a separate capture hazard. The runtime now warms
that path before capture. Vision, action input, action output, time MLP, Gemma,
and Expert all report `cuda_graph_captured=true`. An isolated replay test with
dynamic `[1, 1, 1024]` weights produced `max_abs=0.0` and exact tensor equality.

### 3.6 Historical Board HMM Boundary

The historical low-cosine screenshot belongs to a board-side HMM output, not
to the current software-HMONNX runtime:

| Comparison | Mean cosine | Minimum cosine | MAE |
| --- | ---: | ---: | ---: |
| Historical board HMM vs current LeRobot | `0.73384247` | `0.21465668` | `0.13512515` |
| Current HMONNX software vs matched LeRobot GT | `0.99427173` | `0.93216866` | `0.01701418` |

The historical HMM cannot be reproduced or attributed to a specific graph
stage from this checkout because the compiled `/outputs/xh2` artifact, build
manifest, compiler version, firmware/runtime version, and intermediate dumps
are unavailable. Do not use the current HMONNX result to claim that the old
board HMM issue is fixed; it proves that the exported graph and host software
runtime are accurate under the documented contract.

## 4. LIBERO LeRobot H50

### 4.1 Download and Verify the Checkpoint

The canonical checkpoint is
[`lerobot/pi05_libero_finetuned`](https://www.modelscope.cn/models/lerobot/pi05_libero_finetuned).
It uses mean/std normalization, a 50-step horizon, and 7-dimensional LIBERO
actions.

```bash
modelscope download \
  --model lerobot/pi05_libero_finetuned \
  --local_dir "$PI05_LIBERO_DIR"

sha256sum "$PI05_LIBERO_DIR/model.safetensors"
```

Expected model SHA-256:
`877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae`.

The wrappers change only the tokenizer path or the policy action-chunk
implementation. Model weights, processor state, environment processing,
seeding, action queue, and rollout logic remain the official LeRobot paths.

### 4.2 Export HMONNX

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_workflow.py \
  --variant libero-lerobot-h50 \
  --model-dir "$PI05_LIBERO_DIR" \
  --config-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-output-dir "$PI05_LIBERO_EXPORT_DIR" \
  --quant-output-dir work_dirs/pi05_libero_quant \
  --device cuda:0 \
  --dump-golden \
  --overwrite
```

The artifact contains Vision, Gemma, Expert, and Other components. All floating
graph inputs and outputs are FP16; the validator rejects BF16 tensors in any
HMONNX graph. The export also records dtype checks, attention-mask checks,
golden-data checks, graph hashes, and external-data sizes.

### 4.3 Fixed-Noise Numerical Validation

This validation takes ten initial simulator observations from each of the ten
`libero_object` tasks. Every HMONNX action chunk is compared with LeRobot using
the same observation and fixed flow noise.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_libero_hmonnx_validation.py \
  --model-dir "$PI05_LIBERO_DIR" \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir "$PI05_LIBERO_EXPORT_DIR" \
  --suite libero_object \
  --samples-per-task 10 \
  --seed 1000 \
  --noise-seed 20260810 \
  --hmonnx-runtime v2-cuda-graph \
  --device cuda:0
```

| Compared action | Mean cosine | Minimum cosine | Mean MAE | Mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Normalized HMONNX vs LeRobot | `0.999632` | `0.990900` | `0.012149` | `0.019090` |
| Postprocessed HMONNX vs LeRobot | `0.999426` | `0.973180` | `0.003602` | `0.007611` |

All eight graph dtype checks, four attention-mask checks, and eight HMONNX
golden-data checks passed. Detailed records are stored in
`$PI05_LIBERO_EXPORT_DIR/validation_results/libero_h50_fp_vs_hmonnx.jsonl`; the
aggregate report is
`$PI05_LIBERO_EXPORT_DIR/validation_results/libero_h50_summary.json`.

This is a 100-observation conversion test, not 100 successful robot episodes.
W8A8 parity is tolerance-based rather than bit-exact.

### 4.4 Closed-Loop FP Evaluation

The official LeRobot evaluator is invoked through a small tokenizer wrapper.
For a single partition, use:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_libero_fp_eval.py \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --policy.path="$PI05_LIBERO_DIR" \
  --policy.device=cuda \
  --policy.compile_model=false \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0,1]' \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --output_dir="$PI05_LIBERO_RUN_ROOT/fp_tasks_0_1_retry"
```

### 4.5 Closed-Loop HMONNX Evaluation

The HMONNX wrapper keeps the same LeRobot evaluator and replaces only
`predict_action_chunk` with the four-component software runtime:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=$PWD \
python examples_merak/vla/pi05/pi05_libero_hmonnx_eval.py \
  --tokenizer-dir "$PALIGEMMA_CONFIG_DIR" \
  --export-dir "$PI05_LIBERO_EXPORT_DIR" \
  --hmonnx-runtime v2-cuda-graph \
  --policy.path="$PI05_LIBERO_DIR" \
  --policy.device=cuda \
  --policy.compile_model=false \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[0,1]' \
  --eval.batch_size=1 \
  --eval.n_episodes=20 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --output_dir="$PI05_LIBERO_RUN_ROOT/hmonnx_tasks_0_1"
```

The final direct comparison uses the same process partition for both runtimes:

| Task IDs | FP output | HMONNX output |
| --- | --- | --- |
| `[0,1]` | `fp_tasks_0_1_retry` | `hmonnx_tasks_0_1` |
| `[2,3]` | `fp_tasks_2_3_retry` | `hmonnx_tasks_2_3` |
| `[4,5]` | `fp_tasks_4_5_retry` | `hmonnx_tasks_4_5` |
| `[6]` | `fp_task_6` | `hmonnx_task_6` |
| `[7]` | `fp_task_7` | `hmonnx_task_7` |
| `[8]` | `fp_task_8` | `hmonnx_task_8` |
| `[9]` | `fp_task_9` | `hmonnx_task_9` |

Use seed 1000 and 20 episodes per task for every partition. The policy samples
flow noise from the evaluator process RNG stream, so changing process
partitions changes the closed-loop trajectory even when the top-level seed is
unchanged. The earlier sequential FP run scored `177/200`; it remains a valid
standalone baseline but is excluded from this direct comparison.

### 4.6 Closed-Loop Result

| Task | Instruction | FP | HMONNX | Delta |
| ---: | --- | ---: | ---: | ---: |
| 0 | Pick up the alphabet soup and place it in the basket | 19/20 | 19/20 | 0 |
| 1 | Pick up the cream cheese and place it in the basket | 20/20 | 20/20 | 0 |
| 2 | Pick up the salad dressing and place it in the basket | 19/20 | 19/20 | 0 |
| 3 | Pick up the BBQ sauce and place it in the basket | 19/20 | 19/20 | 0 |
| 4 | Pick up the ketchup and place it in the basket | 20/20 | 20/20 | 0 |
| 5 | Pick up the tomato sauce and place it in the basket | 17/20 | 15/20 | -2 |
| 6 | Pick up the butter and place it in the basket | 20/20 | 20/20 | 0 |
| 7 | Pick up the milk and place it in the basket | 18/20 | 16/20 | -2 |
| 8 | Pick up the chocolate pudding and place it in the basket | 18/20 | 20/20 | +2 |
| 9 | Pick up the orange juice and place it in the basket | 17/20 | 17/20 | 0 |
| **Overall** | **10 tasks, 200 episodes/runtime** | **187/200 (93.5%)** | **185/200 (92.5%)** | **-1.0 pp** |

HMONNX retains `98.93%` of the FP success rate. Seven tasks have identical
success counts, task 8 is two episodes higher, and tasks 5 and 7 are two
episodes lower. The Wilson 95% intervals are `89.20%-96.16%` for FP and
`88.00%-95.40%` for HMONNX.

### 4.7 Accuracy Report Generation

Aggregate the exact matched partitions:

```bash
python examples_merak/vla/pi05/pi05_libero_comparison_report.py \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_tasks_0_1_retry/eval_info.json" \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_tasks_2_3_retry/eval_info.json" \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_tasks_4_5_retry/eval_info.json" \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_task_6/eval_info.json" \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_task_7/eval_info.json" \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_task_8/eval_info.json" \
  --fp-eval-info "$PI05_LIBERO_RUN_ROOT/fp_task_9/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_tasks_0_1/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_tasks_2_3/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_tasks_4_5/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_task_6/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_task_7/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_task_8/eval_info.json" \
  --hmonnx-eval-info "$PI05_LIBERO_RUN_ROOT/hmonnx_task_9/eval_info.json" \
  --output-prefix "$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_fp_vs_hmonnx_accuracy" \
  --episodes-per-task 20 \
  --start-seed 1000
```

This command fails on duplicate tasks, missing episodes, changed process
partitions, or non-boolean success metadata. It writes PNG, CSV, JSON, and
Markdown reports. Video provenance is intentionally separate so the accuracy
report does not hardcode a single sample.

### 4.8 Complete FP vs HMONNX Video

Build the complete persisted-rollout video from the accuracy report:

```bash
python examples_merak/vla/pi05/pi05_libero_video_report.py \
  --report-json "$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_fp_vs_hmonnx_accuracy.json" \
  --output-video "$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_all_tasks_all_persisted_fp_vs_hmonnx_native_rmsnorm.mp4" \
  --output-manifest "$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_all_tasks_all_persisted_fp_vs_hmonnx_native_rmsnorm.json" \
  --overwrite
```

The generator verifies every source `eval_info.json` hash from the accuracy
report, checks all source videos, pairs matching task/episode IDs, records every
source SHA-256, clones only the shorter rollout's final frame for alignment,
and validates the final dimensions, nominal frame rate, and frame count.

| Video property | Value |
| --- | --- |
| Coverage | 10 tasks, episodes 0-9, 100 paired rollouts |
| Source files | 100 FP + 100 HMONNX = 200 videos |
| Resolution | `1024x576` |
| Codec | H.264, `yuv420p` |
| Nominal frame rate | `80 FPS` |
| Frames | `17,276` |
| Duration | `215.971973 s` |
| Size | `26,475,452 bytes` |
| SHA-256 | `ead747ddaafa974d80f7bcc6fb8ffbdffe9ccd9bb025ab5c0c7b74cad8683206` |

The evaluator metadata contains ten persisted videos per task although the
score covers twenty episodes per task. Therefore the video includes all 100
persisted pairs, not a cherry-picked example, while the score remains based on
all 200 episodes per runtime.

Delivery artifacts:

| Artifact | Location |
| --- | --- |
| Complete MP4 | `$MODEL_ZOO2_BASE_URL/model_zoo2/pi05/libero/libero_all_tasks_all_persisted_fp_vs_hmonnx_native_rmsnorm.mp4` |
| Video manifest | `$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_all_tasks_all_persisted_fp_vs_hmonnx_native_rmsnorm.json` |
| Verification contact sheet | `$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_all_tasks_all_persisted_fp_vs_hmonnx_native_rmsnorm_contact_sheet.png` |
| Accuracy dashboard | `$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_fp_vs_hmonnx_accuracy.png` |
| Accuracy JSON | `$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_fp_vs_hmonnx_accuracy.json` |
| Accuracy CSV | `$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_fp_vs_hmonnx_accuracy.csv` |
| Accuracy summary | `$PI05_LIBERO_RUN_ROOT/customer_delivery/libero_fp_vs_hmonnx_accuracy.md` |

## 5. Interpretation and Deployment Boundaries

These workflows deliberately report different evidence types:

| Evidence | Scope | Not implied |
| --- | --- | --- |
| Fixed-input DROID parity | Exported graph versus LeRobot on identical input and noise | Physical-robot success or NPU performance |
| Fixed-observation LIBERO parity | Exported graph versus LeRobot on identical simulator observations and noise | Closed-loop robustness |
| LIBERO closed-loop success | End-to-end simulator policy behavior | Real-robot safety or success |
| HMONNX software runtime | Host CUDA execution of exported HMONNX | Compiled HMM correctness or XH2A board latency |

Final HMM compilation and board execution require the separate XH2 compiler,
firmware, and runtime environment. A board result is reproducible only when the
compiled artifact, build manifest, compiler version, firmware/runtime version,
host preprocessing contract, and intermediate dumps are preserved together.

## 6. Troubleshooting

**Public DROID normalizer files are missing.** Use the paired runtime bundle
through `--runtime-config-dir`; do not copy or rewrite public model weights.

**The PaliGemma tokenizer attempts a gated Hugging Face download.** Pass
`--tokenizer-dir "$PALIGEMMA_CONFIG_DIR"` to every wrapper.

**DROID AV1 decoding fails.** Verify PyTorch 2.8 and TorchCodec 0.7.0 are used
together.

**LIBERO simulator initialization writes outside the workspace.** Set `HOME`
and `LIBERO_CONFIG_PATH` to the documented `work_dirs/pi05_libero_runtime`
locations before launching the evaluator.

**FP and HMONNX scores change after repartitioning.** Match process partitions,
seed, task IDs, and episode count exactly. The policy-noise RNG stream is local
to each evaluator process.

**CUDA Graph capture fails in Expert RMSNorm.** Confirm the native XH2A
RMSNorm canonicalization for singleton-leading dynamic weights and perform
SeFP warmup outside capture. Do not silently replace the native operator with a
software FP32 fallback.

**The complete video cannot be generated.** Confirm `ffmpeg`, `ffprobe`,
`libx264`, `drawtext`, and the DejaVu Sans font are installed. The video script
will identify the first missing or mismatched source artifact.
