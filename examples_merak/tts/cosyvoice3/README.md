# CosyVoice3 (merak)

CosyVoice3-0.5B TTS model migrated into the `xhmodel_merak` architecture. The
model is exported as a set of HMONNX graphs through the unified
`AutoWorkflow` interface.

## Model structure

CosyVoice3 is a multi-submodel TTS pipeline. The export produces one top-level
`export_meta_info.json` containing all component metadata. The LLM component
additionally has its own `meta_info.json` (read by `XHQwen2HMONNXModel`):

| component | source | HMONNX graphs | quant type |
|-----------|--------|---------------|------------|
| llm (Qwen2 0.5B) | llm.pt + HF config | prefill + decode | w8a16_sefp |
| llm_decoder | onnx | 1 (lm-head projection) | w8a16h1_sefp |
| speech_tokenizer_v3 | onnx (mask-instrumented) | 1 | w8a16_sefp |
| campplus | onnx | 1 (speaker encoder) | w8a16_sefp |
| flow_decoder | onnx (DiT) | 1 | w8a16_sefp |
| hift | onnx (HiFT vocoder) | 1 | w8a16_sefp |
| spk_embed_affine_layer | onnx | 1 | w8a16h1_sefp |
| pre_lookahead_layer | onnx | 1 | w8a16h1_sefp |

## Prerequisites

- conda env `<env_name>` with `torch>=2.8`, `xhquant`, `onnx`, `onnxsim`,
  `onnxruntime`, `transformers`.
- For runtime inference (eval / streaming demo), additionally install:
  `torchaudio`, `librosa`, `soundfile`, `openai-whisper`, `wetext`, `inflect`.
- A downloaded Fun-CosyVoice3-0.5B pretrained package. The export expects:
  - `<model_dir>/llm.pt` — CosyVoice3 LLM checkpoint (must contain `speech_embedding.weight`).
  - `<model_dir>/flow.pt` — Flow model checkpoint (must contain `input_embedding.weight`).
  - `<model_dir>/CosyVoice-BlankEN/` (or `<model_dir>`) — Qwen2 HF config.
  - `<model_dir>/onnx/` — source onnx files for the onnx-based submodels:
    `llm_decoder.onnx`, `speech_tokenizer_v3.onnx`, `campplus.onnx`,
    `flow_decoder_estimator_fp32.onnx`, `hift.onnx`,
    `spk_embed_affine_layer.onnx`, `pre_lookahead_layer.onnx`.

Source onnx paths and the llm checkpoint can be overridden in the workflow YAML
(`export.onnx_dir`, `export.llm_checkpoint`, `export.source_onnx`).

## Export

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD conda run -n <env_name> \
  python examples_merak/tts/cosyvoice3/cosyvoice3_workflow.py \
    --model-dir <model_dir> \
    --device cuda:0 \
    --overwrite
```

Export with golden generation:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD conda run -n <env_name> \
  python examples_merak/tts/cosyvoice3/cosyvoice3_workflow.py \
    --model-dir <model_dir> \
    --device cuda:0 \
    --dump-golden \
    --overwrite
```

Override a single component or quant type:

```bash
... --components llm,campplus
... --quant-type w8a16_sefp
```

> **Note:** Exporting only a subset of components (e.g., `--components llm,campplus`)
> produces an **incomplete** export directory.  The full runtime demo
> (`hmonnx_demo.py` / `cosyvoice3_eval.py` / streaming) requires **all 8
> components** and will fail with a partial export.  Subset export is intended
> for single-component debugging or incremental re-export only.

Override the source onnx directory or llm checkpoint (the resolver also falls
back to `<model_dir>` automatically when `<model_dir>/onnx` is absent):

```bash
... --onnx-dir <onnx_dir> --llm-checkpoint <llm_pt_path>
```

## Runtime smoke demo

After export, run every exported HMONNX graph on dummy inputs to verify the
graphs are loadable and runnable:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD conda run -n <env_name> \
  python examples_merak/tts/cosyvoice3/hmonnx_demo.py \
    --export-dir work_dirs/CosyVoice3-0.5B_XH2a \
    --device cuda:0
```

## Runtime inference (eval)

Run batch zero-shot TTS inference using the exported HMONNX graphs. The eval
script reads evaluation data (text, prompt_text, prompt_wav.scp) from a dataset
directory matching the CV3-Eval layout:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD conda run -n <env_name> \
  python examples_merak/tts/cosyvoice3/cosyvoice3_eval.py \
    --work-dir <export_dir> \
    --data-dir <eval_data_dir> \
    --device cuda:0 \
    --max-samples 10
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--work-dir` | Export output directory (the one containing `export_meta_info.json`). |
| `--data-dir` | Eval dataset directory with `text`, `prompt_text`, and `prompt_wav.scp` files. |
| `--device` | Inference device, e.g. `cuda:0`. |
| `--max-samples` | Number of utterances to process (omit to process all). |
| `--output-dir` | Output directory for generated WAV files (default: `<work-dir>/eval_out`). |
| `--seed` | Random seed (default: 1986). |

The script loads all HMONNX graphs referenced by `export_meta_info.json`, builds
a `CosyVoice3HMONNXInference` model, and runs zero-shot TTS for each utterance.
Generated audio (24kHz WAV) and a JSON report (`eval_report.json`) are saved to
the output directory.

## Streaming demo

Run token-level streaming TTS: audio is generated in 25-token chunks and
each chunk is written to disk as soon as it is ready (first-packet latency
< full-sentence generation time). Ported from `examples/audio/Cosyvoice3/cv3_stream.py`.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> PYTHONPATH=$PWD conda run -n <env_name> \
  python examples_merak/tts/cosyvoice3/hmonnx_streaming_demo.py \
    --work-dir <export_dir> \
    --text "<synthesis_text>" \
    --prompt-wav <prompt_wav_path> \
    --prompt-text "<prompt_text>" \
    --device cuda:0 \
    --v3-align --fade-ms 5
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--work-dir` | Export output directory (containing `export_meta_info.json`). |
| `--text` | Text to synthesize. |
| `--prompt-wav` | Prompt wav file (16kHz). |
| `--prompt-text` | Prompt text transcript (optional, default: ""). |
| `--device` | Inference device, e.g. `cuda:0`. |
| `--output` | Output wav path (default: `<work-dir>/streaming_demo.wav`). |
| `--seed` | Random seed (default: 1024). |
| `--token-hop-len` | Tokens per chunk (default: 25). |
| `--pre-lookahead-len` | Pre-lookahead tokens for flow decoder (default: 3). |
| `--v3-align` | Cut hift tail 3840 samples on intermediate chunks (V3 official alignment). |
| `--fade-ms` | Linear fade-in/out per chunk boundary in ms (default: 5). |

Output: per-chunk wav files in `<work-dir>/chunks/` plus a merged final wav.
The script reports first-packet latency and per-chunk wall-clock time.

## Output layout

```
work_dirs/CosyVoice3-0.5B_XH2a/
  cosyvoice3_0_5b.yaml          # dumped effective workflow config
  export_meta_info.json         # top-level entrypoint (component metadata is inline)
  # golden paths recorded in export_meta_info.json (after --dump-golden)
  LLM/
    token_embedding.pt
    meta_info.json              # LLM-specific: wrap_cfg, kv_cache_shape, num_hidden_layers, ...
    Prefill/<...>_prefill.onnx
    Decoder/<...>_decode.onnx
  LLMDecoder/   { onnx/, hmonnx/ }
  SpeechTokenizerV3/ ...
  Campplus/ ...
  FlowDecoder/ ...
  HiFT/ ...
  SpkEmbedAffineLayer/ ...
  PreLookaheadLayer/ ...
```

Non-LLM components do **not** have their own `meta_info.json`; their metadata
(hmonnx/onnx paths, quant_type, input shapes) is stored directly in
`export_meta_info.json` under the respective `components.<name>` key.

## Notes

- The export follows the legacy `examples/audio/Cosyvoice3/*_export_hmonnx.py`
  flow. The onnx transforms (shape fix, simplify, hift scatter/resize rewrite,
  speech tokenizer mask insertion) are ported into
  `xhmodel_merak/xh_other_model/models/cosyvoice3/_export_utils.py` so the
  migrated package does not depend on `./examples`, `./xh_model_zoo`, or
  `./configs` at runtime.
- The migrated package registers `XHCosyVoice3LLM` (main export model, bound to
  `CosyVoice3Workflow` via `WORKFLOW_CLS`) and `XHCosyVoice3LLMHMONNX` (the
  HMONNX inference model used by runtime demos/eval).
- CosyVoice3 is sourced from
  https://github.com/FunAudioLLM/CosyVoice under Apache License 2.0.

