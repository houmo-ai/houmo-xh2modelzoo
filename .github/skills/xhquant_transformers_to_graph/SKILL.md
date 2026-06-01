---
name: xhquant_transformers_to_graph
description: Convert a Hugging Face transformers text model or multimodal model into an xhquant-callable and executable graph by following the wrapper, graph_forward, frontend, HMONNX, and runtime-contract patterns used by qwen3_legacy, qwen3_vl, and qwen3_5.
license: Apache-2.0
metadata:
  author: GitHub Copilot
  version: "1.0"
---

# xhquant Transformers To Graph

Use this skill when the goal is not just to adapt a HF model class, but to make it runnable as an xhquant graph with stable export and inference contracts.

This skill is distilled from these reference implementations:

- `xhmodel_merak/xh_llm/models/qwen3_legacy/`
- `xhmodel_merak/xh_llm/models/qwen3_vl/`
- `xhmodel_merak/xh_llm/models/qwen3_5/`

## What “conversion to xhquant graph” means

The conversion is complete only when all of the following exist:

- HF modules are replaced by traceable wrapper modules registered in `XHLLM_TRACEABLE_MODULES`
- the wrapped model exposes a graph-safe execution path, usually through `graph_forward`
- cache inputs and outputs are explicit and stable for prefill and decode
- the model can be lowered to xhquant frontend graph or ONNX frontend graph
- an HMONNX runtime wrapper can call the exported graph with the same preprocessing and cache contracts

Do not stop at `forward` compatibility alone.

## Pick the right template first

Choose the nearest existing implementation before editing.

- Text causal LM: start from `qwen3_legacy`
- Text + vision multimodal with split visual export: start from `qwen3_vl`
- Multimodal with extra cache families or multiple runtime modes: start from `qwen3_5`

If the target model has mixed full attention and linear attention, or extra recurrent / convolution cache state, treat `qwen3_5` as the primary template even if the upstream architecture name differs.

## Required file layout

### Text-only models

Typical deliverables:

- `_model.py` or `<model>_model_impl.py`: traceable wrappers and registration helpers
- `<model>_model.py`: top-level model registration and wrap entry
- `<model>_hmonnx_inference.py`: HMONNX runtime bridge
- `__init__.py`

### Vision-language models

Typical deliverables:

- `_llm_model_impl.py`: text-side traceable wrappers
- `_vision_model_impl.py`: visual-side traceable wrappers
- `<model>_llm_model.py`: top-level multimodal text model
- `<model>_vision_model.py` or `<model>_visual_model.py`: visual export model
- `<model>_hmonnx_inference.py`: runtime bridge combining visual and text graphs
- processor / preprocess helpers as needed
- `__init__.py`

Do not force visual logic into the text graph when the repository already expects the visual path to be exported separately.

## Conversion workflow

### 1. Identify the execution boundary

Read the upstream HF implementation and decide which submodules must become traceable.

Usually this includes:

- rotary embedding
- attention
- decoder block
- model body
- top-level causal LM head

For VLMs, separately identify:

- vision patch embedding
- vision attention / block
- merger / pooling head
- text-side multimodal embedding insertion path

The wrapper boundary should be as small as possible while still making export legal.

### 2. Register traceable replacements

Each wrapped HF class must be registered through `XHLLM_TRACEABLE_MODULES.register_module`.

Patterns seen in the references:

- wrap RMSNorm and replace with `xhquant.nn.RMSNorm` where needed
- wrap RoPE and prebuild cos / sin caches in `_setup`
- wrap attention and replace opaque ops with `xhquant.nn` or simple tensor ops
- wrap the decoder layer and make the residual, attention, norm, and MLP path explicit
- wrap the model top-level and emit final logits in graph-safe form

Keep names and semantics aligned with the HF model. Do not silently change numerics or cache ordering.

### 3. Move graph-only preparation into `_setup`

Use `_setup(cfg)` to materialize graph-time helpers that are unsafe or inconvenient inside tracing.

Common examples from the references:

- precompute RoPE cos / sin caches
- construct `Slice`, `BatchGather`, `Rope`, `MaskedSoftmax`, `LLMCache`, `LLMCacheV2`, `MatMul`, or similar xhquant ops
- split fused `qkv` weights into separate `q_proj`, `k_proj`, `v_proj`
- pad head dimensions to hardware-friendly multiples
- bind update callbacks to helper ops when sequence length depends on config

If a tensor transform is only needed to make export stable, put it in `_setup` rather than leaving it buried in runtime logic.

### 4. Implement `graph_forward` for export-safe execution

The graph path should receive explicit tensor arguments and avoid opaque HF runtime abstractions.

The common signature pieces are:

- `hidden_states` or `inputs_embeds`
- position embeddings or position ids
- `past_seq_length`
- `current_input_length`
- explicit key/value caches
- extra cache families for advanced models

Observed patterns:

- `qwen3_legacy` attention and decoder wrappers expose `graph_forward` and use explicit KV cache tensors
- `qwen3_vl` text wrappers pass multimodal position embeddings and visual features through explicit graph arguments
- `qwen3_5` extends the contract with linear-attention state such as convolution caches and recurrent states, and may maintain separate prefill / decode modes

Rule: if HF `forward` depends on `Cache`, dynamic helper objects, or Python-side multimodal insertion logic, mirror only the tensor contract in `graph_forward`.

### 5. Make cache contracts explicit and stable

This repository treats cache behavior as part of the model ABI.

You must define clearly:

- cache tensor order
- cache tensor shapes
- prefill vs decode behavior
- whether caches are mutated in place or returned as outputs
- any extra non-KV state such as convolution or recurrent state

Reference patterns:

- `qwen3_legacy`: classic key/value cache only
- `qwen3_vl`: key/value cache plus visual feature insertion during preprocessing
- `qwen3_5`: key/value cache plus linear-attention states and runtime mode switching

Do not hide cache updates behind Python objects in the graph path.

### 6. Build the top-level model wrapper

In the top-level model class:

- register through `@register_llm_model(...)`
- set `HF_MODEL_CLS`, `HF_AUTO_MODEL_CLS`, `HMONNXINFERENCE_CLS`, and config classes
- call the local wrapper registration helper in `init_wrap_model`
- for VLMs, override `_get_language_model` to point to the HF text backbone
- if visual export is split, hold a dedicated visual model instance and keep work directories in sync

For multimodal models, the text-side wrapper should not own raw visual preprocessing logic directly. Use the repository processor / preprocess path.

### 7. Provide an HF-compatible bridge when export still expects HF interfaces

Some export flows still need an HF-shaped model even after xhquant wrapping. In that case, create a compatibility bridge with `_DMRegistryCls("XHCompatible")`.

Typical responsibilities:

- remove heavy unused HF submodules to save memory
- keep only the pieces needed for export
- redirect `forward` to the xh wrapper model
- preserve HF output dataclasses such as `CausalLMOutputWithPast`

Reference patterns:

- `qwen3_vl`: HF-compatible text wrapper calls visual subgraph first, then runs xh text path
- `qwen3_5`: HF-compatible wrapper delegates to xh visual model and xh text model while preserving multimodal HF outputs

Do not use the HF-compatible bridge as the main runtime path. It exists to satisfy export and tooling boundaries.

### 8. Split vision export when appropriate

For VLMs in this repository, the visual branch is usually exported as a separate model.

Expected pattern:

- create a dedicated `BaseVisionModel` subclass
- wrap visual blocks through `_vision_model_impl.py`
- generate dummy visual inputs through the processor
- export visual model to ONNX with named inputs and outputs
- simplify and reload ONNX
- convert ONNX to frontend graph via `to_frontend_graph(..., FrontendType.ONNX, ...)`
- persist visual metadata into the exported model meta

Do not force the entire VLM into one monolithic graph unless the surrounding runtime already expects that.

### 9. Keep preprocessing outside the graph

Token / image preprocessing belongs in processor or data preprocess classes, not in `graph_forward`.

Use preprocess classes to assemble:

- token embeddings
- multimodal embeddings insertion
- M-RoPE position ids or position embeddings
- past / current sequence lengths
- cache tensors
- visual feature tensors
- linear attention masks or auxiliary state tensors

This is the clean separation used by both `qwen3_vl` and `qwen3_5`.

### 10. Build the HMONNX runtime bridge

The HMONNX wrapper must mirror the exported graph contract.

Typical responsibilities:

- load visual and text HMONNX artifacts
- convert `int64` inputs to `int32` if runtime expects it
- instantiate the same data preprocess class used during export
- own persistent cache tensors across calls
- update cache tensors after inference if the model returns new states
- expose `get_tf_processor()` so runtime callers can prepare inputs consistently

This is not optional. Without a runtime bridge, the exported graph is not actually usable by xhquant callers.

## Design rules extracted from the three references

### Rule 1: Replace only the hard parts

Do not rewrite the whole HF model if only attention, RoPE, and cache handling block export.

### Rule 2: Separate model wrapping from runtime preprocessing

Graph modules consume tensors. Processor and preprocess code build those tensors.

### Rule 3: Keep graph arguments explicit

Anything that matters to execution shape or state must be a tensor argument, not implicit Python state.

### Rule 4: Keep visual and text responsibilities split

In multimodal models, visual export, text graph execution, and multimodal assembly are distinct layers.

### Rule 5: Treat cache ABI as part of the public contract

Changing cache order or meaning is a behavior change, not an internal refactor.

### Rule 6: Reuse repository runtime scaffolding

Prefer `TextLLMModel`, `VisionLLMModel`, `BaseVisionModel`, HMONNX wrappers, preprocess helpers, and model meta classes already used nearby.

## Common implementation checklist

- identify nearest template
- add traceable wrappers for unsupported HF modules
- create `_setup` graph helpers
- implement `graph_forward` on attention / block / model layers as needed
- expose top-level logits path
- register top-level model via `register_llm_model`
- add HF-compatible export bridge if required
- add or reuse preprocess class
- for VLMs, add separate visual wrapper and ONNX-to-frontend conversion path
- add HMONNX runtime wrapper and cache update logic
- validate prefill and decode with smallest possible path

## Validation

Run commands after `source env.sh`.

Minimum validation targets:

- import the new model package successfully
- wrap the HF model successfully
- run minimal prefill logits
- run one decode step with cache reuse
- if multimodal, run one image-text sample through processor and visual path
- if visual export exists, verify ONNX export and frontend conversion
- if HMONNX wrapper exists, verify one end-to-end runtime call using the wrapper

Also run formatting / lint on touched Python files when feasible.

## Failure modes to watch for

- wrapping classes but never registering the top-level model
- preserving HF `forward` but forgetting `graph_forward`
- implicit cache mutation with no output contract
- mixing preprocessing logic into wrapped graph modules
- exporting visual features with names or order that do not match runtime preprocessing
- introducing a decode path that does not match prefill cache layout
- forgetting HF-compatible bridge cleanup, causing unnecessary memory retention

## Output standard

A successful conversion summary should state:

- which template was used and why
- which modules were wrapped
- what the graph inputs / outputs are
- how cache flow works in prefill and decode
- whether visual export is separate
- what was validated
- any remaining unsupported upstream features
