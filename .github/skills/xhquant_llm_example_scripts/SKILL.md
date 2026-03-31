---
name: xhquant_llm_example_scripts
description: Build or update example scripts for adapted LLM and VLM models, covering export, generate, and optional quant flows in the style used by this repository.
license: Apache-2.0
metadata:
  author: Lihui
  version: "2.0"
---

# xhquant Example Scripts

Use this skill after model adaptation work when the repository needs runnable example coverage.

## References

- `examples_merak/llm/qwen3_legacy/`
- `examples_merak/llm/qwen3moe/`
- `examples_merak/llm/qwen3_vl/`

Pick the closest existing example set and stay consistent with its argument names, config flow, and output layout.

## Deliverables

For a text LLM, usually provide:

- `<model_name>_xh_export_hmonnx.py`
- `<model_name>_xh_hmonnx_generate.py`
- optional quant script when the workflow requires offline quantization

For a VLM, follow the multimodal input pattern and only add quant scripts when they are actually supported.

## Export Script Requirements

- Support loading from config or model path in a way consistent with existing examples.
- Normalize model naming and persist generated config into the working directory when the flow expects it.
- Use the repository's standard config/model loading path rather than custom wrappers.
- Keep debug and overwrite behavior aligned with nearby examples.
- Format touched Python files with `ruff format` and use `ruff check --fix` for simple follow-up fixes.

## Generate Script Requirements

- Load exported metadata using the repository's hmonnx inference path.
- Support the common performance and golden-check options already used by nearby examples.
- Trim the prompt prefix before decoding generated output when that is the local convention.
- For VLMs, build inputs through the processor path instead of the text-only tokenizer path.

## Validation

Run commands after `source env.sh`.

At minimum verify:

- `ruff format` and `ruff check` on touched Python files when feasible
- export script starts with a valid config or model input
- generate script loads `golden_meta_info.json` and runs a minimal generation path
- README command examples are updated when a new example directory is created
