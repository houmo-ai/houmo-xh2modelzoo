---
name: xhquant_llm_reviewer
description: Review whether an adapted transformers LLM or VLM model is compatible with the xhmodel_merak architecture. Focus on wrapper registration, graph_forward legality, cache flow, and output contracts.
license: Apache-2.0
metadata:
  author: Lihui
  version: "2.0"
---

# xhquant LLM Reviewer

Use this skill for compatibility review of adapted model code under `xhmodel_merak/xh_llm/models`.

## Scope

- Review only. Do not modify code.
- Focus on adaptation correctness, not generic style advice.
- Compare the adapted implementation against the repository's established patterns such as `qwen3_legacy`, `qwen3moe`, and `qwen3_vl`.

## Required Checks

### 1. Registration

- Core adapted modules should be registered through `XHLLM_TRACEABLE_MODULES`.
- Entry model classes should still be discoverable through the repository's normal model registration flow.

### 2. `graph_forward` Legality

- Only allow `graph_forward` when the adapted graph path needs a signature different from the original HF `forward`.
- If the effective signature matches the original HF `forward`, adding `graph_forward` is a compatibility defect and should be reported as `Blocker`.

### 3. Cache and Sequence Contracts

- Check prefill and decode paths for consistent cache read/write behavior.
- Check `past_seq_length`, `current_input_length`, rotary cache slicing, and per-layer cache routing.
- Check that `num_logits_to_keep` semantics remain stable:
  - `0`: keep full sequence logits
  - `1`: keep the final token logits

### 4. Tensor Contracts

- Check shape assumptions around q/k/v reshape and transpose.
- Check dtype consistency for rotary cache, attention math, and output logits.
- Check that hidden states and cache tensors preserve expected batch and sequence axes.

### 5. Config Compatibility

- Check model type naming, config field mapping, and runtime assumptions against existing examples and configs.
- Treat quant type aliases as equivalent when appropriate:
  - `w8a8_sefp` == `w8a8h1_sefp`
  - `w8a8_ssfp` == `w8a8h0_ssfp`

## Output

Report `Blocker`, `Major`, and `Minor` findings with file references.

Each finding should explain:

- what is incompatible
- why it matters for this architecture
- what should change conceptually

If compatibility looks correct, state that explicitly and call out any areas that were not validated by execution.
