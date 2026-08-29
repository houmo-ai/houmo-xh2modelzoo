# ModelZoo CI

`benchmark_test/ci_test_transformer5.16.0/run.sh` is the only active ModelZoo CI
entrypoint. It combines the retained BGE compatibility case, lightweight
recipe contracts, and SHA-pinned Transformers 5.16 block exports, then selects
only the tests affected by the current change.

## Run

```bash
./benchmark_test/ci_test_transformer5.16.0/run.sh
```

The selector discovers local changes by default. When available, it consumes
merge-request metadata from the CI runner so the whole patchset is considered.
Developers do not need to export a base ref or a GPTQModel source directory for
the normal path.

Supported controls:

- `CI_TEST_CHANGED_FILES_FILE`: path to a newline-separated changed-path file;
  preferred in CI so large changes do not enter the process environment.
- `CI_TEST_HEAD_REF`: diff head, default `HEAD`.
- `CI_TEST_FORCE_ALL=1`: run every active CI test without bypassing frozen-path
  or fail-closed policy checks.

Use `select_tests.py --format json` to inspect selected tests and reasons
without running pytest.

## Coverage policy

Selection uses explicit subsystem rules as authoritative boundaries, then
falls back to reverse Python imports and model-family discovery for files that
have no rule. Changes under guarded source/config roots fail before pytest if
no relevant test can be found. Add the test and, for dynamic/config-driven
relationships, update `impact_rules.json` in the same change.

Only the refactored implementation roots `xhmodel_merak`, `examples_merak`,
and `configs_merak` participate in impact selection. The legacy roots
`xh_model_zoo`, `xh_model_zoo_develop`, `examples`, `examples_develop`, and
`configs` are frozen: any change below them is rejected before test selection,
including when `CI_TEST_FORCE_ALL=1`. Existing tests may still execute frozen
code as a compatibility contract; that does not authorize changes in those
directories.

Documentation-only changes still run the fast selector-policy test, so the CI
status is reported instead of disappearing. The removed `ci_test` and
`ci_test_transformer5.5.0` trees are kept as explicit tombstones in the impact
rules. Their deletion paths run only the fast selector-policy test, which also
rejects any later recreation of either retired directory. Other deleted
guarded production/config files remain fail-closed.

## Transformers 5.16 boundary

The directory name tracks the Transformers 5.16 CI boundary. The CI runtime is
pinned to Transformers 5.16.1.

AutoRound runs once when a block checkpoint is published. Pull-request CI does
not quantize weights. It downloads the immutable checkpoint, verifies the
archive SHA-256, and runs the current ModelZoo export path. It does not call
`dump_golden` or create `golden`/`step_*` payloads. The maintained real export
matrix is:

- Qwen3.5-9B: 4 text layers;
- Qwen3.5-35B-A3B: 2 representative text layers;
- Gemma4 E2B: 6 text layers;
- Gemma4 26B-A4B: 2 representative text layers.

The BGE-Reranker case downloads the full model when its cache is absent and
runs the existing W8A8 quantization, HMONNX export, and golden generation path.
The former Qwen3-0.6B and Qwen3-30B-A3B subprocess exports are retired.

For the same four Transformer models, a cheap GPTQ/AutoRound contract layer
checks YAML-to-recipe translation, Python signatures, generated command parser
compatibility, and the public workflow `quant()` / `export()` interface. The
selector maps shared family code to both representative block exports and
model-specific workflow YAML to only that model's export.
