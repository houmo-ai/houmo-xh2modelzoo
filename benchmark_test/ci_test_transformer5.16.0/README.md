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
- `CI_TEST_BASE_REF`: optional base branch/commit; Git discovery uses its merge
  base with the head so unrelated commits on the base branch are not selected.
- `CI_TEST_FORCE_ALL=1`: run every active CI test without bypassing frozen-path
  or fail-closed policy checks.

Use `select_tests.py --format json` (or `--format report`) to inspect every
selected test and reason without running pytest. The normal runner prints a
compact summary, including whether selection is incremental or full and the
reason for any full-suite fallback.

An explicitly supplied empty changed-path file means a known empty diff: only
the fast policy gate runs. An absent diff and unavailable Git history mean an
unknown diff: the registered suite runs in full. Invalid refs and unreadable
path lists fail with an error, rather than quietly running a different diff.
Changed-file lists must include both old and new paths for renames. Local Git
discovery disables rename collapsing to preserve both paths.

## Coverage policy

The selector separates global safety policy from test ownership:

- `impact_policy.json`: frozen/guarded roots, documentation handling, always-run
  policy tests and genuinely global changes (selector, runner, common workflow
  engine, base model, dependencies). Editing this file triggers the full suite.
- `<test-stem>.impact.json`: a test's explicit source/config inputs, stored next
  to its Python file. These declarations are auto-discovered only inside
  `benchmark_test/ci_test_transformer5.16.0`. Adding a model never requires
  editing global policy.

Selection uses declared subsystem boundaries, then falls back to reverse
Python imports and model-family discovery for files that have no declaration.
All declarations matching a changed file contribute their test: shared inputs
can therefore select several families. Explicit declarations suppress noisy
import/family inference for that input, so they must name all intended
representative tests. Import/name inference alone is not proof of coverage of
dynamically loaded models or YAML recipes.

Changes under guarded source/config roots fail before pytest if no relevant
test can be found, even with `--all`. Declarations are validated before any
selection: missing tests, empty inputs, duplicate JSON keys/rule IDs, invalid
paths and unknown fields are errors. A declaration cannot change global policy
or redirect itself to another test. Both incremental and full runs have a hard
test boundary: only Python tests inside `benchmark_test/ci_test_transformer5.16.0`
can be selected. `tests/`, other benchmark lanes and their declarations are
ignored; imports and model-name inference cannot enroll them or use them as
evidence of CI coverage. Put required CI coverage in the active directory.

### Onboard a model

Add a runnable test, e.g. `test_new_model_export.py`, and an adjacent
`test_new_model_export.impact.json` (example placeholders below):

```json
{
  "version": 1,
  "rules": [
    {
      "id": "new-model-export",
      "patterns": [
        "xhmodel_merak/xh_llm/models/new_model/**",
        "configs_merak/workflows/**/new_model/**",
        "examples_merak/llm/new_model/**"
      ]
    }
  ]
}
```

Patterns are positive, repository-relative Python `fnmatch` globs; all matching
rules are combined. No negation/absolute paths are accepted. Keep patterns as
narrow as the test actually exercises. Add any shared helpers the test relies
on, and add the shared input to each existing affected test's declaration.
No list of target tests is needed: the declaration's filename determines its
one owner. Both `test_*.py` and `*_test.py` names are supported.

| Change | Selected tests |
| --- | --- |
| New model + test + declaration | New target and policy gate |
| Edit declaration inputs | Its owner, even without a source change |
| Delete declaration | Its surviving owner; other unmapped source changes still fail closed |
| Rename declarations | Surviving owners at both old and new paths |
| Retire test and declaration together | Remaining affected tests; removed tests are not executed |
| Shared input | All declared consumers |
| Global policy/engine/environment or unknown diff | Full registered suite |

Because ownership is immutable and filename-derived, changed-path-only Jenkins
checkouts need no baseline JSON or Git history to select old/new owners. Keep
the test and declaration together when moving or retiring a target. Changing
the inputs cannot hide that target from the current run.

The old combined `impact_rules.json` schema/CLI option is removed. Existing
input-to-test edges targeting the active suite have been migrated into test-local
declarations. Mappings to tests outside this CI directory have been removed. This
one-time selector/policy refactor itself triggers full CI; subsequent model
onboarding and declaration edits remain incremental.

### Design references

The implementation is repository-specific, standard-library-only code; no
external CI action or build system is required. The design borrows these ideas:

- [dorny/paths-filter](https://github.com/dorny/paths-filter): named path filters,
  explicit change-list inputs and merge-base-aware branch comparisons.
- [Nx affected](https://nx.dev/docs/features/ci-features/affected): map changed
  inputs to owned targets and their dependents, with conservative handling of
  truly global dependency changes.
- [Tinder/bazel-diff](https://github.com/Tinder/bazel-diff#how-it-works): treat
  dependency/rule changes as target-level changes. Bazel compares graph
  snapshots; our fixed test/declaration ownership avoids needing snapshots for
  declaration edits in source-only Jenkins checkouts.

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

MiniCPM-V-4.6 has a CPU-only text-preprocessing interface contract in
`test_minicpm_v_4_6_preprocess_contract.py`. It calls the real export/runtime
preprocessor factories with small tensors, without downloading checkpoints,
loading HMONNX sessions, or generating golden payloads. It covers embedding
padding, visual-feature scattering, sequential positions, chunk continuation,
decode, cache ordering, and the page-attention input ABI. The two MiniCPM
callers and their shared Qwen3.5 preprocessing/cache helpers select this test;
this is not a full MiniCPM vision/export/runtime correctness suite.
