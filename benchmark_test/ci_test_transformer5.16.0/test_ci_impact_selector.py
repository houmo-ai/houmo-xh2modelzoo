from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from select_tests import (
    POLICY_RELATIVE,
    SUITE_RELATIVE,
    FrozenPathError,
    MissingCoverageError,
    _default_suite,
    _report,
    discover_changed_files,
    main,
    select_for_changes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.ci_policy
@pytest.mark.parametrize("filename", ("inference.py", "text_model.py"))
def test_minicpm46_preprocess_callers_select_focused_cpu_contract(filename):
    changed_path = f"xhmodel_merak/xh_llm/models/minicpm_v_4_6/{filename}"
    selection = select_for_changes([changed_path], repo_root=REPO_ROOT)
    contract = f"{SUITE_RELATIVE}/test_minicpm_v_4_6_preprocess_contract.py"
    assert set(selection.tests) == {f"{SUITE_RELATIVE}/test_ci_impact_selector.py", contract}
    assert selection.uncovered_files == ()
    assert selection.full_suite is False
    assert f"rule:minicpm-v46-text-preprocess-contract:{changed_path}" in selection.reasons[contract]


@pytest.mark.ci_policy
@pytest.mark.parametrize("filename", ("data_preprocess.py", "split_conv_cache_utils.py"))
def test_shared_qwen35_preprocess_also_selects_minicpm46_contract(filename):
    selection = select_for_changes(
        [f"xhmodel_merak/xh_llm/models/qwen3_5/{filename}"], repo_root=REPO_ROOT,
    )
    assert f"{SUITE_RELATIVE}/test_minicpm_v_4_6_preprocess_contract.py" in selection.tests
    assert f"{SUITE_RELATIVE}/test_qwen35_9b_block_export.py" in selection.tests
    assert f"{SUITE_RELATIVE}/test_qwen35_35b_a3b_block_export.py" in selection.tests
    assert selection.uncovered_files == ()


@pytest.mark.ci_policy
def test_qwen35_change_selects_only_active_ci_contract_and_exports():
    selection = select_for_changes(
        ["xhmodel_merak/xh_llm/models/qwen3_5/quant_adapter.py"],
        repo_root=REPO_ROOT,
    )

    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_35b_a3b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.py" in selection.tests
    assert "tests/qwen3_5/test_qwen35_quant_adapter.py" not in selection.tests
    assert "tests/gemma4/test_quant_result_contract.py" not in selection.tests
    assert "tests/ling_3_flash/quant_adapter_test.py" not in selection.tests
    assert "tests/qwen3_next/test_workflow_standalone_semantics.py" not in selection.tests


@pytest.mark.ci_policy
@pytest.mark.parametrize("filename", (
    "qwen3_5_moe_model.py", "qwen3_5_moe_hmonnx_inference.py", "_vision_model_impl.py",
))
def test_moe_source_selects_its_registered_block_export(filename):
    changed = f"xhmodel_merak/xh_llm/models/qwen3_5_moe/{filename}"
    selection = select_for_changes([changed], repo_root=REPO_ROOT)
    target = f"{SUITE_RELATIVE}/test_qwen35_35b_a3b_block_export.py"
    assert set(selection.tests) == {f"{SUITE_RELATIVE}/test_ci_impact_selector.py", target}
    assert f"rule:qwen35-moe-source:{changed}" in selection.reasons[target]
    assert selection.full_suite is False


@pytest.mark.ci_policy
def test_gemma4_change_stays_inside_explicit_family_boundary():
    selection = select_for_changes(
        ["xhmodel_merak/xh_llm/models/gemma4_series/quant_adapter.py"],
        repo_root=REPO_ROOT,
    )

    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_e2b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_26b_a4b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.py" in selection.tests
    assert "tests/gemma4/test_runtime_workflow_config_surface.py" not in selection.tests
    assert "tests/gemma4/test_quant_result_contract.py" not in selection.tests
    assert "tests/gemma4/test_flash_attention_export_contract.py" not in selection.tests
    assert "tests/gemma4/test_12b_unified_contract.py" not in selection.tests


@pytest.mark.ci_policy
def test_changed_test_is_always_selected():
    changed_test = f"{SUITE_RELATIVE}/test_transformer516_recipe_contract.py"
    selection = select_for_changes([changed_test], repo_root=REPO_ROOT)

    assert changed_test in selection.tests


@pytest.mark.ci_policy
def test_force_all_discovers_both_supported_test_filename_styles(tmp_path):
    suite = tmp_path / SUITE_RELATIVE
    suite.mkdir(parents=True)
    (suite / "test_prefix.py").touch()
    (suite / "suffix_test.py").touch()
    (suite / "helper.py").touch()

    assert _default_suite(tmp_path) == {
        f"{SUITE_RELATIVE}/suffix_test.py",
        f"{SUITE_RELATIVE}/test_prefix.py",
    }


@pytest.mark.ci_policy
def test_deleted_test_does_not_require_a_replacement_impact_mapping():
    deleted_test = "benchmark_test/ci_test_transformer5.16.0/test_retired_model.py"
    selection = select_for_changes([deleted_test], repo_root=REPO_ROOT)

    assert selection.tests == ("benchmark_test/ci_test_transformer5.16.0/test_ci_impact_selector.py",)


@pytest.mark.ci_policy
def test_qwen35_model_specific_configs_select_only_their_block_export():
    qwen9 = select_for_changes(
        ["configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"],
        repo_root=REPO_ROOT,
    )
    qwen35 = select_for_changes(
        [
            "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/"
            "qwen3_6_35b_a3b_full.yaml"
        ],
        repo_root=REPO_ROOT,
    )

    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" in qwen9.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_35b_a3b_block_export.py" not in qwen9.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_35b_a3b_block_export.py" in qwen35.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" not in qwen35.tests


@pytest.mark.ci_policy
def test_gemma4_model_specific_configs_select_only_their_block_export():
    e2b = select_for_changes(
        [
            "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/"
            "gemma4_e2b_autoround.yaml"
        ],
        repo_root=REPO_ROOT,
    )
    a4b = select_for_changes(
        [
            "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/"
            "gemma4_26b_a4b_autoround.yaml"
        ],
        repo_root=REPO_ROOT,
    )

    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_e2b_block_export.py" in e2b.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_26b_a4b_block_export.py" not in e2b.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_26b_a4b_block_export.py" in a4b.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_e2b_block_export.py" not in a4b.tests


@pytest.mark.ci_policy
@pytest.mark.parametrize(
    "changed_path",
    (
        "xh_model_zoo/xh_llm/models/qwen3_legacy/qwen3_hf_compatible.py",
        "xh_model_zoo_develop/core/converter.py",
        "examples/llm/bge_reranker/bge_reranker_xh2a_export_hmonnx.py",
        "examples_develop/cv/centernet/centernet_export.py",
        "configs/qwen2/7b/qwen2_7b_instruct_xh2a_2k.py",
    ),
)
def test_frozen_legacy_directories_reject_changes(changed_path):
    with pytest.raises(FrozenPathError, match="frozen legacy") as exc_info:
        select_for_changes([changed_path], repo_root=REPO_ROOT)

    assert changed_path in str(exc_info.value)


@pytest.mark.ci_policy
def test_force_all_does_not_bypass_frozen_legacy_policy(tmp_path, capsys):
    changed_path = "xh_model_zoo/xh_llm/models/bge_reranker/bge_reranker_converter.py"
    changed_files = tmp_path / "changed-files.txt"
    changed_files.write_text(f"{changed_path}\n", encoding="utf-8")

    assert (
        main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--changed-files-file",
                str(changed_files),
                "--all",
                "--format",
                "paths",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert changed_path in captured.err
    assert "frozen legacy" in captured.err


@pytest.mark.ci_policy
def test_docs_only_change_keeps_fast_policy_gate():
    selection = select_for_changes(["docs/ci.md"], repo_root=REPO_ROOT)

    assert selection.tests == ("benchmark_test/ci_test_transformer5.16.0/test_ci_impact_selector.py",)


@pytest.mark.ci_policy
def test_retired_ci_test_suite_is_guarded_by_tombstone_policy():
    retired_suite = "benchmark_test/ci_test"
    changed_paths = [
        f"{retired_suite}/run.sh",
        f"{retired_suite}/test_converter.py",
    ]

    selection = select_for_changes(
        changed_paths,
        repo_root=REPO_ROOT,
    )

    assert not (REPO_ROOT / retired_suite).exists()
    assert selection.full_suite is False
    assert selection.tests == ("benchmark_test/ci_test_transformer5.16.0/test_ci_impact_selector.py",)
    reasons = selection.reasons[selection.tests[0]]
    for changed_path in changed_paths:
        assert f"rule:retired-ci-test-suite:{changed_path}" in reasons


@pytest.mark.ci_policy
def test_retired_transformer55_suite_is_guarded_by_tombstone_policy():
    retired_suite = "benchmark_test/ci_test_transformer5.5.0"
    changed_paths = [
        f"{retired_suite}/run.sh",
        f"{retired_suite}/test_trimmed_quant_export.py",
    ]

    selection = select_for_changes(changed_paths, repo_root=REPO_ROOT)

    assert not (REPO_ROOT / retired_suite).exists()
    assert selection.full_suite is False
    assert selection.tests == ("benchmark_test/ci_test_transformer5.16.0/test_ci_impact_selector.py",)
    reasons = selection.reasons[selection.tests[0]]
    for changed_path in changed_paths:
        assert f"rule:retired-transformer55-suite:{changed_path}" in reasons


@pytest.mark.ci_policy
def test_ci_helpers_select_only_the_cases_they_support():
    block_helper = select_for_changes(
        ["benchmark_test/ci_test_transformer5.16.0/_transformer516_block_export.py"],
        repo_root=REPO_ROOT,
    )
    allure_helper = select_for_changes(
        ["benchmark_test/ci_test_transformer5.16.0/_allure_compat.py"],
        repo_root=REPO_ROOT,
    )

    assert block_helper.full_suite is False
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" in block_helper.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_e2b_block_export.py" in block_helper.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_bge_reranker.py" not in block_helper.tests
    assert allure_helper.full_suite is False
    assert "benchmark_test/ci_test_transformer5.16.0/test_bge_reranker.py" in allure_helper.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" not in allure_helper.tests


@pytest.mark.ci_policy
def test_unmapped_new_model_code_fails_closed():
    with pytest.raises(MissingCoverageError, match="no affected CI test"):
        select_for_changes(
            ["xhmodel_merak/xh_llm/models/brand_new_family/adapter.py"],
            repo_root=REPO_ROOT,
        )


@pytest.mark.ci_policy
def test_active_ci_infrastructure_change_falls_back_to_full_suite():
    selection = select_for_changes(
        ["benchmark_test/ci_test_transformer5.16.0/run.sh"],
        repo_root=REPO_ROOT,
    )

    assert selection.full_suite is True
    assert "benchmark_test/ci_test_transformer5.16.0/test_bge_reranker.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_e2b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.py" in selection.tests


@pytest.mark.ci_policy
def test_run_script_forwards_changed_files_file_to_selector(tmp_path, monkeypatch):
    changed_files = tmp_path / "changed files.txt"
    changed_files.write_text("docs/ci.md\n", encoding="utf-8")

    calls_file = tmp_path / "python-calls"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
import sys

with open(os.environ["CI_TEST_FAKE_PYTHON_CALLS"], "ab") as calls:
    calls.write(b"\\0".join(os.fsencode(arg) for arg in sys.argv[1:]) + b"\\n")

if sys.argv[1].endswith("select_tests.py"):
    print("benchmark_test/ci_test_transformer5.16.0/test_ci_impact_selector.py")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    monkeypatch.setenv("CI_TEST_CHANGED_FILES_FILE", str(changed_files))
    monkeypatch.setenv("CI_TEST_FAKE_PYTHON_CALLS", str(calls_file))
    monkeypatch.setenv("PATH", str(fake_bin), prepend=":")

    subprocess.run(
        ["bash", str(REPO_ROOT / SUITE_RELATIVE / "run.sh")],
        cwd=REPO_ROOT,
        check=True,
    )

    calls = [line.split(b"\0") for line in calls_file.read_bytes().splitlines()]
    selector_call = calls[0]
    option_index = selector_call.index(b"--changed-files-file")
    assert selector_call[option_index + 1].decode() == str(changed_files)


@pytest.mark.ci_policy
def test_git_discovery_preserves_unicode_untracked_path(tmp_path, monkeypatch):
    for name in (
        "CI_TEST_BASE_REF",
        "CI_MERGE_REQUEST_DIFF_BASE_SHA",
        "CI_TEST_HEAD_REF",
    ):
        monkeypatch.delenv(name, raising=False)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=tmp_path, check=True)
    baseline = tmp_path / "README.md"
    baseline.write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)

    document = tmp_path / "docs" / "量化流水说明.md"
    document.parent.mkdir()
    document.write_text("docs only\n", encoding="utf-8")

    assert discover_changed_files(tmp_path, None, "HEAD") == ["docs/量化流水说明.md"]


@pytest.mark.ci_policy
def test_git_discovery_uses_merge_base_not_base_branch_tip(tmp_path, monkeypatch):
    for name in (
        "CI_TEST_BASE_REF",
        "CI_MERGE_REQUEST_DIFF_BASE_SHA",
        "CI_TEST_HEAD_REF",
    ):
        monkeypatch.delenv(name, raising=False)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=tmp_path, check=True)

    subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=tmp_path, check=True)
    feature_file = tmp_path / "xhmodel_merak" / "models" / "demo.py"
    feature_file.parent.mkdir(parents=True)
    feature_file.write_text("# feature\n", encoding="utf-8")
    subprocess.run(["git", "add", feature_file.relative_to(tmp_path).as_posix()], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "feature"], cwd=tmp_path, check=True)

    subprocess.run(["git", "branch", "develop", "HEAD^"], cwd=tmp_path, check=True)
    subprocess.run(["git", "switch", "-q", "develop"], cwd=tmp_path, check=True)
    docs_file = tmp_path / "docs" / "base-only.md"
    docs_file.parent.mkdir(parents=True)
    docs_file.write_text("base only\n", encoding="utf-8")
    subprocess.run(["git", "add", docs_file.relative_to(tmp_path).as_posix()], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base moved"], cwd=tmp_path, check=True)
    subprocess.run(["git", "switch", "-q", "feature"], cwd=tmp_path, check=True)

    assert discover_changed_files(tmp_path, "develop", "HEAD") == ["xhmodel_merak/models/demo.py"]


@pytest.fixture
def declaration_repo(tmp_path):
    """A source-only Jenkins checkout: deliberately no .git directory."""
    suite = tmp_path / SUITE_RELATIVE
    suite.mkdir(parents=True)
    (suite / "test_policy.py").touch()
    (suite / "test_unrelated_export.py").touch()
    policy = {
        "version": 2,
        "always_tests": [f"{SUITE_RELATIVE}/test_policy.py"],
        "frozen_patterns": ["xh_model_zoo/**"],
        "documentation_patterns": ["docs/**"],
        "guarded_patterns": ["xhmodel_merak/**/*.py", "configs_merak/**/*.yaml"],
        "full_suite_patterns": [f"{SUITE_RELATIVE}/run.sh"],
    }
    (tmp_path / POLICY_RELATIVE).write_text(json.dumps(policy), encoding="utf-8")
    return tmp_path


def _declare_test(repo, name, patterns, *, in_suite=True):
    test = (Path(SUITE_RELATIVE) if in_suite else Path("tests")) / f"test_{name}.py"
    (repo / test).parent.mkdir(parents=True, exist_ok=True)
    (repo / test).touch()
    manifest = test.with_suffix(".impact.json")
    data = {"version": 1, "rules": [{"id": name, "patterns": patterns}]}
    (repo / manifest).write_text(json.dumps(data), encoding="utf-8")
    return test.as_posix(), manifest.as_posix()


@pytest.mark.ci_policy
def test_onboard_model_without_editing_global_policy_or_git_history(declaration_repo):
    repo = declaration_repo
    before = (repo / POLICY_RELATIVE).read_bytes()
    source = "xhmodel_merak/xh_llm/models/new_model/adapter.py"
    config = "configs_merak/workflows/xh2a/llm_models/new_model/full.yaml"
    test, manifest = _declare_test(repo, "new_model", [source, config])
    selection = select_for_changes([source, config, test, manifest], repo_root=repo)
    assert set(selection.tests) == {f"{SUITE_RELATIVE}/test_policy.py", test}
    assert selection.full_suite is False
    assert f"changed-manifest:{manifest}" in selection.reasons[test]
    assert (repo / POLICY_RELATIVE).read_bytes() == before
    assert not (repo / ".git").exists()


@pytest.mark.ci_policy
def test_minicpm_declaration_change_does_not_select_gemma_or_bge():
    manifest = f"{SUITE_RELATIVE}/test_minicpm_v_4_6_preprocess_contract.impact.json"
    selection = select_for_changes([
        manifest,
        "xhmodel_merak/xh_llm/models/minicpm_v_4_6/inference.py",
        "xhmodel_merak/xh_llm/models/minicpm_v_4_6/text_model.py",
    ], repo_root=REPO_ROOT)
    assert selection.tests == (
        f"{SUITE_RELATIVE}/test_ci_impact_selector.py",
        f"{SUITE_RELATIVE}/test_minicpm_v_4_6_preprocess_contract.py",
    )
    assert selection.full_suite is False


@pytest.mark.ci_policy
def test_shared_preprocess_and_minicpm_declaration_keep_both_qwen_exports():
    selection = select_for_changes([
        f"{SUITE_RELATIVE}/test_minicpm_v_4_6_preprocess_contract.impact.json",
        "xhmodel_merak/xh_llm/models/qwen3_5/data_preprocess.py",
    ], repo_root=REPO_ROOT)
    assert selection.full_suite is False
    for stem in ("qwen35_9b_block_export", "qwen35_35b_a3b_block_export", "minicpm_v_4_6_preprocess_contract"):
        assert f"{SUITE_RELATIVE}/test_{stem}.py" in selection.tests
    assert not any("gemma" in test or "bge" in test for test in selection.tests)


@pytest.mark.ci_policy
@pytest.mark.parametrize("change", ("edit", "delete", "rename", "retire"))
def test_declaration_lifecycle_keeps_surviving_old_and_new_targets(declaration_repo, change):
    repo = declaration_repo
    old, old_manifest = _declare_test(repo, "old", ["xhmodel_merak/old/**"])
    changed = [old_manifest]
    expected = {f"{SUITE_RELATIVE}/test_policy.py", old}
    if change == "edit":
        _declare_test(repo, "old", ["xhmodel_merak/new/**"])
    else:
        (repo / old_manifest).unlink()
    if change == "rename":
        new, new_manifest = _declare_test(repo, "new", ["xhmodel_merak/new/**"])
        expected.add(new)
        changed.append(new_manifest)
    if change == "retire":
        (repo / old).unlink()
        expected.remove(old)
        changed.append(old)
    selection = select_for_changes(changed, repo_root=repo)
    assert set(selection.tests) == expected
    assert selection.full_suite is False


@pytest.mark.ci_policy
def test_manifest_cannot_redirect_ownership_to_a_cheaper_test(declaration_repo):
    test, manifest = _declare_test(declaration_repo, "model", ["xhmodel_merak/model/**"])
    path = declaration_repo / manifest
    data = json.loads(path.read_text())
    data["rules"][0]["tests"] = [f"{SUITE_RELATIVE}/test_policy.py"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only id and patterns"):
        select_for_changes([manifest], repo_root=declaration_repo)


@pytest.mark.ci_policy
@pytest.mark.parametrize("payload", [
    [],
    {"version": 1, "rules": []},
    {"version": 3, "rules": []},
    {"version": True, "rules": []},
    {"version": 1, "rules": [{"id": "demo", "patterns": []}]},
    {"version": 1, "rules": [{"id": "demo", "patterns": "xhmodel_merak/**"}]},
    {"version": 1, "rules": [{"id": "demo", "patterns": ["../outside/**"]}]},
    {"version": 1, "rules": [{"id": "demo", "patterns": ["/tmp/**"]}]},
    {"version": 1, "rules": [{"id": "demo", "patterns": ["!xhmodel_merak/**"]}]},
    {"version": 1, "rules": [
        {"id": "demo", "patterns": ["xhmodel_merak/a/**"]},
        {"id": "demo", "patterns": ["xhmodel_merak/b/**"]},
    ]},
    {"version": 1, "rules": [], "full_suite_patterns": []},
])
def test_invalid_declarations_fail_before_pytest(declaration_repo, payload):
    _, manifest = _declare_test(declaration_repo, "demo", ["xhmodel_merak/demo/**"])
    (declaration_repo / manifest).write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        select_for_changes([manifest], repo_root=declaration_repo)


@pytest.mark.ci_policy
def test_duplicate_json_keys_are_not_silently_overwritten(declaration_repo):
    _, manifest = _declare_test(declaration_repo, "demo", ["xhmodel_merak/demo/**"])
    (declaration_repo / manifest).write_text('{"version": 1, "version": 2, "rules": []}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        select_for_changes([manifest], repo_root=declaration_repo)


@pytest.mark.ci_policy
def test_orphan_declaration_fails_even_for_docs_only_changes(declaration_repo):
    test, _ = _declare_test(declaration_repo, "demo", ["xhmodel_merak/demo/**"])
    (declaration_repo / test).unlink()
    with pytest.raises(ValueError, match="Orphan impact declaration"):
        select_for_changes(["docs/readme.md"], repo_root=declaration_repo)


@pytest.mark.ci_policy
def test_full_suite_ignores_tests_and_declarations_outside_ci_directory(declaration_repo):
    test, _ = _declare_test(declaration_repo, "external", ["xhmodel_merak/external/**"], in_suite=False)
    unrelated = declaration_repo / "tests/test_unregistered.py"
    unrelated.touch()
    selection = select_for_changes([f"{SUITE_RELATIVE}/run.sh"], repo_root=declaration_repo)
    assert test not in selection.tests
    assert "tests/test_unregistered.py" not in selection.tests
    assert all(test.startswith(f"{SUITE_RELATIVE}/") for test in selection.tests)
    assert selection.full_suite is True
    assert selection.full_suite_reasons == (f"full-suite:{SUITE_RELATIVE}/run.sh",)


@pytest.mark.ci_policy
def test_policy_edit_always_runs_full_even_if_its_glob_is_removed(declaration_repo):
    selection = select_for_changes([POLICY_RELATIVE], repo_root=declaration_repo)
    assert selection.full_suite is True
    assert f"{SUITE_RELATIVE}/test_unrelated_export.py" in selection.tests


@pytest.mark.ci_policy
def test_force_all_does_not_bypass_missing_coverage(declaration_repo):
    with pytest.raises(MissingCoverageError, match="no affected CI test"):
        select_for_changes(["xhmodel_merak/new/model.py"], repo_root=declaration_repo, force_all=True)


@pytest.mark.ci_policy
def test_declaration_does_not_bypass_frozen_policy(declaration_repo):
    test, manifest = _declare_test(declaration_repo, "legacy", ["xh_model_zoo/**"])
    with pytest.raises(FrozenPathError):
        select_for_changes(["xh_model_zoo/legacy.py", test, manifest], repo_root=declaration_repo)


@pytest.mark.ci_policy
def test_explicit_empty_diff_does_not_fall_back_to_unrelated_git_changes(declaration_repo, capsys):
    paths = declaration_repo / "changed-files.txt"
    paths.write_text("\n")
    assert main(["--repo-root", str(declaration_repo), "--changed-files-file", str(paths), "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tests"] == [f"{SUITE_RELATIVE}/test_policy.py"]
    assert result["full_suite"] is False


@pytest.mark.ci_policy
def test_missing_git_history_without_path_list_falls_back_with_reason(declaration_repo, capsys, monkeypatch):
    for name in ("CI_TEST_BASE_REF", "CI_TEST_HEAD_REF", "CI_MERGE_REQUEST_DIFF_BASE_SHA"):
        monkeypatch.delenv(name, raising=False)
    assert main(["--repo-root", str(declaration_repo), "--format", "json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["full_suite"] is True
    assert result["full_suite_reasons"] == ["unknown-diff-full-suite"]


@pytest.mark.ci_policy
def test_changed_paths_only_cli_selects_new_model_without_history(declaration_repo):
    source = "xhmodel_merak/new/model.py"
    test, manifest = _declare_test(declaration_repo, "new", [source])
    paths = declaration_repo / "changed files.txt"
    paths.write_text("\n".join((source, test, manifest)))
    process = subprocess.run([
        sys.executable, str(REPO_ROOT / SUITE_RELATIVE / "select_tests.py"),
        "--repo-root", str(declaration_repo), "--changed-files-file", str(paths), "--format", "json",
    ], cwd=declaration_repo, text=True, capture_output=True, check=True)
    result = json.loads(process.stdout)
    assert result["tests"] == [test, f"{SUITE_RELATIVE}/test_policy.py"]
    assert result["full_suite"] is False


@pytest.mark.ci_policy
def test_selection_report_is_compact_but_full_reasons_are_available():
    selection = select_for_changes([
        "xhmodel_merak/xh_llm/models/qwen3_5/workflow.py",
        "xhmodel_merak/xh_llm/models/qwen3_5/data_preprocess.py",
        "xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py",
        "xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_processor.py",
    ], repo_root=REPO_ROOT)
    compact = _report(selection, detailed=False)
    detailed = _report(selection)
    assert "mode: incremental" in compact
    assert "more reasons (use --format json/report)" in compact
    assert len(compact) < len(detailed)


@pytest.mark.ci_policy
def test_actual_runner_onboards_model_in_source_only_checkout(declaration_repo):
    repo = declaration_repo
    suite = repo / SUITE_RELATIVE
    source = "xhmodel_merak/new/model.py"
    test, manifest = _declare_test(repo, "new", [source])
    (repo / test).write_text("def test_new():\n    assert True\n")
    (suite / "test_policy.py").write_text("def test_policy():\n    assert True\n")
    (suite / "test_unrelated_export.py").write_text("raise AssertionError('unrelated export was collected')\n")
    for filename in ("run.sh", "select_tests.py", "_impact_manifest.py"):
        (suite / filename).write_bytes((REPO_ROOT / SUITE_RELATIVE / filename).read_bytes())
    (suite / "pytest.ini").write_text("[pytest]\n")
    paths = repo / "changed files.txt"
    paths.write_text("\n".join((source, test, manifest)))
    env = dict(os.environ)
    env.pop("CI_TEST_FORCE_ALL", None)
    env["CI_TEST_CHANGED_FILES_FILE"] = str(paths)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    process = subprocess.run(
        ["bash", str(suite / "run.sh"), "-q"], cwd=repo, env=env, text=True, capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "2 passed" in process.stdout
    assert "mode: incremental" in process.stderr
    assert "test_unrelated_export" not in process.stderr


@pytest.mark.ci_policy
def test_git_discovery_includes_both_sides_of_manifest_rename(tmp_path, monkeypatch):
    for name in ("CI_TEST_BASE_REF", "CI_TEST_HEAD_REF", "CI_MERGE_REQUEST_DIFF_BASE_SHA"):
        monkeypatch.delenv(name, raising=False)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=tmp_path, check=True)
    old = tmp_path / "test_old.impact.json"
    old.write_text('{"version": 1, "rules": []}')
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    old.rename(tmp_path / "test_new.impact.json")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "rename"], cwd=tmp_path, check=True)
    assert set(discover_changed_files(tmp_path, "HEAD^", "HEAD")) == {
        "test_old.impact.json", "test_new.impact.json",
    }


@pytest.mark.ci_policy
def test_invalid_base_ref_fails_instead_of_silently_changing_diff(declaration_repo, capsys):
    result = main(["--repo-root", str(declaration_repo), "--base-ref", "nonexistent", "--format", "paths"])
    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ci-impact error:" in captured.err


@pytest.mark.ci_policy
def test_missing_changed_paths_file_fails_with_clear_error(declaration_repo, capsys):
    result = main([
        "--repo-root", str(declaration_repo), "--changed-files-file", str(declaration_repo / "missing.txt"),
    ])
    assert result == 2
    assert "missing.txt" in capsys.readouterr().err


@pytest.mark.ci_policy
@pytest.mark.parametrize("changed_path", [
    "tests/qwen3_5/test_qwen35_quant_adapter.py",
    "tests/qwen3_next/test_workflow_standalone_semantics.py",
    "tests/gemma4/test_quant_result_contract.impact.json",
    "benchmark_test/another_lane/test_model.py",
    "benchmark_test/another_lane/conftest.py",
    "benchmark_test/another_lane/run.sh",
])
def test_other_test_directories_do_not_enter_ci_selection(changed_path):
    selection = select_for_changes([changed_path], repo_root=REPO_ROOT)
    assert selection.tests == (f"{SUITE_RELATIVE}/test_ci_impact_selector.py",)
    assert selection.full_suite is False


@pytest.mark.ci_policy
def test_outside_test_imports_and_names_cannot_satisfy_missing_ci_coverage(declaration_repo):
    repo = declaration_repo
    source = repo / "xhmodel_merak/xh_llm/models/brand_new/model.py"
    source.parent.mkdir(parents=True)
    source.touch()
    test, manifest = _declare_test(repo, "brand_new", [source.relative_to(repo).as_posix()], in_suite=False)
    (repo / test).write_text("from xhmodel_merak.xh_llm.models.brand_new import model\n")
    # Even invalid out-of-lane manifests are ignored, not loaded/validated.
    (repo / manifest).write_text("not json")
    with pytest.raises(MissingCoverageError):
        select_for_changes([source.relative_to(repo).as_posix()], repo_root=repo)


@pytest.mark.ci_policy
def test_global_policy_cannot_register_an_outside_ci_test(declaration_repo):
    test, _ = _declare_test(declaration_repo, "outside", ["xhmodel_merak/**"], in_suite=False)
    path = declaration_repo / POLICY_RELATIVE
    policy = json.loads(path.read_text())
    policy["always_tests"].append(test)
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="missing/non-test paths"):
        select_for_changes([POLICY_RELATIVE], repo_root=declaration_repo)


@pytest.mark.ci_policy
def test_import_fallback_still_finds_tests_inside_active_ci(declaration_repo):
    repo = declaration_repo
    source = repo / "xhmodel_merak/demo.py"
    source.parent.mkdir()
    source.touch()
    test = f"{SUITE_RELATIVE}/test_dependency.py"
    (repo / test).write_text("from xhmodel_merak import demo\n")
    selection = select_for_changes(["xhmodel_merak/demo.py"], repo_root=repo)
    assert test in selection.tests
    assert "python-import:xhmodel_merak/demo.py" in selection.reasons[test]
    assert selection.full_suite is False


@pytest.mark.ci_policy
def test_family_fallback_still_finds_tests_inside_active_ci(declaration_repo):
    repo = declaration_repo
    test = f"{SUITE_RELATIVE}/test_brand_new_export.py"
    (repo / test).touch()
    changed = "configs_merak/workflows/llm_models/brand_new/full.yaml"
    selection = select_for_changes([changed], repo_root=repo)
    assert test in selection.tests
    assert f"model-family:{changed}" in selection.reasons[test]


@pytest.mark.ci_policy
def test_every_selection_path_stays_inside_active_ci_directory():
    cases = [
        ["xhmodel_merak/xh_llm/models/qwen3_5/workflow.py"],
        ["xhmodel_merak/xh_llm/models/qwen3_5_moe/qwen3_5_moe_model.py"],
        ["xhmodel_merak/xh_llm/models/gemma4_series/workflow.py"],
        [f"{SUITE_RELATIVE}/select_tests.py"],
        ["tests/qwen3_next/test_workflow_standalone_semantics.py"],
        [],
        None,
    ]
    for changed in cases:
        selection = select_for_changes(changed, repo_root=REPO_ROOT)
        assert all(test.startswith(f"{SUITE_RELATIVE}/") for test in selection.tests)
