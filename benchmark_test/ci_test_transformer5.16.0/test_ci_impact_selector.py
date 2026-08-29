from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from select_tests import (
    SUITE_RELATIVE,
    FrozenPathError,
    MissingCoverageError,
    _default_suite,
    discover_changed_files,
    main,
    select_for_changes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.ci_policy
def test_qwen35_change_selects_contract_and_existing_unit_tests():
    selection = select_for_changes(
        ["xhmodel_merak/xh_llm/models/qwen3_5/quant_adapter.py"],
        repo_root=REPO_ROOT,
    )

    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_9b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_qwen35_35b_a3b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.py" in selection.tests
    assert "tests/qwen3_5/test_qwen35_quant_adapter.py" in selection.tests
    assert "tests/gemma4/test_quant_result_contract.py" not in selection.tests
    assert "tests/ling_3_flash/quant_adapter_test.py" not in selection.tests
    assert "tests/qwen3_next/test_workflow_standalone_semantics.py" not in selection.tests


@pytest.mark.ci_policy
def test_gemma4_change_stays_inside_explicit_family_boundary():
    selection = select_for_changes(
        ["xhmodel_merak/xh_llm/models/gemma4_series/quant_adapter.py"],
        repo_root=REPO_ROOT,
    )

    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_e2b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_gemma4_26b_a4b_block_export.py" in selection.tests
    assert "benchmark_test/ci_test_transformer5.16.0/test_transformer516_recipe_contract.py" in selection.tests
    assert "tests/gemma4/test_runtime_workflow_config_surface.py" in selection.tests
    assert "tests/gemma4/test_quant_result_contract.py" in selection.tests
    assert "tests/gemma4/test_flash_attention_export_contract.py" not in selection.tests
    assert "tests/gemma4/test_12b_unified_contract.py" not in selection.tests


@pytest.mark.ci_policy
def test_changed_test_is_always_selected():
    changed_test = "tests/qwen3_5/test_qwen35_quant_adapter.py"
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
