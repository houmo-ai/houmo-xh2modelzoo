import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / "merak_delivery/tools/merak_model_flow.py"
MODEL_CARD = ROOT / "examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml"
OLD_MODEL_CARD = "merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml"
MODEL_ID = "qwen3_0_6b"
VERSION_ID = "qwen3_0_6b_w8a16h1_sefp_draft"
TOOL_ENTRYPOINTS = [
    "merak_version_flow.py",
    "collect_merak_delivery_manifest.py",
    "check_merak_artifact.py",
    "register_merak_release.py",
    "build_model_catalog.py",
    "render_merak_readme.py",
]


def argparse_namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def run_flow(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FLOW), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_tool(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "merak_delivery/tools" / script), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_tool_directory_exposes_dedicated_entrypoints():
    for script in TOOL_ENTRYPOINTS:
        path = ROOT / "merak_delivery/tools" / script
        assert path.is_file(), f"missing dedicated tool entrypoint: {path}"


def test_run_workflow_dry_run_plans_real_workflow_without_running_heavy_steps(tmp_path):
    model_dir = tmp_path / "hf_model"
    result = run_flow(
        "run-workflow",
        "--model-card",
        str(MODEL_CARD.relative_to(ROOT)),
        "--work-dir",
        str(tmp_path / "flow"),
        "--model-dir",
        str(model_dir),
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["model_id"] == MODEL_ID
    assert plan["version_id"] == VERSION_ID
    assert plan["dry_run"] is True
    assert plan["workflow"]["class"] == "auto"
    assert plan["workflow"]["model_dir"] == str(model_dir)
    assert plan["planned_steps"] == ["quant", "export", "dump_golden", "eval"]
    assert plan["outputs"]["manifest"].endswith("delivery_manifest.json")
    assert plan["outputs"]["eval_report"].endswith("eval_report.json")
    assert plan["outputs"]["eval_work_dir"].endswith("hm_eval")


def test_run_workflow_dry_run_accepts_hm_eval_options(tmp_path):
    result = run_flow(
        "run-workflow",
        "--model-card",
        str(MODEL_CARD.relative_to(ROOT)),
        "--work-dir",
        str(tmp_path / "flow"),
        "--run-eval",
        "--eval-output",
        str(tmp_path / "eval_report.json"),
        "--eval-work-dir",
        str(tmp_path / "hm_eval_outputs"),
        "--eval-model",
        "qwen3_0_6b",
        "--eval-backend",
        "hmonnx",
        "--eval-datasets",
        "ceval",
        "mmlu_pro",
        "--eval-limit",
        "2",
        "--eval-max-tokens",
        "16",
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["outputs"]["eval_report"] == str(tmp_path / "eval_report.json")
    assert plan["outputs"]["eval_work_dir"] == str(tmp_path / "hm_eval_outputs")


def test_run_workflow_default_output_parent_is_work_dirs():
    result = run_flow(
        "run-workflow",
        "--model-card",
        str(MODEL_CARD.relative_to(ROOT)),
        "--dry-run",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert Path(plan["outputs"]["work_dir"]).relative_to(ROOT).parts[0] == "work_dirs"
    assert Path(plan["outputs"]["release_root"]).relative_to(ROOT).parts[0] == "work_dirs"
    assert Path(plan["outputs"]["catalog_root"]).relative_to(ROOT).parts[0] == "work_dirs"


def test_run_eval_writes_merak_eval_report_from_hm_eval_results(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "merak_delivery/tools"))
    import merak_model_flow

    def fake_run_hm_eval(**kwargs):
        assert kwargs["model_id"] == MODEL_ID
        assert kwargs["backend_type"] == "hmonnx"
        assert kwargs["datasets"] == ["ceval", "mmlu_pro"]
        assert kwargs["limit"] == 3
        assert kwargs["max_tokens"] == 12
        assert kwargs["hmonnx_meta"] == tmp_path / "golden_meta_info.json"
        return {
            "model": "Qwen3 0.6B",
            "backend": "hmonnx",
            "datasets": {
                "ceval": {
                    "status": "completed",
                    "metrics": {"accuracy": 0.75},
                    "elapsed_seconds": 1.2,
                    "report_files": [str(tmp_path / "ceval.json")],
                },
                "mmlu_pro": {
                    "status": "completed",
                    "metrics": {"score": 0.5},
                    "elapsed_seconds": 2.3,
                    "report_files": [str(tmp_path / "mmlu_pro.json")],
                },
            },
        }

    monkeypatch.setattr(merak_model_flow, "_run_hm_eval", fake_run_hm_eval)
    (tmp_path / "golden_meta_info.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "eval_report.json"
    status = merak_model_flow.run_eval_command(
        argparse_namespace(
            model_card=str(MODEL_CARD),
            work_dir=str(tmp_path / "work"),
            output=str(output),
            eval_work_dir=str(tmp_path / "hm_eval"),
            eval_model="",
            eval_backend="hmonnx",
            eval_datasets=["ceval", "mmlu_pro"],
            eval_limit=3,
            eval_max_tokens=12,
            hmonnx_meta=str(tmp_path / "golden_meta_info.json"),
            vision_hmonnx_meta="",
        )
    )
    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["model_id"] == MODEL_ID
    assert report["version_id"] == VERSION_ID
    assert report["status"] == "passed"
    assert [task["dataset"] for task in report["tasks"]] == ["ceval", "mmlu_pro"]
    assert report["tasks"][0]["metrics"]["accuracy"] == 0.75
    assert report["logs"]["backend"] == "hmonnx"


def test_run_eval_marks_report_failed_when_any_dataset_fails(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "merak_delivery/tools"))
    import merak_model_flow

    monkeypatch.setattr(
        merak_model_flow,
        "_run_hm_eval",
        lambda **kwargs: {
            "datasets": {
                "ceval": {"status": "completed", "metrics": {"accuracy": 1.0}},
                "mmlu_pro": {"status": "failed", "error": "backend crashed"},
            }
        },
    )
    output = tmp_path / "eval_report.json"
    status = merak_model_flow.run_eval_command(
        argparse_namespace(
            model_card=str(MODEL_CARD),
            work_dir=str(tmp_path / "work"),
            output=str(output),
            eval_work_dir=str(tmp_path / "hm_eval"),
            eval_model=MODEL_ID,
            eval_backend="hmonnx",
            eval_datasets=["ceval", "mmlu_pro"],
            eval_limit=0,
            eval_max_tokens=0,
            hmonnx_meta="",
            vision_hmonnx_meta="",
        )
    )
    assert status == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["tasks"][1]["error"] == "backend crashed"


def test_dedicated_collect_manifest_entrypoint_delegates_to_flow(tmp_path):
    export_dir = tmp_path / "export" / "hmquant_qwen3_0_6b"
    export_dir.mkdir(parents=True)
    golden_meta = export_dir / "golden_meta_info.json"
    golden_meta.write_text(
        json.dumps(
            {
                "model_config": {"context_max_length": 2048, "prefill_chunk_length": 256},
                "subgraphs": [],
            }
        ),
        encoding="utf-8",
    )
    result = run_tool(
        "collect_merak_delivery_manifest.py",
        "--model-card",
        str(MODEL_CARD.relative_to(ROOT)),
        "--work-dir",
        str(tmp_path / "work"),
        "--export-dir",
        str(export_dir.parent),
        "--golden-meta",
        str(golden_meta),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "work" / "delivery_manifest.json").is_file()


def test_full_metadata_flow_generates_manifest_release_catalog_and_readme(tmp_path):
    work_dir = tmp_path / "qwen3_0_6b"
    release_root = tmp_path / "releases" / "merak"
    catalog_root = tmp_path / "model_catalog"
    export_dir = work_dir / "export" / "hmquant_qwen3_0_6b"
    export_dir.mkdir(parents=True)
    (export_dir / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "model_name": "xh2_qwen3_0_6b_w8a16_256_2k",
                "model_config": {
                    "model_type": "Qwen3ForCausalLM",
                    "context_max_length": 2048,
                    "prefill_chunk_length": 256,
                },
                "subgraphs": ["prefill", "decode"],
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "prefill").mkdir()
    (export_dir / "decode").mkdir()
    eval_report = work_dir / "eval_report.json"
    eval_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": MODEL_ID,
                "version_id": VERSION_ID,
                "status": "passed",
                "tasks": [
                    {
                        "dataset": "smoke_eval_placeholder",
                        "metric": "accuracy",
                        "float": 1.0,
                        "quant": 1.0,
                        "delta": 0.0,
                        "status": "passed",
                    }
                ],
                "logs": {"report_path": str(eval_report), "run_log": ""},
            }
        ),
        encoding="utf-8",
    )

    collect = run_flow(
        "collect-manifest",
        "--model-card",
        str(MODEL_CARD.relative_to(ROOT)),
        "--work-dir",
        str(work_dir),
        "--export-dir",
        str(export_dir.parent),
        "--golden-meta",
        str(export_dir / "golden_meta_info.json"),
        "--eval-report",
        str(eval_report),
    )
    assert collect.returncode == 0, collect.stdout + collect.stderr
    manifest_path = work_dir / "delivery_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == MODEL_ID
    assert manifest["version_id"] == VERSION_ID
    assert manifest["artifacts"]["hmonnx_export_path"] == str(export_dir.parent)
    assert manifest["artifacts"]["golden_meta_info"] == str(export_dir / "golden_meta_info.json")

    check = run_flow("check-artifact", "--manifest", str(manifest_path))
    assert check.returncode == 0, check.stdout + check.stderr
    artifact_check_path = work_dir / "artifact_check.json"
    artifact_check = json.loads(artifact_check_path.read_text(encoding="utf-8"))
    assert artifact_check["status"] == "passed"

    register = run_flow(
        "register-release",
        "--manifest",
        str(manifest_path),
        "--artifact-check",
        str(artifact_check_path),
        "--eval-report",
        str(eval_report),
        "--release-root",
        str(release_root),
    )
    assert register.returncode == 0, register.stdout + register.stderr
    release_file = release_root / MODEL_ID / f"{VERSION_ID}.yaml"
    assert release_file.is_file()
    release_state = yaml.safe_load(release_file.read_text(encoding="utf-8"))
    assert release_state["status"] == "accuracy_passed"
    assert release_state["gates"]["artifact_valid"]["status"] == "passed"
    release_state["gates"]["metadata_valid"]["evidence"] = OLD_MODEL_CARD
    release_file.write_text(yaml.safe_dump(release_state, allow_unicode=True, sort_keys=False), encoding="utf-8")

    catalog = run_flow(
        "build-catalog",
        "--release-root",
        str(release_root),
        "--catalog-root",
        str(catalog_root),
        "--model-card-root",
        "examples_merak",
    )
    assert catalog.returncode == 0, catalog.stdout + catalog.stderr
    index_path = catalog_root / "data/index.json"
    detail_path = catalog_root / f"data/models/{MODEL_ID}.json"
    models_path = catalog_root / "data/models.json"
    assert index_path.is_file()
    assert detail_path.is_file()
    assert models_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert any(item["model_id"] == MODEL_ID for item in index["models"])
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    assert detail["model_id"] == MODEL_ID
    assert detail["versions"][0]["version_id"] == VERSION_ID
    assert detail["versions"][0]["gates"]["metadata_valid"]["evidence"] == str(MODEL_CARD.relative_to(ROOT))
    generated = (catalog_root / "data/models.generated.json").read_text(encoding="utf-8")
    assert OLD_MODEL_CARD not in generated
    assert OLD_MODEL_CARD not in models_path.read_text(encoding="utf-8")
    assert OLD_MODEL_CARD not in detail_path.read_text(encoding="utf-8")

    readme = run_flow(
        "render-readme",
        "--model-id",
        MODEL_ID,
        "--release-root",
        str(release_root),
        "--catalog-root",
        str(catalog_root),
    )
    assert readme.returncode == 0, readme.stdout + readme.stderr
    readme_path = release_root / MODEL_ID / f"{VERSION_ID}.md"
    assert readme_path.is_file()
    content = readme_path.read_text(encoding="utf-8")
    assert "# Qwen3 0.6B" in content
    assert VERSION_ID in content


def test_query_catalog_reads_modelzoo_cards_without_writing_catalog(tmp_path):
    release_root = tmp_path / "releases"
    catalog_root = tmp_path / "catalog"
    release_dir = release_root / MODEL_ID
    release_dir.mkdir(parents=True)
    release_file = release_dir / f"{VERSION_ID}.yaml"
    release_file.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model_id": MODEL_ID,
                "version_id": VERSION_ID,
                "status": "accuracy_passed",
                "gates": {
                    "metadata_valid": {
                        "status": "passed",
                        "evidence": OLD_MODEL_CARD,
                    }
                },
                "artifacts": {},
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (release_dir / "latest.yaml").write_text(
        yaml.safe_dump({"latest_version_id": VERSION_ID, "latest_file": release_file.name}, sort_keys=False),
        encoding="utf-8",
    )

    result = run_flow(
        "query-catalog",
        "--release-root",
        str(release_root),
        "--model-card-root",
        "examples_merak",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not catalog_root.exists()
    payload = json.loads(result.stdout)
    cards = list((ROOT / "examples_merak").glob("**/model_cards/*.yaml"))
    assert len(payload["models"]) == len(cards)
    qwen = next(model for model in payload["models"] if model["model_id"] == MODEL_ID)
    metadata_gate = next(gate for gate in qwen["gates"] if gate["name"] == "metadata_valid")
    assert metadata_gate["message"] == str(MODEL_CARD.relative_to(ROOT))
    assert OLD_MODEL_CARD not in result.stdout
