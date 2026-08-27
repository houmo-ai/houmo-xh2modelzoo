from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_merak_model_cards.py"
SPEC = importlib.util.spec_from_file_location("generate_merak_model_cards", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_extracts_llm_precision_and_technical_io(tmp_path: Path):
    config = {
        "quant": None,
        "export": {
            "target_device": "XH2a",
            "model_name": "Demo-0.6B",
            "quant_types": {"talker": "w8a8h1_sefp"},
            "model": {
                "type": "DemoModel",
                "export_cfg": {
                    "input_names": ["inputs_embeds", "past_seq_length"],
                    "output_names": ["logits"],
                },
            },
        },
    }
    card, warnings = generator._make_card(
        "configs_merak/workflows/xh2a/other_models/demo/0_6b/demo_0_6b.yaml",
        config,
        [],
        [],
        tmp_path,
    )

    assert card["model"]["id"] == "demo_0_6b"
    assert card["workflow"]["precision"]["components"]["talker"] == "w8a8h1_sefp"
    assert [item["name"] for item in card["frontend"]["inputs"]] == ["inputs_embeds", "past_seq_length"]
    assert card["frontend"]["outputs"][0]["name"] == "logits"
    assert card["frontend"]["submodels"][0]["id"] == "model"
    assert card["frontend"]["submodels"][0]["type"] == "DemoModel"
    assert card["frontend"]["submodels"][0]["precision"] == "w8a8h1_sefp"
    assert [item["name"] for item in card["frontend"]["submodels"][0]["inputs"]] == [
        "inputs_embeds",
        "past_seq_length",
    ]
    assert warnings


def test_dry_run_discovers_only_referenced_configs(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path
    examples = root / "examples_merak" / "llm" / "demo"
    configs = root / "configs_merak" / "workflows" / "xh2a" / "llm_models" / "demo" / "1b"
    examples.mkdir(parents=True)
    configs.mkdir(parents=True)
    (examples / "demo_workflow.py").write_text(
        'DEFAULT_CONFIG = "configs_merak/workflows/xh2a/llm_models/demo/1b/demo_1b.yaml"\n',
        encoding="utf-8",
    )
    (configs / "demo_1b.yaml").write_text(
        yaml.safe_dump({"quant": None, "export": {"model": {"model_name": "demo_1b"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(generator, "ROOT", root)
    args = SimpleNamespace(
        examples_root="examples_merak",
        workflow_root="configs_merak/workflows/xh2a",
        output_root="examples_merak",
        report="work_dirs/merak_delivery/card_generation_report.json",
        overwrite=False,
        dry_run=True,
    )

    assert generator.generate(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cards"] == 1
    assert not (examples / "model_cards").exists()


def test_generate_supports_absolute_output_root(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "repo"
    examples = root / "examples_merak" / "llm" / "demo"
    config = root / "configs_merak/workflows/xh2a/llm_models/demo/1b/demo_1b.yaml"
    output_root = tmp_path / "generated"
    report_path = tmp_path / "report.json"
    examples.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    relative_config = str(config.relative_to(root))
    (examples / "demo_workflow.py").write_text(f'CONFIG = "{relative_config}"\n', encoding="utf-8")
    config.write_text(yaml.safe_dump({"export": {"model": {}}}), encoding="utf-8")
    monkeypatch.setattr(generator, "ROOT", root)
    args = SimpleNamespace(
        examples_root="examples_merak",
        workflow_root="configs_merak/workflows/xh2a",
        output_root=str(output_root),
        report=str(report_path),
        overwrite=False,
        dry_run=False,
    )

    assert generator.generate(args) == 0
    capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["cards"][0]["card"] == str(output_root / "llm" / "demo" / "model_cards" / "demo_1b.yaml")


def test_formats_qwen_model_names():
    config_path = "configs_merak/workflows/xh2a/llm_models/qwen3_5/0_8b/qwen3_5_0_8b_full.yaml"

    assert generator._display_name({}, config_path) == "Qwen3.5 0.8B"
    assert (
        generator._display_name({}, "configs_merak/workflows/xh2a/llm_models/qwen3/0_6b/qwen3_0_6b_xh2a_w8a16.yaml")
        == "Qwen3 0.6B xh2a w8a16"
    )
    assert (
        generator._display_name({}, "configs_merak/workflows/xh2a/llm_models/qwen3/1_7b/qwen3_1_7b_xh2a_w8a8.yaml")
        == "Qwen3 1.7B xh2a w8a8"
    )


def test_groups_workflow_variants_and_prefers_full(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path
    examples = root / "examples_merak" / "llm" / "qwen3_5"
    configs = root / "configs_merak" / "workflows" / "xh2a" / "llm_models" / "qwen3_5" / "4b"
    examples.mkdir(parents=True)
    configs.mkdir(parents=True)
    config_paths = [
        "configs_merak/workflows/xh2a/llm_models/qwen3_5/4b/qwen3_5_4b_full.yaml",
        "configs_merak/workflows/xh2a/llm_models/qwen3_5/4b/qwen3_5_4b_no_quant.yaml",
        "configs_merak/workflows/xh2a/llm_models/qwen3_5/4b/qwen3_5_4b_visual_only_896.yaml",
        "configs_merak/workflows/xh2a/llm_models/qwen3_5/4b/qwen3_5_4b_dynamic_prune.yaml",
    ]
    (examples / "workflow.py").write_text(
        "\n".join(f'CONFIG_{index} = "{path}"' for index, path in enumerate(config_paths)),
        encoding="utf-8",
    )
    for path in config_paths:
        (root / path).write_text(yaml.safe_dump({"export": {"model": {}}}), encoding="utf-8")
    monkeypatch.setattr(generator, "ROOT", root)
    args = SimpleNamespace(
        examples_root="examples_merak",
        workflow_root="configs_merak/workflows/xh2a",
        output_root="examples_merak",
        report="work_dirs/merak_delivery/card_generation_report.json",
        overwrite=False,
        dry_run=False,
    )

    assert generator.generate(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cards"] == 1
    card = yaml.safe_load(
        (root / args.output_root / "llm" / "qwen3_5" / "model_cards" / "qwen3_5_4b.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(card) == {"schema_version", "external", "internal"}
    assert card["schema_version"] == 2
    assert card["external"]["model"]["display_name"] == "Qwen3.5 4B"
    assert card["internal"]["components"][0]["id"] == "model"
    assert card["internal"]["workflow"]["config_path"] == config_paths[0]
    report = json.loads((root / args.report).read_text(encoding="utf-8"))
    assert report["cards"][0]["variants"] == sorted(config_paths)


def test_owner_prefers_workflow_script(tmp_path: Path, monkeypatch):
    example_dir = tmp_path / "examples_merak" / "llm" / "demo"
    example_dir.mkdir(parents=True)
    workflow = example_dir / "demo_workflow.py"
    generate = example_dir / "demo_generate.py"
    workflow.write_text("", encoding="utf-8")
    generate.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        generator, "_git_first_author", lambda path: "workflow.owner" if path == workflow else "generate.owner"
    )

    assert generator._example_owner([example_dir]) == "workflow.owner"


def test_owner_example_dirs_fall_back_to_model_family(tmp_path: Path):
    examples_root = tmp_path / "examples_merak"
    family_dir = examples_root / "llm" / "qwen3_next"
    family_dir.mkdir(parents=True)

    assert generator._owner_example_dirs(
        "configs_merak/workflows/xh2a/llm_models/qwen3_next/80b/model.yaml",
        [],
        examples_root,
    ) == [family_dir]


def test_card_output_path_uses_nested_model_cards_with_dot_family(tmp_path: Path, monkeypatch):
    root = tmp_path
    examples_root = root / "examples_merak"
    example_dir = examples_root / "llm" / "mineru2.5"
    example_dir.mkdir(parents=True)
    monkeypatch.setattr(generator, "ROOT", root)
    card = {"model": {"id": "mineru2_5_pro", "family": "mineru2_5"}}

    assert generator._card_output_path(
        card,
        "configs_merak/workflows/xh2a/llm_models/mineru2_5/mineru2_5_pro.yaml",
        [],
        examples_root,
        examples_root,
    ) == example_dir / "model_cards" / "mineru2_5_pro.yaml"


def test_card_output_path_prefers_family_dir_over_stale_single_evidence(tmp_path: Path, monkeypatch):
    root = tmp_path
    examples_root = root / "examples_merak"
    qwen35 = examples_root / "llm" / "qwen3_5"
    moe = examples_root / "llm" / "qwen3_5_moe"
    qwen35.mkdir(parents=True)
    moe.mkdir(parents=True)
    monkeypatch.setattr(generator, "ROOT", root)
    card = {"model": {"id": "qwen3_5_122b_a10b", "family": "qwen3_5_moe"}}

    assert generator._card_output_path(
        card,
        "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b/qwen3_5_122b_a10b_full.yaml",
        [qwen35],
        examples_root,
        examples_root,
    ) == moe / "model_cards" / "qwen3_5_122b_a10b.yaml"


def test_overwrite_preserves_accuracy_and_benchmark(tmp_path: Path, monkeypatch, capsys):
    examples = tmp_path / "examples_merak/llm/demo"
    config = tmp_path / "configs_merak/workflows/xh2a/llm_models/demo/1b/demo_1b.yaml"
    examples.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    relative_config = str(config.relative_to(tmp_path))
    (examples / "demo_workflow.py").write_text(f'CONFIG = "{relative_config}"\n', encoding="utf-8")
    config.write_text(yaml.safe_dump({"export": {"model": {}}}), encoding="utf-8")
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    args = SimpleNamespace(
        examples_root="examples_merak",
        workflow_root="configs_merak/workflows/xh2a",
        output_root="examples_merak",
        report="work_dirs/merak_delivery/card_generation_report.json",
        overwrite=False,
        dry_run=False,
    )
    assert generator.generate(args) == 0
    capsys.readouterr()
    card_path = tmp_path / args.output_root / "llm" / "demo" / "model_cards" / "demo_1b.yaml"
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    card["internal"]["source"]["provider"] = "ModelScope"
    card["internal"]["source"]["url"] = "https://modelscope.cn/models/demo"
    card["internal"]["presentation"]["summary"] = "人工确认的摘要"
    card["external"]["precision"]["comparisons"] = [{"name": "verified"}]
    card["external"]["parameters"] = {"total": 1.2, "active": None, "unit": "B", "status": "verified"}
    card["external"]["compute"] = {
        "input_shape": [1, 2048],
        "prefill": 7.5,
        "decode": 0.0037,
        "unit": "TFLOPs",
        "status": "verified",
    }
    card["external"]["precision"]["float_format"] = "BF16"
    card["internal"]["benchmark"] = {"source": "benchmark_test", "cases": [{"script": "test.py"}]}
    card["external"]["owner"] = "model.owner"
    card["internal"]["release"]["quant_models"][0]["release_url"] = "https://example.com/release"
    card_path.write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")

    args.overwrite = True
    assert generator.generate(args) == 0
    updated = yaml.safe_load(card_path.read_text(encoding="utf-8"))

    assert set(updated) == {"schema_version", "external", "internal"}
    assert updated["internal"]["source"]["provider"] == "ModelScope"
    assert updated["internal"]["source"]["url"] == "https://modelscope.cn/models/demo"
    assert updated["internal"]["presentation"]["summary"] == "人工确认的摘要"
    assert updated["external"]["precision"]["comparisons"] == [{"name": "verified"}]
    assert updated["external"]["parameters"] == {"total": 1.2, "active": None, "unit": "B", "status": "verified"}
    assert updated["external"]["compute"] == {
        "input_shape": [1, 2048],
        "prefill": 7.5,
        "decode": 0.0037,
        "unit": "TFLOPs",
        "status": "verified",
    }
    assert updated["external"]["precision"]["float_format"] == "BF16"
    assert updated["internal"]["benchmark"]["cases"] == [{"script": "test.py"}]
    assert updated["external"]["owner"] == "model.owner"
    assert updated["internal"]["release"]["quant_models"][0]["release_url"] == "https://example.com/release"
