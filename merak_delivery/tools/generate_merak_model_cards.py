#!/usr/bin/env python3
"""Discover supported Merak examples and generate one model card per workflow config.

The generator is intentionally metadata-only: it never imports model code or runs a
workflow. It combines strong config references from example scripts/READMEs with
structured workflow YAML and emits warnings for fields that require human input.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from model_card_schema import normalize_model_card, to_v2_model_card

CONFIG_RE = re.compile(r"(?:\.\.?/)?(configs_merak/workflows/xh2a/[A-Za-z0-9_./-]+\.ya?ml)")
URL_RE = re.compile(r"https?://[^\s)`\"'<>]+")
SUPPORTED_CATEGORIES = {"llm_models", "audio_models", "other_models"}


def _sanitize(value: str) -> str:
    value = value.strip().lower().replace(".", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "model"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _relative_or_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _find_quant_types(value: Any, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "quant_type" and isinstance(item, str):
                found[_sanitize(prefix or "model")] = item
            elif key == "quant_types" and isinstance(item, dict):
                found.update({str(k): str(v) for k, v in item.items() if isinstance(v, str)})
            else:
                found.update(_find_quant_types(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_quant_types(item, f"{prefix}[{index}]"))
    return found


def _io_item(name: str, direction: str, submodel: str = "") -> dict[str, Any]:
    meaning = {
        "input": "模型输入张量或预处理后的输入参数。",
        "output": "模型输出张量或后处理前的输出参数。",
    }[direction]
    return {
        "name": name,
        "type": "tensor" if name not in {"text", "audio", "image", "video"} else name,
        "required": direction == "input",
        "description": meaning,
        **({"submodel": submodel} if submodel else {}),
    }


def _extract_io(node: Any, submodel: str = "") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(node, dict):
        return [], []
    cfg = node.get("export_cfg") if isinstance(node.get("export_cfg"), dict) else node
    inputs = cfg.get("input_names", []) if isinstance(cfg, dict) else []
    outputs = cfg.get("output_names", []) if isinstance(cfg, dict) else []
    if not isinstance(inputs, list):
        inputs = []
    if not isinstance(outputs, list):
        outputs = []
    return (
        [_io_item(str(name), "input", submodel) for name in inputs if name],
        [_io_item(str(name), "output", submodel) for name in outputs if name],
    )


def _example_metadata(examples_root: Path) -> tuple[dict[str, list[Path]], dict[str, list[str]]]:
    config_examples: dict[str, list[Path]] = defaultdict(list)
    example_urls: dict[str, list[str]] = defaultdict(list)
    for path in examples_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".markdown"}:
            continue
        text = _read(path)
        example_dir = path.parent
        for match in CONFIG_RE.finditer(text):
            config_examples[match.group(1)].append(example_dir)
        urls = URL_RE.findall(text)
        if urls:
            key = str(example_dir.relative_to(examples_root))
            example_urls[key].extend(urls)
    return config_examples, example_urls


def _git_first_author(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "--follow", "--reverse", "--format=%an", "--", str(path.relative_to(ROOT))],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    authors = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return authors[0] if authors else ""


def _example_owner(example_dirs: list[Path]) -> str:
    candidates: list[tuple[int, str, Path]] = []
    for directory in example_dirs:
        for path in directory.glob("*.py"):
            name = path.stem.lower()
            priority = 5
            if "workflow" in name:
                priority = 0
            elif any(token in name for token in ("generate", "export")):
                priority = 1
            elif any(token in name for token in ("pipeline", "hmonnx", "eval")):
                priority = 2
            candidates.append((priority, name, path))
    for _, _, path in sorted(candidates):
        author = _git_first_author(path)
        if author:
            return author
    return ""


def _owner_example_dirs(config_path: str, example_dirs: list[Path], examples_root: Path) -> list[Path]:
    """Resolve the model-family example directory when the config is not cited directly."""
    if example_dirs:
        return example_dirs
    parts = Path(config_path).parts
    family = ""
    for marker in ("llm_models", "audio_models", "other_models"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                family = parts[index + 1]
            break
    if not family:
        return []
    return [path for path in examples_root.rglob("*") if path.is_dir() and path.name == family]


def _config_family(config_path: str) -> str:
    parts = Path(config_path).parts
    try:
        index = parts.index("llm_models")
    except ValueError:
        try:
            index = parts.index("audio_models")
        except ValueError:
            index = parts.index("other_models")
    return _sanitize(parts[index + 1]) if len(parts) > index + 1 else "other"


def _example_dir_for_card(
    card: dict[str, Any],
    config_path: str,
    example_dirs: list[Path],
    examples_root: Path,
) -> Path:
    family = _sanitize(str(card.get("model", {}).get("family", "")))
    candidates = [path for path in example_dirs if path.exists()]
    if family:
        family_matches = [path for path in candidates if _sanitize(path.name) == family]
        if family_matches:
            return sorted(family_matches)[0]

    discovered = [path for path in examples_root.rglob("*") if path.is_dir() and _sanitize(path.name) == family]
    if discovered:
        return sorted(discovered)[0]
    if len(candidates) == 1:
        return candidates[0]

    parts = Path(config_path).parts
    bucket = "other"
    raw_family = str(card.get("model", {}).get("family") or family or "model")
    for marker, example_bucket in (
        ("llm_models", "llm"),
        ("audio_models", "audio"),
        ("other_models", "other"),
    ):
        if marker in parts:
            bucket = example_bucket
            index = parts.index(marker)
            if index + 1 < len(parts):
                raw_family = parts[index + 1]
            break
    return examples_root / bucket / raw_family


def _card_output_path(
    card: dict[str, Any],
    config_path: str,
    example_dirs: list[Path],
    examples_root: Path,
    output_root: Path,
) -> Path:
    example_dir = _example_dir_for_card(card, config_path, example_dirs, examples_root)
    try:
        relative_example = example_dir.resolve().relative_to(examples_root.resolve())
    except ValueError:
        relative_example = Path(_sanitize(example_dir.name))
    base = examples_root if output_root.resolve() == examples_root.resolve() else output_root
    return base / relative_example / "model_cards" / f"{card['model']['id']}.yaml"


def _legacy_card_path(card: dict[str, Any]) -> Path:
    return ROOT / "merak_delivery" / "model_cards" / "merak" / card["model"]["family"] / f"{card['model']['id']}.yaml"


def _preserve_existing_fields(card: dict[str, Any], existing: dict[str, Any]) -> None:
    existing = normalize_model_card(existing)
    existing_source = existing.get("source", {}) if isinstance(existing, dict) else {}
    for field in ("provider", "name", "url", "license", "raw_model_path"):
        value = existing_source.get(field) if isinstance(existing_source, dict) else None
        if value not in (None, "", [], {}):
            card["source"][field] = deepcopy(value)

    existing_workflow = existing.get("workflow", {}) if isinstance(existing, dict) else {}
    value = existing_workflow.get("model_dir") if isinstance(existing_workflow, dict) else None
    if isinstance(value, str) and value.strip() and value.strip() != "<model_dir>":
        card["workflow"]["model_dir"] = value

    existing_frontend = existing.get("frontend", {}) if isinstance(existing, dict) else {}
    for field in ("summary", "inputs", "outputs", "submodels", "demo", "limitations"):
        value = existing_frontend.get(field) if isinstance(existing_frontend, dict) else None
        if value not in (None, "", [], {}):
            card["frontend"][field] = deepcopy(value)

    existing_release = existing.get("release", {}) if isinstance(existing, dict) else {}
    for field in ("version_id", "target_status", "url", "owner", "reviewer", "quant_models"):
        value = existing_release.get(field) if isinstance(existing_release, dict) else None
        if value not in (None, "", [], {}):
            card["release"][field] = deepcopy(value)

    for field in ("accuracy", "benchmark"):
        value = existing.get(field) if isinstance(existing, dict) else None
        if value:
            card[field] = deepcopy(value)

    for field in ("parameters", "compute", "external_precision"):
        value = existing.get(field) if isinstance(existing, dict) else None
        if isinstance(value, dict):
            card[field] = deepcopy(value)


def _display_name(model: dict[str, Any], config_path: str) -> str:
    export = model.get("export", {}) if isinstance(model.get("export"), dict) else {}
    main = export.get("model", {}) if isinstance(export.get("model"), dict) else {}
    raw = _first(main, "model_name", "model_id") or _first(export, "model_name") or Path(config_path).stem
    text = str(raw).replace("-", "_")
    text = re.sub(r"(?i)_(?:full|autoround|no_quant|visual_only|lora|gptq|mtp|dflash|dynamic_prune)(?:_.*)?$", "", text)
    text = re.sub(r"(?i)^(qwen\d+)_(5|6)(?=_)", r"\1.\2", text)
    text = re.sub(r"(?i)_(\d+)_(\d+)(?=b|m|k)", r" \1.\2", text)
    text = text.replace("_", " ")
    parts = []
    for part in text.split():
        lowered = part.lower()
        if lowered in {"llm", "vlm", "tts", "asr", "moe"}:
            parts.append(lowered.upper())
        elif re.fullmatch(r"\d+(?:\.\d+)?b", lowered):
            parts.append(lowered[:-1] + "B")
        elif lowered.startswith("qwen"):
            parts.append("Qwen" + part[4:])
        elif lowered.startswith("cosyvoice"):
            parts.append("CosyVoice" + part[9:])
        else:
            parts.append(part)
    return " ".join(parts)


def _model_group_key(config_path: str) -> str:
    """Return the base model key shared by full and auxiliary workflow variants."""
    stem = Path(config_path).stem.lower().replace("-", "_")
    stem = re.sub(r"_(?:full|autoround|no_quant|visual_only|lora|gptq|mtp|dflash|dynamic_prune)(?:_.*)?$", "", stem)
    stem = re.sub(r"_w\d+a\d+(?:h\d+)?(?:_[a-z0-9]+)?$", "", stem)
    stem = re.sub(r"_xh2a$", "", stem)
    return _sanitize(stem)


def _config_priority(config_path: str) -> tuple[int, int, str]:
    """Prefer the canonical full workflow, then the least specialized variant."""
    stem = Path(config_path).stem.lower()
    if re.search(r"_full$", stem):
        rank = 0
    elif "_full_" in stem:
        rank = 1
    elif "_autoround" in stem:
        rank = 3
    elif "_no_quant" in stem:
        rank = 4
    elif "_visual_only" in stem:
        rank = 5
    elif "_dynamic_prune" in stem:
        rank = 6
    else:
        rank = 2
    return rank, len(stem), config_path


def _infer_modality(config_path: str, example_dirs: list[Path], model: dict[str, Any]) -> list[str]:
    text = (config_path + " " + " ".join(str(path).lower() for path in example_dirs)).lower()
    modality = []
    for token, value in (
        ("image", "image"),
        ("vision", "image"),
        ("video", "video"),
        ("audio", "audio"),
        ("tts", "audio"),
        ("asr", "audio"),
        ("embedding", "embedding"),
    ):
        if token in text and value not in modality:
            modality.append(value)
    if not modality:
        modality.append("text")
    if "text" not in modality and any(item in modality for item in ("image", "audio", "video")):
        modality.insert(0, "text")
    return modality


def _make_card(
    config_path: str, config: dict[str, Any], example_dirs: list[Path], urls: list[str], examples_root: Path
) -> tuple[dict[str, Any], list[str]]:
    category = next((item for item in SUPPORTED_CATEGORIES if f"/{item}/" in f"/{config_path}"), "other_models")
    family = _config_family(config_path)
    model_id = _model_group_key(config_path)
    export = config.get("export", {}) if isinstance(config.get("export"), dict) else {}
    main = export.get("model", {}) if isinstance(export.get("model"), dict) else {}
    quant_types = _find_quant_types(export)
    overall = next(iter(quant_types.values()), "")
    inputs, outputs = _extract_io(main)
    submodels: list[dict[str, Any]] = []
    components = export.get("components", [])
    component_names = (
        list(components) if isinstance(components, list) else list(components) if isinstance(components, dict) else []
    )
    for key, value in export.items():
        if key in {"model", "components", "quant_types", "target_device", "model_name", "dtype"}:
            continue
        if isinstance(value, dict) and ("export_cfg" in value or key.endswith("_model")):
            component_names.append(key)
            sub_in, sub_out = _extract_io(value, key)
            submodels.append(
                {
                    "id": _sanitize(key),
                    "display_name": key,
                    "type": str(value.get("type", "")),
                    "precision": quant_types.get(key, ""),
                    "inputs": sub_in,
                    "outputs": sub_out,
                }
            )
    for key in dict.fromkeys(str(item) for item in component_names):
        if not any(item["id"] == _sanitize(key) for item in submodels):
            precision = quant_types.get(key, "")
            submodels.append(
                {
                    "id": _sanitize(key),
                    "display_name": key,
                    "type": "",
                    "precision": precision,
                    "inputs": [],
                    "outputs": [],
                }
            )
    if not inputs:
        inputs = [_io_item("input", "input")]
    if not outputs:
        outputs = [_io_item("output", "output")]
    submodels.insert(
        0,
        {
            "id": "model",
            "display_name": _display_name(config, config_path),
            "type": str(main.get("type", "")),
            "precision": overall,
            "inputs": deepcopy(inputs),
            "outputs": deepcopy(outputs),
        },
    )
    example_paths = sorted({str(path.relative_to(ROOT)) for path in example_dirs if path.exists()})
    readmes = [f"{item}/README.md" for item in example_paths if (ROOT / item / "README.md").is_file()]
    source_url = next((url.rstrip(".,") for url in urls if "huggingface.co/" in url or "modelscope.cn/" in url), "")
    source_name = str(
        _first(main, "model_id", "hf_model") or _first(export, "model_name") or _display_name({}, config_path)
    )
    provider = (
        "HuggingFace"
        if "huggingface.co/" in source_url or "huggingface" in source_name.lower()
        else "ModelScope"
        if "modelscope" in source_url or "modelscope" in source_name.lower()
        else "待补"
    )
    owner = _example_owner(_owner_example_dirs(config_path, example_dirs, examples_root)) or "未分配"
    warnings = []
    if not source_url:
        warnings.append("source.url requires manual metadata")
    if provider == "待补":
        warnings.append("source.provider requires manual metadata")
    if not submodels and ("other_models" in config_path or category == "audio_models"):
        warnings.append("submodel IO requires export metadata or manual metadata")
    version = f"{model_id}_{_sanitize(overall or Path(config_path).stem)}_draft"
    card = {
        "schema_version": 1,
        "model": {
            "id": model_id,
            "family": family,
            "display_name": _display_name(config, config_path),
            "modality": _infer_modality(config_path, example_dirs, config),
            "task": ["inference"],
            "tags": ["merak", "xh2a", category.replace("_models", "")],
        },
        "source": {
            "provider": provider,
            "name": source_name,
            "url": source_url,
            "license": "",
            "raw_model_path": "<model_dir>",
        },
        "workflow": {
            "config_path": config_path,
            "model_dir": "<model_dir>",
            "class": "auto",
            "actions": ["quant", "export", "dump_golden", "eval"],
            "category": category,
            "target_device": _first(export, "target_device") or _first(main, "chip_arch") or "XH2a",
            "precision": {"overall": overall, "components": quant_types},
        },
        "runtime": {"device": "cuda:0", "seed": 1024, "work_dir": f"work_dirs/merak_delivery/{model_id}"},
        "frontend": {
            "summary": f"自动发现的 {_display_name(config, config_path)} Merak workflow 卡片。",
            "inputs": inputs,
            "outputs": outputs,
            "submodels": submodels,
            "demo": {"kind": "metadata_only"},
            "limitations": ["自动生成卡片；来源、发版链接、精度实测和部分业务 IO 需要人工确认。"],
            "evidence": {"examples": example_paths, "readme": readmes},
        },
        "accuracy": [],
        "release": {
            "version_id": version,
            "target_status": "draft",
            "url": "",
            "owner": owner,
            "reviewer": "",
            "quant_models": [
                {
                    "name": _display_name(config, config_path),
                    "release_path": "",
                    "release_url": "",
                    "version": "",
                    "date": "",
                    "quant_type": overall,
                    "artifact_type": "hmonnx",
                    "status": "missing",
                }
            ],
        },
    }
    return card, warnings


def generate(args: argparse.Namespace) -> int:
    examples_root = (ROOT / args.examples_root).resolve()
    workflow_root = (ROOT / args.workflow_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    config_examples, example_urls = _example_metadata(examples_root)
    report_path = ROOT / args.report
    previous_generated_cards: set[Path] = set()
    if report_path.is_file():
        try:
            previous_report = json.loads(_read(report_path))
            previous_generated_cards = {
                (ROOT / item["card"]).resolve()
                for item in previous_report.get("cards", [])
                if isinstance(item, dict) and isinstance(item.get("card"), str)
            }
        except (json.JSONDecodeError, OSError):
            previous_generated_cards = set()
    referenced = set(config_examples)
    for path in workflow_root.rglob("*.yaml"):
        rel = str(path.relative_to(ROOT))
        if rel in referenced:
            continue
        # Only add configs with a matching example family directory.
        family = _config_family(rel)
        if any(family in str(item).replace(".", "_") for item in examples_root.rglob("README.md")):
            referenced.add(rel)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "cards": [], "warnings": []}
    grouped_configs: dict[str, list[str]] = defaultdict(list)
    for config_path in referenced:
        grouped_configs[_model_group_key(config_path)].append(config_path)

    for model_group in sorted(grouped_configs):
        config_path = min(grouped_configs[model_group], key=_config_priority)
        path = ROOT / config_path
        if not path.is_file():
            report["warnings"].append(f"missing workflow config: {config_path}")
            continue
        config = yaml.safe_load(_read(path)) or {}
        if not isinstance(config, dict):
            report["warnings"].append(f"invalid workflow mapping: {config_path}")
            continue
        dirs = config_examples.get(config_path, [])
        urls: list[str] = []
        for directory in dirs:
            key = str(directory.relative_to(examples_root))
            urls.extend(example_urls.get(key, []))
        card, warnings = _make_card(config_path, config, dirs, urls, examples_root)
        output = _card_output_path(card, config_path, dirs, examples_root, output_root)
        existing_path = output if output.is_file() else _legacy_card_path(card)
        if existing_path.is_file():
            try:
                existing = yaml.safe_load(_read(existing_path)) or {}
            except yaml.YAMLError:
                existing = {}
            if isinstance(existing, dict):
                _preserve_existing_fields(card, existing)
        if output.exists() and not args.overwrite:
            report["warnings"].append(f"skip existing card: {output.relative_to(ROOT)}")
        elif not args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                yaml.safe_dump(to_v2_model_card(card), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        report["cards"].append(
            {
                "model_id": card["model"]["id"],
                "config": config_path,
                "variants": sorted(grouped_configs[model_group]),
                "card": _relative_or_str(output),
                "warnings": warnings,
            }
        )
    if not args.dry_run:
        current_generated_cards = {(ROOT / item["card"]).resolve() for item in report["cards"]}
        for stale_card in sorted(previous_generated_cards - current_generated_cards):
            if stale_card.is_file() and stale_card.is_relative_to(output_root):
                stale_card.unlink()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"cards": len(report["cards"]), "warnings": len(report["warnings"]), "report": str(report_path)},
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", default="examples_merak")
    parser.add_argument("--workflow-root", default="configs_merak/workflows/xh2a")
    parser.add_argument("--output-root", default="examples_merak")
    parser.add_argument("--report", default="work_dirs/merak_delivery/card_generation_report.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
