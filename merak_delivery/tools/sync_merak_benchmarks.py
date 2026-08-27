#!/usr/bin/env python3
"""Synchronize trustworthy benchmark metadata into Merak model cards.

The scanner never imports or runs benchmark modules. It only reads static source
metadata and records related scripts separately from verified accuracy results.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from model_card_schema import normalize_model_card, to_v2_model_card

TITLE_RE = re.compile(r"@allure\.title\(\s*['\"]([^'\"]+)['\"]\s*\)")
MODEL_RE = re.compile(r"['\"]--model['\"]\s*,\s*['\"]([^'\"]+)['\"]")
QUANT_RE = re.compile(r"['\"]--quant-type['\"]\s*,\s*['\"]([^'\"]+)['\"]")
RESULT_RE = re.compile(r"['\"]([^'\"]*eval_ppl[^'\"]*\.txt)['\"]", re.IGNORECASE)
THRESHOLD_RE = re.compile(r"assert\s+ppl_value\s*([<>]=?)\s*([0-9]+(?:\.[0-9]+)?)")
PARAM_RE = re.compile(r"@pytest\.mark\.parametrize\(\s*['\"](w_bit|a_bit)['\"]\s*,\s*\[([^]]+)]")
SEMANTIC_QUALIFIERS = {"embedding", "reranker", "asr", "tts", "forcealigner", "coder"}


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _active_source(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _basename(value: str) -> str:
    if not value or value.strip().startswith("<"):
        return ""
    return PurePosixPath(value.rstrip("/")).name


def _is_related(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or (min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)))


def _extract_bits(source: str) -> dict[str, list[int]]:
    bits: dict[str, list[int]] = {}
    for name, values in PARAM_RE.findall(source):
        parsed = [int(value) for value in re.findall(r"\d+", values)]
        if parsed:
            bits[name] = parsed
    return bits


def _result_value(root: Path, result_path: str) -> str | int | float:
    if not result_path:
        return ""
    path = Path(result_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return ""
    try:
        data = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return ""
    if isinstance(data, dict):
        value = data.get("wikitext ppl")
        if isinstance(value, (str, int, float)):
            return value
    return ""


def _scan_script(root: Path, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    source = _active_source(text)
    title_match = TITLE_RE.search(source)
    models = list(dict.fromkeys(MODEL_RE.findall(source)))
    quant_types = list(dict.fromkeys(QUANT_RE.findall(source)))
    result_match = RESULT_RE.search(source)
    result_path = result_match.group(1) if result_match else ""
    threshold_match = THRESHOLD_RE.search(source)
    threshold = f"PPL {threshold_match.group(1)} {threshold_match.group(2)}" if threshold_match else ""
    dataset = "WikiText" if "wikitext ppl" in source.lower() else ""
    metric = "PPL" if result_path or threshold else ""
    observed = _result_value(root, result_path)
    none_variables = re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*None\s*$", source)
    has_unavailable_guard = any(
        re.search(rf"if\s+{re.escape(name)}\s+is\s+None\s*:", source) for name in none_variables
    )
    if observed != "":
        run_status = "measured"
    elif has_unavailable_guard and "该模型未完成迁移" in source:
        run_status = "skipped"
    elif "TEST_ALL_MODEL" in source:
        run_status = "on_demand"
    elif not metric and ('acc = "导出成功"' in source or "acc = '导出成功'" in source or "# TODO" in text):
        run_status = "export_only"
    else:
        run_status = "implemented"
    return {
        "name": title_match.group(1) if title_match else path.stem,
        "script": str(path.relative_to(root)),
        "script_key": _normalize(path.stem.removeprefix("benchmark_test_")),
        "model_paths": models,
        "quant_types": quant_types,
        "bits": _extract_bits(source),
        "dataset": dataset,
        "metric": metric,
        "threshold": threshold,
        "result_path": result_path,
        "observed": observed,
        "run_status": run_status,
    }


def _card_ids(card: dict[str, Any]) -> tuple[set[str], set[str]]:
    model = card.get("model", {})
    source = card.get("source", {})
    workflow = card.get("workflow", {})
    general = {
        _normalize(str(model.get("id", ""))),
        _normalize(str(model.get("display_name", ""))),
        _normalize(str(source.get("name", ""))),
    }
    specific = {
        _normalize(_basename(str(source.get("raw_model_path", "")))),
        _normalize(_basename(str(workflow.get("model_dir", "")))),
    }
    general.discard("")
    specific.discard("")
    return general, specific


def _match_case(card: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any] | None:
    general_ids, specific_ids = _card_ids(card)
    script_key = benchmark["script_key"]
    model_ids = {_normalize(_basename(path)) for path in benchmark["model_paths"]}
    model_ids.discard("")
    model_id = _normalize(str(card.get("model", {}).get("id", "")))
    benchmark_identity = script_key + "".join(sorted(model_ids))
    required_qualifiers = {qualifier for qualifier in SEMANTIC_QUALIFIERS if qualifier in model_id}
    if any(qualifier not in benchmark_identity for qualifier in required_qualifiers):
        return None
    candidate = any(_is_related(script_key, identifier) for identifier in general_ids | specific_ids)
    candidate = candidate or any(
        _is_related(model_id, identifier) for model_id in model_ids for identifier in general_ids | specific_ids
    )
    if not candidate:
        return None

    comparison_ids = specific_ids or general_ids
    if model_ids and any(model_id == identifier for model_id in model_ids for identifier in comparison_ids):
        match_status = "exact"
    elif model_ids and any(
        _is_related(model_id, identifier) for model_id in model_ids for identifier in comparison_ids
    ):
        match_status = "variant"
    elif model_ids:
        match_status = "mismatch"
    elif any(script_key == identifier for identifier in comparison_ids):
        match_status = "exact"
    else:
        match_status = "variant"

    card_precision = str(card.get("workflow", {}).get("precision", {}).get("overall", ""))
    precision_mismatch = bool(
        benchmark["quant_types"]
        and card_precision
        and all(_normalize(item) != _normalize(card_precision) for item in benchmark["quant_types"])
    )
    one_layer_variant = any("1layer" in model_id for model_id in model_ids)
    if precision_mismatch or one_layer_variant:
        match_status = "mismatch"

    notes = []
    if model_ids and match_status == "mismatch":
        notes.append("脚本实际模型与当前模型卡版本不一致")
    if precision_mismatch:
        notes.append(f"脚本量化配置 {', '.join(benchmark['quant_types'])} 与模型卡 {card_precision} 不一致")
    if benchmark["run_status"] == "on_demand":
        notes.append("默认跳过，需设置 TEST_ALL_MODEL=true 才执行")
    if benchmark["run_status"] == "export_only":
        notes.append("当前仅验证导出，尚未实现精度评测")
    if benchmark["run_status"] == "skipped":
        notes.append("脚本标记模型未完成迁移并提前返回")
    if benchmark["result_path"] and benchmark["observed"] == "":
        notes.append("仓库中未找到脚本引用的结果文件")

    return {
        "name": benchmark["name"],
        "script": benchmark["script"],
        "match_status": match_status,
        "run_status": benchmark["run_status"],
        "model_path": benchmark["model_paths"][0] if benchmark["model_paths"] else "",
        "quant_type": ", ".join(benchmark["quant_types"]),
        "weight_bits": benchmark["bits"].get("w_bit", []),
        "activation_bits": benchmark["bits"].get("a_bit", []),
        "dataset": benchmark["dataset"],
        "metric": benchmark["metric"],
        "threshold": benchmark["threshold"],
        "result_path": benchmark["result_path"],
        "observed": benchmark["observed"],
        "note": "；".join(notes),
    }


def _benchmark_summary(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any(case["observed"] != "" and case["match_status"] == "exact" for case in cases):
        return "measured", "已找到与当前模型精确匹配的 benchmark 实测结果。"
    if any(case["match_status"] == "exact" for case in cases):
        return "covered", "已找到与当前模型匹配的 benchmark 脚本，但仓库中没有可复制的实测结果。"
    if cases:
        return "mismatch", "检测到关联 benchmark 脚本，但模型版本或量化配置不匹配，不能作为当前卡片精度证据。"
    return "missing", "未找到与当前模型匹配的 benchmark_test 脚本。"


def _discover_model_cards(card_root: Path) -> list[Path]:
    if card_root.is_file():
        return [card_root]
    return sorted(path for path in card_root.rglob("*.yaml") if path.is_file() and path.parent.name == "model_cards")


def sync(root: Path = ROOT, card_root: Path | None = None, benchmark_root: Path | None = None) -> dict[str, int]:
    card_root = card_root or root / "examples_merak"
    benchmark_root = benchmark_root or root / "benchmark_test"
    if not card_root.is_absolute():
        card_root = root / card_root
    if not benchmark_root.is_absolute():
        benchmark_root = root / benchmark_root
    benchmarks = [_scan_script(root, path) for path in sorted(benchmark_root.glob("benchmark_test_*.py"))]
    updated = 0
    matched_cases = 0
    for card_path in _discover_model_cards(card_root):
        card = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
        normalized_card = normalize_model_card(card)
        cases = [case for benchmark in benchmarks if (case := _match_case(normalized_card, benchmark)) is not None]
        if not cases:
            if isinstance(normalized_card.get("benchmark"), dict) and normalized_card["benchmark"].get("source") == "benchmark_test":
                normalized_card.pop("benchmark")
            else:
                continue
        else:
            status, summary = _benchmark_summary(cases)
            normalized_card["benchmark"] = {
                "source": "benchmark_test",
                "status": status,
                "summary": summary,
                "cases": cases,
            }
            matched_cases += len(cases)
        card_path.write_text(
            yaml.safe_dump(to_v2_model_card(normalized_card), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        updated += 1
    return {"cards_updated": updated, "matched_cases": matched_cases, "scripts_scanned": len(benchmarks)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    args = parser.parse_args()
    result = sync(ROOT, args.card_root, args.benchmark_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
