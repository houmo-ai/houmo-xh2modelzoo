#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_reports(
    report_dir: Path,
    *,
    include_running: bool = False,
    tag_regex: str | None = None,
    exclude_regex: str | None = None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    tag_re = re.compile(tag_regex) if tag_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None
    for path in sorted(report_dir.glob("*.json")):
        stem = path.stem
        if tag_re and not tag_re.search(stem):
            continue
        if exclude_re and exclude_re.search(stem):
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not include_running and data.get("status") != "completed":
            continue
        data["_path"] = path
        data["_tag"] = stem
        reports.append(data)
    return reports


def _merged_tag(report: dict[str, Any]) -> str:
    stem = report["_path"].stem
    return re.sub(r"[._-]?shard\d+of\d+$", "", stem)


def _is_sharded_report(report: dict[str, Any]) -> bool:
    stem = report["_path"].stem
    return int(report.get("num_shards", 1) or 1) > 1 or bool(
        re.search(r"[._-]?shard\d+of\d+$", stem)
    )


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        category = row.get("category")
        if category:
            buckets[category].append(row)

    overall_metrics: dict[str, list[float]] = defaultdict(list)
    per_category: dict[str, dict[str, Any]] = {}
    for category, items in sorted(buckets.items()):
        sums: dict[str, list[float]] = defaultdict(list)
        for item in items:
            spec = item.get("spec", {})
            for key in (
                "target_prefill_calls",
                "target_decoder_calls",
                "mtp_prefill_calls",
                "mtp_decode_calls",
                "dflash_prefill_calls",
                "dflash_decode_calls",
                "output_tokens",
                "overall_acceptance_rate",
                "avg_accepted_per_round",
                "num_rounds",
            ):
                if key in spec and spec[key] is not None:
                    sums[f"spec_{key}"].append(float(spec[key]))
        means = {key: (sum(values) / len(values)) for key, values in sums.items() if values}
        per_category[category] = {"n": len(items), **means}
        for key, value in means.items():
            overall_metrics[key].append(value)

    overall = {
        key: (sum(values) / len(values))
        for key, values in overall_metrics.items()
        if values
    }
    return {"per_category": per_category, "overall": overall}


def _pick_examples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        category = row.get("category")
        if category and category not in chosen:
            chosen[category] = row
    return chosen


def _report_think_modes(report: dict[str, Any]) -> list[str]:
    modes = report.get("think_modes")
    if modes:
        return list(modes)
    done = report.get("think_modes_done")
    if done:
        return list(done)
    return sorted(report.get("rows_per_mode", {}).keys())


def _summary_for_mode(report: dict[str, Any], mode: str) -> dict[str, Any]:
    summary = report.get("summary_per_mode", {}).get(mode)
    if summary is not None:
        return summary
    rows = report.get("rows_per_mode", {}).get(mode, [])
    return _summarise(rows) if rows else {}


def _examples_for_mode(report: dict[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    examples = report.get("examples_per_mode", {}).get(mode)
    if examples is not None:
        return examples
    rows = report.get("rows_per_mode", {}).get(mode, [])
    return _pick_examples(rows) if rows else {}


def _merge_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for report in reports:
        if not _is_sharded_report(report):
            merged.append(report)
            continue
        key = (
            report.get("meta_path", report["_tag"]),
            report.get("dataset", ""),
            int(report.get("num_shards", 1) or 1),
        )
        grouped.setdefault(key, []).append(report)

    for group in grouped.values():
        first = group[0]
        per_mode_rows: dict[str, list[dict[str, Any]]] = {}
        think_modes = sorted(
            {
                mode
                for report in group
                for mode in report.get("rows_per_mode", {}).keys()
            }
        )
        for mode in think_modes:
            rows: list[dict[str, Any]] = []
            for report in sorted(group, key=lambda r: int(r.get("shard_index", 0) or 0)):
                rows.extend(report.get("rows_per_mode", {}).get(mode, []))
            rows.sort(key=lambda row: row.get("id", ""))
            per_mode_rows[mode] = rows

        merged.append(
            {
                "_path": first["_path"],
                "_paths": [report["_path"] for report in group],
                "_tag": _merged_tag(first),
                "meta_path": first.get("meta_path"),
                "dataset": first.get("dataset"),
                "num_shards": int(first.get("num_shards", 1) or 1),
                "think_modes": think_modes,
                "rows_per_mode": per_mode_rows,
                "summary_per_mode": {
                    mode: _summarise(rows) for mode, rows in per_mode_rows.items()
                },
                "examples_per_mode": {
                    mode: _pick_examples(rows) for mode, rows in per_mode_rows.items()
                },
            }
        )
    return sorted(merged, key=lambda report: report["_tag"])


def _render_summary_table(reports: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Overall summary",
        "",
        "| model | think | spec_target_decoder | mtp_prefill | mtp_decode | dflash_prefill | dflash_decode | accept_rate | avg_accept/round | output_tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        tag = report["_tag"]
        for mode in _report_think_modes(report):
            overall = _summary_for_mode(report, mode).get("overall", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        tag,
                        mode,
                        _fmt_metric(overall.get("spec_target_decoder_calls")),
                        _fmt_metric(overall.get("spec_mtp_prefill_calls")),
                        _fmt_metric(overall.get("spec_mtp_decode_calls")),
                        _fmt_metric(overall.get("spec_dflash_prefill_calls")),
                        _fmt_metric(overall.get("spec_dflash_decode_calls")),
                        _fmt_metric(overall.get("spec_overall_acceptance_rate")),
                        _fmt_metric(overall.get("spec_avg_accepted_per_round")),
                        _fmt_metric(overall.get("spec_output_tokens")),
                    ]
                )
                + " |"
            )
    lines.append("")
    return lines


def _render_category_sections(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = ["## Per-model details", ""]
    for report in reports:
        tag = report["_tag"]
        lines.append(f"### {tag}")
        lines.append("")
        lines.append(f"- meta: `{report.get('meta_path')}`")
        if report.get("_paths"):
            lines.append("- source json:")
            for path in report["_paths"]:
                lines.append(f"  - `{path}`")
        else:
            lines.append(f"- source json: `{report['_path']}`")
        lines.append("")
        for mode in _report_think_modes(report):
            lines.append(f"#### think={mode}")
            lines.append("")
            per_cat = _summary_for_mode(report, mode).get("per_category", {})
            if per_cat:
                lines.append("| category | n | spec_target_decoder | mtp_decode | dflash_decode | accept_rate | avg_accept/round |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                for cat, stats in per_cat.items():
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                cat,
                                _fmt_metric(stats.get("n")),
                                _fmt_metric(stats.get("spec_target_decoder_calls")),
                                _fmt_metric(stats.get("spec_mtp_decode_calls")),
                                _fmt_metric(stats.get("spec_dflash_decode_calls")),
                                _fmt_metric(stats.get("spec_overall_acceptance_rate")),
                                _fmt_metric(stats.get("spec_avg_accepted_per_round")),
                            ]
                        )
                        + " |"
                    )
                lines.append("")

            chosen = _examples_for_mode(report, mode)
            if chosen:
                lines.append("Representative cases:")
                lines.append("")
                for cat, row in chosen.items():
                    spec = row.get("spec", {})
                    lines.append(f"##### {cat} — `{row.get('id', '')}` (len={row.get('char_length')})")
                    lines.append("")
                    lines.append("**Prompt**")
                    lines.append("")
                    lines.append("```text")
                    lines.append(row.get("prompt", ""))
                    lines.append("```")
                    lines.append("")
                    draft_extra = ""
                    if "mtp_decode_calls" in spec:
                        draft_extra = (
                            f", mtp_prefill={spec.get('mtp_prefill_calls')}, "
                            f"mtp_decode={spec.get('mtp_decode_calls')}"
                        )
                    elif "dflash_decode_calls" in spec:
                        draft_extra = (
                            f", dflash_prefill={spec.get('dflash_prefill_calls')}, "
                            f"dflash_decode={spec.get('dflash_decode_calls')}"
                        )
                    accept_rate = spec.get("overall_acceptance_rate")
                    accept_str = "-" if accept_rate is None else f"{accept_rate:.3f}"
                    lines.append(
                        f"- spec: target_decoder_calls={spec.get('target_decoder_calls')}, "
                        f"output_tokens={spec.get('output_tokens')}, "
                        f"overall_accept_rate={accept_str}{draft_extra}"
                    )
                    lines.append("")
                    lines.append("<details><summary>spec output</summary>")
                    lines.append("")
                    lines.append("```text")
                    lines.append(spec.get("text", ""))
                    lines.append("```")
                    lines.append("</details>")
                    lines.append("")
                lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, help="directory containing bench json files")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--include-running", action="store_true", help="include reports whose status is not completed")
    parser.add_argument("--tag-regex", default=None, help="only include reports whose filename stem matches this regex")
    parser.add_argument("--exclude-regex", default=None, help="exclude reports whose filename stem matches this regex")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    reports = _merge_reports(
        _load_reports(
            report_dir,
            include_running=args.include_running,
            tag_regex=args.tag_regex,
            exclude_regex=args.exclude_regex,
        )
    )
    if not reports:
        raise SystemExit(f"No json reports found under {report_dir}")

    lines = ["# Qwen3.5 spec-decode aggregate report", ""]
    lines.extend(_render_summary_table(reports))
    lines.extend(_render_category_sections(reports))

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
