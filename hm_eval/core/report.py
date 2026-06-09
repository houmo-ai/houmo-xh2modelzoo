"""Report generation — produces detailed evaluation reports."""

from __future__ import annotations

import html as html_lib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_QUALITATIVE_SAMPLE_LIMIT = 3
_QUESTION_MARKER_RE = re.compile(r"(?im)^\s*(?:Question|问题)\s*:\s*")
_ANSWER_MARKER_RE = re.compile(r"(?im)^\s*(?:A|Answer|Assistant)\s*:\s*")
_MARKDOWN_DATA_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>data:image[^)]+)\)", re.IGNORECASE)
_RAW_DATA_IMAGE_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_QUESTION_PREVIEW_LIMIT = 1600
_ANSWER_PREVIEW_LIMIT = 2400
_FINAL_CHOICE_MARKERS = [
    "最终建议选择",
    "最终答案",
    "最终选项",
    "正确选项为",
    "正确选项是",
    "正确选项：",
    "正确选项",
    "正确答案是",
    "正确答案为",
    "正确答案：",
    "正确答案",
    "答案是",
    "答案为",
    "答案：",
    "应选",
    "故选",
    "因此选",
]
_FINAL_FILL_MARKERS = [
    "最终答案",
    "正确答案是",
    "正确答案为",
    "正确答案：",
    "正确答案",
    "答案是",
    "答案为",
    "答案：",
    "填空处应填",
    "应填",
]
_FINAL_JUDGEMENT_MARKERS = [
    "最终判断",
    "最终答案",
    "正确答案是",
    "正确答案为",
    "正确答案：",
    "正确答案",
    "答案是",
    "答案为",
    "答案：",
    "判断为",
]
_FINAL_CHOICE_RE = re.compile(
    r"(?:\s|：|:|为|是|选项|应该是|应为|为：|是：|\*|`|\(|（)*"
    r"([A-D]{1,4})"
    r"(?:\s|。|\.|，|,|；|;|\)|）|\*|`|$)"
)
_FINAL_CHOICE_BLOCK_RE = re.compile(
    r"(?:\s|：|:|为|是|选项|应该是|应为|为：|是：|\*|`|\(|（)*"
    r"(?P<choices>(?:[\(（\[]?\s*[A-D]\s*[\)）\]]?\s*(?:[、,，/和及与]|\s*)?){1,4})"
    r"(?:\s|。|\.|，|,|；|;|\)|）|\*|`|$)"
)
_FINAL_JUDGEMENT_RE = re.compile(
    r"(?:\s|：|:|为|是|\*|`|\(|（)*(正确|错误|对|错|是|否)"
    r"(?:\s|。|\.|，|,|；|;|\(|（|\)|）|\*|`|×|x|X|√|✓|$)"
)
_FINAL_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
_NUMERIC_TARGET_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
_ANSWER_EXPLANATION_SPLITS = ["解析：", "**解析", "解析:", "注：", "*注", "说明：", "理由："]


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _normalize_answer_text(value: Any) -> str:
    text = _as_text(value)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[`*_#$\\]", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[。\.，,；;：:、()（）\[\]【】\"“”'\-_/]", "", text)
    return text.strip().lower()


def _normalize_judgement(value: Any) -> str:
    text = _as_text(value).strip()
    if text in {"正确", "对", "是", "true", "True"}:
        return "对"
    if text in {"错误", "错", "否", "false", "False"}:
        return "错"
    return text


def _find_last_marker(text: str, markers: list[str]) -> tuple[int, Optional[str]]:
    best_index = -1
    best_marker = None
    for marker in markers:
        index = text.rfind(marker)
        if index > best_index:
            best_index = index
            best_marker = marker
    return best_index, best_marker


def _answer_fragment_after_last_marker(text: Any, markers: list[str], width: int = 220) -> tuple[str, Optional[str]]:
    source = _as_text(text)
    index, marker = _find_last_marker(source, markers)
    if marker is None:
        return "", None
    return source[index + len(marker): index + len(marker) + width].strip(), marker


def _extract_final_choice_answer(text: Any) -> tuple[Optional[str], Optional[str], str]:
    fragment, marker = _answer_fragment_after_last_marker(text, _FINAL_CHOICE_MARKERS, width=180)
    if not fragment:
        return None, marker, ""
    block_match = _FINAL_CHOICE_BLOCK_RE.search(fragment)
    if block_match:
        letters = "".join(dict.fromkeys(re.findall(r"[A-D]", block_match.group("choices").upper())))
        if letters:
            return letters, marker, fragment.replace("\n", " ")
    match = _FINAL_CHOICE_RE.search(fragment)
    if not match:
        return None, marker, fragment.replace("\n", " ")
    return match.group(1), marker, fragment.replace("\n", " ")


def _extract_final_judgement_answer(text: Any) -> tuple[Optional[str], Optional[str], str]:
    fragment, marker = _answer_fragment_after_last_marker(text, _FINAL_JUDGEMENT_MARKERS, width=140)
    if not fragment:
        return None, marker, ""
    match = _FINAL_JUDGEMENT_RE.search(fragment)
    if not match:
        return None, marker, fragment.replace("\n", " ")
    return _normalize_judgement(match.group(1)), marker, fragment.replace("\n", " ")


def _is_numeric_target(value: Any) -> bool:
    target = _as_text(value).replace(",", "").strip()
    return bool(_NUMERIC_TARGET_RE.fullmatch(target))


def _extract_final_fill_answer(text: Any, target: Any) -> tuple[Any, Optional[str], str]:
    fragment, marker = _answer_fragment_after_last_marker(text, _FINAL_FILL_MARKERS, width=260)
    if not fragment:
        return None, marker, ""
    for split_marker in _ANSWER_EXPLANATION_SPLITS:
        split_index = fragment.find(split_marker)
        if split_index >= 0:
            fragment = fragment[:split_index]
    fragment = fragment.strip()
    if _is_numeric_target(target):
        numbers = [number.replace(",", "") for number in _FINAL_NUMBER_RE.findall(fragment)]
        return numbers, marker, fragment.replace("\n", " ")

    candidates: list[str] = []
    plain_first_line = re.sub(r"\*+", "", fragment.split("\n", 1)[0]).strip()
    plain_first_line = re.sub(r"^[：:\s]+", "", plain_first_line)
    if _normalize_answer_text(plain_first_line) and len(plain_first_line) <= 160:
        for part in re.split(r"(?:或者|或|（|\(|）|\)|、|/|，|,|；|;)", plain_first_line):
            part = part.strip()
            if _normalize_answer_text(part) and len(part) <= 80:
                candidates.append(part)
    prefix = fragment.split("**", 1)[0].strip()
    prefix = re.sub(r"^[：:\s]+", "", prefix)
    if _normalize_answer_text(prefix) and len(prefix) <= 80:
        candidates.append(prefix)
    bold_answers = re.findall(r"\*\*\s*([^*\n]{1,80}?)\s*\*\*", fragment)
    for bold_answer in bold_answers:
        bold_answer = bold_answer.strip()
        if _normalize_answer_text(bold_answer) and not bold_answer.lstrip().startswith(("或", "（或", "(或")):
            candidates.append(bold_answer)
    if candidates:
        deduped_candidates = list(dict.fromkeys(candidates))
        return deduped_candidates, marker, fragment.replace("\n", " ")

    first_line = re.sub(r"^[：:\s]+", "", fragment.split("\n", 1)[0].strip())
    return first_line, marker, fragment.replace("\n", " ")


def _numeric_answer_equal(left: Any, right: Any) -> bool:
    try:
        left_value = float(_as_text(left).replace(",", "").rstrip("%"))
        right_value = float(_as_text(right).replace(",", "").rstrip("%"))
    except ValueError:
        return False
    return abs(left_value - right_value) <= 1e-6


def _fill_answer_matches(candidate: Any, expected: Any) -> bool:
    if _is_numeric_target(expected):
        if not isinstance(candidate, list):
            return False
        expected_numbers = [number.replace(",", "") for number in _FINAL_NUMBER_RE.findall(_as_text(expected))]
        return any(
            _numeric_answer_equal(candidate_number, expected_number)
            for candidate_number in candidate
            for expected_number in expected_numbers
        )

    if isinstance(candidate, list):
        return any(_fill_answer_matches(candidate_item, expected) for candidate_item in candidate)

    candidate_text = _normalize_answer_text(candidate)
    expected_text = _normalize_answer_text(expected)
    if not candidate_text or not expected_text:
        return False
    if candidate_text == expected_text:
        return True
    return len(candidate_text) <= len(expected_text) + 2 and expected_text in candidate_text


def _format_final_answer_display(final_answer: Any) -> str:
    if isinstance(final_answer, list):
        return ", ".join(_as_text(item) for item in final_answer)
    return _as_text(final_answer)


def _correct_with_final_answer(
    sample_type: str,
    expected: Any,
    prediction_text: Any,
    original_correct: Optional[bool],
) -> tuple[Optional[bool], dict[str, Any]]:
    if original_correct is None:
        return original_correct, {}

    final_answer: Any = None
    marker: Optional[str] = None
    fragment = ""
    corrected = original_correct
    mode = ""
    expected_text = _as_text(expected).strip()

    if sample_type == "选择" and re.fullmatch(r"[A-D]+", expected_text):
        final_answer, marker, fragment = _extract_final_choice_answer(prediction_text)
        if final_answer is None:
            return original_correct, {}
        corrected = final_answer == expected_text
        mode = "choice_final_answer"
    elif sample_type == "判断":
        final_answer, marker, fragment = _extract_final_judgement_answer(prediction_text)
        if final_answer is None:
            return original_correct, {}
        corrected = final_answer == _normalize_judgement(expected_text)
        mode = "judgement_final_answer"
    elif sample_type == "填空":
        final_answer, marker, fragment = _extract_final_fill_answer(prediction_text, expected_text)
        if final_answer is None:
            return original_correct, {}
        corrected = _fill_answer_matches(final_answer, expected_text)
        mode = "fill_final_answer"
    else:
        return original_correct, {}

    final_answer_matched = corrected
    preserved_original_correct = original_correct is True and not final_answer_matched
    if preserved_original_correct:
        corrected = True

    return corrected, {
        "mode": mode,
        "marker": marker or "",
        "fragment": fragment,
        "final_answer": final_answer,
        "final_answer_display": _format_final_answer_display(final_answer),
        "original_correct": original_correct,
        "final_answer_matched": final_answer_matched,
        "preserved_original_correct": preserved_original_correct,
        "corrected": corrected,
        "changed": corrected != original_correct,
    }


def _truncate_text(value: Any, limit: int) -> str:
    text = _as_text(value).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _format_expected_display(sample: Dict[str, Any]) -> str:
    expected = _as_text(sample.get("expected")).strip()
    if expected:
        return expected
    return "无字面标准答案（按规则判分）"


def _format_rule_display(sample: Dict[str, Any]) -> str:
    parts: list[str] = []
    judge_metric = _as_text(sample.get("judge_metric")).strip()
    judge_score = sample.get("judge_score")
    if judge_metric:
        if isinstance(judge_score, (int, float)):
            parts.append(f"{judge_metric}={judge_score:.1f}")
        else:
            parts.append(judge_metric)

    instruction_ids = sample.get("instruction_ids")
    if isinstance(instruction_ids, list):
        normalized_ids = [str(item) for item in instruction_ids if str(item).strip()]
        if normalized_ids:
            parts.append(f"约束: {', '.join(normalized_ids)}")

    return " | ".join(parts)


def _extract_display_question(raw_prompt: Any) -> str:
    prompt = _as_text(raw_prompt).strip()
    if not prompt:
        return ""

    prompt = re.sub(r"^\*\*User\*\*:\s*", "", prompt, count=1, flags=re.IGNORECASE).strip()
    question_markers = list(_QUESTION_MARKER_RE.finditer(prompt))
    if question_markers:
        prompt = prompt[question_markers[-1].start() :].strip()

    answer_marker = _ANSWER_MARKER_RE.search(prompt)
    if answer_marker:
        prompt = prompt[: answer_marker.start()].rstrip()

    prompt = _strip_embedded_data_images(prompt)
    return prompt


def _extract_embedded_data_images(text: str, limit: int = 6) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()

    for match in _MARKDOWN_DATA_IMAGE_RE.finditer(text):
        image_url = match.group("url")
        if image_url not in seen:
            seen.add(image_url)
            images.append(image_url)
            if len(images) >= limit:
                return images

    for match in _RAW_DATA_IMAGE_RE.finditer(text):
        image_url = match.group(0)
        if image_url not in seen:
            seen.add(image_url)
            images.append(image_url)
            if len(images) >= limit:
                return images

    return images


def _strip_embedded_data_images(text: str) -> str:
    image_index = 0

    def replace_markdown(match: re.Match[str]) -> str:
        nonlocal image_index
        image_index += 1
        alt_text = match.group("alt").strip() or f"图像{image_index}"
        return f"[{alt_text} 已提取显示]"

    cleaned = _MARKDOWN_DATA_IMAGE_RE.sub(replace_markdown, text)
    cleaned = _RAW_DATA_IMAGE_RE.sub("[图像数据已提取显示]", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _collect_multimodal_images(value: Any, limit: int = 6) -> list[str]:
    images: list[str] = []
    seen: set[str] = set()

    def append_image(image_url: str) -> None:
        if image_url in seen or len(images) >= limit:
            return
        seen.add(image_url)
        images.append(image_url)

    def visit(node: Any) -> None:
        if len(images) >= limit:
            return
        if isinstance(node, str):
            if node.startswith("data:image"):
                append_image(node)
                return
            for image_url in _extract_embedded_data_images(node, limit=max(limit - len(images), 0)):
                append_image(image_url)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)
                if len(images) >= limit:
                    return
            return
        if not isinstance(node, dict):
            return

        node_type = node.get("type")
        if node_type == "image" and isinstance(node.get("image"), str):
            visit(node["image"])
            return
        if node_type == "image_url" and isinstance(node.get("image_url"), dict):
            visit(node["image_url"].get("url"))
            return

        for key in ("content", "messages", "input", "raw_input", "origin_prompt", "sample_metadata", "metadata"):
            if key in node:
                visit(node[key])
                if len(images) >= limit:
                    return

    visit(value)
    return images[:limit]


def _ensure_report_shape(report: Dict[str, Any]) -> None:
    report.setdefault("summary", {})
    report.setdefault("details", {})
    report.setdefault("qualitative_analysis", {})


def _append_qualitative_sample(report: Dict[str, Any], dataset_name: str, sample: Dict[str, Any]) -> None:
    _ensure_report_shape(report)
    samples = report["qualitative_analysis"].setdefault(dataset_name, [])

    sample_key = (
        sample.get("subset"),
        sample.get("question_id"),
        sample.get("expected"),
        sample.get("predicted"),
    )
    for existing in samples:
        existing_key = (
            existing.get("subset"),
            existing.get("question_id"),
            existing.get("expected"),
            existing.get("predicted"),
        )
        if existing_key == sample_key:
            existing.update(sample)
            return

    if len(samples) >= _QUALITATIVE_SAMPLE_LIMIT:
        return

    samples.append(sample)


def _iter_qualitative_datasets(report: Dict[str, Any]) -> List[tuple[str, list[Dict[str, Any]]]]:
    qualitative = report.get("qualitative_analysis", {})
    if not isinstance(qualitative, dict):
        qualitative = {}

    ordered_dataset_names: list[str] = []
    for dataset_name in report.get("summary", {}).keys():
        if dataset_name not in ordered_dataset_names:
            ordered_dataset_names.append(dataset_name)
    for dataset_name in report.get("details", {}).keys():
        if dataset_name not in ordered_dataset_names:
            ordered_dataset_names.append(dataset_name)
    for dataset_name in sorted(qualitative.keys()):
        if dataset_name not in ordered_dataset_names:
            ordered_dataset_names.append(dataset_name)

    return [(dataset_name, qualitative.get(dataset_name, [])) for dataset_name in ordered_dataset_names]


def _extract_review_record(record: Dict[str, Any], default_question_id: int) -> Dict[str, Any]:
    sample_score = record.get("sample_score", {})
    sample_metadata = record.get("sample_metadata", {}) if isinstance(record.get("sample_metadata"), dict) else {}
    if not sample_metadata and isinstance(sample_score, dict) and isinstance(sample_score.get("sample_metadata"), dict):
        sample_metadata = sample_score.get("sample_metadata", {})
    score_info = sample_score.get("score", {}) if isinstance(sample_score, dict) else {}
    score_value = score_info.get("value", {}) if isinstance(score_info, dict) else {}
    main_score_name = score_info.get("main_score_name") if isinstance(score_info, dict) else None
    main_score_value = None
    if isinstance(main_score_name, str) and main_score_name:
        main_score_value = _as_float(score_value.get(main_score_name))

    is_correct_raw = record.get("is_correct", record.get("correct", None))
    if is_correct_raw is None:
        is_correct = None
        candidate_score_keys: list[str] = []
        if isinstance(main_score_name, str) and main_score_name:
            candidate_score_keys.append(main_score_name)
        candidate_score_keys.extend([
            "acc",
            "accuracy",
            "score",
            "exact_match",
            "pass@1",
            "prompt_level_strict",
            "inst_level_strict",
            "prompt_level_loose",
            "inst_level_loose",
        ])
        for score_key in candidate_score_keys:
            metric_value = _as_float(score_value.get(score_key))
            if metric_value is not None:
                is_correct = metric_value > 0
                break
    else:
        is_correct = bool(is_correct_raw)

    prediction_text = (
        record.get("model_output")
        or score_info.get("prediction", "")
        or record.get("prediction", "")
    )
    predicted = (
        record.get("predicted_answer")
        or score_info.get("extracted_prediction", "")
        or prediction_text
    )
    expected = record.get("correct_answer") or record.get("gold") or record.get("target", "")
    sample_type = _as_text(sample_metadata.get("type", ""))
    corrected_is_correct, postprocess_info = _correct_with_final_answer(
        sample_type,
        expected,
        prediction_text,
        is_correct,
    )
    if postprocess_info.get("final_answer_display"):
        predicted = postprocess_info["final_answer_display"]

    return {
        "question_id": record.get(
            "id",
            record.get(
                "question_id",
                sample_metadata.get("question_id", record.get("index", default_question_id)),
            ),
        ),
        "correct": corrected_is_correct,
        "original_correct": is_correct,
        "predicted": predicted,
        "expected": expected,
        "postprocess": postprocess_info,
        "judge_metric": main_score_name or "",
        "judge_score": main_score_value,
        "instruction_ids": sample_metadata.get("instruction_id_list", []),
        "prediction_text": prediction_text or _as_text(predicted),
        "question": _extract_display_question(record.get("question") or record.get("input") or record.get("prompt")),
        "images": _collect_multimodal_images(record),
    }


def generate_report(eval_results: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
    """Generate a structured report from evaluation results.

    Returns a dict containing:
        - summary: overall metrics per dataset
        - details: per-subset scores
        - metadata: run configuration
    """
    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "model": eval_results.get("model", ""),
        "backend": eval_results.get("backend", ""),
        "total_elapsed_seconds": eval_results.get("total_elapsed_seconds", 0),
        "summary": {},
        "details": {},
        "qualitative_analysis": {},
    }

    for ds_name, ds_result in eval_results.get("datasets", {}).items():
        if ds_result.get("status") != "completed":
            report["summary"][ds_name] = {
                "status": "failed",
                "error": ds_result.get("error", "unknown"),
            }
            continue

        # Summary: top-level metrics
        metrics = ds_result.get("metrics", {})
        report["summary"][ds_name] = {
            "status": "completed",
            "elapsed_seconds": ds_result.get("elapsed_seconds", 0),
            **metrics,
        }

        # Details: per-subset scores
        subset_scores = ds_result.get("subset_scores", {})
        if subset_scores:
            report["details"][ds_name] = subset_scores

    # Also try to extract from evalscope output directories
    enrich_report_from_outputs(report, work_dir)

    return report


def enrich_report_from_outputs(report: Dict[str, Any], work_dir: str) -> None:
    """Walk evalscope output dirs to find detailed per-subset results."""
    _ensure_report_shape(report)
    work_path = Path(work_dir)
    if not work_path.exists():
        return

    # Look for reviews directories (evalscope standard output)
    for reviews_dir in sorted(work_path.rglob("reviews")):
        if not reviews_dir.is_dir():
            continue
        for jsonl_file in sorted(reviews_dir.rglob("*.jsonl")):
            try:
                _parse_reviews_file(report, jsonl_file)
            except Exception:
                continue

    # Look for predictions directories
    for preds_dir in sorted(work_path.rglob("predictions")):
        if not preds_dir.is_dir():
            continue
        for jsonl_file in sorted(preds_dir.rglob("*.jsonl")):
            try:
                _parse_predictions_file(report, jsonl_file)
            except Exception:
                continue

    _sync_summary_from_details(report)


def _enrich_from_evalscope_outputs(report: Dict[str, Any], work_dir: str) -> None:
    enrich_report_from_outputs(report, work_dir)


def _parse_reviews_file(report: Dict[str, Any], jsonl_path: Path) -> None:
    """Parse an evalscope reviews .jsonl file for per-question accuracy."""
    raw_text = jsonl_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return
    lines = raw_text.splitlines()

    subset_name = jsonl_path.stem
    parent_dataset = _guess_dataset_from_path(jsonl_path)
    total = 0
    correct = 0
    correction_changed = 0
    correction_up = 0
    correction_down = 0
    wrong_samples = []

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed_record = _extract_review_record(record, total)
        is_correct = parsed_record["correct"]
        if is_correct is None:
            continue

        total += 1
        postprocess_info = parsed_record.get("postprocess")
        if isinstance(postprocess_info, dict) and postprocess_info.get("changed"):
            correction_changed += 1
            if postprocess_info.get("corrected") is True:
                correction_up += 1
            else:
                correction_down += 1
        if is_correct:
            correct += 1
            continue

        wrong_sample = {
            "subset": subset_name,
            **parsed_record,
        }
        if len(wrong_samples) < _QUALITATIVE_SAMPLE_LIMIT:
            wrong_samples.append(wrong_sample)
        if parent_dataset:
            _append_qualitative_sample(report, parent_dataset, wrong_sample)

    if total > 0:
        if parent_dataset:
            if parent_dataset not in report["details"]:
                report["details"][parent_dataset] = {}
            subset_report = {
                "accuracy": round(correct / total, 4),
                "correct": correct,
                "total": total,
            }
            if correction_changed:
                subset_report["postprocess_correction"] = {
                    "changed": correction_changed,
                    "up": correction_up,
                    "down": correction_down,
                    "rule": "final_explicit_answer",
                }
            if wrong_samples:
                subset_report["wrong_samples"] = wrong_samples
            report["details"][parent_dataset][subset_name] = subset_report


def _parse_predictions_file(report: Dict[str, Any], jsonl_path: Path) -> None:
    """Parse predictions for summary stats only."""
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return
    subset_name = jsonl_path.stem
    parent_dataset = _guess_dataset_from_path(jsonl_path)
    if parent_dataset and parent_dataset not in report["details"]:
        report["details"][parent_dataset] = {}
    if parent_dataset:
        report["details"][parent_dataset].setdefault(subset_name, {})["total_predictions"] = len(lines)


def _sync_summary_from_details(report: Dict[str, Any]) -> None:
    """Keep top-level summary scores consistent with parsed review details."""
    _ensure_report_shape(report)
    for dataset_name, subsets in report.get("details", {}).items():
        if not isinstance(subsets, dict):
            continue

        total = 0
        correct = 0
        changed = 0
        up = 0
        down = 0
        for subset_data in subsets.values():
            if not isinstance(subset_data, dict):
                continue
            subset_correct = subset_data.get("correct")
            subset_total = subset_data.get("total")
            if isinstance(subset_correct, int) and isinstance(subset_total, int):
                correct += subset_correct
                total += subset_total
            correction = subset_data.get("postprocess_correction")
            if isinstance(correction, dict):
                changed += int(correction.get("changed") or 0)
                up += int(correction.get("up") or 0)
                down += int(correction.get("down") or 0)

        if total <= 0:
            continue

        score = round(correct / total, 4)
        summary = report["summary"].setdefault(dataset_name, {"status": "completed"})
        summary.setdefault("status", "completed")
        score_keys = ("macro_acc", "accuracy", "score", "exact_match", "pass@1")
        for score_key in score_keys:
            if score_key in summary:
                summary[score_key] = score
        if not any(score_key in summary for score_key in score_keys):
            summary["score"] = score
        summary["correct"] = correct
        summary["total"] = total
        if changed:
            summary["postprocess_correction"] = {
                "changed": changed,
                "up": up,
                "down": down,
                "rule": "final_explicit_answer",
            }


def _guess_dataset_from_path(path: Path) -> Optional[str]:
    """Try to guess the dataset name from the file path."""
    parts = path.parts
    for ds in ["cmmmu", "mmmu_pro", "mmmu", "math_vision", "omnidoc_bench",
               "ceval", "cmmlu", "mmlu_pro", "mmlu", "arc", "hellaswag",
               "winogrande", "gsm8k", "math_500", "ifeval", "gpqa", "humaneval",
               "bbh", "truthfulqa"]:
        for part in parts:
            if ds in part.lower():
                return ds
    return None


def format_report_text(report: Dict[str, Any]) -> str:
    """Format the report as a human-readable text string."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Evaluation Report")
    lines.append(f"{'='*60}")
    lines.append(f"Model    : {report.get('model', 'N/A')}")
    lines.append(f"Backend  : {report.get('backend', 'N/A')}")
    lines.append(f"Time     : {report.get('generated_at', 'N/A')}")
    lines.append(f"Duration : {report.get('total_elapsed_seconds', 0):.1f}s")
    lines.append("")

    # Summary table
    lines.append(f"{'─'*60}")
    lines.append(f"{'Dataset':<20} {'Status':<12} {'Score':>10}")
    lines.append(f"{'─'*60}")

    for ds_name, ds_summary in report.get("summary", {}).items():
        status = ds_summary.get("status", "unknown")
        if status != "completed":
            lines.append(f"{ds_name:<20} {'FAILED':<12} {'N/A':>10}")
            continue
        # Find the best metric to show
        score = _best_score(ds_summary)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        lines.append(f"{ds_name:<20} {'OK':<12} {score_str:>10}")

    lines.append(f"{'─'*60}")
    lines.append("")

    # Detailed subset scores
    for ds_name, subsets in report.get("details", {}).items():
        if not subsets:
            continue
        lines.append(f"{'─'*60}")
        lines.append(f"  {ds_name} — Per-Subset Scores")
        lines.append(f"{'─'*60}")
        lines.append(f"  {'Subset':<35} {'Accuracy':>10} {'Correct':>8} {'Total':>6}")
        lines.append(f"  {'─'*55}")

        total_correct = 0
        total_count = 0
        for subset_name, subset_data in sorted(subsets.items()):
            if isinstance(subset_data, dict):
                acc = subset_data.get("accuracy", "N/A")
                cor = subset_data.get("correct", "")
                tot = subset_data.get("total", "")
                if isinstance(acc, (int, float)):
                    acc_str = f"{acc:.4f}"
                else:
                    acc_str = str(acc)
                lines.append(f"  {subset_name:<35} {acc_str:>10} {str(cor):>8} {str(tot):>6}")
                if isinstance(cor, int) and isinstance(tot, int):
                    total_correct += cor
                    total_count += tot

        if total_count > 0:
            macro_acc = total_correct / total_count
            lines.append(f"  {'─'*55}")
            lines.append(f"  {'MACRO AVERAGE':<35} {macro_acc:.4f} {total_correct:>8} {total_count:>6}")
        lines.append("")

    qualitative_sections = _iter_qualitative_datasets(report)
    if qualitative_sections:
        lines.append(f"{'─'*60}")
        lines.append("  定性分析（每个数据集最多展示 3 条错误样例）")
        lines.append(f"{'─'*60}")
        lines.append("")
        for ds_name, samples in qualitative_sections:
            lines.append(f"  [{ds_name}]")
            if not samples:
                lines.append("  0 条错误样例（当前数据集未发现可展示的错误题目）")
                lines.append("")
                continue
            for idx, sample in enumerate(samples[:_QUALITATIVE_SAMPLE_LIMIT], start=1):
                subset_name = sample.get("subset") or "N/A"
                question_id = sample.get("question_id", "N/A")
                expected_display = _format_expected_display(sample)
                rule_display = _format_rule_display(sample)
                lines.append(f"  {idx}. subset={subset_name} | question_id={question_id}")
                lines.append(f"     Expected: {expected_display}")
                if rule_display:
                    lines.append(f"     Judge: {rule_display}")
                image_count = len(sample.get("images") or [])
                if image_count:
                    lines.append(f"     Images: {image_count} 张（HTML 报告中可查看缩略图）")
                lines.append(f"     Predicted: {_as_text(sample.get('predicted')) or 'N/A'}")
                lines.append("     Question:")
                question_text = _truncate_text(sample.get("question"), _QUESTION_PREVIEW_LIMIT) or "N/A"
                lines.extend(f"       {line}" for line in question_text.splitlines())
                lines.append("     Model Answer:")
                answer_text = _truncate_text(sample.get("prediction_text"), _ANSWER_PREVIEW_LIMIT) or "N/A"
                lines.extend(f"       {line}" for line in answer_text.splitlines())
                lines.append("")

    return "\n".join(lines)


def format_report_html(report: Dict[str, Any]) -> str:
    """Format the report as an HTML table for Gradio display."""
    html = []
    html.append("<div style='font-family: monospace; padding: 10px;'>")

    # Header
    html.append(f"<h2>Evaluation Report</h2>")
    html.append(f"<p><b>Model:</b> {report.get('model', 'N/A')} | "
                f"<b>Backend:</b> {report.get('backend', 'N/A')} | "
                f"<b>Duration:</b> {report.get('total_elapsed_seconds', 0):.1f}s</p>")

    # Summary table
    html.append("<h3>Summary</h3>")
    html.append("<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>")
    html.append("<tr style='background: #f0f0f0;'><th>Dataset</th><th>Status</th><th>Score</th></tr>")
    for ds_name, ds_summary in report.get("summary", {}).items():
        status = ds_summary.get("status", "unknown")
        color = "#e8f5e9" if status == "completed" else "#ffebee"
        if status != "completed":
            html.append(f"<tr style='background: {color};'><td>{ds_name}</td><td>FAILED</td><td>N/A</td></tr>")
        else:
            score = _best_score(ds_summary)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            html.append(f"<tr style='background: {color};'><td>{ds_name}</td><td>OK</td><td>{score_str}</td></tr>")
    html.append("</table>")

    # Detail tables
    for ds_name, subsets in report.get("details", {}).items():
        if not subsets:
            continue
        html.append(f"<h3>{ds_name} — Per-Subset Scores</h3>")
        html.append("<table border='1' cellpadding='4' cellspacing='0' style='border-collapse: collapse;'>")
        html.append("<tr style='background: #f0f0f0;'>"
                    "<th>Subset</th><th>Accuracy</th><th>Correct</th><th>Total</th></tr>")

        total_correct = 0
        total_count = 0
        for subset_name, subset_data in sorted(subsets.items()):
            if isinstance(subset_data, dict):
                acc = subset_data.get("accuracy", "")
                cor = subset_data.get("correct", "")
                tot = subset_data.get("total", "")
                acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
                html.append(f"<tr><td>{subset_name}</td><td>{acc_str}</td><td>{cor}</td><td>{tot}</td></tr>")
                if isinstance(cor, int) and isinstance(tot, int):
                    total_correct += cor
                    total_count += tot

        if total_count > 0:
            macro_acc = total_correct / total_count
            html.append(f"<tr style='background: #e3f2fd; font-weight: bold;'>"
                        f"<td>MACRO AVG</td><td>{macro_acc:.4f}</td>"
                        f"<td>{total_correct}</td><td>{total_count}</td></tr>")
        html.append("</table>")

    qualitative_sections = _iter_qualitative_datasets(report)
    if qualitative_sections:
        html.append("<h3>定性分析</h3>")
        html.append("<p style='color:#6c757d;'>展开某个数据集，可查看最多 3 条回答错误的问题与模型回答。</p>")
        for ds_name, samples in qualitative_sections:
            escaped_dataset = html_lib.escape(ds_name)
            html.append(
                "<details style='margin: 12px 0 16px; border: 1px solid #dee2e6; border-radius: 10px; background: #fafafa;'>"
            )
            html.append(
                f"<summary style='cursor: pointer; padding: 12px 16px; font-weight: 700;'>{escaped_dataset}"
                f"（当前 {min(len(samples), _QUALITATIVE_SAMPLE_LIMIT)} 条错误样例）</summary>"
            )
            html.append("<div style='padding: 0 16px 16px;'>")
            if not samples:
                html.append(
                    "<p style='margin: 12px 0 0; color:#6c757d;'>"
                    "当前数据集未发现可展示的错误题目，通常意味着精度为 100% 或错误样例不足。"
                    "</p>"
                )
                html.append("</div>")
                html.append("</details>")
                continue
            for idx, sample in enumerate(samples[:_QUALITATIVE_SAMPLE_LIMIT], start=1):
                subset_name = html_lib.escape(_as_text(sample.get("subset")) or "N/A")
                question_id = html_lib.escape(_as_text(sample.get("question_id")) or "N/A")
                expected = html_lib.escape(_format_expected_display(sample))
                rule_display = html_lib.escape(_format_rule_display(sample))
                predicted = html_lib.escape(_as_text(sample.get("predicted")) or "N/A")
                image_urls = [img for img in (sample.get("images") or []) if isinstance(img, str) and img.startswith("data:image")]
                question_text = html_lib.escape(
                    _truncate_text(sample.get("question"), _QUESTION_PREVIEW_LIMIT) or "N/A"
                )
                prediction_text = html_lib.escape(
                    _truncate_text(sample.get("prediction_text"), _ANSWER_PREVIEW_LIMIT) or "N/A"
                )
                html.append(
                    "<div style='margin-top: 12px; padding: 12px; background: #ffffff; border: 1px solid #e9ecef; border-radius: 8px;'>"
                )
                html.append(
                    f"<div style='font-weight: 700; margin-bottom: 8px;'>样例 {idx}"
                    f" <span style='color:#6c757d; font-weight: 500;'>subset={subset_name} | question_id={question_id}</span>"
                    "</div>"
                )
                html.append(
                    f"<p style='margin: 6px 0;'><b>标准答案:</b> {expected} | <b>模型答案:</b> {predicted}</p>"
                )
                if rule_display:
                    html.append(
                        f"<p style='margin: 6px 0; color:#495057;'><b>判分规则:</b> {rule_display}</p>"
                    )
                if image_urls:
                    html.append(
                        "<div style='margin-top: 10px;'>"
                        "<div style='font-weight: 700; margin-bottom: 6px;'>图像</div>"
                        "<div style='display:flex; flex-wrap:wrap; gap:10px;'>"
                    )
                    for img_idx, image_url in enumerate(image_urls[:6], start=1):
                        safe_src = html_lib.escape(image_url, quote=True)
                        html.append(
                            f"<figure style='margin:0; border:1px solid #e9ecef; border-radius:6px; padding:6px; background:#fff;'>"
                            f"<img src='{safe_src}' alt='sample image {img_idx}' "
                            "style='display:block; max-width:260px; max-height:220px; object-fit:contain;'/>"
                            f"<figcaption style='font-size:12px; color:#6c757d; margin-top:4px;'>Image {img_idx}</figcaption>"
                            "</figure>"
                        )
                    html.append("</div></div>")
                html.append(
                    "<div style='margin-top: 10px;'>"
                    "<div style='font-weight: 700; margin-bottom: 6px;'>问题</div>"
                    f"<pre style='white-space: pre-wrap; word-break: break-word; margin: 0; padding: 10px; background: #f8f9fa; border-radius: 6px;'>{question_text}</pre>"
                    "</div>"
                )
                html.append(
                    "<div style='margin-top: 10px;'>"
                    "<div style='font-weight: 700; margin-bottom: 6px;'>模型完整回答</div>"
                    f"<pre style='white-space: pre-wrap; word-break: break-word; margin: 0; padding: 10px; background: #fff8e1; border-radius: 6px;'>{prediction_text}</pre>"
                    "</div>"
                )
                html.append("</div>")
            html.append("</div>")
            html.append("</details>")

    html.append("</div>")
    return "\n".join(html)


def _best_score(summary: Dict[str, Any]) -> Optional[float]:
    """Pick the most relevant score from a dataset summary."""
    for key in ["macro_acc", "accuracy", "score", "exact_match", "pass@1"]:
        val = summary.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def save_report(report: Dict[str, Any], output_dir: str) -> str:
    """Save report as JSON and text to the output directory. Returns the JSON path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    text_path = out / "report.txt"
    text_path.write_text(format_report_text(report), encoding="utf-8")

    return str(json_path)
