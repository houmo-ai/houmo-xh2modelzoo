"""Unified output normalization for all evalscope benchmarks.

Consolidates the 6+ duplicate implementations across examples/ into a single module.
Supports ceval, mmlu_pro, cmmlu, arc, hellaswag, winogrande, gsm8k, math_500, ifeval,
gpqa, humaneval, and wikitext with appropriate normalization per dataset.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Choice letter extraction (A-J range, covers mmlu_pro's 10 options)
# ---------------------------------------------------------------------------

CHOICE_PATTERNS: list[str] = [
    r"答案\s*[:：]\s*([A-J])",
    r"正确答案\s*[:：]?\s*([A-J])",
    r"(?i)answer\s*[:：]\s*([A-J])",
    r"(?i)the answer is\s*[:：]?\s*([A-J])",
    r"(?i)final answer\s*[:：]?\s*([A-J])",
    r"(?i)option\s*([A-J])",
    r"(?i)choice\s*([A-J])",
    r"选项\s*([A-J])",
    r"故选\s*([A-J])",
    r"^[\s\*\-\(\[]*([A-J])[\)\]\s\.:：]*$",
]


def extract_single_choice_letter(text: str) -> Optional[str]:
    """Extract a single choice letter (A-J) from model output text."""
    if not text:
        return None
    for pattern in CHOICE_PATTERNS:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).upper()
    standalone = re.findall(r"\b([A-J])\b", text.upper())
    if standalone:
        return standalone[-1]
    return None


# ---------------------------------------------------------------------------
# Numeric answer extraction (for gsm8k / math)
# ---------------------------------------------------------------------------

_NUMERIC_PATTERNS: list[str] = [
    r"(?i)(?:the answer is|答案是)\s*[:：]?\s*([\-]?\d[\d,]*\.?\d*)",
    r"####\s*([\-]?\d[\d,]*\.?\d*)",
    r"(?i)(?:=|equals)\s*([\-]?\d[\d,]*\.?\d*)\s*$",
    r"([\-]?\d[\d,]*\.?\d*)\s*$",
]


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract a numeric answer from model output (for gsm8k, math benchmarks)."""
    if not text:
        return None
    for pattern in _NUMERIC_PATTERNS:
        match = re.search(pattern, text.strip(), flags=re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")
    return None


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------

def strip_think_content(text: Optional[str]) -> str:
    """Remove <think>...</think> blocks and <|im_end|> tokens."""
    if text is None:
        return ""
    cleaned = text.replace("<|im_end|>", "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Per-dataset system messages
# ---------------------------------------------------------------------------

DATASET_SYSTEM_MESSAGES: dict[str, str] = {
    "ceval": "请直接输出最终选项，不要输出解析。最后一行严格写成：答案：A",
    "cmmlu": "请直接输出最终选项，不要输出解析。最后一行严格写成：答案：A",
    "mmlu_pro": "Return only the final option. The last line must be exactly in the format: Answer: A",
    "arc": "Return only the final option letter. The last line must be exactly: Answer: A",
    "hellaswag": "Return only the final option letter. The last line must be exactly: Answer: A",
    "winogrande": "Return only the final option letter. The last line must be exactly: Answer: A",
    "gsm8k": "Solve the problem step by step. The last line must contain only the final numeric answer after ####.",
    "aime24": "Solve the problem step by step. The last line must contain only the final numeric answer after ####.",
    "aime25": "Solve the problem step by step. The last line must contain only the final numeric answer after ####.",
    "gpqa": "Return only the final option. The last line must be exactly in the format: Answer: A",
}


def get_system_message(dataset_name: str) -> Optional[str]:
    """Return the system message for a given dataset."""
    exact_names = {"ceval", "cmmlu", "mmlu_pro"}
    base_name = dataset_name if dataset_name in exact_names else dataset_name.split("_")[0]
    for key in [dataset_name, base_name]:
        if key in DATASET_SYSTEM_MESSAGES:
            return DATASET_SYSTEM_MESSAGES[key]
    return None


# ---------------------------------------------------------------------------
# Prompt rewriting (dataset-specific)
# ---------------------------------------------------------------------------

def rewrite_messages_for_dataset(
    messages: list[dict[str, str]],
    dataset_name: str,
) -> list[dict[str, str]]:
    """Rewrite messages for datasets that need stricter answer-only prompts."""
    if dataset_name not in {"ceval", "cmmlu", "mmlu_pro"}:
        return messages

    rewritten: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) and dataset_name == "mmlu_pro":
            content = content.replace(
                "Think step by step before answering.",
                "Do not explain. Return only the final answer.",
            )
            content = content.replace(
                "The last line of your response should be of the following format: "
                "'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J.",
                "Return exactly one line in the format: Answer: A",
            )
        elif isinstance(content, str) and dataset_name in {"ceval", "cmmlu"}:
            strict_suffix = "请不要解释，不要列步骤，只输出一行：答案：A/B/C/D。"
            if strict_suffix not in content:
                content = f"{content.rstrip()}\n\n{strict_suffix}"
        rewritten.append({**message, "content": content})
    return rewritten


# ---------------------------------------------------------------------------
# Main normalization entry point
# ---------------------------------------------------------------------------

# Datasets that use multiple-choice letter normalization
_CHOICE_DATASETS = {"ceval", "cmmlu", "mmlu_pro", "arc", "hellaswag", "winogrande", "gpqa"}
# Datasets that use Chinese-style answer formatting
_CHINESE_DATASETS = {"ceval", "cmmlu"}
# Datasets that use numeric answer extraction
_NUMERIC_DATASETS = {"gsm8k", "math_500", "math", "aime24", "aime25"}


def normalize_eval_output(
    output_text: str,
    dataset_name: str,
    prompt_text: Optional[str] = None,
) -> str:
    """Normalize model output for evalscope scoring.

    Args:
        output_text: Raw model output text (already think-stripped).
        dataset_name: Name of the evalscope dataset (e.g. "ceval", "mmlu_pro").
        prompt_text: Optional prompt text for context-aware normalization.

    Returns:
        Normalized output suitable for evalscope scoring.
    """
    if not output_text:
        return output_text

    # Strip thinking content first
    output_text = strip_think_content(output_text)

    # Numeric datasets
    if dataset_name in _NUMERIC_DATASETS:
        num = extract_numeric_answer(output_text)
        if num is not None:
            return f"#### {num}"
        return output_text

    # Choice-based datasets
    if dataset_name in _CHOICE_DATASETS:
        letter = extract_single_choice_letter(output_text)
        if letter is None:
            return output_text
        if dataset_name in _CHINESE_DATASETS:
            return f"答案：{letter}"
        return f"Answer: {letter}"

    # All other datasets: return as-is (ifeval, humaneval, wikitext, etc.)
    return output_text
