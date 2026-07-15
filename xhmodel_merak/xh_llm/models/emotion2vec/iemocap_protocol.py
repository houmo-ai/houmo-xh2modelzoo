from __future__ import annotations

import re


IEMOCAP_LABELS = ("ang", "hap", "neu", "sad")
LABEL_TO_INDEX = {label: index for index, label in enumerate(IEMOCAP_LABELS)}
IEMOCAP_SESSION_COUNTS = (1085, 1023, 1151, 1031, 1241)


def session_from_utterance_id(utterance_id: str) -> str:
    match = re.match(r"^(Ses\d{2})", utterance_id)
    if not match:
        raise ValueError(f"Invalid IEMOCAP utterance id: {utterance_id!r}")
    return match.group(1)


def split_leave_one_session_out(sessions: list[str], leave_out: str):
    train = [session for session in sessions if session != leave_out]
    valid = [session for session in sessions if session == leave_out]
    return train, valid


def compute_metrics(targets: list[int], predictions: list[int], num_classes: int) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    total = len(targets)
    correct = sum(int(target == prediction) for target, prediction in zip(targets, predictions, strict=True))
    wa = correct / total if total else 0.0

    recalls = []
    f1s = []
    supports = []
    for cls_idx in range(num_classes):
        tp = sum(int(t == cls_idx and p == cls_idx) for t, p in zip(targets, predictions, strict=True))
        fn = sum(int(t == cls_idx and p != cls_idx) for t, p in zip(targets, predictions, strict=True))
        fp = sum(int(t != cls_idx and p == cls_idx) for t, p in zip(targets, predictions, strict=True))
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        supports.append(tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1s.append((2 * precision * recalls[-1] / (precision + recalls[-1])) if precision + recalls[-1] else 0.0)

    ua = sum(recalls) / num_classes if num_classes else 0.0
    support_total = sum(supports)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1s, supports, strict=True)) / max(support_total, 1)
    return {"wa": wa, "ua": ua, "weighted_f1": weighted_f1}


def map_labels(labels: list[str]) -> list[int]:
    return [LABEL_TO_INDEX[label] for label in labels]
