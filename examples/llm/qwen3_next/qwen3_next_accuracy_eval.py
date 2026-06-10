#!/usr/bin/env python3
"""Qwen3-Next accuracy evaluation for smoke and full-suite runs.

Supports both HuggingFace floating-point and exported HMONNX backends for:
- WikiText-2 perplexity (PPL)
- CEval multiple-choice accuracy
- MMLU multiple-choice accuracy

The default limits are intentionally small so the 80B model path can be
validated before longer full-suite runs. Pass ``--limit -1`` to evaluate
the full CEVAL/MMLU split selected by ``--ceval-subsets`` or
``--mmlu-subjects``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


# Allow running from repository root or directly from this directory.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

# ruff: noqa: E402
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    from _runtime import load_runtime_from_meta, parse_dtype
except Exception:  # pragma: no cover - only used for hmonnx backend
    load_runtime_from_meta = None
    parse_dtype = None


CEVAL_SUBSETS = [
    "advanced_mathematics",
    "computer_network",
    "computer_architecture",
    "electrical_engineer",
    "high_school_chemistry",
    "operating_system",
    "high_school_physics",
    "law",
    "marxism",
]

MMLU_SUBJECTS = [
    "all",
]


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve multiline examples while still showing argument defaults."""


@dataclass
class ChoiceExample:
    dataset: str
    subset: str
    index: int
    question: str
    choices: Dict[str, str]
    gold: str
    pred: str
    scores: Dict[str, float]
    correct: bool


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_torch_dtype(name: str):
    mapping = {
        "auto": "auto",
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(mapping)}")
    return mapping[key]


def first_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_fp_backend(
    model_dir: str,
    dtype: str,
    device_map: str,
    experts_implementation: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config_kwargs = {}
    if experts_implementation:
        config_kwargs["experts_implementation"] = experts_implementation
        config_kwargs["_experts_implementation"] = experts_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=parse_torch_dtype(dtype),
        device_map=device_map,
        **config_kwargs,
    ).eval()
    if experts_implementation:
        for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
            if cfg is not None:
                setattr(cfg, "_experts_implementation", experts_implementation)
    return FPBackend(model=model, tokenizer=tokenizer, model_dir=model_dir)


class FPBackend:
    def __init__(self, model, tokenizer, model_dir: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_dir = model_dir

    @property
    def device(self) -> torch.device:
        return first_model_device(self.model)

    @torch.no_grad()
    def logits(self, input_ids: torch.LongTensor):
        input_ids = input_ids.to(self.device)
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: Sequence[str]) -> Dict[str, float]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        try:
            out = self.model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        except TypeError:
            out = self.model(input_ids=input_ids, use_cache=False)
        next_logits = out.logits[0, -1, :].to(torch.float32)
        scores: Dict[str, float] = {}
        for choice in choices:
            token_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Choice {choice!r} is not a single token: {token_ids}")
            scores[choice] = float(next_logits[int(token_ids[0])].item())
        return scores

    @torch.no_grad()
    def score_choices_batch(self, prompts: Sequence[str], choices: Sequence[str]) -> List[Dict[str, float]]:
        if not prompts:
            return []
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = original_padding_side
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        try:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        except TypeError:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        next_logits = out.logits[:, -1, :].to(torch.float32)

        choice_token_ids = {}
        for choice in choices:
            token_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Choice {choice!r} is not a single token: {token_ids}")
            choice_token_ids[choice] = int(token_ids[0])
        return [
            {choice: float(row[token_id].item()) for choice, token_id in choice_token_ids.items()}
            for row in next_logits
        ]


class HMONNXBackend:
    def __init__(
        self,
        meta_path: str,
        dtype: str,
        device: str,
        exec_device: str,
        resource_tight_mode: bool,
        enable_cuda_graph: bool = False,
        max_memory_gb: float = 0.0,
    ) -> None:
        if load_runtime_from_meta is None or parse_dtype is None:
            raise RuntimeError("Cannot import qwen3_next _runtime helpers")
        auto_offload_max_memory = None
        if max_memory_gb > 0:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if visible_devices:
                gpu_count = len([item for item in visible_devices.split(",") if item.strip()])
            else:
                gpu_count = torch.cuda.device_count()
            memory_bytes = int(max_memory_gb * 1024**3)
            auto_offload_max_memory = {idx: memory_bytes for idx in range(gpu_count)}
        runtime_kwargs = dict(
            meta_path=meta_path,
            dtype=parse_dtype(dtype),
            device=device,
            exec_device=exec_device,
            auto_offload=True,
            auto_offload_max_memory=auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
        )
        # Newer runtimes may accept enable_cuda_graph; older ones do not.
        if enable_cuda_graph:
            try:
                self.runtime, self.tokenizer, self.meta = load_runtime_from_meta(
                    **runtime_kwargs,
                    enable_cuda_graph=True,  # type: ignore[arg-type]
                )
            except TypeError:
                self.runtime, self.tokenizer, self.meta = load_runtime_from_meta(**runtime_kwargs)
                self.runtime.enable_cuda_graph_requested = True
        else:
            self.runtime, self.tokenizer, self.meta = load_runtime_from_meta(**runtime_kwargs)

    @property
    def device(self) -> torch.device:
        return self.runtime.device

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: Sequence[str]) -> Dict[str, float]:
        from xh_model_zoo.xh_llm.models.qwen3_next.qwen3_next_onnx_model import (
            _alloc_cache_inputs,
            _select_last_valid_logits,
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.runtime.device)
        max_context_tokens = getattr(self.runtime, "max_context_tokens", None)
        if max_context_tokens is not None and input_ids.shape[1] > max_context_tokens:
            input_ids = input_ids[:, -int(max_context_tokens):]

        self.runtime._ensure_prefill_session()
        cache_state = _alloc_cache_inputs(self.runtime.prefill_session, self.runtime.device)
        chunk_len = int(self.runtime._prefill_inputs_info.shape[1])
        past_seq_len = 0
        last_logits = None
        for start in range(0, int(input_ids.shape[1]), chunk_len):
            end = min(start + chunk_len, int(input_ids.shape[1]))
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            feed = self.runtime._build_prefill_feed(chunk_ids, valid_len, past_seq_len, cache_state)
            _, output_map = self.runtime._run_hmonnx(self.runtime.prefill_session, feed)
            logits = self.runtime._extract_logits(output_map)
            last_logits = _select_last_valid_logits(logits, valid_len)
            self.runtime._update_linear_cache(cache_state, output_map)
            past_seq_len += valid_len
        if last_logits is None:
            raise RuntimeError("No logits returned by HMONNX prefill")
        next_logits = last_logits[0, -1, :].to(torch.float32)
        scores: Dict[str, float] = {}
        for choice in choices:
            token_ids = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Choice {choice!r} is not a single token: {token_ids}")
            scores[choice] = float(next_logits[int(token_ids[0])].item())
        return scores

    @torch.no_grad()
    def logits(self, input_ids: torch.LongTensor):
        from xh_model_zoo.xh_llm.models.qwen3_next.qwen3_next_onnx_model import (
            _alloc_cache_inputs,
            _ensure_logits_shape,
        )

        model = self.runtime
        model._ensure_prefill_session()
        chunk_len = int(model._prefill_inputs_info.shape[1])
        cache_state = _alloc_cache_inputs(model.prefill_session, model.device)
        chunks = []
        past_seq_len = 0
        input_ids = input_ids.to(model.device)
        for start in range(0, int(input_ids.shape[1]), chunk_len):
            end = min(start + chunk_len, int(input_ids.shape[1]))
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            feed = model._build_prefill_feed(chunk_ids, valid_len, past_seq_len, cache_state)
            _, output_map = model._run_hmonnx(model.prefill_session, feed)
            logits = _ensure_logits_shape(model._extract_logits(output_map))
            if logits.shape[1] < valid_len:
                raise RuntimeError(
                    f"Prefill returned {logits.shape[1]} logits < {valid_len}; "
                    "PPL requires full prefill logits."
                )
            chunks.append(logits[:, :valid_len, :].to(torch.float32))
            model._update_linear_cache(cache_state, output_map)
            past_seq_len += valid_len
        return torch.cat(chunks, dim=1)


def load_backend(args):
    if args.backend == "fp":
        if not args.model:
            raise ValueError("--model is required for --backend fp")
        return load_fp_backend(args.model, args.dtype, args.device_map, args.experts_implementation)
    if args.backend == "hmonnx":
        if not args.config:
            raise ValueError("--config is required for --backend hmonnx")
        return HMONNXBackend(
            meta_path=args.config,
            dtype=args.dtype,
            device=args.device,
            exec_device=args.exec_device,
            resource_tight_mode=args.resource_tight_mode,
            enable_cuda_graph=args.enable_cuda_graph,
            max_memory_gb=args.hmonnx_max_memory_gb,
        )
    raise ValueError(f"Unsupported backend {args.backend}")


def load_wikitext_ids(tokenizer, split: str, cache_dir: str) -> torch.Tensor:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir, keep_in_memory=True)
    full_ids = []
    for text in ds["text"]:
        if text:
            full_ids.append(tokenizer(text + "\n\n", return_tensors="pt").input_ids)
    return torch.cat(full_ids, dim=-1)


@torch.no_grad()
def evaluate_ppl(backend, split: str, seqlen: int, batch_num: int, cache_dir: str) -> Dict[str, Any]:
    input_ids = load_wikitext_ids(backend.tokenizer, split, cache_dir)
    nsamples = input_ids.numel() // seqlen
    if batch_num > 0:
        nsamples = min(nsamples, batch_num)
    if nsamples <= 0:
        raise ValueError(f"No WikiText samples for seqlen={seqlen}")

    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    for i in tqdm(range(nsamples), desc="PPL"):
        batch = input_ids[:, i * seqlen : (i + 1) * seqlen]
        logits = backend.logits(batch)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].to(shift_logits.device)
        nll = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        total_nll += float(nll.detach().cpu().item())
        total_tokens += int(shift_labels.numel())
    avg_nll = total_nll / total_tokens
    return {
        "wikitext ppl": math.exp(avg_nll),
        "avg_nll": avg_nll,
        "token_count": total_tokens,
        "nsamples": nsamples,
        "seqlen": seqlen,
        "split": split,
    }


def ceval_prompt(item: Dict[str, Any]) -> str:
    return (
        "以下是中国考试的单项选择题。请只给出选项字母 A、B、C 或 D。\n"
        f"问题：{item['question']}\n"
        f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n"
        "答案："
    )


def mmlu_prompt(item: Dict[str, Any]) -> str:
    labels = [chr(ord("A") + i) for i in range(len(item["choices"]))]
    options = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(labels, item["choices"], strict=True)
    )
    return (
        "The following is a multiple choice question. Answer with only the option letter.\n"
        f"Question: {item['question']}\n{options}\nAnswer:"
    )


def argmax_score(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def evaluate_ceval(backend, subsets: Sequence[str], limit: int, cache_dir: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {"subsets": {}, "examples": []}
    total_correct = 0
    total_count = 0
    for subset in subsets:
        ds = load_dataset("ceval/ceval-exam", subset, split="val", cache_dir=cache_dir, trust_remote_code=True)
        n = len(ds) if limit <= 0 else min(limit, len(ds))
        correct = 0
        examples: List[Dict[str, Any]] = []
        for idx in tqdm(range(n), desc=f"CEval/{subset}"):
            item = ds[idx]
            scores = backend.score_choices(ceval_prompt(item), ["A", "B", "C", "D"])
            pred = argmax_score(scores)
            gold = str(item["answer"]).strip().upper()
            ok = pred == gold
            correct += int(ok)
            ex = ChoiceExample(
                dataset="ceval",
                subset=subset,
                index=idx,
                question=item["question"],
                choices={c: item[c] for c in "ABCD"},
                gold=gold,
                pred=pred,
                scores=scores,
                correct=ok,
            ).__dict__
            examples.append(ex)
            results["examples"].append(ex)
        acc = correct / n if n else 0.0
        results["subsets"][subset] = {"accuracy": acc, "correct": correct, "count": n, "examples": examples}
        total_correct += correct
        total_count += n
    results["average_accuracy"] = total_correct / total_count if total_count else 0.0
    results["correct"] = total_correct
    results["count"] = total_count
    return results


def evaluate_mmlu(
    backend,
    subjects: Sequence[str],
    limit: int,
    cache_dir: str,
    choice_batch_size: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"subjects": {}, "examples": [], "dataset_id": "cais/mmlu"}
    total_correct = 0
    total_count = 0
    if any(subject.lower() == "all" for subject in subjects):
        subjects = ["all"]
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test", cache_dir=cache_dir)
        n = len(ds) if limit <= 0 else min(limit, len(ds))
        correct = 0
        examples: List[Dict[str, Any]] = []
        pbar = tqdm(range(0, n, max(1, choice_batch_size)), desc=f"MMLU/{subject}")
        for start in pbar:
            end = min(start + max(1, choice_batch_size), n)
            items = [ds[idx] for idx in range(start, end)]
            labels_per_item = [
                [chr(ord("A") + i) for i in range(len(item["choices"]))]
                for item in items
            ]
            same_labels = all(labels == labels_per_item[0] for labels in labels_per_item)
            if (
                len(items) > 1
                and same_labels
                and hasattr(backend, "score_choices_batch")
            ):
                scores_list = backend.score_choices_batch(
                    [mmlu_prompt(item) for item in items],
                    labels_per_item[0],
                )
            else:
                scores_list = [
                    backend.score_choices(mmlu_prompt(item), labels)
                    for item, labels in zip(items, labels_per_item, strict=True)
                ]
            for offset, (item, labels, scores) in enumerate(
                zip(items, labels_per_item, scores_list, strict=True)
            ):
                idx = start + offset
                pred = argmax_score(scores)
                gold = labels[int(item["answer"])]
                ok = pred == gold
                correct += int(ok)
                ex = ChoiceExample(
                    dataset="mmlu",
                    subset=subject,
                    index=idx,
                    question=item["question"],
                    choices={
                        label: choice
                        for label, choice in zip(labels, item["choices"], strict=True)
                    },
                    gold=gold,
                    pred=pred,
                    scores=scores,
                    correct=ok,
                ).__dict__
                examples.append(ex)
                results["examples"].append(ex)
        acc = correct / n if n else 0.0
        results["subjects"][subject] = {"accuracy": acc, "correct": correct, "count": n, "examples": examples}
        total_correct += correct
        total_count += n
    results["average_accuracy"] = total_correct / total_count if total_count else 0.0
    results["correct"] = total_correct
    results["count"] = total_count
    return results


def compact_for_stdout(payload: Any) -> Any:
    """Return a log-friendly view while preserving detailed files on disk."""
    if isinstance(payload, dict):
        return {
            key: compact_for_stdout(value)
            for key, value in payload.items()
            if key != "examples"
        }
    if isinstance(payload, list):
        return [compact_for_stdout(item) for item in payload]
    return payload


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Qwen3-Next Accuracy Evaluation Report",
        "",
        f"- backend: `{summary['backend']}`",
        f"- model/config: `{summary.get('model') or summary.get('config')}`",
        f"- dtype: `{summary['dtype']}`",
        f"- elapsed_sec: {summary['elapsed_sec']:.1f}",
        "",
        "## Metrics",
        "",
    ]
    ppl = summary.get("ppl")
    if ppl:
        lines.append(
            f"- WikiText-2 PPL: **{ppl['wikitext ppl']:.4f}** "
            f"(nsamples={ppl['nsamples']}, seqlen={ppl['seqlen']})"
        )
    ceval = summary.get("ceval")
    if ceval:
        lines.append(f"- CEval avg accuracy: **{ceval['average_accuracy']:.4f}** ({ceval['correct']}/{ceval['count']})")
    mmlu = summary.get("mmlu")
    if mmlu:
        lines.append(
            f"- MMLU avg accuracy: **{mmlu['average_accuracy']:.4f}** "
            f"({mmlu['correct']}/{mmlu['count']}, dataset={mmlu.get('dataset_id')})"
        )
    lines.extend(["", "## Representative answers", ""])
    for section in ("ceval", "mmlu"):
        data = summary.get(section) or {}
        for ex in data.get("examples", [])[: min(5, len(data.get("examples", [])))]:
            lines.extend([
                f"### {section.upper()} / {ex['subset']} / #{ex['index']}",
                f"- gold: `{ex['gold']}`, pred: `{ex['pred']}`, correct: `{ex['correct']}`",
                f"- question: {ex['question']}",
                f"- scores: `{ex['scores']}`",
                "",
            ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3-Next accuracy on PPL, CEVAL, and MMLU with either "
            "the HuggingFace FP model or an exported HMONNX meta.json."
        ),
        formatter_class=RawDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  FP CEVAL full:\n"
            "    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/llm/qwen3_next/qwen3_next_accuracy_eval.py "
            "--backend fp --model weights/Qwen3-Next-80B-A3B-Instruct --tasks ceval --limit -1 "
            "--output-dir work_dirs/qwen3_next_accuracy_eval/fp_ceval_full\n\n"
            "  HMONNX CEVAL full with CUDA graph and 4-GPU auto-offload:\n"
            "    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/llm/qwen3_next/qwen3_next_accuracy_eval.py "
            "--backend hmonnx --config work_dirs/qwen3_next_verify_20260604_165508/80b/meta.json "
            "--device cpu --exec-device cuda --enable-cuda-graph --hmonnx-max-memory-gb 24 "
            "--tasks ceval --limit -1 --output-dir work_dirs/qwen3_next_accuracy_eval/hmonnx_ceval_full\n\n"
            "  HMONNX MMLU full:\n"
            "    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/llm/qwen3_next/qwen3_next_accuracy_eval.py "
            "--backend hmonnx --config work_dirs/qwen3_next_verify_20260604_165508/80b/meta.json "
            "--device cpu --exec-device cuda --enable-cuda-graph --hmonnx-max-memory-gb 24 "
            "--tasks mmlu --mmlu-subjects all --limit -1 --choice-batch-size 1"
        ),
    )
    p.add_argument("--backend", choices=["fp", "hmonnx"], default="fp")
    p.add_argument("--model", type=str, default="weights/Qwen3-Next-80B-A3B-Instruct")
    p.add_argument("--config", type=str, default=None, help="meta.json for hmonnx backend")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument(
        "--experts-implementation",
        type=str,
        default=None,
        choices=["batched_mm", "grouped_mm"],
        help="Optional Transformers MoE experts implementation for FP backend; use batched_mm on non-SM90 GPUs.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--exec-device", type=str, default="cuda")
    p.add_argument("--resource-tight-mode", action="store_true")
    p.add_argument("--enable-cuda-graph", action="store_true")
    p.add_argument(
        "--hmonnx-max-memory-gb",
        type=float,
        default=0.0,
        help="Per-visible-GPU HMONNX auto-offload memory budget in GiB; <=0 uses auto-detect.",
    )
    p.add_argument("--tasks", nargs="+", default=["ppl", "ceval", "mmlu"], choices=["ppl", "ceval", "mmlu"])
    p.add_argument("--ppl-seqlen", type=int, default=128)
    p.add_argument("--ppl-batch-num", type=int, default=1)
    p.add_argument("--ceval-subsets", nargs="+", default=CEVAL_SUBSETS)
    p.add_argument(
        "--mmlu-subjects",
        nargs="+",
        default=MMLU_SUBJECTS,
        help="MMLU configs to evaluate. Use 'all' for the merged full test set.",
    )
    p.add_argument("--limit", type=int, default=1, help="CEVal/MMLU examples per subset; <=0 means full split")
    p.add_argument("--choice-batch-size", type=int, default=1, help="Batch size for FP MMLU choice scoring")
    p.add_argument("--cache-dir", type=str, default="data/datasets")
    p.add_argument("--output-dir", type=str, default="work_dirs/qwen3_next_accuracy_eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    backend = load_backend(args)

    summary: Dict[str, Any] = {
        "backend": args.backend,
        "model": args.model if args.backend == "fp" else None,
        "config": args.config if args.backend == "hmonnx" else None,
        "dtype": args.dtype,
        "tasks": args.tasks,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if "ppl" in args.tasks:
        summary["ppl"] = evaluate_ppl(backend, "test", args.ppl_seqlen, args.ppl_batch_num, args.cache_dir)
        write_json(out_dir / "ppl_summary.json", summary["ppl"])
    if "ceval" in args.tasks:
        summary["ceval"] = evaluate_ceval(backend, args.ceval_subsets, args.limit, args.cache_dir)
        write_json(out_dir / "ceval_summary.json", summary["ceval"])
    if "mmlu" in args.tasks:
        summary["mmlu"] = evaluate_mmlu(
            backend,
            args.mmlu_subjects,
            args.limit,
            args.cache_dir,
            args.choice_batch_size,
        )
        write_json(out_dir / "mmlu_summary.json", summary["mmlu"])
    summary["elapsed_sec"] = time.time() - started
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir / "report.md", summary)
    print(json.dumps(compact_for_stdout(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
