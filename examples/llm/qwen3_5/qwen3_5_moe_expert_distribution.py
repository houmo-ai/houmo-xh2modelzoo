#!/usr/bin/env python3
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""MoE expert activation distribution analysis for Qwen3.6-35B-A3B.

Phase 1 — measure only. No expert reorder.

Usage:
    python qwen3_5_moe_expert_distribution.py --categories math --n-per-category 2 --max-new-tokens 16
    python qwen3_5_moe_expert_distribution.py  # full run, all 6 categories
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Use non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_MODEL_PATH = str(REPO_ROOT / "weights" / "Qwen3.6-35B-A3B")
DEFAULT_PROMPTS_JSONL = str(SCRIPT_DIR / "spec_decode_eval_prompts.jsonl")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "expert_dist_out")
ALL_CATEGORIES = ["humanities", "social", "tech", "math", "tool_calls", "coding"]


# ---------------------------------------------------------------------------
# Seed / data helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_prompts(
    jsonl_path: str,
    categories: list[str],
    n_per_category: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    by_cat: dict[str, list[str]] = defaultdict(list)
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["category"] in categories:
                by_cat[item["category"]].append(item["prompt"])
    result: dict[str, list[str]] = {}
    for cat in categories:
        pool = by_cat.get(cat, [])
        result[cat] = rng.sample(pool, min(n_per_category, len(pool)))
    return result


def apply_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# MoE layer discovery
# ---------------------------------------------------------------------------

def find_moe_gates(model: Qwen3_5MoeForConditionalGeneration) -> list[tuple[int, torch.nn.Module]]:
    """Return list of (layer_idx, gate_module) for all MoE layers."""
    moe_gates = []
    # Structure: model.model.language_model.layers[i].mlp.gate
    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate"):
            moe_gates.append((i, mlp.gate))
    return moe_gates


# ---------------------------------------------------------------------------
# Hook-based collector
# ---------------------------------------------------------------------------

class ExpertDistributionCollector:
    """Records per-layer expert selections during decode phase via forward hooks.

    Decode detection: the gate module receives hidden states of shape
    (batch * seq_len, hidden_dim). During prefill seq_len > 1, during decode
    seq_len == 1. For single-batch generation (batch_size = 1) this means
    n_tokens == 1 iff we are in a decode step.
    """

    def __init__(self, num_experts: int, top_k: int) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        # layer_idx → list of [top_k] expert-id lists (one per decode token)
        self.decode_selections: dict[int, list[list[int]]] = defaultdict(list)
        self._hooks: list = []

    def register_hooks(self, moe_gates: list[tuple[int, torch.nn.Module]]) -> None:
        for layer_idx, gate_mod in moe_gates:
            h = gate_mod.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            # outputs = (router_logits, router_scores, router_indices)
            # router_indices: (batch * seq_len, top_k)
            router_indices = outputs[2]
            n_tokens = router_indices.shape[0]
            # Skip prefill (n_tokens > 1 for single-batch scenario)
            if n_tokens != 1:
                return
            # Decode step: record expert IDs for this token
            self.decode_selections[layer_idx].append(
                router_indices[0].cpu().tolist()
            )
        return hook

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self) -> None:
        self.decode_selections.clear()

    def compute_stats(self) -> dict[int, dict]:
        E = self.num_experts
        K = self.top_k
        half_E = E // 2
        stats: dict[int, dict] = {}

        for layer_idx in sorted(self.decode_selections.keys()):
            selections = self.decode_selections[layer_idx]
            if not selections:
                continue
            n_tokens = len(selections)
            freq = np.zeros(E, dtype=np.int64)
            half_hist = np.zeros(K + 1, dtype=np.int64)

            for sel in selections:
                for eid in sel:
                    freq[eid] += 1
                n_lower = sum(1 for eid in sel if eid < half_E)
                half_hist[n_lower] += 1

            half_k = K // 2
            frac_perfect = float(half_hist[half_k]) / n_tokens
            lo = max(0, half_k - 1)
            hi = min(K, half_k + 1)
            frac_acceptable = float(half_hist[lo:hi + 1].sum()) / n_tokens
            frac_bad = 1.0 - frac_acceptable

            stats[layer_idx] = {
                "n_tokens": n_tokens,
                "expert_freq": freq.tolist(),
                "half_balance_hist": half_hist.tolist(),
                "frac_perfect": frac_perfect,
                "frac_acceptable": frac_acceptable,
                "frac_bad": frac_bad,
            }
        return stats


# ---------------------------------------------------------------------------
# Per-category runner
# ---------------------------------------------------------------------------

def run_category(
    model: Qwen3_5MoeForConditionalGeneration,
    tokenizer,
    collector: ExpertDistributionCollector,
    category: str,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
) -> dict[int, dict]:
    """Run all prompts for one category; return accumulated per-layer stats."""
    E = collector.num_experts
    K = collector.top_k

    acc_freq: dict[int, np.ndarray] = {}
    acc_hist: dict[int, np.ndarray] = {}
    acc_tokens: dict[int, int] = {}

    for idx, prompt in enumerate(prompts):
        print(f"  [{category}] {idx + 1}/{len(prompts)} ...", end=" ", flush=True)
        collector.reset()

        text = apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        step_stats = collector.compute_stats()
        if step_stats:
            sample_tokens = next(iter(step_stats.values()))["n_tokens"]
        else:
            sample_tokens = 0
        print(f"done ({sample_tokens} decode steps)")

        for layer_idx, s in step_stats.items():
            n = s["n_tokens"]
            if layer_idx not in acc_freq:
                acc_freq[layer_idx] = np.zeros(E, dtype=np.int64)
                acc_hist[layer_idx] = np.zeros(K + 1, dtype=np.int64)
                acc_tokens[layer_idx] = 0
            acc_freq[layer_idx] += np.array(s["expert_freq"], dtype=np.int64)
            acc_hist[layer_idx] += np.array(s["half_balance_hist"], dtype=np.int64)
            acc_tokens[layer_idx] += n

    # Finalise aggregated stats
    final: dict[int, dict] = {}
    for layer_idx in sorted(acc_tokens.keys()):
        n = acc_tokens[layer_idx]
        if n == 0:
            continue
        half_hist = acc_hist[layer_idx]
        half_k = K // 2
        frac_perfect = float(half_hist[half_k]) / n
        lo = max(0, half_k - 1)
        hi = min(K, half_k + 1)
        frac_acceptable = float(half_hist[lo:hi + 1].sum()) / n
        frac_bad = 1.0 - frac_acceptable
        final[layer_idx] = {
            "n_tokens": int(n),
            "expert_freq": acc_freq[layer_idx].tolist(),
            "half_balance_hist": half_hist.tolist(),
            "frac_perfect": frac_perfect,
            "frac_acceptable": frac_acceptable,
            "frac_bad": frac_bad,
        }
    return final


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def plot_heatmap(stats: dict[int, dict], category: str, num_experts: int, out: Path) -> None:
    layers = sorted(stats.keys())
    matrix = np.array([stats[l]["expert_freq"] for l in layers])  # (L, E)
    fig, ax = plt.subplots(figsize=(max(12, num_experts // 8), max(5, len(layers) // 4)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer Index")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_title(f"Expert Selection Frequency — {category}")
    plt.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_half_balance(stats: dict[int, dict], category: str, top_k: int, out: Path) -> None:
    layers = sorted(stats.keys())
    matrix = np.array([stats[l]["half_balance_hist"] for l in layers])  # (L, K+1)
    fig, ax = plt.subplots(figsize=(top_k + 3, max(5, len(layers) // 4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel(f"j = # experts in [0, E/2)   (ideal = {top_k // 2})")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(top_k + 1))
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=6)
    ax.set_title(f"Half-Balance Distribution — {category}")
    # Mark ideal column
    ax.axvline(x=top_k // 2, color="cyan", linewidth=1.5, linestyle="--", alpha=0.7)
    plt.colorbar(im, ax=ax, label="Token Count")
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_cross_task_summary(
    all_stats: dict[str, dict[int, dict]], out: Path
) -> None:
    categories = list(all_stats.keys())
    if not categories:
        return
    all_layers = sorted({l for s in all_stats.values() for l in s.keys()})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, metric, title in [
        (axes[0], "frac_perfect", "frac_perfect  (j == K/2)"),
        (axes[1], "frac_acceptable", "frac_acceptable  (|j − K/2| ≤ 1)"),
    ]:
        for cat, color in zip(categories, colors):
            ys = [all_stats[cat].get(l, {}).get(metric, 0.0) for l in all_layers]
            ax.plot(all_layers, ys, label=cat, color=color, marker="o", markersize=3)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Fraction")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cross-task Half-Balance Per Layer")
    fig.tight_layout()
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def print_console_table(all_stats: dict[str, dict[int, dict]]) -> None:
    print("\n" + "=" * 72)
    print("CROSS-TASK SUMMARY  (mean across layers)")
    print("=" * 72)
    print(f"{'Category':<16} {'frac_perfect':>13} {'frac_acceptable':>16} {'frac_bad':>10}")
    print("-" * 72)
    for cat in sorted(all_stats):
        vals = list(all_stats[cat].values())
        if not vals:
            continue
        mean_p = float(np.mean([v["frac_perfect"] for v in vals]))
        mean_a = float(np.mean([v["frac_acceptable"] for v in vals]))
        mean_b = float(np.mean([v["frac_bad"] for v in vals]))
        print(f"{cat:<16} {mean_p:>13.4f} {mean_a:>16.4f} {mean_b:>10.4f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE expert distribution analysis — Phase 1 (measure only)"
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Path to Qwen3.6-35B-A3B weights directory")
    parser.add_argument("--prompts-jsonl", default=DEFAULT_PROMPTS_JSONL,
                        help="Path to spec_decode_eval_prompts.jsonl")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES,
                        metavar="CAT")
    parser.add_argument("--n-per-category", type=int, default=5,
                        help="Number of prompts per category")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0",
                        help="Primary device for tokenizer inputs; model uses device_map=auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Derive config from model — no hardcoding
    text_cfg = model.config.text_config
    num_layers: int = text_cfg.num_hidden_layers
    num_experts: int = text_cfg.num_experts
    top_k: int = text_cfg.num_experts_per_tok
    print(f"Config: L={num_layers}, E={num_experts}, K={top_k}  (half_E={num_experts // 2})")

    moe_gates = find_moe_gates(model)
    print(f"Found {len(moe_gates)} MoE gate layers")

    collector = ExpertDistributionCollector(num_experts, top_k)
    collector.register_hooks(moe_gates)

    prompts_by_cat = load_prompts(
        args.prompts_jsonl, args.categories, args.n_per_category, args.seed
    )

    all_category_stats: dict[str, dict[int, dict]] = {}
    summary: dict[str, dict] = {}

    for category in args.categories:
        prompts = prompts_by_cat.get(category, [])
        if not prompts:
            print(f"[WARN] No prompts for category '{category}', skipping.")
            continue

        print(f"\n=== {category}  ({len(prompts)} prompts, max_new_tokens={args.max_new_tokens}) ===")
        stats = run_category(
            model, tokenizer, collector, category, prompts,
            args.max_new_tokens, args.device,
        )
        all_category_stats[category] = stats

        # Per-category JSON
        save_json(out_dir / f"expert_dist_{category}.json", {
            "category": category,
            "n_prompts": len(prompts),
            "num_experts": num_experts,
            "top_k": top_k,
            "per_layer": {str(k): v for k, v in stats.items()},
        })

        # Plots
        if stats:
            plot_heatmap(stats, category, num_experts,
                         out_dir / f"heatmap_{category}.png")
            plot_half_balance(stats, category, top_k,
                              out_dir / f"half_balance_{category}.png")

        # Console per-category summary
        if stats:
            mean_p = float(np.mean([v["frac_perfect"] for v in stats.values()]))
            mean_a = float(np.mean([v["frac_acceptable"] for v in stats.values()]))
            print(f"  → mean frac_perfect={mean_p:.4f}  frac_acceptable={mean_a:.4f}")

        summary[category] = {
            str(l): {
                "frac_perfect": v["frac_perfect"],
                "frac_acceptable": v["frac_acceptable"],
                "frac_bad": v["frac_bad"],
                "n_tokens": v["n_tokens"],
            }
            for l, v in stats.items()
        }

    # Global summary files
    save_json(out_dir / "summary.json", summary)
    plot_cross_task_summary(all_category_stats, out_dir / "cross_task_summary.png")

    print_console_table(all_category_stats)

    collector.remove_hooks()

    print(f"\nOutputs → {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
