"""Compare full HMONNX layer hidden states against split-MoE HMONNX chaining.

The full graph must be exported with ``--output-hidden-state-indices`` so its
decode graph exposes a concatenated ``target_hidden`` output.  This script runs
one decode token through both paths and compares each selected full-model layer
hidden state against ``split_layer_{idx}_hidden_out`` from the split chain.

Example:
  conda run -n xh2 --no-capture-output python \
    examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_full_vs_split_hmonnx_compare.py \
    --full-work-dir work_dirs/qwen36moe-no-rotate-attn8-shared8-n256-iter400-XH2a-2k-w8a8h0_sefp \
    --split-work-dir work_dirs/qwen36moe-no-rotate-attn8-shared8-n256-iter400-split-moe-premoe-w8a8h0_sefp-experts-w4a8h0_sefp \
    --token-id 12675
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from xhquant.core import CacheTensor

from examples.llm.qwen3_5_moe.split_qwen36.qwen3_5_moe_xh2a_split_moe_hmonnx_e2e_test import (
    DEFAULT_WORK_DIR as DEFAULT_SPLIT_WORK_DIR,
    SplitMoEHMONNXRunner,
    _create_hmonnx_session,
    _load_token_embedding,
    _resolve_path,
    _run_hmonnx,
    _shape_of,
    _zeros_like_input,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_decode_onnx(work_dir: Path, meta: Dict[str, Any]) -> Path:
    rel_path = meta.get("decode_onnx") or meta.get("decode_onnx_file")
    if rel_path:
        return _resolve_path(work_dir, str(rel_path))
    candidates = sorted((work_dir / "hmonnx" / "decode").glob("*.onnx"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Cannot resolve full decode ONNX under {work_dir}; candidates={candidates}")


def _resolve_embedding(work_dir: Path, meta: Dict[str, Any]) -> Path:
    rel_path = meta.get("quant_embedding") or meta.get("token_embedding_file") or "token_embedding.pt"
    return _resolve_path(work_dir, str(rel_path))


def _as_cache_value(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if isinstance(reference, CacheTensor) and not isinstance(value, CacheTensor):
        return CacheTensor(value)
    return value


def _is_cache_input_name(name: str) -> bool:
    return name.startswith(
        (
            "past_key_cache",
            "past_value_cache",
            "past_conv_cache",
            "past_recurrent_state",
        )
    ) or name.endswith(("_kcache_input", "_vcache_input"))


def _align_shape(full_value: torch.Tensor, split_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if tuple(full_value.shape) == tuple(split_value.shape):
        return full_value, split_value
    if full_value.numel() == split_value.numel():
        return full_value.reshape_as(split_value), split_value
    raise ValueError(f"Cannot compare shapes full={tuple(full_value.shape)} split={tuple(split_value.shape)}")


class FullDecodeHMONNXRunner:
    def __init__(
        self,
        work_dir: Path,
        split_work_dir: Path,
        device: torch.device,
        exec_device: torch.device,
        *,
        embedding_source: str = "split",
    ) -> None:
        self.work_dir = work_dir.resolve()
        self.meta = _load_json(self.work_dir / "meta.json")
        self.split_work_dir = split_work_dir.resolve()
        self.split_meta = _load_json(self.split_work_dir / "split_moe_meta.json")
        self.device = device
        self.exec_device = exec_device
        self.decode_path = _resolve_decode_onnx(self.work_dir, self.meta)
        self.session = _create_hmonnx_session(self.decode_path, self.device, self.exec_device)
        self.past_seq_length = 0
        self._caches: Dict[str, torch.Tensor] = {}

        embedding_meta = self.split_meta if embedding_source == "split" else self.meta
        embedding_dir = self.split_work_dir if embedding_source == "split" else self.work_dir
        self.token_embedding = _load_token_embedding(_resolve_embedding(embedding_dir, embedding_meta)).to(self.device).eval()
        self.hidden_size = int(self.split_meta["hidden_size"])
        self.num_layers = int(self.split_meta["num_hidden_layers"])
        self.layer_indices = self._resolve_layer_indices()

    def close(self) -> None:
        self._caches.clear()
        self.session = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_layer_indices(self) -> List[int]:
        wrap_cfg = self.meta.get("wrap_cfg", {})
        indices = wrap_cfg.get("output_hidden_state_indices")
        if indices is None:
            raise ValueError(
                "Full HMONNX meta.json has no wrap_cfg.output_hidden_state_indices. "
                "Re-export full HMONNX with --output-hidden-state-indices all or a layer list."
            )
        layer_indices = [int(idx) for idx in indices]
        invalid = [idx for idx in layer_indices if idx < 0 or idx >= self.num_layers]
        if invalid:
            raise ValueError(f"Full HMONNX target hidden contains invalid layer ids: {invalid}")
        return layer_indices

    def _cache_input(self, name: str, info) -> torch.Tensor:
        if name not in self._caches:
            self._caches[name] = CacheTensor(_zeros_like_input(info, self.device))
        return self._caches[name]

    def _build_feed(self, inputs_embeds: torch.Tensor) -> Dict[str, torch.Tensor]:
        position_ids = torch.tensor([[self.past_seq_length]], dtype=torch.int32, device=self.device)
        past_seq = torch.tensor([self.past_seq_length], dtype=torch.int32, device=self.device)
        current_len = torch.tensor([1], dtype=torch.int32, device=self.device)
        feed: Dict[str, torch.Tensor] = {}
        for name in self.session.get_input_names():
            info = self.session.get_input(name)
            if name in ("inputs_embeds", "hidden_in", "input_1"):
                feed[name] = inputs_embeds.to(device=self.device, dtype=info.dtype)
            elif name in ("time_position_ids", "hight_position_ids", "height_position_ids", "width_position_ids"):
                feed[name] = position_ids.to(dtype=info.dtype)
            elif name in ("valid_length", "past_seq_length"):
                feed[name] = past_seq.to(dtype=info.dtype)
            elif name in ("current_length", "current_input_length"):
                feed[name] = current_len.to(dtype=info.dtype)
            elif name == "linear_attn_mask":
                feed[name] = torch.ones(_shape_of(info), dtype=info.dtype, device=self.device)
            elif _is_cache_input_name(name):
                feed[name] = self._cache_input(name, info)
            else:
                feed[name] = _zeros_like_input(info, self.device)
        return feed

    def _update_caches(self, outputs: Dict[str, torch.Tensor]) -> None:
        for name, cached in list(self._caches.items()):
            candidates = [name]
            if name.startswith("past_key_cache_"):
                suffix = name[len("past_key_cache_"):]
                candidates.append(f"key_cache_out_{suffix}")
            elif name.startswith("past_value_cache_"):
                suffix = name[len("past_value_cache_"):]
                candidates.append(f"value_cache_out_{suffix}")
            elif name.startswith("past_conv_cache_"):
                suffix = name[len("past_conv_cache_"):]
                candidates.extend([f"conv_cache_out_{suffix}", f"conv_cache_out_{suffix}_0"])
            elif name.startswith("past_recurrent_state"):
                suffix = name[len("past_recurrent_state"):].lstrip("_")
                candidates.append("recurrent_state_out")
                if suffix:
                    candidates.extend([f"recurrent_state_out_{suffix}", f"recurrent_state_out_{suffix}_0"])
            for candidate in candidates:
                if candidate in outputs:
                    self._caches[name] = _as_cache_value(cached, outputs[candidate])
                    break

    def forward_token(self, token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        inputs_embeds = self.token_embedding(token).to(self.device)
        _, outputs = _run_hmonnx(self.session, self._build_feed(inputs_embeds))
        self._update_caches(outputs)
        self.past_seq_length += 1

        logits = outputs.get("logits")
        if logits is None:
            logits = outputs.get("logits_batch_0")
        target_hidden = outputs.get("target_hidden")
        if target_hidden is None:
            target_hidden = outputs.get("target_hidden_batch_0")
        if logits is None:
            raise KeyError(f"Full decode graph did not emit logits. Outputs: {sorted(outputs)}")
        if target_hidden is None:
            raise KeyError(
                "Full decode graph did not emit target_hidden. "
                f"Outputs: {sorted(outputs)}. Re-export with --output-hidden-state-indices."
            )
        return logits, target_hidden

    def split_target_hidden(self, target_hidden: torch.Tensor) -> Dict[int, torch.Tensor]:
        expected_width = len(self.layer_indices) * self.hidden_size
        if int(target_hidden.shape[-1]) != expected_width:
            raise ValueError(
                f"target_hidden last dim {int(target_hidden.shape[-1])} != "
                f"len(indices)({len(self.layer_indices)}) * hidden_size({self.hidden_size})"
            )
        return {
            layer_idx: target_hidden[..., pos * self.hidden_size : (pos + 1) * self.hidden_size]
            for pos, layer_idx in enumerate(self.layer_indices)
        }


def compare_hidden_states(
    full_runner: FullDecodeHMONNXRunner,
    split_runner: SplitMoEHMONNXRunner,
    token_id: int,
    threshold: float,
    *,
    max_split_layers: Optional[int] = None,
    include_layer_details: bool = False,
    use_host_postmoe: bool = False,
) -> bool:
    full_logits, target_hidden = full_runner.forward_token(token_id)
    split_logits, split_intermediates = split_runner.forward_token_with_intermediates(
        token_id,
        max_layers=max_split_layers,
        include_layer_details=include_layer_details,
        use_host_postmoe=use_host_postmoe,
    )
    full_hidden_by_layer = full_runner.split_target_hidden(target_hidden)

    if split_logits is not None:
        full_next = int(torch.argmax(full_logits[:, -1, :], dim=-1).item())
        split_next = int(torch.argmax(split_logits[:, -1, :], dim=-1).item())
        logits_diff = (full_logits.float() - split_logits.float()).abs()
        print("LOGITS")
        print(f"  full next token id:  {full_next}")
        print(f"  split next token id: {split_next}")
        print(f"  max_abs_diff: {float(logits_diff.max()):.6e}")
        print(f"  mean_abs_diff: {float(logits_diff.mean()):.6e}")
        print()
    else:
        print("LOGITS")
        print("  skipped because --max-split-layers stopped before the head graph")
        print()
    print("PER-LAYER HIDDEN_OUT")
    print("  layer  max_abs_diff  mean_abs_diff  status")

    first_bad: Optional[int] = None
    worst_layer = -1
    worst_diff = -1.0
    compared_layers = [
        layer_idx
        for layer_idx in full_runner.layer_indices
        if max_split_layers is None or layer_idx < int(max_split_layers)
    ]
    for layer_idx in compared_layers:
        split_key = f"split_layer_{layer_idx}_hidden_out"
        if split_key not in split_intermediates:
            raise KeyError(f"Missing split intermediate {split_key}")
        full_hidden, split_hidden = _align_shape(full_hidden_by_layer[layer_idx], split_intermediates[split_key])
        diff = (full_hidden.float() - split_hidden.float()).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        if max_diff > worst_diff:
            worst_layer = layer_idx
            worst_diff = max_diff
        status = "OK" if max_diff <= threshold else "DIFF"
        if first_bad is None and max_diff > threshold:
            first_bad = layer_idx
        print(f"  {layer_idx:05d}  {max_diff:.6e}  {mean_diff:.6e}  {status}")
        if include_layer_details:
            _print_layer_details(layer_idx, full_hidden, split_intermediates, threshold)

    print()
    print(f"Worst layer: {worst_layer} max_abs_diff={worst_diff:.6e}")
    if first_bad is None:
        print(f"All compared layers are within threshold {threshold:.6e}.")
        return True
    print(f"First layer exceeding threshold {threshold:.6e}: {first_bad}")
    return False


def _print_layer_details(
    layer_idx: int,
    full_hidden: torch.Tensor,
    split_intermediates: Dict[str, torch.Tensor],
    threshold: float,
) -> None:
    topk_id = split_intermediates.get(f"split_layer_{layer_idx}_topk_id")
    topk_gate = split_intermediates.get(f"split_layer_{layer_idx}_topk_gate")
    if topk_id is not None and topk_gate is not None:
        expert_ids = [int(v) for v in topk_id.reshape(-1).detach().cpu().tolist()]
        gate_vals = [float(v) for v in topk_gate.reshape(-1).float().detach().cpu().tolist()]
        gate_text = ", ".join(f"{value:.6f}" for value in gate_vals)
        print(f"    routed_experts: {expert_ids}")
        print(f"    topk_gate: [{gate_text}]")
    for name in ("moe_input", "shared_out", "residual1"):
        tensor = split_intermediates.get(f"split_layer_{layer_idx}_{name}")
        if tensor is None:
            continue
        print(
            f"    {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"norm={float(tensor.float().norm()):.6e}"
        )
    host_key = f"split_layer_{layer_idx}_hidden_out_host_postmoe"
    hmonnx_key = f"split_layer_{layer_idx}_hidden_out_hmonnx_postmoe"
    if host_key in split_intermediates and hmonnx_key in split_intermediates:
        host_hidden = split_intermediates[host_key]
        hmonnx_hidden = split_intermediates[hmonnx_key]
        host_hmonnx_diff = (host_hidden.float() - hmonnx_hidden.float()).abs()
        full_host, host_hidden = _align_shape(full_hidden, host_hidden)
        full_host_diff = (full_host.float() - host_hidden.float()).abs()
        print(
            "    host_vs_hmonnx_postmoe: "
            f"max_abs_diff={float(host_hmonnx_diff.max()):.6e} "
            f"mean_abs_diff={float(host_hmonnx_diff.mean()):.6e}"
        )
        print(
            "    full_vs_host_postmoe: "
            f"max_abs_diff={float(full_host_diff.max()):.6e} "
            f"mean_abs_diff={float(full_host_diff.mean()):.6e} "
            f"status={'OK' if float(full_host_diff.max()) <= threshold else 'DIFF'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full HMONNX target_hidden against split-MoE chain hidden_out.")
    parser.add_argument("--full-work-dir", required=True, help="Full HMONNX export directory containing meta.json.")
    parser.add_argument("--split-work-dir", default=DEFAULT_SPLIT_WORK_DIR, help="Split-MoE export directory.")
    parser.add_argument("--token-id", type=int, default=12675, help="Single decode token id to compare.")
    parser.add_argument("--device", default="cuda", help="Torch device for HMONNX inputs and cached tensors.")
    parser.add_argument("--execution-device", default="cuda", help="HMONNX execution device.")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Per-layer max-abs threshold for OK/DIFF.")
    parser.add_argument(
        "--embedding-source",
        choices=("split", "full"),
        default="split",
        help="Token embedding artifact used to feed the full graph. Use split by default for identical input.",
    )
    parser.add_argument("--cache-experts", action="store_true", help="Cache split expert sessions.")
    parser.add_argument(
        "--max-split-layers",
        type=int,
        default=None,
        help="Run only the first N split layers. Useful for fast first-bad-layer diagnosis.",
    )
    parser.add_argument(
        "--include-layer-details",
        action="store_true",
        help="Print split premoe routing, tensor stats, and host-vs-HMONNX postmoe detail for compared layers.",
    )
    parser.add_argument(
        "--use-host-postmoe",
        action="store_true",
        help="Use torch host aggregation instead of the postmoe HMONNX graph for split hidden_out.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    exec_device = torch.device(args.execution_device)
    full_runner = FullDecodeHMONNXRunner(
        Path(args.full_work_dir),
        Path(args.split_work_dir),
        device,
        exec_device,
        embedding_source=args.embedding_source,
    )
    split_runner = SplitMoEHMONNXRunner(
        Path(args.split_work_dir),
        device=device,
        exec_device=exec_device,
        cache_experts=args.cache_experts,
    )
    try:
        ok = compare_hidden_states(
            full_runner,
            split_runner,
            args.token_id,
            args.threshold,
            max_split_layers=args.max_split_layers,
            include_layer_details=args.include_layer_details,
            use_host_postmoe=args.use_host_postmoe,
        )
    finally:
        full_runner.close()
        split_runner.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
