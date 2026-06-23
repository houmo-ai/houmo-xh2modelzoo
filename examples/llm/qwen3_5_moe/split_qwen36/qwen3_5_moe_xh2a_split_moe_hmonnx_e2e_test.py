from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xhquant.core import CacheTensor
from xhquant.xhonnxruntime import HMONNXGrapInference, HMONNXInference


DEFAULT_WORK_DIR = (
    "work_dirs/qwen36moe-no-rotate-attn8-shared8-n256-iter400-split-moe-premoe-w8a8h0_sefp-"
    "experts-w4a8h0_sefp"
)


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_token_embedding(embed_path: Path) -> nn.Module:
    try:
        obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(embed_path), map_location="cpu")
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj
    if isinstance(obj, dict) and "weight" in obj:
        embedding = nn.Embedding(
            obj["weight"].shape[0],
            obj["weight"].shape[1],
            dtype=obj["weight"].dtype,
        )
        embedding.load_state_dict(obj)
        embedding.eval()
        return embedding
    raise TypeError(f"Unsupported token embedding object type from {embed_path}: {type(obj)}")


def _shape_of(info) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in info.shape)


def _zeros_like_input(info, device: torch.device) -> torch.Tensor:
    return torch.zeros(_shape_of(info), dtype=info.dtype, device=device)


def _as_cache_value(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if isinstance(reference, CacheTensor) and not isinstance(value, CacheTensor):
        return CacheTensor(value)
    return value


def _run_hmonnx(
    session,
    input_feed: Dict[str, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    outputs = session.run(input_feed)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    output_names = session.get_output_names()
    output_map = {name: out for name, out in zip(output_names, outputs, strict=False)}
    return tuple(outputs), output_map


def _create_hmonnx_session(onnx_path: Path, device: torch.device, exec_device: torch.device):
    try:
        session = HMONNXGrapInference(str(onnx_path))
    except AttributeError as exc:
        if "'str' object has no attribute 'name'" not in str(exc):
            raise
        session = HMONNXInference(str(onnx_path))
    session.to(device)
    session.exec_device = exec_device
    return session


def _default_head_record(work_dir: Path) -> Optional[dict]:
    head_path = work_dir / "hmonnx" / "head" / "head.onnx"
    if not head_path.exists():
        return None
    return {"onnx": str(head_path.relative_to(work_dir))}


def _count_onnx(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.onnx"))


def _component_summary(work_dir: Path, meta: dict) -> Dict[str, int]:
    num_layers = int(meta.get("num_hidden_layers", 0))
    num_experts = int(meta.get("num_experts", 0))
    return {
        "premoe_prefill": _count_onnx(work_dir / "hmonnx" / "premoe_prefill"),
        "premoe_decode": _count_onnx(work_dir / "hmonnx" / "premoe"),
        "experts": _count_onnx(work_dir / "hmonnx" / "experts"),
        "postmoe": _count_onnx(work_dir / "hmonnx" / "postmoe"),
        "head": _count_onnx(work_dir / "hmonnx" / "head"),
        "expected_premoe_decode": num_layers,
        "expected_experts": num_layers * num_experts,
    }


def _missing_components(work_dir: Path, meta: dict) -> List[str]:
    summary = _component_summary(work_dir, meta)
    missing: List[str] = []
    if summary["premoe_decode"] < summary["expected_premoe_decode"]:
        missing.append(f"premoe decode graphs: {summary['premoe_decode']}/{summary['expected_premoe_decode']}")
    if summary["experts"] < summary["expected_experts"]:
        missing.append(f"expert graphs: {summary['experts']}/{summary['expected_experts']}")
    if summary["postmoe"] < 1:
        missing.append("postmoe graph")
    if summary["head"] < 1:
        missing.append("head graph")
    if not (work_dir / str(meta.get("token_embedding_file", "token_embedding.pt"))).exists():
        missing.append("token embedding")
    if not (work_dir / str(meta.get("hf_config", "hf_config"))).exists():
        missing.append("hf tokenizer/config")
    return missing


def _head_export_command(work_dir: Path, meta: dict) -> str:
    model = meta.get("loaded_model_path") or meta.get("hf_model_path") or "/path/to/model"
    premoe_quant = meta.get("split_quant_types", {}).get("premoe") or meta.get("quant_type") or "w8a8h0_sefp"
    context_length = int(meta.get("max_context_tokens", 2048))
    decode_length = int(meta.get("decode_sequence_length", 1))
    return (
        "conda run -n xh2 --no-capture-output python "
        "examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_split_moe_hmonnx.py "
        f"--model {model} --context-length {context_length} --decode-sequence-length {decode_length} "
        f"--export-parts head --premoe-quant-type {premoe_quant} --work-dir {work_dir}"
    )


class SplitMoEHMONNXRunner:
    def __init__(
        self,
        work_dir: Path,
        device: torch.device,
        exec_device: torch.device,
        cache_experts: bool = False,
        verbose: bool = False,
    ) -> None:
        self.work_dir = work_dir.resolve()
        self.meta_path = self.work_dir / "split_moe_meta.json"
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.device = device
        self.exec_device = exec_device
        self.cache_experts = cache_experts
        self.verbose = verbose
        self.num_layers = int(self.meta["num_hidden_layers"])
        self.num_experts_per_tok = int(self.meta["num_experts_per_tok"])
        self.past_seq_length = 0
        self._sessions: Dict[Path, Any] = {}
        self._layer_caches: Dict[int, Dict[str, torch.Tensor]] = {}

        tokenizer_dir = _resolve_path(self.work_dir, self.meta.get("hf_config", "hf_config"))
        embed_path = _resolve_path(self.work_dir, self.meta.get("token_embedding_file", "token_embedding.pt"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.token_embedding = _load_token_embedding(embed_path).to(self.device).eval()

        self.premoe_paths = self._resolve_premoe_paths()
        self.postmoe_path = self._resolve_component_path(
            self.meta.get("postmoe"),
            "hmonnx/postmoe/postmoe_decode_npu_agg.onnx",
        )
        head_record = self.meta.get("head") or _default_head_record(self.work_dir)
        self.head_path = self._resolve_component_path(head_record, "hmonnx/head/head.onnx")

    def close(self) -> None:
        self._sessions.clear()
        self._layer_caches.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resolve_component_path(self, record: Optional[dict], fallback: str) -> Path:
        rel_path = record.get("onnx") if isinstance(record, dict) else fallback
        return _resolve_path(self.work_dir, rel_path)

    def _resolve_premoe_paths(self) -> List[Path]:
        records = self.meta.get("premoe", {}).get("decode", {}).get("layers", [])
        if records:
            return [_resolve_path(self.work_dir, record["onnx"]) for record in records]
        return [self.work_dir / "hmonnx" / "premoe" / f"layer_{idx:03d}_premoe.onnx" for idx in range(self.num_layers)]

    def _expert_path(self, layer_idx: int, expert_idx: int) -> Path:
        return self.work_dir / "hmonnx" / "experts" / f"layer_{layer_idx:03d}" / f"expert_{expert_idx:03d}.onnx"

    def _session(self, path: Path, *, cache: bool = True):
        path = path.resolve()
        if cache and path in self._sessions:
            return self._sessions[path]
        session = _create_hmonnx_session(path, self.device, self.exec_device)
        if cache:
            self._sessions[path] = session
        return session

    def _run_session(self, path: Path, feed: Dict[str, torch.Tensor], *, cache: bool = True) -> Dict[str, torch.Tensor]:
        session = self._session(path, cache=cache)
        _, output_map = _run_hmonnx(session, feed)
        if not cache:
            del session
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return output_map

    def _cache_input(self, layer_idx: int, name: str, info) -> torch.Tensor:
        layer_cache = self._layer_caches.setdefault(layer_idx, {})
        if name not in layer_cache:
            layer_cache[name] = CacheTensor(_zeros_like_input(info, self.device))
        return layer_cache[name]

    def _cast_to_input(self, value: torch.Tensor, info) -> torch.Tensor:
        return value.to(device=self.device, dtype=info.dtype)

    def _build_premoe_feed(self, session, layer_idx: int, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        position_ids = torch.tensor([[self.past_seq_length]], dtype=torch.int32, device=self.device)
        past_seq = torch.tensor([self.past_seq_length], dtype=torch.int32, device=self.device)
        current_len = torch.tensor([1], dtype=torch.int32, device=self.device)
        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            info = session.get_input(name)
            if name in ("hidden_in", "inputs_embeds", "input_1"):
                feed[name] = self._cast_to_input(hidden, info)
            elif name in ("time_position_ids", "hight_position_ids", "height_position_ids", "width_position_ids"):
                feed[name] = position_ids.to(dtype=info.dtype)
            elif name in ("valid_length", "past_seq_length"):
                feed[name] = past_seq.to(dtype=info.dtype)
            elif name in ("current_length", "current_input_length"):
                feed[name] = current_len.to(dtype=info.dtype)
            elif name == "linear_attn_mask":
                feed[name] = torch.ones(_shape_of(info), dtype=info.dtype, device=self.device)
            elif name.startswith(("past_key_cache", "past_value_cache", "past_conv_cache", "past_recurrent_state")):
                feed[name] = self._cache_input(layer_idx, name, info)
            else:
                feed[name] = _zeros_like_input(info, self.device)
        return feed

    def _update_layer_cache(self, layer_idx: int, output_map: Dict[str, torch.Tensor]) -> None:
        layer_cache = self._layer_caches.get(layer_idx)
        if not layer_cache:
            return
        for name, cached in list(layer_cache.items()):
            if name in output_map:
                layer_cache[name] = _as_cache_value(cached, output_map[name])
                continue
            candidates: List[str] = []
            if name.startswith("past_conv_cache_"):
                suffix = name[len("past_conv_cache_"):]
                candidates.extend([f"conv_cache_out_{suffix}", f"conv_cache_out_{suffix}_0"])
            elif name.startswith("past_recurrent_state"):
                suffix = name[len("past_recurrent_state"):].lstrip("_")
                candidates.extend(["recurrent_state_out"])
                if suffix:
                    candidates.extend([f"recurrent_state_out_{suffix}", f"recurrent_state_out_{suffix}_0"])
            for output_name in candidates:
                if output_name in output_map:
                    layer_cache[name] = _as_cache_value(cached, output_map[output_name])
                    break

    def _run_expert(self, layer_idx: int, expert_idx: int, moe_input: torch.Tensor) -> torch.Tensor:
        path = self._expert_path(layer_idx, expert_idx)
        session = self._session(path, cache=self.cache_experts)
        input_name = session.get_input_names()[0]
        info = session.get_input(input_name)
        _, output_map = _run_hmonnx(session, {input_name: self._cast_to_input(moe_input, info)})
        if not self.cache_experts:
            del session
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if "expert_out" in output_map:
            return output_map["expert_out"]
        return next(iter(output_map.values()))

    def _run_postmoe(
        self,
        expert_outputs: Sequence[torch.Tensor],
        topk_gate: torch.Tensor,
        shared_out: torch.Tensor,
        residual1: torch.Tensor,
    ) -> torch.Tensor:
        session = self._session(self.postmoe_path)
        values: Dict[str, torch.Tensor] = {
            **{f"expert_out_{idx}": expert_outputs[idx] for idx in range(len(expert_outputs))},
            "topk_gate": topk_gate,
            "shared_out": shared_out,
            "residual1": residual1,
        }
        feed = {}
        for name in session.get_input_names():
            info = session.get_input(name)
            if name not in values:
                raise KeyError(f"Cannot feed postmoe input {name}; known inputs: {sorted(values)}")
            feed[name] = self._cast_to_input(values[name], info)
        _, output_map = _run_hmonnx(session, feed)
        if "hidden_out" in output_map:
            return output_map["hidden_out"]
        return next(iter(output_map.values()))

    def _run_head(self, hidden: torch.Tensor) -> torch.Tensor:
        session = self._session(self.head_path)
        input_name = session.get_input_names()[0]
        info = session.get_input(input_name)
        _, output_map = _run_hmonnx(session, {input_name: self._cast_to_input(hidden, info)})
        if "logits" in output_map:
            return output_map["logits"]
        return next(iter(output_map.values()))

    def forward_token(self, token_id: int) -> torch.Tensor:
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        hidden = self.token_embedding(token).to(self.device)
        for layer_idx, premoe_path in enumerate(self.premoe_paths):
            premoe_session = self._session(premoe_path)
            premoe_feed = self._build_premoe_feed(premoe_session, layer_idx, hidden)
            _, premoe_output = _run_hmonnx(premoe_session, premoe_feed)
            self._update_layer_cache(layer_idx, premoe_output)

            moe_input = premoe_output["moe_input"]
            topk_id = premoe_output["topk_id"].to(torch.long)
            if topk_id.numel() < self.num_experts_per_tok:
                raise ValueError(f"topk_id has too few elements: {tuple(topk_id.shape)}")
            expert_ids = [int(topk_id.reshape(-1)[idx].item()) for idx in range(self.num_experts_per_tok)]
            if self.verbose:
                print(f"  layer {layer_idx:02d}: experts={expert_ids}")
            expert_outputs = [self._run_expert(layer_idx, expert_idx, moe_input) for expert_idx in expert_ids]
            hidden = self._run_postmoe(
                expert_outputs,
                premoe_output["topk_gate"],
                premoe_output["shared_out"],
                premoe_output["residual1"],
            )
        logits = self._run_head(hidden)
        self.past_seq_length += 1
        return logits

    def forward_token_with_intermediates(
        self,
        token_id: int,
        *,
        max_layers: Optional[int] = None,
        include_layer_details: bool = False,
        use_host_postmoe: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        hidden = self.token_embedding(token).to(self.device)
        intermediates: Dict[str, torch.Tensor] = {}
        layer_limit = self.num_layers if max_layers is None else min(int(max_layers), self.num_layers)

        for layer_idx, premoe_path in enumerate(self.premoe_paths[:layer_limit]):
            premoe_session = self._session(premoe_path)
            premoe_feed = self._build_premoe_feed(premoe_session, layer_idx, hidden)
            _, premoe_output = _run_hmonnx(premoe_session, premoe_feed)
            self._update_layer_cache(layer_idx, premoe_output)

            moe_input = premoe_output["moe_input"]
            topk_id = premoe_output["topk_id"].to(torch.long)
            topk_gate = premoe_output["topk_gate"]
            shared_out = premoe_output["shared_out"]
            residual1 = premoe_output["residual1"]
            if topk_id.numel() < self.num_experts_per_tok:
                raise ValueError(f"topk_id has too few elements: {tuple(topk_id.shape)}")
            expert_ids = [int(topk_id.reshape(-1)[idx].item()) for idx in range(self.num_experts_per_tok)]
            if self.verbose:
                print(f"  layer {layer_idx:02d}: experts={expert_ids}")
            expert_outputs = [self._run_expert(layer_idx, expert_idx, moe_input) for expert_idx in expert_ids]

            hmonnx_hidden = self._run_postmoe(expert_outputs, topk_gate, shared_out, residual1)
            hidden = hmonnx_hidden
            if include_layer_details or use_host_postmoe:
                host_hidden = residual1 + shared_out
                for route_idx, expert_out in enumerate(expert_outputs):
                    host_hidden = host_hidden + expert_out * topk_gate[..., route_idx : route_idx + 1]
                intermediates[f"split_layer_{layer_idx}_hidden_out_host_postmoe"] = host_hidden
                intermediates[f"split_layer_{layer_idx}_hidden_out_hmonnx_postmoe"] = hmonnx_hidden
                if use_host_postmoe:
                    hidden = host_hidden

            intermediates[f"split_layer_{layer_idx}_hidden_out"] = hidden
            if include_layer_details:
                intermediates[f"split_layer_{layer_idx}_moe_input"] = moe_input
                intermediates[f"split_layer_{layer_idx}_topk_id"] = topk_id
                intermediates[f"split_layer_{layer_idx}_topk_gate"] = topk_gate
                intermediates[f"split_layer_{layer_idx}_shared_out"] = shared_out
                intermediates[f"split_layer_{layer_idx}_residual1"] = residual1

        logits: Optional[torch.Tensor] = None
        if layer_limit == self.num_layers:
            logits = self._run_head(hidden)
            self.past_seq_length += 1
        return logits, intermediates

    def encode_prompt(self, prompt: str, *, use_chat_template: bool, enable_thinking: bool) -> Tuple[str, List[int]]:
        if use_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                enable_thinking=enable_thinking,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        input_ids = self.tokenizer(text, padding=False, return_tensors="pt").input_ids[0].tolist()
        return text, [int(token_id) for token_id in input_ids]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        use_chat_template: bool = True,
        enable_thinking: bool = False,
    ) -> Tuple[List[int], str, List[int]]:
        prompt_text, input_ids = self.encode_prompt(
            prompt,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )
        print(f"Prompt tokens: {len(input_ids)}")
        last_logits: Optional[torch.Tensor] = None
        for idx, token_id in enumerate(input_ids):
            print(f"[prefill-by-decode] token {idx + 1}/{len(input_ids)} id={token_id}")
            last_logits = self.forward_token(token_id)
        if last_logits is None:
            raise ValueError("Prompt produced no input tokens.")

        generated: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        for step in range(max_new_tokens):
            next_token = int(torch.argmax(last_logits[:, -1, :], dim=-1).item())
            generated.append(next_token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            print(f"[generate] step {step + 1}/{max_new_tokens} id={next_token} text={text!r}")
            if eos_token_id is not None and next_token == eos_token_id:
                break
            last_logits = self.forward_token(next_token)
        return input_ids, prompt_text, generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run decode-only end-to-end validation for split Qwen3.5-MoE HMONNX parts."
    )
    parser.add_argument(
        "--work-dir",
        default=DEFAULT_WORK_DIR,
        help="Split-MoE export directory containing split_moe_meta.json.",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句话介绍你自己。",
        help="User prompt for chat-template generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Number of greedy tokens to generate after consuming the prompt.",
    )
    parser.add_argument(
        "--single-token-id",
        type=int,
        default=None,
        help="Run exactly one decode-token forward through premoe->experts->postmoe->head and print logits summary.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device used for HMONNX inputs and cached tensors.")
    parser.add_argument("--execution-device", default="cuda", help="HMONNX execution device.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check component completeness; do not run forward.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Treat --prompt as raw model text instead of applying tokenizer chat template.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to the Qwen chat template.",
    )
    parser.add_argument(
        "--cache-experts",
        action="store_true",
        help="Cache expert HMONNX sessions after first use; faster but uses much more memory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print routed expert ids for each layer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    meta_path = work_dir / "split_moe_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing split meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    summary = _component_summary(work_dir, meta)
    print("Component summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    missing = _missing_components(work_dir, meta)
    if missing:
        print("Missing components:")
        for item in missing:
            print(f"  - {item}")
        if "head graph" in missing:
            print("Export missing head with:")
            print(f"  {_head_export_command(work_dir, meta)}")
        raise SystemExit(2)
    if args.check_only:
        print("All required decode-chain components are present.")
        return

    if args.single_token_id is None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive when not using --check-only.")

    runner = SplitMoEHMONNXRunner(
        work_dir,
        device=torch.device(args.device),
        exec_device=torch.device(args.execution_device),
        cache_experts=args.cache_experts,
        verbose=args.verbose,
    )
    start = time.time()
    try:
        if args.single_token_id is not None:
            logits = runner.forward_token(args.single_token_id)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            print("\nSingle-token split-HMONNX forward succeeded.")
            print(f"Input token id: {args.single_token_id}")
            print(f"Logits shape: {tuple(logits.shape)}")
            print(f"Greedy next token id: {next_token}")
            print(f"Greedy next token text: {runner.tokenizer.decode([next_token], skip_special_tokens=False)!r}")
        else:
            input_ids, prompt_text, generated = runner.generate(
                args.prompt,
                args.max_new_tokens,
                use_chat_template=not args.no_chat_template,
                enable_thinking=args.enable_thinking,
            )
            print("\nPrompt text:")
            print(prompt_text)
            print(f"Input token ids: {input_ids}")
            print(f"Generated token ids: {generated}")
            print("Generated text:")
            print(runner.tokenizer.decode(generated, skip_special_tokens=True))
        print(f"Elapsed: {time.time() - start:.2f}s")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
