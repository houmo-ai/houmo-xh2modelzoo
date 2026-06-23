"""Compare split-HMONNX outputs against PyTorch GPTQModel layer by layer.

Usage:
  python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_split_moe_compare.py \
      --work-dir work_dirs/qwen36moe-no-rotate-attn8-shared8-n256-iter400-split-moe-premoe-w8a8h0_sefp-experts-w4a8h0_sefp \
      --token-id 12675
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeConvertConfig
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter import Qwen3_5MoeConverterXH2a
from xhquant.api import DeviceType, QuantScheme
from xhquant.core import CacheTensor
from xhquant.xhonnxruntime import HMONNXGrapInference, HMONNXInference


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


def _run_hmonnx(session, input_feed: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    outputs = session.run(input_feed)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    output_names = session.get_output_names()
    return {name: out for name, out in zip(output_names, outputs, strict=False)}


def _resolve_path(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _load_token_embedding(embed_path: Path) -> nn.Module:
    obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj
    if isinstance(obj, dict) and "weight" in obj:
        emb = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1], dtype=obj["weight"].dtype)
        emb.load_state_dict(obj)
        emb.eval()
        return emb
    raise TypeError(f"Unsupported embedding type: {type(obj)}")


def _load_gptq_model(model_dir: str, device: torch.device) -> Tuple[Any, Any]:
    """Load GPTQModel and return (native_model, wrapped_text_model)."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok("Hi", return_tensors="pt").input_ids
    config = Qwen3_5MoeConvertConfig(
        batch_size=1,
        context_length=256,
        input_sequence_length=ids.shape[1],
        quant_scheme=QuantScheme(DeviceType.XH2a, "w8a8h0_sefp"),
        quant_weight=None,
    )
    converter = Qwen3_5MoeConverterXH2a(config)
    model = converter.get_hf_model(model_dir, torch_dtype=torch.float16, device_map=device)
    model.eval()

    # Get text model (unwrap model.language_model)
    text_model = model
    if hasattr(model, "model"):
        text_model = model.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model

    return model, text_model, tok


class LayerComparator:
    def __init__(self, work_dir: str, gptq_dir: str, device_id: int = 0):
        self.work_dir = Path(work_dir).resolve()
        self.device = torch.device(f"cuda:{device_id}")
        self.exec_device = torch.device(f"cuda:{device_id}")

        # Load split meta
        meta_path = self.work_dir / "split_moe_meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.num_layers = int(self.meta["num_hidden_layers"])
        self.num_experts_per_tok = int(self.meta["num_experts_per_tok"])
        self.num_experts = int(self.meta["num_experts"])
        self.hidden_size = int(self.meta["hidden_size"])

        # Load token embedding
        embed_path = _resolve_path(self.work_dir, self.meta["token_embedding_file"])
        self.token_embedding = _load_token_embedding(embed_path).to(self.device).eval()

        # Load GPTQModel
        print(f"Loading GPTQModel from {gptq_dir} onto {self.device}...")
        self.gptq_model, self.text_model, self.tokenizer = _load_gptq_model(gptq_dir, self.device)
        self.gptq_dtype = next(self.text_model.parameters()).dtype

        # Resolve split paths
        self.premoe_paths = [
            self.work_dir / "hmonnx" / "premoe" / f"layer_{idx:03d}_premoe.onnx"
            for idx in range(self.num_layers)
        ]
        self.postmoe_path = self.work_dir / "hmonnx" / "postmoe" / "postmoe_decode_npu_agg.onnx"
        self.head_path = self.work_dir / "hmonnx" / "head" / "head.onnx"

        self._premoe_sessions: Dict[int, Any] = {}
        self._expert_sessions: Dict[Tuple[int, int], Any] = {}
        self._postmoe_session = None
        self._head_session = None
        self._layer_caches: Dict[int, Dict[str, torch.Tensor]] = {}

    def close(self):
        self._premoe_sessions.clear()
        self._expert_sessions.clear()
        self._postmoe_session = None
        self._head_session = None
        self._layer_caches.clear()
        del self.gptq_model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- PyTorch reference -------------------------------------------------
    def _pt_forward_one_token(self, token_id: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run one token through GPTQModel, capturing hidden states after each layer."""
        with torch.no_grad():
            # Embed
            embed = self.text_model.get_input_embeddings()
            embed_device = embed.weight.device
            token = torch.tensor([[token_id]], dtype=torch.long, device=embed_device)
            hidden = embed(token).to(dtype=self.gptq_dtype)

            layer_hidden_states = []
            for layer_idx, layer in enumerate(self.text_model.layers):
                layer_device = next(layer.parameters()).device
                hidden = hidden.to(layer_device)
                hidden = layer(hidden)[0]  # layer returns (hidden, ...)
                layer_hidden_states.append(hidden.clone())

            # Final norm + lm_head
            norm_device = next(self.text_model.norm.parameters()).device
            hidden = hidden.to(norm_device)
            normed = self.text_model.norm(hidden)
            lm_head = self.gptq_model.lm_head if hasattr(self.gptq_model, "lm_head") else self.text_model.lm_head
            normed = normed.to(next(lm_head.parameters()).device)
            logits = lm_head(normed)

        return logits, layer_hidden_states

    def _pt_forward_with_intermediates(self, token_id: int) -> Dict[str, torch.Tensor]:
        """Run one token with hooks to capture premoe-level intermediates."""
        intermediates: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            embed = self.text_model.get_input_embeddings()
            embed_device = embed.weight.device
            token = torch.tensor([[token_id]], dtype=torch.long, device=embed_device)
            hidden = embed(token).to(dtype=self.gptq_dtype)

            for layer_idx, layer in enumerate(self.text_model.layers):
                layer_device = next(layer.parameters()).device
                hidden = hidden.to(layer_device)
                # input_layernorm + attention
                residual = hidden
                normed = layer.input_layernorm(hidden)

                # Attention
                attn_out = layer.self_attn(normed)[0]
                hidden = residual + attn_out

                # post_attention_layernorm
                residual = hidden
                normed = layer.post_attention_layernorm(hidden)

                # Router
                router_logits = layer.mlp.gate(normed)
                router_prob = torch.softmax(router_logits, dim=-1)
                topk_gate, topk_id = torch.topk(router_prob, k=self.num_experts_per_tok, dim=-1)
                topk_gate = topk_gate / topk_gate.sum(dim=-1, keepdim=True)

                # Shared expert
                shared_out = layer.mlp.shared_expert(normed)
                shared_out = torch.sigmoid(layer.mlp.shared_expert_gate(normed)) * shared_out

                # Save premoe outputs
                intermediates[f"layer_{layer_idx}_residual1"] = residual.clone()
                intermediates[f"layer_{layer_idx}_moe_input"] = normed.clone()
                intermediates[f"layer_{layer_idx}_topk_id"] = topk_id.clone()
                intermediates[f"layer_{layer_idx}_topk_gate"] = topk_gate.clone()
                intermediates[f"layer_{layer_idx}_shared_out"] = shared_out.clone()

                # Routed experts (PyTorch)
                expert_outs = []
                flat_id = topk_id.reshape(-1)
                for k_idx in range(self.num_experts_per_tok):
                    expert_idx = int(flat_id[k_idx].item())
                    expert = layer.mlp.experts[expert_idx]
                    expert_out = expert.down_proj(
                        torch.nn.functional.silu(expert.gate_proj(normed)) * expert.up_proj(normed)
                    )
                    expert_outs.append(expert_out)
                    intermediates[f"layer_{layer_idx}_expert_{expert_idx}_out"] = expert_out.clone()

                # Aggregate
                routed_sum = expert_outs[0] * topk_gate[..., 0:1]
                for k_idx in range(1, self.num_experts_per_tok):
                    routed_sum = routed_sum + expert_outs[k_idx] * topk_gate[..., k_idx : k_idx + 1]
                hidden = residual + shared_out + routed_sum

            # Head
            norm_device = next(self.text_model.norm.parameters()).device
            hidden = hidden.to(norm_device)
            normed = self.text_model.norm(hidden)
            lm_head = self.gptq_model.lm_head if hasattr(self.gptq_model, "lm_head") else self.text_model.lm_head
            normed = normed.to(next(lm_head.parameters()).device)
            logits = lm_head(normed)
            intermediates["final_logits"] = logits.clone()
            intermediates["final_hidden"] = hidden.clone()

        return intermediates

    # ---- Split HMONNX ------------------------------------------------------
    def _premoe_session(self, layer_idx: int):
        if layer_idx not in self._premoe_sessions:
            path = self.premoe_paths[layer_idx]
            self._premoe_sessions[layer_idx] = _create_hmonnx_session(path, self.device, self.exec_device)
        return self._premoe_sessions[layer_idx]

    def _expert_session(self, layer_idx: int, expert_idx: int):
        key = (layer_idx, expert_idx)
        if key not in self._expert_sessions:
            path = self.work_dir / "hmonnx" / "experts" / f"layer_{layer_idx:03d}" / f"expert_{expert_idx:03d}.onnx"
            self._expert_sessions[key] = _create_hmonnx_session(path, self.device, self.exec_device)
        return self._expert_sessions[key]

    def _postmoe_sess(self):
        if self._postmoe_session is None:
            self._postmoe_session = _create_hmonnx_session(self.postmoe_path, self.device, self.exec_device)
        return self._postmoe_session

    def _head_sess(self):
        if self._head_session is None:
            self._head_session = _create_hmonnx_session(self.head_path, self.device, self.exec_device)
        return self._head_session

    def _cache_input(self, layer_idx: int, name: str, session) -> torch.Tensor:
        info = session.get_input(name)
        shape = tuple(int(d) for d in info.shape)
        layer_cache = self._layer_caches.setdefault(layer_idx, {})
        if name not in layer_cache:
            layer_cache[name] = CacheTensor(torch.zeros(shape, dtype=info.dtype, device=self.device))
        return layer_cache[name]

    def _build_premoe_feed(self, session, layer_idx: int, hidden: torch.Tensor, past_seq_len: int) -> Dict[str, torch.Tensor]:
        pos = torch.tensor([[past_seq_len]], dtype=torch.int32, device=self.device)
        past = torch.tensor([past_seq_len], dtype=torch.int32, device=self.device)
        cur = torch.tensor([1], dtype=torch.int32, device=self.device)
        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            info = session.get_input(name)
            if name in ("hidden_in", "inputs_embeds", "input_1"):
                feed[name] = hidden.to(dtype=info.dtype)
            elif name in ("time_position_ids", "hight_position_ids", "height_position_ids", "width_position_ids"):
                feed[name] = pos.to(dtype=info.dtype)
            elif name in ("valid_length", "past_seq_length"):
                feed[name] = past.to(dtype=info.dtype)
            elif name in ("current_length", "current_input_length"):
                feed[name] = cur.to(dtype=info.dtype)
            elif name == "linear_attn_mask":
                feed[name] = torch.ones(tuple(int(d) for d in info.shape), dtype=info.dtype, device=self.device)
            elif name.startswith(("past_key_cache", "past_value_cache", "past_conv_cache", "past_recurrent_state")):
                feed[name] = self._cache_input(layer_idx, name, session)
            else:
                feed[name] = torch.zeros(tuple(int(d) for d in info.shape), dtype=info.dtype, device=self.device)
        return feed

    def _split_forward_one_token(self, token_id: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run one token through split HMONNX, capturing all intermediate outputs."""
        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        hidden = self.token_embedding(token).to(self.device)
        intermediates: Dict[str, torch.Tensor] = {}
        past_seq_len = 0

        for layer_idx in range(self.num_layers):
            session = self._premoe_session(layer_idx)
            feed = self._build_premoe_feed(session, layer_idx, hidden, past_seq_len)
            outputs = _run_hmonnx(session, feed)

            moe_input = outputs["moe_input"]
            topk_id = outputs["topk_id"].to(torch.long)
            topk_gate = outputs["topk_gate"]
            shared_out = outputs["shared_out"]
            residual1 = outputs["residual1"]

            intermediates[f"split_layer_{layer_idx}_moe_input"] = moe_input.clone()
            intermediates[f"split_layer_{layer_idx}_topk_id"] = topk_id.clone()
            intermediates[f"split_layer_{layer_idx}_topk_gate"] = topk_gate.clone()
            intermediates[f"split_layer_{layer_idx}_shared_out"] = shared_out.clone()
            intermediates[f"split_layer_{layer_idx}_residual1"] = residual1.clone()

            # Update caches
            for name in list(self._layer_caches.get(layer_idx, {})):
                if name in outputs:
                    self._layer_caches[layer_idx][name] = outputs[name]

            # Run experts
            flat_id = topk_id.reshape(-1)
            expert_outs = []
            for k_idx in range(self.num_experts_per_tok):
                expert_idx = int(flat_id[k_idx].item())
                expert_sess = self._expert_session(layer_idx, expert_idx)
                einfo = expert_sess.get_input(expert_sess.get_input_names()[0])
                eout = _run_hmonnx(expert_sess, {"expert_input": moe_input.to(dtype=einfo.dtype)})
                expert_out = eout.get("expert_out", next(iter(eout.values())))
                expert_outs.append(expert_out)
                intermediates[f"split_layer_{layer_idx}_expert_{expert_idx}_out"] = expert_out.clone()

            # PostMoE
            post_sess = self._postmoe_sess()
            post_feed: Dict[str, torch.Tensor] = {}
            for name in post_sess.get_input_names():
                pinfo = post_sess.get_input(name)
                if name.startswith("expert_out_"):
                    idx = int(name.split("_")[-1])
                    post_feed[name] = expert_outs[idx].to(dtype=pinfo.dtype)
                elif name == "topk_gate":
                    post_feed[name] = topk_gate.to(dtype=pinfo.dtype)
                elif name == "shared_out":
                    post_feed[name] = shared_out.to(dtype=pinfo.dtype)
                elif name == "residual1":
                    post_feed[name] = residual1.to(dtype=pinfo.dtype)
            post_outputs = _run_hmonnx(post_sess, post_feed)
            hidden = post_outputs.get("hidden_out", next(iter(post_outputs.values())))
            intermediates[f"split_layer_{layer_idx}_hidden_out"] = hidden.clone()

            past_seq_len += 1

        # Head
        head_sess = self._head_sess()
        head_info = head_sess.get_input(head_sess.get_input_names()[0])
        head_outputs = _run_hmonnx(head_sess, {"hidden_in": hidden.to(dtype=head_info.dtype)})
        logits = head_outputs.get("logits", next(iter(head_outputs.values())))
        intermediates["split_final_logits"] = logits.clone()
        intermediates["split_final_hidden"] = hidden.clone()

        return logits, intermediates

    # ---- Comparison ---------------------------------------------------------
    def compare(self, token_id: int):
        print(f"\n{'='*60}")
        print(f"Comparing PyTorch GPTQModel vs Split HMONNX")
        print(f"Token ID: {token_id}, Token: {self.tokenizer.decode([token_id])!r}")
        print(f"{'='*60}")

        # PyTorch reference with intermediates
        print("\n[1/2] Running PyTorch GPTQModel with intermediate hooks...")
        t0 = time.time()
        pt_intermediates = self._pt_forward_with_intermediates(token_id)
        pt_logits = pt_intermediates["final_logits"]
        pt_time = time.time() - t0
        print(f"  Done in {pt_time:.1f}s, logits shape: {tuple(pt_logits.shape)}")

        # Split HMONNX
        print("\n[2/2] Running Split HMONNX...")
        t0 = time.time()
        split_logits, split_intermediates = self._split_forward_one_token(token_id)
        split_time = time.time() - t0
        print(f"  Done in {split_time:.1f}s, logits shape: {tuple(split_logits.shape)}")

        # Compare logits
        print(f"\n{'='*60}")
        print("LOGITS COMPARISON")
        print(f"{'='*60}")
        pt_next = int(torch.argmax(pt_logits[:, -1, :], dim=-1).item())
        split_next = int(torch.argmax(split_logits[:, -1, :], dim=-1).item())
        logits_diff = (pt_logits.float().cpu() - split_logits.float().cpu()).abs()

        print(f"PyTorch  next token: {pt_next} ({self.tokenizer.decode([pt_next])!r})")
        print(f"Split   next token: {split_next} ({self.tokenizer.decode([split_next])!r})")
        print(f"Logits match: {pt_next == split_next}")
        print(f"Logits max abs diff: {logits_diff.max().item():.6e}")
        print(f"Logits mean abs diff: {logits_diff.mean().item():.6e}")

        # Compare per-layer hidden states
        print(f"\n{'='*60}")
        print("PER-LAYER HIDDEN STATE COMPARISON")
        print(f"{'='*60}")
        all_match = pt_next == split_next
        max_layer_diff = 0.0
        for layer_idx in range(self.num_layers):
            pt_key = f"layer_{layer_idx}_residual1"
            split_key = f"split_layer_{layer_idx}_residual1"
            # Compare hidden_out (after postmoe)
            pt_hidden = pt_intermediates.get(f"final_hidden")
            split_hidden = split_intermediates.get(f"split_layer_{layer_idx}_hidden_out")
            if split_hidden is not None and layer_idx == self.num_layers - 1:
                # Compare last layer output vs final hidden
                if pt_hidden is not None:
                    diff = (pt_hidden.float().cpu() - split_hidden.float().cpu()).abs()
                    max_layer_diff = max(max_layer_diff, diff.max().item())
                    print(f"  Final hidden: max_abs_diff={diff.max().item():.6e}, mean_abs_diff={diff.mean().item():.6e}")

            # Compare moe_input
            pt_moe = pt_intermediates.get(f"layer_{layer_idx}_moe_input")
            split_moe = split_intermediates.get(f"split_layer_{layer_idx}_moe_input")
            if pt_moe is not None and split_moe is not None:
                diff = (pt_moe.float().cpu() - split_moe.float().cpu()).abs()
                max_layer_diff = max(max_layer_diff, diff.max().item())
                if layer_idx == 0 or diff.max().item() > 1e-3:
                    print(f"  Layer {layer_idx:02d} moe_input:  max_abs_diff={diff.max().item():.6e}, mean_abs_diff={diff.mean().item():.6e}")

        if max_layer_diff < 1e-5:
            print("\n*** ALL LAYERS MATCH (max diff < 1e-5) ***")
        else:
            print(f"\n*** MISMATCH DETECTED (max layer diff = {max_layer_diff:.6e}) ***")

        return all_match


def parse_args():
    parser = argparse.ArgumentParser(description="Compare split HMONNX vs PyTorch GPTQModel")
    parser.add_argument("--work-dir", required=True, help="Split MoE work directory")
    parser.add_argument("--gptq-dir", default="/data01/datasets/qwen36moe-no-rotate-attn8-shared8-n256-iter400")
    parser.add_argument("--token-id", type=int, default=12675, help="Token ID to test (12675=Hi)")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    comparator = LayerComparator(args.work_dir, args.gptq_dir, device_id=0)
    try:
        comparator.compare(args.token_id)
    finally:
        comparator.close()


if __name__ == "__main__":
    main()
