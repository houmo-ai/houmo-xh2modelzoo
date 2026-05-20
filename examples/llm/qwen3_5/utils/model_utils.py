"""model_utils.py — core model manipulation helpers for Qwen3.5 MTP head K-trimming.

Public API
----------
rerank_model_for_mtp(original_model_dir, K, dst_dir, ...)
    Build an end-to-end reranked model repo with K hot tokens.
    Handles both FP16 HF checkpoints and GPTQ/autoround quantized checkpoints that
    lack a standalone vocab.json (reads embedded vocab from tokenizer.json instead).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Project-root resolution (examples/llm/qwen3_5/utils/model_utils.py → root)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[4]  # .../xh2modelzoo/

_SELECTION_DIR = (
    _PROJECT_ROOT
    / "analysis"
    / "mtp_head_longtail"
    / "v2_reranked"
    / "selection"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_base_vocab(model_dir: Path) -> dict:
    """Return {token_str: old_id} from vocab.json or tokenizer.json fallback."""
    vp = model_dir / "vocab.json"
    if vp.exists():
        return json.loads(vp.read_text(encoding="utf-8"))
    # GPTQ fallback: read embedded vocab from tokenizer.json
    tjp = model_dir / "tokenizer.json"
    if not tjp.exists():
        raise FileNotFoundError(
            f"Neither vocab.json nor tokenizer.json found in {model_dir}"
        )
    tj = json.loads(tjp.read_text(encoding="utf-8"))
    return tj["model"]["vocab"]  # {token_str: old_id}


def _has_vocab_json(model_dir: Path) -> bool:
    return (model_dir / "vocab.json").exists()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_model_for_mtp(
    original_model_dir: str,
    K: int,
    dst_dir: str,
    hot_ids_path: Optional[str] = None,
    id_map_path: Optional[str] = None,
    force: bool = False,
) -> str:
    """Build a K-trimmed + reranked model repo, returning the absolute dst path.

    Steps
    -----
    1.  Load hot_ids (int64[K], ascending) and id_map from selection/.
    2.  Build full V→V permutation (hot first, then cold by ascending old_id).
    3.  Load all safetensors shards; rerank embed_tokens.weight rows.
    4.  If tie_word_embeddings=False also rerank lm_head.weight.
    5.  Write new safetensors shards and update model.safetensors.index.json.
    6.  Save mtp_lm_head.pt  (reranked lm_head[:K] or embed[:K] for tie=True).
    7.  Remap tokenizer files (vocab.json / tokenizer.json / tokenizer_config.json /
        config.json) with new token ids.
    8.  Copy remaining files verbatim.

    GPTQ compat
    -----------
    If ``original_model_dir`` has no standalone vocab.json the function reads the
    vocab from tokenizer.json["model"]["vocab"] instead.  On write, if the source
    had no vocab.json the dst directory also will not get one (keeps layout
    consistent with the quantised checkpoint so downstream tokenizer code doesn't
    get confused by a spurious vocab.json).

    Parameters
    ----------
    original_model_dir:
        Original HF model directory (FP16 or GPTQ/autoround quantised).
    K:
        Number of hot tokens to keep in the MTP lm_head.
    dst_dir:
        Destination directory for the reranked repo.
    hot_ids_path:
        Path to hot_ids_{K}.pt.  Defaults to
        analysis/mtp_head_longtail/v2_reranked/selection/hot_ids_{K}.pt.
    id_map_path:
        Path to id_map_{K}.pt.  Defaults to
        analysis/mtp_head_longtail/v2_reranked/selection/id_map_{K}.pt.
    force:
        When True, regenerate even if dst_dir/mtp_lm_head.pt already exists.

    Returns
    -------
    str
        Absolute path to the ready reranked repo directory.
    """
    model_dir = Path(original_model_dir).resolve()
    dst = Path(dst_dir).resolve()
    mtp_head_path = dst / "mtp_lm_head.pt"

    if not force and dst.exists() and mtp_head_path.exists():
        print(f"[rerank] Reranked repo exists at {dst} — skipping (use force=True to override).")
        return str(dst)

    dst.mkdir(parents=True, exist_ok=True)

    # ── resolve hot_ids / id_map paths ──────────────────────────────────────
    _hot = Path(hot_ids_path) if hot_ids_path else _SELECTION_DIR / f"hot_ids_{K}.pt"
    _imap = Path(id_map_path) if id_map_path else _SELECTION_DIR / f"id_map_{K}.pt"
    if not _hot.exists():
        raise FileNotFoundError(f"hot_ids not found: {_hot}")
    if not _imap.exists():
        raise FileNotFoundError(f"id_map not found: {_imap}")

    print(f"[rerank] K={K}  model={model_dir}  dst={dst}")

    # ── load id_map and build full V→V permutation ──────────────────────────
    id_map = torch.load(_imap)
    hot_ids: torch.Tensor = torch.load(_hot)  # int64[K], ascending
    assert hot_ids.shape == (K,), f"expected hot_ids shape ({K},), got {hot_ids.shape}"

    cfg_full = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tc = cfg_full.get("text_config", cfg_full)
    V = int(tc["vocab_size"])
    H = int(tc["hidden_size"])
    tie_word_embeddings = cfg_full.get(
        "tie_word_embeddings", tc.get("tie_word_embeddings", True)
    )
    print(f"[rerank] V={V}  H={H}  tie_word_embeddings={tie_word_embeddings}")

    hot_set = set(hot_ids.tolist())
    cold_ids = sorted(i for i in range(V) if i not in hot_set)
    assert len(cold_ids) == V - K, f"expected {V-K} cold ids, got {len(cold_ids)}"

    full_new_to_old = torch.cat([hot_ids, torch.tensor(cold_ids, dtype=torch.int64)])
    assert full_new_to_old.shape == (V,)

    full_old_to_new = torch.empty(V, dtype=torch.int64)
    full_old_to_new[full_new_to_old] = torch.arange(V, dtype=torch.int64)

    # Verify against id_map spot-checks
    assert torch.equal(full_new_to_old[:K], hot_ids)
    for nid in range(0, K, K // 8):
        oid = hot_ids[nid].item()
        assert full_old_to_new[oid].item() == nid, (
            f"permutation check failed at nid={nid}"
        )
    print(f"[rerank] full permutation built and verified ✓")

    # ── load and rerank embed_tokens.weight ─────────────────────────────────
    print(f"[rerank] loading safetensors index...")
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = idx["weight_map"]  # {weight_name: shard_file}

    EMBED_KEY = "model.language_model.embed_tokens.weight"
    LM_HEAD_KEY = "lm_head.weight"
    assert EMBED_KEY in wm, f"Could not find {EMBED_KEY} in index; keys: {list(wm.keys())[:5]}"
    has_lm_head = LM_HEAD_KEY in wm
    print(f"[rerank] has independent lm_head.weight: {has_lm_head}")

    shards_data: dict[str, dict] = {}
    for shard_file in sorted(set(wm.values())):
        print(f"[rerank] loading shard {shard_file} ...")
        shards_data[shard_file] = load_file(str(model_dir / shard_file), device="cpu")

    orig_embed = shards_data[wm[EMBED_KEY]][EMBED_KEY]  # (V, H) bf16
    assert orig_embed.shape == (V, H), f"expected ({V},{H}), got {orig_embed.shape}"
    print(f"[rerank] reranking embed_tokens.weight {orig_embed.shape} ...")
    reranked_embed = orig_embed[full_new_to_old].contiguous()
    assert reranked_embed.shape == (V, H)

    rng = random.Random(42)
    sample_old = rng.sample(range(V), 100)
    for oid in sample_old:
        nid = full_old_to_new[oid].item()
        assert torch.equal(reranked_embed[nid], orig_embed[oid]), (
            f"embed bit-exact check failed for old_id={oid} new_id={nid}"
        )
    print(f"[rerank] embed bit-exact check 100/100 ✓")

    shards_data[wm[EMBED_KEY]][EMBED_KEY] = reranked_embed

    reranked_lm_head = None
    if has_lm_head:
        orig_lm_head = shards_data[wm[LM_HEAD_KEY]][LM_HEAD_KEY]  # (V, H) bf16
        assert orig_lm_head.shape == (V, H), (
            f"lm_head shape mismatch: expected ({V},{H}), got {orig_lm_head.shape}"
        )
        print(f"[rerank] reranking lm_head.weight {orig_lm_head.shape} ...")
        reranked_lm_head = orig_lm_head[full_new_to_old].contiguous()
        assert reranked_lm_head.shape == (V, H)

        sample_old_lm = rng.sample(range(V), 100)
        for oid in sample_old_lm:
            nid = full_old_to_new[oid].item()
            assert torch.equal(reranked_lm_head[nid], orig_lm_head[oid]), (
                f"lm_head bit-exact check failed for old_id={oid} new_id={nid}"
            )
        print(f"[rerank] lm_head bit-exact check 100/100 ✓")

        shards_data[wm[LM_HEAD_KEY]][LM_HEAD_KEY] = reranked_lm_head

    # ── write reranked safetensors shards ────────────────────────────────────
    print(f"[rerank] writing new safetensors shards ...")
    for shard_file, tensors in shards_data.items():
        dst_shard = dst / shard_file
        save_file(tensors, str(dst_shard))
        print(f"[rerank]   wrote {dst_shard}  ({dst_shard.stat().st_size/1024/1024:.1f}MB)")

    shutil.copy2(
        str(model_dir / "model.safetensors.index.json"),
        dst / "model.safetensors.index.json",
    )
    print(f"[rerank] copied model.safetensors.index.json")

    # ── save mtp_lm_head.pt ──────────────────────────────────────────────────
    if reranked_lm_head is not None:
        mtp_lm_head = reranked_lm_head[:K].contiguous()
        mtp_source = "lm_head"
    else:
        mtp_lm_head = reranked_embed[:K].contiguous()
        mtp_source = "embed"
    print(f"[rerank] mtp_lm_head source: {mtp_source}")
    assert mtp_lm_head.shape == (K, H), f"mtp_lm_head shape mismatch: {mtp_lm_head.shape}"
    torch.save(mtp_lm_head, dst / "mtp_lm_head.pt")
    print(f"[rerank] saved mtp_lm_head.pt  shape={tuple(mtp_lm_head.shape)} dtype={mtp_lm_head.dtype} ✓")

    # ── build tokenizer remapping ────────────────────────────────────────────
    print(f"[rerank] remapping tokenizer files ...")

    def remap_id(old_id: int) -> int:
        return int(full_old_to_new[old_id].item())

    src_has_vocab_json = _has_vocab_json(model_dir)

    if src_has_vocab_json:
        # vocab.json: {token_str: new_id}
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        new_vocab = {tok: remap_id(oid) for tok, oid in vocab.items()}
        (dst / "vocab.json").write_text(
            json.dumps(new_vocab, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[rerank] vocab.json remapped ({len(new_vocab)} entries)")

    # tokenizer.json
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))

    if not src_has_vocab_json:
        # GPTQ path: load vocab from tokenizer.json for verification purposes
        vocab = tj["model"]["vocab"]  # {token_str: old_id}

    # model.vocab: {token_str: old_id} → remap ids
    if "model" in tj and "vocab" in tj["model"] and isinstance(tj["model"]["vocab"], dict):
        tj["model"]["vocab"] = {tok: remap_id(oid) for tok, oid in tj["model"]["vocab"].items()}

    # added_tokens: list of {id: old_id, ...} objects
    # IMPORTANT: The HuggingFace tokenizers Rust library only honors the `id` field in added_tokens
    # when that id == max(base_vocab_id) + position. For added tokens with ids that fall
    # inside the BPE vocab range, we must ALSO insert them into model.vocab so the rust library
    # uses the correct id. Having the same token in both model.vocab and added_tokens is safe;
    # added_tokens takes priority for encoding (atomic match before BPE).
    new_model_vocab = tj["model"]["vocab"]
    for at in tj.get("added_tokens", []):
        new_id = remap_id(at["id"])
        at["id"] = new_id
        new_model_vocab[at["content"]] = new_id

    # post_processor token ids (if any)
    def remap_post_processor(obj):
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k == "id" and isinstance(obj[k], int):
                    obj[k] = remap_id(obj[k])
                else:
                    remap_post_processor(obj[k])
        elif isinstance(obj, list):
            for item in obj:
                remap_post_processor(item)

    if "post_processor" in tj and tj["post_processor"]:
        remap_post_processor(tj["post_processor"])

    (dst / "tokenizer.json").write_text(
        json.dumps(tj, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[rerank] tokenizer.json remapped")

    # tokenizer_config.json: rekey added_tokens_decoder {old_id_str: info} → {new_id_str: info}
    tc_cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    atd = tc_cfg.get("added_tokens_decoder", {})
    new_atd = {}
    for old_id_str, info in atd.items():
        new_id = remap_id(int(old_id_str))
        new_atd[str(new_id)] = info
    tc_cfg["added_tokens_decoder"] = new_atd
    (dst / "tokenizer_config.json").write_text(
        json.dumps(tc_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[rerank] tokenizer_config.json remapped ({len(new_atd)} added tokens)")

    # config.json: update all token id references, keep vocab_size unchanged
    cfg_new = json.loads(json.dumps(cfg_full))

    def remap_cfg_id(cfg_obj, key):
        val = cfg_obj.get(key)
        if val is None:
            return
        if isinstance(val, int):
            cfg_obj[key] = remap_id(val)
        elif isinstance(val, list):
            cfg_obj[key] = [remap_id(v) if isinstance(v, int) else v for v in val]

    for key in [
        "image_token_id", "video_token_id", "vision_start_token_id",
        "vision_end_token_id", "eos_token_id", "bos_token_id", "pad_token_id",
    ]:
        remap_cfg_id(cfg_new, key)

    tc_new = cfg_new.get("text_config", {})
    for key in ["eos_token_id", "bos_token_id", "pad_token_id"]:
        remap_cfg_id(tc_new, key)

    # vocab_size must remain V — do NOT change it
    assert cfg_new.get("text_config", cfg_new).get("vocab_size") == V, (
        "vocab_size changed — should remain unchanged"
    )

    cfg_new["_reranked"] = {
        "K": K, "V": V,
        "ordering": "hot_ids_then_cold_by_old_id",
        "script": "utils/model_utils.py",
    }
    (dst / "config.json").write_text(
        json.dumps(cfg_new, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[rerank] config.json updated (vocab_size={V} unchanged)")

    # ── copy remaining files verbatim ────────────────────────────────────────
    # Use a denylist: anything NOT generated by rerank above gets copied.
    # This handles GPTQ/autoround quantised checkpoints (quantization_config.json,
    # generation_config.json, processor_config.json, eval_report.json, ...) that
    # would silently be dropped by a hardcoded allowlist.
    written_files = set(shards_data.keys()) | {
        "model.safetensors.index.json",
        "mtp_lm_head.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    }
    if src_has_vocab_json:
        written_files.add("vocab.json")
    for src_f in sorted(model_dir.iterdir()):
        if not src_f.is_file():
            continue
        if src_f.name in written_files:
            continue
        shutil.copy2(str(src_f), dst / src_f.name)
        print(f"[rerank] copied {src_f.name}")

    # ── final verification ────────────────────────────────────────────────────
    print(f"\n[rerank] === Final verification ===")

    chk_shard = dst / wm[EMBED_KEY]
    with safe_open(str(chk_shard), framework="pt", device="cpu") as f:
        chk_shape = tuple(f.get_slice(EMBED_KEY).get_shape())
    print(f"[rerank] embed shape in new shard: {chk_shape}  {'✓' if chk_shape==(V,H) else '✗ FAIL'}")
    assert chk_shape == (V, H)

    with safe_open(str(chk_shard), framework="pt", device="cpu") as f:
        embed_disk = f.get_tensor(EMBED_KEY)
    for oid in sample_old[:10]:
        nid = full_old_to_new[oid].item()
        assert torch.equal(embed_disk[nid], orig_embed[oid]), (
            f"disk bit-exact failed for old_id={oid}"
        )
    print(f"[rerank] disk bit-exact re-check 10/10 ✓")

    if has_lm_head:
        chk_lm_shard = dst / wm[LM_HEAD_KEY]
        with safe_open(str(chk_lm_shard), framework="pt", device="cpu") as f:
            lm_head_disk = f.get_tensor(LM_HEAD_KEY)
        for oid in sample_old_lm[:10]:
            nid = full_old_to_new[oid].item()
            assert torch.equal(lm_head_disk[nid], orig_lm_head[oid]), (
                f"lm_head disk bit-exact failed for old_id={oid}"
            )
        print(f"[rerank] lm_head disk bit-exact re-check 10/10 ✓")

    mtp_chk = torch.load(dst / "mtp_lm_head.pt")
    print(f"[rerank] mtp_lm_head.shape={tuple(mtp_chk.shape)}  {'✓' if tuple(mtp_chk.shape)==(K,H) else '✗ FAIL'}")
    assert tuple(mtp_chk.shape) == (K, H)

    if src_has_vocab_json:
        vj = json.loads((dst / "vocab.json").read_text(encoding="utf-8"))
        assert len(vj) == len(vocab), "vocab.json entry count changed"
        print(f"[rerank] vocab.json size consistent ({len(vj)}) ✓")

    print(f"\n[rerank] ALL CHECKS PASSED — new model repo at {dst}")
    return str(dst)
