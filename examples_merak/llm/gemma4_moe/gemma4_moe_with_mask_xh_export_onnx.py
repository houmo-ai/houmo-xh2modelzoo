"""Gemma 4 MoE 26B-A4B-it — LLM (with-mask) Merak-side ONNX export.

Strictly aligned with the source script
``xhquant_llm/examples/gemma4_moe/gemma4_moe_with_mask_xh2a_llm_export.py``.

Stages: WRAPED → FRONTEND → QUANTED_ALIGNED → EXPORTED → ONNX [→ GOLDEN]

The vision encoder is exported separately by
``gemma4_moe_visual_xh_export_onnx.py``; this script handles only the LLM
text path with explicit ``local_attention_mask`` / ``global_attention_mask``
inputs.

Default arguments mirror the source script while keeping the merak naming
convention for files and work directories.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path


def bootstrap_runtime() -> None:
    """Make vendored transformers 5.5 importable for Gemma4.

    Mirrors the bootstrap used by the rest of the merak Gemma4 entry points so
    that ``transformers.models.gemma4`` is available in the active xhquant env.
    """

    workspace_root = Path(__file__).resolve().parents[4]
    vendor_transformers = workspace_root / ".vendor" / "python" / "transformers"
    if vendor_transformers.exists():
        link_root = Path(tempfile.gettempdir()) / "xh2a_vendor_transformers_only"
        link_root.mkdir(parents=True, exist_ok=True)
        link_path = link_root / "transformers"
        if link_path.exists() or link_path.is_symlink():
            if not link_path.is_symlink() or link_path.resolve() != vendor_transformers:
                if link_path.is_dir() and not link_path.is_symlink():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
        if not link_path.exists():
            link_path.symlink_to(vendor_transformers, target_is_directory=True)
        sys.path.insert(0, str(link_root))


bootstrap_runtime()

import torch
import torch.nn as nn

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager
from xhquant.api import Config, get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


# ──────────────────────────────────────────────────────────────────────────
# Helpers (mirrors the source script behavior while staying repo-local)
# ──────────────────────────────────────────────────────────────────────────


def _resolve_hf_asset(primary_dir: str, fallback_dir: str | None, name: str) -> Path | None:
    primary = Path(primary_dir) / name
    if primary.exists():
        return primary
    if fallback_dir is None:
        return None
    fallback = Path(fallback_dir) / name
    return fallback if fallback.exists() else None


def _decode_next_token(tokenizer, logits: torch.Tensor):
    logits = logits.detach()
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    next_id = torch.argmax(logits, dim=-1, keepdim=True)
    text = tokenizer.batch_decode(next_id, skip_special_tokens=True)
    return next_id, text


def _extract_generated_reply(tokenizer, input_ids: torch.Tensor, generated_ids: torch.Tensor):
    prompt_len = int(input_ids.shape[-1])
    if generated_ids.dim() != 2:
        raise RuntimeError(f"Unexpected generated_ids shape: {tuple(generated_ids.shape)}")
    if int(generated_ids.shape[-1]) <= prompt_len:
        raise RuntimeError("Generation did not append any new token.")
    output_ids = generated_ids[0][prompt_len:].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return output_ids, output_text


def _reset_validation_runtime(model) -> None:
    model.release_inference_model()
    if getattr(model, "hf_compatible_model", None) is not None:
        model.hf_compatible_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_validation_input_dtype(stage_name: str, default_dtype: torch.dtype) -> torch.dtype:
    if stage_name.startswith("quanted"):
        return torch.float16
    return default_dtype


def _should_move_validation_model(stage_name: str) -> bool:
    return stage_name.startswith("quanted")


def _cast_floating_tensors_for_validation(value, target_dtype: torch.dtype):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(target_dtype)
        return value
    if isinstance(value, list):
        return [_cast_floating_tensors_for_validation(item, target_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_floating_tensors_for_validation(item, target_dtype) for item in value)
    return value


def _run_forward_validation(
    *, model, tokenizer, input_ids: torch.Tensor, stage_name: str, device, dtype, logger
) -> None:
    validation_input_dtype = _get_validation_input_dtype(stage_name, dtype)
    validation_input_ids = input_ids.detach().clone()
    if _should_move_validation_model(stage_name):
        validation_input_ids = validation_input_ids.to(device)
    debug_batch = {"input_ids": validation_input_ids, "past_seq_length": 0}
    contexts = [
        TimeProfiler(f"valid_{stage_name}_forward", logger),
        MemoryTracker(device=device, name=f"valid_{stage_name}_forward", logger=logger),
        LLMInferenceContextManager(model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        if _should_move_validation_model(stage_name):
            model.to(device=device, dtype=validation_input_dtype)
        model.eval()
        data_processor = model.get_data_preprocessor()
        if _should_move_validation_model(stage_name):
            data_processor.to(device=device, dtype=validation_input_dtype)
        processed_inputs = data_processor(debug_batch)
        processed_inputs = _cast_floating_tensors_for_validation(processed_inputs, validation_input_dtype)
        logits = model(*processed_inputs)

    next_token_id, next_token_text = _decode_next_token(tokenizer, logits)
    logger.info(
        f"[valid/{stage_name}] forward logits={tuple(logits.shape)} "
        f"next_token_id={int(next_token_id[0][0].item())} next_token={next_token_text[0]!r}"
    )
    _reset_validation_runtime(model)


def _run_chat_validation(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    stage_name: str,
    device,
    dtype,
    max_new_tokens: int,
    logger,
) -> None:
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_ids = {int(token_id) for token_id in eos_token_id}
    elif eos_token_id is None:
        eos_token_ids = set()
    else:
        eos_token_ids = {int(eos_token_id)}
    validation_input_dtype = _get_validation_input_dtype(stage_name, dtype)
    generated_ids = input_ids.detach().clone()
    if _should_move_validation_model(stage_name):
        generated_ids = generated_ids.to(device)

    contexts = [
        TimeProfiler(f"valid_{stage_name}_chat", logger),
        MemoryTracker(device=device, name=f"valid_{stage_name}_chat", logger=logger),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        for _ in range(max_new_tokens):
            with ContextManagers([LLMInferenceContextManager(model)]):
                if _should_move_validation_model(stage_name):
                    model.to(device=device, dtype=validation_input_dtype)
                model.eval()
                data_processor = model.get_data_preprocessor()
                if _should_move_validation_model(stage_name):
                    data_processor.to(device=device, dtype=validation_input_dtype)
                processed_inputs = data_processor({"input_ids": generated_ids, "past_seq_length": 0})
                processed_inputs = _cast_floating_tensors_for_validation(processed_inputs, validation_input_dtype)
                logits = model(*processed_inputs)

            next_token_id, _ = _decode_next_token(tokenizer, logits)
            next_token_id = next_token_id.to(generated_ids.device)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            _reset_validation_runtime(model)

            if eos_token_ids and int(next_token_id[0][0].item()) in eos_token_ids:
                break

    output_ids, output_text = _extract_generated_reply(tokenizer, input_ids.cpu(), generated_ids.cpu())
    logger.info(f"[valid/{stage_name}] chat generated_tokens={len(output_ids)} reply={output_text[:200]!r}")


def _run_stage_validation(
    *,
    model,
    tokenizer,
    hf_inputs: dict[str, torch.Tensor],
    stage_name: str,
    device,
    dtype,
    max_new_tokens: int,
    logger,
) -> None:
    logger.info("=" * 60)
    logger.info(f"[valid/{stage_name}] start")
    logger.info("=" * 60)
    input_ids = hf_inputs["input_ids"]
    _run_forward_validation(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        stage_name=stage_name,
        device=device,
        dtype=dtype,
        logger=logger,
    )
    _run_chat_validation(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        stage_name=stage_name,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        logger=logger,
    )
    logger.info(f"[valid/{stage_name}] done")


def _build_calibration_inputs(processor_dir: str, tokenizer, exec_device: torch.device):
    """Build text-only chat-template input_ids without AutoProcessor.

    The vendored transformers 5.5 ``AutoProcessor`` path imports
    ``tokenization_mistral_common`` which in turn requires
    ``mistral_common.ReasoningEffort`` — a symbol absent in the xhquant env's
    installed ``mistral_common``. Since this script only needs ``input_ids``
    for a text-only Gemma4 LLM export, we apply the chat template directly via
    the tokenizer (already loaded by the model). ``processor_dir`` is kept for
    parity with the source script but is currently only consulted when the
    tokenizer cannot resolve a chat template on its own.
    """

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "你好，请详细介绍一下大语言模型的原理。"}],
        }
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        # Some Gemma4 templates expect plain string content instead of the
        # multi-modal list form. Fall back gracefully.
        flat_messages = [
            {"role": m["role"], "content": "".join(c.get("text", "") for c in m["content"])} for m in messages
        ]
        rendered = tokenizer.apply_chat_template(
            flat_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    enc = tokenizer(rendered, return_tensors="pt")
    input_ids = enc["input_ids"].to(torch.long).to(exec_device)
    return {"input_ids": input_ids}


def _bake_token_embedding(model, output_dir: Path, logger) -> Path:
    """Save the input embedding with ``embed_scale`` baked in.

    Mirrors the ``token_embedding`` artefact produced by the source script,
    which is what ``LLMWithMaskONNXModel.set_input_embeddings`` expects when
    running HMONNX golden inference.
    """

    token_embedding = model.get_input_embeddings()
    if token_embedding is None:
        raise RuntimeError("Model does not expose a token embedding; cannot bake embed_scale.")
    token_embedding_file = output_dir / "token_embedding.pt"
    state_dict = {k: v.detach().clone() for k, v in token_embedding.state_dict().items()}
    embed_scale = getattr(token_embedding, "embed_scale", None)
    if embed_scale is not None:
        weight = state_dict["weight"]
        state_dict["weight"] = weight * embed_scale.to(weight.dtype)
        try:
            scale_value = float(embed_scale.detach().reshape(-1)[0].item())
        except Exception:
            scale_value = float("nan")
        logger.info(f"Baked embed_scale={scale_value:.4f} into token_embedding weights")
    else:
        logger.info("token_embedding has no embed_scale attribute; saving raw weights")
    torch.save(state_dict, str(token_embedding_file))
    return token_embedding_file


def _generate_hmonnx_golden(
    *,
    work_dir: Path,
    prefill_onnx_file: str,
    decode_onnx_file: str,
    input_ids: torch.Tensor,
    tokenizer,
    input_sequence_length: int,
    num_hidden_layers: int,
    kv_cache_shape: list,
    kv_cache_shapes_per_layer: list | None,
    sliding_window_cfg: dict,
    token_embedding_file: Path,
    pad_token_id: int,
    max_decode_steps: int,
    logger,
) -> None:
    """Replicate source-style HMONNX golden generation without xhquant_llm.

    Uses the repo-local copy of ``LLMWithMaskONNXModel`` so the golden flow
    keeps the same masks, KV layout and I/O plumbing without importing the
    sibling xhquant_llm package.
    """

    import xhquant.utils.suppress_printing

    xhquant.utils.suppress_printing.disable_printing = True
    from xh_model_zoo.xh_llm.models.llm_with_mask_onnx_model import LLMWithMaskONNXModel

    golden_output_dir = work_dir / "golden"
    golden_output_dir.mkdir(parents=True, exist_ok=True)
    prefill_golden_dir = golden_output_dir / "prefill"
    decode_golden_dir = golden_output_dir / "decode"

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    dtype = torch.float16

    state_dict = torch.load(str(token_embedding_file), map_location="cpu", weights_only=True)
    token_embedding = nn.Embedding(state_dict["weight"].shape[0], state_dict["weight"].shape[1])
    token_embedding.load_state_dict(state_dict)

    model = LLMWithMaskONNXModel(
        prefill=dict(onnx=prefill_onnx_file, input_sequence_length=input_sequence_length),
        decode=dict(onnx=decode_onnx_file),
        kv_cache=dict(num_hidden_layers=num_hidden_layers, shape=kv_cache_shape),
        sliding_window_cfg=sliding_window_cfg,
        pad_token_id=pad_token_id,
    )
    model.set_input_embeddings(token_embedding)

    if kv_cache_shapes_per_layer is not None:
        from xhquant.core import HybridCacheTensor

        for layer_idx, shape in enumerate(kv_cache_shapes_per_layer):
            setattr(
                model,
                f"past_k_cache_{layer_idx}",
                HybridCacheTensor(torch.zeros(shape, dtype=torch.float16)),
            )
            setattr(
                model,
                f"past_v_cache_{layer_idx}",
                HybridCacheTensor(torch.zeros(shape, dtype=torch.float16)),
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.init_prefill()

    input_len = input_ids.shape[-1]
    data_batch = {"input_ids": input_ids.to(device), "past_seq_length": 0}

    meta_payload = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "input_ids": input_ids.tolist(),
    }
    with open(golden_output_dir / "golden_meta_info.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=4)

    model.eval()
    model.to(device)
    model.set_exec_device(exec_device)
    model.to(dtype)

    model.save_prefill_golden(str(prefill_golden_dir))
    logger.info(f"[Golden] data_batch input_ids shape={data_batch['input_ids'].shape}")
    logger.info(f"[Golden] prefill_input_sequence_length={model.prefill_input_sequence_length}")
    with torch.no_grad():
        prefill_logits = model.prefill(data_batch)
        next_token_id, next_token_text = _decode_next_token(tokenizer, prefill_logits)
    logger.info(f"[Golden] Prefill next token: {next_token_text}")
    model.release_prefill_session()

    model.init_decode()
    model.to(device)
    model.set_exec_device(exec_device)
    model.to(dtype)

    past_seq_length = input_len
    data_batch = {"input_ids": next_token_id.to(device), "past_seq_length": past_seq_length}

    decode_texts: list[str] = []
    eos_token_id = tokenizer.eos_token_id
    decode_step = 0
    model.save_decode_golden(str(decode_golden_dir))
    while True:
        with torch.no_grad():
            decode_logits = model.decode(data_batch)
            next_token_id, next_token_text = _decode_next_token(tokenizer, decode_logits)
        logger.info(f"[Golden] Decode step {decode_step}: {next_token_text}")
        token_id = int(next_token_id[0][0].item())
        if eos_token_id is not None and token_id == eos_token_id:
            break
        decode_texts.extend(next_token_text)
        past_seq_length += 1
        decode_step += 1
        data_batch = {"input_ids": next_token_id.to(device), "past_seq_length": past_seq_length}
        if decode_step >= max_decode_steps:
            break

    logger.info(f"[Golden] Decode output: {''.join(decode_texts)}")
    logger.info("[Golden] HMONNX golden generation done.")


# ──────────────────────────────────────────────────────────────────────────
# Stage runner
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma4 MoE with-mask LLM export (Merak)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_26b_a4b_it_xh2a_w8a8_256_2k.py",
    )
    parser.add_argument("--output", default="", help="output dir; default work_dirs/<cfg_stem>")
    parser.add_argument(
        "--valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run WRAP / QUANTED_ALIGNED validation: one forward smoke plus one short deterministic chat generation.",
    )
    parser.add_argument(
        "--valid-max-new-tokens",
        type=int,
        default=16,
        help="Max new tokens used by each validation chat generation.",
    )
    parser.add_argument(
        "--golden",
        action="store_true",
        help="Generate HMONNX golden (prefill + decode) on the just-exported ONNX.",
    )
    parser.add_argument(
        "--max-decode-steps",
        type=int,
        default=1,
        help="Max decode steps for golden generation (matches source default).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output dir.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # ── 1. Load config + bootstrap output dir / logging ─────────────
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    output_dir = Path(args.output) if args.output else Path("work_dirs") / cfg_name
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(output_dir / "with_mask_export.log"), args.debug)
    logger = get_xhquant_logger()

    cfg.dump(output_dir / Path(args.config).name)
    logger.info(f"Config:\n{cfg.pretty_text}")

    # ── 2. Build model and drop visual sub-model ────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exec_device = device
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    logger.info(f"device={device} exec_device={exec_device} dtype={dtype}")

    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model = AutoLLMModel.from_pretrained(model_cfg)

    # Visual encoder is exported separately by gemma4_moe_visual_xh_export_onnx.py.
    if isinstance(getattr(model, "_models", None), dict):
        model._models.pop("visual", None)
    if hasattr(model, "visual"):
        try:
            object.__delattr__(model, "visual")
        except (AttributeError, TypeError):
            try:
                del model.__dict__["visual"]
            except KeyError:
                pass

    # Build calibration inputs from real chat template (matches source).
    fallback_dir = getattr(model, "fallback_hf_model_dir", None)
    primary_dir = model.hf_model_dir
    processor_dir = primary_dir
    if _resolve_hf_asset(primary_dir, fallback_dir, "preprocessor_config.json") is None and fallback_dir is not None:
        processor_dir = fallback_dir
    logger.info(f"AutoProcessor source dir: {processor_dir}")
    tokenizer = model.get_tokenizer(trust_remote_code=True)
    hf_inputs = _build_calibration_inputs(processor_dir, tokenizer, exec_device)

    # ── 3. Stage 1: WRAPED ─────────────────────────────────────────
    logger.info("[stage] -> WRAPED")
    model.to_wrap()
    if args.valid:
        _run_stage_validation(
            model=model,
            tokenizer=tokenizer,
            hf_inputs=hf_inputs,
            stage_name="wrap",
            device=device,
            dtype=dtype,
            max_new_tokens=args.valid_max_new_tokens,
            logger=logger,
        )

    # ── 4. Stage 2: FRONTEND ──────────────────────────────────────
    # The fix in Gemma4MoeWithMaskInputProcessor.forward keeps the KV cache
    # lists grouped (matching source semantics), so the wrap signature
    # (inputs_embeds, past_seq_length, seq_length, local_mask, global_mask,
    # past_key_caches, past_value_caches) lines up at trace time.
    logger.info("[stage] -> FRONTEND")
    model.to_fronted()

    # ── 5. Stage 3: QUANTED_ALIGNED ──────────────────────────────
    logger.info("[stage] -> QUANTED_ALIGNED")
    model.to_quanted_aligned()
    if args.valid:
        _run_stage_validation(
            model=model,
            tokenizer=tokenizer,
            hf_inputs=hf_inputs,
            stage_name="quanted_aligned",
            device=device,
            dtype=dtype,
            max_new_tokens=args.valid_max_new_tokens,
            logger=logger,
        )

    # ── 6. Stage 4: EXPORTED + prefill/decode ONNX + meta ────────
    logger.info("[stage] -> EXPORTED")
    model._quanted_model.fixed()
    model._quanted_model.to(device=device, dtype=_get_validation_input_dtype("quanted_aligned", dtype))
    model._quanted_model.eval()
    exported_info = model.get_export_info(str(output_dir))
    model._export_hmonnx(exported_info)

    meta_info = exported_info.meta
    exported_dir_path = Path(exported_info.exported_dir)

    meta_dict = meta_info.to_dict()
    with open(exported_dir_path / "golden_meta_info.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=4, ensure_ascii=False, default=str)
    logger.info(f"merak meta saved to {exported_dir_path / 'golden_meta_info.json'}")

    # Resolve absolute prefill/decode ONNX paths for downstream consumers.
    prefill_onnx_abs = str(exported_dir_path / meta_info.prefill_hmonnx)
    decode_onnx_abs = str(exported_dir_path / meta_info.decode_hmonnx)

    # ── 7. Source-compatible meta + token embedding side artifact ─
    # Provides everything LLMWithMaskONNXModel / source generate_golden needs,
    # while the merak `golden_meta_info.json` above remains the canonical Merak
    # export descriptor.
    token_embedding_file = _bake_token_embedding(model, output_dir, logger)

    sliding_window_cfg = dict(model.sliding_window_cfg)
    populated_caches = list(model.past_key_caches)
    if populated_caches:
        kv_cache_shape = list(populated_caches[0].shape)
        kv_cache_shapes_per_layer = [list(c.shape) for c in populated_caches]
        num_hidden_layers = len(populated_caches)
    else:
        kv_cache_shape = list(model.kvcache_config.kv_cache_shape)
        num_hidden_layers = int(model.kvcache_config.num_layers)
        kv_cache_shapes_per_layer = [kv_cache_shape for _ in range(num_hidden_layers)]

    src_meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config": Path(args.config).name,
        "hf_model": model.hf_model_dir,
        "hf_config": str((exported_dir_path / meta_info.hf_config).relative_to(output_dir)),
        "token_embedding_file": str(token_embedding_file.relative_to(output_dir)),
        "prefill_onnx": str(Path(prefill_onnx_abs).relative_to(output_dir)),
        "decode_onnx": str(Path(decode_onnx_abs).relative_to(output_dir)),
        "input_sequence_length": int(model.config.prefill_chunk_length),
        "num_hidden_layers": num_hidden_layers,
        "use_cache": True,
        "sliding_window_cfg": sliding_window_cfg,
        "kv_cache_shape": kv_cache_shape,
        "kv_cache_shapes_per_layer": kv_cache_shapes_per_layer,
        "exported_dir": str(exported_dir_path.relative_to(output_dir)),
    }
    src_meta_path = output_dir / "export_meta_info.json"
    with open(src_meta_path, "w", encoding="utf-8") as f:
        json.dump(src_meta, f, indent=4, ensure_ascii=False, default=str)
    logger.info(f"export_meta_info.json saved to {src_meta_path}")

    # ── 8. Optional Stage 5: HMONNX golden ─────────────────────────
    if args.golden:
        logger.info("=" * 60)
        logger.info("Generating HMONNX golden (prefill + decode) with mask")
        logger.info("=" * 60)
        _generate_hmonnx_golden(
            work_dir=output_dir,
            prefill_onnx_file=prefill_onnx_abs,
            decode_onnx_file=decode_onnx_abs,
            input_ids=hf_inputs["input_ids"].cpu(),
            tokenizer=tokenizer,
            input_sequence_length=int(model.config.prefill_chunk_length),
            num_hidden_layers=num_hidden_layers,
            kv_cache_shape=kv_cache_shape,
            kv_cache_shapes_per_layer=kv_cache_shapes_per_layer,
            sliding_window_cfg=sliding_window_cfg,
            token_embedding_file=token_embedding_file,
            pad_token_id=int(getattr(model, "pad_token_id", 0) or 0),
            max_decode_steps=args.max_decode_steps,
            logger=logger,
        )
    else:
        logger.info("Skip HMONNX golden generation (enable with --golden)")

    print(f"export OK {src_meta_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
