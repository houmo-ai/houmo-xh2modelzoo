"""
FireRedASR LLM common quant script (quarot + gptq).

Supports both:
  - merge_lora: merge LoRA into base model first.
  - keep_lora: keep runtime LoRA branches for downstream export.
"""

import argparse
import gc
import glob
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoTokenizer
from xhquant.api import set_random_seed

# Ensure local repo package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.api import Config, decode_next_token, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen2_legacy import XHQwen2LegacyModel

try:
    from examples.audio.fireredasr.audio_llm_xh2a_export import (
        _resolve_hf_model_dir,
        _edit_distance,
        _find_ref_text,
        _find_wav_dir,
        _load_ref_texts,
        load_fireredasr_lora_weights,
        merge_lora_into_base_model,
        register_lora_buffers,
    )
except ModuleNotFoundError:
    export_file = Path(__file__).resolve().parent / "audio_llm_xh2a_export.py"
    spec = importlib.util.spec_from_file_location("firered_export_local", export_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FireRedASR export helpers from {export_file}")
    export_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_module)
    _resolve_hf_model_dir = export_module._resolve_hf_model_dir
    _edit_distance = export_module._edit_distance
    _find_ref_text = export_module._find_ref_text
    _find_wav_dir = export_module._find_wav_dir
    _load_ref_texts = export_module._load_ref_texts
    load_fireredasr_lora_weights = export_module.load_fireredasr_lora_weights
    merge_lora_into_base_model = export_module.merge_lora_into_base_model
    register_lora_buffers = export_module.register_lora_buffers


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fireredasr/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py",
    )
    parser.add_argument("--fireredasr_model_dir", type=str, default="weights/FireRedASR-LLM-L")
    parser.add_argument("--hf_model_dir", type=str, default=None)
    parser.add_argument("--lora_mode", type=str, choices=["merge_lora", "keep_lora"], default="merge_lora")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_quarot", action="store_true")
    parser.add_argument("--skip_gptq", action="store_true")
    parser.add_argument("--quarot_matrix_size", type=int, default=None)
    parser.add_argument("--save_rotation_matrix", action="store_true")
    parser.add_argument("--rotation_matrix_out", type=str, default=None)
    parser.add_argument("--resume_weight", type=str, default=None)
    parser.add_argument("--gptq_calib_dataset", type=str, default=None)
    parser.add_argument("--gptq_calib_samples", type=int, default=None)
    parser.add_argument("--gptq_seqlen", type=int, default=None)
    parser.add_argument("--gptq_cache_dir", type=str, default=None)
    parser.add_argument("--gptq_data_files", type=str, nargs="*", default=None)
    parser.add_argument(
        "--reuse_gptq_layer_cache",
        action="store_true",
        help="复用 work_dir/layers_cache（默认关闭，避免误用旧cache）",
    )
    parser.add_argument("--rotate_audio_projector", action="store_true")
    parser.add_argument("--rotated_adapter_out", type=str, default=None)
    parser.add_argument("--valid", action="store_true", help="在common quant脚本内做量化前后和回读一致性验证")
    parser.add_argument("--valid_asr", action="store_true", help="在common quant脚本内做真实语音ASR对比验证")
    parser.add_argument("--wav_dir", type=str, default=None, help="真实语音目录，默认自动检测")
    parser.add_argument("--ref_text", type=str, default=None, help="参考文本文件，可选")
    parser.add_argument("--use_gpu", action="store_true", help="ASR验证阶段使用GPU")
    parser.add_argument("--prompt", type=str, default="请转写音频为文字")
    return parser


def msg_output_format(title):
    return f"{'*' * 10} {title} {'*' * 10}"


def rotate_adapter_with_llm_matrix(adapter_state_dict: dict, q_matrix: torch.Tensor, logger):
    """Rotate adapter outputs into the same hidden-space basis as rotated LLM."""
    if q_matrix is None:
        logger.warning("No quarot rotation matrix found on model, skip adapter rotation.")
        return adapter_state_dict

    q = q_matrix.detach().clone().to(dtype=torch.float64)
    rotated = {}
    rotated_keys = []
    skipped_keys = []
    for key, value in adapter_state_dict.items():
        v = value.detach().clone()
        # FireRedASR adapter contains a nonlinearity between linear1 and linear2.
        # Only the final projection (linear2) can be output-rotated safely.
        if key == "linear2.weight" and v.ndim == 2 and v.shape[0] == q.shape[0]:
            # output-side rotation: W <- Q^T @ W
            v = torch.matmul(q.T, v.to(dtype=torch.float64)).to(dtype=value.dtype)
            rotated_keys.append(key)
        elif key == "linear2.bias" and v.ndim == 1 and v.shape[0] == q.shape[0]:
            # output bias rotation: b <- Q^T @ b
            v = torch.matmul(q.T, v.to(dtype=torch.float64)).to(dtype=value.dtype)
            rotated_keys.append(key)
        else:
            skipped_keys.append(key)
        rotated[key] = v
    logger.info(
        "Applied quarot matrix to audio projector state_dict "
        f"(rotated={rotated_keys}, skipped={len(skipped_keys)} keys)."
    )
    return rotated


def _load_rotation_matrix_file(path: str | None, logger) -> torch.Tensor | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        q = torch.load(str(p), map_location="cpu")
    except Exception as exc:
        logger.warning(f"Failed to load rotation matrix from {p}: {exc}")
        return None
    if not isinstance(q, torch.Tensor):
        logger.warning(f"Rotation matrix file {p} is not a tensor, got {type(q)}")
        return None
    logger.info(f"Loaded quarot rotation matrix from {p}")
    return q


def compute_tensor_diff_metrics(ref: torch.Tensor, cur: torch.Tensor):
    ref_fp32 = ref.float().reshape(1, -1)
    cur_fp32 = cur.float().reshape(1, -1)
    diff = torch.abs(ref_fp32 - cur_fp32)
    cosine = torch.nn.functional.cosine_similarity(ref_fp32, cur_fp32, dim=1).item()
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "cosine_sim": cosine,
    }


def log_logits_compare(logger, name: str, ref_logits: torch.Tensor, cur_logits: torch.Tensor):
    metrics = compute_tensor_diff_metrics(ref_logits, cur_logits)
    logger.info(
        f"{name} logits diff: max={metrics['max_diff']:.6e}, "
        f"mean={metrics['mean_diff']:.6e}, cosine={metrics['cosine_sim']:.6f}"
    )


def _load_wraped_model_state_dict_compat(
    xh_model_reload,
    model: torch.nn.Module,
    checkpoint_file: Path,
    strict_resume: bool,
):
    """Compatibility wrapper for xh_model_zoo/xhquant_llm API differences."""
    try:
        xh_model_reload.load_wraped_model_state_dict(model, str(checkpoint_file), strict=strict_resume)
    except TypeError:
        xh_model_reload.load_wraped_model_state_dict(model, str(checkpoint_file))


def _build_valid_input_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt").input_ids


def _get_eval_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


@torch.no_grad()
def _run_single_forward_logits(model: torch.nn.Module, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    eval_device = torch.device(device)
    eval_dtype = _get_eval_dtype(eval_device)
    model.eval().to(eval_device).to(eval_dtype)
    outputs = model(input_ids=input_ids.to(eval_device), use_cache=False)
    logits = outputs.logits[:, -1:, :].detach().cpu().to(torch.float32)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def _streaming_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor, chunk_rows: int = 512):
    if lhs.shape != rhs.shape:
        return float("inf"), float("inf")
    if lhs.ndim == 1:
        lhs = lhs.unsqueeze(0)
        rhs = rhs.unsqueeze(0)
    max_diff = 0.0
    sum_diff = 0.0
    total_count = 0
    for start in range(0, lhs.shape[0], chunk_rows):
        end = min(start + chunk_rows, lhs.shape[0])
        diff = (lhs[start:end].float() - rhs[start:end].float()).abs()
        max_diff = max(max_diff, diff.max().item())
        sum_diff += diff.sum().item()
        total_count += diff.numel()
    mean_diff = sum_diff / max(total_count, 1)
    return max_diff, mean_diff


def _validate_saved_quant_checkpoint(
    checkpoint_file: Path,
    cfg: Config,
    args,
    valid_input_ids: torch.Tensor,
    in_memory_logits: torch.Tensor,
    tokenizer,
    lora_state_dict: dict,
    lora_config: dict,
    logger,
):
    logger.info(msg_output_format("Start saved-checkpoint reload validation"))

    xh_model_reload: XHQwen2LegacyModel = MODELS.build(cfg.model)  # type: ignore
    reload_model = xh_model_reload.get_hf_model(device_map="cpu")
    reload_model.eval().to(torch.float16)

    if args.lora_mode == "merge_lora":
        reload_model = merge_lora_into_base_model(reload_model, lora_state_dict, lora_config)
        strict_resume = True
    else:
        reload_model = register_lora_buffers(reload_model, lora_state_dict, lora_config)
        strict_resume = False

    _load_wraped_model_state_dict_compat(
        xh_model_reload=xh_model_reload,
        model=reload_model,
        checkpoint_file=checkpoint_file,
        strict_resume=strict_resume,
    )
    reload_logits = _run_single_forward_logits(reload_model, valid_input_ids, cfg.device)

    log_logits_compare(logger, "valid_reload_vs_in_memory", in_memory_logits, reload_logits)
    in_mem_next_id, in_mem_next_text = decode_next_token(tokenizer, in_memory_logits)
    reload_next_id, reload_next_text = decode_next_token(tokenizer, reload_logits)
    logger.info(f"in-memory next token: {in_mem_next_id} {in_mem_next_text}")
    logger.info(f"reloaded next token: {reload_next_id} {reload_next_text}")

    with safe_open(str(checkpoint_file), framework="pt", device="cpu") as f:
        embed_key = None
        for key in f.keys():
            if key.endswith("embed_tokens.weight"):
                embed_key = key
                break
        if embed_key is None:
            logger.warning("No embed_tokens.weight found in saved checkpoint.")
        else:
            checkpoint_embedding = f.get_tensor(embed_key).cpu()
            model_embedding = reload_model.model.embed_tokens.weight.detach().cpu()
            max_diff, mean_diff = _streaming_abs_diff(checkpoint_embedding, model_embedding)
            logger.info(
                f"embedding reload check ({embed_key}): max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
            )

    del reload_model
    del xh_model_reload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(msg_output_format("End saved-checkpoint reload validation"))


@torch.no_grad()
def _materialize_runtime_lora_into_weight(model: torch.nn.Module, lora_config: dict, logger) -> int:
    """For keep_lora path, merge runtime LoRA buffers into Linear weights for HF ASR validation."""
    scaling = float(lora_config["lora_alpha"]) / float(lora_config["r"])
    merged_count = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not hasattr(module, "weight_lora_a") or not hasattr(module, "weight_lora_b"):
            continue

        lora_a = module.weight_lora_a
        lora_b = module.weight_lora_b
        if lora_a is None or lora_b is None:
            continue
        if lora_a.shape[0] != lora_b.shape[1]:
            lora_a = lora_a.T
            lora_b = lora_b.T
        if (lora_b.shape[0], lora_a.shape[1]) != tuple(module.weight.shape):
            continue

        calc_dtype = module.weight.dtype
        if module.weight.device.type == "cpu" and calc_dtype in (torch.float16, torch.bfloat16):
            calc_dtype = torch.float32
        delta = (lora_b.to(dtype=calc_dtype, device=module.weight.device) @ lora_a.to(dtype=calc_dtype, device=module.weight.device)) * scaling
        module.weight.data.add_(delta.to(dtype=module.weight.dtype))
        merged_count += 1
    logger.info(f"Materialized runtime LoRA into {merged_count} linear layers for ASR validation.")
    return merged_count


def _load_quantized_hf_model_from_checkpoint(
    checkpoint_file: Path,
    cfg: Config,
    args,
    lora_state_dict: dict,
    lora_config: dict,
):
    xh_model_reload: XHQwen2LegacyModel = MODELS.build(cfg.model)  # type: ignore
    model = xh_model_reload.get_hf_model(device_map="cpu")
    model.eval().to(torch.float16)

    if args.lora_mode == "merge_lora":
        model = merge_lora_into_base_model(model, lora_state_dict, lora_config)
        strict_resume = True
    else:
        model = register_lora_buffers(model, lora_state_dict, lora_config)
        strict_resume = False

    _load_wraped_model_state_dict_compat(
        xh_model_reload=xh_model_reload,
        model=model,
        checkpoint_file=checkpoint_file,
        strict_resume=strict_resume,
    )
    del xh_model_reload
    gc.collect()
    return model


def _sync_fireredasr_llm_special_tokens(
    quantized_llm: torch.nn.Module, reference_llm: torch.nn.Module, logger
) -> None:
    """Align special token ids expected by FireRedASR after replacing LLM."""
    copied = {}
    for key in ("pad_token_id", "bos_token_id", "eos_token_id", "default_speech_token_id"):
        ref_val = getattr(reference_llm.config, key, None)
        cur_val = getattr(quantized_llm.config, key, None)
        # default_speech_token_id must follow FireRedASR tokenizer; others fill when missing.
        if key == "default_speech_token_id":
            if ref_val is not None and cur_val != ref_val:
                setattr(quantized_llm.config, key, ref_val)
                copied[key] = ref_val
        elif cur_val is None and ref_val is not None:
            setattr(quantized_llm.config, key, ref_val)
            copied[key] = ref_val

    if getattr(quantized_llm.config, "pad_token_id", None) is None:
        eos_id = getattr(quantized_llm.config, "eos_token_id", None)
        fallback = int(eos_id) if eos_id is not None else 0
        quantized_llm.config.pad_token_id = fallback
        copied["pad_token_id_fallback"] = fallback

    gen_cfg = getattr(quantized_llm, "generation_config", None)
    if gen_cfg is not None:
        for key in ("pad_token_id", "bos_token_id", "eos_token_id"):
            val = getattr(quantized_llm.config, key, None)
            if val is not None:
                setattr(gen_cfg, key, val)

    logger.info(f"Synchronized LLM special tokens for FireRedASR: {copied}")


@torch.no_grad()
def validate_asr_quantized_hf_llm(
    fireredasr_model_dir: str,
    quantized_llm: torch.nn.Module,
    wav_dir: str | None,
    ref_text_file: str | None,
    use_gpu: bool,
    logger,
    rotated_adapter_path: str | None = None,
):
    try:
        from fireredasr.models.fireredasr import FireRedAsr
    except Exception as exc:
        logger.warning(f"Skip valid_asr: cannot import FireRedAsr ({exc})")
        return {"skipped": True, "reason": "fireredasr_import_error"}

    wav_dir = wav_dir or _find_wav_dir()
    if wav_dir is None:
        logger.warning("No wav_dir found, skip ASR validation.")
        return {"skipped": True, "reason": "no_wav_dir"}

    wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_paths:
        logger.warning(f"No wav files found in {wav_dir}, skip ASR validation.")
        return {"skipped": True, "reason": "no_wav_files"}

    ref_text_file = ref_text_file or _find_ref_text(wav_dir)
    ref_texts = _load_ref_texts(ref_text_file)
    uttids = [Path(p).stem for p in wav_paths]
    infer_args = {
        "use_gpu": use_gpu and torch.cuda.is_available(),
        "beam_size": 1,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 1.0,
        "llm_length_penalty": 0.0,
        "temperature": 1.0,
    }

    native_model = FireRedAsr.from_pretrained("llm", fireredasr_model_dir)
    if infer_args["use_gpu"]:
        native_model.model.cuda()
    native_results = []
    for uttid, wav_path in zip(uttids, wav_paths):
        native_results.extend(native_model.transcribe([uttid], [wav_path], infer_args))
    native_texts = {item["uttid"]: item["text"] for item in native_results}
    if infer_args["use_gpu"]:
        native_model.model.cpu()
        torch.cuda.empty_cache()
    del native_model
    gc.collect()

    quant_model = FireRedAsr.from_pretrained("llm", fireredasr_model_dir)
    _sync_fireredasr_llm_special_tokens(quantized_llm, quant_model.model.llm, logger)
    quantized_llm.eval()
    quant_model.model.llm = quantized_llm
    if rotated_adapter_path is not None and Path(rotated_adapter_path).exists():
        rotated_adapter_sd = load_safetensors_file(str(rotated_adapter_path))
        missing, unexpected = quant_model.model.encoder_projector.load_state_dict(rotated_adapter_sd, strict=False)
        logger.info(
            f"Loaded rotated audio projector from {rotated_adapter_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    if infer_args["use_gpu"]:
        quant_model.model.cuda()
    quant_results = []
    for uttid, wav_path in zip(uttids, wav_paths):
        quant_results.extend(quant_model.transcribe([uttid], [wav_path], infer_args))
    quant_texts = {item["uttid"]: item["text"] for item in quant_results}

    details = []
    total_chars = 0
    total_edit = 0
    total_ref_chars = 0
    total_ref_edit = 0
    for uttid in uttids:
        native_text = native_texts.get(uttid, "")
        quant_text = quant_texts.get(uttid, "")
        is_match = native_text == quant_text

        cer_nat_q = 0.0
        if len(native_text) > 0:
            edit = _edit_distance(list(native_text), list(quant_text))
            total_edit += edit
            total_chars += len(native_text)
            cer_nat_q = edit / len(native_text)

        cer_ref_quant = None
        ref_text = ref_texts.get(uttid, None)
        if ref_text is not None and len(ref_text) > 0:
            edit_quant = _edit_distance(list(ref_text), list(quant_text))
            cer_ref_quant = edit_quant / len(ref_text)
            total_ref_chars += len(ref_text)
            total_ref_edit += edit_quant

        logger.info(
            f"[ASR-CMP-QUANT] {uttid}: match={is_match}, cer(native->quant)={cer_nat_q:.4f}, "
            f"native='{native_text}', quant='{quant_text}'"
        )
        details.append(
            dict(
                uttid=uttid,
                native_text=native_text,
                quant_text=quant_text,
                ref_text=ref_text,
                match=is_match,
                cer_native_vs_quant=cer_nat_q,
                cer_ref_quant=cer_ref_quant,
            )
        )

    summary = {
        "num_utts": len(uttids),
        "num_match": sum(1 for item in details if item["match"]),
        "match_rate": sum(1 for item in details if item["match"]) / max(len(details), 1),
        "cer_native_vs_quant": (total_edit / total_chars) if total_chars > 0 else None,
        "cer_ref_quant": (total_ref_edit / total_ref_chars) if total_ref_chars > 0 else None,
        "details": details,
    }
    logger.info(
        f"ASR quant summary: match_rate={summary['match_rate']:.4f}, "
        f"cer(native->quant)={summary['cer_native_vs_quant']}"
    )

    if infer_args["use_gpu"]:
        quant_model.model.cpu()
        torch.cuda.empty_cache()
    del quant_model
    gc.collect()
    return summary


def main():
    args = parse_arguments().parse_args()

    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg.work_dir = str(Path("./work_dirs") / f"{cfg_name}_fireredasr_{args.lora_mode}")
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cpu:
        cfg.device = "cpu"
    cfg.dtype = "float16"
    if "quarot" not in cfg:
        cfg.quarot = False
    if "gptq" not in cfg:
        cfg.gptq = False
    if not (cfg.quarot or cfg.gptq):
        raise ValueError("quarot or gptq must be True")

    set_random_seed(args.seed)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    xhquant_llm_init(log_file, False)
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")

    resolved_hf_model_dir = _resolve_hf_model_dir(
        cfg_hf_model_dir=str(cfg.hf_model_dir),
        fireredasr_model_dir=args.fireredasr_model_dir,
        cli_hf_model_dir=args.hf_model_dir,
    )
    cfg.hf_model_dir = resolved_hf_model_dir
    cfg.model.hf_model = resolved_hf_model_dir
    logger.info(f"Resolved hf_model_dir: {resolved_hf_model_dir}")

    xh_model: XHQwen2LegacyModel = MODELS.build(cfg.model)  # type: ignore
    native_model = xh_model.get_hf_model(device_map="cpu")
    native_model.eval().to(torch.float16)

    lora_state_dict, lora_config, _, adapter_state_dict, _ = load_fireredasr_lora_weights(args.fireredasr_model_dir)
    logger.info(f"Loaded LoRA keys: {len(lora_state_dict)}")
    if args.lora_mode == "merge_lora":
        native_model = merge_lora_into_base_model(native_model, lora_state_dict, lora_config)
    else:
        native_model = register_lora_buffers(native_model, lora_state_dict, lora_config)

    if args.resume_weight is not None and Path(args.resume_weight).exists():
        state_dict = load_safetensors_file(args.resume_weight)
        remove = [k for k in state_dict if "quant_weight" in k]
        for key in remove:
            state_dict.pop(key)
        strict_resume = args.lora_mode == "merge_lora"
        missing_keys, unexpected_keys = native_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"resume_weight missing keys: {len(missing_keys)}; first 5: {missing_keys[:5]}")
        if unexpected_keys:
            logger.warning(
                f"resume_weight unexpected keys: {len(unexpected_keys)}; first 5: {unexpected_keys[:5]}"
            )
        if strict_resume and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Strict resume load failed for {args.resume_weight}: "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )
        logger.info(msg_output_format(f"Load state_dict from {args.resume_weight}"))

    tokenizer = None
    valid_input_ids = None
    baseline_logits = None
    stage_logits = None
    if args.valid:
        tokenizer = AutoTokenizer.from_pretrained(resolved_hf_model_dir, trust_remote_code=True)
        valid_input_ids = _build_valid_input_ids(tokenizer, args.prompt)
        baseline_logits = _run_single_forward_logits(native_model, valid_input_ids, cfg.device)
        stage_logits = baseline_logits
        baseline_next_id, baseline_next_text = decode_next_token(tokenizer, baseline_logits)
        logger.info(f"baseline next token: {baseline_next_id} {baseline_next_text}")
        if args.lora_mode == "keep_lora":
            logger.warning(
                "keep_lora common_quant valid compares base branch consistency only; runtime LoRA accuracy uses export --valid."
            )

    quant_methods = []
    quant_checkpoint_file: Path | None = None
    cached_q_matrix: torch.Tensor | None = None
    default_rotation_out = Path(cfg.work_dir) / "quarot_rotation_matrix.pt"
    device = cfg.device
    if cfg.quarot and not args.skip_quarot:
        quant_methods.append("quarot")
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

        logger.info(msg_output_format("Start quarot quantization"))
        native_model = quarot(native_model, device=device, quarot_matrix_size=args.quarot_matrix_size)
        logger.info(msg_output_format("End quarot quantization"))

        q_matrix = getattr(native_model, "_quarot_rotation_matrix", None)
        if isinstance(q_matrix, torch.Tensor):
            cached_q_matrix = q_matrix.detach().cpu()
        if cached_q_matrix is not None and (args.save_rotation_matrix or args.rotate_audio_projector):
            rotation_out = args.rotation_matrix_out
            if rotation_out is None:
                rotation_out = str(default_rotation_out)
            torch.save(cached_q_matrix, rotation_out)
            logger.info(f"Saved quarot rotation matrix to {rotation_out}")
        elif args.save_rotation_matrix or args.rotate_audio_projector:
            logger.warning(
                "QuaRot finished but no _quarot_rotation_matrix found on model; "
                "will try --rotation_matrix_out/default file fallback before rotating audio projector."
            )

        if args.valid and valid_input_ids is not None and stage_logits is not None:
            quarot_logits = _run_single_forward_logits(native_model, valid_input_ids, cfg.device)
            log_logits_compare(logger, "valid_quarot_vs_previous", stage_logits, quarot_logits)
            stage_logits = quarot_logits

    if cfg.gptq and not args.skip_gptq:
        quant_methods.append("gptq")
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

        gptq_config = dict(cfg.get("gptq_config", {}))
        if args.gptq_calib_dataset is not None:
            gptq_config["calib_dataset"] = args.gptq_calib_dataset
        if args.gptq_calib_samples is not None:
            gptq_config["calib_samples"] = args.gptq_calib_samples
        if args.gptq_seqlen is not None:
            gptq_config["seqlen"] = args.gptq_seqlen
        if args.gptq_cache_dir is not None:
            gptq_config["cache_dir"] = args.gptq_cache_dir
        if args.gptq_data_files:
            gptq_config["data_files"] = [str(Path(p).expanduser()) for p in args.gptq_data_files]

        layers_cache_dir = Path(cfg.work_dir) / "layers_cache"
        if layers_cache_dir.exists() and not args.reuse_gptq_layer_cache:
            logger.info(f"Removing stale GPTQ layer cache: {layers_cache_dir}")
            shutil.rmtree(layers_cache_dir)
        layers_cache_dir.mkdir(exist_ok=True, parents=True)

        logger.info(msg_output_format("Start gptq quantization"))
        native_model = gptq(
            native_model,
            args=args,
            model_name=resolved_hf_model_dir,
            **gptq_config,
            device=device,
            layers_cache_dir=str(layers_cache_dir),
        )
        logger.info(msg_output_format("End gptq quantization"))
        quant_weight_count = sum(1 for k in native_model.state_dict().keys() if k.endswith("quant_weight"))
        logger.info(f"GPTQ quant_weight tensor count: {quant_weight_count}")

        if args.valid and valid_input_ids is not None and stage_logits is not None:
            gptq_logits = _run_single_forward_logits(native_model, valid_input_ids, cfg.device)
            log_logits_compare(logger, "valid_gptq_vs_previous", stage_logits, gptq_logits)
            stage_logits = gptq_logits

    if quant_methods:
        quant_name = "_".join(quant_methods)
        filename = Path(cfg.work_dir) / f"{quant_name}-state-dict.safetensors"
        quant_checkpoint_file = filename
        state_dict = native_model.state_dict()
        for key in tqdm(state_dict, desc="cast state_dict"):
            value = state_dict[key]
            if key.endswith("quant_weight"):
                if value.min().item() >= -(2**7) and value.max().item() <= (2**7) - 1:
                    value = value.to(torch.int8)
                elif value.min().item() >= -(2**15) and value.max().item() <= (2**15) - 1:
                    value = value.to(torch.int16)
                else:
                    value = value.to(torch.float32)
            else:
                value = value.to(torch.float16)
            state_dict[key] = value
        save_safetensors_file(state_dict, str(filename))
        logger.info(f"Saved quant checkpoint to {filename}")

    if args.rotate_audio_projector:
        q_matrix = cached_q_matrix
        if q_matrix is None:
            q_matrix = getattr(native_model, "_quarot_rotation_matrix", None)
        if q_matrix is None:
            q_matrix = _load_rotation_matrix_file(args.rotation_matrix_out, logger)
        if q_matrix is None:
            q_matrix = _load_rotation_matrix_file(str(default_rotation_out), logger)
        if q_matrix is None:
            raise RuntimeError(
                "rotate_audio_projector enabled, but no quarot rotation matrix is available. "
                "Please enable quarot and ensure matrix is exported/saved correctly."
            )
        rotated_adapter = rotate_adapter_with_llm_matrix(adapter_state_dict, q_matrix, logger)
        out_file = args.rotated_adapter_out
        if out_file is None:
            out_file = str(Path(cfg.work_dir) / "audio_projector_rotated.safetensors")
        save_safetensors_file(rotated_adapter, out_file)
        logger.info(f"Saved rotated audio projector to {out_file}")

    if args.valid and quant_checkpoint_file is not None and quant_checkpoint_file.exists():
        if stage_logits is None and valid_input_ids is not None:
            stage_logits = _run_single_forward_logits(native_model, valid_input_ids, cfg.device)
        if stage_logits is not None and tokenizer is not None and valid_input_ids is not None:
            native_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _validate_saved_quant_checkpoint(
                checkpoint_file=quant_checkpoint_file,
                cfg=cfg,
                args=args,
                valid_input_ids=valid_input_ids,
                in_memory_logits=stage_logits,
                tokenizer=tokenizer,
                lora_state_dict=lora_state_dict,
                lora_config=lora_config,
                logger=logger,
            )

    if args.valid_asr:
        if quant_checkpoint_file is None or not quant_checkpoint_file.exists():
            logger.warning("Skip valid_asr: quant checkpoint not found.")
            return
        native_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_quant_model = _load_quantized_hf_model_from_checkpoint(
            checkpoint_file=quant_checkpoint_file,
            cfg=cfg,
            args=args,
            lora_state_dict=lora_state_dict,
            lora_config=lora_config,
        )
        if args.lora_mode == "keep_lora":
            _materialize_runtime_lora_into_weight(eval_quant_model, lora_config, logger)

        rotated_adapter_path = args.rotated_adapter_out
        if rotated_adapter_path is None:
            inferred_rotated = Path(cfg.work_dir) / "audio_projector_rotated.safetensors"
            if inferred_rotated.exists():
                rotated_adapter_path = str(inferred_rotated)

        asr_summary = validate_asr_quantized_hf_llm(
            fireredasr_model_dir=args.fireredasr_model_dir,
            quantized_llm=eval_quant_model,
            wav_dir=args.wav_dir,
            ref_text_file=args.ref_text,
            use_gpu=args.use_gpu,
            logger=logger,
            rotated_adapter_path=rotated_adapter_path,
        )
        logger.info(f"valid_asr summary: {asr_summary}")


if __name__ == "__main__":
    main()
