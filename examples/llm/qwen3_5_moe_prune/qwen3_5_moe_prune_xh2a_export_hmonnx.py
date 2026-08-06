import argparse
import json
import logging
import os
import os.path as osp
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import onnx
import torch
import torch.nn as nn
from safetensors import safe_open

from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler
from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_5_moe_prune import Qwen3_5MoePruneConvertConfig
from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init


# Suppress noisy onnxscript logs during golden/export
logging.getLogger("onnxscript.rewriter.rules.common._collapse_slices").setLevel(logging.WARNING)
logging.getLogger("onnxscript.optimizer._constant_folding").setLevel(logging.WARNING)
logging.getLogger("onnx_ir.passes.common.initializer_deduplication").setLevel(logging.WARNING)


FP16_MAX_FINITE_POSITION = 65504


def _validate_offline_rope_required_for_fp16_limit(args):
    if getattr(args, "support_long_context_over_fp16_limit", True):
        return

    checked_ranges = {
        "max_pe_length": int(getattr(args, "max_pe_length", 0) or 0),
        "context_length": int(getattr(args, "context_length", 0) or 0),
        "input_sequence_length": int(getattr(args, "input_sequence_length", 0) or 0),
    }
    offenders = {name: value for name, value in checked_ranges.items() if value > FP16_MAX_FINITE_POSITION}
    if not offenders:
        return

    offender_text = ", ".join(f"{name}={value}" for name, value in offenders.items())
    raise ValueError(
        "--no-support_long_context_over_fp16_limit is invalid because "
        f"{offender_text} exceeds fp16 max finite value {FP16_MAX_FINITE_POSITION}. "
        "Keep offline RoPE enabled for long-context exports."
    )


_NORM_WEIGHT_SUFFIXES = (
    "pre_feedforward_layernorm_2.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "input_layernorm.weight",
)


def _load_weight_map(hf_model_dir: Path) -> Dict[str, str]:
    index_path = hf_model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {str(key): str(value) for key, value in payload["weight_map"].items()}

    weight_map: Dict[str, str] = {}
    for weight_file in sorted(hf_model_dir.glob("*.safetensors")):
        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = weight_file.name
    if not weight_map:
        raise FileNotFoundError(f"No safetensors checkpoint found under {hf_model_dir}")
    return weight_map


def _load_checkpoint_tensor(hf_model_dir: Path, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    with safe_open(hf_model_dir / weight_map[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _find_fused_moe_layer_keys(weight_map: Dict[str, str]):
    layer_keys = []
    pattern = re.compile(r"^(?P<prefix>.*layers\.(?P<layer_idx>\d+)\.)mlp\.experts\.gate_up_proj(?:\.weight)?$")
    for gate_up_key in weight_map:
        match = pattern.match(gate_up_key)
        if match is None:
            continue

        layer_idx = int(match.group("layer_idx"))
        prefix = match.group("prefix")
        down_key = f"{prefix}mlp.experts.down_proj"
        if down_key not in weight_map:
            down_key = f"{down_key}.weight"
        if down_key not in weight_map:
            raise KeyError(f"Unable to find down_proj for layer {layer_idx} from {gate_up_key}")

        gamma_key = None
        for suffix in _NORM_WEIGHT_SUFFIXES:
            candidate = f"{prefix}{suffix}"
            if candidate in weight_map:
                gamma_key = candidate
                break
        if gamma_key is None:
            raise KeyError(f"Unable to find norm gamma weight for layer {layer_idx}")

        layer_keys.append((layer_idx, gamma_key, gate_up_key, down_key))
    if not layer_keys:
        raise RuntimeError("No fused Qwen3.5-MoE expert weights found in checkpoint")
    return sorted(layer_keys)


def _resolve_s_scalar_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _compute_expert_slanc_exact(
    gamma: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    e_weight = up_proj_weight.transpose(0, 1).contiguous()
    b_weight = gate_proj_weight.transpose(0, 1).contiguous()
    g_weight = down_proj_weight.transpose(0, 1).contiguous()

    norm_gamma_e = torch.norm(gamma[:, None] * e_weight, p="fro")
    norm_gamma_b = torch.norm(gamma[:, None] * b_weight, p="fro")
    bg = b_weight @ g_weight
    eg = e_weight @ g_weight
    a_e = torch.norm(gamma[:, None] * (norm_gamma_e * bg), p="fro")
    a_b = torch.norm(gamma[:, None] * (norm_gamma_b * eg), p="fro")
    return torch.sqrt(a_e * a_b + eps)


@torch.no_grad()
def _build_s_scalar_for_layer(
    gamma: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    compute_device: str,
    eps: float,
) -> torch.Tensor:
    gamma = gamma.to(device=compute_device, dtype=torch.float32)
    raw_scores = []
    for expert_idx in range(gate_up_proj.shape[0]):
        gate_up_weight = gate_up_proj[expert_idx].to(device=compute_device, dtype=torch.float32)
        gate_proj_weight, up_proj_weight = gate_up_weight.chunk(2, dim=0)
        down_proj_weight = down_proj[expert_idx].to(device=compute_device, dtype=torch.float32)
        raw_scores.append(
            _compute_expert_slanc_exact(
                gamma,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
                eps,
            ).cpu()
        )
        del gate_up_weight, gate_proj_weight, up_proj_weight, down_proj_weight

    raw = torch.stack(raw_scores).float()
    return raw / (raw.mean() + eps)


def _build_method1_s_scalars(hf_model_dir: Path, compute_device: str, eps: float, logger):
    weight_map = _load_weight_map(hf_model_dir)
    layer_keys = _find_fused_moe_layer_keys(weight_map)
    s_scalars = {}
    for layer_idx, gamma_key, gate_up_key, down_key in layer_keys:
        logger.info(f"Computing method1 s_scalar for layer {layer_idx}: {gate_up_key}")
        gamma = _load_checkpoint_tensor(hf_model_dir, weight_map, gamma_key)
        gate_up_proj = _load_checkpoint_tensor(hf_model_dir, weight_map, gate_up_key)
        down_proj = _load_checkpoint_tensor(hf_model_dir, weight_map, down_key)
        s_scalars[str(layer_idx)] = _build_s_scalar_for_layer(
            gamma,
            gate_up_proj,
            down_proj,
            compute_device,
            eps,
        )
        del gamma, gate_up_proj, down_proj
        if compute_device.startswith("cuda"):
            torch.cuda.empty_cache()
    return s_scalars


def _resolve_s_scalar_path(args, hf_model_dir: Path, work_dir: Path, logger):
    if args.s_scalar_path is not None:
        return args.s_scalar_path
    if not args.auto_s_scalar:
        return None

    s_scalar_path = Path(args.s_scalar_output) if args.s_scalar_output else work_dir / "method1_s_scalar.pt"
    if s_scalar_path.exists() and not args.overwrite_s_scalar:
        logger.info(f"Reuse existing method1 s_scalar: {s_scalar_path}")
        return str(s_scalar_path)

    s_scalar_path.parent.mkdir(exist_ok=True, parents=True)
    compute_device = _resolve_s_scalar_device(args.s_scalar_compute_device)
    logger.info(f"Computing method1 s_scalar on {compute_device}, output={s_scalar_path}")
    s_scalars = _build_method1_s_scalars(hf_model_dir, compute_device, args.s_scalar_eps, logger)
    torch.save(s_scalars, s_scalar_path)
    logger.info(f"Saved method1 s_scalar for {len(s_scalars)} layers to {s_scalar_path}")
    return str(s_scalar_path)


# ─────────────────────────────────────────────────────────────────────────────
# Golden / release helpers (verbatim from qwen3_5_moe_xh2a_export_hmonnx.py)
# ─────────────────────────────────────────────────────────────────────────────


def _get_default_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


_PURE_FIXED_POINT_WMIX_RE = re.compile(r"^w\d+a\d+$")


def _normalize_wmix_amix(value: str) -> str:
    if value is None:
        return "wmix_amix"
    s = str(value).strip().lower()
    if not s:
        return "wmix_amix"
    if _PURE_FIXED_POINT_WMIX_RE.match(s):
        return s
    return "wmix_amix"


def _detect_release_wmix_amix(hf_model_dir: str) -> str:
    return "wmix_amix"


def _resolve_token_embedding_path(work_dir: Path) -> Optional[Path]:
    for name in ("quant_embedding.pt", "token_embedding.pt"):
        candidate = work_dir / name
        if candidate.exists():
            return candidate
    return None


def _build_release_prefix_moe(args, hf_model_path: str) -> str:
    xh_version = (getattr(args, "release_xh_version", None) or "xh2").strip().lower()
    if xh_version not in {"xh1", "xh2"}:
        raise ValueError(f"release_xh_version must be one of 'xh1' / 'xh2', got: {xh_version!r}")

    modelscope_name = getattr(args, "release_modelscope_name", None)
    if not modelscope_name:
        modelscope_name = Path(hf_model_path).name
    modelscope_name = str(modelscope_name).strip().lower().replace(".", "_").replace("-", "_").replace(" ", "_")

    wmix_amix_raw = getattr(args, "release_wmix_amix", None) or _detect_release_wmix_amix(hf_model_path)
    wmix_amix = _normalize_wmix_amix(wmix_amix_raw)

    prefill_len = int(args.input_sequence_length)
    ctx_len = int(args.context_length)
    ctx_str = f"{ctx_len // 1024}k" if ctx_len % 1024 == 0 else str(ctx_len)

    date_str = getattr(args, "release_date", None)
    if not date_str:
        date_str = time.strftime("%Y%m%d")
    date_str = str(date_str).strip().lower()

    return f"hmquant_{xh_version}_{modelscope_name}_{wmix_amix}_{prefill_len}_{ctx_str}_{date_str}"


def _save_onnx_with_renamed_external_data(src_onnx: Path, dst_onnx: Path, new_external_data_name: str, logger) -> None:
    dst_onnx.parent.mkdir(parents=True, exist_ok=True)
    stale = dst_onnx.parent / new_external_data_name
    if stale.exists() or stale.is_symlink():
        try:
            stale.unlink()
        except OSError:
            pass
    if dst_onnx.exists() or dst_onnx.is_symlink():
        try:
            dst_onnx.unlink()
        except OSError:
            pass
    logger.info(
        f"Resaving ONNX with renamed external_data: {src_onnx.name} -> {dst_onnx.name} (ext={new_external_data_name})"
    )
    model = onnx.load(str(src_onnx), load_external_data=True)
    onnx.save_model(
        model,
        str(dst_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=new_external_data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _rename_golden_to_short_format(golden_dir: Path, release_prefix: str, role: str, logger) -> None:
    if role is None:
        return
    step0_dir = golden_dir / "step_0"
    if not step0_dir.is_dir():
        return

    prefix_stripped = release_prefix.split("_", 2)[-1]
    segments = prefix_stripped.split("_")
    scheme_keywords = {
        "xh1",
        "xh2",
        "wmix",
        "amix",
        "w4a8",
        "w8a8",
        "w4",
        "w8",
        "a4",
        "a8",
        "256",
        "2k",
        "4k",
        "8k",
        "16k",
        "32k",
        "h0",
        "h1",
        "ssfp",
        "sefp",
        "fp16",
        "fp32",
        "gptq",
    }
    model_parts = []
    for seg in segments:
        if seg.lower() in scheme_keywords:
            break
        model_parts.append(seg)
    model_name = "_".join(model_parts)
    short_prefix = f"hmquant_{model_name}"

    role_tokens = [
        "prefill",
        "decode",
        "draft_prefill",
        "draft_decode",
        "draft_context",
        "draft_context_decode",
    ]

    for fpath in list(step0_dir.iterdir()):
        if fpath.is_dir():
            if fpath.name.endswith("_with_act") and not fpath.name.startswith(short_prefix):
                new_name = f"{short_prefix}_with_act"
                new_path = step0_dir / new_name
                if new_path.exists():
                    shutil.rmtree(new_path)
                logger.info(f"Rename golden dir: {fpath.name} → {new_name}")
                shutil.move(str(fpath), str(new_path))
            continue
        if not fpath.name.endswith(".npy"):
            continue

        name_without_ext = fpath.name[:-4]
        if "_output" in name_without_ext:
            suffix_idx = name_without_ext.rfind("_output")
            direction = "_output"
        elif "_input" in name_without_ext:
            suffix_idx = name_without_ext.rfind("_input")
            direction = "_input"
        else:
            logger.warning(f"Cannot parse golden file name (no _output/_input): {fpath.name}")
            continue

        model_prefix = short_prefix
        after_model = name_without_ext[len(model_prefix) : suffix_idx]
        rest = after_model.lstrip("_")
        io_name = rest
        for rt in role_tokens:
            if rest.startswith(rt + "_"):
                io_name = rest[len(rt) + 1 :]
                break
            for mode_prefix in ("xh2a_", "xh2_", "xh1a_", "xh1_"):
                combined = mode_prefix + rt + "_"
                if rest.startswith(combined):
                    io_name = rest[len(combined) :]
                    break
                if rt in ("prefill", "decode") and rest.startswith(rt + "_"):
                    io_name = rest[len(rt) + 1 :]
                    break
            else:
                continue
            break

        new_name = f"{model_prefix}_{io_name}{direction}.npy"
        new_path = step0_dir / new_name
        logger.info(f"Rename golden file: {fpath.name} → {new_name}")
        shutil.move(str(fpath), str(new_path))


def _create_step0_onnx_symlinks(golden_dir: Path, release_prefix: str, logger) -> None:
    step0_dir = golden_dir / "step_0"
    step0_dir.mkdir(parents=True, exist_ok=True)

    prefix_stripped = release_prefix.split("_", 2)[-1]
    segments = prefix_stripped.split("_")
    scheme_keywords = {
        "xh1",
        "xh2",
        "wmix",
        "amix",
        "w4a8",
        "w8a8",
        "w4",
        "w8",
        "a4",
        "a8",
        "256",
        "2k",
        "4k",
        "8k",
        "16k",
        "32k",
        "h0",
        "h1",
        "ssfp",
        "sefp",
        "fp16",
        "fp32",
        "gptq",
    }
    model_parts = []
    for seg in segments:
        if seg.lower() in scheme_keywords:
            break
        model_parts.append(seg)
    model_name = "_".join(model_parts)

    onnx_short = f"hmquant_{model_name}_with_act.onnx"
    onnx_actual = f"{release_prefix}_with_act.onnx"
    ext_actual = f"{release_prefix}_external_data"

    src = golden_dir / onnx_actual
    if src.exists():
        link = step0_dir / onnx_short
        if link.exists() or link.is_symlink():
            try:
                link.unlink()
            except OSError:
                pass
        try:
            os.symlink(os.path.relpath(src, start=step0_dir), link)
            logger.info(f"step_0 symlink: {link} -> {os.readlink(link)}")
        except OSError as exc:
            logger.warning(f"Symlink failed ({exc}); falling back to copy {src} -> {link}")
            shutil.copy2(src, link)

    src = golden_dir / ext_actual
    if src.exists():
        link = step0_dir / ext_actual
        if link.exists() or link.is_symlink():
            try:
                link.unlink()
            except OSError:
                pass
        try:
            os.symlink(os.path.relpath(src, start=step0_dir), link)
            logger.info(f"step_0 symlink: {link} -> {os.readlink(link)}")
        except OSError as exc:
            logger.warning(f"Symlink failed ({exc}); falling back to copy {src} -> {link}")
            shutil.copy2(src, link)


def _copy_release_root_assets(args, release_dir: Path, release_prefix: str, work_dir: Path, logger) -> None:
    release_dir.mkdir(parents=True, exist_ok=True)
    script_src = Path(__file__).resolve()
    script_dst = release_dir / f"{release_prefix}_hmonnx.py"
    try:
        if script_dst.exists() or script_dst.is_symlink():
            script_dst.unlink()
        shutil.copy2(script_src, script_dst)
        logger.info(f"Release root: copied script -> {script_dst.name}")
    except OSError as exc:
        logger.warning(f"Failed to copy export script into release_dir: {exc}")

    log_pairs = [
        (work_dir / "convert.log", release_dir / f"{release_prefix}_hmonnx_debug.log"),
        (work_dir / "golden.log", release_dir / f"{release_prefix}_hmonnx_debug.log"),
    ]
    for src, dst in log_pairs:
        if not src.exists():
            continue
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst)
            logger.info(f"Release root: copied log {src.name} -> {dst.name}")
        except OSError as exc:
            logger.warning(f"Failed to copy log {src} into release_dir: {exc}")


def _load_token_embedding(embed_path: Path) -> nn.Module:
    try:
        obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(embed_path), map_location="cpu")
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj
    if isinstance(obj, dict):
        if "weight" not in obj:
            raise ValueError(f"Unsupported token embedding state dict format: {embed_path}")
        emb = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1])
        emb.load_state_dict(obj)
        emb.eval()
        return emb
    raise TypeError(f"Unsupported token embedding object type: {type(obj)}")


def _build_inputs_embeds(
    token_embedding: nn.Module,
    input_ids: torch.Tensor,
    target_seq_len: int,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    seq_len = input_ids.shape[1]
    if seq_len < target_seq_len:
        pad = torch.full(
            (input_ids.shape[0], target_seq_len - seq_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, pad], dim=1)
    elif seq_len > target_seq_len:
        input_ids = input_ids[:, :target_seq_len]
    embedding_device = token_embedding.weight.device
    with torch.no_grad():
        embeds = token_embedding(input_ids.to(embedding_device))
    return embeds.to(dtype=dtype, device=device)


def _is_cache_input_name(name: str) -> bool:
    return "cache" in name.lower() or "state" in name.lower()


def _ensure_cache_tensor(tensor: torch.Tensor):
    from xhquant.xhonnxruntime.parsers.llm_cache import CacheTensor

    if isinstance(tensor, CacheTensor):
        return tensor
    return CacheTensor(tensor)


def _alloc_cache_inputs(session, device: torch.device) -> Dict[str, torch.Tensor]:
    cache_inputs = {}
    for name in session.get_input_names():
        if _is_cache_input_name(name):
            info = session.get_input(name)
            cache_inputs[name] = _ensure_cache_tensor(torch.zeros(info.shape, dtype=info.dtype, device=device))
    return cache_inputs


def _build_linear_attn_mask(valid_len: int, mask_info, device: torch.device) -> torch.Tensor:
    mask = torch.ones(mask_info.shape, dtype=mask_info.dtype, device=device)
    return mask


def _resolve_input_name(session, candidates, fallback=None) -> str:
    input_names = session.get_input_names()
    for name in candidates:
        if name in input_names:
            return name
        batch0 = f"{name}_batch_0"
        if batch0 in input_names:
            return batch0
    if fallback is not None:
        return fallback(session)
    raise ValueError(f"None of {candidates} found in inputs: {input_names}")


def _batch_suffix(name: str, base: str) -> Optional[int]:
    prefix = f"{base}_batch_"
    if name.startswith(prefix):
        return int(name[len(prefix) :])
    return None


def _feed_batched_tensor(feed: Dict[str, torch.Tensor], name: str, base: str, tensor: torch.Tensor) -> bool:
    if name == base:
        feed[name] = tensor
        return True
    batch_idx = _batch_suffix(name, base)
    if batch_idx is not None:
        feed[name] = tensor[batch_idx : batch_idx + 1]
        return True
    return False


def _feed_any_batched_tensor(feed: Dict[str, torch.Tensor], name: str, bases, tensor: torch.Tensor) -> bool:
    for base in bases:
        if _feed_batched_tensor(feed, name, base, tensor):
            return True
    return False


def _extract_logits_from_output_map(output_map: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if "logits" in output_map:
        return output_map["logits"]
    logits_items = []
    idx = 0
    while f"logits_batch_{idx}" in output_map:
        logits_items.append(output_map[f"logits_batch_{idx}"])
        idx += 1
    if logits_items:
        return torch.cat(logits_items, dim=0)
    return None


def _infer_inputs_embeds_name(session) -> str:
    for name in session.get_input_names():
        info = session.get_input(name)
        if info.dtype in (torch.float16, torch.float32, torch.bfloat16) and len(info.shape) == 3:
            return name
    return session.get_input_names()[0]


def _create_golden_session(onnx_file: str, golden_dir: Path, device: torch.device, logger):
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXGoldenInference

    golden_dir.mkdir(exist_ok=True, parents=True)
    session = HMONNXGoldenInference(onnx_file)
    session.exec_device = device
    session.save_golden = True
    session.golden_dir = golden_dir
    session.initialize()
    # Move the parsed full-network model before cache/input tensors are
    # allocated. Otherwise run() moves the model after inputs already occupy
    # GPU memory, causing an avoidable peak-memory OOM.
    session.to(device)
    return session


def _run_hmonnx_with_golden(session, input_feed: Dict[str, torch.Tensor]):
    outputs = session.run(input_feed)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    output_names = session.get_output_names()
    output_map = {name: out for name, out in zip(output_names, outputs, strict=True)}
    return tuple(outputs), output_map


def _find_external_data(onnx_path: Path) -> Optional[Path]:
    ext_path = onnx_path.parent / f"{onnx_path.stem}_external_data"
    if ext_path.exists():
        return ext_path
    return None


def _copy_path(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.resolve())


def _ensure_step0_layout(golden_dir: Path, logger) -> None:
    step0_dir = golden_dir / "step_0"
    step0_dir.mkdir(exist_ok=True, parents=True)
    for item in list(golden_dir.iterdir()):
        if item.name == "step_0":
            continue
        if item.name.endswith(".onnx") or item.name.endswith("_external_data"):
            continue
        if item.is_dir() and item.name.startswith("step_"):
            continue
        if item.name == "logits.npy" or item.name.startswith("hmquant_") or item.name.startswith("Qwen"):
            target = step0_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            logger.info(f"Move golden artifact into step_0: {item} -> {target}")
            shutil.move(str(item), str(target))


def _cleanup_hmonnx_in_golden(golden_dir: Path, release_prefix: str, role: Optional[str], logger) -> None:
    _ensure_step0_layout(golden_dir, logger)
    if role:
        _rename_golden_to_short_format(golden_dir, release_prefix, role, logger)

    for step_dir in sorted(golden_dir.glob("step_*")):
        if not step_dir.is_dir():
            continue
        if step_dir.name != "step_0":
            logger.info(f"Cleanup: removing extra golden step dir {step_dir}")
            shutil.rmtree(step_dir)
            continue
        for fpath in step_dir.iterdir():
            if fpath.is_dir() and not fpath.is_symlink():
                continue
            if fpath.name.endswith(".onnx") or fpath.name.endswith("_external_data"):
                logger.info(f"Cleanup: removing {fpath}")
                fpath.unlink()

    for fpath in golden_dir.iterdir():
        if fpath.name.startswith(release_prefix):
            continue
        if fpath.name == "step_0":
            continue
        if fpath.is_dir() and not fpath.is_symlink():
            continue
        if fpath.name.endswith(".onnx"):
            logger.info(f"Cleanup: removing {fpath}")
            fpath.unlink()


def _generate_draft_golden_for_onnx(
    work_dir: Path,
    args,
    onnx_file: str,
    golden_dir: Path,
    token_embedding: nn.Module,
    pad_token_id: int,
    input_ids_full: torch.Tensor,
    is_decode: bool,
    device: torch.device,
    logger,
) -> None:
    """Run draft ONNX once and let HMONNXGoldenInference dump step_0 golden."""
    golden_dir.mkdir(exist_ok=True, parents=True)
    valid_len = input_ids_full.shape[1]
    past_seq_val = valid_len if is_decode else 0

    session = _create_golden_session(onnx_file, golden_dir, device, logger)
    if hasattr(session, "legacy_mode"):
        session.legacy_mode = False

    cache_inputs = _alloc_cache_inputs(session, device)
    float3d_names = [
        name
        for name in session.get_input_names()
        if session.get_input(name).dtype in (torch.float16, torch.float32, torch.bfloat16)
        and len(session.get_input(name).shape) == 3
        and name not in cache_inputs
    ]
    model_seq_len = session.get_input(float3d_names[0]).shape[1] if float3d_names else 1

    input_feed: Dict[str, torch.Tensor] = {}
    for name in session.get_input_names():
        if name in cache_inputs:
            input_feed[name] = cache_inputs[name]
            continue

        info = session.get_input(name)
        if name in ("past_seq_length", "valid_length"):
            batch = info.shape[0] if info.shape else 1
            input_feed[name] = torch.tensor([past_seq_val] * batch, dtype=info.dtype, device=device)
        elif name in ("current_input_length", "current_length"):
            batch = info.shape[0] if info.shape else 1
            input_feed[name] = torch.tensor([model_seq_len] * batch, dtype=info.dtype, device=device)
        elif name in ("linear_attn_mask", "attention_mask", "attn_mask"):
            input_feed[name] = torch.ones(info.shape, dtype=info.dtype, device=device)
        elif info.dtype in (torch.float16, torch.float32, torch.bfloat16) and len(info.shape) == 3:
            embed_dim = token_embedding.weight.shape[1]
            if info.shape[2] == embed_dim:
                input_feed[name] = _build_inputs_embeds(
                    token_embedding, input_ids_full, info.shape[1], pad_token_id, device, info.dtype
                )
            else:
                input_feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=device)
        elif info.dtype in (torch.int32, torch.int64) and len(info.shape) == 2:
            seq_len = info.shape[1]
            if seq_len <= input_ids_full.shape[1]:
                ids = input_ids_full[:, :seq_len].to(device=device, dtype=info.dtype)
            else:
                pad_ids = torch.full(
                    (input_ids_full.shape[0], seq_len - input_ids_full.shape[1]),
                    pad_token_id,
                    dtype=info.dtype,
                    device=device,
                )
                ids = torch.cat([input_ids_full.to(device=device, dtype=info.dtype), pad_ids], dim=1)
            input_feed[name] = ids
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids"):
            numel = 1
            for dim in info.shape:
                numel *= dim
            pos = torch.arange(past_seq_val, past_seq_val + numel, device=device, dtype=info.dtype)
            input_feed[name] = pos.reshape(info.shape)
        else:
            tensor = torch.zeros(info.shape, dtype=info.dtype, device=device)
            if _is_cache_input_name(name):
                tensor = _ensure_cache_tensor(tensor)
            input_feed[name] = tensor

    _run_hmonnx_with_golden(session, input_feed)
    del session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_golden(
    work_dir: Path,
    args,
    input_ids_full: torch.Tensor,
    tokenizer,
    prefill_onnx_file: str,
    decode_onnx_file: str,
    logger,
    draft_onnx_files: Optional[Dict[str, str]] = None,
    spec_decode_mode: Optional[str] = None,
) -> Path:
    device = _get_default_device()
    dtype = torch.float16

    token_embedding_file = _resolve_token_embedding_path(work_dir)
    if token_embedding_file is None:
        raise FileNotFoundError(
            f"Neither quant_embedding.pt nor token_embedding.pt found under {work_dir}. "
            "Re-run the export step or supply a completed work_dir."
        )
    # Keep the ~1 GB embedding module on CPU. Only the small lookup result is
    # transferred to the execution device, leaving more GPU memory for the
    # full-network HMONNX weights during golden generation.
    token_embedding = _load_token_embedding(token_embedding_file).to(dtype)
    token_embedding.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    valid_len = input_ids_full.shape[1]
    work_meta = _load_work_meta(work_dir, logger)
    export_batch = int(work_meta.get("wrap_cfg", {}).get("batch_size", getattr(args, "batch_size", 1)))
    if input_ids_full.shape[0] == export_batch:
        input_ids_batch = input_ids_full.contiguous()
    elif input_ids_full.shape[0] == 1:
        input_ids_batch = input_ids_full.expand(export_batch, -1).contiguous()
    else:
        raise ValueError(
            f"Cannot build golden inputs for export_batch={export_batch} from prompt batch={input_ids_full.shape[0]}"
        )
    logger.info(f"Golden export batch: {export_batch}")

    hf_model_path = osp.normpath(osp.abspath(args.model))
    release_prefix = _build_release_prefix_moe(args, hf_model_path)
    release_dir = work_dir / release_prefix
    prefill_dir = release_dir / "prefill"
    decode_dir = release_dir / "decode"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    decode_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Release prefix: {release_prefix}")
    logger.info(f"Release directory: {release_dir}")

    prefill_onnx_path = Path(prefill_onnx_file)
    decode_onnx_path = Path(decode_onnx_file)
    named_prefill_onnx = prefill_dir / f"{release_prefix}_with_act.onnx"
    named_decode_onnx = decode_dir / f"{release_prefix}_with_act.onnx"

    quant_embedding_file = release_dir / "quant_embedding.pt"
    if token_embedding_file.exists() and not quant_embedding_file.exists():
        shutil.copy2(token_embedding_file, quant_embedding_file)
        logger.info(f"Copied quant_embedding.pt to {quant_embedding_file}")

    hf_config_src = work_dir / "hf_config"
    hf_config_dst = release_dir / "hf_config"
    if hf_config_src.exists() and not hf_config_dst.exists():
        shutil.copytree(hf_config_src, hf_config_dst)
        logger.info(f"Copied hf_config -> {hf_config_dst}")
    elif not hf_config_src.exists():
        logger.warning(f"hf_config not found under {work_dir}; release dir will be incomplete per HM 命名规则")

    logger.info("Generating prefill golden...")
    prefill_session = _create_golden_session(str(prefill_onnx_path), prefill_dir, device, logger)
    if hasattr(prefill_session, "legacy_mode"):
        prefill_session.legacy_mode = False

    prefill_inputs_name = _resolve_input_name(
        prefill_session, ("inputs_embeds", "input_1"), fallback=_infer_inputs_embeds_name
    )
    prefill_past_seq_name = _resolve_input_name(prefill_session, ("past_seq_length", "valid_length"))
    prefill_current_seq_name = _resolve_input_name(prefill_session, ("current_input_length", "current_length"))
    prefill_mask_name = _resolve_input_name(prefill_session, ("linear_attn_mask", "attention_mask", "attn_mask"))

    prefill_inputs_info = prefill_session.get_input(prefill_inputs_name)
    prefill_mask_info = prefill_session.get_input(prefill_mask_name)
    prefill_past_seq_info = prefill_session.get_input(prefill_past_seq_name)
    prefill_current_seq_info = prefill_session.get_input(prefill_current_seq_name)

    prefill_inputs_embeds = _build_inputs_embeds(
        token_embedding, input_ids_batch, prefill_inputs_info.shape[1], pad_token_id, device, prefill_inputs_info.dtype
    )
    prefill_linear_attn_mask = _build_linear_attn_mask(valid_len, prefill_mask_info, device)
    if export_batch > 1:
        prefill_linear_attn_mask = prefill_linear_attn_mask.expand(export_batch, -1).contiguous()
    prefill_batch = export_batch if prefill_inputs_name.endswith("_batch_0") else prefill_inputs_info.shape[0]
    prefill_past_seq_length = torch.tensor([0] * prefill_batch, dtype=prefill_past_seq_info.dtype, device=device)
    prefill_current_input_length = torch.tensor(
        [valid_len] * prefill_batch, dtype=prefill_current_seq_info.dtype, device=device
    )

    prefill_seq_len = prefill_inputs_info.shape[1]
    prefill_cache_inputs = _alloc_cache_inputs(prefill_session, device)
    prefill_input_feed: Dict[str, torch.Tensor] = {}
    for name in prefill_session.get_input_names():
        if _feed_any_batched_tensor(prefill_input_feed, name, ("inputs_embeds", "input_1"), prefill_inputs_embeds):
            pass
        elif _feed_any_batched_tensor(
            prefill_input_feed,
            name,
            ("past_seq_length", "valid_length"),
            prefill_past_seq_length,
        ):
            pass
        elif _feed_any_batched_tensor(
            prefill_input_feed,
            name,
            ("current_input_length", "current_length"),
            prefill_current_input_length,
        ):
            pass
        elif _feed_any_batched_tensor(
            prefill_input_feed,
            name,
            ("linear_attn_mask", "attention_mask", "attn_mask"),
            prefill_linear_attn_mask,
        ):
            pass
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids") or any(
            _batch_suffix(name, base) is not None
            for base in ("time_position_ids", "hight_position_ids", "width_position_ids")
        ):
            info = prefill_session.get_input(name)
            pos_batch = (
                torch.arange(0, prefill_seq_len, device=device, dtype=info.dtype).view(1, -1).expand(export_batch, -1)
            )
            for base in ("time_position_ids", "hight_position_ids", "width_position_ids"):
                batch_idx = _batch_suffix(name, base)
                if batch_idx is not None:
                    prefill_input_feed[name] = pos_batch[batch_idx : batch_idx + 1]
                    break
            else:
                prefill_input_feed[name] = pos_batch.reshape(info.shape)
        elif name in prefill_cache_inputs:
            prefill_input_feed[name] = prefill_cache_inputs[name]
        else:
            info = prefill_session.get_input(name)
            t = torch.zeros(info.shape, dtype=info.dtype, device=device)
            if _is_cache_input_name(name):
                t = _ensure_cache_tensor(t)
            prefill_input_feed[name] = t

    _, prefill_output_map = _run_hmonnx_with_golden(prefill_session, prefill_input_feed)
    prefill_logits = _extract_logits_from_output_map(prefill_output_map)
    if prefill_logits is not None:
        if args.num_logits_to_keep == 0:
            prefill_logits = prefill_logits[:, valid_len - 1 : valid_len, :]
        next_token_id = prefill_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        logger.info(f"Prefill golden next token ids: {next_token_id.reshape(-1).tolist()}")
    else:
        next_token_id = torch.zeros((export_batch, 1), dtype=torch.int64, device=device)

    del prefill_session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Generating decode golden...")
    decode_session = _create_golden_session(str(decode_onnx_path), decode_dir, device, logger)
    if hasattr(decode_session, "legacy_mode"):
        decode_session.legacy_mode = False

    decode_inputs_name = _resolve_input_name(
        decode_session, ("inputs_embeds", "input_1"), fallback=_infer_inputs_embeds_name
    )
    decode_past_seq_name = _resolve_input_name(decode_session, ("past_seq_length", "valid_length"))
    decode_current_seq_name = _resolve_input_name(decode_session, ("current_input_length", "current_length"))
    decode_mask_name = _resolve_input_name(decode_session, ("linear_attn_mask", "attention_mask", "attn_mask"))

    decode_inputs_info = decode_session.get_input(decode_inputs_name)
    decode_mask_info = decode_session.get_input(decode_mask_name)
    decode_past_seq_info = decode_session.get_input(decode_past_seq_name)
    decode_current_seq_info = decode_session.get_input(decode_current_seq_name)

    decode_inputs_embeds = _build_inputs_embeds(
        token_embedding, next_token_id, decode_inputs_info.shape[1], pad_token_id, device, decode_inputs_info.dtype
    )
    decode_linear_attn_mask = _build_linear_attn_mask(1, decode_mask_info, device)
    if export_batch > 1:
        decode_linear_attn_mask = decode_linear_attn_mask.expand(export_batch, -1).contiguous()
    decode_batch = export_batch if decode_inputs_name.endswith("_batch_0") else decode_inputs_info.shape[0]
    decode_past_seq_length = torch.tensor([valid_len] * decode_batch, dtype=decode_past_seq_info.dtype, device=device)
    decode_current_input_length = torch.tensor([1] * decode_batch, dtype=decode_current_seq_info.dtype, device=device)

    decode_input_feed: Dict[str, torch.Tensor] = {}
    for name in decode_session.get_input_names():
        if _feed_any_batched_tensor(decode_input_feed, name, ("inputs_embeds", "input_1"), decode_inputs_embeds):
            pass
        elif _feed_any_batched_tensor(
            decode_input_feed,
            name,
            ("past_seq_length", "valid_length"),
            decode_past_seq_length,
        ):
            pass
        elif _feed_any_batched_tensor(
            decode_input_feed,
            name,
            ("current_input_length", "current_length"),
            decode_current_input_length,
        ):
            pass
        elif _feed_any_batched_tensor(
            decode_input_feed,
            name,
            ("linear_attn_mask", "attention_mask", "attn_mask"),
            decode_linear_attn_mask,
        ):
            pass
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids") or any(
            _batch_suffix(name, base) is not None
            for base in ("time_position_ids", "hight_position_ids", "width_position_ids")
        ):
            info = decode_session.get_input(name)
            n_decode_tokens = decode_inputs_info.shape[1]
            decode_pos_batch = (
                torch.arange(
                    valid_len,
                    valid_len + n_decode_tokens,
                    device=device,
                    dtype=info.dtype,
                )
                .view(1, -1)
                .expand(export_batch, -1)
            )
            for base in ("time_position_ids", "hight_position_ids", "width_position_ids"):
                batch_idx = _batch_suffix(name, base)
                if batch_idx is not None:
                    decode_input_feed[name] = decode_pos_batch[batch_idx : batch_idx + 1]
                    break
            else:
                decode_input_feed[name] = decode_pos_batch.reshape(info.shape)
        elif name.startswith("past_conv_cache_"):
            suffix = name[len("past_conv_cache_") :]
            out_keys = [f"conv_cache_out_{suffix}"]
            if "_batch_" in suffix:
                base, batch = suffix.rsplit("_batch_", 1)
                out_keys.append(f"conv_cache_out_{base}_0_batch_{batch}")
            else:
                out_keys.append(f"conv_cache_out_{suffix}_0")
            out_key = next((key for key in out_keys if key in prefill_output_map), None)
            if out_key is not None:
                decode_input_feed[name] = _ensure_cache_tensor(prefill_output_map[out_key])
            else:
                info = decode_session.get_input(name)
                decode_input_feed[name] = _ensure_cache_tensor(torch.zeros(info.shape, dtype=info.dtype, device=device))
        elif name.startswith("past_recurrent_state_"):
            suffix = name[len("past_recurrent_state_") :]
            out_keys = [f"recurrent_state_out_{suffix}"]
            if "_batch_" in suffix:
                base, batch = suffix.rsplit("_batch_", 1)
                out_keys.append(f"recurrent_state_out_{base}_0_batch_{batch}")
            else:
                out_keys.append(f"recurrent_state_out_{suffix}_0")
            out_key = next((key for key in out_keys if key in prefill_output_map), None)
            if out_key is not None:
                decode_input_feed[name] = _ensure_cache_tensor(prefill_output_map[out_key])
            else:
                info = decode_session.get_input(name)
                decode_input_feed[name] = _ensure_cache_tensor(torch.zeros(info.shape, dtype=info.dtype, device=device))
        else:
            if name in prefill_cache_inputs:
                decode_input_feed[name] = prefill_cache_inputs[name]
            else:
                info = decode_session.get_input(name)
                t = torch.zeros(info.shape, dtype=info.dtype, device=device)
                if _is_cache_input_name(name):
                    t = _ensure_cache_tensor(t)
                decode_input_feed[name] = t

    _, decode_output_map = _run_hmonnx_with_golden(decode_session, decode_input_feed)
    decode_logits = _extract_logits_from_output_map(decode_output_map) if isinstance(decode_output_map, dict) else None
    if decode_logits is not None:
        import numpy as np

        np.save(str(decode_dir / "logits.npy"), decode_logits.detach().cpu().numpy())

    del decode_session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if prefill_onnx_path.exists():
        _save_onnx_with_renamed_external_data(
            prefill_onnx_path,
            named_prefill_onnx,
            f"{release_prefix}_external_data",
            logger,
        )
    if decode_onnx_path.exists():
        _save_onnx_with_renamed_external_data(
            decode_onnx_path,
            named_decode_onnx,
            f"{release_prefix}_external_data",
            logger,
        )

    _cleanup_hmonnx_in_golden(prefill_dir, release_prefix, "prefill", logger)
    _cleanup_hmonnx_in_golden(decode_dir, release_prefix, "decode", logger)
    _create_step0_onnx_symlinks(prefill_dir, release_prefix, logger)
    _create_step0_onnx_symlinks(decode_dir, release_prefix, logger)

    draft_golden_paths: Dict[str, Path] = {}
    if draft_onnx_files and spec_decode_mode in ("mtp", "dflash"):
        if spec_decode_mode == "mtp":
            draft_items: List[Tuple[str, str, bool]] = [
                ("draft_prefill_onnx", "draft_prefill", False),
                ("draft_decode_onnx", "draft_decode", True),
            ]
        else:
            draft_items = [
                ("draft_context_onnx", "draft_context", False),
                ("draft_context_decode_onnx", "draft_context_decode", False),
                ("draft_decode_onnx", "draft_decode", True),
            ]

        for onnx_key, dir_name, is_decode_step in draft_items:
            onnx_path_str = draft_onnx_files.get(onnx_key)
            if not onnx_path_str:
                logger.warning(f"[draft golden] '{onnx_key}' not found in draft_onnx_files, skipping.")
                continue
            onnx_path = Path(onnx_path_str)
            if not onnx_path.exists():
                logger.warning(f"[draft golden] ONNX not on disk: {onnx_path}, skipping.")
                continue

            draft_golden_dir = release_dir / f"{spec_decode_mode}_{dir_name}"
            draft_golden_dir.mkdir(exist_ok=True, parents=True)
            named_draft_onnx = draft_golden_dir / f"{release_prefix}_with_act.onnx"
            _save_onnx_with_renamed_external_data(
                onnx_path,
                named_draft_onnx,
                f"{release_prefix}_external_data",
                logger,
            )

            logger.info(f"Generating draft golden [{spec_decode_mode}/{dir_name}] from {onnx_path.name} ...")
            _generate_draft_golden_for_onnx(
                work_dir=work_dir,
                args=args,
                onnx_file=str(onnx_path),
                golden_dir=draft_golden_dir,
                token_embedding=token_embedding,
                pad_token_id=pad_token_id,
                input_ids_full=input_ids_full,
                is_decode=is_decode_step,
                device=device,
                logger=logger,
            )
            role = dir_name.split("_", 1)[1] if dir_name.startswith("draft_") else dir_name
            _cleanup_hmonnx_in_golden(draft_golden_dir, release_prefix, role, logger)
            _create_step0_onnx_symlinks(draft_golden_dir, release_prefix, logger)
            draft_golden_paths[f"{spec_decode_mode}_{dir_name}"] = draft_golden_dir
            logger.info(f"Draft golden [{spec_decode_mode}/{dir_name}] saved: {draft_golden_dir}")

    golden_meta = {
        "release_prefix": release_prefix,
        "zip_name": f"{release_prefix}.zip",
        "zip_cmd": f"zip -r -y {release_prefix}.zip {release_prefix}/",
        "prefill_onnx": str(named_prefill_onnx.relative_to(release_dir)) if named_prefill_onnx.exists() else None,
        "decode_onnx": str(named_decode_onnx.relative_to(release_dir)) if named_decode_onnx.exists() else None,
        "hf_config": "hf_config",
        "token_embedding_file": "quant_embedding.pt",
        "quant_embedding": "quant_embedding.pt",
        "pad_token_id": pad_token_id,
        "spec_decode_mode": spec_decode_mode,
    }
    for key in (
        "max_context_tokens",
        "wrap_cfg",
        "kv_cache",
        "linear_cache",
        "model_config",
    ):
        if key in work_meta:
            golden_meta[key] = work_meta[key]
    for key, dir_path in draft_golden_paths.items():
        golden_meta[f"{key}_golden_dir"] = key
        named_onnx = dir_path / f"{release_prefix}_with_act.onnx"
        golden_meta[f"{key}_onnx"] = str(named_onnx.relative_to(release_dir)) if named_onnx.exists() else None
    with (release_dir / "golden_meta_info.json").open("w", encoding="utf-8") as fout:
        json.dump(golden_meta, fout, ensure_ascii=False, indent=2)

    _copy_release_root_assets(args, release_dir, release_prefix, work_dir, logger)

    logger.info(f"Golden generation done. Release dir: {release_dir}")
    logger.info(f"To package: {golden_meta['zip_cmd']}")

    if getattr(args, "package_release", False):
        _package_release_dir(release_dir, logger)
    return release_dir


def _package_release_dir(release_dir: Path, logger) -> Optional[Path]:
    if not release_dir.exists():
        return None
    zip_file = release_dir.parent / f"{release_dir.name}.zip"
    if zip_file.exists():
        zip_file.unlink()
    logger.info(f"Packing release: {release_dir} -> {zip_file}")
    subprocess.run(
        ["zip", "-r", "-y", "-0", "-q", zip_file.name, release_dir.name],
        cwd=str(release_dir.parent),
        check=True,
    )
    logger.info(f"Package done: {zip_file}")
    return zip_file


def _resolve_onnx_file(work_dir: Path, subdir: str, logger) -> Optional[str]:
    onnx_dir = work_dir / "hmonnx" / subdir
    if not onnx_dir.exists():
        onnx_dir = work_dir / f"{subdir}_onnx"
    if not onnx_dir.exists():
        logger.warning(f"ONNX directory not found: {onnx_dir}")
        return None
    onnx_files = list(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        logger.warning(f"No .onnx files in {onnx_dir}")
        return None
    return str(onnx_files[0])


def _load_work_meta(work_dir: Path, logger) -> Dict:
    meta_path = work_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as fin:
            return json.load(fin)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to load work_dir meta.json: {exc}")
        return {}


def _resolve_draft_onnx_files(
    work_dir: Path, spec_decode_mode: Optional[str], meta_info: Dict, logger
) -> Optional[Dict[str, str]]:
    if spec_decode_mode not in ("mtp", "dflash"):
        return None

    if spec_decode_mode == "mtp":
        draft_items = [
            ("draft_prefill_onnx_file", "draft_prefill_onnx", "*mtp_prefill.onnx"),
            ("draft_decode_onnx_file", "draft_decode_onnx", "*mtp_decode.onnx"),
        ]
    else:
        draft_items = [
            ("draft_context_onnx_file", "draft_context_onnx", "*dflash_context.onnx"),
            ("draft_context_decode_onnx_file", "draft_context_decode_onnx", "*dflash_context_decode.onnx"),
            ("draft_decode_onnx_file", "draft_decode_onnx", "*dflash_decode.onnx"),
        ]

    draft_onnx_dir = work_dir / "draft_onnx"
    draft_onnx_files: Dict[str, str] = {}
    for meta_key, onnx_key, pattern in draft_items:
        candidates = []
        rel_path = meta_info.get(meta_key)
        if rel_path:
            candidates.append(work_dir / rel_path)
        candidates.extend(sorted(draft_onnx_dir.glob(pattern)))

        for candidate in candidates:
            if candidate.exists():
                draft_onnx_files[onnx_key] = str(candidate.resolve())
                logger.info(f"Found draft ONNX [{onnx_key}]: {candidate}")
                break
        else:
            logger.warning(f"Draft ONNX not found for {onnx_key} (meta key={meta_key}, glob={pattern})")

    return draft_onnx_files or None


def _run_golden(work_dir: Path, args, logger) -> Optional[Path]:
    from transformers import AutoTokenizer

    meta_info = _load_work_meta(work_dir, logger)
    spec_decode_mode = getattr(args, "spec_decode_mode", None) or meta_info.get("spec_decode_mode")
    if spec_decode_mode == "none":
        spec_decode_mode = None

    hf_model_path = osp.normpath(osp.abspath(args.model))
    meta_hf_model_path = meta_info.get("hf_model_path")
    if meta_hf_model_path and getattr(args, "existing_work_dir", None) and Path(args.model).name == "Qwen3.5-35B-A3B":
        candidate_model_path = osp.normpath(osp.abspath(meta_hf_model_path))
        if Path(candidate_model_path).exists():
            hf_model_path = candidate_model_path
            args.model = candidate_model_path
            logger.info(f"Golden-only: using hf_model_path from meta.json: {hf_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    prompt = "你好，请介绍一下你自己。"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    prefill_onnx = _resolve_onnx_file(work_dir, "prefill", logger)
    decode_onnx = _resolve_onnx_file(work_dir, "decode", logger)
    if prefill_onnx is None or decode_onnx is None:
        logger.error("Cannot find prefill/decode ONNX files for golden generation.")
        return None

    draft_onnx_files = _resolve_draft_onnx_files(work_dir, spec_decode_mode, meta_info, logger)

    logger.info(f"Prefill ONNX: {prefill_onnx}")
    logger.info(f"Decode ONNX: {decode_onnx}")
    if draft_onnx_files:
        logger.info(f"Draft ONNX files: {draft_onnx_files}")

    return _generate_golden(
        work_dir,
        args,
        input_ids,
        tokenizer,
        prefill_onnx,
        decode_onnx,
        logger,
        draft_onnx_files=draft_onnx_files,
        spec_decode_mode=spec_decode_mode,
    )


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)

    _validate_offline_rope_required_for_fp16_limit(args)

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        prefix = f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}"
        if args.quant_weight:
            prefix += "-gptq"
        work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()

    s_scalar_path = _resolve_s_scalar_path(args, Path(hf_model_path), work_dir, logger)

    config = Qwen3_5MoePruneConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        max_layers=None,
        max_pe_length=args.max_pe_length,
        support_long_context_over_fp16_limit=getattr(args, "support_long_context_over_fp16_limit", True),
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=args.num_logits_to_keep,
        linear_attention_mode=args.linear_attention_mode,
        linear_chunk_size=args.linear_chunk_size,
        split_conv_cache=args.split_conv_cache,
        normalize_force_fp32=getattr(args, "normalize_force_fp32", False),
        use_manual_depthwise_conv1d=getattr(args, "use_manual_depthwise_conv1d", False),
        fuse_gdr_ops=getattr(args, "fuse_gdr_ops", False),
        mix_search=args.mix_search,
        threshold=args.threshold,
        s_scalar_path=s_scalar_path,
    )

    logger.info(f"model: {hf_model_path}")
    logger.info(f"quant_weight: {args.quant_weight}")
    logger.info(f"s_scalar_path: {s_scalar_path}")
    logger.info(f"output: {work_dir}")

    architecture = args.architecture

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

    logger.info(f"Done. Artifacts in: {work_dir}")

    if getattr(args, "golden", False):
        logger.info("=" * 60)
        logger.info("Generating HMONNX golden (prefill + decode)")
        logger.info("=" * 60)
        _run_golden(work_dir, args, logger)


def main_golden_only(args):
    _validate_offline_rope_required_for_fp16_limit(args)

    if getattr(args, "existing_work_dir", None):
        work_dir = Path(args.existing_work_dir)
    elif args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        raise ValueError("--golden-only requires --work-dir or --existing-work-dir")

    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    token_embedding_file = _resolve_token_embedding_path(work_dir)
    if token_embedding_file is None:
        raise FileNotFoundError(
            f"Neither quant_embedding.pt nor token_embedding.pt found under work_dir: {work_dir}. "
            "Please use a completed export directory."
        )

    log_file = work_dir / "golden.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    logger.info(f"Golden-only mode. work_dir: {work_dir}")

    _run_golden(work_dir, args, logger)
    logger.info("Golden-only done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pruned Qwen3.5-MoE to prefill/decode HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3.6-35B-A3B")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export model inputs")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument(
        "--max-pe-length",
        "--max_pe_length",
        dest="max_pe_length",
        type=int,
        default=262144,
        help="RoPE cache length; 256K is required for long context over fp16 position limit",
    )
    parser.set_defaults(support_long_context_over_fp16_limit=True)
    long_context_group = parser.add_mutually_exclusive_group()
    long_context_group.add_argument(
        "--support_long_context_over_fp16_limit",
        dest="support_long_context_over_fp16_limit",
        action="store_true",
        help=(
            "Use precomputed rotary cache (offline RoPE) so exported graphs can "
            "support position_id > 65504. Enabled by default; offline RoPE is required for 256K context."
        ),
    )
    long_context_group.add_argument(
        "--no-support_long_context_over_fp16_limit",
        "--no-support-long-context-over-fp16-limit",
        dest="support_long_context_over_fp16_limit",
        action="store_false",
        help="Disable offline RoPE cache support; invalid for 256K long-context exports.",
    )
    parser.add_argument("--quant-type", default="w4a8h0_ssfp")
    parser.add_argument("--num_logits_to_keep", type=int, default=1)
    parser.add_argument("--quant-weight", type=str, default=None)
    parser.add_argument("--mix_search", type=str, default=None)
    parser.add_argument("--linear-attention-mode", type=str, default="auto")
    parser.add_argument("--linear-chunk-size", type=int, default=64)
    parser.set_defaults(split_conv_cache=True)
    parser.add_argument("--split-conv-cache", dest="split_conv_cache", action="store_true")
    parser.add_argument("--no-split-conv-cache", dest="split_conv_cache", action="store_false")
    parser.add_argument("--normalize-force-fp32", dest="normalize_force_fp32", action="store_true", default=False)
    parser.add_argument(
        "--use-manual-depthwise-conv1d",
        dest="use_manual_depthwise_conv1d",
        action="store_true",
        default=False,
    )
    parser.add_argument("--fuse-gdr-ops", dest="fuse_gdr_ops", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--s-scalar-path", type=str, default=None)
    parser.add_argument("--auto-s-scalar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--s-scalar-output", type=str, default=None)
    parser.add_argument("--s-scalar-compute-device", type=str, default="auto")
    parser.add_argument("--s-scalar-eps", type=float, default=1e-8)
    parser.add_argument("--overwrite-s-scalar", action="store_true")
    parser.add_argument(
        "--architecture",
        type=str,
        default="Qwen3_5MoeForConditionalGeneration_prune",
    )
    # ── Golden validation ──
    parser.add_argument("--golden", action="store_true", help="Generate HMONNX golden after export")
    parser.add_argument(
        "--golden-only",
        "--golden_only",
        dest="golden_only",
        action="store_true",
        help="Skip export and only generate golden from existing work_dir ONNX files",
    )
    parser.add_argument(
        "--existing-work-dir",
        "--existing_work_dir",
        dest="existing_work_dir",
        type=str,
        default=None,
        help="Path to existing work_dir for --golden-only.",
    )
    # ── HM 模型版本发布命名规则 ──
    parser.add_argument(
        "--release-xh-version",
        "--release_xh_version",
        dest="release_xh_version",
        type=str,
        default=None,
        choices=["xh1", "xh2"],
        help="Release xh_version. Must be 'xh1' or 'xh2' (default: xh2).",
    )
    parser.add_argument(
        "--release-modelscope-name",
        "--release_modelscope_name",
        dest="release_modelscope_name",
        type=str,
        default=None,
        help="Release modelscope_name. Defaults to lowercased basename of --model.",
    )
    parser.add_argument(
        "--release-wmix-amix",
        "--release_wmix_amix",
        dest="release_wmix_amix",
        type=str,
        default=None,
        help=(
            "Release wmix_amix field. Pure 'w<bits>a<bits>' is preserved; anything else is normalised to 'wmix_amix'."
        ),
    )
    parser.add_argument(
        "--release-date",
        "--release_date",
        dest="release_date",
        type=str,
        default=None,
        help="Release date string (YYYYMMDD). Defaults to today.",
    )
    parser.add_argument(
        "--package-release",
        "--package_release",
        dest="package_release",
        action="store_true",
        help="Zip the release directory after golden generation.",
    )
    args = parser.parse_args()
    if getattr(args, "golden_only", False):
        main_golden_only(args)
    else:
        main(args)
