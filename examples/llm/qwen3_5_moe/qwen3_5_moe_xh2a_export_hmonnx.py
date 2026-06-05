# Copyright 2025 HOUMO AI
#
# File: qwen3_5_moe_xh2a_export_hmonnx.py
# Description:
#   Export script: Qwen3.5-MoE LLM -> prefill/decode HMONNX via LLMConverter.
#
# Usage (float weights):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
#       --model /data01/nfs_shared/Qwen3.5-35B-A3B \
#       --context-length 2048 --input-sequence-length 256 \
#       --quant-type w8a8h0_ssfp
#
# Usage (GPTQModel weights):
#   python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
#       --model /data01/nfs_shared/Qwen3.5-35B-A3B \
#       --quant-weight /data01/home/huxing/gptqmodel/work_dirs/Qwen35_35B_A3B_attn4_e4_se4_0324 \
#       --quant-type w4a8h0_ssfp \
#       --context-length 2048 --input-sequence-length 256
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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

import torch
import torch.nn as nn

from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.qwen3_5_moe import Qwen3_5MoeConvertConfig
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter import Qwen3_5MoeConverterXH2a


from xhquant.api import DeviceType, QuantScheme, get_root_logger, xhquant_init  # isort:skip
from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

# The large continue-batch graphs generate thousands of benign Slice-collapse
# info messages from onnxscript.  Keep export logs focused on actionable stages
# and tracebacks.
logging.getLogger("onnxscript.rewriter.rules.common._collapse_slices").setLevel(logging.WARNING)
logging.getLogger("onnxscript.optimizer._constant_folding").setLevel(logging.WARNING)
logging.getLogger("onnx_ir.passes.common.initializer_deduplication").setLevel(logging.WARNING)


def _get_default_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# HM 模型版本发布命名规则 helpers (verbatim copy from dense script —
# kept self-contained per project rule "复制粘贴更安全，不要去改 dense 的导入结构")
# ─────────────────────────────────────────────────────────────────────────────

_PURE_FIXED_POINT_WMIX_RE = re.compile(r"^w\d+a\d+$")


def _normalize_wmix_amix(value: str) -> str:
    """Per HM 命名规则: pure ``w<bits>a<bits>`` is preserved; everything else
    (sub-mode tags like ``h0_ssfp``, underscores like ``w4_a8``, mixed) → ``wmix_amix``."""
    if value is None:
        return "wmix_amix"
    s = str(value).strip().lower()
    if not s:
        return "wmix_amix"
    if _PURE_FIXED_POINT_WMIX_RE.match(s):
        return s
    return "wmix_amix"


def _detect_release_wmix_amix(hf_model_dir: str) -> str:
    """Always return the conservative ``wmix_amix`` for MoE.

    MoE quant schemes (e.g. ``w8a8h0_sefp``, ``w4a8h0_ssfp``) carry sub-mode
    tags that are not legal under the release naming rule, so we normalise
    to ``wmix_amix`` and let the user override via ``--release-wmix-amix``.
    """
    return "wmix_amix"


def _resolve_token_embedding_path(work_dir: Path) -> Optional[Path]:
    """Look up embedding file. Prefer ``quant_embedding.pt``, fall back to ``token_embedding.pt``."""
    for name in ("quant_embedding.pt", "token_embedding.pt"):
        candidate = work_dir / name
        if candidate.exists():
            return candidate
    return None


def _build_release_prefix_moe(args, hf_model_path: str) -> str:
    """Build the release prefix for MoE per HM 模型版本发布命名规则.

    Layout: ``hmquant_<xh_version>_<modelscope_name>_<wmix_amix>_<prefill>_<context>_<date>``
    All-lowercase. ``xh_version`` must be one of ``xh1``/``xh2``.
    """
    xh_version = (getattr(args, "release_xh_version", None) or "xh2").strip().lower()
    if xh_version not in {"xh1", "xh2"}:
        raise ValueError(
            f"release_xh_version must be one of 'xh1' / 'xh2', got: {xh_version!r}"
        )

    modelscope_name = getattr(args, "release_modelscope_name", None)
    if not modelscope_name:
        modelscope_name = Path(hf_model_path).name
    modelscope_name = (
        str(modelscope_name)
        .strip()
        .lower()
        .replace(".", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    wmix_amix_raw = getattr(args, "release_wmix_amix", None) or _detect_release_wmix_amix(hf_model_path)
    wmix_amix = _normalize_wmix_amix(wmix_amix_raw)

    prefill_len = int(args.input_sequence_length)
    ctx_len = int(args.context_length)
    if ctx_len % 1024 == 0:
        ctx_str = f"{ctx_len // 1024}k"
    else:
        ctx_str = str(ctx_len)

    date_str = getattr(args, "release_date", None)
    if not date_str:
        date_str = time.strftime("%Y%m%d")
    date_str = str(date_str).strip().lower()

    return f"hmquant_{xh_version}_{modelscope_name}_{wmix_amix}_{prefill_len}_{ctx_str}_{date_str}"


def _save_onnx_with_renamed_external_data(
    src_onnx: Path, dst_onnx: Path, new_external_data_name: str, logger
) -> None:
    """Resave an ONNX file under a new external-data filename so the protobuf
    reference inside the model matches the renamed sidecar.

    Drops any stale destination external_data file first to avoid mismatches.
    """
    import onnx

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
    logger.info(f"Resaving ONNX with renamed external_data: {src_onnx.name} -> {dst_onnx.name} (ext={new_external_data_name})")
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
    """Rename golden .npy files and with_act/ dir inside step_0 to short format.

    Fu Shengguo naming rule:
      hmquant_{model_name}_{io_name}_{direction}.npy
      e.g. hmquant_qwen3_5_9b_conv_cache_out_0_output.npy

    HMONNX golden files use tensor names (e.g. xh2a_prefill_conv_cache_out_0_output)
    as the filename prefix, NOT the release_prefix. The tensor name in the golden file
    embeds the mode/role (e.g. xh2a_prefill), which must be stripped.
    """
    if role is None:
        return
    step0_dir = golden_dir / "step_0"
    if not step0_dir.is_dir():
        return

    # Extract model_name from release_prefix = "hmquant_{backend}_{model_name}_{rest}"
    prefix_stripped = release_prefix.split("_", 2)[-1]
    segments = prefix_stripped.split("_")
    scheme_keywords = {"xh1", "xh2", "wmix", "amix", "w4a8", "w8a8", "w4", "w8", "a4", "a8",
                       "256", "2k", "4k", "8k", "16k", "32k", "h0", "h1",
                       "ssfp", "sefp", "fp16", "fp32", "gptq"}
    model_parts = []
    for seg in segments:
        if seg.lower() in scheme_keywords:
            break
        model_parts.append(seg)
    model_name = "_".join(model_parts)
    short_prefix = f"hmquant_{model_name}"

    # Role tokens that appear in golden file tensor names between model and io_name.
    role_tokens = [
        "prefill", "decode",
        "draft_prefill", "draft_decode",
        "draft_context", "draft_context_decode",
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
        after_model = name_without_ext[len(model_prefix):suffix_idx]
        rest = after_model.lstrip("_")
        io_name = rest
        for rt in role_tokens:
            if rest.startswith(rt + "_"):
                io_name = rest[len(rt) + 1:]
                break
            for mode_prefix in ("xh2a_", "xh2_", "xh1a_", "xh1_"):
                combined = mode_prefix + rt + "_"
                if rest.startswith(combined):
                    io_name = rest[len(combined):]
                    break
                if rt in ("prefill", "decode") and rest.startswith(rt + "_"):
                    io_name = rest[len(rt) + 1:]
                    break
            else:
                continue
            break

        new_name = f"{model_prefix}_{io_name}{direction}.npy"
        new_path = step0_dir / new_name
        logger.info(f"Rename golden file: {fpath.name} → {new_name}")
        shutil.move(str(fpath), str(new_path))


def _create_step0_onnx_symlinks(golden_dir: Path, release_prefix: str, logger) -> None:
    """Inside ``step_0/`` create *relative* symlinks back to the named ONNX
    and its external_data sidecar at ``golden_dir`` root.

    Fu Shengguo's naming rule: step_0 symlink name uses only the model-name portion
    (e.g. hmquant_qwen3_5_9b_with_act.onnx), while the actual outer ONNX file
    at golden_dir root keeps the full release_prefix.
    """
    step0_dir = golden_dir / "step_0"
    step0_dir.mkdir(parents=True, exist_ok=True)

    # Extract model_name from release_prefix = "hmquant_{backend}_{model_name}_{rest}"
    # e.g. "hmquant_xh2_qwen3_5_9b_wmix_amix_256_2k_20260526" -> "qwen3_5_9b"
    prefix_stripped = release_prefix.split("_", 2)[-1]
    segments = prefix_stripped.split("_")
    scheme_keywords = {"xh1", "xh2", "wmix", "amix", "w4a8", "w8a8", "w4", "w8", "a4", "a8",
                       "256", "2k", "4k", "8k", "16k", "32k", "h0", "h1",
                       "ssfp", "sefp", "fp16", "fp32", "gptq"}
    model_parts = []
    for seg in segments:
        if seg.lower() in scheme_keywords:
            break
        model_parts.append(seg)
    model_name = "_".join(model_parts)

    # ONNX short name (Fu Shengguo rule); external_data keeps full name to avoid runtime risk
    onnx_short = f"hmquant_{model_name}_with_act.onnx"
    onnx_actual = f"{release_prefix}_with_act.onnx"
    # external_data symlink: keep full release_prefix (do NOT shorten)
    ext_actual = f"{release_prefix}_external_data"

    # ONNX symlink: short name -> actual
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

    # external_data symlink: full name -> full name (no shortening)
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
    """Drop the export script and convert log into release_dir root per HM 命名规则."""
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
    with torch.no_grad():
        embeds = token_embedding(input_ids.to(device))
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
        return int(name[len(prefix):])
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
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.initialize()
    return session


def _run_hmonnx_with_golden(session, input_feed: Dict[str, torch.Tensor]):
    outputs = session.run(input_feed)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    output_names = session.get_output_names()
    output_map = {name: out for name, out in zip(output_names, outputs)}
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


def _cleanup_hmonnx_in_golden(
    golden_dir: Path, release_prefix: str, role: Optional[str], logger
) -> None:
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
    token_embedding = _load_token_embedding(token_embedding_file).to(device).to(dtype)
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
            f"Cannot build golden inputs for export_batch={export_batch} "
            f"from prompt batch={input_ids_full.shape[0]}"
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
    # Fu Shengguo naming rule: ONNX file is hmquant_{prefix}_with_act.onnx (no role suffix)
    named_prefill_onnx = prefill_dir / f"{release_prefix}_with_act.onnx"
    named_decode_onnx = decode_dir / f"{release_prefix}_with_act.onnx"

    quant_embedding_file = release_dir / "quant_embedding.pt"
    if token_embedding_file.exists() and not quant_embedding_file.exists():
        shutil.copy2(token_embedding_file, quant_embedding_file)
        logger.info(f"Copied quant_embedding.pt to {quant_embedding_file}")

    # HM 模型版本发布命名规则: bundle hf_config/ (config.json, tokenizer*, etc.) into release_dir
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
        if _feed_batched_tensor(prefill_input_feed, name, "inputs_embeds", prefill_inputs_embeds):
            pass
        elif _feed_batched_tensor(prefill_input_feed, name, "past_seq_length", prefill_past_seq_length):
            pass
        elif _feed_batched_tensor(prefill_input_feed, name, "current_input_length", prefill_current_input_length):
            pass
        elif _feed_batched_tensor(prefill_input_feed, name, "linear_attn_mask", prefill_linear_attn_mask):
            pass
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids") or any(
            _batch_suffix(name, base) is not None for base in ("time_position_ids", "hight_position_ids", "width_position_ids")
        ):
            info = prefill_session.get_input(name)
            pos_batch = torch.arange(0, prefill_seq_len, device=device, dtype=info.dtype).view(1, -1).expand(export_batch, -1)
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
        if _feed_batched_tensor(decode_input_feed, name, "inputs_embeds", decode_inputs_embeds):
            pass
        elif _feed_batched_tensor(decode_input_feed, name, "past_seq_length", decode_past_seq_length):
            pass
        elif _feed_batched_tensor(decode_input_feed, name, "current_input_length", decode_current_input_length):
            pass
        elif _feed_batched_tensor(decode_input_feed, name, "linear_attn_mask", decode_linear_attn_mask):
            pass
        elif name in ("time_position_ids", "hight_position_ids", "width_position_ids") or any(
            _batch_suffix(name, base) is not None for base in ("time_position_ids", "hight_position_ids", "width_position_ids")
        ):
            info = decode_session.get_input(name)
            n_decode_tokens = decode_inputs_info.shape[1]
            decode_pos_batch = torch.arange(valid_len, valid_len + n_decode_tokens, device=device, dtype=info.dtype).view(1, -1).expand(export_batch, -1)
            for base in ("time_position_ids", "hight_position_ids", "width_position_ids"):
                batch_idx = _batch_suffix(name, base)
                if batch_idx is not None:
                    decode_input_feed[name] = decode_pos_batch[batch_idx : batch_idx + 1]
                    break
            else:
                decode_input_feed[name] = decode_pos_batch.reshape(info.shape)
        elif name.startswith("past_conv_cache_"):
            suffix = name[len("past_conv_cache_"):]
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
            suffix = name[len("past_recurrent_state_"):]
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

    # Li Wanyu: flat draft release dirs, e.g. mtp_draft_prefill/ and mtp_draft_decode/.
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

    # ── HM 模型版本发布命名规则: drop hmonnx.py + debug logs at release_dir root ──
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


def _build_draft_only_default_work_dir(
    existing_work_dir: Path, spec_decode_mode: str, draft_head_weight_bits: int
) -> Path:
    existing_work_dir = existing_work_dir.resolve()
    suffix = f"draft_{spec_decode_mode}_w{draft_head_weight_bits}"
    base = existing_work_dir.with_name(f"{existing_work_dir.name}-{suffix}")
    if not base.exists() and base.resolve() != existing_work_dir:
        return base
    return existing_work_dir.with_name(f"{existing_work_dir.name}-{suffix}-{time.strftime('%Y%m%d%H%M%S')}")


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)

    spec_decode_mode = args.spec_decode_mode or None
    if spec_decode_mode == "none":
        spec_decode_mode = None
        args.spec_decode_mode = None
    num_draft_tokens = args.num_draft_tokens

    config = Qwen3_5MoeConvertConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=quant_scheme,
        quant_weight=args.quant_weight,
        num_logits_to_keep=args.num_logits_to_keep,
        linear_attention_mode=args.linear_attention_mode,
        linear_chunk_size=args.linear_chunk_size,
        spec_decode_mode=spec_decode_mode,
        num_draft_tokens=num_draft_tokens,
        dflash_model_dir=args.dflash_model_dir,
        spec_draft_head_weight_bits=args.spec_draft_head_weight_bits,
        split_conv_cache=args.split_conv_cache,
        normalize_force_fp32=getattr(args, "normalize_force_fp32", False),
        use_manual_depthwise_conv1d=getattr(args, "use_manual_depthwise_conv1d", False),
        fuse_gdr_ops=getattr(args, "fuse_gdr_ops", False),
    )

    if args.draft_only:
        if args.existing_work_dir is None:
            raise ValueError("--draft-only requires --existing-work-dir")
        if spec_decode_mode not in {"mtp", "dflash"}:
            raise ValueError("--draft-only requires --spec-decode-mode to be one of {'mtp', 'dflash'}")
        if args.work_dir:
            work_dir = Path(args.work_dir)
        else:
            work_dir = _build_draft_only_default_work_dir(
                Path(args.existing_work_dir),
                spec_decode_mode,
                int(args.spec_draft_head_weight_bits),
            )
        if work_dir.resolve() == Path(args.existing_work_dir).resolve():
            raise ValueError("--draft-only --work-dir must not overwrite --existing-work-dir")
    elif args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        prefix = f"{model_name}-{target_device}-{args.context_length // 1024}k-{quant_type}"
        if args.quant_weight:
            prefix += "-gptq"
        if spec_decode_mode:
            prefix += f"-spec_{spec_decode_mode}"
        work_dir = Path("work_dirs") / prefix
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug)
    logger = get_root_logger()
    logger.info(f"model: {hf_model_path}")
    logger.info(f"quant_weight: {args.quant_weight}")
    logger.info(f"spec_decode_mode: {spec_decode_mode}")
    logger.info(f"spec_draft_head_weight_bits: {args.spec_draft_head_weight_bits}")
    logger.info(f"output: {work_dir}")

    if args.draft_only:
        with TimeProfiler("draft-only convert", logger), MemoryTracker("cuda:0", "draft-only convert", logger):
            Qwen3_5MoeConverterXH2a(config).export_draft_only(
                hf_model_path,
                args.existing_work_dir,
                str(work_dir),
            )
        logger.info(f"Done. Draft-only artifacts in: {work_dir}")
        return

    # Detect architecture from config.json automatically
    architecture = args.architecture  # may be None → auto-detect

    with TimeProfiler("convert", logger), MemoryTracker("cuda:0", "convert", logger):
        LLMConverter.from_pretrained(hf_model_path, architecture, config, str(work_dir))

    logger.info(f"Done. Artifacts in: {work_dir}")

    if getattr(args, "golden", False):
        logger.info("=" * 60)
        logger.info("Generating HMONNX golden (prefill + decode)")
        logger.info("=" * 60)
        _run_golden(work_dir, args, logger)


def main_golden_only(args):
    if args.existing_work_dir:
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
        description="Export Qwen3.5-MoE to prefill/decode HMONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="weights/Qwen3.5-35B-A3B", help="HuggingFace model directory"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Architecture string (auto-detected if None). "
        "Use 'Qwen3_5MoeForConditionalGeneration' or 'Qwen3_5MoeForCausalLM'",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export model inputs")
    parser.add_argument("--context-length", type=int, default=2048, help="Maximum context length (kv cache size)")
    parser.add_argument("--input-sequence-length", type=int, default=256, help="Prefill chunk size")
    parser.add_argument("--quant-type", type=str, default="w8a8h0_sefp", help="Quantisation type string")
    parser.add_argument("--quant-weight", type=str, default=None, help="Path to GPTQModel quantised weights (optional)")
    parser.add_argument(
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=str,
        default=None,
        help="Output work directory. Defaults to the standard export prefix, or a non-overwriting draft-only sibling.",
    )
    parser.add_argument(
        "--num-logits-to-keep", type=int, default=1, help="How many final logit positions to keep (1 = last only)"
    )
    parser.add_argument(
        "--linear-attention-mode",
        type=str,
        default="auto",
        choices=["auto", "chunk", "recurrent"],
        help="Linear attention computation mode",
    )
    parser.add_argument("--linear-chunk-size", type=int, default=64, help="Chunk size for linear attention")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--draft-only",
        "--draft_only",
        dest="draft_only",
        action="store_true",
        help="Reuse target artifacts from --existing-work-dir and export only MTP/DFlash draft ONNX/meta.",
    )
    parser.add_argument(
        "--existing-work-dir",
        "--existing_work_dir",
        dest="existing_work_dir",
        type=str,
        default=None,
        help="Existing target export work_dir containing meta.json for --draft-only.",
    )
    # Speculative decoding
    parser.add_argument(
        "--spec-decode-mode",
        "--spec_decode_mode",
        dest="spec_decode_mode",
        type=str,
        default=None,
        choices=["none", "mtp", "dflash"],
        help="Speculative decoding mode.  'mtp' exports MTP draft graphs; "
        "'dflash' exports DFlash context/decode draft graphs.",
    )
    parser.add_argument(
        "--dflash-model-dir",
        "--dflash_model_dir",
        dest="dflash_model_dir",
        type=str,
        default=None,
        help="Path to DFlash draft model dir (required for --spec-decode-mode dflash)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        "--num_draft_tokens",
        dest="num_draft_tokens",
        type=int,
        default=4,
        help=(
            "Number of draft tokens per spec-decode round (verify_length = N + 1). "
            "For DFlash the draft decode input length is also verify_length."
        ),
    )
    parser.add_argument(
        "--spec-draft-head-weight-bits",
        "--spec_draft_head_weight_bits",
        dest="spec_draft_head_weight_bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Weight bits for MTP/DFlash draft lm_head. Default uses w4 head; set 8 to keep previous w8 head.",
    )
    parser.set_defaults(split_conv_cache=True)
    parser.add_argument(
        "--split-conv-cache",
        "--split_conv_cache",
        dest="split_conv_cache",
        action="store_true",
        help=(
            "Split linear attention conv_cache into 3 separate tensors (q, k, v). "
            "This is the default export format."
        ),
    )
    parser.add_argument(
        "--no-split-conv-cache",
        "--no_split_conv_cache",
        dest="split_conv_cache",
        action="store_false",
        help="Use the legacy merged single-tensor conv_cache format.",
    )
    parser.add_argument(
        "--normalize-force-fp32",
        "--normalize_force_fp32",
        dest="normalize_force_fp32",
        action="store_true",
        default=False,
        help="Force fp32 accumulation in Normalize operator.",
    )
    parser.add_argument(
        "--use_manual_depthwise_conv1d",
        "--use-manual-depthwise-conv1d",
        dest="use_manual_depthwise_conv1d",
        action="store_true",
        default=False,
        help=(
            "QTL-341: fall back to the slice/mul/add manual depthwise conv1d "
            "unroll (legacy path). Default False routes the conv tail through "
            "the self.conv1d_* nn.Conv1d module so hmonnx export emits a clean "
            "Conv op."
        ),
    )
    parser.add_argument(
        "--fuse-gdr-ops",
        "--fuse_gdr_ops",
        dest="fuse_gdr_ops",
        action="store_true",
        default=False,
        help="Enable fused GDR ops when supported. Default False.",
    )
    parser.add_argument("--golden", action="store_true", help="Generate HMONNX golden after export")
    parser.add_argument(
        "--golden-only",
        "--golden_only",
        dest="golden_only",
        action="store_true",
        help="Skip export and only generate golden from existing work_dir ONNX files",
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
            "Release wmix_amix field. Pure 'w<bits>a<bits>' is preserved; "
            "anything else is normalised to 'wmix_amix'."
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
    if args.spec_decode_mode == "none":
        args.spec_decode_mode = None
    if getattr(args, "golden_only", False):
        main_golden_only(args)
    else:
        main(args)
