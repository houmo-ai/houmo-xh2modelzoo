"""Workflow-owned Gemma4 Series MTP export and golden helpers."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[4]
MTP_DRAFT_DECODE_DIR = "mtp_draft_decode"
REQUIRED_SHARED_KV_INPUTS = (
    "shared_key_cache_sliding",
    "shared_value_cache_sliding",
    "shared_key_cache_full",
    "shared_value_cache_full",
)


def quant_type_weight_bits(quant_type: str | None, default: int = 4) -> int:
    if not quant_type or not str(quant_type).startswith("w"):
        return default
    digits: list[str] = []
    for char in str(quant_type)[1:]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else default


def model_config_dict(meta: dict[str, Any]) -> dict[str, Any]:
    model_config = meta.get("model_config") or {}
    return model_config if isinstance(model_config, dict) else {}


def mtp_config_dict(meta: dict[str, Any]) -> dict[str, Any]:
    mtp_config = model_config_dict(meta).get("mtp_config") or {}
    return mtp_config if isinstance(mtp_config, dict) else {}


def resolve_context_length(model_cfg: dict[str, Any], mtp_cfg: dict[str, Any]) -> int:
    return int(mtp_cfg.get("context_max_length") or model_cfg.get("context_max_length") or 2048)


def is_mtp_manifest(meta: dict[str, Any]) -> bool:
    spec_decode = meta.get("spec_decode") or {}
    mode = spec_decode.get("mode") if isinstance(spec_decode, dict) else None
    return str(meta.get("spec_decode_mode") or mode).lower() == "mtp"


def resolve_model_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def require_model_dir(path: str, *, role: str) -> Path:
    model_dir = resolve_model_path(path)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Gemma4 MTP {role} model directory does not exist: {model_dir}"
        )
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Gemma4 MTP {role} model directory is missing config.json: "
            f"{model_dir}"
        )
    return model_dir


def validate_mtp_model_inputs(model_cfg: dict[str, Any]) -> dict[str, Path]:
    mtp_cfg = model_cfg.get("mtp_config") or {}
    if not isinstance(mtp_cfg, dict):
        raise TypeError("Gemma4 MTP export.model.mtp_config must be a mapping")

    assistant_dir = mtp_cfg.get("assistant_hf_model")
    target_dir = mtp_cfg.get("target_hf_model")
    missing = [
        f"export.model.mtp_config.{name}"
        for name, value in (
            ("assistant_hf_model", assistant_dir),
            ("target_hf_model", target_dir),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Gemma4 MTP export requires model paths before graph export: "
            f"missing {', '.join(missing)}. "
            "Inject them with workflow config_overrides, the "
            "--mtp-assistant-model-dir/--mtp-target-model-dir CLI options, "
            "or a public Gemma4 preset."
        )

    shared_kv_inputs = mtp_cfg.get("shared_kv_inputs")
    if list(shared_kv_inputs or []) != list(REQUIRED_SHARED_KV_INPUTS):
        raise ValueError(
            "Gemma4 MTP export.model.mtp_config.shared_kv_inputs must be "
            f"{list(REQUIRED_SHARED_KV_INPUTS)!r}; got "
            f"{shared_kv_inputs!r}. Duplicate and legacy KV names are not "
            "accepted."
        )

    model_paths = {
        "assistant": require_model_dir(str(assistant_dir), role="assistant"),
        "target": require_model_dir(str(target_dir), role="target"),
    }
    from .gemma4_series_mtp_model import validate_assistant_target_contract

    validate_assistant_target_contract(
        model_paths["assistant"],
        model_paths["target"],
        mtp_cfg,
    )
    return model_paths


def resolve_exported_dir(export_result: Any) -> Path:
    exported_dir = getattr(export_result, "exported_dir", None)
    if exported_dir:
        return Path(exported_dir)

    work_dir = Path(getattr(export_result, "work_dir", ""))
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Export work_dir does not exist: {work_dir}")

    prefill_hmonnx = getattr(getattr(export_result, "meta", None), "prefill_hmonnx", None)
    if prefill_hmonnx:
        matches = [
            path
            for path in work_dir.iterdir()
            if path.is_dir() and path.name.startswith("hmquant") and (path / str(prefill_hmonnx)).exists()
        ]
        if len(matches) == 1:
            return matches[0]

    meta_files = [
        path / "golden_meta_info.json"
        for path in work_dir.iterdir()
        if path.is_dir() and path.name.startswith("hmquant") and (path / "golden_meta_info.json").is_file()
    ]
    if len(meta_files) == 1:
        return meta_files[0].parent
    if not meta_files:
        raise RuntimeError(f"Cannot locate exported HMONNX directory under {work_dir}")
    meta_file_list = ", ".join(str(path) for path in meta_files)
    raise RuntimeError(f"Found multiple exported HMONNX directories under {work_dir}: {meta_file_list}")


def infer_shared_cache_lengths(meta: dict[str, Any]) -> tuple[int | None, int | None]:
    spec = dict(meta.get("spec_decode") or {})
    sliding_len = spec.get("shared_sliding_cache_length")
    full_len = spec.get("shared_full_cache_length")
    if sliding_len and full_len:
        return int(sliding_len), int(full_len)

    model_cfg = model_config_dict(meta)
    context_len = int(model_cfg.get("context_max_length") or meta.get("context_max_length") or 0)
    sliding_window = int(meta.get("sliding_window") or model_cfg.get("sliding_window") or 0)
    prefill_len = int(model_cfg.get("prefill_chunk_length") or model_cfg.get("input_sequence_length") or 0)
    if not sliding_len and sliding_window > 0 and prefill_len > 0:
        sliding_len = ((sliding_window + prefill_len + 15) // 16) * 16
    if not full_len and context_len > 0:
        full_len = context_len

    layer_shapes = meta.get("layer_kv_shapes") or meta.get("kv_cache_shapes_per_layer") or []
    lengths = sorted({int(shape[2]) for shape in layer_shapes if len(shape) > 2})
    if not sliding_len and lengths:
        candidates = [length for length in lengths if context_len <= 0 or length < context_len]
        sliding_len = max(candidates or lengths)
    if not full_len and lengths:
        full_len = max(lengths)
    return (int(sliding_len) if sliding_len else None, int(full_len) if full_len else None)


def relative_to_export_dir(path: Path, export_dir: Path) -> str:
    try:
        return path.resolve().relative_to(export_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _first_present(*values: Any) -> Any:
    for value in values:
        if _present(value):
            return value
    return None


def update_manifest_with_draft(
    meta_path: Path,
    draft_onnx: Path,
    *,
    lm_head_quant_type: str,
    shared_sliding_len: int | None,
    shared_full_len: int | None,
    context_length: int | None = None,
    draft_rope_max_pe_length: int | None = None,
    target_max_pe_length_source: str | None = None,
    target_max_pe_length_hf_value: int | None = None,
    target_max_pe_length_hf_source: str | None = None,
) -> None:
    export_dir = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cfg = model_config_dict(meta)
    spec_decode = dict(meta.get("spec_decode") or {})
    model_num_draft_tokens = model_cfg.get("num_draft_tokens")
    block_size = int(
        _first_present(
            spec_decode.get("block_size"),
            model_num_draft_tokens,
            meta.get("spec_decode_block_size"),
            4,
        )
    )
    if _present(spec_decode.get("verify_length")):
        verify_length = int(spec_decode["verify_length"])
    elif _present(model_cfg.get("spec_decode_verify_length")):
        verify_length = int(model_cfg["spec_decode_verify_length"])
    elif _present(model_cfg.get("verify_length")):
        verify_length = int(model_cfg["verify_length"])
    elif _present(model_num_draft_tokens):
        verify_length = block_size + 1
    else:
        verify_length = int(_first_present(meta.get("spec_decode_verify_length"), block_size + 1))
    draft_head_bits = quant_type_weight_bits(lm_head_quant_type)
    rel_draft = relative_to_export_dir(draft_onnx, export_dir)
    sliding_window = int(meta.get("sliding_window") or model_cfg.get("sliding_window") or 0)
    target_decode_sliding = None
    if sliding_window > 0:
        target_decode_sliding = ((sliding_window + verify_length - 1 + 15) // 16) * 16

    shared_sliding_len = spec_decode.get("shared_sliding_cache_length") or shared_sliding_len
    shared_full_len = spec_decode.get("shared_full_cache_length") or shared_full_len
    spec_decode.update(
        mode="mtp",
        block_size=block_size,
        verify_length=verify_length,
        hidden_output_name="target_hidden_state",
        draft_head_weight_bits=draft_head_bits,
        draft_decode_onnx=rel_draft,
        draft_onnx=rel_draft,
        shared_sliding_cache_length=shared_sliding_len,
        shared_full_cache_length=shared_full_len,
        target_decode_sliding_output_length=target_decode_sliding,
    )
    if context_length is not None:
        spec_decode["context_length"] = int(context_length)
    if draft_rope_max_pe_length is not None:
        draft_rope_max_pe_length = int(draft_rope_max_pe_length)
        spec_decode["draft_rope_max_pe_length"] = draft_rope_max_pe_length
        if isinstance(model_cfg, dict):
            model_cfg["max_pe_length"] = draft_rope_max_pe_length
    if target_max_pe_length_source:
        source = str(target_max_pe_length_source)
        spec_decode["target_max_pe_length_source"] = source
        if isinstance(model_cfg, dict):
            model_cfg["max_pe_length_source"] = source
    if target_max_pe_length_hf_value is not None:
        hf_value = int(target_max_pe_length_hf_value)
        spec_decode["target_max_pe_length_hf_value"] = hf_value
        if isinstance(model_cfg, dict):
            model_cfg["max_pe_length_hf_value"] = hf_value
    if target_max_pe_length_hf_source:
        hf_source = str(target_max_pe_length_hf_source)
        spec_decode["target_max_pe_length_hf_source"] = hf_source
        if isinstance(model_cfg, dict):
            model_cfg["max_pe_length_hf_source"] = hf_source
    meta.update(
        spec_decode_mode="mtp",
        spec_decode_block_size=block_size,
        spec_decode_hidden_output_name="target_hidden_state",
        spec_decode_verify_length=verify_length,
        spec_decode_draft_head_weight_bits=draft_head_bits,
        draft_decode_onnx_file=rel_draft,
        draft_onnx_file=rel_draft,
        spec_decode=spec_decode,
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")


def resolve_and_update_manifest_max_pe_length(
    meta_path: Path,
    target_model_dir: str | Path,
    *,
    max_pe_length_explicit: bool,
) -> dict[str, Any]:
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        resolve_target_max_pe_length,
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cfg = model_config_dict(meta)
    resolved = resolve_target_max_pe_length(
        target_model_dir,
        model_cfg,
        max_pe_length_explicit=max_pe_length_explicit,
        return_metadata=True,
    )
    assert isinstance(resolved, dict)
    model_cfg["max_pe_length"] = int(resolved["value"])
    model_cfg["max_pe_length_source"] = str(resolved["source"])
    if resolved.get("hf_value") is not None:
        model_cfg["max_pe_length_hf_value"] = int(resolved["hf_value"])
    if resolved.get("hf_source") is not None:
        model_cfg["max_pe_length_hf_source"] = str(resolved["hf_source"])
    spec_decode = meta.get("spec_decode")
    if isinstance(spec_decode, dict) and str(spec_decode.get("mode") or meta.get("spec_decode_mode")).lower() == "mtp":
        spec_decode["draft_rope_max_pe_length"] = int(resolved["value"])
        spec_decode["target_max_pe_length_source"] = str(resolved["source"])
        if resolved.get("hf_value") is not None:
            spec_decode["target_max_pe_length_hf_value"] = int(resolved["hf_value"])
        if resolved.get("hf_source") is not None:
            spec_decode["target_max_pe_length_hf_source"] = str(resolved["hf_source"])
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")
    return resolved


def draft_quant_config(body_quant_type: str, lm_head_quant_type: str) -> Any:
    from xhquant.api import ConfigDict

    return ConfigDict(
        quant_type=body_quant_type,
        nodes=dict(
            lm_head=dict(quant_type=lm_head_quant_type),
            masked_embedding=dict(quant_type=lm_head_quant_type),
            masked_embedding_centroids=dict(quant_type=lm_head_quant_type),
            **{"masked_embedding.centroids": dict(quant_type=lm_head_quant_type)},
        ),
        ops={},
    )


def export_mtp_draft(
    export_result: Any,
    *,
    hf_model_dir: str,
    chip_arch: str | None = None,
    draft_dtype: str = "float16",
) -> Path | None:
    exported_dir = resolve_exported_dir(export_result)
    meta_path = exported_dir / "golden_meta_info.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not is_mtp_manifest(meta):
        return None

    model_cfg = model_config_dict(meta)
    mtp_cfg = mtp_config_dict(meta)
    target_dir = mtp_cfg.get("target_hf_model") or hf_model_dir
    if target_dir and not mtp_cfg.get("target_hf_model"):
        mtp_cfg["target_hf_model"] = target_dir
    model_paths = validate_mtp_model_inputs(
        {**model_cfg, "mtp_config": mtp_cfg}
    )

    body_quant_type = str(mtp_cfg.get("body_quant_type") or "w8a8h1_sefp")
    lm_head_quant_type = str(mtp_cfg.get("lm_head_quant_type") or "w4a8h0_ssfp")
    chip_arch = str(model_cfg.get("chip_arch") or meta.get("chip_arch") or chip_arch or "XH2a")
    context_length = resolve_context_length(model_cfg, mtp_cfg)
    dtype = str(mtp_cfg.get("dtype") or mtp_cfg.get("draft_dtype") or draft_dtype)
    shared_sliding_len, shared_full_len = infer_shared_cache_lengths(meta)
    output_dir = exported_dir / MTP_DRAFT_DECODE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import (
        XHGemma4SeriesAssistantDraftModel,
        resolve_target_max_pe_length,
        validate_assistant_target_contract,
    )
    from xhquant.api import ConfigDict, PrecisionMode, get_xhquant_logger, ptq_quantize

    target_path = model_paths["target"]
    assistant_path = model_paths["assistant"]
    validate_assistant_target_contract(assistant_path, target_path, mtp_cfg)
    target_max_pe_length_info = resolve_target_max_pe_length(
        target_path,
        model_cfg,
        max_pe_length_explicit=model_cfg.get("max_pe_length_source") == "model_config.max_pe_length",
        return_metadata=True,
    )
    assert isinstance(target_max_pe_length_info, dict)
    target_max_pe_length = int(target_max_pe_length_info["value"])
    target_max_pe_length_source = str(target_max_pe_length_info["source"])
    logger = get_xhquant_logger()
    logger.info("Exporting Gemma4 Series MTP assistant draft ONNX to %s", output_dir)
    model = XHGemma4SeriesAssistantDraftModel(
        assistant_model_dir=str(assistant_path),
        target_model_dir=str(target_path),
        wrap_cfg=ConfigDict(
            input_sequence_length=int(mtp_cfg.get("input_sequence_length") or 1),
            context_length=context_length,
            dtype=dtype,
            cache_axis=2,
            shared_sliding_cache_length=shared_sliding_len,
            shared_full_cache_length=shared_full_len,
            target_max_pe_length=target_max_pe_length,
            model_config=dict(model_cfg),
        ),
        quant_config=draft_quant_config(body_quant_type, lm_head_quant_type),
    )
    model.init_wrap_model()
    dummy = model.prepare_inputs(None)
    model.convert_to_fronted_graph(dummy)
    model.convert_to_quant_graph(chip_arch.lower())
    ptq_quantize(model.quanted_model, [model.prepare_inputs(None)], PrecisionMode.ALIGNED, [torch.device("cpu")])
    model.convert_to_export_graph(dummy)
    onnx_file = Path(
        model.to_export_onnx(
            dummy,
            str(output_dir),
            prefix=f"gemma4_series_{Path(hf_model_dir).name}_assistant_decode",
        )[0]
    )
    model.release_exported_model()
    model.release_quanted_model()
    model.release_frontend_model()
    model.release_wraped_model()
    gc.collect()

    update_manifest_with_draft(
        meta_path,
        onnx_file,
        lm_head_quant_type=lm_head_quant_type,
        shared_sliding_len=shared_sliding_len,
        shared_full_len=shared_full_len,
        context_length=context_length,
        draft_rope_max_pe_length=target_max_pe_length,
        target_max_pe_length_source=target_max_pe_length_source,
        target_max_pe_length_hf_value=target_max_pe_length_info.get("hf_value"),
        target_max_pe_length_hf_source=target_max_pe_length_info.get("hf_source"),
    )
    return onnx_file


def find_mtp_draft_onnx(export_result: Any) -> Path | None:
    exported_dir = resolve_exported_dir(export_result)
    meta_path = exported_dir / "golden_meta_info.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return resolve_mtp_draft_onnx(meta_path, meta=meta)


def resolve_mtp_draft_onnx(meta_path: Path, *, meta: dict[str, Any] | None = None) -> Path | None:
    meta = meta or json.loads(meta_path.read_text(encoding="utf-8"))
    if not is_mtp_manifest(meta):
        return None
    spec_decode = meta.get("spec_decode") or {}
    candidates: list[Any] = []
    if isinstance(spec_decode, dict):
        candidates.extend([spec_decode.get("draft_decode_onnx"), spec_decode.get("draft_onnx")])
    candidates.extend([meta.get("draft_decode_onnx_file"), meta.get("draft_onnx_file")])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        return path if path.is_absolute() else meta_path.parent / path
    return None


def dump_mtp_draft_golden(meta_file: str | Path, device: str, *, logger: Any | None = None) -> Path | None:
    meta_path = Path(meta_file)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    draft_onnx = resolve_mtp_draft_onnx(meta_path, meta=meta)
    if draft_onnx is None:
        return None
    if not draft_onnx.is_file():
        raise FileNotFoundError(f"MTP draft ONNX referenced by manifest does not exist: {draft_onnx}")

    golden_dir = meta_path.parent / MTP_DRAFT_DECODE_DIR
    if golden_dir.is_dir() and any(golden_dir.glob("step_*/*.npy")):
        return golden_dir
    golden_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from xhquant.api import HMONNXGraphGoldenInference

    if logger is not None:
        logger.info("Generating Gemma4 MTP draft golden from %s", draft_onnx)
    session = HMONNXGraphGoldenInference(str(draft_onnx))
    session.exec_device = torch.device(device)
    session.to(device)
    session.save_golden = True
    session.golden_dir = golden_dir
    session.initialize()
    input_feed = build_zero_golden_input_feed(session, torch.device(device))
    session.run(input_feed)
    del session
    gc.collect()
    return golden_dir


def build_zero_golden_input_feed(session: Any, device: Any) -> dict[str, Any]:
    import torch

    input_feed: dict[str, Any] = {}
    for name in session.get_input_names():
        info = session.get_input(name)
        shape = tuple(int(dim) for dim in info.shape)
        dtype = info.dtype
        if name in {"past_seq_length", "valid_length"}:
            input_feed[name] = torch.zeros(shape, dtype=dtype, device=device)
        elif name in {"current_input_length", "current_length"}:
            seq_len = 1
            for dim in shape:
                if dim > 1:
                    seq_len = dim
                    break
            input_feed[name] = torch.full(shape, seq_len, dtype=dtype, device=device)
        else:
            input_feed[name] = torch.zeros(shape, dtype=dtype, device=device)
    return input_feed
