from __future__ import annotations

import argparse
import gc
import json
import os.path as osp
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemma4_moe_mtp_common import XHGemma4AssistantDraftModel
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name, support_llm_model_types
from xhmodel_merak.xh_llm.types import LLMModelState, ModelSwitcher
from xhmodel_merak.xh_llm.utils import unfold_args
from xhmodel_merak.utils import calculate_file_md5
from xhquant.api import Config, ConfigDict, HMONNXGoldenInference, PrecisionMode, get_xhquant_logger, ptq_quantize, set_random_seed, xhquant_init
from xhquant.api import to_export_graph, to_export_hmonnx_v2
from xhquant.utils import MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4_moe import XHGemma4MoeWithMaskConfig, XHGemma4MoeWithMaskModel


def _build_cfg_from_model(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    assistant_model_path = osp.normpath(osp.abspath(args.assistant_model))
    fallback_hf_model_path = osp.normpath(osp.abspath(args.fallback_hf_model)) if args.fallback_hf_model else hf_model_path
    model_name = Path(hf_model_path).name
    target_device = args.chip_arch
    quant_type = args.quant_type
    prefill_chunk_length = args.prefill_chunk_length
    context_length = args.context_length

    cfg_name = (
        f"{target_device}_{model_name}_with_mask_mtp_{quant_type}_{prefill_chunk_length}_{context_length // 1024}k_cli"
    )
    cfg = dict(
        chip_arch=target_device,
        model=dict(
            model_type=args.model_type,
            hf_model=hf_model_path,
            fallback_hf_model=fallback_hf_model_path,
            model_name=f"{model_name}_mtp",
            context_max_length=context_length,
            prefill_chunk_length=prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type=quant_type,
            ),
            only_first_block=False,
            quant_weight=args.quant_weight,
        ),
        mtp=dict(
            assistant_model=assistant_model_path,
            assistant_quant_type=args.assistant_quant_type,
            num_draft_tokens=args.num_draft_tokens,
        ),
    )
    cfg = format_model_name(cfg)
    return cfg_name.lower(), Config(cfg)


def _drop_visual_model(xh_model) -> None:
    if hasattr(xh_model, "visual"):
        try:
            delattr(xh_model, "visual")
        except AttributeError:
            pass
    if hasattr(xh_model, "_models") and isinstance(xh_model._models, dict):
        xh_model._models.pop("visual", None)


def _find_export_dir(work_dir: Path) -> Path:
    export_dirs = sorted(path for path in work_dir.glob("hmquant_*") if path.is_dir())
    if len(export_dirs) != 1:
        raise RuntimeError(f"Expected exactly one hmquant export dir under {work_dir}, got {export_dirs}")
    return export_dirs[0]


def _load_target_model(cfg, logger):
    model_cfg: XHGemma4MoeWithMaskConfig = AutoLLMConfig.from_pretrained(cfg.model)
    assert type(model_cfg).__name__ == "XHGemma4MoeWithMaskConfig", (
        f"Expected model config type XHGemma4MoeWithMaskConfig, but got {type(model_cfg).__name__}"
    )
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

    xh_model: XHGemma4MoeWithMaskModel = AutoLLMModel.from_pretrained(config=model_cfg)
    assert type(xh_model).__name__ == "XHGemma4MoeWithMaskModel", (
        f"Expected model type XHGemma4MoeWithMaskModel, but got {type(xh_model).__name__}"
    )
    _drop_visual_model(xh_model)
    return xh_model


def _release_model_memory(xh_model) -> None:
    if xh_model is None:
        return

    try:
        kvcache_mixin = xh_model.get_kvcache_mixin()
        kvcache_mixin.clear_kv_cache()
        kvcache_mixin.clear_other_cache()
    except Exception:
        pass

    attrs_to_clear = (
        "_inference_model",
        "_exported_model",
        "_quanted_model",
        "_frontend_model",
        "_wrap_model",
        "_data_processor",
        "hf_compatible_model",
    )
    for attr_name in attrs_to_clear:
        if hasattr(xh_model, attr_name):
            setattr(xh_model, attr_name, None)

    for sub_model in getattr(xh_model, "_models", {}).values():
        for attr_name in attrs_to_clear:
            if hasattr(sub_model, attr_name):
                setattr(sub_model, attr_name, None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_target_hmonnx(xh_model, export_root: Path, logger, profile_name: str) -> Path:
    with TimeProfiler(profile_name, logger), MemoryTracker(
        "cuda" if torch.cuda.is_available() else "cpu", profile_name, logger
    ):
        xh_model.export_hmonnx(str(export_root))
    export_dir = _find_export_dir(export_root)
    _ensure_base_meta(export_dir, xh_model.config, logger)
    _repair_target_hmonnx_files(export_dir, logger)
    _ensure_base_meta(export_dir, xh_model.config, logger)
    return export_dir


def _to_plain_data(value):
    if hasattr(value, "to_dict"):
        return _to_plain_data(value.to_dict())
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def _load_exported_text_config(export_dir: Path, cfg) -> dict:
    candidate_paths = [
        export_dir / "hf_config" / "config.json",
        Path(cfg.hf_model) / "config.json",
    ]
    fallback_hf_model = getattr(cfg, "fallback_hf_model", None)
    if fallback_hf_model:
        candidate_paths.append(Path(fallback_hf_model) / "config.json")
    for config_path in candidate_paths:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                hf_config = json.load(f)
            text_config = dict(hf_config.get("text_config", hf_config))
            for token_key in ("image_token_id", "audio_token_id", "video_token_id"):
                if token_key not in text_config and token_key in hf_config:
                    text_config[token_key] = hf_config[token_key]
            return text_config
    raise FileNotFoundError(f"Cannot find Gemma4 config.json for rebuilding meta under {export_dir}")


def _resolve_single_onnx(export_dir: Path, subdir: str, suffix: str) -> Path:
    candidates = sorted((export_dir / subdir).glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find {subdir}/*{suffix} under {export_dir}")
    if len(candidates) > 1:
        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_layer_kv_shapes(text_config: dict, context_length: int) -> list[list[int]]:
    layer_types = text_config.get("layer_types") or ["sliding_attention"] * int(text_config["num_hidden_layers"])
    num_key_value_heads = int(text_config["num_key_value_heads"])
    head_dim = int(text_config["head_dim"])
    num_global_key_value_heads = int(text_config.get("num_global_key_value_heads", num_key_value_heads))
    global_head_dim = int(text_config.get("global_head_dim", head_dim))
    layer_kv_shapes = []
    for layer_type in layer_types:
        if layer_type == "full_attention":
            layer_kv_shapes.append([1, num_global_key_value_heads, context_length, global_head_dim])
        else:
            layer_kv_shapes.append([1, num_key_value_heads, context_length, head_dim])
    return layer_kv_shapes


def _ensure_base_meta(export_dir: Path, cfg, logger) -> Path:
    base_meta_path = export_dir / "golden_meta_info.json"
    if base_meta_path.exists():
        return base_meta_path

    prefill_hmonnx = _resolve_single_onnx(export_dir, "prefill", "_prefill_with_act.onnx")
    decode_hmonnx = _resolve_single_onnx(export_dir, "decode", "_decode_with_act.onnx")
    quant_embedding = export_dir / "quant_embedding.pt"
    if not quant_embedding.exists():
        quant_embedding = export_dir / "token_embedding.pt"
    if not quant_embedding.exists():
        raise FileNotFoundError(f"Cannot find quant_embedding.pt or token_embedding.pt under {export_dir}")

    text_config = _load_exported_text_config(export_dir, cfg)
    context_length = int(getattr(cfg, "context_max_length", 2048))
    sliding_window = int(text_config.get("sliding_window", getattr(cfg, "sliding_window", 1024)))
    model_config = _to_plain_data(cfg)
    model_config.update(
        image_token_id=int(text_config.get("image_token_id", model_config.get("image_token_id", -1) or -1)),
        sliding_window=sliding_window,
        local_attention_window_size=int(model_config.get("local_attention_window_size", sliding_window)),
        global_attention_window_size=int(model_config.get("global_attention_window_size", context_length)),
        has_local_attention=True,
        has_global_attention=True,
        num_hidden_layers=int(text_config["num_hidden_layers"]),
        num_key_value_heads=int(text_config["num_key_value_heads"]),
        num_global_key_value_heads=int(text_config.get("num_global_key_value_heads", text_config["num_key_value_heads"])),
        head_dim=int(text_config["head_dim"]),
    )
    if "fallback_hf_model" not in model_config and getattr(cfg, "fallback_hf_model", None):
        model_config["fallback_hf_model"] = str(cfg.fallback_hf_model)

    layer_kv_shapes = _build_layer_kv_shapes(text_config, context_length)
    kv_cache_shape = next(
        (shape for shape in layer_kv_shapes if shape[1] == int(text_config["num_key_value_heads"])),
        layer_kv_shapes[0],
    )
    pad_token_id = int(text_config.get("pad_token_id") or text_config.get("eos_token_id") or 1)
    meta = dict(
        create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_config=model_config,
        hf_config="hf_config",
        quant_embedding=quant_embedding.name,
        quant_embedding_md5=calculate_file_md5(str(quant_embedding)),
        kv_cache=dict(
            num_layers=len(layer_kv_shapes),
            kv_cache_shape=kv_cache_shape,
            cache_axis=2,
            batch_size=1,
            cache_dtype="float16",
            use_cache=True,
        ),
        prefill_hmonnx_md5=calculate_file_md5(str(prefill_hmonnx)),
        decode_hmonnx_md5=calculate_file_md5(str(decode_hmonnx)),
        prefill_hmonnx=str(prefill_hmonnx.relative_to(export_dir)),
        decode_hmonnx=str(decode_hmonnx.relative_to(export_dir)),
        meta=dict(class_name="VLLMModelMeta"),
        pad_token_id=pad_token_id,
        visual_config=None,
        sliding_window_cfg=dict(
            sliding_window=sliding_window,
            local_attention_window_size=int(model_config["local_attention_window_size"]),
            global_attention_window_size=int(model_config["global_attention_window_size"]),
            has_local_attention=True,
            has_global_attention=True,
        ),
        kv_cache_shapes_per_layer=layer_kv_shapes,
    )
    with open(base_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    logger.warning(f"Rebuilt missing base golden_meta_info.json at {base_meta_path}")
    return base_meta_path


def _repair_dangling_clone_edges(onnx_file: str | Path, logger) -> bool:
    import onnx
    from onnx import TensorProto

    onnx_file = Path(onnx_file)
    model = onnx.load_model(str(onnx_file), load_external_data=False)

    produced = {value.name for value in model.graph.input}
    produced |= {init.name for init in model.graph.initializer}
    produced |= {sparse.name for sparse in model.graph.sparse_initializer}
    for node in model.graph.node:
        produced.update(name for name in node.output if name)

    missing_inputs = {
        input_name
        for node in model.graph.node
        for input_name in node.input
        if input_name and input_name not in produced
    }

    output_counts: dict[str, int] = {}
    for node in model.graph.node:
        for output_name in node.output:
            if output_name:
                output_counts[output_name] = output_counts.get(output_name, 0) + 1

    repaired_edges: list[tuple[str, str]] = []
    for node in model.graph.node:
        if node.op_type != "Transpose" or not node.name.startswith("node_") or len(node.output) != 1:
            continue
        expected_output = node.name.removeprefix("node_")
        actual_output = node.output[0]
        if expected_output in missing_inputs and output_counts.get(actual_output, 0) > 1:
            node.output[0] = expected_output
            repaired_edges.append((actual_output, expected_output))

    produced = {value.name for value in model.graph.input}
    produced |= {init.name for init in model.graph.initializer}
    produced |= {sparse.name for sparse in model.graph.sparse_initializer}
    for node in model.graph.node:
        produced.update(name for name in node.output if name)

    output_names = {output.name for output in model.graph.output}
    added_hidden_output = False
    if "last_hidden_state" not in output_names:
        lm_head_node = next(
            (
                node
                for node in model.graph.node
                if node.op_type == "Linear"
                and len(node.input) >= 3
                and node.input[1] == "lm_head.qweight"
                and node.input[2] == "lm_head.scale_or_exp"
            ),
            None,
        )
        logits_output = next((output for output in model.graph.output if output.name == "logits"), None)
        if lm_head_node is not None and logits_output is not None:
            hidden_name = lm_head_node.input[0]
            hidden_output = onnx.helper.make_node(
                "Identity",
                [hidden_name],
                ["last_hidden_state"],
                name="node_mtp_last_hidden_state",
            )
            model.graph.node.append(hidden_output)
            logits_shape = logits_output.type.tensor_type.shape.dim
            in_features = next(
                (onnx.helper.get_attribute_value(attr) for attr in lm_head_node.attribute if attr.name == "in_features"),
                None,
            )
            hidden_shape = [
                logits_shape[0].dim_value or logits_shape[0].dim_param,
                logits_shape[1].dim_value or logits_shape[1].dim_param,
                int(in_features),
            ]
            model.graph.output.append(
                onnx.helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT16, hidden_shape)
            )
            added_hidden_output = True

    added_lm_head = False
    if "logits" in output_names and "logits" not in produced and "last_hidden_state" in produced:
        hidden_output = next((output for output in model.graph.output if output.name == "last_hidden_state"), None)
        logits_output = next((output for output in model.graph.output if output.name == "logits"), None)
        qweight = next((init for init in model.graph.initializer if init.name == "lm_head.qweight"), None)
        scale_or_exp = next((init for init in model.graph.initializer if init.name == "lm_head.scale_or_exp"), None)
        if hidden_output is not None and logits_output is not None and qweight is not None and scale_or_exp is not None:
            hidden_shape = hidden_output.type.tensor_type.shape.dim
            logits_shape = logits_output.type.tensor_type.shape.dim
            if len(hidden_shape) == 3 and len(logits_shape) == 3:
                logits_shape[0].dim_value = hidden_shape[0].dim_value
                logits_shape[1].dim_value = hidden_shape[1].dim_value
                logits_shape[2].dim_value = int(qweight.dims[-1])
            logits_node = onnx.helper.make_node(
                "Linear",
                ["last_hidden_state", "lm_head.qweight", "lm_head.scale_or_exp"],
                ["logits"],
                name="node_mtp_lm_head",
                domain="ai.houmo.xh2a",
                hmfp_psum_dtype=b"fp24",
                hmfp_psum_round_mode=b"trunc",
                hmfp_weight_hidden_bit=1,
                output_dtype=b"float16",
                hmfp_weight_rounding=b"RNE",
                have_bias=0,
                in_features=int(hidden_shape[-1].dim_value),
                out_features=int(qweight.dims[-1]),
                mode=b"sefp",
                hmfp_act_exp_bit=5,
                hmfp_act_man_bit=8,
                hmfp_act_rounding=b"RNE",
                hmfp_act_nshare=64,
                hmfp_act_hidden_bit=1,
                hmfp_weight_exp_bit=5,
                hmfp_weight_nshare=64,
                hmfp_weight_man_bit=8,
            )
            model.graph.node.append(logits_node)
            added_lm_head = True
        else:
            logger.warning(
                f"Cannot add lm_head logits producer in {onnx_file}: "
                f"hidden_output={hidden_output is not None}, logits_output={logits_output is not None}, "
                f"qweight={qweight is not None}, scale_or_exp={scale_or_exp is not None}"
            )

    if not repaired_edges and not added_lm_head and not added_hidden_output:
        if missing_inputs:
            logger.warning(f"No dangling clone edges repaired in {onnx_file}, missing inputs: {sorted(missing_inputs)[:8]}")
        return False

    onnx.save_model(model, str(onnx_file))
    logger.info(
        f"Repaired target HMONNX outputs in {onnx_file}: "
        f"dangling_edges={repaired_edges}, added_hidden_output={added_hidden_output}, added_lm_head={added_lm_head}"
    )
    return True


def _add_mtp_external_hmfp_input_attr(onnx_file: str | Path, logger) -> bool:
    """Add external_hmfp_input attr to MTP decode model's KV cache inputs.

    The MTP decode (assistant draft) model takes KV cache from the main model
    as external inputs.  Marking these inputs lets the compiler recognise them
    as hmfp-carrying tensors even when --llm-opt / --flash-attention does not
    convert them automatically.

    ``shared_value_cache_sliding`` is set to ``false`` so that the v_sliding
    path remains in fp16.
    """
    import onnx

    EXTERNAL_HMFP_INPUTS = {
        "shared_key_cache_sliding": "true",
        "shared_value_cache_sliding": "false",
        "shared_key_cache_full": "true",
        "shared_value_cache_full": "true",
    }

    onnx_file = Path(onnx_file)
    model = onnx.load_model(str(onnx_file), load_external_data=False)

    modified = False
    for value_info in model.graph.input:
        if value_info.name in EXTERNAL_HMFP_INPUTS:
            attr_value = EXTERNAL_HMFP_INPUTS[value_info.name]
            logger.info(
                f"Adding external_hmfp_input={attr_value} attr to input '{value_info.name}' in {onnx_file.name}"
            )
            meta = onnx.StringStringEntryProto()
            meta.key = "external_hmfp_input"
            meta.value = attr_value
            value_info.metadata_props.append(meta)
            modified = True

    if modified:
        onnx.save_model(model, str(onnx_file))
        logger.info(f"Added external_hmfp_input attr to draft ONNX: {onnx_file.name}")
    else:
        logger.warning(f"No KV cache inputs found in {onnx_file.name} to add external_hmfp_input attr")

    return modified


def _repair_target_hmonnx_files(export_dir: Path, logger) -> None:
    base_meta_path = export_dir / "golden_meta_info.json"
    if not base_meta_path.exists():
        return
    with open(base_meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    changed = False
    for hmonnx_key, md5_key in (
        ("prefill_hmonnx", "prefill_hmonnx_md5"),
        ("decode_hmonnx", "decode_hmonnx_md5"),
    ):
        hmonnx_path = export_dir / meta[hmonnx_key]
        if _repair_dangling_clone_edges(hmonnx_path, logger):
            meta[md5_key] = calculate_file_md5(str(hmonnx_path))
            changed = True

    if changed:
        with open(base_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)


def _resolve_decode_quanted_model(xh_model):
    if xh_model._state != LLMModelState.QUANTED_ALIGNED:
        xh_model.to_quanted_aligned()
    xh_model._quanted_model.fixed()
    if isinstance(xh_model._quanted_model, ModelSwitcher):
        decode_quanted_model = xh_model._quanted_model.decode
    else:
        decode_quanted_model = xh_model._quanted_model
    if not decode_quanted_model.is_fixed():
        raise ValueError("decode_quanted_model model is not fixed, Please call `fixed` first.")
    decode_quanted_model.to("cpu")
    return decode_quanted_model


def _export_target_verify_hmonnx(xh_model, export_dir: Path, cfg, logger) -> tuple[str, str]:
    num_draft_tokens = int(cfg.mtp.num_draft_tokens)
    verify_length = num_draft_tokens + 1
    export_model_name = export_dir.name
    verify_dir = export_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    text_config = _load_exported_text_config(export_dir, cfg.model)
    eos_token_id = text_config.get("eos_token_id", 1)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    xh_model.pad_token_id = int(text_config.get("pad_token_id") or eos_token_id or 1)

    xh_model._llm_prefill = False
    xh_model.set_input_sequence_length(verify_length)

    with TimeProfiler("convert_target_verify", logger), MemoryTracker(
        "cuda" if torch.cuda.is_available() else "cpu", "convert_target_verify", logger
    ):
        original_get_decode_dummy_inputs = xh_model.get_decode_dummy_inputs

        def get_verify_decode_dummy_inputs():
            return {
                "input_ids": torch.randint(0, 100, (1, verify_length), dtype=torch.long),
                "past_seq_length": xh_model.config.prefill_chunk_length,
            }

        xh_model.get_decode_dummy_inputs = get_verify_decode_dummy_inputs
        try:
            decode_quanted_model = _resolve_decode_quanted_model(xh_model)
        finally:
            xh_model.get_decode_dummy_inputs = original_get_decode_dummy_inputs
        with xh_model.get_kvcache_mixin().kv_cache_scope(device="meta"):
            xh_model.set_decode()
            xh_model.set_input_sequence_length(verify_length)
            data_processor = xh_model.get_data_preprocessor()
            data_processor.input_sequence_length = verify_length
            dummy_input = {
                "input_ids": torch.randint(0, 100, (1, verify_length), dtype=torch.long),
                "past_seq_length": xh_model.config.prefill_chunk_length,
            }
            inputs = data_processor(dummy_input)
            inputs = unfold_args(inputs)
            verify_exported_model = to_export_graph(decode_quanted_model, inputs)

            verify_hmonnx_file = str(verify_dir / f"{export_model_name}_verify.onnx")
            export_cfg = xh_model.get_export_cfg()
            verify_hmonnx_file = to_export_hmonnx_v2(
                verify_exported_model,
                inputs,
                verify_hmonnx_file,
                export_cfg,
                normalize_onnx_name=True,
            )

    verify_path = Path(verify_hmonnx_file)
    _repair_dangling_clone_edges(verify_path, logger)
    verify_rel = str(verify_path.relative_to(export_dir))
    verify_md5 = calculate_file_md5(verify_hmonnx_file)
    logger.info(f"Target verify HMONNX exported to {verify_hmonnx_file}")
    return verify_rel, verify_md5


def _export_assistant_draft(cfg, export_dir: Path, logger) -> str:
    mtp_cfg = cfg.mtp
    draft_onnx_dir = export_dir / "draft_onnx"
    draft_onnx_dir.mkdir(parents=True, exist_ok=True)

    assistant_model = XHGemma4AssistantDraftModel(
        assistant_model_dir=mtp_cfg.assistant_model,
        target_model_dir=cfg.model.hf_model,
        wrap_cfg=ConfigDict(
            input_sequence_length=1,
            max_sequence_length=cfg.model.context_max_length,
            dtype="float16",
        ),
        quant_config=ConfigDict(quant_type=mtp_cfg.assistant_quant_type),
    )
    assistant_model.init_wrap_model()
    dummy_data = assistant_model.prepare_inputs(None)
    assistant_model.convert_to_fronted_graph(dummy_data)
    assistant_model.convert_to_quant_graph(cfg.chip_arch)
    ptq_quantize(
        assistant_model.quanted_model,
        [assistant_model.prepare_inputs(None)],
        PrecisionMode.ALIGNED,
        [torch.device("cpu")],
    )
    assistant_model.convert_to_export_graph(dummy_data)
    onnx_file = assistant_model.to_export_onnx(dummy_data, str(draft_onnx_dir), prefix="gemma4_assistant_decode")[0]
    assistant_model.release_exported_model()
    assistant_model.release_quanted_model()
    assistant_model.release_frontend_model()
    assistant_model.release_wraped_model()
    _add_mtp_external_hmfp_input_attr(onnx_file, logger)
    logger.info(f"Assistant draft ONNX exported to {onnx_file}")
    return onnx_file


def _write_mtp_meta(base_meta_path: Path, draft_onnx_file: str, verify_hmonnx: str, verify_hmonnx_md5: str, cfg) -> Path:
    if not base_meta_path.exists():
        _ensure_base_meta(base_meta_path.parent, cfg.model, get_xhquant_logger())
    with open(base_meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    num_draft_tokens = int(cfg.mtp.num_draft_tokens)
    verify_length = num_draft_tokens + 1
    draft_rel = Path(draft_onnx_file).resolve().relative_to(base_meta_path.parent.resolve())
    meta["verify_hmonnx"] = verify_hmonnx
    meta["verify_hmonnx_md5"] = verify_hmonnx_md5
    meta["spec_decode"] = dict(
        mode="mtp",
        draft_onnx=str(draft_rel),
        draft_decode_onnx=str(draft_rel),
        verify_hmonnx=verify_hmonnx,
        verify_decode_onnx=verify_hmonnx,
        block_size=num_draft_tokens,
        verify_length=verify_length,
        hidden_output_name="last_hidden_state",
        assistant_hidden_output_name="assistant_hidden_state",
        assistant_model_dir=str(Path(cfg.mtp.assistant_model).resolve()),
    )
    meta["mtp"] = dict(cfg.mtp)

    output_path = base_meta_path.parent / "golden_meta_info_mtp.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    return output_path


def _generate_golden_for_hmonnx(hmonnx_file: str, golden_dir: Path, device: torch.device, logger) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    session = HMONNXGoldenInference(hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.initialize()

    input_names = session.get_input_names()
    net_inputs = []
    for input_name in input_names:
        input_info = session.get_input(input_name)
        if input_info.dtype in (torch.float32, torch.float16, torch.float64):
            tensor = torch.randn(input_info.shape, dtype=input_info.dtype, device=device)
        elif input_info.dtype in (torch.int32, torch.int64, torch.int16):
            tensor = torch.randint(0, 100, input_info.shape, dtype=input_info.dtype, device=device)
        elif input_info.dtype == torch.bool:
            tensor = torch.randint(0, 2, input_info.shape, dtype=input_info.dtype, device=device)
        else:
            tensor = torch.zeros(input_info.shape, dtype=input_info.dtype, device=device)
        net_inputs.append(tensor)

    with torch.no_grad():
        session(*net_inputs)
    logger.info(f"Golden saved to: {golden_dir}")


def _generate_golden_for_export(export_dir: Path, device: torch.device, logger) -> None:
    hmonnx_patterns = [
        ("prefill", "*_prefill_with_act.onnx"),
        ("decode", "*_decode_with_act.onnx"),
        ("verify", "*_verify_with_act.onnx"),
        ("draft_onnx", "*.onnx"),
    ]
    for subdir, pattern in hmonnx_patterns:
        onnx_dir = export_dir / subdir
        if not onnx_dir.exists():
            continue
        onnx_files = sorted(onnx_dir.glob(pattern))
        for onnx_file in onnx_files:
            golden_name = onnx_file.stem
            golden_path = export_dir / "golden" / golden_name
            logger.info(f"Generating golden for {onnx_file.relative_to(export_dir)}")
            try:
                _generate_golden_for_hmonnx(str(onnx_file), golden_path, device, logger)
            except Exception as e:
                logger.warning(f"Failed to generate golden for {onnx_file}: {e}")


def main(args):
    config_file = args.config
    model_dir = args.model
    if config_file and model_dir:
        raise ValueError("Cannot specify both --config and --model at the same time. Please choose one.")
    if config_file:
        cfg_name = Path(config_file).stem
        cfg = Config.fromfile(args.config)
    elif model_dir:
        cfg_name, cfg = _build_cfg_from_model(args)
    else:
        raise ValueError("Either --config or --model must be specified.")

    if args.debug:
        cfg_name += "_debug"
    if args.valid:
        cfg_name += "_valid"
        cfg.model.only_first_block = True

    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = Path(args.work_dir)
    if work_dir.exists():
        if args.force:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            from loguru import logger

            logger.info(f"Exported model already exists at {work_dir}, use --force to overwrite.")
            return -1

    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / "export_hmonnx.log")

    xhquant_init(log_file, args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg.seed = seed
    logger.info(f"Config:\n{cfg.pretty_text}")
    dumped_config_file = work_dir / f"{cfg_name}.py"
    cfg.dump(dumped_config_file)

    logger.info("Loading target model for prefill export")
    target_model = _load_target_model(cfg, logger)
    target_model.wrap_cfg.output_hidden_states_for_export = True
    export_dir = _export_target_hmonnx(target_model, work_dir, logger, "convert_target")
    logger.info("Releasing target export model before verify export")
    _release_model_memory(target_model)
    del target_model

    logger.info("Loading target model for verify export")
    verify_model = _load_target_model(cfg, logger)
    verify_model.wrap_cfg.num_logits_to_keep = 0
    verify_model.wrap_cfg.output_hidden_states_for_export = True
    verify_hmonnx, verify_hmonnx_md5 = _export_target_verify_hmonnx(verify_model, export_dir, cfg, logger)
    logger.info("Releasing target verify model before assistant export")
    _release_model_memory(verify_model)
    del verify_model

    base_meta_path = export_dir / "golden_meta_info.json"
    draft_onnx_file = _export_assistant_draft(cfg, export_dir, logger)
    mtp_meta_path = _write_mtp_meta(base_meta_path, draft_onnx_file, verify_hmonnx, verify_hmonnx_md5, cfg)
    logger.info(f"MTP meta saved to {mtp_meta_path}")

    if args.golden:
        golden_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Generating golden data for all exported HMONNX modules")
        _generate_golden_for_export(export_dir, golden_device, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_mtp_26b_a4b_it_xh2a_w4a8_256_2k.py",
        help="model config file, for development and debugging.",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--force", default=True, action="store_true", help="Whether to force export even if the model exists.")
    parser.add_argument("--golden", action="store_true", help="Generate golden data for each exported HMONNX module.")
    parser.add_argument(
        "--valid",
        default=False,
        action="store_true",
        help="Wrap only the first decoder block for a fast smoke export.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="Gemma4ForConditionalGeneration_with_mask",
        choices=support_llm_model_types,
    )
    parser.add_argument(
        "--chip-arch",
        type=str,
        default="XH2a",
        choices=["XH2a", "YueHui"],
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--fallback-hf-model",
        type=str,
        default="",
        help="Optional float HF model directory used to fill missing non-quantized tensors when --model points to GPTQ weights.",
    )
    parser.add_argument(
        "--assistant-model",
        type=str,
        default="/data01/home/chenzx/model/gemma-4-26B-A4B-it-assistant",
        help="Gemma4 assistant model path.",
    )
    parser.add_argument("--context-length", type=int, default=2048, help="max context sequence length")
    parser.add_argument("--prefill-chunk-length", type=int, default=256, help="prefill chunk length")
    parser.add_argument("--quant-type", default="w4a8h1_ssfp", help="target quant type")
    parser.add_argument("--assistant-quant-type", default="w8a8h1_sefp", help="assistant quant type")
    parser.add_argument("--num-draft-tokens", type=int, default=4, help="assistant draft tokens per round")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use config quantization.",
    )
    main(parser.parse_args())