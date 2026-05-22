import gc
import json
import shutil
import sys
import time

from pathlib import Path
from typing import List, Tuple

import torch
import xhquant.utils.suppress_printing
from PIL import Image, ImageOps
from torch import Tensor
from xhquant.api import ConfigDict, HMONNXGoldenInference, PrecisionMode, QTensor, ptq_quantize, set_random_seed

# Keep behavior aligned with glm4v export script: ensure project root is importable
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import decode_next_token, xhquant_llm_init, get_root_logger
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.glm_ocr import GlmOcrHFCompatible, GlmOcrProcessor, XHGlmOcrLLMModel
from xh_model_zoo.xh_llm.models.glm_ocr.utils import build_inputs, build_messages
from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_model_dir", type=str, default="/data01/datasets/GLM-OCR",
                        help="HuggingFace model directory")
    parser.add_argument("--work_dir", type=str, default="work_dirs/glm_ocr_llm_xh2a_2k_export",
                        help="output work directory")
    parser.add_argument("--image_path", type=str, default="examples/llm/glm_ocr/data/img3.png")
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--valid", default=True, help="evaluate the model")
    parser.add_argument("--profile_nodes", default=False, action="store_true", help="profile traced/quanted graph node outputs")
    parser.add_argument("--profile_decode_steps", type=int, default=3, help="number of decode steps to profile when --profile_nodes is set")
    parser.add_argument("--profile_dir", type=str, default=None, help="node profile output dir, default work_dir/node_profile")
    parser.add_argument("--image_size_w", type=int, default=672, help="image width")
    parser.add_argument("--image_size_h", type=int, default=672, help="image height")
    parser.add_argument("--max_sequence_length", type=int, default=2048, help="max sequence length")
    parser.add_argument("--input_sequence_length", type=int, default=256, help="prefill input sequence length")
    parser.add_argument("--target_device", type=str, default="XH2a", help="target device")
    parser.add_argument("--skip_golden", action="store_true", help="skip HMONNX golden generation")
    parser.add_argument("--golden_dir", type=str, default=None, help="golden output dir, default work_dir/golden")
    return parser


def _copy_hf_configs(src_dir: Path, dst_dir: Path, logger):
    dst_dir.mkdir(parents=True, exist_ok=True)
    hf_config_file_candidates = [
        ["chat_template.json", "chat_template.jinja"],
        ["config.json"],
        ["generation_config.json"],
        ["preprocessor_config.json"],
        ["tokenizer_config.json"],
        ["vocab.json"],
        ["tokenizer.json"],
        ["merges.txt"],
        ["special_tokens_map.json"],
    ]
    for candidates in hf_config_file_candidates:
        copied = False
        for cfg_file in candidates:
            src_file = src_dir / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_dir / cfg_file)
                copied = True
                break
        if not copied:
            logger.warning(f"Skip copying hf config files, missing {candidates} in {src_dir}")


def _flatten_args(args):
    flat_args = []
    for arg in args:
        if isinstance(arg, (List, Tuple)):
            flat_args.extend(arg)
        else:
            flat_args.append(arg)
    return flat_args


def _decode_new_tokens(processor, input_ids, generated_ids):
    if isinstance(generated_ids, torch.Tensor):
        return processor.decode(generated_ids[0][input_ids.shape[1] :], skip_special_tokens=False)
    if isinstance(generated_ids, (list, tuple)) and len(generated_ids) > 0:
        return processor.decode(generated_ids[0][input_ids.shape[1] :], skip_special_tokens=False)
    return ""


def _load_and_process_image(image_path: str, target_w: int, target_h: int):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if (orig_w, orig_h) != (target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114))
    return image


def _build_stop_token_ids(processor, eos_token_id):
    stop_token_ids = set()
    if eos_token_id is not None:
        if isinstance(eos_token_id, (list, tuple)):
            stop_token_ids.update(int(v) for v in eos_token_id)
        else:
            stop_token_ids.add(int(eos_token_id))

    tokenizer_eos = getattr(processor.tokenizer, "eos_token_id", None)
    tokenizer_pad = getattr(processor.tokenizer, "pad_token_id", None)
    if tokenizer_eos is not None:
        stop_token_ids.add(int(tokenizer_eos))
    if tokenizer_pad is not None:
        stop_token_ids.add(int(tokenizer_pad))

    for token in ["<|user|>", "<|assistant|>", "<|observation|>", "<eop>"]:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and int(token_id) >= 0:
            stop_token_ids.add(int(token_id))

    return sorted(stop_token_ids)


def _align_inputs_device(inputs, device):
    aligned = {}
    for key, value in inputs.items():
        if isinstance(value, Tensor):
            aligned[key] = value.to(device)
        else:
            aligned[key] = value
    return aligned


def _prepare_generate_inputs(inputs):
    generate_inputs = dict(inputs)
    generate_inputs.pop("token_type_ids", None)
    generate_inputs.pop("mm_token_type_ids", None)
    return generate_inputs


def _fix_image_token_id_if_needed(model_config, input_ids: Tensor, image_grid_thw: Tensor, logger):
    current_id = int(model_config.image_token_id)
    current_count = int((input_ids == current_id).sum().item())
    spatial_merge_size = int(model_config.vision_config.spatial_merge_size)
    expected_count = int((image_grid_thw.prod(-1) // (spatial_merge_size**2)).sum().item())

    if current_count == expected_count:
        return

    unique_ids, counts = torch.unique(input_ids, return_counts=True)
    matched = unique_ids[counts == expected_count]
    if matched.numel() == 1:
        detected_id = int(matched[0].item())
        logger.warning(
            f"image_token_id mismatch: configured={current_id}, observed_count={current_count}, "
            f"expected={expected_count}. Auto-fix to detected image_token_id={detected_id}."
        )
        model_config.image_token_id = detected_id


def _safe_generate_text(model, inputs, processor, input_ids, max_new_tokens, logger, stage: str, stop_token_ids=None):

    llm_model = getattr(model, "_llm_model", None)
    if isinstance(llm_model, XHGlmOcrLLMModel):
        _reset_kv_cache_buffers(llm_model)
        llm_model.rope_deltas = None
    elif isinstance(model, XHGlmOcrLLMModel):
        _reset_kv_cache_buffers(model)
        model.rope_deltas = None

    model_device = input_ids.device
    try:
        model_device = next(model.parameters()).device
    except Exception:
        pass
    aligned_inputs = _align_inputs_device(_prepare_generate_inputs(inputs), model_device)

    with torch.no_grad():
        generate_kwargs = dict(max_new_tokens=max_new_tokens)
        if stop_token_ids is not None and len(stop_token_ids) > 0:
            generate_kwargs["eos_token_id"] = stop_token_ids
            if getattr(processor.tokenizer, "pad_token_id", None) is not None:
                generate_kwargs["pad_token_id"] = int(processor.tokenizer.pad_token_id)
        generated_ids = model.generate(**aligned_inputs, **generate_kwargs)
    return _decode_new_tokens(processor, input_ids, generated_ids)


def _safe_test_next_token(model, tokenizer, data_batch, logger, stage: str):
    try:
        outs = model.test_step(data_batch)
        logits = outs.logits.detach()
        return decode_next_token(tokenizer, logits)
    except Exception as exc:
        logger.warning(f"{stage} test_step failed, skip token validation: {exc}")
        return None, None


def _reset_kv_cache_buffers(model: XHGlmOcrLLMModel):
    for cache_name in ("past_key_caches", "past_value_caches"):
        caches = getattr(model, cache_name, None)
        if caches is None:
            continue
        for cache in caches:
            if isinstance(cache, Tensor):
                cache.zero_()


def _move_graph_tensor_constants_(graph, device, dtype=None) -> int:
    target_device = torch.device(device)
    moved = 0
    for module in graph.modules():
        for name, value in list(getattr(module, "_buffers", {}).items()):
            if isinstance(value, Tensor):
                need_move = value.device != target_device
                need_cast = dtype is not None and value.is_floating_point() and value.dtype != dtype
                if need_move or need_cast:
                    new_val = value.to(device=target_device, dtype=dtype if need_cast else value.dtype)
                    module._buffers[name] = new_val
                    moved += 1

        buffer_names = set(getattr(module, "_buffers", {}).keys())
        for name, value in list(module.__dict__.items()):
            if name in buffer_names:
                continue
            if isinstance(value, Tensor):
                need_move = value.device != target_device
                need_cast = dtype is not None and value.is_floating_point() and value.dtype != dtype
                if need_move or need_cast:
                    new_val = value.to(device=target_device, dtype=dtype if need_cast else value.dtype)
                    setattr(module, name, new_val)
                    moved += 1
    return moved


def _prepare_ptq_calibration_batches(glm_ocr_llm_model: XHGlmOcrLLMModel, data_prefill: dict, logger):
    _reset_kv_cache_buffers(glm_ocr_llm_model)
    glm_ocr_llm_model.rope_deltas = None

    prefill_calib = glm_ocr_llm_model.prepare_inputs_for_graph(data_prefill)
    prefill_calib = _flatten_args(prefill_calib)
    calib_batches = [prefill_calib]

    try:
        prompt_len = int(data_prefill["input_ids"].shape[-1])
        decode_data = {
            "input_ids": data_prefill["input_ids"][:, :1],
            "past_seq_length": prompt_len,
        }
        decode_calib = glm_ocr_llm_model.prepare_inputs_for_graph(decode_data)
        decode_calib = _flatten_args(decode_calib)
        calib_batches.append(decode_calib)
        logger.info(f"PTQ calibration batches prepared: {len(calib_batches)} (prefill + decode)")
    except Exception as exc:
        logger.warning(f"Build decode calibration batch failed, fallback to prefill only: {exc}")
        logger.info(f"PTQ calibration batches prepared: {len(calib_batches)} (prefill only)")

    return calib_batches


def _prepare_image_embeds(native_model, inputs, execution_device, dtype):
    visual = native_model.model.visual
    visual.to(execution_device)
    visual.to(dtype)

    with torch.no_grad():
        image_embeds = visual(
            inputs["pixel_values"].to(execution_device).to(dtype),
            grid_thw=inputs["image_grid_thw"].to(execution_device),
            return_dict=True,
        ).pooler_output
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        image_embeds = image_embeds.to(execution_device).to(dtype)
    return image_embeds


def _generate_hmonnx_golden(onnx_file: str, inputs, golden_dir: Path, execution_device, logger, tag: str):
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_model = HMONNXGoldenInference(onnx_file)
    golden_model.save_golden = True
    golden_model.golden_dir = str(golden_dir)
    golden_model.step = 0
    golden_model.to("cpu")
    golden_model.exec_device = execution_device

    golden_inputs = []
    for item in inputs:
        if isinstance(item, Tensor):
            value = item.to(execution_device)
            if value.is_floating_point():
                value = value.half()
            golden_inputs.append(value)
        else:
            golden_inputs.append(item)
    with torch.no_grad():
        golden_model.forward(*golden_inputs)
    logger.info(f"{tag} golden generated at: {golden_dir}")


def _prepare_exported_graph_inputs(model: XHGlmOcrLLMModel, data: dict):
    prepared = model.prepare_inputs_for_graph(data)
    flat = _flatten_args(prepared)
    if len(flat) >= 4 and isinstance(flat[0], Tensor) and isinstance(flat[3], Tensor):
        exported_input_length = int(flat[0].shape[1])
        if int(flat[3].reshape(-1)[0].item()) != exported_input_length:
            flat[3] = torch.tensor([exported_input_length], dtype=flat[3].dtype, device=flat[3].device)
    return flat


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    execution_device,
    dtype,
    logger,
    valid: bool = True,
):
    logger = get_root_logger()

    xh_model.to("cpu")
    torch.cuda.empty_cache()

    logger.info("Start exporting graph.............")
    with TimeProfiler("export graph"):
        xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exported graph.")

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(execution_device)
        xh_model.to(dtype)
        xh_model.set_exec_device(execution_device)
        try:
            with torch.no_grad():
                outs = xh_model.test_step(data_batch)
                logits = outs.logits.detach()
                next_tokens, next_token_str = decode_next_token(tokenizer, logits)
            logger.info(f"Exported model next token: {next_tokens} {next_token_str}")
        except Exception as exc:
            logger.warning(f"exported test_step failed, skip token validation: {exc}")

        xh_model.to("cpu")
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting onnx ***************")
    with TimeProfiler("export onnx"):
        onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _export_impl(cfg, args):
    logger = get_root_logger()
    is_valid_model = args.valid
    config_file = cfg.config_file
    cfg_name = cfg.cfg_name

    device = torch.device(cfg.device)
    execution_device = torch.device(cfg.execution_device)
    dtype = getattr(torch, cfg.dtype)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(cfg.work_dir)),
        )
    )

    model_cfg = ConfigDict(cfg.model.copy())
    target_w = int(model_cfg.pop("image_size_w", 672))
    target_h = int(model_cfg.pop("image_size_h", 672))

    glm_ocr_llm_model: XHGlmOcrLLMModel = MODELS.build(model_cfg)
    native_model = glm_ocr_llm_model.get_hf_model(attn_implementation=args.attn_implementation)

    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    _copy_hf_configs(Path(hf_model_dir), hf_config_dir, logger)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = native_model.model.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding, str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    processor = GlmOcrProcessor.from_pretrained(hf_model_dir)
    image = _load_and_process_image(args.image_path, target_w=target_w, target_h=target_h)
    messages = build_messages(image, args.prompt)
    inputs = build_inputs(processor, messages, device=execution_device)
    tokenizer = processor.tokenizer

    _fix_image_token_id_if_needed(native_model.config, inputs["input_ids"], inputs["image_grid_thw"], logger)
    cfg_eos_token_id = getattr(native_model.config, "eos_token_id", None)
    if cfg_eos_token_id is None:
        cfg_eos_token_id = getattr(native_model.generation_config, "eos_token_id", None)
    stop_token_ids = _build_stop_token_ids(processor, cfg_eos_token_id)

    native_model.to(execution_device)  # pyright: ignore[reportArgumentType]
    native_model.to(dtype)

    image_embeds = _prepare_image_embeds(native_model, inputs, execution_device, dtype)
    native_model.cpu()

    glm_ocr_llm_model.init_wrap_model(native_model)
    del native_model

    glm_ocr_llm_model.change_eval_type(EvalModelType.WRAPED)
    glm_ocr_llm_model.token_embedding = torch.load(token_embedding_file, weights_only=False)

    xh2a_hf_compatible_model = None
    native_model = glm_ocr_llm_model.get_hf_model(attn_implementation=args.attn_implementation)
    xh2a_hf_compatible_model = GlmOcrHFCompatible.to_hf_compatible(native_model, glm_ocr_llm_model)
    xh2a_hf_compatible_model.to(execution_device)
    xh2a_hf_compatible_model.to(dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if glm_ocr_llm_model.past_key_caches is not None and len(glm_ocr_llm_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = glm_ocr_llm_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(glm_ocr_llm_model.past_key_caches)

    data_prefill = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "image_embeds": image_embeds,
        "past_seq_length": 0,
        "image_grid_thw": inputs["image_grid_thw"],
    }
    profile_root = Path(args.profile_dir) if args.profile_dir is not None else Path(cfg.work_dir) / "node_profile"

    glm_ocr_llm_model.change_eval_type(EvalModelType.WRAPED)
    glm_ocr_llm_model.to(dtype)
    glm_ocr_llm_model.to(device)

    glm_ocr_llm_model.cpu()
    logger.info("************* convert to frontend graph *************")
    glm_ocr_llm_model.convert_to_fronted_graph(data_prefill, release_wraped_model=False)

    wraped_output_text = None
    if is_valid_model and xh2a_hf_compatible_model is not None:
        glm_ocr_llm_model.to(execution_device)
        glm_ocr_llm_model.to(dtype)
        wraped_output_text = _safe_generate_text(
            xh2a_hf_compatible_model,
            inputs,
            processor,
            inputs["input_ids"],
            args.max_new_tokens,
            logger,
            "wraped",
            stop_token_ids,
        )
        if wraped_output_text is not None:
            logger.info("***************** wraped model output *****************")
            logger.info(wraped_output_text)

    if is_valid_model and xh2a_hf_compatible_model is not None:
        glm_ocr_llm_model.change_eval_type(EvalModelType.FRONTEND)
        glm_ocr_llm_model.to(execution_device)
        glm_ocr_llm_model.to(dtype)
        moved = _move_graph_tensor_constants_(glm_ocr_llm_model.frontend_model, execution_device, dtype=dtype)
        if moved > 0:
            logger.info(f"Moved {moved} frontend graph tensor constants to {execution_device}.")
        frontend_output_text = _safe_generate_text(
            xh2a_hf_compatible_model,
            inputs,
            processor,
            inputs["input_ids"],
            args.max_new_tokens,
            logger,
            "frontend-traced",
            stop_token_ids,
        )
        if frontend_output_text is not None:
            logger.info("***************** frontend traced model output *****************")
            logger.info(frontend_output_text)

        glm_ocr_llm_model.change_eval_type(EvalModelType.WRAPED)

    if args.profile_nodes:
        from glm_ocr_profile_nodes import profile_traced_quanted_graph_nodes
        profile_traced_quanted_graph_nodes(
            glm_ocr_llm_model=glm_ocr_llm_model,
            data_prefill=data_prefill,
            profile_root=profile_root,
            execution_device=execution_device,
            dtype=dtype,
            target_device=cfg.target_device,
            logger=logger,
            decode_steps=args.profile_decode_steps,
            tokenizer=tokenizer,
        )

    logger.info("************* convert to quanted graph *************")
    glm_ocr_llm_model.convert_to_quant_graph(cfg.target_device)

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    glm_ocr_llm_model.to(execution_device)
    glm_ocr_llm_model.to(dtype)
    moved = _move_graph_tensor_constants_(glm_ocr_llm_model.quanted_model, execution_device, dtype=dtype)
    if moved > 0:
        logger.info(f"Moved {moved} quanted graph tensor constants to {execution_device} before PTQ.")

    logger.info("*************** Start PTQ Quantize ***************")
    calib_batches = _prepare_ptq_calibration_batches(glm_ocr_llm_model, data_prefill, logger)
    ptq_quantize(glm_ocr_llm_model.quanted_model, calib_batches, PrecisionMode.ALIGNED, [execution_device])
    logger.info("*************** Finished PTQ Quantize ***************")

    moved = _move_graph_tensor_constants_(glm_ocr_llm_model.quanted_model, execution_device, dtype=dtype)
    if moved > 0:
        logger.info(f"Moved {moved} quanted graph tensor constants to {execution_device} after PTQ.")

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    glm_ocr_llm_model.to(execution_device)
    glm_ocr_llm_model.to(dtype)

    if is_valid_model and xh2a_hf_compatible_model is not None:
        quanted_aligned_output_text = _safe_generate_text(
            xh2a_hf_compatible_model,
            inputs,
            processor,
            inputs["input_ids"],
            args.max_new_tokens,
            logger,
            "quanted-aligned",
            stop_token_ids,
        )
        if quanted_aligned_output_text is not None:
            logger.info("***************** quanted (aligned) model output *****************")
            logger.info(quanted_aligned_output_text)

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    logger.info("*************** Start exporting prefill model ***************")
    prefill_onnx_file = xhmodel_export_onnx(
        glm_ocr_llm_model,
        tokenizer,
        data_prefill,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        execution_device,
        dtype,
        logger,
        is_valid_model,
    )
    glm_ocr_llm_model.release_exported_model()
    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished export prefill model ***************")

    golden_root = Path(args.golden_dir) if args.golden_dir is not None else Path(cfg.work_dir) / "golden"
    if not args.skip_golden:
        prefill_golden_inputs = _prepare_exported_graph_inputs(glm_ocr_llm_model, data_prefill)
        _generate_hmonnx_golden(
            prefill_onnx_file,
            prefill_golden_inputs,
            golden_root / "prefill",
            execution_device,
            logger,
            "Prefill",
        )

    glm_ocr_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    glm_ocr_llm_model.set_input_sequence_length(1)

    data_decode = {
        "input_ids": torch.randint(0, 1000, (1, 1)),
        "past_seq_length": int(data_prefill["input_ids"].shape[-1]),
    }

    if is_valid_model:
        glm_ocr_llm_model.to(execution_device)
        next_token_id, next_token_text = _safe_test_next_token(
            glm_ocr_llm_model, tokenizer, data_decode, logger, "decode-quanted"
        )
        if next_token_id is not None:
            logger.info(f"Decode Quanted Model next token: {next_token_id} {next_token_text}")

    torch.cuda.empty_cache()
    logger.info("*************** Start exporting decode model ***************")
    with TimeProfiler("export decode onnx", logger):
        decode_onnx_file = xhmodel_export_onnx(
            glm_ocr_llm_model,
            tokenizer,
            data_decode,
            str(decode_onnx_dir),
            f"{cfg_name}_decode",
            execution_device,
            dtype,
            logger,
            is_valid_model,
        )
    glm_ocr_llm_model.release_exported_model()
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    logger.info("*************** Finished export decode model ***************")

    if not args.skip_golden:
        decode_golden_inputs = _prepare_exported_graph_inputs(glm_ocr_llm_model, data_decode)
        _generate_hmonnx_golden(
            decode_onnx_file,
            decode_golden_inputs,
            golden_root / "decode",
            execution_device,
            logger,
            "Decode",
        )

    torch.save(
        {
            "input_ids": inputs["input_ids"].cpu(),
            "attention_mask": inputs["attention_mask"].cpu(),
            "image_grid_thw": inputs["image_grid_thw"].cpu(),
            "image_embeds": image_embeds.cpu(),
            "decode_input_ids": data_decode["input_ids"].cpu(),
            "decode_past_seq_length": int(data_decode["past_seq_length"]),
        },
        Path(cfg.work_dir) / "export_samples.pt",
    )

    patch_size = int(processor.image_processor.patch_size)
    meta_info.image_size_w = int(inputs["image_grid_thw"][0, 2].item()) * patch_size
    meta_info.image_size_h = int(inputs["image_grid_thw"][0, 1].item()) * patch_size
    meta_info.model_type = "glm_ocr"

    meta_file = str(Path(cfg.work_dir) / "export_meta_info.json")
    json.dump(meta_info, open(meta_file, "w"), indent=4)
    logger.info(f"Save meta info to {meta_file}")


def main(args):
    from types import SimpleNamespace

    work_dir = args.work_dir
    cfg_name = "glm_ocr_llm_xh2a_2k_export"

    # Build quant_config and model dict inline (no external config file)
    quant_config = dict(
        inputs=dict(
            inputs_embeds=dict(quantizer=dict(qspec=dict(fake_dtype="float16"))),
            past_seq_length=dict(quantizer=dict(qspec=dict(fake_dtype="int32"))),
            current_input_length=dict(quantizer=dict(qspec=dict(fake_dtype="int32"))),
            position_ids=dict(quantizer=dict(qspec=dict(fake_dtype="float16"))),
        ),
        w_schema=dict(bits=8, fp_mode="sefp"),
        act_schema=dict(bits=16, fp_mode="sefp"),
        nodes_cfg=dict(
            lm_head=dict(
                w_schema=dict(bits=8, fp_mode="sefp"),
                act_schema=dict(bits=8, fp_mode="sefp"),
            )
        ),
    )

    model_dict = dict(
        type="XHGlmOcrLLMModel",
        hf_model=args.hf_model_dir,
        wrap_cfg=dict(
            max_sequence_length=args.max_sequence_length,
            input_sequence_length=args.input_sequence_length,
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(cache_axis=2),
        ),
        quant_config=quant_config,
        frontend_type="TorchFX",
        export_cfg=dict(
            input_names=["inputs_embeds", "position_ids", "past_seq_length", "current_input_length"],
            output_names=["logits"],
        ),
        image_size_w=args.image_size_w,
        image_size_h=args.image_size_h,
    )

    # Construct a SimpleNamespace to keep _export_impl interface stable
    cfg = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        execution_device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype="float16",
        debug=args.debug,
        seed=args.seed,
        work_dir=work_dir,
        hf_model_dir=args.hf_model_dir,
        target_device=args.target_device,
        model=model_dict,
        config_file=str(Path(work_dir) / f"{cfg_name}_runtime.py"),
        cfg_name=cfg_name,
    )

    log_file = Path(work_dir) / f"{cfg_name}_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)

    set_random_seed(args.seed)

    xhquant_llm_init(log_file, args.debug)
    logger = get_root_logger()

    xhquant.utils.suppress_printing.disable_printing = True

    logger.info(f"Args: {args}")
    with TimeProfiler(f"{cfg_name} export", logger), MemoryTracker(0, "export", logger):
        _export_impl(cfg, args)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
