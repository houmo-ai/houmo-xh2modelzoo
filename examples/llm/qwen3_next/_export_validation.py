import gc
from copy import deepcopy
from typing import Dict, List, Optional

import accelerate
import torch
import torch.nn as nn
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from xhquant.api import PrecisionMode, ptq_quantize

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType


_GIB = 1024**3


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _flatten_inputs(inputs):
    flat = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _max_abs_diff(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    valid_len: int,
    num_logits_to_keep: int,
) -> float:
    if num_logits_to_keep == 0:
        return (lhs[:, :valid_len, :] - rhs[:, :valid_len, :]).abs().max().item()
    return (lhs - rhs).abs().max().item()


def _format_dtype(dtype) -> str:
    if dtype is None:
        return "None"
    return str(dtype).replace("torch.", "")


def _format_bytes_as_gib_str(num_bytes: int) -> str:
    gib = num_bytes / _GIB
    return f"{gib:.1f}GiB"


def _normalize_device_map_to_modules(model: nn.Module, device_map):
    module_names = set(dict(model.named_modules()).keys())
    normalized_device_map = {}

    for key, device in device_map.items():
        if key == "" or key in module_names:
            normalized_key = key
        else:
            parts = key.split(".")
            normalized_key = ""
            while len(parts) > 0:
                candidate = ".".join(parts)
                if candidate in module_names:
                    normalized_key = candidate
                    break
                parts.pop()

        existing_device = normalized_device_map.get(normalized_key, None)
        if existing_device is not None and existing_device != device:
            continue
        normalized_device_map[normalized_key] = device

    return normalized_device_map


def _resolve_no_split_module_classes(model: nn.Module, no_split_module_classes):
    if no_split_module_classes is None:
        return []
    if not isinstance(no_split_module_classes, (tuple, list)):
        no_split_module_classes = [no_split_module_classes]

    original_class_names = list(no_split_module_classes)
    found_class_names = []
    all_module_class_names = set()

    for _, module in model.named_modules():
        module_class_name = module.__class__.__name__
        all_module_class_names.add(module_class_name)
        mro_names = [cls.__name__ for cls in module.__class__.__mro__[:]]
        for cls_name in original_class_names:
            if cls_name in mro_names:
                found_class_names.append(module_class_name)

    found_class_names = list(set(found_class_names))
    if found_class_names:
        return found_class_names

    partial_matches = []
    for cls_name in original_class_names:
        core_name = cls_name
        if cls_name.startswith("Q") and len(cls_name) > 1:
            core_name = cls_name[1:]
        elif cls_name.startswith("Quant"):
            core_name = cls_name[5:]

        for module_class_name in all_module_class_names:
            if core_name.lower() in module_class_name.lower():
                partial_matches.append(module_class_name)

    if partial_matches:
        return list(set(partial_matches))
    return original_class_names


def _auto_offload(
    model: nn.Module,
    no_split_module_classes=None,
    target_dtype: torch.dtype = torch.float16,
    max_memory=None,
):
    resolved_no_split_module_classes = _resolve_no_split_module_classes(model, no_split_module_classes)
    device_map_kwargs = {"no_split_module_classes": resolved_no_split_module_classes}
    max_memory = get_balanced_memory(
        model,
        dtype=target_dtype,
        low_zero=False,
        max_memory=max_memory,
        **device_map_kwargs,
    )
    device_map_kwargs["max_memory"] = max_memory
    device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)
    device_map = _normalize_device_map_to_modules(model, device_map)
    dispatch_model(model, device_map=device_map, offload_buffers=True)


def _remove_offload(module):
    accelerate.hooks.remove_hook_from_module(module, recurse=True)
    if hasattr(module, "hf_device_map"):
        delattr(module, "hf_device_map")


def _build_gpu_only_validation_graph_max_memory(logger) -> Optional[Dict[int, str]]:
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        return None

    reserve_bytes = 12 * _GIB
    min_budget_bytes = 20 * _GIB
    max_memory: Dict[int, str] = {}
    debug_budgets = []

    for gpu_idx in range(num_gpus):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
        usable_bytes = min(free_bytes, max(total_bytes - reserve_bytes, min_budget_bytes))
        usable_bytes = max(usable_bytes, min_budget_bytes)
        planner_bytes = usable_bytes * 2 if gpu_idx == 0 else usable_bytes
        max_memory[gpu_idx] = _format_bytes_as_gib_str(planner_bytes)
        debug_budgets.append(
            f"cuda:{gpu_idx}: usable={_format_bytes_as_gib_str(usable_bytes)}, "
            f"planner={max_memory[gpu_idx]}, free={_format_bytes_as_gib_str(free_bytes)}"
        )

    logger.info(
        "Validation graph auto_offload_max_memory (GPU-only): "
        + "; ".join(debug_budgets)
    )
    return max_memory


def _run_with_first_layer_dtype_capture(model, fn):
    captured_dtype = [None]

    def hook(module, args, kwargs):
        if args:
            tensor = args[0]
        elif kwargs:
            tensor = next(iter(kwargs.values()))
        else:
            return
        if isinstance(tensor, torch.Tensor):
            captured_dtype[0] = tensor.dtype

    target = None
    for _, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "DecoderLayer" in cls_name or "Block" in cls_name:
            target = module
            break

    handle = None
    if target is not None:
        handle = target.register_forward_pre_hook(hook, with_kwargs=True)

    result = fn()

    if handle is not None:
        handle.remove()

    return result, captured_dtype[0]


def _offload_wrap_model(qwen3_next_model, no_split_modules: List[str], dtype: torch.dtype, logger):
    if qwen3_next_model.wrap_model is None:
        raise RuntimeError("wrap_model is None before auto_offload")
    logger.info(
        f"Dispatching wrap model via auto_offload, no_split_modules={no_split_modules}, "
        f"target_dtype={_format_dtype(dtype)}"
    )
    _auto_offload(
        qwen3_next_model.wrap_model,
        no_split_module_classes=no_split_modules,
        target_dtype=dtype,
        max_memory=qwen3_next_model.wrap_cfg.get("auto_offload_max_memory", None),
    )


def _de_offload_wrap_model(qwen3_next_model, logger):
    if qwen3_next_model.wrap_model is not None:
        logger.info("Removing auto_offload hooks, moving wrap model to CPU")
        _remove_offload(qwen3_next_model.wrap_model)
    qwen3_next_model.to("cpu")
    cleanup_memory()


def _set_validation_graph_auto_offload_max_memory(qwen3_next_model, logger):
    original_max_memory = qwen3_next_model.wrap_cfg.get("auto_offload_max_memory", None)
    validation_max_memory = _build_gpu_only_validation_graph_max_memory(logger)
    if validation_max_memory is not None:
        qwen3_next_model.wrap_cfg.auto_offload_max_memory = validation_max_memory
    return original_max_memory


def _restore_validation_graph_auto_offload_max_memory(qwen3_next_model, original_max_memory):
    qwen3_next_model.wrap_cfg.auto_offload_max_memory = original_max_memory


def _de_dispatch_to_cpu(model):
    _remove_offload(model)
    model.to("cpu")
    cleanup_memory()


def _clone_tensor_to_cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().to("cpu").clone()


def _clone_tensor_list_to_cpu(tensors) -> Optional[List[torch.Tensor]]:
    if tensors is None:
        return None
    return [_clone_tensor_to_cpu(tensor) for tensor in tensors]


def _capture_generation_state(qwen3_next_model) -> Dict[str, Optional[torch.Tensor]]:
    return {
        "past_key_caches": _clone_tensor_list_to_cpu(getattr(qwen3_next_model, "past_key_caches", None)),
        "past_value_caches": _clone_tensor_list_to_cpu(getattr(qwen3_next_model, "past_value_caches", None)),
        "past_conv_caches": _clone_tensor_list_to_cpu(getattr(qwen3_next_model, "past_conv_caches", None)),
        "past_recurrent_states": _clone_tensor_list_to_cpu(getattr(qwen3_next_model, "past_recurrent_states", None)),
    }


def _restore_generation_state(qwen3_next_model, state_snapshot):
    for attr_name in [
        "past_key_caches",
        "past_value_caches",
        "past_conv_caches",
        "past_recurrent_states",
    ]:
        saved_tensors = state_snapshot.get(attr_name, None)
        target_tensors = getattr(qwen3_next_model, attr_name, None)
        if saved_tensors is None:
            continue
        if target_tensors is None or len(target_tensors) != len(saved_tensors):
            raise RuntimeError(
                f"State migration failed for {attr_name}: expected {len(saved_tensors)} tensors, "
                f"got {0 if target_tensors is None else len(target_tensors)}"
            )
        for idx, saved_tensor in enumerate(saved_tensors):
            restored_tensor = saved_tensor.to(
                device=target_tensors[idx].device,
                dtype=target_tensors[idx].dtype,
            )
            target_tensors[idx].copy_(restored_tensor)


def _build_prefill_batch(
    input_ids: torch.Tensor,
    valid_len: int,
    input_sequence_length: int,
    pad_token_id: int,
) -> Dict[str, object]:
    input_ids_pad = torch.cat(
        [
            input_ids.cpu(),
            torch.full(
                (input_ids.shape[0], input_sequence_length - valid_len),
                pad_token_id,
                dtype=torch.long,
            ),
        ],
        dim=-1,
    )
    return {
        "input_ids": input_ids_pad,
        "past_seq_length": [0],
        "current_input_length": [valid_len],
    }


def _build_isolated_validation_model(cfg, dtype: torch.dtype, logger, stage_name: str):
    model_cfg = deepcopy(cfg.model)
    isolated_model = MODELS.build(model_cfg)
    cleanup_memory()
    logger.info(f"[{stage_name}] Loading HF model with device_map='cpu' for isolated full-output validation")
    native_model = isolated_model.get_hf_model(
        device_map="cpu",
        use_safetensors=True,
        torch_dtype=dtype,
    )
    _de_dispatch_to_cpu(native_model)
    isolated_model.init_wrap_model(native_model)
    del native_model

    isolated_model.change_eval_type(EvalModelType.WRAPED)
    isolated_model.set_exec_device(torch.device("cpu"))
    isolated_model.to("cpu")
    isolated_model.to(dtype)
    _set_validation_graph_auto_offload_max_memory(isolated_model, logger)
    return isolated_model


def _release_isolated_validation_model(qwen3_next_model):
    if qwen3_next_model is None:
        return
    from xhquant.xhonnxruntime import AutoOffloadGraphModel

    # Remove AutoOffloadGraphModel wrappers and move sub-models to CPU
    # (_frontend_model / _quanted_model are NOT registered submodules,
    #  so parent .to("cpu") won't reach them)
    for attr in ("_frontend_model", "_quanted_model"):
        sub = getattr(qwen3_next_model, attr, None)
        if sub is None:
            continue
        if AutoOffloadGraphModel.is_auto_offload_model(sub):
            AutoOffloadGraphModel.remove_auto_offload_model(sub)
        sub.to("cpu")
    qwen3_next_model.set_exec_device(torch.device("cpu"))
    qwen3_next_model.to("cpu")

    qwen3_next_model.release_exported_model()
    qwen3_next_model.release_quanted_model()
    qwen3_next_model.release_frontend_model()
    qwen3_next_model.release_wraped_model()
    del qwen3_next_model
    cleanup_memory()


def _prepare_isolated_stage_model(
    qwen3_next_model,
    cfg,
    dtype: torch.dtype,
    gpu_device: torch.device,
    stage_eval_type,
    trace_batch,
    logger,
):
    from xhquant.xhonnxruntime import AutoOffloadGraphModel

    qwen3_next_model.set_exec_device(torch.device("cpu"))
    qwen3_next_model.to("cpu")

    if stage_eval_type == EvalModelType.FRONTEND:
        qwen3_next_model.convert_to_fronted_graph(trace_batch, release_wraped_model=False)
        qwen3_next_model.change_eval_type(EvalModelType.FRONTEND)
        qwen3_next_model.to(dtype)
        validation_max_memory = _build_gpu_only_validation_graph_max_memory(logger)
        qwen3_next_model._frontend_model = AutoOffloadGraphModel.from_graph_model(
            qwen3_next_model._frontend_model, max_memory=validation_max_memory,
        )
        qwen3_next_model.set_exec_device(gpu_device)
        return

    if stage_eval_type != EvalModelType.QUANTED_ALIGNED:
        raise ValueError(f"Unsupported isolated stage eval type: {stage_eval_type}")

    qwen3_next_model.convert_to_fronted_graph(trace_batch, release_wraped_model=False)
    qwen3_next_model.convert_to_quant_graph(cfg.target_device)
    calib_data = _flatten_inputs(qwen3_next_model.prepare_inputs_for_graph(trace_batch))
    ptq_quantize(
        qwen3_next_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [gpu_device],
        auto_release_unused_parameters=True,
    )
    # ptq_quantize puts weights on GPU; move back to CPU so GPU is clean
    # before AutoOffloadGraphModel distributes across GPUs
    qwen3_next_model._quanted_model.to("cpu")
    cleanup_memory()

    qwen3_next_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    qwen3_next_model.to(dtype)
    validation_max_memory = _build_gpu_only_validation_graph_max_memory(logger)
    qwen3_next_model._quanted_model = AutoOffloadGraphModel.from_graph_model(
        qwen3_next_model._quanted_model, max_memory=validation_max_memory,
    )
    qwen3_next_model.set_exec_device(gpu_device)


@torch.no_grad()
def collect_hf_references(
    native_model,
    tokenizer,
    input_ids: torch.Tensor,
    gpu_device: torch.device,
    dtype: torch.dtype,
    num_logits_to_keep: int,
    compare_full_output: bool,
    compare_max_new_tokens: int,
    logger,
) -> Dict[str, Optional[torch.Tensor]]:
    refs: Dict[str, Optional[torch.Tensor]] = {
        "hf_logits": None,
        "hf_generated_ids": None,
    }
    native_model.eval()
    hf_out, hf_runtime_dtype = _run_with_first_layer_dtype_capture(
        native_model,
        lambda: native_model(input_ids=input_ids.to(gpu_device), use_cache=True),
    )
    logger.info(
        f"HF runtime dtype: first_decoder_input={_format_dtype(hf_runtime_dtype)}, "
        f"logits={_format_dtype(hf_out.logits.dtype)}"
    )
    if hf_runtime_dtype is not None and hf_runtime_dtype != dtype:
        logger.warning(
            f"HF runtime first decoder input dtype mismatch: "
            f"observed={_format_dtype(hf_runtime_dtype)}, target={_format_dtype(dtype)}"
        )
    if num_logits_to_keep == 0:
        refs["hf_logits"] = hf_out.logits.detach().cpu()
    else:
        refs["hf_logits"] = hf_out.logits[:, -1:, :].detach().cpu()
    del hf_out

    if compare_full_output and compare_max_new_tokens > 0:
        generated = native_model.generate(
            input_ids=input_ids.to(gpu_device),
            max_new_tokens=compare_max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        refs["hf_generated_ids"] = generated[:, input_ids.shape[1] :].detach().cpu()
    return refs


@torch.no_grad()
def _generate_full_output(
    qwen3_next_model,
    tokenizer,
    input_ids: torch.Tensor,
    valid_len: int,
    input_sequence_length: int,
    device: torch.device,
    max_new_tokens: int,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return torch.empty((1, 0), dtype=torch.long)
    original_input_sequence_length = qwen3_next_model.get_input_sequence_length()
    original_linear_attention_mode = qwen3_next_model.wrap_cfg.linear_attention_mode
    try:
        qwen3_next_model.reset_kvcache()
        qwen3_next_model.set_linear_attention_mode("chunk")
        qwen3_next_model.set_input_sequence_length(input_sequence_length)

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids_pad = torch.cat(
            [
                input_ids,
                torch.full((input_ids.shape[0], input_sequence_length - valid_len), pad_token_id, dtype=torch.long),
            ],
            dim=-1,
        )
        data_batch_prefill = {
            "input_ids": input_ids_pad.to(device),
            "past_seq_length": [0],
            "current_input_length": [valid_len],
        }
        prefill_out = qwen3_next_model.test_step(data_batch_prefill)
        prefill_logits = prefill_out.logits.detach()
        del prefill_out
        next_token = prefill_logits[:, -1, :].argmax(dim=-1).view(1, 1)
        generated_ids = [int(next_token[0, 0].item())]

        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None and generated_ids[-1] == eos_token_id:
            return torch.tensor([generated_ids], dtype=torch.long)

        qwen3_next_model.set_linear_attention_mode("recurrent")
        qwen3_next_model.set_input_sequence_length(1)
        for step in range(max_new_tokens - 1):
            data_batch_decode = {
                "input_ids": next_token.to(device),
                "past_seq_length": [valid_len + step],
                "current_input_length": [1],
            }
            decode_out = qwen3_next_model.test_step(data_batch_decode)
            decode_logits = decode_out.logits.detach()
            del decode_out
            next_token = decode_logits[:, -1, :].argmax(dim=-1).view(1, 1)
            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
        return torch.tensor([generated_ids], dtype=torch.long)
    finally:
        qwen3_next_model.reset_kvcache()
        qwen3_next_model.set_linear_attention_mode(original_linear_attention_mode)
        qwen3_next_model.set_input_sequence_length(original_input_sequence_length)


@torch.no_grad()
def _generate_full_output_with_isolated_instances(
    cfg,
    tokenizer,
    input_ids: torch.Tensor,
    valid_len: int,
    input_sequence_length: int,
    dtype: torch.dtype,
    gpu_device: torch.device,
    max_new_tokens: int,
    stage_eval_type,
    stage_name: str,
    logger,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return torch.empty((1, 0), dtype=torch.long)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    prefill_batch = _build_prefill_batch(input_ids, valid_len, input_sequence_length, pad_token_id)
    prefill_model = None
    decode_model = None
    state_snapshot = None

    try:
        prefill_model = _build_isolated_validation_model(cfg, dtype, logger, f"{stage_name}-prefill")
        prefill_model.reset_kvcache()
        prefill_model.set_linear_attention_mode("chunk")
        prefill_model.set_input_sequence_length(input_sequence_length)
        _prepare_isolated_stage_model(
            qwen3_next_model=prefill_model,
            cfg=cfg,
            dtype=dtype,
            gpu_device=gpu_device,
            stage_eval_type=stage_eval_type,
            trace_batch=prefill_batch,
            logger=logger,
        )

        prefill_out = prefill_model.test_step(prefill_batch)
        prefill_logits = prefill_out.logits.detach()
        del prefill_out
        next_token = prefill_logits[:, -1, :].argmax(dim=-1).view(1, 1).cpu()
        generated_ids = [int(next_token[0, 0].item())]
        state_snapshot = _capture_generation_state(prefill_model)
    finally:
        _release_isolated_validation_model(prefill_model)
        prefill_model = None
        cleanup_memory()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and generated_ids[-1] == eos_token_id:
        return torch.tensor([generated_ids], dtype=torch.long)

    try:
        decode_model = _build_isolated_validation_model(cfg, dtype, logger, f"{stage_name}-decode")
        decode_model.reset_kvcache()
        decode_model.set_linear_attention_mode("recurrent")
        decode_model.set_input_sequence_length(1)
        _restore_generation_state(decode_model, state_snapshot)

        decode_trace_batch = {
            "input_ids": next_token.clone(),
            "past_seq_length": [valid_len],
            "current_input_length": [1],
        }
        _prepare_isolated_stage_model(
            qwen3_next_model=decode_model,
            cfg=cfg,
            dtype=dtype,
            gpu_device=gpu_device,
            stage_eval_type=stage_eval_type,
            trace_batch=decode_trace_batch,
            logger=logger,
        )
        _restore_generation_state(decode_model, state_snapshot)

        for step in range(max_new_tokens - 1):
            data_batch_decode = {
                "input_ids": next_token.clone(),
                "past_seq_length": [valid_len + step],
                "current_input_length": [1],
            }
            decode_out = decode_model.test_step(data_batch_decode)
            decode_logits = decode_out.logits.detach()
            del decode_out
            next_token = decode_logits[:, -1, :].argmax(dim=-1).view(1, 1).cpu()
            token_id = int(next_token[0, 0].item())
            generated_ids.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
    finally:
        _release_isolated_validation_model(decode_model)

    return torch.tensor([generated_ids], dtype=torch.long)


def _log_stage_output(marker: str, token_ids: Optional[torch.Tensor], tokenizer, logger):
    if token_ids is None:
        return
    token_list = token_ids.view(-1).tolist()
    text = tokenizer.decode(token_list, skip_special_tokens=True)
    logger.info(f"{marker}: {text}")


def _log_full_output_comparison(
    lhs_name: str,
    lhs_ids: torch.Tensor,
    rhs_name: str,
    rhs_ids: torch.Tensor,
    tokenizer,
    logger,
):
    lhs = lhs_ids.view(-1).tolist()
    rhs = rhs_ids.view(-1).tolist()
    matched_len = 0
    max_len = min(len(lhs), len(rhs))
    for idx in range(max_len):
        if lhs[idx] != rhs[idx]:
            break
        matched_len += 1
    exact_match = len(lhs) == len(rhs) and matched_len == len(lhs)
    logger.info(
        f"{lhs_name} vs {rhs_name} full output token compare: "
        f"exact_match={exact_match}, {lhs_name}_len={len(lhs)}, {rhs_name}_len={len(rhs)}, "
        f"prefix_match_len={matched_len}"
    )
    if matched_len < max_len:
        logger.info(
            f"First mismatch index={matched_len}, "
            f"{lhs_name}_token={lhs[matched_len]}, {rhs_name}_token={rhs[matched_len]}"
        )
    lhs_text = tokenizer.decode(lhs, skip_special_tokens=True)
    rhs_text = tokenizer.decode(rhs, skip_special_tokens=True)
    text_match = lhs_text == rhs_text
    logger.info(f"{lhs_name} vs {rhs_name} full output text compare: exact_match={text_match}")
    if not text_match:
        logger.info(f"{lhs_name} text: {lhs_text}")
        logger.info(f"{rhs_name} text: {rhs_text}")


@torch.no_grad()
def run_conversion_validation(
    qwen3_next_model,
    cfg,
    args,
    tokenizer,
    input_ids: torch.Tensor,
    valid_len: int,
    input_sequence_length: int,
    dtype: torch.dtype,
    no_split_modules: List[str],
    gpu_device: torch.device,
    hf_logits: Optional[torch.Tensor],
    hf_generated_ids: Optional[torch.Tensor],
    logger,
):
    qwen3_next_model.change_eval_type(EvalModelType.WRAPED)
    qwen3_next_model.set_linear_attention_mode("chunk")
    _offload_wrap_model(qwen3_next_model, no_split_modules, dtype, logger)
    qwen3_next_model.set_exec_device(gpu_device)

    qwen3_next_model.set_input_sequence_length(valid_len)
    data_batch_check = {"input_ids": input_ids.to(gpu_device), "past_seq_length": [0]}
    wrap_out, wrap_runtime_dtype = _run_with_first_layer_dtype_capture(
        qwen3_next_model.wrap_model,
        lambda: qwen3_next_model.test_step(data_batch_check),
    )
    logger.info(
        f"Wrap runtime dtype: first_decoder_input={_format_dtype(wrap_runtime_dtype)}, "
        f"logits={_format_dtype(wrap_out.logits.dtype)}"
    )
    if wrap_runtime_dtype is not None and wrap_runtime_dtype != dtype:
        logger.warning(
            f"Wrap runtime first decoder input dtype mismatch: "
            f"observed={_format_dtype(wrap_runtime_dtype)}, target={_format_dtype(dtype)}"
        )
    wrap_logits = wrap_out.logits.detach().cpu()
    del wrap_out

    if hf_logits is not None:
        diff = (wrap_logits - hf_logits).abs().max().item()
        logger.info(f"Wrap vs HF (unpadded, len={valid_len}) max abs error: {diff:.6e}")

    qwen3_next_model.reset_kvcache()
    qwen3_next_model.set_input_sequence_length(input_sequence_length)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids_pad = torch.cat(
        [
            input_ids,
            torch.full((input_ids.shape[0], input_sequence_length - valid_len), pad_token_id, dtype=torch.long),
        ],
        dim=-1,
    )
    data_batch_pad = {
        "input_ids": input_ids_pad.to(gpu_device),
        "past_seq_length": [0],
        "current_input_length": [valid_len],
    }
    wrap_pad_out = qwen3_next_model.test_step(data_batch_pad)
    wrap_pad_logits = wrap_pad_out.logits.detach().cpu()
    del wrap_pad_out

    diff_mask = _max_abs_diff(wrap_pad_logits, wrap_logits, valid_len, args.num_logits_to_keep)
    logger.info(f"Mask pad vs unpad max abs error: {diff_mask:.6e}")

    compare_full_output = bool(getattr(args, "valid_compare_full_output", True))
    compare_max_new_tokens = int(getattr(args, "valid_compare_max_new_tokens", 64))
    wrap_generated_ids = None
    if compare_full_output and compare_max_new_tokens > 0:
        wrap_generated_ids = _generate_full_output(
            qwen3_next_model=qwen3_next_model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            valid_len=valid_len,
            input_sequence_length=input_sequence_length,
            device=gpu_device,
            max_new_tokens=compare_max_new_tokens,
        )
        _log_stage_output("wraped model output", wrap_generated_ids, tokenizer, logger)
        if hf_generated_ids is not None:
            _log_full_output_comparison("HF", hf_generated_ids, "Wrap", wrap_generated_ids, tokenizer, logger)

    # ── Wrap(pad) vs Frontend(pad) — CPU trace, GPU inference (following qwen3.5) ──
    from xhquant.xhonnxruntime import AutoOffloadGraphModel

    _de_offload_wrap_model(qwen3_next_model, logger)
    qwen3_next_model.set_exec_device(torch.device("cpu"))
    qwen3_next_model.reset_kvcache()
    qwen3_next_model.set_input_sequence_length(input_sequence_length)
    data_batch_pad_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data_batch_pad.items()}
    validation_max_memory = _build_gpu_only_validation_graph_max_memory(logger)
    qwen3_next_model.convert_to_fronted_graph(data_batch_pad_cpu, release_wraped_model=True)
    qwen3_next_model.change_eval_type(EvalModelType.FRONTEND)
    qwen3_next_model.to(dtype)
    # Load frontend model to GPU(s) for inference
    qwen3_next_model._frontend_model = AutoOffloadGraphModel.from_graph_model(
        qwen3_next_model._frontend_model, max_memory=validation_max_memory,
    )
    qwen3_next_model.set_exec_device(gpu_device)
    data_batch_pad_gpu = {
        k: v.to(gpu_device) if isinstance(v, torch.Tensor) else v
        for k, v in data_batch_pad_cpu.items()
    }
    frontend_out = qwen3_next_model.test_step(data_batch_pad_gpu)
    frontend_logits = frontend_out.logits.detach().cpu()
    del frontend_out

    diff_wrap_frontend = _max_abs_diff(wrap_pad_logits, frontend_logits, valid_len, args.num_logits_to_keep)
    logger.info(f"Wrap(pad) vs Frontend(pad) max abs error: {diff_wrap_frontend:.6e}")

    # ── Frontend(pad) vs Quant(pad) — CPU quant, GPU inference (following qwen3.5) ──
    qwen3_next_model.reset_kvcache()
    frontend_model = qwen3_next_model._frontend_model
    if AutoOffloadGraphModel.is_auto_offload_model(frontend_model):
        AutoOffloadGraphModel.remove_auto_offload_model(frontend_model)
    qwen3_next_model._frontend_model = frontend_model
    qwen3_next_model.set_exec_device(torch.device("cpu"))
    qwen3_next_model.to("cpu")
    qwen3_next_model.set_input_sequence_length(input_sequence_length)
    cleanup_memory()

    qwen3_next_model.convert_to_quant_graph(cfg.target_device)
    calib_data = _flatten_inputs(qwen3_next_model.prepare_inputs_for_graph(data_batch_pad_cpu))
    ptq_quantize(
        qwen3_next_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [gpu_device],
        auto_release_unused_parameters=True,
    )
    qwen3_next_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    qwen3_next_model.to(dtype)
    # Load quant model to GPU(s) for inference
    qwen3_next_model._quanted_model = AutoOffloadGraphModel.from_graph_model(
        qwen3_next_model._quanted_model, max_memory=validation_max_memory,
    )
    qwen3_next_model.set_exec_device(gpu_device)
    quant_out = qwen3_next_model.test_step(data_batch_pad_gpu)
    quant_logits = quant_out.logits.detach().cpu()
    del quant_out

    diff_frontend_quant = _max_abs_diff(frontend_logits, quant_logits, valid_len, args.num_logits_to_keep)
    logger.info(f"Frontend(pad) vs Quant(pad) max abs error: {diff_frontend_quant:.6e}")

    if compare_full_output and compare_max_new_tokens > 0:
        logger.info("Releasing main validation graphs before isolated full-output comparison")
        # Must remove AutoOffloadGraphModel wrappers and move to CPU before releasing,
        # otherwise GPU tensors won't be freed.
        for attr in ("_frontend_model", "_quanted_model"):
            sub = getattr(qwen3_next_model, attr, None)
            if sub is not None:
                if AutoOffloadGraphModel.is_auto_offload_model(sub):
                    AutoOffloadGraphModel.remove_auto_offload_model(sub)
                sub.to("cpu")
        qwen3_next_model.set_exec_device(torch.device("cpu"))
        qwen3_next_model.to("cpu")
        qwen3_next_model.change_eval_type(EvalModelType.NONE)
        qwen3_next_model.release_quanted_model()
        qwen3_next_model.release_frontend_model()
        qwen3_next_model.release_wraped_model()
        cleanup_memory()

        frontend_generated_ids = _generate_full_output_with_isolated_instances(
            cfg=cfg,
            tokenizer=tokenizer,
            input_ids=input_ids,
            valid_len=valid_len,
            input_sequence_length=input_sequence_length,
            dtype=dtype,
            gpu_device=gpu_device,
            max_new_tokens=compare_max_new_tokens,
            stage_eval_type=EvalModelType.FRONTEND,
            stage_name="frontend",
            logger=logger,
        )
        _log_stage_output("frontend traced model output", frontend_generated_ids, tokenizer, logger)
        if wrap_generated_ids is not None:
            _log_full_output_comparison("Wrap", wrap_generated_ids, "Frontend", frontend_generated_ids, tokenizer, logger)

        quant_generated_ids = _generate_full_output_with_isolated_instances(
            cfg=cfg,
            tokenizer=tokenizer,
            input_ids=input_ids,
            valid_len=valid_len,
            input_sequence_length=input_sequence_length,
            dtype=dtype,
            gpu_device=gpu_device,
            max_new_tokens=compare_max_new_tokens,
            stage_eval_type=EvalModelType.QUANTED_ALIGNED,
            stage_name="quant",
            logger=logger,
        )
        _log_stage_output("quanted (aligned) model output", quant_generated_ids, tokenizer, logger)
        _log_full_output_comparison("Frontend", frontend_generated_ids, "Quant", quant_generated_ids, tokenizer, logger)
    else:
        frontend_next = frontend_logits[:, -1, :].argmax(dim=-1).view(1, 1)
        quant_next = quant_logits[:, -1, :].argmax(dim=-1).view(1, 1)
        _log_stage_output("frontend traced model output", frontend_next, tokenizer, logger)
        _log_stage_output("quanted (aligned) model output", quant_next, tokenizer, logger)
        qwen3_next_model.release_wraped_model()
