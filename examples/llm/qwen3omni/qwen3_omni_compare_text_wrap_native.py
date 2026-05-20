import argparse
import json
import os.path as osp
import time
import types
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from xhquant.api import CacheTensor, Config

from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.qwen3_omni._text_model import register_wrap_modules as register_text_wrap_modules
from xh_model_zoo.xh_llm.models.qwen3_omni.qwen3_omni_converter import (
    Qwen3OmniMoeConverterXH2a,
    _install_qwen3omni_thinker_auto_class_compat,
)

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from qwen_omni_utils import process_mm_info
except ImportError:

    def process_mm_info(conversation, use_audio_in_video=False):
        audios, images, videos = [], [], []
        for turn in conversation:
            for item in turn.get("content", []):
                item_type = item.get("type")
                if item_type == "audio":
                    audios.append(item.get("audio"))
                elif item_type == "image":
                    images.append(item.get("image"))
                elif item_type == "video":
                    videos.append(item.get("video"))
        return audios, images, videos

try:
    from ._thinker_gptq_view import is_qwen3_omni_checkpoint, prepare_qwen3_omni_thinker_text_view
except ImportError:
    from _thinker_gptq_view import is_qwen3_omni_checkpoint, prepare_qwen3_omni_thinker_text_view


def build_conversation(case: str, text_prompt: Optional[str] = None):
    image_path = str(SCRIPT_DIR / "data" / "cars.jpg")
    audio_path = str(SCRIPT_DIR / "data" / "cough.wav")

    if case == "text":
        return [
            {"role": "user", "content": [{"type": "text", "text": text_prompt or "请用一句话介绍你自己。"}]},
        ], False
    if case == "vision":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt or "请描述这张图。"},
                ],
            },
        ], False
    if case == "audio":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": text_prompt or "请描述你听到了什么。"},
                ],
            },
        ], False
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": text_prompt or "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ], True


def _ensure_mistral_common_reasoning_effort():
    try:
        import mistral_common.protocol.instruct.request as request_module
    except ImportError:
        return

    if hasattr(request_module, "ReasoningEffort"):
        return

    class ReasoningEffort(str):
        none = "none"
        high = "high"

    request_module.ReasoningEffort = ReasoningEffort


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "auto":
        return "auto"
    raise ValueError(f"unsupported dtype: {name}")


def _parse_device_map(value: str):
    if value in ("none", "None", ""):
        return None
    if value in ("cuda", "cuda:0"):
        return {"": 0}
    if value.startswith("cuda:"):
        return {"": int(value.split(":", 1)[1])}
    return value


def _maybe_skip_transformers_cuda_allocator_warmup(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        return
    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def _first_parameter_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _move_batch_with_dtype(
    batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _resolve_accept_hidden_layer(model_dir: Path, model) -> int:
    config = getattr(model, "config", None)
    for owner in (config, getattr(config, "text_config", None), getattr(config, "talker_config", None)):
        value = getattr(owner, "accept_hidden_layer", None)
        if value is not None:
            return int(value)

    config_path = model_dir / "config.json"
    if config_path.exists():
        payload = _load_json(config_path)
        for section_name in ("talker_config", "thinker_config", "text_config"):
            section = payload.get(section_name, {})
            if isinstance(section, dict) and section.get("accept_hidden_layer") is not None:
                return int(section["accept_hidden_layer"])
        if payload.get("accept_hidden_layer") is not None:
            return int(payload["accept_hidden_layer"])

    raise ValueError("cannot resolve accept_hidden_layer from model/config; pass --accept-hidden-layer")


def _load_thinker_model(args, model_path: Path, work_dir: Path):
    torch_dtype = _torch_dtype(args.dtype)
    device_map = _parse_device_map(args.device_map)

    if args.load_kind == "text-view" or (
        args.load_kind == "auto" and is_qwen3_omni_checkpoint(str(model_path)) and args.force_text_view
    ):
        view_dir = work_dir / f"{model_path.name}-thinker-text-view"
        prepare_qwen3_omni_thinker_text_view(str(model_path), view_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(view_dir),
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        return model, view_dir, "text-view"

    if args.load_kind in ("auto", "thinker") and is_qwen3_omni_checkpoint(str(model_path)):
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        _install_qwen3omni_thinker_auto_class_compat()
        model = Qwen3OmniMoeConverterXH2a._load_model_from_pretrained(
            Qwen3OmniMoeThinkerForConditionalGeneration,
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        return model, model_path, "thinker"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    return model, model_path, "causal-lm"


def _pad_sequence_tensor(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = int(tensor.shape[1])
    if current_length == target_length:
        return tensor
    if current_length > target_length:
        raise ValueError(f"input length {current_length} exceeds input_sequence_length {target_length}")
    padding = torch.zeros(
        (tensor.shape[0], target_length - current_length, tensor.shape[2]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)


def _build_position_ids(actual_length: int, target_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position_ids = torch.arange(actual_length, dtype=dtype, device=device)
    if actual_length < target_length:
        position_ids = torch.cat(
            [position_ids, torch.zeros(target_length - actual_length, dtype=dtype, device=device)], dim=0
        )
    return position_ids


def _get_head_dim(model) -> int:
    first_layer = model.model.layers[0]
    head_dim = getattr(first_layer.self_attn, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    config = model.model.config
    return int(config.hidden_size // config.num_attention_heads)


def _get_hidden_size(model) -> int:
    return int(model.model.config.hidden_size)


def _pad_position_id_vector(
    position_ids: torch.Tensor,
    target_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position_ids = position_ids.to(device=device, dtype=dtype)
    current_length = int(position_ids.shape[0])
    if current_length == target_length:
        return position_ids
    if current_length > target_length:
        raise ValueError(f"position id length {current_length} exceeds input_sequence_length {target_length}")
    return torch.cat(
        [position_ids, torch.zeros(target_length - current_length, dtype=dtype, device=device)],
        dim=0,
    )


def _build_empty_caches(model, context_length: int, dtype: torch.dtype, device: torch.device):
    num_layers = int(model.model.config.num_hidden_layers)
    num_key_value_heads = int(model.model.config.num_key_value_heads)
    head_dim = _get_head_dim(model)
    kv_shape = [1, num_key_value_heads, context_length, head_dim]
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=dtype, device=device)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=dtype, device=device)) for _ in range(num_layers)]
    return kv_shape, past_key_caches, past_value_caches


def _force_moe_fallback(model) -> int:
    from xhquant.nn.modules.moeblock import MoeBlock

    patched = 0
    for module in model.modules():
        if not isinstance(module, MoeBlock):
            continue

        def _forward_fallback(self, hidden_states, routing_weights, selected_experts=None):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            out_dtype = hidden_states.dtype
            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            routing = routing_weights.reshape(flat_hidden.shape[0], -1).float()
            num_experts = int(self.expert_gate_proj_weight.shape[0])
            if getattr(self, "topk_outside", False):
                if selected_experts is None:
                    raise ValueError("selected_experts is required when MoeBlock.topk_outside=True")
                selected = selected_experts.reshape(flat_hidden.shape[0], -1).to(torch.long)
                if routing.shape[-1] == selected.shape[-1]:
                    topk_weights = routing.reshape(flat_hidden.shape[0], -1)
                else:
                    topk_weights = routing.gather(-1, selected)
            else:
                topk_weights, selected = torch.topk(routing, k=int(self.k), dim=-1)
            if self.normalize_routing_weights:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(out_dtype)

            final = torch.zeros_like(flat_hidden)
            active_experts = torch.unique(selected).tolist()
            for expert_idx in active_experts:
                expert_idx = int(expert_idx)
                if expert_idx < 0 or expert_idx >= num_experts:
                    continue
                token_idx, rank_idx = torch.where(selected == expert_idx)
                if token_idx.numel() == 0:
                    continue
                current_state = flat_hidden[token_idx]
                gate_bias = self.expert_gate_proj_bias
                up_bias = self.expert_up_proj_bias
                down_bias = self.expert_down_proj_bias
                gate_state = F.linear(
                    current_state,
                    self.expert_gate_proj_weight[expert_idx].to(out_dtype),
                    gate_bias[expert_idx].to(out_dtype) if gate_bias is not None else None,
                )
                if self.limit > 0:
                    gate_state = gate_state.clamp(min=None, max=self.limit)
                if "silu" in self.activation_type:
                    gate_state = F.silu(gate_state)
                elif "gelu" in self.activation_type:
                    gate_state = F.gelu(gate_state)
                elif self.activation_type in {"relu2", "relu_squared"}:
                    gate_state = F.relu(gate_state).pow(2)
                else:
                    gate_state = self.act_fn(gate_state)

                if self.expert_up_proj_constant_one:
                    mlp_state = gate_state
                else:
                    up_state = F.linear(
                        current_state,
                        self.expert_up_proj_weight[expert_idx].to(out_dtype),
                        up_bias[expert_idx].to(out_dtype) if up_bias is not None else None,
                    )
                    if self.limit > 0:
                        up_state = up_state.clamp(min=-self.limit, max=self.limit) + 1
                    mlp_state = gate_state * up_state
                down_state = F.linear(
                    mlp_state,
                    self.expert_down_proj_weight[expert_idx].to(out_dtype),
                    down_bias[expert_idx].to(out_dtype) if down_bias is not None else None,
                )
                current_hidden_states = down_state * topk_weights[token_idx, rank_idx, None]
                final.index_add_(0, token_idx, current_hidden_states.to(out_dtype))
            return final.reshape(batch_size, sequence_length, hidden_dim)

        module.forward = types.MethodType(_forward_fallback, module)
        patched += 1
    return patched


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def _resolve_case_names(case: str) -> list[str]:
    if case == "all":
        return ["text", "vision", "audio", "multimodal"]
    return [case]


def _prepare_case_inputs(
    model_path: Path,
    tokenizer,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    case: str,
    text_prompt: Optional[str],
):
    conversation, use_audio_in_video = build_conversation(case, text_prompt=text_prompt)
    if case == "text":
        if getattr(tokenizer, "chat_template", None):
            rendered_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        else:
            rendered_text = text_prompt or "请用一句话介绍你自己。"
        inputs = tokenizer(text=rendered_text, return_tensors="pt", padding=True)
        return rendered_text, _move_batch(inputs, device), use_audio_in_video

    if processor is None:
        from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))

    rendered_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=rendered_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        seconds_per_chunk=2.0,
        position_id_per_seconds=13,
        use_audio_in_video=use_audio_in_video,
    )
    inputs.pop("hm_pixel_values", None)
    inputs.pop("hm_pixel_values_videos", None)
    return rendered_text, _move_batch_with_dtype(inputs, device, dtype), use_audio_in_video


def _build_prefill_rope_inputs(model, inputs: dict[str, torch.Tensor], use_audio_in_video: bool):
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs["input_ids"])
    audio_feature_lengths = None
    if "feature_attention_mask" in inputs:
        audio_feature_lengths = torch.sum(inputs["feature_attention_mask"], dim=1)
    position_ids, rope_deltas = model.get_rope_index(
        inputs["input_ids"],
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        attention_mask,
        use_audio_in_video,
        audio_feature_lengths,
        inputs.get("video_second_per_grid"),
    )
    delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
    rope_deltas = rope_deltas - delta0
    return position_ids.to(torch.long), rope_deltas.to(torch.long), attention_mask


def _build_decode_position_ids(actual_length: int, rope_deltas: torch.Tensor, device: torch.device) -> torch.Tensor:
    delta = actual_length + int(rope_deltas.reshape(-1)[0].item())
    return torch.full((3, 1, 1), delta, dtype=torch.long, device=device)


def _build_rope_probe_hidden_states(model, sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(
        (1, sequence_length, _get_hidden_size(model)),
        dtype=next(model.parameters()).dtype,
        device=device,
    )


def _compose_wrapped_rope(wrapped_model, time_position_ids: torch.Tensor, height_position_ids: torch.Tensor, width_position_ids: torch.Tensor):
    text_model = wrapped_model.model
    cos_cache = text_model.rotary_emb.cos_cached[0, 0]
    sin_cache = text_model.rotary_emb.sin_cached[0, 0]

    time_cos = cos_cache[time_position_ids] * text_model.time_mask
    time_sin = sin_cache[time_position_ids] * text_model.time_mask
    height_cos = cos_cache[height_position_ids] * text_model.height_mask
    height_sin = sin_cache[height_position_ids] * text_model.height_mask
    width_cos = cos_cache[width_position_ids] * text_model.width_mask
    width_sin = sin_cache[width_position_ids] * text_model.width_mask
    return (time_cos + height_cos + width_cos).unsqueeze(0), (time_sin + height_sin + width_sin).unsqueeze(0)


def _mask_and_scatter_modal_features(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    token_id: int,
    modal_features: torch.Tensor,
    modal_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    modal_mask = input_ids == token_id
    expanded_mask = modal_mask.unsqueeze(-1).expand_as(inputs_embeds)
    modal_features = modal_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    if inputs_embeds[expanded_mask].numel() != modal_features.numel():
        raise ValueError(
            f"{modal_name} features and placeholder tokens do not match: "
            f"tokens={int(modal_mask.sum())}, features={tuple(modal_features.shape)}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, modal_features)
    return inputs_embeds, modal_mask


def _build_dense_deepstack_tensors(
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    deepstack_outputs: list[Any],
) -> list[torch.Tensor]:
    dense_tensors = []
    for deepstack_output in deepstack_outputs:
        if not isinstance(deepstack_output, torch.Tensor):
            deepstack_tensor = torch.as_tensor(deepstack_output)
        else:
            deepstack_tensor = deepstack_output
        deepstack_tensor = deepstack_tensor.to(device=inputs_embeds.device, dtype=torch.float16)
        dense_tensor = torch.zeros_like(inputs_embeds, dtype=torch.float16)
        dense_tensor[image_mask] = deepstack_tensor.to(dense_tensor.dtype)
        dense_tensors.append(dense_tensor)
    return dense_tensors


def _extract_image_feature_outputs(image_outputs):
    if hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
        return image_outputs.pooler_output, getattr(image_outputs, "deepstack_features", None)
    if hasattr(image_outputs, "last_hidden_state"):
        return image_outputs.last_hidden_state, getattr(image_outputs, "deepstack_features", None)
    if isinstance(image_outputs, tuple):
        return image_outputs
    return image_outputs, None


def _clone_tensor_or_none(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value


def _clone_tensor_list_or_none(values):
    if values is None:
        return None
    return [_clone_tensor_or_none(value) for value in values]


def _dense_deepstack_from_visual_mask(
    inputs_embeds: torch.Tensor,
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_visual_embeds: Optional[list[torch.Tensor]],
) -> list[torch.Tensor]:
    if visual_pos_masks is None or deepstack_visual_embeds is None:
        return [torch.zeros_like(inputs_embeds, dtype=torch.float16) for _ in range(3)]

    mask = visual_pos_masks[..., 0] if visual_pos_masks.ndim == 3 else visual_pos_masks
    mask = mask.to(device=inputs_embeds.device, dtype=torch.bool)
    dense_tensors = []
    for deepstack_tensor in deepstack_visual_embeds:
        dense_tensor = torch.zeros_like(inputs_embeds, dtype=torch.float16)
        dense_tensor[mask] = deepstack_tensor.to(device=inputs_embeds.device, dtype=dense_tensor.dtype)
        dense_tensors.append(dense_tensor)
    return dense_tensors


def _parse_layer_indices(value: Optional[str], num_layers: int) -> list[int]:
    if value is None or value.strip() == "":
        candidates = [0, 1, 2, 3, 7, 8, 15, 16, 23, 24, num_layers - 1]
    elif value == "all":
        candidates = list(range(num_layers))
    else:
        candidates = [int(item.strip()) for item in value.split(",") if item.strip()]
    return sorted({idx for idx in candidates if 0 <= idx < num_layers})


def _register_layer_output_captures(
    layers,
    layer_indices: Optional[list[int]] = None,
    to_cpu: bool = False,
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captures: dict[int, torch.Tensor] = {}
    handles = []
    wanted = set(range(len(layers))) if layer_indices is None else set(layer_indices)

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in wanted:
            continue

        def _capture_layer_output(_module, _inputs, output, idx=layer_idx):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            captured = tensor.detach().clone()
            captures[idx] = captured.cpu() if to_cpu else captured

        handles.append(layer.register_forward_hook(_capture_layer_output))

    return captures, handles


def _register_layer_submodule_captures(
    layer,
    to_cpu: bool = True,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[Any]]:
    captures: dict[str, dict[str, torch.Tensor]] = {}
    handles = []
    submodule_names = ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]

    def _to_capture(tensor: torch.Tensor) -> torch.Tensor:
        captured = tensor.detach().clone()
        return captured.cpu() if to_cpu else captured

    def _first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = _first_tensor(item)
                if isinstance(tensor, torch.Tensor):
                    return tensor
        return None

    for name in submodule_names:
        if not hasattr(layer, name):
            continue
        module = getattr(layer, name)
        captures[name] = {}

        def _pre_hook(_module, inputs, module_name=name):
            tensor = _first_tensor(inputs)
            if isinstance(tensor, torch.Tensor):
                captures[module_name]["input"] = _to_capture(tensor)

        def _post_hook(_module, _inputs, output, module_name=name):
            tensor = _first_tensor(output)
            if isinstance(tensor, torch.Tensor):
                captures[module_name]["output"] = _to_capture(tensor)

        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))

    return captures, handles


def _submodule_diagnostics(
    native_captures: dict[str, dict[str, torch.Tensor]],
    wrapped_captures: dict[str, dict[str, torch.Tensor]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    results = []
    first_not_allclose = None
    for module_name in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]:
        if module_name not in native_captures or module_name not in wrapped_captures:
            continue
        module_result = {"module": module_name, "tensors": {}}
        for tensor_name in ["input", "output"]:
            native_tensor = native_captures[module_name].get(tensor_name)
            wrapped_tensor = wrapped_captures[module_name].get(tensor_name)
            if native_tensor is None or wrapped_tensor is None:
                continue
            metrics = _tensor_metrics(native_tensor, wrapped_tensor, atol, rtol)
            module_result["tensors"][tensor_name] = metrics
            if first_not_allclose is None and not metrics["allclose"]:
                first_not_allclose = {"module": module_name, "tensor": tensor_name}
        results.append(module_result)
    return {
        "first_not_allclose": first_not_allclose,
        "modules": results,
    }


def _remove_handles(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _stack_position_ids_from_wrapped_inputs(wrapped_inputs: dict[str, Any], actual_length: int) -> torch.Tensor:
    return torch.stack(
        [
            wrapped_inputs["time_position_ids"][:actual_length].to(torch.long),
            wrapped_inputs["height_position_ids"][:actual_length].to(torch.long),
            wrapped_inputs["width_position_ids"][:actual_length].to(torch.long),
        ],
        dim=0,
    ).unsqueeze(1)


def _layer_diagnostics(
    native_layers: dict[int, torch.Tensor],
    wrapped_layers: dict[int, torch.Tensor],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    records = []
    first_not_allclose = None
    for layer_idx in sorted(set(native_layers) & set(wrapped_layers)):
        metrics = _tensor_metrics(native_layers[layer_idx], wrapped_layers[layer_idx], atol, rtol)
        if first_not_allclose is None and not metrics["allclose"]:
            first_not_allclose = layer_idx
        records.append({"layer": int(layer_idx), **metrics})
    return {
        "first_not_allclose_layer": first_not_allclose,
        "num_compared_layers": len(records),
        "layers": records,
    }


def _prepare_forward_wrapped_inputs(
    model,
    inputs: dict[str, torch.Tensor],
    use_audio_in_video: bool,
    input_sequence_length: int,
    position_dtype: torch.dtype,
):
    input_ids = inputs["input_ids"]
    device = input_ids.device
    inputs_embeds = model.get_input_embeddings()(input_ids).detach().to(device=device)
    deepstack_tensors = [torch.zeros_like(inputs_embeds, dtype=torch.float16) for _ in range(3)]

    feature_attention_mask = inputs.get("feature_attention_mask")
    audio_feature_lengths = None
    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

    input_features = inputs.get("input_features")
    if input_features is not None:
        audio_outputs = model.get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
        )
        audio_features = getattr(audio_outputs, "last_hidden_state", audio_outputs)
        inputs_embeds, _ = _mask_and_scatter_modal_features(
            inputs_embeds,
            input_ids,
            int(model.config.audio_token_id),
            audio_features,
            "audio",
        )

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is not None:
        image_embeds, image_embeds_multiscale = _extract_image_feature_outputs(
            model.get_image_features(pixel_values, image_grid_thw)
        )
        inputs_embeds, image_mask = _mask_and_scatter_modal_features(
            inputs_embeds,
            input_ids,
            int(model.config.image_token_id),
            image_embeds,
            "image",
        )
        if image_embeds_multiscale is not None:
            deepstack_tensors = _build_dense_deepstack_tensors(
                inputs_embeds,
                image_mask,
                list(image_embeds_multiscale),
            )

    if inputs.get("pixel_values_videos") is not None:
        raise NotImplementedError("Video inputs are not supported by this comparison script yet")

    actual_length = int(inputs_embeds.shape[1])
    if actual_length > input_sequence_length:
        raise ValueError(
            f"prompt length {actual_length} exceeds input_sequence_length {input_sequence_length}"
        )

    if actual_length < input_sequence_length:
        pad_embeds = torch.zeros(
            (inputs_embeds.shape[0], input_sequence_length - actual_length, inputs_embeds.shape[2]),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
        deepstack_tensors = [
            torch.cat(
                [tensor, torch.zeros_like(pad_embeds, dtype=torch.float16)],
                dim=1,
            )
            for tensor in deepstack_tensors
        ]

    position_ids, _, _ = _build_prefill_rope_inputs(model, inputs, use_audio_in_video)
    time_position_ids = _pad_position_id_vector(position_ids[0, 0], input_sequence_length, device, position_dtype)
    height_position_ids = _pad_position_id_vector(position_ids[1, 0], input_sequence_length, device, position_dtype)
    width_position_ids = _pad_position_id_vector(position_ids[2, 0], input_sequence_length, device, position_dtype)
    return {
        "actual_length": actual_length,
        "inputs_embeds": inputs_embeds.to(torch.float16),
        "time_position_ids": time_position_ids,
        "height_position_ids": height_position_ids,
        "width_position_ids": width_position_ids,
        "deepstack_tensors": [tensor.to(torch.float16) for tensor in deepstack_tensors],
    }


def _tensor_metrics(native: torch.Tensor, wrapped: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    native_f = native.detach().float().cpu()
    wrapped_f = wrapped.detach().float().cpu()
    diff = wrapped_f - native_f
    abs_diff = diff.abs()
    denom = native_f.abs().clamp_min(1e-6)
    rel_diff = abs_diff / denom
    flat_native = native_f.reshape(-1)
    flat_wrapped = wrapped_f.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_native, flat_wrapped, dim=0).item()
    return {
        "native_shape": list(native_f.shape),
        "wrapped_shape": list(wrapped_f.shape),
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "max_relative_error": float(rel_diff.max().item()),
        "mean_relative_error": float(rel_diff.mean().item()),
        "cosine_similarity": float(cosine),
        "allclose": bool(torch.allclose(native_f, wrapped_f, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _extract_generated_sequences(generated) -> torch.Tensor:
    if isinstance(generated, torch.Tensor):
        return generated
    sequences = getattr(generated, "sequences", None)
    if isinstance(sequences, torch.Tensor):
        return sequences
    if isinstance(generated, (tuple, list)):
        for item in generated:
            if isinstance(item, torch.Tensor) and item.ndim == 2 and not torch.is_floating_point(item):
                return item
            sequences = getattr(item, "sequences", None)
            if isinstance(sequences, torch.Tensor):
                return sequences
    raise RuntimeError(f"cannot extract generated token sequences from {type(generated)!r}")


def _generated_new_tokens(sequences: torch.Tensor, prompt_length: int) -> torch.Tensor:
    if sequences.shape[1] > prompt_length:
        return sequences[:, prompt_length:]
    return sequences


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    if token_ids.ndim == 2:
        token_ids = token_ids[0]
    return tokenizer.decode(
        token_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _levenshtein_distance(left: list[Any], right: list[Any]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        curr = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _sequence_similarity(left: list[Any], right: list[Any]) -> float:
    denom = max(len(left), len(right), 1)
    return 1.0 - (_levenshtein_distance(left, right) / denom)


def _char_bigrams(text: str) -> set[str]:
    compact = "".join(text.split())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[idx : idx + 2] for idx in range(len(compact) - 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _common_prefix_len(left: list[Any], right: list[Any]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def _generation_similarity(
    native_new_tokens: torch.Tensor,
    wrapped_new_tokens: torch.Tensor,
    native_text: str,
    wrapped_text: str,
    semantic_threshold: float,
) -> dict[str, Any]:
    native_ids = native_new_tokens.reshape(-1).detach().cpu().tolist()
    wrapped_ids = wrapped_new_tokens.reshape(-1).detach().cpu().tolist()
    native_chars = list("".join(native_text.split()))
    wrapped_chars = list("".join(wrapped_text.split()))
    token_similarity = _sequence_similarity(native_ids, wrapped_ids)
    char_similarity = _sequence_similarity(native_chars, wrapped_chars)
    bigram_jaccard = _jaccard(_char_bigrams(native_text), _char_bigrams(wrapped_text))
    exact_text_match = native_text == wrapped_text
    exact_token_match = native_ids == wrapped_ids
    semantic_score = max(token_similarity, char_similarity, bigram_jaccard)
    return {
        "native_token_count": len(native_ids),
        "wrapped_token_count": len(wrapped_ids),
        "common_prefix_token_count": _common_prefix_len(native_ids, wrapped_ids),
        "exact_token_match": bool(exact_token_match),
        "exact_text_match": bool(exact_text_match),
        "token_edit_similarity": float(token_similarity),
        "char_edit_similarity": float(char_similarity),
        "char_bigram_jaccard": float(bigram_jaccard),
        "semantic_score": float(semantic_score),
        "semantic_threshold": float(semantic_threshold),
        "semantic_close": bool(exact_text_match or semantic_score >= semantic_threshold),
        "note": "semantic_close uses deterministic token/text overlap heuristics; exact text/token match implies identical semantics.",
    }


def _resolve_eos_token_ids(tokenizer, model) -> set[int]:
    values = []
    for candidate in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, int):
            values.append(candidate)
        elif isinstance(candidate, (list, tuple, set)):
            values.extend(int(item) for item in candidate if item is not None)
    return set(values)


def _decode_position_vectors(
    past_length: int,
    rope_deltas: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = past_length + int(rope_deltas.reshape(-1)[0].item())
    position_ids = torch.full((3, 1), delta, dtype=dtype, device=device)
    return position_ids[0], position_ids[1], position_ids[2]


def _wrapped_greedy_generate(
    model,
    wrapped_model,
    wrapped_inputs: dict[str, Any],
    actual_length: int,
    context_length: int,
    max_new_tokens: int,
    rope_deltas: torch.Tensor,
    tokenizer,
    position_dtype: torch.dtype,
    device: torch.device,
    autocast_enabled: bool,
) -> torch.Tensor:
    if actual_length + max_new_tokens > context_length:
        raise ValueError(
            f"actual_length + max_new_tokens must be <= context_length, got "
            f"{actual_length} + {max_new_tokens} > {context_length}"
        )

    eos_token_ids = _resolve_eos_token_ids(tokenizer, model)
    _, past_key_caches, past_value_caches = _build_empty_caches(
        wrapped_model, context_length, torch.float16, device
    )
    zero_decode_deepstack = [
        torch.zeros((1, 1, wrapped_inputs["inputs_embeds"].shape[-1]), dtype=torch.float16, device=device)
        for _ in range(3)
    ]
    generated_tokens: list[torch.Tensor] = []

    past_seq_length = torch.tensor([0], dtype=torch.int32, device=device)
    current_input_length = torch.tensor([actual_length], dtype=torch.int32, device=device)
    with torch.inference_mode(), _autocast_context(device, autocast_enabled):
        wrapped_logits, _ = wrapped_model(
            wrapped_inputs["inputs_embeds"],
            wrapped_inputs["time_position_ids"],
            wrapped_inputs["height_position_ids"],
            wrapped_inputs["width_position_ids"],
            past_seq_length,
            current_input_length,
            wrapped_inputs["deepstack_tensors"][0],
            wrapped_inputs["deepstack_tensors"][1],
            wrapped_inputs["deepstack_tensors"][2],
            past_key_caches,
            past_value_caches,
        )

    next_token = wrapped_logits.argmax(dim=-1).to(torch.long)
    for step_idx in range(max_new_tokens):
        generated_tokens.append(next_token.detach().clone())
        token_id = int(next_token.reshape(-1)[0].item())
        if token_id in eos_token_ids:
            break
        if step_idx == max_new_tokens - 1:
            break

        decode_past_length = actual_length + step_idx
        time_position_ids, height_position_ids, width_position_ids = _decode_position_vectors(
            decode_past_length,
            rope_deltas,
            device,
            position_dtype,
        )
        token_embeds = model.get_input_embeddings()(next_token.to(device)).to(torch.float16)
        past_seq_length = torch.tensor([decode_past_length], dtype=torch.int32, device=device)
        current_input_length = torch.tensor([1], dtype=torch.int32, device=device)
        with torch.inference_mode(), _autocast_context(device, autocast_enabled):
            wrapped_logits, _ = wrapped_model(
                token_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_input_length,
                zero_decode_deepstack[0],
                zero_decode_deepstack[1],
                zero_decode_deepstack[2],
                past_key_caches,
                past_value_caches,
            )
        next_token = wrapped_logits.argmax(dim=-1).to(torch.long)

    if not generated_tokens:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.cat(generated_tokens, dim=1)


def _extract_native_accept_hidden(outputs, captured_hidden: Optional[torch.Tensor], accept_hidden_layer: int):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > accept_hidden_layer:
        return hidden_states[accept_hidden_layer]
    if captured_hidden is not None:
        return captured_hidden
    raise RuntimeError("native forward did not expose hidden_states and hook did not capture accept hidden")


def compare(args) -> dict[str, Any]:
    model_path = Path(osp.abspath(args.model))
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    _maybe_skip_transformers_cuda_allocator_warmup(args.skip_cuda_allocator_warmup)
    model, resolved_model_path, loaded_as = _load_thinker_model(args, model_path, work_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    accept_hidden_layer = args.accept_hidden_layer
    if accept_hidden_layer is None:
        accept_hidden_layer = _resolve_accept_hidden_layer(model_path, model)
    accept_hidden_layer = int(accept_hidden_layer)

    device = _first_parameter_device(model)
    model_dtype = next(model.parameters()).dtype
    processor = None
    if args.case != "text":
        _ensure_mistral_common_reasoning_effort()
        from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))

    rendered_text, prepared_inputs, use_audio_in_video = _prepare_case_inputs(
        model_path,
        tokenizer,
        processor,
        device,
        model_dtype,
        args.case,
        args.prompt,
    )
    input_ids = prepared_inputs["input_ids"]
    attention_mask = prepared_inputs.get("attention_mask")
    actual_length = int(input_ids.shape[1])
    input_sequence_length = int(args.input_sequence_length or actual_length)
    if actual_length > input_sequence_length:
        raise ValueError(f"prompt token length {actual_length} exceeds input_sequence_length {input_sequence_length}")
    if input_sequence_length > args.context_length:
        raise ValueError("input_sequence_length must be <= context_length")

    native_text_input_capture: dict[str, Any] = {}

    def _capture_text_model_inputs(_module, _inputs, kwargs):
        native_text_input_capture["inputs_embeds"] = _clone_tensor_or_none(kwargs.get("inputs_embeds"))
        native_text_input_capture["position_ids"] = _clone_tensor_or_none(kwargs.get("position_ids"))
        native_text_input_capture["visual_pos_masks"] = _clone_tensor_or_none(kwargs.get("visual_pos_masks"))
        native_text_input_capture["deepstack_visual_embeds"] = _clone_tensor_list_or_none(
            kwargs.get("deepstack_visual_embeds")
        )

    text_input_hook = model.model.register_forward_pre_hook(_capture_text_model_inputs, with_kwargs=True)

    hook_capture: dict[str, torch.Tensor] = {}
    hook_handle = None
    if accept_hidden_layer > 0:
        layer_index = accept_hidden_layer - 1
        layer = model.model.layers[layer_index]

        def _capture_layer_output(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            hook_capture["accept_hidden"] = tensor.detach()

        hook_handle = layer.register_forward_hook(_capture_layer_output)

    with torch.inference_mode(), _autocast_context(device, args.autocast):
        native_outputs = model(
            **prepared_inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            use_audio_in_video=use_audio_in_video,
        )
    text_input_hook.remove()
    if hook_handle is not None:
        hook_handle.remove()

    native_logits = native_outputs.logits[:, actual_length - 1 : actual_length, :].detach()
    native_accept_hidden = _extract_native_accept_hidden(
        native_outputs, hook_capture.get("accept_hidden"), accept_hidden_layer
    ).detach()

    position_dtype = torch.int32 if args.position_dtype == "int32" else torch.long
    wrapped_inputs = _prepare_forward_wrapped_inputs(
        model,
        prepared_inputs,
        use_audio_in_video,
        input_sequence_length,
        position_dtype,
    )
    native_accept_hidden = native_accept_hidden[:, :actual_length, :]

    diagnostics: dict[str, Any] = {}
    captured_inputs_embeds = native_text_input_capture.get("inputs_embeds")
    if isinstance(captured_inputs_embeds, torch.Tensor):
        captured_inputs_embeds = captured_inputs_embeds[:, :actual_length, :]
        manual_inputs_embeds = wrapped_inputs["inputs_embeds"][:, :actual_length, :]
        captured_position_ids = native_text_input_capture.get("position_ids")
        manual_position_ids = _stack_position_ids_from_wrapped_inputs(wrapped_inputs, actual_length)
        input_diagnostics = {
            "inputs_embeds": _tensor_metrics(
                captured_inputs_embeds,
                manual_inputs_embeds,
                args.hidden_atol,
                args.hidden_rtol,
            ),
            "position_ids": _tensor_metrics(
                captured_position_ids[:, :, :actual_length].to(torch.float32),
                manual_position_ids.to(torch.float32),
                0.0,
                0.0,
            ),
            "deepstack": [],
        }
        captured_dense_deepstack = _dense_deepstack_from_visual_mask(
            captured_inputs_embeds,
            native_text_input_capture.get("visual_pos_masks"),
            native_text_input_capture.get("deepstack_visual_embeds"),
        )
        for stack_idx, (native_deepstack, manual_deepstack) in enumerate(
            zip(captured_dense_deepstack, wrapped_inputs["deepstack_tensors"])
        ):
            input_diagnostics["deepstack"].append(
                {
                    "index": int(stack_idx),
                    **_tensor_metrics(
                        native_deepstack[:, :actual_length, :],
                        manual_deepstack[:, :actual_length, :],
                        args.hidden_atol,
                        args.hidden_rtol,
                    ),
                }
            )
        diagnostics["native_thinker_internal_vs_manual_inputs"] = input_diagnostics

    native_text_direct_layers: dict[int, torch.Tensor] = {}
    native_text_direct_logits = None
    native_text_direct_accept_hidden = None
    diagnostic_layer_indices = _parse_layer_indices(args.diagnose_layer_indices, len(model.model.layers))
    if accept_hidden_layer > 0:
        diagnostic_layer_indices = sorted(
            set(diagnostic_layer_indices) | {max(accept_hidden_layer - 1, 0)}
        )
    diagnose_submodules_layer = args.diagnose_submodules_layer
    native_submodule_captures = None
    wrapped_submodule_captures = None
    if args.diagnose_layers:
        native_text_direct_layers, native_text_handles = _register_layer_output_captures(
            model.model.layers,
            layer_indices=diagnostic_layer_indices,
            to_cpu=True,
        )
        native_submodule_handles: list[Any] = []
        if diagnose_submodules_layer is not None:
            native_submodule_captures, native_submodule_handles = _register_layer_submodule_captures(
                model.model.layers[int(diagnose_submodules_layer)],
                to_cpu=True,
            )
        direct_inputs_embeds = wrapped_inputs["inputs_embeds"][:, :actual_length, :]
        direct_position_ids = _stack_position_ids_from_wrapped_inputs(wrapped_inputs, actual_length)
        direct_attention_mask = torch.ones(
            (direct_inputs_embeds.shape[0], actual_length), dtype=torch.long, device=device
        )
        direct_visual_pos_masks = native_text_input_capture.get("visual_pos_masks")
        direct_deepstack = native_text_input_capture.get("deepstack_visual_embeds")
        with torch.inference_mode(), _autocast_context(device, args.autocast):
            native_text_direct_outputs = model.model(
                attention_mask=direct_attention_mask,
                position_ids=direct_position_ids,
                inputs_embeds=direct_inputs_embeds,
                use_cache=False,
                deepstack_visual_embeds=direct_deepstack,
                visual_pos_masks=direct_visual_pos_masks,
            )
            native_text_direct_logits = model.lm_head(
                native_text_direct_outputs.last_hidden_state[:, -1:, :]
            ).detach()
        _remove_handles(native_text_handles)
        _remove_handles(native_submodule_handles)
        native_text_direct_accept_hidden = native_text_direct_layers[accept_hidden_layer - 1].to(device)
        diagnostics["native_full_vs_native_text_direct"] = {
            "logits": _tensor_metrics(
                native_logits,
                native_text_direct_logits,
                args.logits_atol,
                args.logits_rtol,
            ),
            "accept_hidden": _tensor_metrics(
                native_accept_hidden,
                native_text_direct_accept_hidden[:, :actual_length, :],
                args.hidden_atol,
                args.hidden_rtol,
            ),
        }

    register_text_wrap_modules()
    wrap_cfg = Config(
        dict(
            batch_size=1,
            max_sequence_length=int(args.context_length),
            input_sequence_length=input_sequence_length,
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(cache_axis=2),
            accept_hidden_layer=accept_hidden_layer,
            use_multimodal_position_ids=True,
            prefill_full_accept_hidden=True,
        )
    )
    wrapped_model = wrap_llm_model(model, wrap_cfg)
    wrapped_model.to(device)
    patched_moe_blocks = _force_moe_fallback(wrapped_model) if args.moe_fallback else 0
    wrapped_model.eval()

    past_seq_length = torch.tensor([0], dtype=torch.int32, device=device)
    current_input_length = torch.tensor([actual_length], dtype=torch.int32, device=device)
    _, past_key_caches, past_value_caches = _build_empty_caches(
        wrapped_model, int(args.context_length), torch.float16, device
    )

    wrapped_layer_outputs: dict[int, torch.Tensor] = {}
    wrapped_layer_handles: list[Any] = []
    if args.diagnose_layers:
        wrapped_layer_outputs, wrapped_layer_handles = _register_layer_output_captures(
            wrapped_model.model.layers,
            layer_indices=diagnostic_layer_indices,
            to_cpu=True,
        )
        wrapped_submodule_handles: list[Any] = []
        if diagnose_submodules_layer is not None:
            wrapped_submodule_captures, wrapped_submodule_handles = _register_layer_submodule_captures(
                wrapped_model.model.layers[int(diagnose_submodules_layer)],
                to_cpu=True,
            )
    else:
        wrapped_submodule_handles = []

    with torch.inference_mode(), _autocast_context(device, args.autocast):
        wrapped_logits, wrapped_accept_hidden = wrapped_model(
            wrapped_inputs["inputs_embeds"],
            wrapped_inputs["time_position_ids"],
            wrapped_inputs["height_position_ids"],
            wrapped_inputs["width_position_ids"],
            past_seq_length,
            current_input_length,
            wrapped_inputs["deepstack_tensors"][0],
            wrapped_inputs["deepstack_tensors"][1],
            wrapped_inputs["deepstack_tensors"][2],
            past_key_caches,
            past_value_caches,
        )
    if wrapped_layer_handles:
        _remove_handles(wrapped_layer_handles)
    if wrapped_submodule_handles:
        _remove_handles(wrapped_submodule_handles)

    wrapped_accept_hidden = wrapped_accept_hidden[:, :actual_length, :]
    if args.diagnose_layers and native_text_direct_layers:
        diagnostics["wrapped_vs_native_text_direct_layers"] = _layer_diagnostics(
            native_text_direct_layers,
            wrapped_layer_outputs,
            args.hidden_atol,
            args.hidden_rtol,
        )
        diagnostics["diagnostic_layer_indices"] = diagnostic_layer_indices
        if native_submodule_captures is not None and wrapped_submodule_captures is not None:
            diagnostics["wrapped_vs_native_text_direct_submodules"] = {
                "layer": int(diagnose_submodules_layer),
                **_submodule_diagnostics(
                    native_submodule_captures,
                    wrapped_submodule_captures,
                    args.hidden_atol,
                    args.hidden_rtol,
                ),
            }

    logits_metrics = _tensor_metrics(native_logits, wrapped_logits, args.logits_atol, args.logits_rtol)
    hidden_metrics = _tensor_metrics(
        native_accept_hidden, wrapped_accept_hidden, args.hidden_atol, args.hidden_rtol
    )
    native_top1 = int(native_logits.argmax(dim=-1).reshape(-1)[0].item())
    wrapped_top1 = int(wrapped_logits.argmax(dim=-1).reshape(-1)[0].item())

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model": str(model_path),
        "resolved_model_path": str(resolved_model_path),
        "loaded_as": loaded_as,
        "device": str(device),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "autocast": bool(args.autocast),
        "moe_fallback": bool(args.moe_fallback),
        "patched_moe_blocks": int(patched_moe_blocks),
        "case": args.case,
        "rendered_text": rendered_text,
        "prompt": args.prompt,
        "input_ids": input_ids.detach().cpu().tolist(),
        "actual_sequence_length": actual_length,
        "input_sequence_length": input_sequence_length,
        "context_length": int(args.context_length),
        "accept_hidden_layer": accept_hidden_layer,
        "native_top1_token_id": native_top1,
        "wrapped_top1_token_id": wrapped_top1,
        "top1_match": native_top1 == wrapped_top1,
        "logits": logits_metrics,
        "accept_hidden": hidden_metrics,
        "diagnostics": diagnostics,
        "passed": bool(logits_metrics["allclose"] and hidden_metrics["allclose"] and native_top1 == wrapped_top1),
    }
    return report


def compare_generate(args) -> dict[str, Any]:
    model_path = Path(osp.abspath(args.model))
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    _maybe_skip_transformers_cuda_allocator_warmup(args.skip_cuda_allocator_warmup)
    model, resolved_model_path, loaded_as = _load_thinker_model(args, model_path, work_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    accept_hidden_layer = args.accept_hidden_layer
    if accept_hidden_layer is None:
        accept_hidden_layer = _resolve_accept_hidden_layer(model_path, model)
    accept_hidden_layer = int(accept_hidden_layer)

    device = _first_parameter_device(model)
    model_dtype = next(model.parameters()).dtype
    processor = None
    if args.case != "text":
        _ensure_mistral_common_reasoning_effort()
        from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))

    rendered_text, prepared_inputs, use_audio_in_video = _prepare_case_inputs(
        model_path,
        tokenizer,
        processor,
        device,
        model_dtype,
        args.case,
        args.prompt,
    )
    input_ids = prepared_inputs["input_ids"]
    actual_length = int(input_ids.shape[1])
    input_sequence_length = int(args.input_sequence_length or actual_length)
    if actual_length > input_sequence_length:
        raise ValueError(f"prompt token length {actual_length} exceeds input_sequence_length {input_sequence_length}")
    if actual_length + int(args.max_new_tokens) > int(args.context_length):
        raise ValueError(
            f"actual_length + max_new_tokens must be <= context_length, got "
            f"{actual_length} + {args.max_new_tokens} > {args.context_length}"
        )

    generation_kwargs = dict(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=False,
        use_audio_in_video=use_audio_in_video,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode(), _autocast_context(device, args.autocast):
        native_generated = model.generate(
            **prepared_inputs,
            **generation_kwargs,
        )
    native_sequences = _extract_generated_sequences(native_generated)
    native_new_tokens = _generated_new_tokens(native_sequences, actual_length)
    native_text = _decode_tokens(tokenizer, native_new_tokens)

    position_dtype = torch.int32 if args.position_dtype == "int32" else torch.long
    wrapped_inputs = _prepare_forward_wrapped_inputs(
        model,
        prepared_inputs,
        use_audio_in_video,
        input_sequence_length,
        position_dtype,
    )
    _, rope_deltas, _ = _build_prefill_rope_inputs(model, prepared_inputs, use_audio_in_video)

    register_text_wrap_modules()
    wrap_cfg = Config(
        dict(
            batch_size=1,
            max_sequence_length=int(args.context_length),
            input_sequence_length=input_sequence_length,
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(cache_axis=2),
            accept_hidden_layer=accept_hidden_layer,
            use_multimodal_position_ids=True,
            prefill_full_accept_hidden=True,
        )
    )
    wrapped_model = wrap_llm_model(model, wrap_cfg)
    wrapped_model.to(device)
    patched_moe_blocks = _force_moe_fallback(wrapped_model) if args.moe_fallback else 0
    wrapped_model.eval()

    wrapped_new_tokens = _wrapped_greedy_generate(
        model=model,
        wrapped_model=wrapped_model,
        wrapped_inputs=wrapped_inputs,
        actual_length=actual_length,
        context_length=int(args.context_length),
        max_new_tokens=int(args.max_new_tokens),
        rope_deltas=rope_deltas,
        tokenizer=tokenizer,
        position_dtype=position_dtype,
        device=device,
        autocast_enabled=args.autocast,
    )
    wrapped_text = _decode_tokens(tokenizer, wrapped_new_tokens)
    similarity = _generation_similarity(
        native_new_tokens,
        wrapped_new_tokens,
        native_text,
        wrapped_text,
        args.semantic_threshold,
    )

    return {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "compare_target": "generate",
        "model": str(model_path),
        "resolved_model_path": str(resolved_model_path),
        "loaded_as": loaded_as,
        "device": str(device),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "autocast": bool(args.autocast),
        "moe_fallback": bool(args.moe_fallback),
        "patched_moe_blocks": int(patched_moe_blocks),
        "case": args.case,
        "rendered_text": rendered_text,
        "prompt": args.prompt,
        "actual_sequence_length": actual_length,
        "input_sequence_length": input_sequence_length,
        "context_length": int(args.context_length),
        "max_new_tokens": int(args.max_new_tokens),
        "accept_hidden_layer": accept_hidden_layer,
        "native_new_token_ids": native_new_tokens.detach().cpu().tolist(),
        "wrapped_new_token_ids": wrapped_new_tokens.detach().cpu().tolist(),
        "native_text": native_text,
        "wrapped_text": wrapped_text,
        "generation_similarity": similarity,
        "passed": bool(similarity["semantic_close"]),
    }


def compare_rope(args) -> dict[str, Any]:
    model_path = Path(osp.abspath(args.model))
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    _maybe_skip_transformers_cuda_allocator_warmup(args.skip_cuda_allocator_warmup)
    model, resolved_model_path, loaded_as = _load_thinker_model(args, model_path, work_dir)
    if not hasattr(model, "get_rope_index"):
        raise ValueError("rope comparison requires a Qwen3-Omni thinker load path; use --load-kind thinker or auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    accept_hidden_layer = args.accept_hidden_layer
    if accept_hidden_layer is None:
        accept_hidden_layer = _resolve_accept_hidden_layer(model_path, model)
    accept_hidden_layer = int(accept_hidden_layer)

    device = _first_parameter_device(model)
    model_dtype = next(model.parameters()).dtype
    case_names = _resolve_case_names(args.case)

    processor = None
    if any(case_name != "text" for case_name in case_names):
        _ensure_mistral_common_reasoning_effort()
        from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))

    case_records = []
    max_actual_length = 1

    with torch.inference_mode(), _autocast_context(device, args.autocast):
        for case_name in case_names:
            rendered_text, inputs, use_audio_in_video = _prepare_case_inputs(
                model_path,
                tokenizer,
                processor,
                device,
                model_dtype,
                case_name,
                args.prompt,
            )
            position_ids, rope_deltas, attention_mask = _build_prefill_rope_inputs(model, inputs, use_audio_in_video)
            actual_length = int(attention_mask.reshape(-1).sum().item()) if attention_mask is not None else int(inputs["input_ids"].shape[1])
            position_ids = position_ids[:, :, :actual_length]
            decode_position_ids = _build_decode_position_ids(actual_length, rope_deltas, device)
            native_prefill_cos, native_prefill_sin = model.model.rotary_emb(
                _build_rope_probe_hidden_states(model, actual_length, device),
                position_ids,
            )
            native_decode_cos, native_decode_sin = model.model.rotary_emb(
                _build_rope_probe_hidden_states(model, 1, device),
                decode_position_ids,
            )

            max_actual_length = max(max_actual_length, actual_length)
            case_records.append(
                {
                    "case": case_name,
                    "rendered_text": rendered_text,
                    "input_ids": inputs["input_ids"].detach().cpu().tolist(),
                    "actual_sequence_length": actual_length,
                    "time_position_ids": position_ids[0, 0].detach().clone(),
                    "height_position_ids": position_ids[1, 0].detach().clone(),
                    "width_position_ids": position_ids[2, 0].detach().clone(),
                    "rope_deltas": rope_deltas.detach().clone(),
                    "decode_position_ids": decode_position_ids.detach().clone(),
                    "native_prefill_cos": native_prefill_cos.detach().clone(),
                    "native_prefill_sin": native_prefill_sin.detach().clone(),
                    "native_decode_cos": native_decode_cos.detach().clone(),
                    "native_decode_sin": native_decode_sin.detach().clone(),
                }
            )

    register_text_wrap_modules()
    wrap_cfg = Config(
        dict(
            batch_size=1,
            max_sequence_length=int(args.context_length),
            input_sequence_length=max(int(args.input_sequence_length or 0), max_actual_length),
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(cache_axis=2),
            accept_hidden_layer=accept_hidden_layer,
            use_multimodal_position_ids=True,
            prefill_full_accept_hidden=True,
        )
    )
    wrapped_model = wrap_llm_model(model, wrap_cfg)
    wrapped_model.to(device)
    wrapped_model.eval()

    case_results = []
    for record in case_records:
        time_position_ids = record["time_position_ids"].to(device=device, dtype=torch.long)
        height_position_ids = record["height_position_ids"].to(device=device, dtype=torch.long)
        width_position_ids = record["width_position_ids"].to(device=device, dtype=torch.long)
        decode_position_ids = record["decode_position_ids"].to(device=device, dtype=torch.long)

        wrapped_prefill_cos, wrapped_prefill_sin = _compose_wrapped_rope(
            wrapped_model,
            time_position_ids,
            height_position_ids,
            width_position_ids,
        )
        wrapped_decode_cos, wrapped_decode_sin = _compose_wrapped_rope(
            wrapped_model,
            decode_position_ids[0, 0],
            decode_position_ids[1, 0],
            decode_position_ids[2, 0],
        )

        prefill_cos_metrics = _tensor_metrics(record["native_prefill_cos"], wrapped_prefill_cos, args.rope_atol, args.rope_rtol)
        prefill_sin_metrics = _tensor_metrics(record["native_prefill_sin"], wrapped_prefill_sin, args.rope_atol, args.rope_rtol)
        decode_cos_metrics = _tensor_metrics(record["native_decode_cos"], wrapped_decode_cos, args.rope_atol, args.rope_rtol)
        decode_sin_metrics = _tensor_metrics(record["native_decode_sin"], wrapped_decode_sin, args.rope_atol, args.rope_rtol)

        case_passed = all(
            metric["allclose"]
            for metric in (prefill_cos_metrics, prefill_sin_metrics, decode_cos_metrics, decode_sin_metrics)
        )
        case_results.append(
            {
                "case": record["case"],
                "rendered_text": record["rendered_text"],
                "input_ids": record["input_ids"],
                "actual_sequence_length": record["actual_sequence_length"],
                "prefill_position_ids": {
                    "time": record["time_position_ids"].detach().cpu().tolist(),
                    "height": record["height_position_ids"].detach().cpu().tolist(),
                    "width": record["width_position_ids"].detach().cpu().tolist(),
                },
                "rope_deltas": record["rope_deltas"].detach().cpu().tolist(),
                "decode_position_ids": {
                    "time": record["decode_position_ids"][0, 0].detach().cpu().tolist(),
                    "height": record["decode_position_ids"][1, 0].detach().cpu().tolist(),
                    "width": record["decode_position_ids"][2, 0].detach().cpu().tolist(),
                },
                "prefill_cos": prefill_cos_metrics,
                "prefill_sin": prefill_sin_metrics,
                "decode_cos": decode_cos_metrics,
                "decode_sin": decode_sin_metrics,
                "passed": case_passed,
            }
        )

    return {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "compare_target": "rope",
        "model": str(model_path),
        "resolved_model_path": str(resolved_model_path),
        "loaded_as": loaded_as,
        "device": str(device),
        "device_map": args.device_map,
        "dtype": args.dtype,
        "autocast": bool(args.autocast),
        "case": args.case,
        "rope_atol": float(args.rope_atol),
        "rope_rtol": float(args.rope_rtol),
        "results": case_results,
        "passed": bool(all(item["passed"] for item in case_results)),
    }


def main(args) -> None:
    if args.compare_target == "rope":
        report = compare_rope(args)
    elif args.compare_target == "generate":
        report = compare_generate(args)
    else:
        report = compare(args)
    report_path = Path(args.output).resolve() if args.output else Path(args.work_dir).resolve() / "wrap_native_compare_report.json"
    _save_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report saved to: {report_path}")
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare native Qwen3-Omni thinker text forward with the xhquant wrapped PyTorch forward."
    )
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni_wrap_compare")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="请用一句话介绍你自己。")
    parser.add_argument("--compare-target", type=str, default="forward", choices=["forward", "rope", "generate"])
    parser.add_argument(
        "--load-kind",
        type=str,
        default="auto",
        choices=["auto", "thinker", "text-view", "causal-lm"],
        help="auto loads a Qwen3-Omni root as thinker-only; text-view materializes a Qwen3Moe text view first.",
    )
    parser.add_argument(
        "--force-text-view",
        action="store_true",
        help="With --load-kind auto and a Qwen3-Omni root checkpoint, compare through the prepared Qwen3Moe text view.",
    )
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--accept-hidden-layer", type=int, default=None)
    parser.add_argument("--position-dtype", type=str, default="int32", choices=["int32", "int64"])
    parser.add_argument(
        "--skip-cuda-allocator-warmup",
        action="store_true",
        help="Diagnostic-only: skip transformers CUDA allocator warmup before loading very large models.",
    )
    parser.add_argument("--case", type=str, default="text", choices=["text", "vision", "audio", "multimodal", "all"])
    parser.add_argument("--logits-atol", type=float, default=5e-2)
    parser.add_argument("--logits-rtol", type=float, default=5e-2)
    parser.add_argument("--hidden-atol", type=float, default=5e-2)
    parser.add_argument("--hidden-rtol", type=float, default=5e-2)
    parser.add_argument("--rope-atol", type=float, default=5e-2)
    parser.add_argument("--rope-rtol", type=float, default=5e-2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.6,
        help="Heuristic text/token similarity threshold for --compare-target generate.",
    )
    parser.add_argument(
        "--diagnose-layers",
        action="store_true",
        help="Run an extra native text-tower pass and report per-layer wrapped-vs-native divergence.",
    )
    parser.add_argument(
        "--diagnose-layer-indices",
        type=str,
        default="0,1,2,3,7,8,15,16,23,24,47",
        help="Comma-separated layer indices to capture for --diagnose-layers, or 'all'.",
    )
    parser.add_argument(
        "--diagnose-submodules-layer",
        type=int,
        default=None,
        help="When --diagnose-layers is set, capture input/output for major submodules in this layer.",
    )
    parser.add_argument("--no-autocast", dest="autocast", action="store_false")
    parser.set_defaults(autocast=True)
    parser.add_argument(
        "--fast-moe",
        dest="moe_fallback",
        action="store_false",
        help="Use MoeBlock's Triton fast path. The default uses the PyTorch fallback for robust numerical comparison.",
    )
    parser.set_defaults(moe_fallback=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())