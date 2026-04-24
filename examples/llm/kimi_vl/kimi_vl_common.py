from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable
import types

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoProcessor
from xhquant.api import CacheTensor

from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.kimi_moe._moe_model import register_wrap_modules
from xh_model_zoo.xh_llm.models.kimi_moe.configuration_deepseek import DeepseekV3Config
from xh_model_zoo.xh_llm.models.kimi_moe.modeling_deepseek import DeepseekV3ForCausalLM


def finalize_modelscope_weights(model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    temp_dir = model_dir / "._____temp"
    if not temp_dir.exists():
        return

    for shard in temp_dir.glob("model-*.safetensors"):
        target = model_dir / shard.name
        if not target.exists():
            shard.rename(target)


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load module spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_kimi_remote_modules(model_dir: str | Path) -> tuple[ModuleType, ModuleType]:
    model_dir = Path(model_dir)
    package_name = "kimi_vl_remote_pkg"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    config_mod = _load_module_from_file(f"{package_name}.configuration_kimi_vl", model_dir / "configuration_kimi_vl.py")
    modeling_mod = _load_module_from_file(f"{package_name}.modeling_kimi_vl", model_dir / "modeling_kimi_vl.py")
    return config_mod, modeling_mod


def patch_kimi_visual_for_export(modeling_mod: ModuleType) -> None:
    if getattr(modeling_mod, "_xh_export_safe_rope_patched", False):
        return

    class ExportSafeRope2DPosEmb(modeling_mod.Rope2DPosEmb):
        def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
            n = self.max_height * self.max_width
            flat_pos = torch.arange(0, n, device=device, dtype=torch.float32)
            x_pos = flat_pos % self.max_width
            y_pos = flat_pos // self.max_width
            dim_range = torch.arange(0, self.dim, 4, device=device, dtype=torch.float32)[: (self.dim // 4)]
            freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
            x_freqs = torch.outer(x_pos, freqs).float()
            y_freqs = torch.outer(y_pos, freqs).float()
            x_cos = x_freqs.cos()
            x_sin = x_freqs.sin()
            y_cos = y_freqs.cos()
            y_sin = y_freqs.sin()
            xy = torch.stack([x_cos, x_sin, y_cos, y_sin], dim=-1)
            return xy.reshape(self.max_height, self.max_width, -1, 4)

        def get_freqs_cis(self, grid_hws: torch.Tensor) -> torch.Tensor:
            if self.freqs_cis is None:
                self.freqs_cis = self._precompute_freqs_cis(grid_hws.device)
            shapes = grid_hws.tolist()
            return torch.cat([self.freqs_cis[:h, :w].reshape(-1, self.dim // 4, 4) for h, w in shapes], dim=0)

    def export_safe_apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = freqs_cis.unsqueeze(-3)
        x_cos = freqs_cis[..., 0]
        x_sin = freqs_cis[..., 1]
        y_cos = freqs_cis[..., 2]
        y_sin = freqs_cis[..., 3]

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x = x.float().reshape(*x.shape[:-1], -1, 4)
            x0 = x[..., 0]
            x1 = x[..., 1]
            x2 = x[..., 2]
            x3 = x[..., 3]
            out0 = x0 * x_cos - x1 * x_sin
            out1 = x1 * x_cos + x0 * x_sin
            out2 = x2 * y_cos - x3 * y_sin
            out3 = x3 * y_cos + x2 * y_sin
            return torch.stack([out0, out1, out2, out3], dim=-1).flatten(-2)

        return _rotate(xq).type_as(xq), _rotate(xk).type_as(xk)

    modeling_mod.Rope2DPosEmb = ExportSafeRope2DPosEmb
    modeling_mod.apply_rope = export_safe_apply_rope
    modeling_mod._xh_export_safe_rope_patched = True


def load_config_json(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    return json.loads((model_dir / "config.json").read_text())


def load_weight_map(model_dir: str | Path) -> dict[str, str]:
    model_dir = Path(model_dir)
    finalize_modelscope_weights(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def load_prefixed_state_dict(model_dir: str | Path, prefixes: Iterable[str]) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    weight_map = load_weight_map(model_dir)
    prefix_list = tuple(prefixes)
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        if key.startswith(prefix_list):
            shard_to_keys.setdefault(shard_name, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, keys in shard_to_keys.items():
        shard_state = load_safetensors_file(str(model_dir / shard_name), device="cpu")
        for key in keys:
            state_dict[key] = shard_state[key]
        del shard_state
    return state_dict


def strip_prefix_state_dict(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    prefix = prefix.rstrip(".") + "."
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


@dataclass
class PreparedInputs:
    prompt_text: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_hws: torch.Tensor
    image_embeds: torch.Tensor
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    current_input_length: int


class KimiVLVisualModel(nn.Module):
    def __init__(self, vision_tower: nn.Module, projector: nn.Module, image_grid_hws: torch.Tensor):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        self.register_buffer("image_grid_hws", image_grid_hws.to(torch.int64), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.vision_tower(pixel_values, self.image_grid_hws)
        return self.projector(image_features)


def build_visual_model(model_dir: str | Path, image_grid_hws: torch.Tensor, dtype: torch.dtype = torch.float16) -> KimiVLVisualModel:
    model_dir = Path(model_dir)
    config_mod, modeling_mod = load_kimi_remote_modules(model_dir)
    patch_kimi_visual_for_export(modeling_mod)
    cfg_json = load_config_json(model_dir)
    config = config_mod.KimiVLConfig(
        vision_config=cfg_json["vision_config"],
        text_config=cfg_json["text_config"],
        ignore_index=cfg_json.get("ignore_index", -100),
        media_placeholder_token_id=cfg_json["media_placeholder_token_id"],
        pad_token_id=cfg_json["text_config"]["pad_token_id"],
        tie_word_embeddings=cfg_json.get("tie_word_embeddings", False),
    )

    vision_tower = modeling_mod.MoonVitPretrainedModel(config.vision_config).eval()
    projector = modeling_mod.KimiVLMultiModalProjector(config).eval()

    visual_state = load_prefixed_state_dict(model_dir, ("vision_tower.", "multi_modal_projector."))
    vision_state = {k[len("vision_tower.") :]: v for k, v in visual_state.items() if k.startswith("vision_tower.")}
    projector_state = {
        k[len("multi_modal_projector.") :]: v
        for k, v in visual_state.items()
        if k.startswith("multi_modal_projector.")
    }
    vision_tower.load_state_dict(vision_state, strict=True)
    projector.load_state_dict(projector_state, strict=True)

    model = KimiVLVisualModel(vision_tower, projector, image_grid_hws)
    model.to(dtype=dtype)
    return model.eval()


def build_language_model(model_dir: str | Path, dtype: torch.dtype = torch.float16) -> tuple[DeepseekV3ForCausalLM, nn.Embedding, dict]:
    model_dir = Path(model_dir)
    cfg_json = load_config_json(model_dir)
    text_config = DeepseekV3Config(**cfg_json["text_config"])
    native_model = DeepseekV3ForCausalLM(text_config).eval()
    state_dict = strip_prefix_state_dict(load_prefixed_state_dict(model_dir, ("language_model.",)), "language_model")
    expected_keys = native_model.state_dict().keys()
    state_dict = {key: value for key, value in state_dict.items() if key in expected_keys}
    native_model.load_state_dict(state_dict, strict=False)
    native_model.to(dtype=dtype)
    token_embedding = deepcopy(native_model.model.get_input_embeddings()).to(dtype=dtype)
    return native_model, token_embedding, cfg_json


def build_wrapped_language_model(native_model: DeepseekV3ForCausalLM, batch_size: int, context_length: int, input_sequence_length: int):
    wrap_cfg = {
        "batch_size": batch_size,
        "max_sequence_length": context_length,
        "input_sequence_length": input_sequence_length,
        "use_cache": True,
        "num_logits_to_keep": 1,
        "kv_cache": {"cache_axis": 2},
        "enable_rope": True,
    }
    register_wrap_modules()
    return wrap_llm_model(native_model, wrap_cfg), wrap_cfg


def resize_image(image_path: str | Path, image_size_h: int, image_size_w: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image.resize((image_size_w, image_size_h), Image.Resampling.BICUBIC)


def build_processor_inputs(
    model_dir: str | Path,
    image_path: str | Path,
    prompt: str,
    image_size_h: int,
    image_size_w: int,
):
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    image = resize_image(image_path, image_size_h=image_size_h, image_size_w=image_size_w)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
    inputs["image_grid_hws"] = inputs["image_grid_hws"].to(torch.int64)
    return processor, prompt_text, inputs


def merge_image_embeds(
    token_embedding: nn.Embedding,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    media_placeholder_token_id: int,
) -> torch.Tensor:
    input_ids = input_ids.to(torch.long)
    inputs_embeds = token_embedding(input_ids)
    image_token_count = int((input_ids == media_placeholder_token_id).sum().item())
    if image_token_count != int(image_embeds.shape[0]):
        raise ValueError(
            f"placeholder token count mismatch: tokens={image_token_count}, image_embeds={int(image_embeds.shape[0])}"
        )
    image_mask = (input_ids == media_placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))


def prepare_multimodal_inputs(
    model_dir: str | Path,
    image_path: str | Path,
    prompt: str,
    token_embedding: nn.Embedding,
    image_size_h: int,
    image_size_w: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoProcessor, PreparedInputs]:
    processor, prompt_text, inputs = build_processor_inputs(model_dir, image_path, prompt, image_size_h, image_size_w)
    visual_model = build_visual_model(model_dir, inputs["image_grid_hws"], dtype=dtype)
    with torch.no_grad():
        image_embeds = visual_model(inputs["pixel_values"].to(dtype=dtype)).cpu()

    cfg_json = load_config_json(model_dir)
    merged = merge_image_embeds(
        token_embedding=token_embedding.cpu(),
        input_ids=inputs["input_ids"].cpu(),
        image_embeds=image_embeds.cpu(),
        media_placeholder_token_id=cfg_json["media_placeholder_token_id"],
    )
    input_ids = inputs["input_ids"].cpu()
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)
    prepared = PreparedInputs(
        prompt_text=prompt_text,
        input_ids=input_ids,
        pixel_values=inputs["pixel_values"].cpu(),
        image_grid_hws=inputs["image_grid_hws"].cpu(),
        image_embeds=image_embeds.cpu(),
        inputs_embeds=merged.cpu(),
        position_ids=position_ids.cpu(),
        current_input_length=seq_len,
    )
    return processor, prepared


def pad_prefill_inputs(
    prepared: PreparedInputs,
    token_embedding: nn.Embedding,
    input_sequence_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs_embeds = prepared.inputs_embeds
    position_ids = prepared.position_ids
    current_seq_len = prepared.current_input_length
    if current_seq_len > input_sequence_length:
        raise ValueError(
            f"input sequence too long: current_seq_len={current_seq_len}, input_sequence_length={input_sequence_length}"
        )
    if current_seq_len == input_sequence_length:
        return inputs_embeds, position_ids.to(torch.int32), torch.tensor([current_seq_len], dtype=torch.int32)

    pad_count = input_sequence_length - current_seq_len
    pad_ids = torch.full((1, pad_count), pad_token_id, dtype=torch.long)
    pad_embeds = token_embedding(pad_ids)
    padded_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
    pad_position_ids = torch.ones((1, pad_count), dtype=position_ids.dtype)
    padded_position_ids = torch.cat([position_ids, pad_position_ids], dim=1)
    return padded_embeds, padded_position_ids.to(torch.int32), torch.tensor([current_seq_len], dtype=torch.int32)


def create_kv_caches(
    batch_size: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    context_length: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype = torch.float16,
    cache_tensor: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    key_caches = []
    value_caches = []
    for _ in range(num_hidden_layers):
        k = torch.zeros((batch_size, num_key_value_heads, context_length, key_head_dim), dtype=dtype)
        v = torch.zeros((batch_size, num_key_value_heads, context_length, value_head_dim), dtype=dtype)
        if cache_tensor:
            k = CacheTensor(k)
            v = CacheTensor(v)
        key_caches.append(k)
        value_caches.append(v)
    return key_caches, value_caches


def flatten_hmonnx_inputs(args: tuple) -> list[torch.Tensor]:
    flat: list[torch.Tensor] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flat.extend(flatten_hmonnx_inputs(tuple(arg)))
        else:
            flat.append(arg)
    return flat
