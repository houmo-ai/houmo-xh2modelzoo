import json
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Qwen3OmniMoeForConditionalGeneration

from xhquant.api import CacheTensor
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

from xh_model_zoo.xh_llm.models.qwen3_omni.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
from xh_model_zoo.xh_llm.models.qwen3_omni.monkey_patch import Qwen3OmniMoeThinkerForConditionalGeneration_forward
from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

# Monkey-patch HMONNXInference to auto-set exec_device=cuda (GPU execution
# without moving all weights to GPU, avoiding OOM).
_orig_hmonnx_init = HMONNXInference.__init__

def _hmonnx_init_cuda_exec(self, onnx_file: str) -> None:
    _orig_hmonnx_init(self, onnx_file)
    if torch.cuda.is_available():
        self.exec_device = torch.device("cuda")

HMONNXInference.__init__ = _hmonnx_init_cuda_exec

SCRIPT_DIR = Path(__file__).resolve().parent
_HMONNX_RUNTIME_FIX_CACHE: Dict[str, Path] = {}
_GIB = 1024**3


def release_export_cuda_memory(logger=None, label: Optional[str] = None):
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if logger is not None:
        suffix = f" after {label}" if label else ""
        logger.info(f"released export resources and cleared CUDA cache{suffix}")


def _pick_best_validation_single_gpu(logger=None) -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    best_gpu_idx = None
    best_free_bytes = -1
    debug_entries = []

    for gpu_idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
        min_required_bytes = max(40 * _GIB, int(total_bytes * 0.9))
        debug_entries.append(
            f"cuda:{gpu_idx}: free={_format_bytes_as_gib_str(free_bytes)}, "
            f"required={_format_bytes_as_gib_str(min_required_bytes)}"
        )
        if free_bytes < min_required_bytes:
            continue
        if free_bytes > best_free_bytes:
            best_gpu_idx = gpu_idx
            best_free_bytes = free_bytes

    if logger is not None:
        logger.info("validation single-gpu candidates: " + "; ".join(debug_entries))

    if best_gpu_idx is None:
        return None
    return f"cuda:{best_gpu_idx}"


def _resolve_validation_device_map(device_map: str, logger=None) -> str:
    if device_map != "auto":
        return device_map
    if not torch.cuda.is_available():
        resolved = "cpu"
    else:
        resolved = _pick_best_validation_single_gpu(logger) or "auto"
    if logger is not None:
        logger.info(f"validation device_map resolved from auto to {resolved}")
    return resolved


def _format_bytes_as_gib_str(num_bytes: int) -> str:
    gib = max(num_bytes, 0) / _GIB
    return f"{gib:.1f}GiB"


def _build_safe_validation_max_memory(logger=None) -> Optional[Dict[Any, str]]:
    if not torch.cuda.is_available():
        return None

    reserve_bytes = 8 * _GIB
    min_free_bytes = 12 * _GIB
    max_memory: Dict[Any, str] = {}
    debug_entries = []

    for gpu_idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(gpu_idx)
        if free_bytes < min_free_bytes:
            debug_entries.append(f"cuda:{gpu_idx}: skipped free={_format_bytes_as_gib_str(free_bytes)}")
            continue

        planner_bytes = free_bytes - reserve_bytes
        if planner_bytes < min_free_bytes:
            planner_bytes = int(free_bytes * 0.9)
        if planner_bytes <= 0:
            debug_entries.append(f"cuda:{gpu_idx}: skipped after reserve free={_format_bytes_as_gib_str(free_bytes)}")
            continue

        max_memory[gpu_idx] = _format_bytes_as_gib_str(planner_bytes)
        debug_entries.append(
            f"cuda:{gpu_idx}: planner={max_memory[gpu_idx]}, free={_format_bytes_as_gib_str(free_bytes)}"
        )

    try:
        import psutil

        cpu_available = psutil.virtual_memory().available
        cpu_budget = max(cpu_available - 16 * _GIB, 32 * _GIB)
        max_memory["cpu"] = _format_bytes_as_gib_str(cpu_budget)
    except Exception:
        max_memory["cpu"] = "64.0GiB"

    if logger is not None:
        logger.info("validation max_memory: " + "; ".join(debug_entries + [f"cpu: planner={max_memory['cpu']}"]))

    return max_memory


def _build_inputs_embeds_input_ids(model_kwargs: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if not model_kwargs:
        return None

    inputs_embeds = model_kwargs.get("inputs_embeds")
    if not isinstance(inputs_embeds, torch.Tensor):
        return None

    batch_size = 1
    for value in model_kwargs.values():
        if isinstance(value, torch.Tensor):
            batch_size = int(value.shape[0])
            break

    return torch.ones((batch_size, 0), dtype=torch.long, device=inputs_embeds.device)


def _patch_inputs_embeds_generation_device(module, module_name: str, logger=None):
    if getattr(module, "_xh_inputs_embeds_generation_device_patched", False):
        return

    original = getattr(module, "_maybe_initialize_input_ids_for_generation", None)
    if not callable(original):
        return

    def patched(self, inputs=None, bos_token_id=None, model_kwargs=None):
        generated_input_ids = None
        if inputs is None:
            generated_input_ids = _build_inputs_embeds_input_ids(model_kwargs)
        if generated_input_ids is not None:
            return generated_input_ids
        return original(inputs, bos_token_id, model_kwargs)

    module._maybe_initialize_input_ids_for_generation = types.MethodType(patched, module)
    module._xh_inputs_embeds_generation_device_patched = True
    if logger is not None:
        logger.info(f"patched inputs_embeds generation device for {module_name}")


def _resolve_runtime_execution_device(module) -> torch.device:
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        device = torch.device(execution_device)
        if device.type != "meta":
            return device

    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    for buffer in module.buffers():
        if buffer.device.type != "meta":
            return buffer.device

    hf_device_map = getattr(module, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped_device in hf_device_map.values():
            if mapped_device in (None, "disk"):
                continue
            try:
                candidate = torch.device(mapped_device)
            except (TypeError, RuntimeError, ValueError):
                continue
            if candidate.type != "meta":
                return candidate

    raise RuntimeError(f"Cannot resolve a concrete runtime device for {module.__class__.__name__}")


def _patch_runtime_device_property(module, module_name: str, logger=None):
    if getattr(module, "_xh_runtime_device_property_patched", False):
        return

    patched_cls = type(
        f"{module.__class__.__name__}XHRuntimeDevicePatched",
        (module.__class__,),
        {"device": property(lambda self: _resolve_runtime_execution_device(self))},
    )
    module.__class__ = patched_cls
    module._xh_runtime_device_property_patched = True
    if logger is not None:
        logger.info(f"patched runtime device property for {module_name}")


def _create_hmonnx_session(onnx_path: Path) -> HMONNXInference:
    try:
        return HMONNXInference(str(onnx_path))
    except KeyError as exc:
        if "indices_or_sections" not in str(exc):
            raise

    cache_key = str(onnx_path.resolve())
    fixed_path = _HMONNX_RUNTIME_FIX_CACHE.get(cache_key)
    if fixed_path is None or not fixed_path.exists():
        import onnx
        from onnx import helper
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        initializer_map = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
        value_shape_map = {}
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
            tensor_type = value.type.tensor_type
            dims = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(None)
            value_shape_map[value.name] = dims

        def _infer_split_sections(node) -> list[int]:
            has_indices = any(attr.name == "indices_or_sections" for attr in node.attribute)
            if has_indices:
                indices_attr = next(attr for attr in node.attribute if attr.name == "indices_or_sections")
                return list(indices_attr.ints)

            split_attr = next((attr for attr in node.attribute if attr.name == "split"), None)
            if split_attr is not None:
                split_sections = list(split_attr.ints)
                if not split_sections and split_attr.i != 0:
                    split_sections = [int(split_attr.i)]
                if split_sections:
                    return split_sections

            if len(node.input) > 1 and node.input[1] in initializer_map:
                return initializer_map[node.input[1]].reshape(-1).astype("int64").tolist()

            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            num_outputs_attr = next((attr for attr in node.attribute if attr.name == "num_outputs"), None)
            input_shape = value_shape_map.get(node.input[0], [])
            if axis_attr is None or num_outputs_attr is None or not input_shape:
                return []

            axis = int(axis_attr.i)
            if axis < 0:
                axis += len(input_shape)
            if not (0 <= axis < len(input_shape)):
                return []

            axis_dim = input_shape[axis]
            num_outputs = int(num_outputs_attr.i)
            if axis_dim is None or num_outputs <= 0 or axis_dim % num_outputs != 0:
                return []
            return [axis_dim // num_outputs] * num_outputs

        patched = False
        rewritten_nodes = []
        for node in model.graph.node:
            if node.op_type != "Split":
                rewritten_nodes.append(node)
                continue

            split_sections = _infer_split_sections(node)
            if not split_sections:
                raise KeyError(f"indices_or_sections missing and cannot infer split sections from {onnx_path}")

            axis_attr = next((attr for attr in node.attribute if attr.name == "axis"), None)
            axis = int(axis_attr.i) if axis_attr is not None else 0

            if len(node.output) > 1:
                current_start = 0
                for index, output_name in enumerate(node.output):
                    section = int(split_sections[index])
                    starts_name = f"{node.name}_starts_{index}"
                    ends_name = f"{node.name}_ends_{index}"
                    axes_name = f"{node.name}_axes_{index}"
                    steps_name = f"{node.name}_steps_{index}"
                    model.graph.initializer.extend(
                        [
                            numpy_helper.from_array(np.asarray([current_start], dtype=np.int64), starts_name),
                            numpy_helper.from_array(np.asarray([current_start + section], dtype=np.int64), ends_name),
                            numpy_helper.from_array(np.asarray([axis], dtype=np.int64), axes_name),
                            numpy_helper.from_array(np.asarray([1], dtype=np.int64), steps_name),
                        ]
                    )
                    rewritten_nodes.append(
                        helper.make_node(
                            "Slice",
                            inputs=[node.input[0], starts_name, ends_name, axes_name, steps_name],
                            outputs=[output_name],
                            name=f"{node.name}_slice_{index}",
                        )
                    )
                    current_start += section
                patched = True
                continue

            has_indices = any(attr.name == "indices_or_sections" for attr in node.attribute)
            if not has_indices:
                node.attribute.extend([helper.make_attribute("indices_or_sections", split_sections)])
                patched = True
            rewritten_nodes.append(node)

        model.graph.ClearField("node")
        model.graph.node.extend(rewritten_nodes)

        if not patched:
            raise KeyError(f"No runtime-compatible graph fixes were applied for {onnx_path}")

        fixed_path = onnx_path.with_name(f"{onnx_path.stem}.runtimefix.onnx")
        onnx.save(model, str(fixed_path))
        _HMONNX_RUNTIME_FIX_CACHE[cache_key] = fixed_path

    return HMONNXInference(str(fixed_path))


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


def build_conversation(case: str):
    image_path = str(SCRIPT_DIR / "data" / "cars.jpg")
    audio_path = str(SCRIPT_DIR / "data" / "cough.wav")

    if case == "text":
        return [
            {"role": "user", "content": [{"type": "text", "text": "请用一句话介绍你自己。"}]},
        ], False
    if case == "vision":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "请描述这张图。"},
                ],
            },
        ], False
    if case == "audio":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "请描述你听到了什么。"},
                ],
            },
        ], False
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ], True


def save_json(file_path: Path, data: Dict[str, Any]):
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _latest_matching(root_dir: Path, pattern: str, required_key: Optional[Any] = None) -> Optional[Path]:
    candidates = sorted(root_dir.rglob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if required_key is None:
        return candidates[0] if candidates else None

    if isinstance(required_key, (list, tuple, set)):
        required_keys = tuple(required_key)
    else:
        required_keys = (required_key,)

    for candidate in candidates:
        try:
            meta = load_json(candidate)
        except Exception:
            continue
        if any(key in meta for key in required_keys):
            return candidate
    return None


def discover_artifacts(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    discovered: Dict[str, Dict[str, Any]] = {}
    mapping = {
        "text": ("meta.json", "prefill_onnx"),
        "audio": ("meta_audio.json", "audio_encoder_onnx"),
        "vision": ("meta_vision.json", "vision_encoder_onnx"),
        "talker": ("meta_talker.json", "talker_prefill_onnx"),
        "talker_prediction": ("meta_talker_prediction.json", "talker_prediction_prefill_onnx"),
        "projection": ("meta_talker_projection.json", ("talker_projection_hmonnx", "hidden_projection_hmonnx")),
        "code2wav": ("meta_code2wav.json", "code2wav_hmonnx"),
    }
    for key, (pattern, required_key) in mapping.items():
        meta_path = _latest_matching(root_dir, pattern, required_key)
        if meta_path is None:
            continue
        meta = load_json(meta_path)
        meta["_meta_path"] = str(meta_path)
        meta["_root_dir"] = str(meta_path.parent)
        discovered[key] = meta
    return discovered


def _resolve_meta_path(meta: Dict[str, Any], key: str) -> Path:
    return Path(meta["_root_dir"]) / meta[key]


def _ensure_tensor(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None):
    if value is None:
        raise ValueError("Cannot convert None to tensor")
    
    # Handle model output objects
    if hasattr(value, 'last_hidden_state'):
        value = value.last_hidden_state
    elif hasattr(value, 'hidden_states'):
        value = value.hidden_states
    
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as e:
            raise ValueError(f"Cannot convert {type(value)} to tensor: {e}")
    
    tensor = tensor.to(device)
    if dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype)
    return tensor


def _extract_primary_output(output: Any):
    if output is None:
        raise ValueError("HMONNX session returned None output")
    
    # Handle model output objects with .last_hidden_state or similar attributes
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state
    if hasattr(output, 'hidden_states'):
        return output.hidden_states
    
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("HMONNX session returned empty output list/tuple")
        return output[0]
    return output


def _get_modal_token_id(model_config: Any, attr_name: str) -> int:
    if hasattr(model_config, attr_name):
        return int(getattr(model_config, attr_name))
    thinker_config = getattr(model_config, "thinker_config", None)
    if thinker_config is not None and hasattr(thinker_config, attr_name):
        return int(getattr(thinker_config, attr_name))
    raise AttributeError(f"Cannot resolve {attr_name} from model config")


def _extract_outputs(output: Any):
    if isinstance(output, (list, tuple)):
        return list(output)
    return [output]


def _build_dense_deepstack_tensors(
    inputs_embeds: torch.Tensor,
    image_mask: torch.Tensor,
    deepstack_outputs: list[Any],
) -> list[torch.Tensor]:
    dense_tensors = []
    for deepstack_output in deepstack_outputs:
        deepstack_tensor = _ensure_tensor(deepstack_output, torch.device("cpu"), torch.float16)
        dense_tensor = torch.zeros_like(inputs_embeds, dtype=torch.float16)
        dense_tensor[image_mask] = deepstack_tensor.to(dense_tensor.dtype)
        dense_tensors.append(dense_tensor)
    return dense_tensors


def _ensure_hm_pixel_values(inputs: Dict[str, Any]) -> Dict[str, Any]:
    if "hm_pixel_values" not in inputs and "pixel_values" in inputs:
        inputs["hm_pixel_values"] = inputs["pixel_values"]
    if "hm_pixel_values_videos" not in inputs and "pixel_values_videos" in inputs:
        inputs["hm_pixel_values_videos"] = inputs["pixel_values_videos"]
    return inputs


def _prepare_vision_hmonnx_input(pixel_values: torch.Tensor, expected_shape) -> torch.Tensor:
    vision_input = pixel_values
    if vision_input.ndim == 6 and vision_input.shape[1] == 1:
        vision_input = vision_input[:, 0]
    if vision_input.ndim != 5:
        raise ValueError(f"Unexpected vision input rank: {vision_input.shape}")

    batch, channels, frames, height, width = vision_input.shape
    target_batch, target_channels, target_frames, target_height, target_width = [int(v) for v in expected_shape]

    if channels != target_channels:
        raise ValueError(f"Unexpected vision channels: got {channels}, expected {target_channels}")
    if frames != target_frames:
        if frames == 1 and target_frames > 1:
            vision_input = vision_input.repeat(1, 1, target_frames, 1, 1)
            frames = target_frames
        else:
            raise ValueError(f"Unexpected vision frames: got {frames}, expected {target_frames}")

    if height != target_height or width != target_width:
        vision_input = vision_input.to(torch.float32)
        vision_input = vision_input.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        vision_input = torch.nn.functional.interpolate(
            vision_input,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        vision_input = vision_input.reshape(batch, frames, channels, target_height, target_width).permute(0, 2, 1, 3, 4)

    if batch != target_batch:
        if batch == 1 and target_batch == 1:
            pass
        else:
            raise ValueError(f"Unexpected vision batch: got {batch}, expected {target_batch}")

    return vision_input.to(torch.float16)


def _make_cache_state(kv_cache_info: Dict[str, Any]):
    kv_shape = kv_cache_info["shape"]
    num_layers = kv_cache_info["num_decoder_layers"]
    return {
        "past_seq_length": 0,
        "past_key_caches": [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)],
        "past_value_caches": [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)],
    }


def _replace_code2wav(native_model, code2wav_hmonnx_path: Path, static_code_len: int, logger):
    session = _create_hmonnx_session(code2wav_hmonnx_path)
    native_model.code2wav.hmonnx = session
    native_model.code2wav.hmonnx_max_code_len = static_code_len

    def forward(self, codes):
        max_code_len = int(self.hmonnx_max_code_len)
        code_len = int(codes.shape[-1])
        if code_len > max_code_len:
            raise ValueError(f"code2wav hmonnx max code len is {max_code_len}, but got {code_len}")
        hmonnx_input = codes.to(torch.int32)
        if code_len < max_code_len:
            hmonnx_input = torch.nn.functional.pad(hmonnx_input, (0, max_code_len - code_len))
        wav = self.hmonnx.forward(hmonnx_input)
        wav = _extract_primary_output(wav)
        wav = _ensure_tensor(wav, codes.device)
        return wav[..., : code_len * self.total_upsample]

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        max_code_len = int(self.hmonnx_max_code_len)
        safe_chunk = min(chunk_size, max(1, max_code_len - left_context_size))
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + safe_chunk, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            chunk_token_len = end_index - start_index
            if chunk_token_len + context_size > max_code_len:
                context_size = max(0, max_code_len - chunk_token_len)
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.forward(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    native_model.code2wav.forward = types.MethodType(forward, native_model.code2wav)
    native_model.code2wav.chunked_decode = types.MethodType(chunked_decode, native_model.code2wav)
    logger.info(f"code2wav replaced with HMONNX: {code2wav_hmonnx_path}")


def _replace_projection(module, hmonnx_path: Path, name: str, logger):
    session = _create_hmonnx_session(hmonnx_path)
    module._hmonnx_session = session
    input_shape = getattr(session.inputs[0], "shape", None)
    static_seq_len = None
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
        try:
            static_seq_len = int(input_shape[1])
        except (TypeError, ValueError):
            static_seq_len = None

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        try:
            # Handle 2D input [seq_len, hidden_dim]
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(0)
            
            hmonnx_input = hidden_states.cpu().to(torch.float16)
            if static_seq_len is not None and static_seq_len > 0 and hmonnx_input.shape[1] != static_seq_len:
                outputs = []
                for i in range(hmonnx_input.shape[1]):
                    step_in = hmonnx_input[:, i : i + 1, :]
                    step_out = self._hmonnx_session.forward(step_in)
                    step_out = _extract_primary_output(step_out)
                    step_out = _ensure_tensor(step_out, hidden_states.device, hidden_states.dtype)
                    if step_out.ndim == 2:
                        step_out = step_out.unsqueeze(1)
                    elif step_out.ndim == 1:
                        step_out = step_out.unsqueeze(0).unsqueeze(0)
                    outputs.append(step_out)
                out = torch.cat(outputs, dim=1)
            else:
                out = self._hmonnx_session.forward(hmonnx_input)
                out = _extract_primary_output(out)
                out = _ensure_tensor(out, hidden_states.device, hidden_states.dtype)
                if out.ndim == 2:
                    out = out.unsqueeze(1)
                elif out.ndim == 1:
                    out = out.unsqueeze(0).unsqueeze(0)
            
            # Reshape back to original input shape
            if len(original_shape) == 2 and out.shape[0] == 1:
                out = out.squeeze(0)
            
            return out
        except Exception as e:
            print(f"[ERROR] Projection forward failed: input_shape={original_shape}, error={e}")
            raise

    module.forward = types.MethodType(forward, module)
    logger.info(f"{name} replaced with HMONNX: {hmonnx_path}")


def _replace_projection_bundle(native_model, projection_hmonnx_path: Path, logger):
    session = _create_hmonnx_session(projection_hmonnx_path)

    def _patch_projection(module, output_index: int, name: str):
        module._hmonnx_session = session
        module._hmonnx_output_index = output_index
        input_shape = getattr(session.inputs[0], "shape", None)
        static_seq_len = None
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            try:
                static_seq_len = int(input_shape[1])
            except (TypeError, ValueError):
                static_seq_len = None

        def forward(self, hidden_states):
            original_shape = hidden_states.shape
            try:
                if hidden_states.ndim == 2:
                    hidden_states = hidden_states.unsqueeze(0)

                hmonnx_input = hidden_states.cpu().to(torch.float16)
                if static_seq_len is not None and static_seq_len > 0 and hmonnx_input.shape[1] != static_seq_len:
                    outputs = []
                    for i in range(hmonnx_input.shape[1]):
                        step_in = hmonnx_input[:, i : i + 1, :]
                        step_outputs = self._hmonnx_session.forward(step_in)
                        if not isinstance(step_outputs, (list, tuple)) or len(step_outputs) <= self._hmonnx_output_index:
                            raise RuntimeError("Projection bundle returned unexpected outputs")
                        step_out = step_outputs[self._hmonnx_output_index]
                        step_out = _ensure_tensor(step_out, hidden_states.device, hidden_states.dtype)
                        if step_out.ndim == 2:
                            step_out = step_out.unsqueeze(1)
                        elif step_out.ndim == 1:
                            step_out = step_out.unsqueeze(0).unsqueeze(0)
                        outputs.append(step_out)
                    out = torch.cat(outputs, dim=1)
                else:
                    bundle_outputs = self._hmonnx_session.forward(hmonnx_input)
                    if not isinstance(bundle_outputs, (list, tuple)) or len(bundle_outputs) <= self._hmonnx_output_index:
                        raise RuntimeError("Projection bundle returned unexpected outputs")
                    out = bundle_outputs[self._hmonnx_output_index]
                    out = _ensure_tensor(out, hidden_states.device, hidden_states.dtype)
                    if out.ndim == 2:
                        out = out.unsqueeze(1)
                    elif out.ndim == 1:
                        out = out.unsqueeze(0).unsqueeze(0)

                if len(original_shape) == 2 and out.shape[0] == 1:
                    out = out.squeeze(0)

                return out
            except Exception as e:
                print(f"[ERROR] {name} forward failed: input_shape={original_shape}, error={e}")
                raise

        module.forward = types.MethodType(forward, module)
        logger.info(f"{name} replaced with projection bundle output {output_index}: {projection_hmonnx_path}")

    _patch_projection(native_model.talker.hidden_projection, 0, "hidden_projection")
    _patch_projection(native_model.talker.text_projection, 1, "text_projection")


def _patch_multimodal_encoders(native_model, audio_hmonnx_path: Optional[Path], vision_hmonnx_path: Optional[Path], logger):
    thinker = native_model.thinker
    thinker.forward = types.MethodType(Qwen3OmniMoeThinkerForConditionalGeneration_forward, thinker)

    if audio_hmonnx_path is not None:
        thinker._audio_hmonnx_session = _create_hmonnx_session(audio_hmonnx_path)

        def get_audio_features(self, input_features, feature_attention_mask=None, audio_feature_lengths=None):
            original_device = input_features.device
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

            feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
            aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
            n_window = self.audio_tower.n_window
            n_window_infer = self.audio_tower.n_window_infer

            chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
            chunk_lengths = torch.tensor(
                [n_window * 2] * int(chunk_num.sum().item()),
                dtype=torch.long,
                device=feature_lens.device,
            )
            tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
            chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
            chunk_lengths[chunk_lengths == 0] = n_window * 2

            chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
            padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
            feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
            padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
                [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
                batch_first=True,
            )

            cu_chunk_lens = [0]
            window_aftercnn = padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window * 2))
            for cnn_len in aftercnn_lens:
                cu_chunk_lens += [window_aftercnn] * int(cnn_len.item() // window_aftercnn)
                remainder = int(cnn_len.item() % window_aftercnn)
                if remainder != 0:
                    cu_chunk_lens += [remainder]
            cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

            # Process one chunk at a time (HMONNX exported with batch_size=1)
            all_outputs = []
            for i in range(padded_feature.shape[0]):
                single_feature = padded_feature[i:i+1].cpu().to(torch.float16)  # [1, mel, len]
                single_cu = torch.tensor([0, int(feature_lens_after_cnn[i])], dtype=torch.int32)
                out_i = self._audio_hmonnx_session.forward(single_feature, single_cu)
                out_i = _extract_primary_output(out_i)
                out_i = _ensure_tensor(out_i, original_device, torch.float16)
                out_i = out_i[: int(feature_lens_after_cnn[i])]
                all_outputs.append(out_i)
            output = torch.cat(all_outputs, dim=0) if len(all_outputs) > 1 else all_outputs[0]
            return output

        thinker.get_audio_features = types.MethodType(get_audio_features, thinker)
        logger.info(f"audio encoder replaced with HMONNX: {audio_hmonnx_path}")

    if vision_hmonnx_path is not None:
        thinker._vision_hmonnx_session = _create_hmonnx_session(vision_hmonnx_path)
        thinker._vision_hmonnx_input_shape = thinker._vision_hmonnx_session.inputs[0].shape

        def get_image_features(self, pixel_values, image_grid_thw=None):
            try:
                hmonnx_input = _prepare_vision_hmonnx_input(pixel_values, self._vision_hmonnx_input_shape)
                output = self._vision_hmonnx_session.forward(hmonnx_input.cpu())
                all_outputs = _extract_outputs(output)
                image_embeds = _ensure_tensor(all_outputs[0], pixel_values.device, torch.float16)
                deepstack_features = tuple(
                    _ensure_tensor(ds, pixel_values.device, torch.float16)
                    for ds in all_outputs[1:4]
                ) if len(all_outputs) > 1 else ()
                return image_embeds, deepstack_features
            except Exception as e:
                print(f"[ERROR] get_image_features failed: {e}")
                raise

        def get_video_features(self, pixel_values_videos, video_grid_thw=None):
            try:
                hmonnx_input = _prepare_vision_hmonnx_input(pixel_values_videos, self._vision_hmonnx_input_shape)
                output = self._vision_hmonnx_session.forward(hmonnx_input.cpu())
                all_outputs = _extract_outputs(output)
                video_embeds = _ensure_tensor(all_outputs[0], pixel_values_videos.device, torch.float16)
                deepstack_features = tuple(
                    _ensure_tensor(ds, pixel_values_videos.device, torch.float16)
                    for ds in all_outputs[1:4]
                ) if len(all_outputs) > 1 else ()
                return video_embeds, deepstack_features
            except Exception as e:
                print(f"[ERROR] get_video_features failed: {e}")
                raise

        thinker.get_image_features = types.MethodType(get_image_features, thinker)
        thinker.get_video_features = types.MethodType(get_video_features, thinker)
        logger.info(f"vision encoder replaced with HMONNX: {vision_hmonnx_path}")


def _patch_talker_shadow(native_model, talker_meta: Dict[str, Any], logger):
    prefill_session = _create_hmonnx_session(_resolve_meta_path(talker_meta, "talker_prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(talker_meta, "talker_decode_onnx"))
    state = _make_cache_state(talker_meta["talker_kv_cache"])
    static_prefill_len = int(talker_meta.get("talker_input_sequence_length", 0))
    original_forward = native_model.talker.forward

    def forward(self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])
            if seq_len > 1:
                state["past_seq_length"] = 0
                state["past_key_caches"] = _make_cache_state(talker_meta["talker_kv_cache"])["past_key_caches"]
                state["past_value_caches"] = _make_cache_state(talker_meta["talker_kv_cache"])["past_value_caches"]
            session = prefill_session if seq_len > 1 else decode_session
            shadow_inputs_embeds = inputs_embeds.detach().cpu().to(torch.float16)
            shadow_seq_len = seq_len
            if seq_len > 1 and static_prefill_len > 0:
                if seq_len > static_prefill_len:
                    raise ValueError(f"talker shadow seq_len {seq_len} exceeds exported static length {static_prefill_len}")
                if seq_len < static_prefill_len:
                    pad = torch.zeros(
                        shadow_inputs_embeds.shape[0],
                        static_prefill_len - seq_len,
                        shadow_inputs_embeds.shape[2],
                        dtype=shadow_inputs_embeds.dtype,
                    )
                    shadow_inputs_embeds = torch.cat([shadow_inputs_embeds, pad], dim=1)
                    shadow_seq_len = static_prefill_len
            session.forward(
                shadow_inputs_embeds,
                torch.tensor([state["past_seq_length"]], dtype=torch.int32),
                torch.tensor([shadow_seq_len], dtype=torch.int32),
                *state["past_key_caches"],
                *state["past_value_caches"],
            )
            state["past_seq_length"] += seq_len
        # Ensure attention_mask is not None (generate may skip it when pad==eos)
        if kwargs.get("attention_mask") is None and kwargs.get("inputs_embeds") is not None:
            ie = kwargs["inputs_embeds"]
            past_kv = kwargs.get("past_key_values")
            if past_kv is not None and hasattr(past_kv, "get_seq_length"):
                past_len = past_kv.get_seq_length()
            elif past_kv is not None and isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
                past_len = past_kv[0][0].shape[2]
            else:
                past_len = 0
            total_len = past_len + ie.shape[1]
            kwargs["attention_mask"] = torch.ones(ie.shape[0], total_len, device=ie.device, dtype=torch.long)
        return original_forward(*args, **kwargs)

    native_model.talker.forward = types.MethodType(forward, native_model.talker)
    # Skip custom kwarg validation in talker.generate()
    native_model.talker._validate_model_kwargs = types.MethodType(
        lambda self, model_kwargs: None, native_model.talker
    )
    logger.info("talker model inserted in shadow mode")


def _patch_talker_prediction_shadow(native_model, predictor_meta: Dict[str, Any], logger):
    prefill_session = _create_hmonnx_session(_resolve_meta_path(predictor_meta, "talker_prediction_prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(predictor_meta, "talker_prediction_decode_onnx"))
    state = _make_cache_state(predictor_meta["talker_prediction_kv_cache"])
    static_prefill_len = int(predictor_meta.get("talker_prediction_input_sequence_length", 0))
    original_forward = native_model.talker.code_predictor.forward

    def forward(self, *args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            seq_len = int(inputs_embeds.shape[1])
            if seq_len > 1:
                state["past_seq_length"] = 0
                state["past_key_caches"] = _make_cache_state(predictor_meta["talker_prediction_kv_cache"])["past_key_caches"]
                state["past_value_caches"] = _make_cache_state(predictor_meta["talker_prediction_kv_cache"])["past_value_caches"]
            session = prefill_session if seq_len > 1 else decode_session
            shadow_inputs_embeds = inputs_embeds.detach().cpu().to(torch.float16)
            shadow_seq_len = seq_len
            if seq_len > 1 and static_prefill_len > 0:
                if seq_len > static_prefill_len:
                    raise ValueError(
                        f"talker prediction shadow seq_len {seq_len} exceeds exported static length {static_prefill_len}"
                    )
                if seq_len < static_prefill_len:
                    pad = torch.zeros(
                        shadow_inputs_embeds.shape[0],
                        static_prefill_len - seq_len,
                        shadow_inputs_embeds.shape[2],
                        dtype=shadow_inputs_embeds.dtype,
                    )
                    shadow_inputs_embeds = torch.cat([shadow_inputs_embeds, pad], dim=1)
                    shadow_seq_len = static_prefill_len
            session.forward(
                shadow_inputs_embeds,
                torch.tensor([state["past_seq_length"]], dtype=torch.int32),
                torch.tensor([shadow_seq_len], dtype=torch.int32),
                *state["past_key_caches"],
                *state["past_value_caches"],
            )
            state["past_seq_length"] += seq_len
        # Ensure attention_mask is not None (generate may skip it when pad==eos)
        if kwargs.get("attention_mask") is None and kwargs.get("inputs_embeds") is not None:
            ie = kwargs["inputs_embeds"]
            past_kv = kwargs.get("past_key_values")
            if past_kv is not None and hasattr(past_kv, "get_seq_length"):
                past_len = past_kv.get_seq_length()
            elif past_kv is not None and isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
                past_len = past_kv[0][0].shape[2]
            else:
                past_len = 0
            total_len = past_len + ie.shape[1]
            kwargs["attention_mask"] = torch.ones(ie.shape[0], total_len, device=ie.device, dtype=torch.long)
        return original_forward(*args, **kwargs)

    native_model.talker.code_predictor.forward = types.MethodType(forward, native_model.talker.code_predictor)
    # Skip custom kwarg validation in code_predictor.model.generate()
    native_model.talker.code_predictor.model._validate_model_kwargs = types.MethodType(
        lambda self, model_kwargs: None, native_model.talker.code_predictor.model
    )
    logger.info("talker prediction inserted in shadow mode")


def apply_artifact_replacements(native_model, artifacts: Dict[str, Dict[str, Any]], logger):
    if "code2wav" in artifacts:
        _replace_code2wav(
            native_model,
            _resolve_meta_path(artifacts["code2wav"], "code2wav_hmonnx"),
            int(artifacts["code2wav"]["static_code_len"]),
            logger,
        )

    if "projection" in artifacts:
        projection_meta = artifacts["projection"]
        if "talker_projection_hmonnx" in projection_meta:
            _replace_projection_bundle(
                native_model,
                _resolve_meta_path(projection_meta, "talker_projection_hmonnx"),
                logger,
            )
        else:
            _replace_projection(
                native_model.talker.hidden_projection,
                _resolve_meta_path(projection_meta, "hidden_projection_hmonnx"),
                "hidden_projection",
                logger,
            )
            _replace_projection(
                native_model.talker.text_projection,
                _resolve_meta_path(projection_meta, "text_projection_hmonnx"),
                "text_projection",
                logger,
            )

    audio_hmonnx_path = None
    if "audio" in artifacts:
        audio_hmonnx_path = _resolve_meta_path(artifacts["audio"], "audio_encoder_onnx")

    vision_hmonnx_path = None
    if "vision" in artifacts:
        vision_hmonnx_path = _resolve_meta_path(artifacts["vision"], "vision_encoder_onnx")

    if audio_hmonnx_path is not None or vision_hmonnx_path is not None:
        _patch_multimodal_encoders(native_model, audio_hmonnx_path, vision_hmonnx_path, logger)

    if "talker" in artifacts:
        _patch_talker_shadow(native_model, artifacts["talker"], logger)

    if "talker_prediction" in artifacts:
        _patch_talker_prediction_shadow(native_model, artifacts["talker_prediction"], logger)


def run_dialogue_validation(
    model_path: str,
    work_dir: Path,
    logger,
    case: str,
    max_new_tokens: int = 64,
    device_map: str = "auto",
    max_memory: Optional[Dict[Any, str]] = None,
    artifacts: Optional[Dict[str, Dict[str, Any]]] = None,
    report_name: str = "dialogue_validation.json",
    output_prefix: str = "dialogue",
    talker_max_new_tokens: Optional[int] = None,
):
    device_map = _resolve_validation_device_map(device_map, logger)
    if device_map == "auto" and max_memory is None:
        max_memory = _build_safe_validation_max_memory(logger)

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs,
    )
    native_model.eval()
    _patch_inputs_embeds_generation_device(native_model.talker, "talker", logger)
    _patch_inputs_embeds_generation_device(native_model.talker.code_predictor, "talker.code_predictor", logger)
    if hasattr(native_model, "code2wav"):
        _patch_runtime_device_property(native_model.code2wav, "code2wav", logger)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    if artifacts:
        apply_artifact_replacements(native_model, artifacts, logger)

    conversation, use_audio_in_video = build_conversation(case)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        seconds_per_chunk=2.0,
        position_id_per_seconds=13,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = _ensure_hm_pixel_values(inputs)
    # When vision artifact replacement is active, monkey-patched thinker expects hm_* tensors.
    use_vision_hmonnx = artifacts is not None and "vision" in artifacts
    if not use_vision_hmonnx:
        # Pure HF path: keep only canonical keys to avoid unsupported kwargs in generate.
        inputs.pop("hm_pixel_values", None)
        inputs.pop("hm_pixel_values_videos", None)

    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype
    inputs = inputs.to(device).to(dtype)

    try:
        generate_kwargs = dict(
            **inputs,
            speaker="Ethan",
            thinker_return_dict_in_generate=True,
            use_audio_in_video=use_audio_in_video,
            max_new_tokens=max_new_tokens,
        )
        if talker_max_new_tokens is not None:
            generate_kwargs["talker_max_new_tokens"] = talker_max_new_tokens
        with torch.no_grad():
            text_ids, audio = native_model.generate(**generate_kwargs)
    except Exception as e:
        import traceback
        if logger is not None:
            logger.warning(f"dialogue validation generate failed: {e}")
            logger.warning(traceback.format_exc())
        raise

    sequences = text_ids.sequences if hasattr(text_ids, 'sequences') else text_ids
    output_text = processor.batch_decode(
        sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "case": case,
        "output_text": output_text,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "applied_artifacts": sorted(list(artifacts.keys())) if artifacts else [],
    }

    if audio is not None:
        wav_path = work_dir / f"{output_prefix}_{case}.wav"
        sf.write(str(wav_path), audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        report["audio_file"] = str(wav_path.relative_to(work_dir))

    report_path = work_dir / report_name
    save_json(report_path, report)
    logger.info(f"dialogue validation report saved to {report_path}")
    return report


def _run_audio_encoder_hmonnx(session: HMONNXInference, audio_tower, input_features, feature_attention_mask):
    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    feature_lens = audio_feature_lengths
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    n_window = audio_tower.n_window
    n_window_infer = audio_tower.n_window_infer
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [n_window * 2] * int(chunk_num.sum().item()),
        dtype=torch.long,
        device=feature_lens.device,
    )
    tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2
    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    cu_chunk_lens = [0]
    window_aftercnn = padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window * 2))
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * int(cnn_len.item() // window_aftercnn)
        remainder = int(cnn_len.item() % window_aftercnn)
        if remainder != 0:
            cu_chunk_lens += [remainder]
    cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
    # Process one chunk at a time (HMONNX exported with batch_size=1)
    all_outputs = []
    for i in range(padded_feature.shape[0]):
        single_feature = padded_feature[i:i+1].cpu().to(torch.float16)
        single_cu = torch.tensor([0, int(feature_lens_after_cnn[i])], dtype=torch.int32)
        out_i = session.forward(single_feature, single_cu)
        out_i = _extract_primary_output(out_i)
        out_i = _ensure_tensor(out_i, torch.device("cpu"), torch.float16)
        out_i = out_i[: int(feature_lens_after_cnn[i])]
        all_outputs.append(out_i)
    output = torch.cat(all_outputs, dim=0) if len(all_outputs) > 1 else all_outputs[0]
    return output


def run_text_hmonnx_chain_forward(
    model_path: str,
    text_meta: Dict[str, Any],
    logger,
    case: str = "multimodal",
    audio_meta: Optional[Dict[str, Any]] = None,
    vision_meta: Optional[Dict[str, Any]] = None,
    report_path: Optional[Path] = None,
    max_new_tokens: int = 256,
    device_map: str = "auto",
):
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    if logger is not None and device_map != "cpu":
        logger.info(
            f"text chain validation keeps HF model on cpu regardless of requested device_map={device_map}"
        )
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()

    conversation, use_audio_in_video = build_conversation(case)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        seconds_per_chunk=2.0,
        position_id_per_seconds=13,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = _ensure_hm_pixel_values(inputs)

    token_embedding_state_dict = torch.load(
        Path(text_meta["_root_dir"]) / text_meta["token_embedding_file"],
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(token_embedding_state_dict, dict) and "weight" in token_embedding_state_dict:
        token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        token_embedding.load_state_dict(token_embedding_state_dict)
    else:
        raise RuntimeError("token embedding file does not contain a standard state_dict")

    inputs_embeds = token_embedding(inputs["input_ids"].cpu())
    prefill_token_length = int(inputs_embeds.shape[1])

    if audio_meta is not None and "input_features" in inputs and "feature_attention_mask" in inputs:
        audio_session = _create_hmonnx_session(_resolve_meta_path(audio_meta, "audio_encoder_onnx"))
        audio_features = _run_audio_encoder_hmonnx(
            audio_session,
            native_model.thinker.audio_tower,
            inputs["input_features"].cpu(),
            inputs["feature_attention_mask"].cpu(),
        )
        audio_mask = inputs["input_ids"].cpu() == _get_modal_token_id(native_model.config, "audio_token_id")
        inputs_embeds[audio_mask] = audio_features.to(inputs_embeds.dtype)

    if vision_meta is not None and "hm_pixel_values" in inputs:
        vision_session = _create_hmonnx_session(_resolve_meta_path(vision_meta, "vision_encoder_onnx"))
        vision_hmonnx_input = _prepare_vision_hmonnx_input(inputs["hm_pixel_values"].cpu(), vision_session.inputs[0].shape)
        vision_output = vision_session.forward(vision_hmonnx_input.to(torch.float16))
        vision_outputs = _extract_outputs(vision_output)
        vision_embeds = _ensure_tensor(vision_outputs[0], torch.device("cpu"), torch.float16)
        image_mask = inputs["input_ids"].cpu() == _get_modal_token_id(native_model.config, "image_token_id")
        inputs_embeds[image_mask] = vision_embeds.to(inputs_embeds.dtype)
        deepstack_tensors = _build_dense_deepstack_tensors(inputs_embeds, image_mask, vision_outputs[1:4])
    else:
        deepstack_tensors = [torch.zeros_like(inputs_embeds, dtype=torch.float16) for _ in range(3)]

    kv_cache_info = text_meta["kv_cache"]
    kv_shape = kv_cache_info["shape"]
    num_layers = kv_cache_info["num_decoder_layers"]
    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)]

    prefill_session = _create_hmonnx_session(_resolve_meta_path(text_meta, "prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(text_meta, "decode_onnx"))

    input_sequence_length = int(text_meta["wrap_cfg"]["input_sequence_length"])
    if prefill_token_length > input_sequence_length:
        raise RuntimeError(
            f"text chain input length {prefill_token_length} exceeds exported input_sequence_length {input_sequence_length}"
        )
    if prefill_token_length < input_sequence_length:
        pad_embeds = torch.zeros(
            (inputs_embeds.shape[0], input_sequence_length - prefill_token_length, inputs_embeds.shape[2]),
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
        deepstack_tensors = [
            torch.cat([tensor, torch.zeros_like(pad_embeds, dtype=torch.float16)], dim=1) for tensor in deepstack_tensors
        ]

    current_input_length = torch.tensor([prefill_token_length], dtype=torch.int32)
    past_seq_length = torch.tensor([0], dtype=torch.int32)
    zero_decode_deepstack = [torch.zeros((1, 1, inputs_embeds.shape[2]), dtype=torch.float16) for _ in range(3)]

    prefill_inputs = [
        inputs_embeds.to(torch.float16),
        past_seq_length,
        current_input_length,
    ]
    text_supports_deepstack = len(prefill_session.inputs) == 3 + 3 + (2 * num_layers)
    if text_supports_deepstack:
        prefill_inputs.extend(deepstack_tensors)

    # --- Prefill ---
    prefill_logits = prefill_session.forward(
        *prefill_inputs,
        *past_key_caches,
        *past_value_caches,
    )
    prefill_logits = _ensure_tensor(_extract_primary_output(prefill_logits), torch.device("cpu"), torch.float32)
    if prefill_logits.ndim == 2:
        prefill_logits = prefill_logits.unsqueeze(1)
    next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)

    # Determine EOS token ids for stopping
    eos_token_id = processor.tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_ids = {eos_token_id}
    elif isinstance(eos_token_id, (list, tuple)):
        eos_token_ids = set(eos_token_id)
    else:
        eos_token_ids = set()
    # Also add common chat stop tokens if available
    if hasattr(native_model.config, "eos_token_id"):
        cfg_eos = native_model.config.eos_token_id
        if isinstance(cfg_eos, int):
            eos_token_ids.add(cfg_eos)
        elif isinstance(cfg_eos, (list, tuple)):
            eos_token_ids.update(cfg_eos)

    # --- Autoregressive decode loop ---
    generated_tokens = [next_token]  # first token from prefill
    decode_past_seq_length = current_input_length.clone()
    one_length = torch.ones_like(current_input_length)

    kv_max_seq = kv_shape[2] if len(kv_shape) > 2 else kv_shape[-1]

    for step in range(max_new_tokens - 1):
        token_id = int(next_token.item())
        if token_id in eos_token_ids:
            break
        # Check KV cache capacity
        if int(decode_past_seq_length.item()) + 1 > kv_max_seq:
            logger.warning(f"KV cache full at step {step + 1}, stopping decode")
            break

        decode_inputs = [
            token_embedding(next_token).to(torch.float16),
            decode_past_seq_length,
            one_length,
        ]
        if text_supports_deepstack:
            decode_inputs.extend(zero_decode_deepstack)
        decode_logits = decode_session.forward(
            *decode_inputs,
            *past_key_caches,
            *past_value_caches,
        )
        decode_logits = _ensure_tensor(_extract_primary_output(decode_logits), torch.device("cpu"), torch.float32)
        if decode_logits.ndim == 2:
            decode_logits = decode_logits.unsqueeze(1)
        next_token = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        decode_past_seq_length = decode_past_seq_length + 1

    # Concat all generated token ids and decode to text
    all_token_ids = torch.cat(generated_tokens, dim=-1)  # [1, num_tokens]
    output_text = processor.tokenizer.batch_decode(all_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    logger.info(f"generated {all_token_ids.shape[-1]} tokens: {output_text}")

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "case": case,
        "prefill_logits_shape": list(prefill_logits.shape),
        "num_generated_tokens": int(all_token_ids.shape[-1]),
        "output_text": output_text,
        "used_audio_hmonnx": audio_meta is not None,
        "used_vision_hmonnx": vision_meta is not None,
        "used_deepstack": text_supports_deepstack,
    }
    if report_path is not None:
        save_json(report_path, report)
        logger.info(f"text HMONNX chain report saved to {report_path}")
    return report


def validate_golden_outputs(golden_dir: Path, selected_cases):
    meta_file = golden_dir / "golden_meta.json"
    if not meta_file.exists() or meta_file.stat().st_size == 0:
        raise RuntimeError(f"missing or empty golden meta file: {meta_file}")

    meta = load_json(meta_file)
    missing = []
    for case_name in selected_cases:
        ids_file = golden_dir / f"golden_{case_name}_ids.pt"
        if not ids_file.exists() or ids_file.stat().st_size == 0:
            missing.append(str(ids_file))
        if case_name in ("audio", "multimodal"):
            audio_file = golden_dir / f"golden_{case_name}.wav"
            if not audio_file.exists() or audio_file.stat().st_size == 0:
                missing.append(str(audio_file))
        if case_name not in meta.get("results", {}):
            missing.append(f"golden_meta.json::results::{case_name}")

    if missing:
        raise RuntimeError("golden outputs incomplete: " + ", ".join(missing))

    return {
        "golden_dir": str(golden_dir),
        "validated_cases": selected_cases,
        "result_count": len(meta.get("results", {})),
    }
