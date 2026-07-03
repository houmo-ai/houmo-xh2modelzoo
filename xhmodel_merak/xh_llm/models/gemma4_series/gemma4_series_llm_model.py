"""Unified public Gemma4 Series LLM model class."""

from __future__ import annotations

import copy
import gc
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import onnx
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.types import ExportData, KVCacheConfig, LLMModelState, ModelSwitcher
from xhmodel_merak.xh_llm.vision_llm_model import VisionLLMModel
from xhquant.utils import get_xhquant_logger, log_function_call

from .data_preprocess import (
    Gemma4DataPreprocess,
    Gemma4MoeDataPreprocess,
    Gemma4PerLayerInputEmbedding,
)
from .llm_text import (
    Gemma4KVCacheMixin,
    _Gemma4DecodeNoFullMaskBridge,
    _Gemma4DecodeNoFullMaskMTPBridge,
    _Gemma4TextExportBridgePLE,
    _Gemma4TextExportBridgePLEMTPDecode,
    _copy_model_shared_params,
    _gemma4_cache_seq_len_for_layer,
    _make_text_export_bridge_if_needed,
    build_gemma4_hf_compatible_model,
)
from .gemma4_series_audio_model import XHGemma4SeriesAudioModel
from .gemma4_series_hmonnx_inference import XHGemma4SeriesHMONNXModel
from .gemma4_series_vision_model import XHGemma4SeriesVisionModel
from .xh_gemma4_series_config import Gemma4SeriesModelMeta, XHGemma4SeriesModelConfig


def _is_mtp_mode(value: Any) -> bool:
    return str(value).lower() == "mtp"


def _cfg_get_value(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        value = getattr(cfg, key)
        return default if value is None else value
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
        return default if value is None else value
    return default


def _cfg_set_value(cfg: Any, key: str, value: Any) -> None:
    if hasattr(cfg, key):
        setattr(cfg, key, value)
    else:
        cfg[key] = value


def _quant_type_weight_bits(quant_type: Any, default: int = 4) -> int:
    if not quant_type:
        return default
    text = str(quant_type)
    if not text.startswith("w"):
        return default
    digits = []
    for char in text[1:]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else default


def _mtp_draft_head_weight_bits(cfg: Any, default: int = 4) -> int:
    mtp_config = _cfg_get_value(cfg, "mtp_config", None)
    lm_head_quant_type = _cfg_get_value(mtp_config, "lm_head_quant_type", None) if mtp_config is not None else None
    return _quant_type_weight_bits(lm_head_quant_type, default)

def _strip_default_llmcache_only_handle_old_cache_attrs(onnx_file: str | Path) -> int:
    """Remove explicit default-false LLMCache attrs from exported ONNX.

    xhquant's parser defaults ``only_handle_old_cache`` to false when the attr
    is absent.  Keeping explicit ``0`` bloats the graph and makes new Gemma4
    exports look different from older models for no semantic reason.  Do not
    remove explicit true values; assistant draft graphs rely on them to mark
    read-only shared KV inputs.
    """

    onnx_path = Path(onnx_file)
    if not onnx_path.exists():
        return 0
    # Graph attrs live in the main protobuf; do not load external weights.
    # Loading weights here would make large 26B/31B exports expensive and risks
    # rewriting external tensors when the only required change is metadata.
    model = onnx.load(str(onnx_path), load_external_data=False)
    removed = 0
    for node in model.graph.node:
        if node.op_type != "KVcache":
            continue
        kept_attrs = []
        for attr in node.attribute:
            if attr.name in {"only_handle_old_cache", "only-handle-old-cache"}:
                value = onnx.helper.get_attribute_value(attr)
                if value in (0, False):
                    removed += 1
                    continue
            kept_attrs.append(attr)
        if len(kept_attrs) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(kept_attrs)
    if removed:
        onnx.save(model, str(onnx_path))
    return removed


@register_llm_model("Gemma4ForConditionalGeneration", force=True)
class XHGemma4SeriesModel(VisionLLMModel):
    """Single public Gemma4 entry for E4B, 31B dense, and 26B-A4B MoE.

    Public owner for shared Gemma4 construction. Implementation details are
    being migrated into this package phase-by-phase while preserving the
    verified export behavior.
    """

    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.gemma4_series.workflow:XHGemma4SeriesHMONNXWorkflow"
    transformers_min_version = "5.5.0"
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Gemma4SeriesModelMeta
    HMONNXINFERENCE_CLS = XHGemma4SeriesHMONNXModel
    CONFIG_CLS = XHGemma4SeriesModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_gemma4_hf_compatible_model)

    @classmethod
    def from_pretrained(cls, config: XHGemma4SeriesModelConfig):
        # Keep the public model type anchored in gemma4_series.  The inherited
        # legacy class used to return a private gemma4_moe subclass for 26B-A4B,
        # which made the public registration look unified while construction
        # still escaped this package.
        return cls(config)

    def __init__(self, config: XHGemma4SeriesModelConfig):
        # Intentionally bypass the legacy XHGemma4Model.__init__ because it
        # constructs visual/audio submodels from gemma4/gemma4e.  The series
        # package owns those children so later audio/video/PLE changes have one
        # implementation site.
        VisionLLMModel.__init__(self, config)
        self.config = cast(XHGemma4SeriesModelConfig, self.config)
        self.visual = (
            XHGemma4SeriesVisionModel(config.visual_config)
            if config.visual_config is not None
            else None
        )
        self.video_visual = (
            XHGemma4SeriesVisionModel(config.video_visual_config)
            if config.video_visual_config is not None
            else None
        )
        self.audio = (
            XHGemma4SeriesAudioModel(config.audio_config)
            if config.audio_config is not None
            else None
        )
        self.per_layer_input_embedding: Gemma4PerLayerInputEmbedding | None = None
        self._kvcache_config = KVCacheConfig()
        self._kvcache_config.use_cache = self.config.use_cache
        self._kvcache_mixin = Gemma4KVCacheMixin(self.kvcache_config)
        self._decode_input_sequence_length = self._mtp_verify_length() if self._is_mtp_export() else 1
        self._validate_mtp_sliding_kv_cache_input_mode()
        if self._is_mtp_export():
            self._decode_wrap_cfg_overrides = {"num_logits_to_keep": 0}

    def _is_mtp_export(self) -> bool:
        return _is_mtp_mode(getattr(self.config, "spec_decode_mode", None))

    def _uses_target_verify_decode_accepted_count(self) -> bool:
        return (
            self._is_mtp_export()
            and getattr(self.config, "sliding_kv_cache_input_mode", "slice_window") == "slice_window"
            and self.is_decode()
        )

    def _validate_mtp_sliding_kv_cache_input_mode(self) -> None:
        if (
            self._is_mtp_export()
            and getattr(self.config, "sliding_kv_cache_input_mode", "slice_window") != "slice_window"
        ):
            raise ValueError(
                "Gemma4 Series MTP requires sliding_kv_cache_input_mode='slice_window'; "
                f"got {getattr(self.config, 'sliding_kv_cache_input_mode', None)!r}."
            )

    def _mtp_num_draft_tokens(self) -> int:
        return int(getattr(self.config, "num_draft_tokens", None) or 4)

    def _mtp_verify_length(self) -> int:
        return self._mtp_num_draft_tokens() + 1

    def _mtp_shared_sliding_cache_length(self) -> int:
        return _gemma4_cache_seq_len_for_layer(
            layer_type="sliding_attention",
            context_max_length=self.config.context_max_length,
            sliding_window=self.sliding_window,
            input_seq_len=self._prefill_cache_input_length(),
            sliding_kv_cache_input_mode=getattr(
                self.config, "sliding_kv_cache_input_mode", "slice_window"
            ),
        )

    def _mtp_target_decode_sliding_output_length(self) -> int:
        verify_length = self._mtp_verify_length()
        return ((int(self.sliding_window) + verify_length - 1 + 15) // 16) * 16

    def _mtp_decode_wrap_cfg_overrides(self) -> dict[str, int]:
        return {"num_logits_to_keep": 0, "enable_accepted_count_input": True}

    def _prefill_cache_input_length(self) -> int:
        return int(getattr(self.config, "prefill_chunk_length", 320))

    def _apply_wrap_cfg_to_modules(self, module: nn.Module) -> None:
        def _update(child: nn.Module) -> None:
            update_cfg = getattr(child, "_update_cfg", None)
            if update_cfg is not None:
                update_cfg(self.wrap_cfg)

        module.apply(_update)

    def _apply_mtp_phase_wrap_cfg(self, *, decode: bool) -> None:
        if not self._is_mtp_export():
            return
        input_sequence_length = self._mtp_verify_length() if decode else self.config.prefill_chunk_length
        _cfg_set_value(self.wrap_cfg, "input_sequence_length", input_sequence_length)
        _cfg_set_value(
            self.wrap_cfg,
            "num_logits_to_keep",
            0 if decode else self.config.num_logits_to_keep,
        )
        if self._data_processor is not None:
            self._data_processor.input_sequence_length = input_sequence_length
        self.update_cfg(self.wrap_cfg)

    def set_prefill(self):
        super().set_prefill()
        self._apply_mtp_phase_wrap_cfg(decode=False)

    def set_decode(self):
        super().set_decode()
        self._apply_mtp_phase_wrap_cfg(decode=True)

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        if self.visual is not None:
            self.visual.work_dir = str(Path(work_dir) / "visual")
        if self.video_visual is not None:
            self.video_visual.work_dir = str(Path(work_dir) / "video_visual")
        if self.audio is not None:
            self.audio.work_dir = str(Path(work_dir) / "audio")

    def get_tf_processor(self):
        if self.visual is not None:
            processor = self.visual.get_tf_processor()
        elif self.video_visual is not None:
            processor = self.video_visual.get_tf_processor()
        else:
            processor = super().get_tf_processor()
        if self.video_visual is not None:
            processor.video_max_patches = self.video_visual.config.max_patches
            processor.video_image_seq_length = self.video_visual.config.image_seq_length
            processor.video_pooling_kernel_size = self.video_visual.config.pooling_kernel_size
        if self.audio is not None:
            processor.config.sampling_rate = self.audio.config.sampling_rate
            processor.config.audio_feature_length = self.audio.config.input_feature_length
        return processor

    def _get_language_model(self, hf_model: Any) -> Any:
        if hasattr(hf_model, "language_model"):
            return hf_model.language_model
        return hf_model.model.language_model

    @staticmethod
    def _is_gguf_quant_weight_path(quant_weight_path: str | None) -> bool:
        if not quant_weight_path:
            return False
        path = Path(quant_weight_path)
        if path.is_file():
            return path.suffix.lower() == ".gguf"
        if path.is_dir():
            return any(child.suffix.lower() == ".gguf" for child in path.iterdir())
        return str(quant_weight_path).lower().endswith(".gguf")

    @staticmethod
    def _resolve_gguf_artifact_files(gguf_path: str) -> tuple[str, str | None]:
        path = Path(gguf_path)
        if path.is_file():
            if path.suffix.lower() != ".gguf":
                raise ValueError(f"GGUF quant_weight must be a .gguf file or directory, got: {gguf_path}")
            sibling_ggufs = sorted(
                child for child in path.parent.iterdir() if child.is_file() and child.suffix.lower() == ".gguf"
            )
            if "mmproj" in path.name.lower():
                main_files = [child for child in sibling_ggufs if "mmproj" not in child.name.lower()]
                if len(main_files) != 1:
                    candidate_list = ", ".join(str(child) for child in main_files)
                    raise ValueError(
                        f"GGUF mmproj file path requires exactly one sibling non-mmproj .gguf file, got "
                        f"{len(main_files)}. Candidates: {candidate_list}"
                    )
                return str(main_files[0]), str(path)
            mmproj_files = [child for child in sibling_ggufs if "mmproj" in child.name.lower()]
            if len(mmproj_files) > 1:
                candidate_list = ", ".join(str(child) for child in mmproj_files)
                raise ValueError(
                    f"GGUF file path has multiple sibling mmproj .gguf files; pass an artifact directory with "
                    f"a single mmproj file or remove ambiguity. Candidates: {candidate_list}"
                )
            return str(path), str(mmproj_files[0]) if mmproj_files else None
        if not path.is_dir():
            raise FileNotFoundError(f"GGUF quant_weight path does not exist: {gguf_path}")

        gguf_files = sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() == ".gguf")
        if not gguf_files:
            raise FileNotFoundError(f"GGUF quant_weight directory does not contain any .gguf files: {gguf_path}")
        mmproj_files = [child for child in gguf_files if "mmproj" in child.name.lower()]
        main_files = [child for child in gguf_files if child not in mmproj_files]
        if len(main_files) != 1:
            candidate_list = ", ".join(str(child) for child in main_files)
            raise ValueError(
                f"GGUF quant_weight directory must contain exactly one non-mmproj .gguf file, got "
                f"{len(main_files)}. Candidates: {candidate_list}"
            )
        if len(mmproj_files) > 1:
            candidate_list = ", ".join(str(child) for child in mmproj_files)
            raise ValueError(
                f"GGUF quant_weight directory contains multiple mmproj .gguf files; pass one artifact directory "
                f"with a single mmproj file. Candidates: {candidate_list}"
            )
        return str(main_files[0]), str(mmproj_files[0]) if mmproj_files else None

    @classmethod
    def _load_hf_model_from_gguf(cls, hf_model_dir: str, gguf_path: str, **kwargs) -> nn.Module:
        try:
            from accelerate import init_empty_weights
            from accelerate.utils.modeling import set_module_tensor_to_device
        except ImportError as e:
            raise ImportError("Loading GGUF quant_weight requires accelerate.") from e

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            no_init_weights = init_empty_weights

        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        auto_model_cls = cls.get_hf_auto_model_cls()
        model_dtype = cls.get_hf_model_dtype()
        torch_dtype = kwargs.pop("torch_dtype", kwargs.pop("dtype", model_dtype))
        model_kwargs = dict(kwargs)
        model_kwargs.pop("device_map", None)
        model_kwargs.pop("device", None)
        model_kwargs.setdefault("trust_remote_code", True)
        with no_init_weights(), init_empty_weights():
            native_model = auto_model_cls.from_config(
                config,
                **model_kwargs,
            )
        cls._load_gemma4_gguf_weights(
            gguf_path=gguf_path,
            native_hf_model=native_model,
            torch_dtype=torch_dtype,
            set_module_tensor_to_device=set_module_tensor_to_device,
        )
        if native_model.can_generate():
            try:
                native_model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
            except OSError:
                logger = get_xhquant_logger()
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
        return native_model.eval()

    @classmethod
    def _load_gemma4_gguf_weights(
        cls,
        *,
        gguf_path: str,
        native_hf_model: nn.Module,
        torch_dtype: torch.dtype,
        set_module_tensor_to_device: Callable,
    ) -> None:
        import numpy as np
        from gguf import GGUFReader
        from gguf import quants as gguf_quants

        logger = get_xhquant_logger()
        main_file, mmproj_file = cls._resolve_gguf_artifact_files(gguf_path)
        readers = [GGUFReader(main_file)]
        if mmproj_file is not None:
            readers.append(GGUFReader(mmproj_file))

        architecture = cls._gguf_field_string(readers[0], "general.architecture")
        if architecture != "gemma4":
            raise NotImplementedError(
                f"GGUF quant_weight loading currently supports Gemma4 GGUF only, got architecture={architecture!r}."
            )

        state_shapes = {name: tuple(tensor.shape) for name, tensor in native_hf_model.state_dict().items()}
        loaded: set[str] = set()
        skipped: list[str] = []
        attached_quant_weights = 0

        def _reshape_gguf_array(array: np.ndarray, tensor_name: str, target_shape: tuple[int, ...]) -> np.ndarray:
            if tensor_name == "v.patch_embd.weight" and array.ndim == 4 and len(target_shape) == 2:
                array = array.transpose(0, 2, 3, 1)
            if array.shape != target_shape:
                if array.size != int(np.prod(target_shape)):
                    raise RuntimeError(
                        f"GGUF tensor {tensor_name} shape {array.shape} cannot be reshaped to HF shape {target_shape}."
                    )
                array = array.reshape(target_shape)
            return array

        def _q4_0_quant_weight(tensor: Any, target_shape: tuple[int, ...]) -> torch.Tensor:
            blocks = np.asarray(tensor.data).view(np.uint8).reshape(-1, 18)
            _, qs = np.hsplit(blocks, [2])
            qcodes = qs.reshape((blocks.shape[0], -1, 1, 16)) >> np.array(
                [0, 4], dtype=np.uint8
            ).reshape((1, 1, 2, 1))
            qcodes = (qcodes & np.uint8(0x0F)).reshape((blocks.shape[0], -1)).astype(np.int8) - np.int8(8)
            qcodes = qcodes.reshape(gguf_quants.quant_shape_from_byte_shape(tensor.data.shape, tensor.tensor_type))
            qcodes = _reshape_gguf_array(qcodes, tensor.name, target_shape)
            return torch.from_numpy(np.asarray(qcodes)).contiguous()

        def _attach_quant_weight(native_model: nn.Module, hf_name: str, quant_weight: torch.Tensor | None) -> bool:
            if quant_weight is None:
                return False
            module_name, _, attr_name = hf_name.rpartition(".")
            if attr_name != "weight":
                return False
            module = native_model.get_submodule(module_name) if module_name else native_model
            if not isinstance(module, nn.Linear):
                return False
            if hasattr(module, "quant_weight"):
                setattr(module, "quant_weight", quant_weight)
            else:
                module.register_buffer("quant_weight", quant_weight, persistent=False)
            return True

        def _gguf_tensor_to_torch(
            tensor: Any,
            target_shape: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            tensor_type = int(tensor.tensor_type)
            if tensor_type == 0:
                array = np.asarray(tensor.data)
                quant_weight = None
            else:
                array = gguf_quants.dequantize(tensor.data, tensor.tensor_type)
                quant_weight = _q4_0_quant_weight(tensor, target_shape) if tensor_type == 2 else None
            array = _reshape_gguf_array(array, tensor.name, target_shape)
            tensor_value = torch.from_numpy(np.asarray(array))
            if tensor_value.is_floating_point():
                tensor_value = tensor_value.to(dtype=torch_dtype)
            return tensor_value.contiguous(), quant_weight

        total_tensors = sum(len(reader.tensors) for reader in readers)
        pbar = tqdm(total=total_tensors, desc="Loading Gemma4 GGUF tensors")
        for reader in readers:
            for gguf_tensor in reader.tensors:
                hf_name = cls._map_gemma4_gguf_tensor_name(gguf_tensor.name)
                pbar.update(1)
                if hf_name is None:
                    skipped.append(gguf_tensor.name)
                    continue
                target_shape = state_shapes.get(hf_name)
                if target_shape is None:
                    raise KeyError(f"GGUF tensor {gguf_tensor.name!r} maps to unknown HF tensor {hf_name!r}.")
                value, quant_weight = _gguf_tensor_to_torch(gguf_tensor, target_shape)
                set_module_tensor_to_device(native_hf_model, hf_name, "cpu", value=value)
                if _attach_quant_weight(native_hf_model, hf_name, quant_weight):
                    attached_quant_weights += 1
                loaded.add(hf_name)
                del value, quant_weight
        pbar.close()

        if "lm_head.weight" in state_shapes and "lm_head.weight" not in loaded:
            embed = native_hf_model.model.language_model.embed_tokens.weight
            set_module_tensor_to_device(native_hf_model, "lm_head.weight", "cpu", value=embed.detach())
            loaded.add("lm_head.weight")

        language_model = getattr(getattr(native_hf_model, "model", None), "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                source_idx = getattr(attn, "kv_shared_layer_index", None)
                if not getattr(attn, "is_kv_shared_layer", False) or source_idx is None:
                    continue
                for suffix in ("k_norm.weight", "k_proj.weight", "v_norm.weight", "v_proj.weight"):
                    target_name = f"model.language_model.layers.{layer_idx}.self_attn.{suffix}"
                    source_name = f"model.language_model.layers.{int(source_idx)}.self_attn.{suffix}"
                    if target_name not in state_shapes or target_name in loaded:
                        continue
                    source_tensor = native_hf_model.state_dict().get(source_name)
                    if source_tensor is None or getattr(source_tensor, "device", None) is None:
                        continue
                    if source_tensor.device.type == "meta":
                        continue
                    set_module_tensor_to_device(
                        native_hf_model,
                        target_name,
                        "cpu",
                        value=source_tensor.detach().clone(),
                    )
                    loaded.add(target_name)

        missing = sorted(name for name in state_shapes if name not in loaded and not name.endswith("inv_freq"))
        meta_missing = []
        for name in missing:
            tensor = native_hf_model.state_dict()[name]
            if getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
                meta_missing.append(name)
        if meta_missing:
            preview = ", ".join(meta_missing[:20])
            raise RuntimeError(
                f"Gemma4 GGUF load left {len(meta_missing)} HF tensors on meta device. "
                f"First missing tensors: {preview}"
            )
        logger.info(
            "Loaded Gemma4 GGUF weights from %s%s; loaded=%d, skipped=%d, attached_quant_weights=%d",
            main_file,
            f" and {mmproj_file}" if mmproj_file else "",
            len(loaded),
            len(skipped),
            attached_quant_weights,
        )

    @staticmethod
    def _gguf_field_string(reader: Any, name: str) -> str | None:
        field = reader.fields.get(name)
        if field is None or not field.parts:
            return None
        value = field.parts[-1]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "tolist"):
            value = value.tolist()
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, list) and all(isinstance(item, int) for item in value):
                return bytes(value).decode("utf-8")
        return str(value)

    @staticmethod
    def _map_gemma4_gguf_tensor_name(name: str) -> str | None:
        if name == "token_embd.weight":
            return "model.language_model.embed_tokens.weight"
        if name == "output_norm.weight":
            return "model.language_model.norm.weight"
        if name == "output.weight":
            return "lm_head.weight"
        if name == "per_layer_model_proj.weight":
            return "model.language_model.per_layer_model_projection.weight"
        if name == "per_layer_proj_norm.weight":
            return "model.language_model.per_layer_projection_norm.weight"
        if name == "per_layer_token_embd.weight":
            return "model.language_model.embed_tokens_per_layer.weight"
        if name == "rope_freqs.weight":
            return None

        block_match = re.match(r"blk\.(\d+)\.(.+)", name)
        if block_match:
            layer_idx, suffix = block_match.groups()
            suffix_map = {
                "layer_output_scale.weight": "layer_scalar",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "ffn_gate.weight": "mlp.gate_proj.weight",
                "ffn_up.weight": "mlp.up_proj.weight",
                "ffn_down.weight": "mlp.down_proj.weight",
                "inp_gate.weight": "per_layer_input_gate.weight",
                "proj.weight": "per_layer_projection.weight",
                "post_norm.weight": "post_per_layer_input_norm.weight",
                "attn_norm.weight": "input_layernorm.weight",
                "post_attention_norm.weight": "post_attention_layernorm.weight",
                "ffn_norm.weight": "pre_feedforward_layernorm.weight",
                "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
                "ffn_gate_inp.scale": "router.scale",
                "ffn_down_exps.scale": "router.per_expert_scale",
                "ffn_gate_inp.weight": "router.proj.weight",
                "ffn_gate_up_exps.weight": "experts.gate_up_proj",
                "ffn_down_exps.weight": "experts.down_proj",
                "post_ffw_norm_1.weight": "post_feedforward_layernorm_1.weight",
                "post_ffw_norm_2.weight": "post_feedforward_layernorm_2.weight",
                "pre_ffw_norm_2.weight": "pre_feedforward_layernorm_2.weight",
            }
            mapped_suffix = suffix_map.get(suffix)
            if mapped_suffix is None:
                raise KeyError(f"Unsupported Gemma4 GGUF tensor name: {name}")
            return f"model.language_model.layers.{layer_idx}.{mapped_suffix}"

        if name == "mm.input_projection.weight":
            return "model.embed_vision.embedding_projection.weight"
        if name == "mm.a.input_projection.weight":
            return "model.embed_audio.embedding_projection.weight"
        audio_top_map = {
            "a.pre_encode.out.bias": "model.audio_tower.output_proj.bias",
            "a.pre_encode.out.weight": "model.audio_tower.output_proj.weight",
            "a.input_projection.weight": "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
            "a.conv1d.0.weight": "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
            "a.conv1d.0.norm.weight": "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
            "a.conv1d.1.weight": "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
            "a.conv1d.1.norm.weight": "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
        }
        if name in audio_top_map:
            return audio_top_map[name]

        audio_match = re.match(r"a\.blk\.(\d+)\.(.+)", name)
        if audio_match:
            layer_idx, suffix = audio_match.groups()
            audio_prefix_map = {
                "ffn_up": "feed_forward1.ffw_layer_1",
                "ffn_down": "feed_forward1.ffw_layer_2",
                "ffn_up_1": "feed_forward2.ffw_layer_1",
                "ffn_down_1": "feed_forward2.ffw_layer_2",
                "attn_q": "self_attn.q_proj",
                "attn_k": "self_attn.k_proj",
                "attn_v": "self_attn.v_proj",
                "attn_out": "self_attn.post",
                "conv_pw1": "lconv1d.linear_start",
                "conv_pw2": "lconv1d.linear_end",
            }
            for gguf_prefix, hf_prefix in audio_prefix_map.items():
                if suffix == f"{gguf_prefix}.weight":
                    return f"model.audio_tower.layers.{layer_idx}.{hf_prefix}.linear.weight"
                for stat_name in ("input_min", "input_max", "output_min", "output_max"):
                    if suffix == f"{gguf_prefix}.{stat_name}":
                        return f"model.audio_tower.layers.{layer_idx}.{hf_prefix}.{stat_name}"
            suffix_map = {
                "ffn_norm.weight": "feed_forward1.pre_layer_norm.weight",
                "ffn_post_norm.weight": "feed_forward1.post_layer_norm.weight",
                "ffn_norm_1.weight": "feed_forward2.pre_layer_norm.weight",
                "ffn_post_norm_1.weight": "feed_forward2.post_layer_norm.weight",
                "attn_pre_norm.weight": "norm_pre_attn.weight",
                "attn_post_norm.weight": "norm_post_attn.weight",
                "attn_k_rel.weight": "self_attn.relative_k_proj.weight",
                "per_dim_scale.weight": "self_attn.per_dim_scale",
                "conv_dw.weight": "lconv1d.depthwise_conv1d.weight",
                "norm_conv.weight": "lconv1d.pre_layer_norm.weight",
                "conv_norm.weight": "lconv1d.conv_norm.weight",
                "ln2.weight": "norm_out.weight",
            }
            mapped_suffix = suffix_map.get(suffix)
            if mapped_suffix is None:
                raise KeyError(f"Unsupported Gemma4 audio GGUF tensor name: {name}")
            return f"model.audio_tower.layers.{layer_idx}.{mapped_suffix}"
        vision_top_map = {
            "v.patch_embd.weight": "model.vision_tower.patch_embedder.input_proj.weight",
            "v.position_embd.weight": "model.vision_tower.patch_embedder.position_embedding_table",
            "v.std_bias": "model.vision_tower.std_bias",
            "v.std_scale": "model.vision_tower.std_scale",
            "v.post_ln.weight": "model.vision_tower.post_layernorm.weight",
        }
        if name in vision_top_map:
            return vision_top_map[name]

        vision_match = re.match(r"v\.blk\.(\d+)\.(.+)", name)
        if vision_match:
            layer_idx, suffix = vision_match.groups()
            suffix_map = {
                "attn_q.weight": "self_attn.q_proj.linear.weight",
                "attn_k.weight": "self_attn.k_proj.linear.weight",
                "attn_v.weight": "self_attn.v_proj.linear.weight",
                "attn_out.weight": "self_attn.o_proj.linear.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "ffn_gate.weight": "mlp.gate_proj.linear.weight",
                "ffn_up.weight": "mlp.up_proj.linear.weight",
                "ffn_down.weight": "mlp.down_proj.linear.weight",
                "ln1.weight": "input_layernorm.weight",
                "attn_post_norm.weight": "post_attention_layernorm.weight",
                "ln2.weight": "pre_feedforward_layernorm.weight",
                "ffn_post_norm.weight": "post_feedforward_layernorm.weight",
            }
            mapped_suffix = suffix_map.get(suffix)
            if mapped_suffix is None:
                vision_prefix_map = {
                    "attn_q": "self_attn.q_proj",
                    "attn_k": "self_attn.k_proj",
                    "attn_v": "self_attn.v_proj",
                    "attn_out": "self_attn.o_proj",
                    "ffn_gate": "mlp.gate_proj",
                    "ffn_up": "mlp.up_proj",
                    "ffn_down": "mlp.down_proj",
                }
                for gguf_prefix, hf_prefix in vision_prefix_map.items():
                    for stat_name in ("input_min", "input_max", "output_min", "output_max"):
                        if suffix == f"{gguf_prefix}.{stat_name}":
                            mapped_suffix = f"{hf_prefix}.{stat_name}"
                            break
                    if mapped_suffix is not None:
                        break
            if mapped_suffix is None:
                raise KeyError(f"Unsupported Gemma4 vision GGUF tensor name: {name}")
            return f"model.vision_tower.encoder.layers.{layer_idx}.{mapped_suffix}"

        raise KeyError(f"Unsupported Gemma4 GGUF tensor name: {name}")

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        # Keep Gemma4 Series on the Merak quantized-HF loading path.  GPTQModel
        # Gemma4 MoE checkpoints are not safe to load through Transformers/
        # optimum directly: GPTQModel may split/fill MoE expert weights and the
        # base Merak path already owns the required torch-backend load + dequant
        # conversion.  This mirrors the legacy gemma4_moe loader instead of
        # inventing a second quantized load path here.
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("attn_implementation", "eager")
        kwargs.setdefault("device_map", "cpu")
        kwargs.setdefault("dtype", torch.bfloat16)
        if cls._is_gguf_quant_weight_path(quant_weight):
            return cls._load_hf_model_from_gguf(hf_model_dir, quant_weight, **kwargs)
        return super().get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)

    @classmethod
    def _postprocess_gptqmodel_structure(cls, native_hf_model: nn.Module, **kwargs) -> nn.Module:
        native_hf_model.quantization_method = None  # type: ignore[attr-defined]
        native_hf_model._is_hf_initialized = False  # type: ignore[attr-defined]
        if hasattr(native_hf_model.config, "quantization_config"):
            native_hf_model.config.quantization_config = None
        if getattr(native_hf_model.config, "tie_word_embeddings", False):
            native_hf_model.config.torchscript = True
            native_hf_model.tie_weights()
            native_hf_model.config.tie_word_embeddings = False
            native_hf_model.config.torchscript = False
        return native_hf_model.eval()

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        # GPTQModel registers a split-expert Gemma4NaiveExperts implementation
        # in-process for MoE quant checkpoints.  It intentionally exposes
        # per-expert Linear modules instead of HF's packed gate_up_proj/down_proj
        # Parameters.  During HMONNX compatible generate we only need an empty
        # facade model; however Transformers still walks _init_weights even
        # under no_init/init_empty_weights.  Guard just this empty-model
        # construction so the GPTQModel split-expert class is not initialized
        # through HF's packed-expert branch.
        from transformers.models.gemma4 import modeling_gemma4

        original_init_weights = modeling_gemma4.Gemma4PreTrainedModel._init_weights

        def _guard_gptqmodel_split_expert_init(self, module):
            if (
                module.__class__.__name__ == "Gemma4NaiveExperts"
                and hasattr(module, "experts")
                and not hasattr(module, "gate_up_proj")
                and not hasattr(module, "down_proj")
            ):
                return
            return original_init_weights(self, module)

        modeling_gemma4.Gemma4PreTrainedModel._init_weights = _guard_gptqmodel_split_expert_init
        try:
            return super().get_empty_hf_model(hf_model_dir, **kwargs)
        finally:
            modeling_gemma4.Gemma4PreTrainedModel._init_weights = original_init_weights

    def _wraped_pre(self, hf_model: Gemma4ForConditionalGeneration):
        model = getattr(hf_model, "model", None)
        if model is None:
            return hf_model
        for name in ["vision_tower", "embed_vision", "audio_tower", "embed_audio"]:
            if hasattr(model, name):
                delattr(model, name)
        return hf_model

    def init_wrap_model(self, hf_model: Gemma4ForConditionalGeneration) -> object:
        from ._llm_model_impl import register_wrap_modules

        register_wrap_modules()
        hf_model = _make_text_export_bridge_if_needed(
            hf_model,
            self.config.num_logits_to_keep,
            enable_mtp_outputs=self.config.enable_mtp_outputs,
        )
        return super().init_wrap_model(hf_model)

    def _wraped_post(self, hf_model: Gemma4ForConditionalGeneration):
        self.config.image_token_id = getattr(hf_model.config, "image_token_id", None)
        self.config.video_token_id = getattr(hf_model.config, "video_token_id", None)
        self.config.audio_token_id = getattr(hf_model.config, "audio_token_id", None)
        self.config.boi_token_id = getattr(hf_model.config, "boi_token_id", None)
        self.config.eoi_token_id = getattr(hf_model.config, "eoi_token_id", None)

        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)
        # Deep-copy embedding on CPU to avoid GPU OOM for large vocab models.
        orig_embed = llm_model.get_input_embeddings()
        orig_device = orig_embed.weight.device
        embed_copy = copy.deepcopy(orig_embed.cpu())
        orig_embed.to(orig_device)
        if hasattr(embed_copy, "embed_scale"):
            embed_copy.weight.data = (embed_copy.weight.float() * embed_copy.embed_scale).to(embed_copy.weight.dtype)
        self.embed_tokens = nn.Embedding(
            embed_copy.num_embeddings,
            embed_copy.embedding_dim,
            _weight=embed_copy.weight,
        ).to(orig_device)
        self.pad_token_id = int(getattr(llm_model.config, "pad_token_id", 0) or 0)
        self.sliding_window = int(getattr(llm_model.config, "sliding_window", 1024))
        self.layer_types = list(getattr(llm_model.config, "layer_types", []))
        if getattr(llm_model.config, "hidden_size_per_layer_input", 0):
            self.per_layer_input_embedding = Gemma4PerLayerInputEmbedding.from_language_model(llm_model)

        layer_kv_shapes: list[list[int]] = []
        layer_cache_types: list[str | None] = []
        layer_cache_indices: list[int] = []
        for layer_idx, layer in enumerate(llm_model.layers):
            attn = layer.self_attn
            if getattr(attn, "is_kv_shared_layer", False):
                continue
            num_key_value_heads = attn.k_proj.out_features // attn.head_dim
            layer_type = self.layer_types[layer_idx] if layer_idx < len(self.layer_types) else None
            cache_seq_len = _gemma4_cache_seq_len_for_layer(
                layer_type=layer_type,
                context_max_length=self.config.context_max_length,
                sliding_window=self.sliding_window,
                input_seq_len=self._prefill_cache_input_length(),
                sliding_kv_cache_input_mode=getattr(
                    self.config, "sliding_kv_cache_input_mode", "slice_window"
                ),
            )
            layer_kv_shapes.append([1, num_key_value_heads, cache_seq_len, attn.head_dim])
            layer_cache_types.append(layer_type)
            layer_cache_indices.append(layer_idx)
        self._kvcache_mixin.set_layer_kv_shapes(layer_kv_shapes)
        self.layer_cache_types = layer_cache_types
        self.layer_cache_indices = layer_cache_indices

    def _get_data_preprocessor(self):
        common_kwargs = dict(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            context_length=self.config.context_max_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=self.pad_token_id,
            image_token_id=self.config.image_token_id or -1,
            audio_token_id=self.config.audio_token_id or -1,
            video_token_id=self.config.video_token_id or -1,
            bidirectional_vision_attention=getattr(self.config, "bidirectional_vision_attention", False),
            # Full-attention layers receive None and use xhquant.nn.MaskedSoftmax's
            # causal path.  Only sliding attention consumes the explicit mask.
            emit_full_attention_mask=False,
            emit_accepted_count_input=self._uses_target_verify_decode_accepted_count(),
        )
        if getattr(self.config, "enable_moe_block", False):
            return Gemma4MoeDataPreprocess(
                **common_kwargs,
                sliding_window_cfg={
                    "sliding_window": self.config.sliding_window,
                    "local_attention_window_size": self.config.local_attention_window_size,
                    "global_attention_window_size": self.config.global_attention_window_size,
                    "has_local_attention": self.config.has_local_attention,
                    "has_global_attention": self.config.has_global_attention,
                },
            )
        return Gemma4DataPreprocess(
            **common_kwargs,
            per_layer_input_embedding=self.per_layer_input_embedding,
            sliding_window=self.sliding_window,
        )

    def get_data_preprocessor(self):
        # Recreate the lightweight processor whenever callers ask so it captures
        # current prefill/decode length, dual-prefill shape, and KV cache objects.
        self._data_processor = None
        return super().get_data_preprocessor()

    def _set_device(self, device: torch.device | str | None):
        super()._set_device(device)
        if self.visual is not None:
            self.visual.to(device)
        if self.video_visual is not None:
            self.video_visual.to(device)
        if self.audio is not None:
            self.audio.to(device)
        if self.per_layer_input_embedding is not None:
            self.per_layer_input_embedding.to(device)

    def _to_fronted(self, wrap_model):
        self.set_prefill()
        decode_wrap_model = _copy_model_shared_params(wrap_model)

        if isinstance(wrap_model, _Gemma4TextExportBridgePLE):
            prefill_wrap_model = wrap_model
        else:
            prefill_wrap_model = _Gemma4DecodeNoFullMaskBridge(
                wrap_model,
                num_logits_to_keep=self.config.num_logits_to_keep,
                language_model_keeps_last_logit=(int(self.config.num_logits_to_keep or 0) == 1),
                language_model_returns_tensor=True,
                enable_mtp_outputs=self.config.enable_mtp_outputs,
            )

        if self._is_mtp_export() and isinstance(decode_wrap_model, _Gemma4TextExportBridgePLE):
            decode_wrap_model.__class__ = _Gemma4TextExportBridgePLEMTPDecode
        elif not isinstance(decode_wrap_model, _Gemma4TextExportBridgePLE):
            decode_num_logits_to_keep = 0 if self._is_mtp_export() else self.config.num_logits_to_keep
            decode_bridge_cls = (
                _Gemma4DecodeNoFullMaskMTPBridge if self._is_mtp_export() else _Gemma4DecodeNoFullMaskBridge
            )
            decode_wrap_model = decode_bridge_cls(
                decode_wrap_model,
                num_logits_to_keep=decode_num_logits_to_keep,
                language_model_keeps_last_logit=(int(decode_num_logits_to_keep or 0) == 1),
                language_model_returns_tensor=True,
                enable_mtp_outputs=self.config.enable_mtp_outputs,
            )
        self._wrap_model = prefill_wrap_model
        prefill_frontend_model = super()._to_fronted(prefill_wrap_model)

        self._wrap_model = decode_wrap_model
        self.set_decode()
        original_input_sequence_length = _cfg_get_value(self.wrap_cfg, "input_sequence_length")
        original_num_logits_to_keep = _cfg_get_value(self.wrap_cfg, "num_logits_to_keep", None)
        if self._is_mtp_export():
            _cfg_set_value(self.wrap_cfg, "input_sequence_length", self._mtp_verify_length())
            for key, value in self._mtp_decode_wrap_cfg_overrides().items():
                _cfg_set_value(self.wrap_cfg, key, value)
            self._apply_wrap_cfg_to_modules(decode_wrap_model)
        try:
            decode_frontend_model = super()._to_fronted(decode_wrap_model)
        finally:
            if original_input_sequence_length is not None:
                _cfg_set_value(self.wrap_cfg, "input_sequence_length", original_input_sequence_length)
            if original_num_logits_to_keep is not None:
                _cfg_set_value(self.wrap_cfg, "num_logits_to_keep", original_num_logits_to_keep)
        self._frontend_model = prefill_frontend_model
        self._wrap_model = prefill_frontend_model
        self.set_prefill()
        models = {
            "prefill": prefill_frontend_model,
            "decode": decode_frontend_model,
        }
        fronted_model = ModelSwitcher(models)
        fronted_model.set_activate_model("prefill")
        return fronted_model

    def _to_quanted(self, frontend_model, state):
        decode_fronted_model = frontend_model.decode
        prefill_fronted_model = frontend_model.prefill
        self.set_prefill()
        # Let the xhquant pipeline own device placement.  Pre-moving the full
        # 31B prefill graph to CUDA leaves only a few MB free on an 80GB card
        # and can OOM inside graph_module_guard while adding input/output
        # quant points.  Qwen3.5's Merak path uses the same base pipeline
        # without an eager .cuda() here.
        prefill_quanted_model = super()._to_quanted(prefill_fronted_model, state, infer_shape=False)

        prefill_quanted_model.cpu()
        prefill_fronted_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        decode_fronted_model.cpu()
        torch.cuda.empty_cache()

        self.set_decode()
        original_input_sequence_length = _cfg_get_value(self.wrap_cfg, "input_sequence_length")
        original_num_logits_to_keep = _cfg_get_value(self.wrap_cfg, "num_logits_to_keep", None)
        if self._is_mtp_export():
            _cfg_set_value(self.wrap_cfg, "input_sequence_length", self._mtp_verify_length())
            for key, value in self._mtp_decode_wrap_cfg_overrides().items():
                _cfg_set_value(self.wrap_cfg, key, value)
            self._apply_wrap_cfg_to_modules(decode_fronted_model)
        try:
            decode_quanted_model = super()._to_quanted(decode_fronted_model, state, infer_shape=False)
        finally:
            if original_input_sequence_length is not None:
                _cfg_set_value(self.wrap_cfg, "input_sequence_length", original_input_sequence_length)
            if original_num_logits_to_keep is not None:
                _cfg_set_value(self.wrap_cfg, "num_logits_to_keep", original_num_logits_to_keep)
        self.set_prefill()
        models = {
            "prefill": prefill_quanted_model,
            "decode": decode_quanted_model,
        }
        quanted_model = ModelSwitcher(models)
        quanted_model.set_activate_model("prefill")
        return quanted_model

    def get_export_cfg(self) -> dict[str, list[str]]:
        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
        ]
        input_names.append("sliding_attention_mask")
        if self.per_layer_input_embedding is not None or getattr(self.config, "hidden_size_per_layer_input", 0):
            # E4B PLE belongs with the text payload, immediately before KV caches.
            input_names.append("per_layer_inputs")
        export_cfg = {
            "input_names": input_names,
            "output_names": ["logits"],
        }
        if self._is_mtp_export():
            export_cfg["output_names"].append("target_hidden_state")
        if self._uses_target_verify_decode_accepted_count():
            export_cfg["input_names"].append("accepted_count")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info):
        meta_info.variant = getattr(self.config, "variant", None)
        meta_info.capabilities = dict(getattr(self.config, "capabilities", {}) or {})
        meta_info.layer_types = self.layer_types
        meta_info.layer_kv_shapes = self._kvcache_mixin.layer_kv_shapes
        meta_info.layer_cache_types = list(getattr(self, "layer_cache_types", []) or [])
        meta_info.layer_cache_indices = list(getattr(self, "layer_cache_indices", []) or [])
        meta_info.sliding_window = self.sliding_window
        meta_info.prefill_chunk_length = int(self.config.prefill_chunk_length)
        meta_info.prefill_graphs = {
            "prefill": {
                "input_sequence_length": int(self.config.prefill_chunk_length),
                "hmonnx": "",
            },
        }
        meta_info.sliding_cache_storage_length = self._mtp_shared_sliding_cache_length()
        meta_info.sliding_kv_cache_input_mode = getattr(
            self.config, "sliding_kv_cache_input_mode", "slice_window"
        )
        if self._is_mtp_export():
            block_size = self._mtp_num_draft_tokens()
            verify_length = self._mtp_verify_length()
            draft_head_bits = _mtp_draft_head_weight_bits(self.config)
            spec_decode = {
                "mode": "mtp",
                "block_size": block_size,
                "verify_length": verify_length,
                "hidden_output_name": "target_hidden_state",
                "draft_head_weight_bits": draft_head_bits,
                "shared_sliding_cache_length": self._mtp_shared_sliding_cache_length(),
                "shared_full_cache_length": int(self.config.context_max_length),
                "target_decode_sliding_output_length": self._mtp_target_decode_sliding_output_length(),
                "shared_sliding_cache_length_basis": "slice_window + prefill_chunk_length",
                "target_decode_sliding_output_length_basis": "aligned(sliding_window + verify_length - 1, 16)",
            }
            meta_info.spec_decode_mode = "mtp"
            meta_info.spec_decode_block_size = block_size
            meta_info.spec_decode_verify_length = verify_length
            meta_info.spec_decode_hidden_output_name = "target_hidden_state"
            meta_info.spec_decode_draft_head_weight_bits = draft_head_bits
            meta_info.spec_decode = spec_decode
        if self.per_layer_input_embedding is not None:
            artifact_path = Path(output_dir) / "per_layer_input_embedding.pt"
            self.per_layer_input_embedding.save_artifact(artifact_path)
            meta_info.per_layer_input_embedding = artifact_path.relative_to(output_dir).as_posix()
        return meta_info

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        output_dir = Path(output_dir) / f"hmquant_{model_name}_{str_datetime}"
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = f"hmquant_{model_name}_{str_datetime}"
        export_data.str_datetime = str_datetime
        return export_data

    @log_function_call()
    def _export_hmonnx(self, exported_info: ExportData):
        exported_info = super()._export_hmonnx(exported_info)
        meta_info = exported_info.meta
        if hasattr(meta_info, "prefill_graphs") and isinstance(meta_info.prefill_graphs, dict):
            meta_info.prefill_graphs["prefill"]["hmonnx"] = meta_info.prefill_hmonnx
        return exported_info

    @log_function_call()
    def export_hmonnx(self, output_dir: str):
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        if self.visual is not None:
            if self.visual.quanted_model is None:
                self.visual.to_quanted_aligned()
            self.visual.quanted_model.fixed()
        if self.video_visual is not None:
            if self.video_visual.quanted_model is None:
                self.video_visual.to_quanted_aligned()
            self.video_visual.quanted_model.fixed()
        if self.audio is not None:
            if self.audio.quanted_model is None:
                self.audio.to_quanted_aligned()
            self.audio.quanted_model.fixed()
        if self._is_mtp_export():
            self._decode_input_sequence_length = self._mtp_verify_length()
            self._decode_wrap_cfg_overrides = self._mtp_decode_wrap_cfg_overrides()
        exported_info = self.get_export_info(output_dir)
        meta_info = cast(Gemma4SeriesModelMeta, exported_info.meta)
        if self.visual is not None:
            visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
            visual_meta = self.visual.export_hmonnx(visual_output_dir)
            visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
            if getattr(visual_meta, "onnx", None):
                visual_meta.onnx = str(Path(visual_meta.onnx).relative_to(exported_info.exported_dir).as_posix())
            meta_info.visual_config = visual_meta
        if self.video_visual is not None:
            video_visual_output_dir = str(Path(exported_info.exported_dir) / "video_visual")
            video_visual_meta = self.video_visual.export_hmonnx(video_visual_output_dir)
            video_visual_meta.hmonnx = str(
                Path(video_visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix()
            )
            if getattr(video_visual_meta, "onnx", None):
                video_visual_meta.onnx = str(
                    Path(video_visual_meta.onnx).relative_to(exported_info.exported_dir).as_posix()
                )
            meta_info.video_visual_config = video_visual_meta
        if self.audio is not None:
            audio_output_dir = str(Path(exported_info.exported_dir) / "audio")
            audio_meta = self.audio.export_hmonnx(audio_output_dir)
            audio_meta.hmonnx = str(Path(audio_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
            if getattr(audio_meta, "onnx", None):
                audio_meta.onnx = str(Path(audio_meta.onnx).relative_to(exported_info.exported_dir).as_posix())
            meta_info.audio_config = audio_meta
        self._export_hmonnx(exported_info)
        stripped_default_cache_attrs = 0
        for hmonnx_path in (
            meta_info.prefill_hmonnx,
            meta_info.decode_hmonnx,
        ):
            if hmonnx_path:
                onnx_path = Path(hmonnx_path)
                if not onnx_path.is_absolute():
                    onnx_path = Path(exported_info.exported_dir) / onnx_path
                stripped_default_cache_attrs += _strip_default_llmcache_only_handle_old_cache_attrs(onnx_path)
        if stripped_default_cache_attrs:
            logger.info(
                "Removed %d default-false only_handle_old_cache attributes from Gemma4 Series ONNX graphs.",
                stripped_default_cache_attrs,
            )
        json.dump(
            meta_info.to_dict(),
            open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"),
            indent=4,
        )
        logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info


# Compatibility export name for callers that have not renamed imports yet.
XHGemma4Model = XHGemma4SeriesModel


__all__ = ["XHGemma4Model", "XHGemma4SeriesModel", "build_gemma4_hf_compatible_model"]
