import torch

from xhmodel_merak.xh_llm.kv_cache_mixin import KVCacheWithLinearMixin
from xhquant.core import CacheTensor
from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import LLMModelMeta
from .data_preprocess import Qwen3_5_DataPreprocess
from .hybrid_cache_runtime import (
    commit_hybrid_cache_outputs,
    get_spec_decode_verify_steps,
    model_config_prefill_recurrent_state_uses_cache,
    normalize_hybrid_hmonnx_args,
)
from .qwen3_5_processor import XHQwen3_5Processor
from .split_conv_cache_utils import (
    _flatten_split_conv_cache_outputs,
    _regroup_flat_split_conv_cache,  # noqa: F401 - compatibility re-export
)
from .visual_token_gears import (
    VISUAL_ATTENTION_MASK_FORMAT,
    VISUAL_ATTENTION_MASK_OPERATOR,
    VISUAL_ATTENTION_MASK_SHAPE,
    VISUAL_INPUT_PATCHES,
    VISUAL_ROTARY_POSITION_FORMAT,
    build_visual_token_gear_inputs,
    pad_flattened_patches,
    patch_token_capacity,
    select_image_token_gear,
)


def _model_config_prefill_recurrent_state_uses_cache(model_config) -> bool:
    """Compatibility alias for callers that imported the former local helper."""

    return model_config_prefill_recurrent_state_uses_cache(model_config)


class VisualHMONNXModel(HMONNXModel):
    def forward(self, *args):
        out = super().forward(*args)
        return out


class VisualTokenGearHMONNXModel:
    """Route one image at a time across shared-weight static visual graphs."""

    INPUT_NAMES = (
        "pixel_values",
        "position_ids",
        "position_weights",
        "rotary_position_ids",
        "attention_mask",
    )

    def __init__(self, visual_meta, *, device, enable_golden: bool = False):
        if not HMONNXModel._is_inference_v2_env_enabled():
            raise RuntimeError(
                "multi-gear Qwen3.5 visual runtime requires ENABLE_HMINFERENCE_V2=1 "
                "so MultiHMONNXLoader can preserve shared GPU weights"
            )
        expected_contract = {
            "visual_input_mode": VISUAL_INPUT_PATCHES,
            "attention_mask_format": VISUAL_ATTENTION_MASK_FORMAT,
            "attention_mask_shape": VISUAL_ATTENTION_MASK_SHAPE,
            "attention_mask_operator": VISUAL_ATTENTION_MASK_OPERATOR,
            "rotary_position_format": VISUAL_ROTARY_POSITION_FORMAT,
        }
        for field, expected in expected_contract.items():
            actual = getattr(visual_meta, field, None)
            if actual != expected:
                raise ValueError(f"Qwen3.5 visual token gears require {field}={expected!r}, got {actual!r}")
        merge_size = int(visual_meta.spatial_merge_size)
        rope_cache_length = int(visual_meta.visual_rope_cache_length)
        for gear in visual_meta.gears:
            image_capacity = int(gear.image_token_capacity)
            expected_patch_capacity = patch_token_capacity(image_capacity, merge_size)
            actual_patch_capacity = int(gear.patch_token_capacity)
            if actual_patch_capacity != expected_patch_capacity:
                raise ValueError(
                    f"visual gear {image_capacity} declares patch capacity "
                    f"{actual_patch_capacity}, expected {expected_patch_capacity}"
                )
            minimum_rope_cache_length = image_capacity * merge_size
            if rope_cache_length < minimum_rope_cache_length:
                raise ValueError(
                    f"visual gear {image_capacity} needs RoPE cache length "
                    f"{minimum_rope_cache_length}, got {rope_cache_length}"
                )
        self.visual_meta = visual_meta
        self.gears = tuple(sorted(int(gear.image_token_capacity) for gear in visual_meta.gears))
        graph_files = {f"m{int(gear.image_token_capacity)}": str(gear.hmonnx) for gear in visual_meta.gears}
        self._loader = MultiHMONNXLoader(graph_files)
        self.models: dict[int, VisualHMONNXModel] = {}
        for gear in visual_meta.gears:
            capacity = int(gear.image_token_capacity)
            graph_name = f"m{capacity}"
            self.models[capacity] = VisualHMONNXModel(
                str(gear.hmonnx),
                onnx_graph=self._loader.graphs[graph_name],
                device_map=[device],
                enable_golden=enable_golden,
            )
        self._device = self.models[self.gears[-1]].device
        self._dtype = torch.float16
        self._enable_golden = enable_golden

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def enable_golden(self):
        return self._enable_golden

    @property
    def shared_weight_summary(self):
        return self._loader.initializer_pool.summary()

    @enable_golden.setter
    def enable_golden(self, enabled: bool):
        for model in self.models.values():
            model.enable_golden = enabled
        self._enable_golden = enabled

    def to(self, device):
        for model in self.models.values():
            model.to(device)
        # HMONNXInferenceV2 may intentionally keep an already materialized
        # session on its construction device.  Report the session's actual
        # device instead of claiming that a no-op move succeeded.
        self._device = self.models[self.gears[-1]].device
        return self

    def _set_dtype(self, dtype):
        for model in self.models.values():
            model._set_dtype(dtype)
        self._dtype = dtype
        return self

    @staticmethod
    def _session_input_info(model: VisualHMONNXModel, name: str):
        session = model.hmonnx_session
        get_input = getattr(session, "get_input", None)
        if not callable(get_input):
            raise RuntimeError(f"HMONNX session does not expose input metadata for {name!r}")
        return get_input(name)

    def _coerce_inputs(self, model: VisualHMONNXModel, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        inputs = []
        for name in self.INPUT_NAMES:
            info = self._session_input_info(model, name)
            inputs.append(values[name].to(device=model.device, dtype=info.dtype))
        return tuple(inputs)

    def _prepare_image(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> tuple[int, tuple, int]:
        grid_thw = grid_thw.reshape(1, 3)
        valid_patch_tokens = int(grid_thw.prod())
        merge_unit = int(self.visual_meta.spatial_merge_size) ** 2
        if valid_patch_tokens % merge_unit:
            raise ValueError(
                f"visual patch count {valid_patch_tokens} is not divisible by spatial merge unit {merge_unit}"
            )
        valid_image_tokens = valid_patch_tokens // merge_unit
        gear = select_image_token_gear(valid_image_tokens, self.gears)
        capacity = patch_token_capacity(gear, int(self.visual_meta.spatial_merge_size))
        padded_pixels, tensor_patch_tokens = pad_flattened_patches(pixel_values, capacity)
        if tensor_patch_tokens != valid_patch_tokens:
            raise ValueError(
                f"pixel_values has {tensor_patch_tokens} patches but image_grid_thw declares {valid_patch_tokens}"
            )
        model = self.models[gear]
        position_dtype = self._session_input_info(model, "position_weights").dtype
        geometry = build_visual_token_gear_inputs(
            grid_thw,
            patch_capacity=capacity,
            num_position_embeddings=int(self.visual_meta.num_position_embeddings),
            spatial_merge_size=int(self.visual_meta.spatial_merge_size),
            dtype=position_dtype,
            rotary_cache_length=int(self.visual_meta.visual_rope_cache_length),
        )
        values = {"pixel_values": padded_pixels}
        values.update({name: geometry[name] for name in self.INPUT_NAMES if name != "pixel_values"})
        return gear, self._coerce_inputs(model, values), valid_image_tokens

    @staticmethod
    def _slice_valid_output(output: torch.Tensor, valid_image_tokens: int) -> torch.Tensor:
        if output.ndim == 3:
            return output[:, :valid_image_tokens]
        if output.ndim == 2:
            return output[:valid_image_tokens]
        raise ValueError(f"unexpected visual output shape: {tuple(output.shape)}")

    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        gear, inputs, valid_image_tokens = self._prepare_image(pixel_values, grid_thw)
        output = self.models[gear].forward(*inputs)
        return self._slice_valid_output(output, valid_image_tokens)

    def encode_many(self, pixel_values, image_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        grids = [grid.reshape(1, 3) for grid in image_grid_thw]
        if isinstance(pixel_values, torch.Tensor):
            split_sizes = [int(grid.prod()) for grid in grids]
            pixels = list(torch.split(pixel_values, split_sizes, dim=0))
        else:
            pixels = list(pixel_values)
        if len(pixels) != len(grids):
            raise ValueError(f"got {len(pixels)} pixel tensors for {len(grids)} image grids")

        prepared = [self._prepare_image(image, grid) for image, grid in zip(pixels, grids, strict=True)]
        outputs: list[torch.Tensor | None] = [None] * len(prepared)
        # Grouping by gear improves graph/CUDA-graph locality while preserving
        # the caller-visible image order through indexed output placement.
        for gear in self.gears:
            for index, (selected_gear, inputs, valid_image_tokens) in enumerate(prepared):
                if selected_gear != gear:
                    continue
                output = self.models[gear].forward(*inputs)
                outputs[index] = self._slice_valid_output(output, valid_image_tokens)
        if any(output is None for output in outputs):
            missing = [index for index, output in enumerate(outputs) if output is None]
            raise RuntimeError(f"visual gear scheduler did not produce outputs for image indices {missing}")
        return [output for output in outputs if output is not None]

    def forward(self, *args):
        if len(args) == 2:
            return self.encode(args[0], args[1])
        if len(args) != len(self.INPUT_NAMES):
            raise ValueError(
                "visual token gear forward expects (pixel_values, grid_thw) or the five padded graph inputs"
            )
        patch_capacity_value = int(args[0].shape[1])
        matching = [
            gear
            for gear in self.gears
            if patch_token_capacity(gear, int(self.visual_meta.spatial_merge_size)) == patch_capacity_value
        ]
        if len(matching) != 1:
            raise ValueError(f"no visual gear has patch capacity {patch_capacity_value}")
        attention_mask = args[-1]
        expected_mask_shape = (1, 1, 1, patch_capacity_value)
        if tuple(attention_mask.shape) != expected_mask_shape:
            raise ValueError(
                "visual attention mask must be a compact additive key-padding bias with shape "
                f"{expected_mask_shape}, got {tuple(attention_mask.shape)}"
            )
        flattened_mask = attention_mask.reshape(-1)
        valid_patch_tokens = int(torch.count_nonzero(flattened_mask == 0).item())
        merge_unit = int(self.visual_meta.spatial_merge_size) ** 2
        if valid_patch_tokens <= 0 or valid_patch_tokens > patch_capacity_value or valid_patch_tokens % merge_unit:
            raise ValueError(
                f"visual attention mask declares invalid patch length {valid_patch_tokens} "
                f"for capacity {patch_capacity_value} and merge unit {merge_unit}"
            )
        if not bool(torch.all(flattened_mask[:valid_patch_tokens] == 0).item()) or (
            valid_patch_tokens < patch_capacity_value
            and not bool(torch.all(flattened_mask[valid_patch_tokens:] < 0).item())
        ):
            raise ValueError("visual attention mask must contain a zero prefix followed by negative padding bias")
        valid_image_tokens = valid_patch_tokens // merge_unit
        return self._slice_valid_output(self.models[matching[0]].forward(*args), valid_image_tokens)


class Qwen3_5HMONNXKVCacheMixin(KVCacheWithLinearMixin):  # noqa: N801
    """HMONNX cache mixin that mirrors the exported split q/k/v conv-cache signature."""

    def __init__(self, kv_cache_config) -> None:
        super().__init__(kv_cache_config)
        self.split_conv_cache: bool = False

    def prepare_other_cache(self):
        if not self.split_conv_cache:
            return super().prepare_other_cache()

        linear_cfg = self.kvcache_config.linear_kv_cache_config
        value_dim = linear_cfg.num_v_heads * linear_cfg.head_v_dim
        key_dim = (linear_cfg.conv_dim - value_dim) // 2
        if key_dim <= 0 or linear_cfg.conv_dim != key_dim * 2 + value_dim:
            raise RuntimeError(
                "Invalid linear kv cache config for split conv cache: "
                f"conv_dim={linear_cfg.conv_dim}, value_dim={value_dim}"
            )

        cache_dtype = linear_cfg.cache_torch_dtype
        batch_size = linear_cfg.batch_size
        kernel_size = linear_cfg.conv_kernel_size
        for _i in range(linear_cfg.num_layers):
            conv_cache_q = CacheTensor(
                torch.zeros(
                    batch_size,
                    key_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            conv_cache_k = CacheTensor(
                torch.zeros(
                    batch_size,
                    key_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            conv_cache_v = CacheTensor(
                torch.zeros(
                    batch_size,
                    value_dim,
                    kernel_size,
                    dtype=cache_dtype,
                    device=self._device,
                )
            )
            self.past_conv_caches.append((conv_cache_q, conv_cache_k, conv_cache_v))

            recurrent_cache_shape = [
                batch_size,
                linear_cfg.num_v_heads,
                linear_cfg.head_k_dim,
                linear_cfg.head_v_dim,
            ]
            self.past_recurrent_states.append(
                CacheTensor(torch.zeros(recurrent_cache_shape, dtype=cache_dtype, device=self._device))
            )

    def _set_device(self, device):
        self._device = device
        for cache_tensor in _flatten_split_conv_cache_outputs(self.past_conv_caches):
            cache_tensor.to(device)
        for cache_tensor in self.past_recurrent_states:
            cache_tensor.to(device)


class XHQwen3_5_HMONNXModel(VisonLLMHMONNXModel):  # noqa: N801
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        enable_golden = kwargs.get("enable_golden", False)
        if getattr(self.visual_meta, "gears", None):
            self.visual = VisualTokenGearHMONNXModel(
                self.visual_meta,
                device=self.prefill_model.device,
                enable_golden=enable_golden,
            )
        else:
            self.visual = VisualHMONNXModel(
                self.visual_meta.hmonnx, device_map=[self.prefill_model.device], enable_golden=enable_golden
            )
        self._kvcache_mixin = Qwen3_5HMONNXKVCacheMixin(self.kvcache_config)
        self._kvcache_mixin.split_conv_cache = bool(getattr(meta_info.model_config, "split_conv_cache", False))
        self._sync_page_attention_mode_to_kvcache()

    @property
    def past_conv_caches(self):
        return self._kvcache_mixin.past_conv_caches

    @property
    def past_recurrent_states(self):
        return self._kvcache_mixin.past_recurrent_states

    def _set_device(self, device):
        super()._set_device(device)
        self.visual.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def _get_spec_decode_verify_steps(self) -> int:
        return get_spec_decode_verify_steps(self.meta_info)

    def _prefill_recurrent_state_uses_cache(self) -> bool:
        return _model_config_prefill_recurrent_state_uses_cache(self.meta_info.model_config)

    def get_input_sequence_length(self) -> int:
        if getattr(self, "_llm_prefill", True):
            return self.meta_info.model_config.prefill_chunk_length
        return self._get_spec_decode_verify_steps()

    def get_tf_processor(self):
        processor = XHQwen3_5Processor.from_pretrained(self.hf_model_dir)
        meta_info = self.meta_info.model_config
        processor.config.patch_size = meta_info.visual_config.patch_size
        processor.config.max_size_h = meta_info.visual_config.max_size_h
        processor.config.max_size_w = meta_info.visual_config.max_size_w
        processor.config.visual_input_mode = getattr(meta_info.visual_config, "visual_input_mode", "image")
        if getattr(meta_info.visual_config, "gears", None):
            max_patch_capacity = max(int(gear.patch_token_capacity) for gear in meta_info.visual_config.gears)
            processor.config.max_pixels = max_patch_capacity * int(meta_info.visual_config.patch_size) ** 2
        return processor

    def to_fast(self):
        """转换为快速推理模式，返回一个新的模型实例。"""
        # 默认实现直接返回自己，子类可以重写此方法以支持快速推理模式
        if self.fast_mode:
            return self
        self.fast_mode = True
        self.prefill_model.to_fast()
        # self.decode_model.to_fast()
        # self.visual.to_fast()
        return self

    def forward(self, *args):
        args = normalize_hybrid_hmonnx_args(args)
        outs = super().forward(*args)

        return commit_hybrid_cache_outputs(self, outs, model_label="Qwen3.5")

    def _get_data_preprocessor(self) -> Qwen3_5_DataPreprocess:
        input_sequence_length = self.get_input_sequence_length()

        data_preprocess = Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=input_sequence_length,
            image_size_w=self.visual_meta.max_size_w,
            image_size_h=self.visual_meta.max_size_h,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=self.visual_meta.patch_size,
            # temporal_patch_size=self.visual_meta.temporal_patch_size,
            image_token_id=self.meta_info.model_config.image_token_id,
            video_token_id=self.meta_info.model_config.video_token_id,
            vision_start_token_id=self.meta_info.model_config.vision_start_token_id,
            vision_end_token_id=self.meta_info.model_config.vision_end_token_id,
            spatial_merge_size=self.meta_info.model_config.spatial_merge_size,
            enable_page_attention=self.enable_page_attention,
        )
        return data_preprocess

    def _set_enable_golden(self, enable: bool) -> None:
        super()._set_enable_golden(enable)
        self.visual.enable_golden = enable
