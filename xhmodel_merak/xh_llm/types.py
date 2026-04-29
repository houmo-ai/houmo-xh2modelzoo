from pathlib import Path
from typing import Any, Iterable, Optional, TypeVar

import torch
import torch.nn as nn

from xhquant.api import DeviceType, QuantScheme

from ..configuration_utils import BaseConfig, HFModelConfig
from ..utils import CaseInsensitiveEnum


T = TypeVar("T")


class CacheList(list[T]):
    """轻量 list 子类，支持 [] 访问与切片。"""

    def __init__(self, items: Optional[Iterable[T]] = None):
        super().__init__(items if items is not None else [])

    def to_list(self) -> list[T]:
        return list(self)


class LLMModelState(CaseInsensitiveEnum):
    NONE = "none"
    WRAP = "wrap"
    EAGER_FAST = "eager_fast"
    EAGER_ALIGNED = "eager_aligned"
    FRONTED = "fronted"
    QUANTED_DISABLE = "quanted_disable"
    QUANTED_FAST = "quanted_fast"
    QUANTED_ALIGNED = "quanted_aligned"
    EXPORTED = "exported"

    @classmethod
    def from_string(cls, value: str) -> "LLMModelState":
        """从字符串构造 ModelState（大小写不敏感）。

        Args:
            value: 模型状态字符串，可以是任意大小写格式

        Returns:
            对应的 ModelState 枚举实例

        Raises:
            ValueError: 如果字符串不对应任何有效的模型状态

        Examples:
            >>> ModelState.from_string("wrap")
            ModelState.WRAP
            >>> ModelState.from_string("WRAP")
            ModelState.WRAP
            >>> ModelState.from_string("Wrap")
            ModelState.WRAP
        """
        try:
            return cls(value)
        except ValueError as e:
            raise ValueError(f"无效的模型状态: '{value}'. 有效的值为: {[m.value for m in cls]}") from e

    @classmethod
    def get_all_values(cls) -> list[str]:
        """获取 LLMModelState 所有可用值。

        Returns:
            包含所有模型状态值的列表

        Examples:
            >>> LLMModelState.get_all_values()
            ['none', 'wrap', 'fronted', 'quanted_fast', 'quanted_aligned', 'exported']
        """
        return [member.value for member in cls]


class BaseLLMModelConfig(HFModelConfig):
    """Base configuration class for model conversion."""

    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: Optional[str] = None,
        hf_model: str | None = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: Optional[int] = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        enable_prefill_chunk=False,
        max_pe_length: int = 32768,
        **kwargs,
    ):
        # 内部调试参数
        only_first_block = False
        if "only_first_block" in kwargs:
            only_first_block = kwargs.pop("only_first_block")

        max_layers = None
        if "max_layers" in kwargs:
            max_layers = kwargs.pop("max_layers")

        enable_auto_offload = False
        if "enable_auto_offload" in kwargs:
            enable_auto_offload = kwargs.pop("enable_auto_offload")
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            quant_scheme=quant_scheme,
            quant_weight=quant_weight,
            hf_model=hf_model,
            enable_prefill_chunk=enable_prefill_chunk,
            **kwargs,
        )

        self.hf_model = hf_model
        if isinstance(quant_scheme, dict):
            target_device = DeviceType(self.chip_arch)
            if "target_device" not in quant_scheme:
                quant_scheme["target_device"] = target_device
            quant_scheme = QuantScheme(**quant_scheme)
        assert quant_scheme is None or isinstance(quant_scheme, QuantScheme), (
            "quant_scheme must be a dict or QuantScheme instance"
        )
        self.quant_scheme = quant_scheme

        self.only_first_block = only_first_block
        self.max_layers = max_layers
        self.enable_auto_offload = enable_auto_offload

        self.batch_size = batch_size
        self.context_max_length = context_max_length
        self.prefill_chunk_length = prefill_chunk_length
        self.num_logits_to_keep = num_logits_to_keep
        self.quant_weight = quant_weight
        self.mix_search = mix_search
        self.use_cache = use_cache
        self.max_pe_length = max_pe_length

    def get_max_decode_layers(self) -> int:
        max_layers = -1
        if hasattr(self, "only_first_block") and self.only_first_block:
            max_layers = 1
        if max_layers <= 0 and hasattr(self, "max_layers") and self.max_layers is not None:
            max_layers = self.max_layers
        return max_layers

    # def to_json_string(self, use_diff: bool = True) -> str:
    #     config_dict = self.to_dict()
    #     return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    # def __repr__(self):
    #     return f"{self.__class__.__name__} {self.to_json_string()}"


TensorShape = list[int]


class KVCacheConfig(BaseConfig):
    def __init__(
        self,
        *,
        num_layers: int = -1,
        kv_cache_shape: TensorShape | list[TensorShape] | None = None,
        cache_axis: int = 2,
        batch_size: int = 1,
        cache_dtype: str = "float16",
        use_cache: bool = True,
    ):
        self.num_layers = num_layers
        self.kv_cache_shape = kv_cache_shape
        self.cache_axis = cache_axis
        self.batch_size = batch_size
        self.cache_dtype = cache_dtype
        self.use_cache = use_cache

    @property
    def cache_torch_dtype(self):
        return getattr(torch, self.cache_dtype)


class LinearKVCacheConfig(BaseConfig):
    def __init__(
        self,
        conv_dim: int = -1,
        conv_kernel_size: int = -1,
        num_v_heads: int = -1,
        head_k_dim: int = -1,
        head_v_dim: int = -1,
        num_layers: int = -1,
        batch_size: int = 1,
        cache_dtype: str = "float16",
    ):
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cache_dtype = cache_dtype

    @property
    def cache_torch_dtype(self):
        return getattr(torch, self.cache_dtype)


class KVCacheWithLinearConfig(KVCacheConfig):
    def __init__(
        self,
        *,
        num_layers: int = -1,
        kv_cache_shape: TensorShape | list[TensorShape] | None = None,
        cache_axis: int = 2,
        batch_size: int = 1,
        cache_dtype: str = "float16",
        linear_kv_cache_config: dict = None,
    ):
        super().__init__(
            num_layers=num_layers,
            kv_cache_shape=kv_cache_shape,
            cache_axis=cache_axis,
            batch_size=batch_size,
            cache_dtype=cache_dtype,
        )
        self.linear_kv_cache_config: LinearKVCacheConfig = (
            LinearKVCacheConfig(**linear_kv_cache_config)
            if linear_kv_cache_config is not None
            else LinearKVCacheConfig()
        )


class ModelMeta(BaseConfig):
    pass


class LLMModelMeta(ModelMeta):
    KVCACHE_CONFOG_CLS = KVCacheConfig

    def __init__(
        self,
        *,
        create_time: str = "",
        model_config: dict | BaseLLMModelConfig = None,
        hf_config: str = "",
        quant_embedding: str = "",
        quant_embedding_md5: str = "",
        kv_cache: dict | KVCacheConfig = None,
        prefill_hmonnx_md5: str = "",
        decode_hmonnx_md5: str = "",
        prefill_hmonnx: str = "",
        decode_hmonnx: str = "",
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.create_time: str = create_time
        self.model_config = model_config
        self.hf_config: str = hf_config

        self.quant_embedding: str = quant_embedding
        self.quant_embedding_md5: str = quant_embedding_md5
        if kv_cache is not None:
            self.kv_cache: KVCacheConfig = (
                kv_cache if isinstance(kv_cache, KVCacheConfig) else self.KVCACHE_CONFOG_CLS(**kv_cache)
            )
        else:
            self.kv_cache: KVCacheConfig = kv_cache
        self.prefill_hmonnx_md5: str = prefill_hmonnx_md5
        self.decode_hmonnx_md5: str = decode_hmonnx_md5
        self.prefill_hmonnx: str = prefill_hmonnx
        self.decode_hmonnx: str = decode_hmonnx
        self.meta = dict(
            class_name=type(self).__name__,
        )
        self.pad_token_id = pad_token_id
        for kw_name, value in kwargs.items():
            if not kw_name.startswith("_"):
                setattr(self, kw_name, value)

    @classmethod
    def from_dict(cls, config_dict: dict):
        if "meta" in config_dict:
            meta_info = config_dict["meta"]
            if meta_info.get("class_name", None) != cls.__name__:
                raise ValueError(
                    f"Invalid meta class name: {meta_info.get('class_name', None)}, expected: {cls.__name__}"
                )
            config_dict.pop("meta")
        if "_meta_path_" in config_dict:
            _meta_path_ = config_dict.get("_meta_path_")
            config_dict["hf_config"] = str(Path(_meta_path_).parent / config_dict["hf_config"])
            config_dict["quant_embedding"] = str(Path(_meta_path_).parent / config_dict["quant_embedding"])
            config_dict["prefill_hmonnx"] = str(Path(_meta_path_).parent / config_dict["prefill_hmonnx"])
            config_dict["decode_hmonnx"] = str(Path(_meta_path_).parent / config_dict["decode_hmonnx"])
        meta_info = super().from_dict(config_dict)
        return meta_info


class VisualModelMeta(ModelMeta):
    def __init__(self, *, image_size_w: int = None, image_size_h: int = None):
        self.image_size_w = image_size_w
        self.image_size_h = image_size_h
        self.hmonnx: str = None


class VLLMModelMeta(LLMModelMeta):
    def __init__(
        self,
        *,
        visual_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_config = visual_config
        if "_meta_path_" in kwargs and self.visual_config is not None and getattr(self.visual_config, "hmonnx", None):
            _meta_path_ = kwargs.get("_meta_path_")
            self.visual_config.hmonnx = str(Path(_meta_path_).parent / self.visual_config.hmonnx)


class TextLLMModelConfig(BaseLLMModelConfig):
    """Configuration class for Large Language Model conversion.

    Attributes:
        batch_size: Batch size for inference
        context_max_length: Maximum context length
        prefill_chunk_length: input sequence length for prefill chunk
        quant_scheme: Quantization scheme
        quant_weight: Path to quantization weight file (optional)
        mix_search: Whether to use mixed search (optional)
        num_logits_to_keep: Number of logits to keep (default: 1, which means keeping only the last token's logit)
    """

    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str | None = None,
        hf_model: str | None = None,
        quant_scheme: dict | QuantScheme | None = None,
        quant_weight: Optional[str] = None,
        batch_size: int = 1,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: Optional[int] = 1,
        mix_search: bool = False,
        use_cache: bool = True,
        max_pe_length: int = 32768,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            quant_scheme=quant_scheme,
            quant_weight=quant_weight,
            hf_model=hf_model,
            batch_size=batch_size,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            num_logits_to_keep=num_logits_to_keep,
            mix_search=mix_search,
            use_cache=use_cache,
            max_pe_length=max_pe_length,
            **kwargs,
        )


class ExportData:
    def __init__(self) -> None:
        self.exported_dir = None
        self.meta: LLMModelMeta = None
        self.model_name: str = None
        self.str_datetime: str = None


class ModelSwitcher:
    def __init__(self, models: dict[str, Any]):
        self._models = models
        self._activate_model = None

    @property
    def activate_model(self):
        return self._activate_model

    def eval(self):
        for model in self._models.values():
            model.eval()

    def set_activate_model(self, model_name: str):
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found in GroupModel")
        self._activate_model = self._models[model_name]

    def apply(self, fn):
        for model in self._models.values():
            if isinstance(model, nn.Module):
                model.apply(fn)
        fn(self)
        return self

    def __getattr__(self, key: str):
        # 只有 _models 中存在的 key 才走属性代理，避免无限递归
        _models = object.__getattribute__(self, "_models")
        if key in _models:
            return _models[key]
        raise AttributeError(f"'ModelSwitcher' object has no attribute '{key}'")

    def __setattr__(self, key: str, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._models[key] = value

    def __getitem__(self, key):
        return self._models[key]

    def __setitem__(self, key, value):
        self._models[key] = value

    def __delitem__(self, key):
        del self._models[key]

    def __len__(self):
        return len(self._models)

    def __iter__(self):
        return iter(self._models)

    def __contains__(self, key):
        return key in self._models

    def keys(self):
        return self._models.keys()

    def values(self):
        return self._models.values()

    def items(self):
        return self._models.items()

    def get(self, key, default=None):
        return self._models.get(key, default)

    @property
    def __class__(self):  # type: ignore[override]
        if self._activate_model is not None:
            return type(self._activate_model)
        return type(self)

    def __call__(self, *args, **kwargs):
        return self._activate_model(*args, **kwargs)

    def to(self, *args, **kwargs):
        for model in self._models.values():
            model.to(*args, **kwargs)
        return self
