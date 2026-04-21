import json
import shutil
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PreTrainedTokenizer

from xhmodel_merak.configuration_utils import BaseAttrDict
from xhmodel_merak.xh_llm.llm_data_processor import BaseInputProcessorConfig
from xhquant.api import (
    ConfigDict,
    get_xhquant_logger,
    to_export_graph,
    to_export_hmonnx_v2,
    to_frontend_graph,
)
from xhquant.utils import log_function_call

from ..utils import calculate_file_md5
from .base_model import XHBaseModel
from .kv_cache_mixin import KVCacheMixin
from .llm_data_processor import BaseLLMInputProcessor
from .types import BaseLLMModelConfig, ExportData, KVCacheConfig, LLMModelMeta, LLMModelState, ModelSwitcher
from .utils import is_graph_module, unfold_args


def qlinear_cuda_old_converter(self: nn.Module):
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear

    assert isinstance(self, CudaOldQuantLinear)
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**self.bits) - 1
        )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )

        zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
        weight = weight.reshape(-1, self.group_size, weight.shape[2])
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    quant_weight = weight - zeros
    weight = scales * quant_weight
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    max_val = 2 ** (self.bits - 1)
    min_val = -max_val

    assert quant_weight.max() < max_val and quant_weight.min() >= min_val, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def general_qlinear_converter(self: nn.Module):
    if self.bits in [2, 4, 8]:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        zeros = torch.bitwise_and(zeros, (2**self.bits) - 1)

        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = torch.bitwise_right_shift(
            torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1),
            self.wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**self.bits) - 1)
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )
        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf.unsqueeze(-1)) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = torch.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    else:
        raise NotImplementedError("Only 2,3,4,8 bits are supported.")

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    # quant_weight = quant_weight.reshape(quant_weight.shape[0] * quant_weight.shape[1], quant_weight.shape[2])

    maxq = (2**self.bits) / 2

    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear
    self.out_features = self.outfeatures
    self.in_features = self.infeatures


def gptqmodel_torch_qlinear_converter(self: nn.Module):
    import torch as t  # conflict with torch.py

    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = 2 ** (self.bits - 1)
    # diff = quant_weight.to(torch.int32) - quant_weight
    # error = diff.abs().float()
    # assert torch.allclose(error, t.tensor(0.0), atol=1e-3), f"{error.max()}"
    assert quant_weight.max() < maxq and quant_weight.min() >= -maxq, (
        f"min={quant_weight.min()}, max={quant_weight.max()}, not in [{-maxq}, {maxq})"
    )
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t().to(torch.int8).contiguous()
    self.register_parameter("weight", nn.Parameter(weight))
    self.__class__ = nn.Linear
    # Store as regular attribute (not buffer) so .to(device) does NOT migrate it.
    # xhquant's ssfp linear.quantize() will lazily move it to weight.device/dtype.
    self.quant_weight = quant_weight


def _dequantize_gptqmodel_hf_model(native_hf_model):
    from transformers.utils import is_gptqmodel_available

    assert is_gptqmodel_available(), "We need gptqmodel to dequantize auto-gptq model"
    converter = gptqmodel_torch_qlinear_converter
    from gptqmodel.nn_modules.qlinear import PackableQuantLinear

    for name, module in native_hf_model.named_modules():  # type: ignore
        if isinstance(module, PackableQuantLinear):
            converter(module)
    return native_hf_model


class BaseLLMModel(XHBaseModel):
    CONFIG_CLS = BaseLLMModelConfig

    def __init__(self, config: BaseLLMModelConfig):
        super().__init__(config)

        self._kvcache_config = KVCacheConfig()  # 不能重新赋值
        self.embed_tokens: nn.Module | None = None

        # self.wrap_cfg会在init_wrap中传入_model.py中
        self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

        # 兼容旧代码
        self.wrap_cfg["max_sequence_length"] = self.wrap_cfg["context_max_length"]
        self.wrap_cfg["input_sequence_length"] = self.wrap_cfg["prefill_chunk_length"]

        self.frontend_type = "TorchFX"

        self._default_pad_token_id = None
        self._inference_model = None
        self._data_processor: Optional[BaseLLMInputProcessor] = None
        self._kvcache_mixin = KVCacheMixin(self.kvcache_config)
        self._llm_prefill: bool | None = None

    @property
    def kvcache_config(self):
        return self._kvcache_config

    @property
    def past_key_caches(self):
        return self._kvcache_mixin.past_key_caches

    @property
    def past_value_caches(self):
        return self._kvcache_mixin.past_value_caches

    @property
    def use_cache(self) -> bool:
        return self._kvcache_mixin.use_cache

    @use_cache.setter
    def use_cache(self, value: bool):
        self._kvcache_mixin.use_cache = value

    def get_kvcache_mixin(self) -> KVCacheMixin:
        return self._kvcache_mixin

    def eval(self):
        super().eval()
        if self.hf_compatible_model is not None:
            self.hf_compatible_model.eval()

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer | None:
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        if self._default_pad_token_id is None or self._default_pad_token_id == 0:
            try:
                self._default_pad_token_id = self.tokenizer.pad_token_id
            except Exception:
                pass
        return self._default_pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._default_pad_token_id = value

    ### 输入处理
    def get_prefill_dummy_inputs(self) -> dict[str, str]:
        # 生成dummy_inputs，供to_frontend使用
        dummy_inputs = {
            "input_ids": torch.randint(0, 100, (1, self.wrap_cfg.prefill_chunk_length), dtype=torch.long),
            "past_seq_length": 0,
        }
        return dummy_inputs

    def get_decode_dummy_inputs(self) -> dict[str, str]:
        dummy_inputs = {
            "input_ids": torch.randint(0, 100, (1, 1), dtype=torch.long),
            "past_seq_length": self.config.prefill_chunk_length,
        }
        return dummy_inputs

    def _to_wrap(self, hf_model):
        # 包装模型，准备进行转换
        self.init_wrap_model(hf_model)
        for _, module in self._wrap_model.named_modules():
            if hasattr(module, "graph_forward"):
                module._eager_forward = module.forward
                module.forward = module.graph_forward

    def to_exported(self):
        raise NotImplementedError

    ### 模型加载
    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        """
        仅仅加载模型结构,不初始化权重,不占用显存
        """
        from accelerate import init_empty_weights

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            # Fallback for older transformers versions
            no_init_weights = init_empty_weights

        with no_init_weights(), init_empty_weights():
            auto_model_cls = cls.HF_AUTO_MODEL_CLS
            model_dtype = cls.HF_MODEL_DTYPE
            if "dtype" not in kwargs:
                kwargs["dtype"] = model_dtype
            config = AutoConfig.from_pretrained(hf_model_dir)
            hf_model: nn.Module = auto_model_cls.from_config(
                config,
                **kwargs,
            )
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
                except OSError:
                    logger = get_xhquant_logger()
                    logger.info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
                pass
        return hf_model

    def get_tokenizer(self, **kwargs):
        assert self.hf_model_dir is not None
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, **kwargs)
        return tokenizer

    ## 模型状态
    def _to_fronted(self, wrap_model):
        # 将模型转换成前端图，准备进行量化
        with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
            data_processor = self.get_data_preprocessor()
            dummy_inputs = self.get_dummy_inputs()
            assert isinstance(dummy_inputs, (dict,)), (
                "Dummy inputs should be a dictionary, but get {type(dummy_inputs)}."
            )
            inputs = data_processor(dummy_inputs)
            assert isinstance(inputs, (list, tuple)), (
                f"Processed dummy inputs should be a list or tuple of tensors, but get {type(inputs)}."
            )

            extra_args = {}
            frontend_model = to_frontend_graph(wrap_model, self.frontend_type, inputs, **extra_args)
            return frontend_model

    ### 模型推理
    def get_dummy_inputs(self) -> dict[str, str]:
        if self._llm_prefill is None or self._llm_prefill:
            return self.get_prefill_dummy_inputs()
        else:
            return self.get_decode_dummy_inputs()

    def is_prefill(self) -> bool:
        return self._llm_prefill if self._llm_prefill is not None else True

    def is_decode(self):
        return not self.is_prefill()

    def set_prefill(self):
        self._llm_prefill = True
        data_processor = self.get_data_preprocessor()
        data_processor.input_sequence_length = self.config.prefill_chunk_length
        self.wrap_cfg.input_sequence_length = self.config.prefill_chunk_length

        self.update_cfg(self.wrap_cfg)

    def set_decode(self):
        self._llm_prefill = False
        data_processor = self.get_data_preprocessor()
        data_processor.input_sequence_length = 1
        self.wrap_cfg.input_sequence_length = 1
        self.update_cfg(self.wrap_cfg)

    def is_support_dynamic_input(self) -> bool:
        if self._state in [
            LLMModelState.EAGER_ALIGNED,
            LLMModelState.EAGER_FAST,
        ]:
            return True
        if self._state in [
            LLMModelState.WRAP,
            LLMModelState.FRONTED,
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
        ]:
            return not self.config.enable_prefill_chunk
        return False

    def _set_device(self, device: torch.device | str | None):
        super()._set_device(device)
        if self.embed_tokens is not None:
            self.embed_tokens = self.embed_tokens.to(device)
        self._kvcache_mixin.to(device)

    def get_num_logits_to_keep(self) -> int:
        return self.config.num_logits_to_keep

    def get_input_embeddings(self):
        return self.embed_tokens

    ### 模型推理

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        preprocessor = BaseLLMInputProcessor(
            BaseInputProcessorConfig(
                embed_tokens=self.embed_tokens,
                input_sequence_length=self.wrap_cfg.input_sequence_length,
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                pad_token_id=self.pad_token_id,
            )
        )
        return preprocessor

    def get_tf_processor(self):
        return XHLLMModelProcessor.from_pretraind(self.hf_model_dir)

    @classmethod
    def _get_hf_model_for_compatible(cls, hf_model_dir=None):
        return cls.get_empty_hf_model(hf_model_dir)

    def generate(self, *args, **kwargs):
        infer_model = None
        old_enable_hf_compatible = self.enable_hf_compatible
        self.enable_hf_compatible = True
        if self._state == LLMModelState.NONE:
            raise RuntimeError("Model is not ready for generation, please set state to fronted or quanted.")
        elif self._state in [LLMModelState.EAGER_FAST, LLMModelState.EAGER_ALIGNED]:
            infer_model = self._wrap_model
        else:
            if self.hf_compatible_model is None:
                hf_model = type(self)._get_hf_model_for_compatible(self.hf_model_dir)
                # 从类中直接获取函数，避免自动绑定 self
                hf_compatible_model = type(self).build_hf_compatible_model(hf_model, self)
                assert isinstance(hf_compatible_model, self.get_hf_model_cls())
                if not self.config.enable_auto_offload:
                    hf_compatible_model.to(device=self.device, dtype=self.dtype)
                self.hf_compatible_model = hf_compatible_model
            infer_model = self.hf_compatible_model
        assert infer_model is not None
        out = infer_model.generate(*args, **kwargs)
        self.enable_hf_compatible = old_enable_hf_compatible
        return out

    def update_cfg(self, cfg: ConfigDict | None):
        inference_model = self.get_inference_model()

        def apply_fn(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(cfg)

        inference_model.apply(apply_fn)

    def set_input_sequence_length(self, input_sequence_length: int):
        self.wrap_cfg.input_sequence_length = input_sequence_length
        self.update_cfg(self.wrap_cfg)
        if self._data_processor is not None:
            self._data_processor.input_sequence_length = input_sequence_length

    def get_input_sequence_length(self) -> int:
        return self.wrap_cfg.input_sequence_length

    def prepare_for_inference(self, *args, **kwargs):
        super().prepare_for_inference(*args, **kwargs)
        if self._data_processor is not None:
            self._data_processor.input_sequence_length = self.config.prefill_chunk_length

    def forward(self, *args, **kwargs):
        inference_model = self._inference_model
        assert inference_model is not None, f"Inference model is not initialized for state: {self._state}"
        if is_graph_module(inference_model):
            args = unfold_args(args)
        out = self._inference_model(*args, **kwargs)
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        return out

    def get_export_fname(self):
        if self.config.model_name is None or len(self.config.model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        return f"{model_name}_{str_datetime}"

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = dict(
            input_names=[
                "inputs_embeds",
                "past_seq_length",
                "current_input_length",
            ],
            output_names=["logits"],
        )

        num_decoder_layers = self.kvcache_config.num_layers
        for layer_idx in range(num_decoder_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")
        return export_cfg

    ## 模型导出
    def _extra_export_metadata(self, output_dir: str, meta_info: LLMModelMeta) -> LLMModelMeta:
        """
        子类可以重写此函数，在meta_info中添加额外的导出信息。"""
        return meta_info

    def create_export_metadata(self, output_dir: str) -> LLMModelMeta:
        logger = get_xhquant_logger()
        meta_info = self.get_export_metadata_cls()()
        meta_info.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_info.pad_token_id = self.pad_token_id
        meta_info.model_config = self.config
        hf_model_path = Path(self.config.hf_model)
        hf_config_files = list(hf_model_path.glob("*.json")) + list(hf_model_path.glob("*.jinja"))
        hf_config_dir = Path(output_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        for cfg_file in hf_config_files:
            src_file = cfg_file
            dst_file = hf_config_dir / cfg_file.name
            if src_file.exists():
                shutil.copyfile(src_file, dst_file)
            else:
                logger.warning(f"{src_file} not exists, skip copy")
        meta_info.hf_config = str(hf_config_dir.relative_to(output_dir))

        token_embedding = self.embed_tokens

        token_embedding_file_path = Path(output_dir) / "quant_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file_path))
        meta_info.quant_embedding = str(token_embedding_file_path.relative_to(output_dir))
        meta_info.quant_embedding_md5 = calculate_file_md5(str(token_embedding_file_path))
        meta_info.kv_cache = self.kvcache_config
        self._extra_export_metadata(output_dir, meta_info)
        return meta_info

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        if self.config.model_name is None or len(self.config.model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        model_name = self.config.model_name.lower()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_name = f"hmquant_{model_name}_{str_datetime}"
        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = model_name
        export_data.str_datetime = str_datetime
        return export_data

    @log_function_call()
    def _export_hmonnx(self, exported_info: ExportData):
        meta_info = exported_info.meta
        logger = get_xhquant_logger()
        model_name = exported_info.model_name
        export_model_name = exported_info.model_name
        output_dir_path = Path(exported_info.exported_dir)
        meta_info.kv_cache = self.kvcache_config
        with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
            self.set_input_sequence_length(self.wrap_cfg.prefill_chunk_length)

            if isinstance(self._quanted_model, ModelSwitcher):
                prefill_quanted_model = self._quanted_model.prefill
                decode_quanted_model = self._quanted_model.decode
            else:
                prefill_quanted_model = self._quanted_model
                decode_quanted_model = self._quanted_model

            if not prefill_quanted_model.is_fixed():
                raise ValueError("prefill_quanted_model model is not fixed, Please call `fixed` first.")
            prefill_quanted_model.to("cpu")

            if not decode_quanted_model.is_fixed():
                raise ValueError("decode_quanted_model model is not fixed, Please call `fixed` first.")
            decode_quanted_model.to("cpu")

            logger.info(f"Exporting Prefill for {model_name} model .........")

            self.set_prefill()
            data_processor = self.get_data_preprocessor()
            dummy_input = self.get_dummy_inputs()
            data_processor.input_sequence_length = self.config.prefill_chunk_length
            inputs = data_processor(dummy_input)
            inputs = unfold_args(inputs)
            prefill_exported_model = to_export_graph(prefill_quanted_model, inputs)

            # logger.info(f"Prefill exported model graph: \n{prefill_exported_model.graph}")

            logger.info(f"Exporting Prefill to HMONNX format for {model_name} .........")
            prefill_dir = output_dir_path / "prefill"
            prefill_dir.mkdir(parents=True, exist_ok=True)
            # 导出的文件格式必须是 hmquant_{model_name}_{prefill/decode}_with_act.onnx
            prefill_hmonnx_file = str(prefill_dir / f"{export_model_name}_prefill.onnx")
            export_cfg = self.get_export_cfg()
            prefill_hmonnx_file = to_export_hmonnx_v2(
                prefill_exported_model, inputs, str(prefill_hmonnx_file), export_cfg, normalize_onnx_name=True
            )
            meta_info.prefill_hmonnx_md5 = calculate_file_md5(prefill_hmonnx_file)
            meta_info.prefill_hmonnx = str(Path(prefill_hmonnx_file).relative_to(output_dir_path))

            logger.info(f"Exporting Decode for {model_name} model .........")
            self.set_decode()
            data_processor = self.get_data_preprocessor()
            self.set_input_sequence_length(1)
            dummy_input = self.get_dummy_inputs()
            inputs = data_processor(dummy_input)
            inputs = unfold_args(inputs)
            decode_exported_model = to_export_graph(decode_quanted_model, inputs)

            logger.info(f"Exporting Decode to HMONNX format for {model_name} .........")
            decode_dir = output_dir_path / "decode"
            decode_dir.mkdir(parents=True, exist_ok=True)
            decode_hmonnx_file = str(decode_dir / f"{export_model_name}_decode.onnx")
            export_cfg = self.get_export_cfg()
            decode_hmonnx_file = to_export_hmonnx_v2(
                decode_exported_model, inputs, str(decode_hmonnx_file), export_cfg, normalize_onnx_name=True
            )
            meta_info.decode_hmonnx_md5 = calculate_file_md5(decode_hmonnx_file)
            meta_info.decode_hmonnx = str(Path(decode_hmonnx_file).relative_to(output_dir_path))
        return exported_info

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> LLMModelMeta:
        """
        导出模型的目录结构如下：
        hmquant_{model_name}_{date}/
            golden_meta_info.json  # 包含模型的基本信息和导出信息
            quant_embedding.pt  # 量化后的输入嵌入权重
            prefill/
                hmquant_{model_name}_{date}_prefill_with_act.onnx  # prefill阶段的HMONNX模型
                hmquant_{model_name}_{date}_prefill_external_data
            decode/
                hmquant_{model_name}_{date}_decode_with_act.onnx  # decode阶段的HMONNX模型
                hmquant_{model_name}_{date}_decode_external_data
        to_export_hmonnx_v2的normalize_onnx_name参数会对导出的模型进行规范化命名。
        """
        logger = get_xhquant_logger()
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()

        exported_info = self.get_export_info(output_dir)
        self._export_hmonnx(exported_info)
        meta_info = exported_info.meta
        exported_dir_path = exported_info.exported_dir

        meta_info = meta_info.to_dict()
        json.dump(meta_info, open(Path(exported_dir_path) / "golden_meta_info.json", "w"), indent=4)
        logger.info(f"Exporting completed! Exported model is saved at: {exported_dir_path}")
        return meta_info


class XHLLMModelProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretraind(cls, pretrained_model_name_or_path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(tokenizer)

    def apply_chat_template(
        self,
        message: list[dict[str, str]],
        enable_think=False,
        **kwargs,
    ):
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_think,  # Switches between thinking and non-thinking modes. Default is True.
            **kwargs,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True)
        return model_inputs
