# Copyright 2025 HOUMO AI
#
# File: base_model.py
# Description:
#   Base model implementation for xh_model_zoo.
#   This module provides the BaseModel class with support for model loading,
#   compression, and various optimization techniques.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import copy
import re
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from compressed_tensors import (
    ModelCompressor,
    SparsityCompressionConfig,
    delete_offload_parameter,
    has_offloaded_params,
    register_offload_parameter,
)
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerationConfig
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    # transformers >= 5.x removed no_init_weights; provide a minimal fallback
    import contextlib

    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield
from xhquant.api import (
    ConfigDict,
    ExportedGraph,
    FrontendGraph,
    FrontendType,
    FXInterpreter,
    Hook,
    PrecisionMode,
    QuantGraph,
    export_onnx,
    to_export_graph,
    to_export_hmonnx,
    to_export_hmonnx_v2,
    to_frontend_graph,
    to_quant_graph,
)
from xhquant.core import CacheTensor

from xhquant.api import Config, get_root_logger
from xhquant.utils.registry.dynamic_module import DynamicModule
from .builder import XHLLM_TRACEABLE_MODULES, wrap_llm_model
from .eval_model_type import EvalModelType
from .generation_mixin import BaseGenerationMixin


def untie_word_embeddings(model: PreTrainedModel):
    """
    Patches bug where HF transformers will fail to untie weights under specific
    circumstances (https://github.com/huggingface/transformers/issues/33689).

    This function detects those cases and unties the tensors if applicable

    :param model: model to fix
    """
    input_embed = model.get_input_embeddings()
    output_embed = model.get_output_embeddings()
    logger = get_root_logger()
    for module in (input_embed, output_embed):
        if module is None or not hasattr(module, "weight"):
            logger.warning(f"Cannot untie {module} which does not have weight param")
            continue

        # this could be replaced by a `get_offloaded_parameter` util
        if not has_offloaded_params(module):
            untied_data = module.weight.data.clone()
        else:
            untied_data = module._hf_hook.weights_map["weight"].clone()

        requires_grad = module.weight.requires_grad
        new_parameter = torch.nn.Parameter(untied_data, requires_grad=requires_grad)
        delete_offload_parameter(module, "weight")
        register_offload_parameter(module, "weight", new_parameter)

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


class BaseModel(nn.Module):
    frontend_type: FrontendType = FrontendType.TorchExport

    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type=None,
        allow_quant=True,
        export_cfg=None,
    ):
        """
        hf_model: huggingface model or path to model
        wrap_cfg: 改写模型的配置,用于torch.fx追踪计算图
        quant_config: 量化配置,参看xhquanttool的配置
        frontend_type: TorchFX or ONNX or TorchExport
        allow_quant: whether to allow quantization
        export_onnx_cfg: export onnx 的配置
        """
        super().__init__()

        self.hf_model_dir = hf_model

        self.quant_cfg = quant_config
        self._wrap_model: Optional[DynamicModule] = None
        self._frontend_model: Optional[FrontendGraph] = None
        self._quanted_model: Optional[QuantGraph] = None
        self._activate_eval_type = EvalModelType.NONE
        self._exported_model: Optional[ExportedGraph] = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.allow_quant = allow_quant

        self._hooks: List[Hook] = []
        self.interactive_mode = False
        self.frontend_type = frontend_type

        self._exec_device = None
        self._tokenizer = None

        # kv cache始终放在CPU上
        self.past_key_caches: List[CacheTensor] = []
        self.past_value_caches: List[CacheTensor] = []

        # self.use_cache = wrap_cfg.use_cache
        # # Set up caching length based on configuration settings.
        # if wrap_cfg.use_cache:
        #     self.cache_length = wrap_cfg.max_sequence_length
        # else:
        #     self.cache_length = 0
        self.export_cfg = export_cfg
        # self.input_sequence_length = wrap_cfg.input_sequence_length
        self.wrap_cfg = wrap_cfg
        # self.max_sequence_length = wrap_cfg.max_sequence_length
        # self.pad_token_id = 0

        # 适配GenerationMixin的Generate方法
        # self.main_input_name = "input_ids"
        # self._supports_cache_class = False
        # self.generation_config: Optional[GenerationConfig] = None
        # self.config: Optional[PreTrainedModel.Config] = None

        # 由特定模型创建
        # Register buffers for past key and value caches with non-persistent storage.
        # self.register_buffer("past_k_cache", None, persistent=False)
        # self.register_buffer("past_v_cache", None, persistent=False)

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        if self._tokenizer is None:
            self._tokenizer = self.get_tokenizer()
        return self._tokenizer

    @property
    def wrap_model(self) -> DynamicModule:
        if self._wrap_model is not None:
            return weakref.proxy(self._wrap_model)
        return self._wrap_model

    @property
    def frontend_model(self) -> FrontendGraph:
        if self._frontend_model is not None:
            return weakref.proxy(self._frontend_model)
        return self._frontend_model

    @property
    def quanted_model(self) -> QuantGraph:
        if self._quanted_model is not None:
            return weakref.proxy(self._quanted_model)
        return self._quanted_model

    @property
    def exported_model(self) -> ExportedGraph:
        if self._exported_model is not None:
            return weakref.proxy(self._exported_model)
        return self._exported_model

    def release_exported_model(self):
        del self._exported_model
        self._exported_model = None

    def release_wraped_model(self):
        del self._wrap_model
        self._wrap_model = None

    def release_frontend_model(self):
        del self._frontend_model
        self._frontend_model = None

    def release_quanted_model(self):
        del self._quanted_model
        self._quanted_model = None

    @property
    def activate_eval_type(self):
        return self._activate_eval_type

    @activate_eval_type.setter
    def activate_eval_type(self, value: EvalModelType):
        self._activate_eval_type = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, dev):
        self._device = dev

    @property
    def execution_device(self):
        return self.device if self._exec_device is None else self._exec_device

    @execution_device.setter
    def execution_device(self, dev):
        self._device = dev

    @property
    def dtype(self):
        return self._dtype

    def get_empty_hf_model(self, device_map="cpu", **kwargs) -> Any:
        """
        仅仅加载模型结构,不初始化权重,不占用显存
        """
        config = AutoConfig.from_pretrained(self.hf_model_dir)
        with no_init_weights(), init_empty_weights():
            hf_model: nn.Module = AutoModelForCausalLM.from_config(
                config,
                # dtype=torch.float16,
                **kwargs,
            )
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(self.hf_model_dir)
                except OSError:
                    logger = get_root_logger()
                    logger.info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
                pass
        return hf_model

    def untied_weights(self, module: nn.Module) -> nn.Module:
        param_ids = {}
        duplicate_params = []
        for name, param in module.named_parameters(remove_duplicate=False):
            param_id = id(param)
            if param_id not in param_ids:
                param_ids[param_id] = param
            else:
                duplicate_params.append(name)

        duplicate_params = list(set(duplicate_params))
        for param_name in duplicate_params:
            fields = param_name.split(".")[:-1]
            m_name = ".".join(fields)
            attr_name = param_name.split(".")[-1]
            m = module.get_submodule(m_name)
            param = getattr(m, attr_name)
            setattr(m, attr_name, nn.Parameter(param.clone()))
        return module

    def _load_hf_model(self, device_map="cpu", **kwargs) -> Any:
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16
        if device_map == "meta":
            import accelerate

            with accelerate.init_empty_weights(include_buffers=False):
                config = AutoConfig.from_pretrained(self.hf_model_dir)
                hf_model: nn.Module = AutoModelForCausalLM.from_config(
                    config,
                    # dtype=torch.float16,
                    # torch_dtype=torch.float16,
                    **kwargs,
                )
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_dir,
                # dtype=torch.float16,
                # torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=device_map,
                # low_cpu_mem_usage=True,
                **kwargs,
            ).eval()
        return hf_model

    def get_hf_model(self, device_map="cpu", **kwargs) -> Any:
        assert self.hf_model_dir is not None
        hf_model = self._load_hf_model(device_map, **kwargs)
        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False

        hf_model = self.untied_weights(hf_model)
        import accelerate
        import accelerate.hooks

        accelerate.hooks.remove_hook_from_module(hf_model)
        return hf_model

    def get_tokenizer(self, **kwargs):
        assert self.hf_model_dir is not None
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, **kwargs)
        return tokenizer

    def init_wrap_model(self, hf_model: Optional[PreTrainedModel] = None) -> Any:
        if hf_model is None:
            hf_model = self.get_hf_model()
        self._wrap_model = wrap_llm_model(hf_model, self.wrap_cfg)  # id(model) == id(wrap_model)

        def check_wraped(wraped_model: nn.Module):
            """
            检查是否被正确的wrap
            """
            for name, module in wraped_model.named_modules():
                if not isinstance(module, DynamicModule) and type(module) in XHLLM_TRACEABLE_MODULES:
                    raise RuntimeError(
                        f"Module {name}[{type(module)}] is  not wrapped by XHQuant, please use XHLLM_TRACEABLE_MODULES.register_module to wrap it."
                    )

                return self._wrap_model

        check_wraped(self._wrap_model)
        return self._wrap_model

    @property
    def need_quant(self):
        return True

    def register_hook(self, hook: Hook):
        self._hooks.append(hook)

    def clear_hooks(self):
        self._hooks = []

    def set_property(self, proerty_name, value):
        def apply_fn(module):
            if not hasattr(module, proerty_name):
                return
            setattr(module, proerty_name, value)

        self.apply(apply_fn)

    def _set_device(self, device: torch.device) -> None:
        """Recursively set device for `BaseDataPreprocessor` instance.

        Args:
            device (torch.device): the desired device of the parameters and
                buffers in this module.
        """

        all_models = []
        if self._wrap_model is not None and isinstance(self._wrap_model, nn.Module):
            all_models.append(self._wrap_model)
        if self._frontend_model is not None and isinstance(self._frontend_model, nn.Module):
            all_models.append(self._frontend_model)
        if self._quanted_model is not None and isinstance(self._quanted_model, nn.Module):
            all_models.append(self._quanted_model)
        if self._exported_model is not None and isinstance(self._exported_model, nn.Module):
            all_models.append(self._exported_model)

        activate_model = self.get_activate_model()

        if device != torch.device("meta"):
            cpu_models = [model for model in all_models if model != activate_model]
            for model in cpu_models:
                model.to(torch.device("cpu"))

        # exported_moded、quanted_model会共享参数，所以当前模型要最后设置device
        if activate_model is not None and isinstance(activate_model, nn.Module):
            activate_model.to(device)

        self._device = device

    def _set_dtype(self, dtype: torch.dtype) -> None:
        if self._wrap_model is not None:
            self._wrap_model.to(dtype)

        if self._frontend_model is not None:
            self._frontend_model.to(dtype)

        if self._quanted_model is not None:
            self._quanted_model.to(dtype)

        self._dtype = dtype

    def _set_exec_device(self, device):
        self._exec_device = device

    def set_exec_device(self, device) -> None:
        def apply_fn(module):
            if not hasattr(module, "_set_exec_device"):
                return
            module._set_exec_device(device)

        self.apply(apply_fn)
        return self

    def half(self):
        return self.to(dtype=torch.float16)

    def to(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.

        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            # self._set_device(torch.device(device))
            def apply_fn(module):
                if not hasattr(module, "_set_device"):
                    return
                module._set_device(device)

            self.apply(apply_fn)
            # return super().to(*args, **kwargs)
            return self
        if dtype is not None:
            # self._set_dtype(dtype)
            def apply_fn(module):
                if not hasattr(module, "_set_dtype"):
                    return
                module._set_dtype(dtype)

            self.apply(apply_fn)
            # return super().to(*args, **kwargs)
            return self

    def cuda(
        self,
        device: Optional[Union[int, str, torch.device]] = None,
    ) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        if device is None or isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._set_device(torch.device(device))
        return super().cuda(device)

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
        additionally.

        Returns:
            nn.Module: The model itself.
        """
        self._set_device(torch.device("cpu"))
        return super().cpu()

    def prepare_inputs(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        pass

    def prepare_inputs_for_graph(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        return self.prepare_inputs(data)

    def forward(
        self,
        *args,
        **kwargs,
    ):
        if self.activate_eval_type in [
            EvalModelType.FRONTEND,
            EvalModelType.QUANTED_DISABLED,
            EvalModelType.QUANTED_ALIGNED,
            EvalModelType.QUANTED_FAST,
            EvalModelType.EXPORTED,
            EvalModelType.CALIBRATION,
        ]:
            ## 将输入的List展开
            new_args = []
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    new_args.extend(arg)
                else:
                    new_args.append(arg)
            args = new_args

            if isinstance(self.execution_device, str):
                self.set_exec_device(torch.device(self.execution_device))
            if isinstance(self.device, str):
                self.device = torch.device(self.device)

            # if self.execution_device != self.device:
            #     assert self.interactive_mode, f"{self.execution_device} != {self.device}, must be in interactive mode"

        if self.activate_eval_type == EvalModelType.WRAPED:
            out = self._wrap_model(
                *args,
                **kwargs,
            )

        elif self.activate_eval_type == EvalModelType.FRONTEND:
            assert self._frontend_model is not None, "Traced model is not available."
            if self.interactive_mode:
                interpreter = FXInterpreter(self._frontend_model)
                interpreter.register_hooks(self._hooks)
                out = interpreter.run(
                    *args,
                    **kwargs,
                )
            else:
                out = self._frontend_model(
                    *args,
                    **kwargs,
                )
        elif self.activate_eval_type in [
            EvalModelType.QUANTED_DISABLED,
            EvalModelType.QUANTED_ALIGNED,
            EvalModelType.QUANTED_FAST,
            EvalModelType.CALIBRATION,
        ]:
            assert self._quanted_model is not None, "Quantized model is not available."
            if self.interactive_mode:
                interpreter = FXInterpreter(self._quanted_model)
                interpreter.register_hooks(self._hooks)
                out = interpreter.run(
                    *args,
                    **kwargs,
                )
            else:
                out = self._quanted_model(
                    *args,
                    **kwargs,
                )
        elif self.activate_eval_type == EvalModelType.EXPORTED:
            assert self._exported_model is not None, "Exported model is not available."

            if self.interactive_mode:
                interpreter = FXInterpreter(self._exported_model)
                interpreter.register_hooks(self._hooks)
                out = interpreter.run(
                    *args,
                    **kwargs,
                )
            else:
                out = self._exported_model(
                    *args,
                    **kwargs,
                )
            if isinstance(out, (tuple, list)) and len(out) == 1:
                out = out[0]
        else:
            # try:
            out = self._wrap_model(
                *args,
                **kwargs,
            )
            # except Exception as e:
            #     raise ValueError(f"Unsupported eval type: {self.activate_eval_type}.")
        return out

    def _forward(sef, *args, **kwargs):
        pass

    @torch.no_grad()
    def test_step(self, data: Union[Dict, Tuple, List]):
        inputs = self.prepare_inputs(data)
        output = self._forward(*inputs)
        return output

    @torch.no_grad()
    def test_step_with_fake_mode(self, data: Union[Dict, Tuple, List]):
        """
        仅权重量化以及设值必要的配置参数
        """
        assert self.activate_eval_type == EvalModelType.CALIBRATION
        inputs = self.prepare_inputs(data)

        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        converter = fake_mode.fake_tensor_converter

        # 递归遍历所有Tensor，将Tensor转换为FakeTensor
        def cast_fake_tensor(data):
            if isinstance(data, (list, tuple)):
                return [cast_fake_tensor(x) for x in data]
            elif isinstance(data, dict):
                return {k: cast_fake_tensor(v) for k, v in data.items()}
            elif isinstance(data, Tensor):
                return converter.from_real_tensor(fake_mode, data)

        inputs = cast_fake_tensor(inputs)
        output = self.forward(*inputs)
        return output

    def _check_quant_model(self):
        assert (
            self._quanted_model is not None
        ), "Quantized model is not available, please call `convert2quantizer` first."

    def _check_frontend_model(self):
        assert self._frontend_model is not None, "Traced model is not available, please call `convert2graph` first."

    def enable_fast_quant(self):
        """
        开启快速量化模型
        """
        if self.need_quant:
            self._check_quant_model()
            self._quanted_model.enable_fast_precision_mode()

    def enable_calibration(self):
        if self.need_quant:
            self._check_quant_model()
            self._quanted_model.enable_calibration()

    def enable_align_hardward_quant(self):
        """
        开启硬件对齐量化模式
        """
        if self.need_quant:
            self._check_quant_model()
            self._quanted_model.enable_aligned_precision_mode()

    def disable_quant(self):
        """
        关闭量化模型
        """
        if self.need_quant:
            self._check_quant_model()
            self._quanted_model.disable_quant()

    def fixed(self):
        def apply_fn(module):
            if isinstance(module, BaseModel):
                if self._quanted_model is not None:
                    self._quanted_model.fixed()

        apply_fn(self)
        self.apply(apply_fn)

    @classmethod
    def can_generate(cls) -> bool:
        return False

    def get_activate_model(self) -> Any:
        if self.activate_eval_type == EvalModelType.WRAPED:
            return self._wrap_model
        elif self.activate_eval_type == EvalModelType.FRONTEND:
            return self._frontend_model
        elif self.activate_eval_type in [
            EvalModelType.QUANTED_DISABLED,
            EvalModelType.QUANTED_ALIGNED,
            EvalModelType.QUANTED_FAST,
            EvalModelType.CALIBRATION,
        ]:
            return self._quanted_model
        elif self.activate_eval_type == EvalModelType.EXPORTED:
            return self._exported_model
        elif self.activate_eval_type == EvalModelType.NONE:
            return None
        else:
            raise ValueError(f"Unsupported eval type: {self.activate_eval_type}.")

    def change_eval_type(self, eval_type: EvalModelType):
        if eval_type == self.activate_eval_type:
            return
        old_eval_type = self.activate_eval_type
        logger = get_root_logger()

        if not self.need_quant and eval_type not in [EvalModelType.WRAPED, EvalModelType.NONE]:
            logger.warning(f"{type(self).__name__} does not allow quantization, skip change eval type.")
            return

        if eval_type == EvalModelType.WRAPED:
            assert self._wrap_model is not None, "wraped model is not available."
        if eval_type == EvalModelType.FRONTEND:
            if old_eval_type in [
                EvalModelType.QUANTED_DISABLED,
                EvalModelType.QUANTED_ALIGNED,
                EvalModelType.QUANTED_FAST,
                EvalModelType.CALIBRATION,
            ]:
                raise ValueError(f"Cannot change eval type from {old_eval_type} to {eval_type}.")
            assert self._frontend_model is not None, "Traced model is not available."
        elif eval_type == EvalModelType.QUANTED_DISABLED:
            assert self._quanted_model is not None, "Quantized model is not available."
            self._quanted_model.disable_quant()
        elif eval_type == EvalModelType.QUANTED_ALIGNED:
            assert self._quanted_model is not None, "Quantized model is not available."
            self._quanted_model.enable_quant()
            self._quanted_model.enable_aligned_precision_mode()
        elif eval_type == EvalModelType.QUANTED_FAST:
            assert self._quanted_model is not None, "Quantized model is not available."
            self._quanted_model.enable_quant(precisionMode=PrecisionMode.FAST)
            self._quanted_model.enable_fast_precision_mode()
        elif eval_type == EvalModelType.EXPORTED:
            assert self._exported_model is not None, "Exported model is not available."
        self.activate_eval_type = eval_type

        logger = get_root_logger()
        logger.info(f"{type(self).__name__} eval type changed from {old_eval_type} to {self.activate_eval_type}.")

        # 由调用方控制模型的设备
        # activate_model = self.get_activate_model()
        # if activate_model is not None and isinstance(activate_model, nn.Module):
        #     activate_model.to(self.device)

    def convert_to_fronted_graph(self, data: Union[dict, tuple, list], release_wraped_model: bool = True, **extra_args) -> Optional[FrontendGraph]:
        if not self.need_quant:
            return None
        if self._frontend_model is not None:
            return self._frontend_model
        if isinstance(data, dict):
            inputs = self.prepare_inputs_for_graph(data)
        else:
            inputs = data
        str_frontend_type = self.frontend_type
        frontend_type = FrontendType(str_frontend_type)
        # extra_args: Dict[str, Any] = {}
        frontend_model: FrontendGraph = to_frontend_graph(self._wrap_model, frontend_type, inputs, **extra_args)
        self._frontend_model = frontend_model
        if release_wraped_model:
            self.release_wraped_model()
        return self._frontend_model

    def convert_to_quant_graph(self, target_device: str) -> Optional[QuantGraph]:
        if not self.need_quant:
            return None

        if self._quanted_model is not None:
            return self._quanted_model
        assert self._frontend_model is not None, "Frontend model is not available."
        if "inputs" in self.quant_cfg:
            self.quant_cfg.pop("inputs")

        self._quanted_model = to_quant_graph(self._frontend_model, target_device, self.quant_cfg)
        self._frontend_model = None  # quant_model会复用frontend_model的计算图，所以这里置空
        return self._quanted_model

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        args = self.prepare_inputs(data)
        return args

    def convert_to_export_graph(self, data: Union[dict, tuple, list]) -> Optional[ExportedGraph]:
        if not self.need_quant:
            return None
        logger = get_root_logger()
        if not self.need_quant:
            logger.warning(f"{type(self).__name__} does not allow quantization, skip convert to exported graph.")
            return None

        if self._exported_model is not None:
            return self._exported_model
        assert self._quanted_model is not None, "Quantized model is not available, Please call `to_quant_graph` first."
        assert self._quanted_model.is_fixed(), "Quantized model is not fixed, Please call `fixed` first."
        self._quanted_model.to("cpu")
        # if isinstance(data, dict):
        #     inputs = self.prepare_inputs_for_graph(data)
        #     ## 将输入的List展开
        #     new_args = []
        #     for arg in inputs:
        #         if isinstance(arg, (list, tuple)):
        #             new_args.extend(arg)
        #         else:
        #             new_args.append(arg)
        #     inputs = new_args
        # else:
        #     inputs = data

        # inputs = [t.to("cpu") if isinstance(t, Tensor) else t for t in inputs]
        inputs = self._get_export_dummy_data(data)
        exported_model = to_export_graph(self._quanted_model, inputs)

        logger.debug(f"************* Start Exported model *************")
        logger.debug(f"{exported_model.graph}")
        logger.debug(f"************* End Exported model *************")

        self._exported_model = exported_model
        return self._exported_model

    @staticmethod
    def xh1_hmonnx_compatible(export_cfg: Dict[str, Any]):
        export_cfg = copy.deepcopy(export_cfg)
        input_names_mapping = {
            "inputs_embeds": "input_1",
            "past_seq_length": "valid_length",
            "current_input_length": "current_length",
        }
        if "input_names" in export_cfg:
            for idx in range(len(export_cfg["input_names"])):
                in_name = export_cfg["input_names"][idx]
                if in_name in input_names_mapping:
                    export_cfg["input_names"][idx] = input_names_mapping[in_name]
                else:
                    # 匹配past_key_cache_后面跟数字的字符串
                    kcache_pattern = r"^past_key_cache_\d+$"  # \d+表示匹配一个或多个数字
                    kcache_match = re.match(kcache_pattern, in_name)
                    if kcache_match:
                        kcache_idx = kcache_match.group(0).split("_")[-1]
                        export_cfg["input_names"][idx] = "model_layers_{}_self_attn_kcache_input".format(kcache_idx)
                    else:
                        vcache_pattern = r"^past_value_cache_\d+$"  # \d+表示匹配一个或多个数字
                        vcache_match = re.match(vcache_pattern, in_name)
                        if vcache_match:
                            vcache_idx = vcache_match.group(0).split("_")[-1]
                            export_cfg["input_names"][idx] = "model_layers_{}_self_attn_vcache_input".format(vcache_idx)
        return export_cfg

    def _get_export_dummy_data(self, data):
        if isinstance(data, dict):
            inputs = self.prepare_inputs_for_graph(data)
            ## 将输入的List展开
            new_args = []
            for arg in inputs:
                if isinstance(arg, (list, tuple)):
                    new_args.extend(arg)
                else:
                    new_args.append(arg)
            inputs = new_args
        else:
            inputs = data

        inputs = [t.to("cpu") if isinstance(t, Tensor) else t for t in inputs]
        return inputs

    def to_export_onnx(self, data: Union[dict, tuple, list], output_dir: str, prefix: str = None) -> List[str]:
        self.convert_to_export_graph(data)
        exported_graph = self._exported_model
        if prefix is None or len(prefix) == 0:
            prefix = type(self).__name__
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        onnx_file = Path(output_dir) / f"{prefix}.onnx"
        export_cfg = BaseModel.xh1_hmonnx_compatible(self.export_cfg)

        inputs = self._get_export_dummy_data(data)

        to_export_hmonnx_v2(exported_graph, inputs, str(onnx_file), export_cfg)
        return [str(onnx_file)]

    def _update_cfg(self, cfg: ConfigDict):
        self.input_sequence_length = cfg["input_sequence_length"]

    def update_cfg(self, cfg: Optional[ConfigDict]):
        def apply_fn(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(cfg)

        self.apply(apply_fn)

    def get_num_logits_to_keep(self):
        return self.wrap_cfg.num_logits_to_keep

    def set_num_logits_to_keep(self, num_logits_to_keep: int = 1):
        """
        num_logits_to_keep: int, 保留的最后一个logits的数量,默认为1,即只保留最后一个logits。
        当num_logits_to_keep=0时,会保留所有的logits。
        """
        self.wrap_cfg.num_logits_to_keep = num_logits_to_keep
        self.update_cfg(self.wrap_cfg)

    def set_input_sequence_length(self, input_sequence_length: int):
        self.wrap_cfg.input_sequence_length = input_sequence_length
        self.update_cfg(self.wrap_cfg)

    def get_input_sequence_length(self) -> int:
        return self.wrap_cfg.input_sequence_length

    def set_batch_size(self, batch_size: int):
        self.wrap_cfg.batch_size = batch_size
        self.update_cfg(self.wrap_cfg)
        # 更新kv Cache
        if len(self.past_key_caches) > 0:
            kv_cache_shape = list(self.past_key_caches[0].shape)
            old_batch_size = kv_cache_shape[0]

            if batch_size != old_batch_size:
                kv_cache_shape[0] = batch_size
                self.past_key_caches = [
                    torch.zeros(kv_cache_shape, dtype=cache.dtype, device=cache.device)
                    for cache in self.past_key_caches
                ]
                self.past_value_caches = [
                    torch.zeros(kv_cache_shape, dtype=cache.dtype, device=cache.device)
                    for cache in self.past_value_caches
                ]

    def load_wraped_model_state_dict(self, native_model, checkpoint: str):
        logger = get_root_logger()

        archive_file = checkpoint
        logger.info(f"Load previously saved checkpoint from: {archive_file}")
        is_safetensors = archive_file.endswith(".safetensors")

        if is_safetensors:
            state_dict: Dict[str, Tensor] = load_safetensors_file(archive_file, device="cpu")
        else:
            state_dict: Dict[str, Tensor] = torch.load(archive_file, weights_only=True, map_location="cpu")

        model_state_dict = native_model.state_dict()
        unexpect_state_dict: List[str] = []
        for k, v in state_dict.items():
            if k not in model_state_dict:
                unexpect_state_dict.append(k)

        for k in unexpect_state_dict:
            paths = k.split(".")
            if paths[-1] == "quant_weight":
                submodule_name = ".".join(paths[:-1])
                submodule = native_model.get_submodule(submodule_name)
                # submodule = get_submodule(native_model, k)
                v = state_dict[k]
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
                submodule.register_buffer("quant_weight", v, persistent=False)
                logger.debug(f"add quant_weight to {submodule_name}")
            else:
                logger.warning(f"ignore unexpect state dict: {k}")
            state_dict.pop(k)

        native_model.load_state_dict(state_dict)
        del state_dict

    def kv_cache_to(self, device: Union[str, torch.device] = torch.device("cpu")):
        for i in range(len(self.past_key_caches)):
            self.past_key_caches[i] = self.past_key_caches[i].to(device)
            self.past_value_caches[i] = self.past_value_caches[i].to(device)

    def reset_kvcache(self):
        for i in range(len(self.past_key_caches)):
            self.past_key_caches[i].reset()
            self.past_value_caches[i].reset()
