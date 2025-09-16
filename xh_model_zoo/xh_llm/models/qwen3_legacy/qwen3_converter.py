import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional,Callable

import torch
import torch.nn as nn
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, Qwen3ForCausalLM

from ..base_converter import BaseConverter, HFTransfromersConverter,update_cfg_after_quanted
from ..builder import wrap_llm_model
from .qwen3_convert_config import Qwen3LegacyConvertConfig
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
)

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
    assert quant_weight.max() < 8 and quant_weight.min() >= -8, f"{quant_weight.max()} {quant_weight}.min()"
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

class Qwen3LegacyConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3LegacyConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None
    
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

    def load_gptq_model(self, hf_model_dir:str, **kwargs):
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **kwargs).eval()  # quantization_config={"use_exllama": False}
        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False

        hf_model = self.untied_weights(hf_model)
        
        assert hasattr(hf_model, "hf_quantizer")
        hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

        from transformers.utils import is_auto_gptq_available, is_gptqmodel_available

        converter: Optional[Callable] = None

        QuantLinear = hf_quantizer.optimum_quantizer.quant_linear  # type: ignore
        if is_auto_gptq_available():
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear as GeneralQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as CudaOldQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_exllama import QuantLinear as ExllamaQuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import QuantLinear as Exllamav2QuantLinear
            from auto_gptq.nn_modules.qlinear.qlinear_marlin import QuantLinear as MarlinQuantLinear

            if QuantLinear is GeneralQuantLinear:
                converter = general_qlinear_converter
            elif QuantLinear is CudaOldQuantLinear:
                converter = qlinear_cuda_old_converter
            elif QuantLinear is ExllamaQuantLinear:
                converter = None
            elif QuantLinear is Exllamav2QuantLinear:
                converter = None
            elif QuantLinear is MarlinQuantLinear:
                converter = None

        if is_gptqmodel_available():
            from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
            from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

            if QuantLinear is TorchQuantLinear:
                converter = gptqmodel_torch_qlinear_converter
            elif QuantLinear is MarlinQuantLinear:
                converter = None

        assert converter is not None, f"Not implemented for {QuantLinear} yet"

        for name, module in hf_model.named_modules():  # type: ignore
            if isinstance(module, QuantLinear):
                if converter is not None:
                    converter(module)
                else:
                    raise NotImplementedError(f"Not implemented for {type(QuantLinear)} yet")

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model
    
    def load_hf_model(self, hf_model_dir: str, **kwargs):
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        assert not hasattr(config, "quantization_config")
        native_model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **kwargs)
        assert not hasattr(native_model, "hf_quantizer")
        assert isinstance(
            native_model, Qwen3ForCausalLM
        ), f"The model is not Qwen2ForCausalLM, but {type(native_model)}"
        native_model: Qwen3ForCausalLM = native_model  # type: ignore

        if native_model.config.tie_word_embeddings:  # type: ignore
            old_torchscript = native_model.config.torchscript  # type: ignore
            native_model.config.torchscript = True  # type: ignore
            native_model.tie_weights()  # type: ignore
            native_model.config.tie_word_embeddings = False  # type: ignore
            native_model.config.torchscript = old_torchscript  # type: ignore

        self.hf_model_path = hf_model_dir
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config
        if config.gptqmodel_cfg:
            native_model = self.load_gptq_model(hf_model_path,trust_remote_code=True,torch_dtype=torch.float16, device_map="cpu")
        else:
            native_model = self.load_hf_model(
                hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
            )

        # 融合GPTQ权重
        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)
        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        # batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["hf_model_path"] = hf_model_path
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["quant_weight"] = resume_from

        work_dir = Path(output_dir)
        hf_config_dir = Path(work_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
        ]

        for cfg_file in hf_config_files:
            src_file = Path(hf_model_path) / cfg_file
            dst_file = Path(hf_config_dir) / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, dst_file)
            else:
                logger.warning(f"{src_file} not exists, skip copy")

        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        token_embedding = native_model.model.get_input_embeddings()

        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        from ._model import register_wrap_modules as qwen3_register_wrap_modules  # noqa: F403, F401

        qwen3_register_wrap_modules(native_model)

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                # batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )

        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)

        # 设置kv cache
        num_hidden_layers = wraped_qwen_model.model.config.num_hidden_layers
        head_dim = wraped_qwen_model.model.layers[0].self_attn.head_dim
        # pad_token_id = wraped_qwen_model.config.eos_token_id

        # head_dim = wraped_qwen_model.model.config.hidden_size // wraped_qwen_model.model.config.num_attention_heads
        # num_hidden_layers = wraped_qwen_model.model.config.num_hidden_layers
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            1,
            wraped_qwen_model.model.config.num_key_value_heads,
            wrap_cfg.max_sequence_length,
            head_dim,
        ]
        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=num_decoder_layers,
        )

        past_key_caches = []
        past_value_caches = []
        for _ in range(num_decoder_layers):
            past_key_caches.append(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_value_caches.append(torch.zeros(kv_cache_shape, dtype=torch.float16))

        # 导出Prefill模型
        input_ids = []
        current_input_length = []
        # position_ids = []
        for _ in range(1):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long)
            seq_length = input_id.shape[0]
            # past_seq_length = 0
            # position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long)
            current_input_length.append(seq_length)

            input_id = input_id.unsqueeze(0)
            # position_id = position_id.unsqueeze(0)
            # position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids_t = torch.cat(input_ids, dim=0)
        # position_ids = torch.cat(position_ids, dim=0)

        inputs_embeds = token_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor(current_input_length, dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            # position_ids,
            past_key_caches,
            past_value_caches,
        )
        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            # "position_ids",
        ]
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-{context_length//1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(f"********************* start export prefill model *********************")

        quanted_model = convert_fx_model_to_quanted_model(
            wraped_qwen_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )
        # quant_info_onnx_file = str(Path(output_dir) / "quant_info.onnx")
        # quanted_model.dump_quant_info_to_onnx(quant_info_onnx_file)
        if self.config.update_cfg is not None:
            with open(self.config.update_cfg, "r") as f:
                yaml_dict = yaml.safe_load(f)
            quanted_model = update_cfg_after_quanted(quanted_model,yaml_dict)
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        logger.info(f"********************* start export decode model *********************")
        decode_inputs = (
            inputs_embeds[:, :1, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            # position_ids[:, :1],
            past_key_caches,
            past_value_caches,
        )

        # 更新与input_sequence_length相关的Module
        wrap_cfg.input_sequence_length = 1
        for _, m in quanted_model.named_modules():
            if hasattr(m, "_update_cfg"):
                m._update_cfg(wrap_cfg)

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, decode_inputs, str(decode_onnx_file), input_names, output_names)

        logger.info(f"Export decode model to {decode_onnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3LegacyConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp and (not config.gptqmodel_cfg):
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        Qwen3LegacyConverterXH2a(config)._convert(hf_model_path, output_dir)
