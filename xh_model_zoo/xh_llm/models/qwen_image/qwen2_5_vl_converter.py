import json
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import onnx
import torch
import torch.nn as nn
from PIL import Image
from .vision_process import process_vision_info
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from xhquant.api import convert_fx_model_to_quanted_model, convert_onnx_to_hmonnx, convert_quanted_model_to_hmonnx
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .data_preprocess import Qwen2_5_VLDataPreprocess
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .qwen2_5_vl_convert_config import Qwen2_5_VLConvertConfig

from xhquant.api import (  # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)


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


class Qwen2_5_VLConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen2_5_VLConvertConfig):
        super().__init__()
        self.config = config
        self.work_dir = None
        self.pad_token_id = 0

        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_end_token_id = 151653
        self.vision_token_id = 151654
        self.eos_token_id = [151645, 151643]
        self.spatial_merge_size = 2
        self.window_size = 112

    def load_hf_model(self, hf_model_path, **kwargs) -> Qwen2_5_VLForConditionalGeneration:
        native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(hf_model_path, **kwargs)
        if native_model.config.tie_word_embeddings:
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
        native_model.eval()
        return native_model

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

    def load_gptq_model(self, hf_model_dir: str, **kwargs):
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_model_dir, **kwargs
        ).eval()  # quantization_config={"use_exllama": False}
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

    def _export_vision(
        self, inputs, hf_model: Qwen2_5_VLForConditionalGeneration, vision_hmonnx_file: str, golden_dir: str
    ):
        logger = get_root_logger()
        logger.info("********************* start export vision model *********************")
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        visual = hf_model.visual
        visual.eval()
        visual.cpu()
        wrap_cfg = dict(
            max_size_w=self.config.visual_config.image_max_size_w,
            max_size_h=self.config.visual_config.image_max_size_h,
            max_size_t=self.config.visual_config.image_max_size_t,
            patch_size=self.config.visual_config.patch_size,
            temporal_patch_size=self.config.visual_config.temporal_patch_size,
        )
        vision_register_wrap_cls(hf_model)
        wraped_vision_model = wrap_llm_model(visual, wrap_cfg)
        wraped_vision_model.float().eval()
        wraped_vision_model.cpu()

        hm_pixel_values = inputs["hm_pixel_values"][0].type(wraped_vision_model.dtype).to(wraped_vision_model.device)
        hm_pixel_values = hm_pixel_values.unsqueeze(2).repeat(1, 1, self.config.visual_config.image_max_size_t, 1, 1)
        window_index = wraped_vision_model.window_index
        attention_bias = wraped_vision_model.attention_bias

        logger.info(f"start export vision model to onnx format............")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_onnx_file = str(Path(tmp_dir) / "visual.onnx")
            torch.onnx.export(
                wraped_vision_model,
                (hm_pixel_values.float().cpu(), window_index.cpu(), attention_bias.cpu()),
                tmp_onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["pixel_values", "window_index", "window_mask"],
                output_names=["image_embeds"],
                verbose=False,
            )
            onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)

        logger.info(f"simplify onnx model............")
        onnx_model, check = simplify_large_onnx(onnx_model)

        input_args = [hm_pixel_values.float().cpu(), window_index.cpu()]
        if not (self.config.visual_config.image_max_size_w % 112 and self.config.visual_config.image_max_size_h % 112):
            pass
        else:
            input_args.append(attention_bias.cpu())

        convert_onnx_to_hmonnx(
            onnx_model,
            input_args,
            self.target_device,
            vision_hmonnx_file,
        )

        logger.info(f"Export vision model to {vision_hmonnx_file}")

        logger.info(f"start export vision model golden............")
        from xhquant.api import HMONNXGoldenInference

        vision_model = HMONNXGoldenInference(vision_hmonnx_file)
        vision_model.save_golden = True
        vision_model.exec_device = torch.device("cuda:0")

        Path(golden_dir).mkdir(exist_ok=True, parents=True)
        vision_model.golden_dir = str(golden_dir)

        input_args = [hm_pixel_values.to(torch.device("cuda:0")).half(), window_index.to(torch.device("cuda:0")).int()]
        if not (self.config.visual_config.image_max_size_w % 112 and self.config.visual_config.image_max_size_h % 112):
            pass
        else:
            input_args.append(attention_bias.to(torch.device("cuda:0")).half())

        with torch.no_grad():
            vision_model.forward(*input_args)       
        logger.info(f"Export vision model golden to {golden_dir}")

    def _convert(self, hf_model, output_dir: str):
        logger = get_root_logger()
        config = self.config

        native_model = hf_model

        # 融合GPTQ权重
        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)

        lm_head = native_model.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        token_embedding = deepcopy(native_model.model.get_input_embeddings())

        model_name = "qwen_image_text_encoder"
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        max_pe_length = config.max_pe_length
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"

        quant_config = create_quant_config(config.quant_scheme)
        print(quant_config)
        work_dir = Path(output_dir)
        self.work_dir = output_dir

        quant_config = ConfigDict(quant_config)

        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                max_pe_length=max_pe_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
                visual=dict(
                    image_max_size_h=config.visual_config.image_max_size_h,
                    image_max_size_w=config.visual_config.image_max_size_w,
                    image_max_size_t=config.visual_config.image_max_size_t,
                    temporal_patch_size=config.visual_config.temporal_patch_size,
                    patch_size=config.visual_config.patch_size,
                ),
            )
        )

        meta_info = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        token_embedding = native_model.model.get_input_embeddings()
        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.cpu(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        Path(work_dir / "hmonnx").mkdir(exist_ok=True, parents=True)

        quant_type = config.quant_scheme.quant_type
        prefix = f"{model_name}-{target_device}-{quant_type}"

        input_ids = torch.randint(0, 1000, (1, 160))
        attention_mask = torch.ones(1, 160)

        # 设置kv cache
        hf_model = native_model
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        head_dim = hf_model.model.config.hidden_size // hf_model.model.config.num_attention_heads
        num_hidden_layers = hf_model.model.config.num_hidden_layers
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            wrap_cfg.batch_size,
            hf_model.model.config.num_key_value_heads,
            wrap_cfg.max_sequence_length,
            head_dim,
        ]
        meta_info["kv_cache"] = {
            "shape": kv_cache_shape,
            "num_decoder_layers": num_decoder_layers,
        }

        past_key_caches = []
        past_value_caches = []
        for _ in range(num_decoder_layers):
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

        # Export LLM model, prefill
        data_prefill = {
            "input_ids": input_ids,
            "image_embeds": None,
            "past_seq_length": 0,
            "image_grid_thw": None,
        }

        input_sequence_length = wrap_cfg.input_sequence_length

        input_seq_len = data_prefill["input_ids"].shape[-1]
        steps = (input_seq_len + input_sequence_length - 1) // input_sequence_length

        data_preprocess = Qwen2_5_VLDataPreprocess(
            token_embedding,
            input_sequence_length * steps,
        )
        data_input = data_preprocess(data_prefill)

        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
        ) = data_input

        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls  # noqa: F401

        llm_register_wrap_cls(hf_model)

        wraped_llm_model = wrap_llm_model(hf_model, wrap_cfg)
        wraped_llm_model.cpu()
        wraped_llm_model.to(torch.float16)

        target_device = self.config.quant_scheme.target_device

        ## prefill
        prefill_inputs = (
            inputs_embeds[:, :input_sequence_length, :],
            time_position_ids[:input_sequence_length],
            height_position_ids[:input_sequence_length],
            width_position_ids[:input_sequence_length],
            past_seq_length,
            torch.tensor([input_sequence_length], dtype=torch.int32).to(inputs_embeds.device),
            past_key_caches,
            past_value_caches,
        )

        onnx_input_names = [
            "inputs_embeds",
            "time_position_ids",
            "height_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
        ]
        for layer_idx in range(num_decoder_layers):
            onnx_input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            onnx_input_names.append(f"past_value_cache_{layer_idx}")

        onnx_output_names = ["logits"]

        prefill_onnx_file = str(work_dir / "hmonnx" / f"{prefix}-llm-prefill.onnx")
        quant_graph_model = None
        if not Path(prefill_onnx_file).exists():
            logger.info("********************* start export llm prefill model *********************")
            quant_graph_model = convert_fx_model_to_quanted_model(
                wraped_llm_model, prefill_inputs, target_device, quant_config
            )
            onnx_input_names = BaseConverter.xh1_hmonnx_compatible(onnx_input_names)
            convert_quanted_model_to_hmonnx(
                quant_graph_model, prefill_inputs, prefill_onnx_file, onnx_input_names, onnx_output_names
            )
        else:
            logger.info(f"{prefill_onnx_file} exists, skip export prefill model.")

        meta_info["prefill_onnx"] = str(Path(prefill_onnx_file).relative_to(work_dir))

        prefill_golden_dir = str(work_dir / "golden" / f"{prefix}-llm-prefill")

        if not Path(prefill_golden_dir).exists():
            logger.info(f"start export vision model golden............")
            from xhquant.api import HMONNXGoldenInference

            prefill_model = HMONNXGoldenInference(prefill_onnx_file)
            prefill_model.save_golden = True
            prefill_model.exec_device = torch.device("cuda:0")

            Path(prefill_golden_dir).mkdir(exist_ok=True, parents=True)
            prefill_model.golden_dir = str(prefill_golden_dir)

            with torch.no_grad():
                input_args = []
                for arg in prefill_inputs:
                    if isinstance(arg, (list, tuple)):
                        input_args.extend(arg)
                    else:
                        input_args.append(arg)
                prefill_model.forward(*input_args)
            logger.info(f"Export prefill model golden to {prefill_golden_dir}")
        else:
            logger.info(f"{prefill_golden_dir} exists, skip export prefill model golden.")

        ## decode
        wrap_cfg.input_sequence_length = 1

        def update_cfg_fn(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(wrap_cfg)

        past_seq_length[0] = wrap_cfg.input_sequence_length
        decode_inputs = (
            inputs_embeds[:, :1, :],
            time_position_ids[:1],
            height_position_ids[:1],
            width_position_ids[:1],
            past_seq_length,
            torch.tensor([1], dtype=torch.int32).to(inputs_embeds.device),
            past_key_caches,
            past_value_caches,
        )

        decode_onnx_file = str(work_dir / "hmonnx" / f"{prefix}-llm-decode.onnx")
        if not Path(decode_onnx_file).exists():
            logger.info("********************* start export llm decode model *********************")
            if quant_graph_model is None:
                quant_graph_model = convert_fx_model_to_quanted_model(
                    wraped_llm_model, prefill_inputs, target_device, quant_config
                )
            quant_graph_model.apply(update_cfg_fn)
            onnx_input_names = BaseConverter.xh1_hmonnx_compatible(onnx_input_names)
            convert_quanted_model_to_hmonnx(
                quant_graph_model, decode_inputs, decode_onnx_file, onnx_input_names, onnx_output_names
            )
        else:
            logger.info(f"{decode_onnx_file} exists, skip export decode model.")

        meta_info["decode_onnx"] = str(Path(decode_onnx_file).relative_to(work_dir))

        decoder_golden_dir = str(work_dir / "golden" / f"{prefix}-llm-decode")
        if not Path(decoder_golden_dir).exists():
            logger.info(f"start export decode model golden............")
            from xhquant.api import HMONNXGoldenInference

            decoder_model = HMONNXGoldenInference(decode_onnx_file)
            decoder_model.save_golden = True
            decoder_model.exec_device = torch.device("cuda:0")
            Path(decoder_golden_dir).mkdir(exist_ok=True, parents=True)
            decoder_model.golden_dir = str(decoder_golden_dir)
            with torch.no_grad():
                input_args = []
                for arg in decode_inputs:
                    if isinstance(arg, (list, tuple)):
                        input_args.extend(arg)
                    else:
                        input_args.append(arg)
                decoder_model.forward(*input_args)
            logger.info(f"Export decode model golden to {decoder_golden_dir}")
        else:
            logger.info(f"{decoder_golden_dir} exists, skip export decode model golden.")

        meta_info["decoder_golden_dir"] = str(Path(decoder_golden_dir).relative_to(work_dir))
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    # @classmethod
    # def convert(cls, hf_model_path: str, config: Qwen2_5_VLConvertConfig, output_dir: str):
    #     quant_config = create_quant_config(config.quant_scheme)
    #     is_ssfp = is_ssfp_quant_config(quant_config)
    #     if is_ssfp:
    #         if not config.gptqmodel_cfg:
    #             assert config.quant_weight is not None and Path(config.quant_weight).exists()
    #     Qwen2_5_VLConverterXH2a(config)._convert(hf_model_path, output_dir)
