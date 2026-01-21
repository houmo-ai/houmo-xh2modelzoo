from pyparsing import str_type
import torch
import torch.nn as nn
import yaml
from torch import Tensor
import time
from functools import cached_property
from typing import List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
)
from xh_model_zoo.xh_llm.llm_converter import LLMConverter
from xh_model_zoo.xh_llm.llm_convert_config import LLMConvertConfig
from dataclasses import dataclass
from pathlib import Path
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xhquant.api import (
    CacheTensor,
    ConfigDict,
    HMONNXGoldenInference,
    PrecisionMode,
    QuantGraph,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
)
from transformers.utils.quantization_config import QuantizationMethod
from xhquant.utils import TimeProfiler
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.datasets.preprocess.mix_search_preprocess import (
    ms_data_preprocess,
)
import data
from dataclasses import dataclass, field
import shutil,json
from copy import deepcopy

@dataclass
class DeepseekOCRConverterConfig(LLMConvertConfig):
    export_mode: str = field(default_factory=str_type)
    input_resolution: int = field(default_factory=int)



class DeepseekOCRConverter(LLMConverter):
    MODEL_KEYS: List[str] = ["DeepseekOCRModel", "DeepseekOCR", "deepseek_ocr"]
    config: DeepseekOCRConverterConfig

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> nn.Module:
        hf_config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)

        # Load AutoRoundModel
        if (
            hasattr(hf_config, "quantization_config")
            and hf_config.quantization_config["quant_method"].lower() == "auto-round"
        ):
            # from auto_round import AutoRoundConfig ##must import for auto-round format
            from transformers import AutoModelForCausalLM

            native_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, torch_dtype="auto", revision="14dbc8", **kwargs
            )
        # Load GPTQModel
        elif (
            hasattr(hf_config, "quantization_config")
            and hf_config.quantization_config["quant_method"].lower() == "gptq"
            and hf_config.quantization_config.get("meta", None) is not None
            and ("gptqmodel" in str(hf_config.quantization_config["meta"].get("quantizer", None)))
        ):
            from gptqmodel import GPTQModel

            kwargs["device"] = kwargs.pop("device_map", "cpu")
            native_model = GPTQModel.from_quantized(hf_model_dir, **kwargs)
        # Native Load
        else:
            from xh_model_zoo.xh_llm.models.deepseek_ocr._model_deepseekv2 import DeepseekOCRForCausalLM
            native_model = DeepseekOCRForCausalLM.from_pretrained(hf_model_dir, 
                                _attn_implementation='eager', 
                                trust_remote_code=True, 
                                use_safetensors=True)

        if hasattr(native_model.config, "tie_word_embeddings") and native_model.config.tie_word_embeddings:  # type: ignore
            old_torchscript = native_model.config.torchscript  # type: ignore
            native_model.config.torchscript = True  # type: ignore
            native_model.tie_weights()  # type: ignore
            native_model.config.tie_word_embeddings = False  # type: ignore
            native_model.config.torchscript = old_torchscript  # type: ignore
        native_model.eval()
        if hasattr(native_model.config, "quantization_config"):
            native_model = self.dequantize_hf_model(native_model)
        if self.config.quant_weight is not None:
            self.load_quant_weight(self.config.quant_weight, native_model)
        self.token_embedding = native_model.get_input_embeddings()
        return native_model
    
    def _register_wrap_module(self, native_model: nn.Module = None):
        from ._model_deepseekv2 import register_wrap_modules as deepseek_ocr_llm_register_wrap_modules  #

        deepseek_ocr_llm_register_wrap_modules(native_model)

    def convert_llm_to_wrap_module(self, native_model: nn.Module) -> nn.Module:
        from ._model_deepseekv2 import register_wrap_modules as deepseek_ocr_llm_register_wrap_modules  #
        deepseek_ocr_llm_register_wrap_modules(native_model)
        wraped_model = wrap_llm_model(native_model, self.wrap_cfg)
        # del self.native_model
        return wraped_model
    
    def convert_llm_to_quant_module(self, wraped_model: nn.Module, input_args: dict) -> QuantGraph:
        config = self.config
        quant_config = self.quant_cfg

        quanted_model = convert_fx_model_to_quanted_model(
            wraped_model, input_args, config.quant_scheme.target_device, quant_config
        )
        # del self.wraped_model
        if self.config.mix_search:
            assert torch.cuda.is_available(), "CUDA is not available for MixSearch"
            quanted_model = quanted_model.cuda()
            quanted_model.enable_fast_precision_mode()
            from xhquant.mix_precision.mix_precision import MixPrecisionSearch

            with open(self.config.mix_search, "r") as f:
                ms_cfg = yaml.safe_load(f)
            ms = MixPrecisionSearch(quanted_model, ms_cfg)
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
            # data preprocess
            gpu_loader, label_dataloader = ms_data_preprocess(wraped_model, tokenizer, input_args[-2], input_args[-1])
            precision_mode = PrecisionMode.FAST
            ms.search(gpu_loader, precision_mode, label_dataloader, self.wrap_cfg)
            quanted_model.enable_aligned_precision_mode()  # adjust precision mode
            quanted_model = quanted_model.to("cpu")
        return quanted_model

    def export_llm_to_hmonnx(self, quant_model: QuantGraph, input_args: tuple, dst_file: str):
        input_names = [
            "input_embeds",
            "past_seq_length",
            "current_input_length",
            *[f"past_key_cache_{i}" for i in range(len(input_args[-1]))],
            *[f"past_value_cache_{i}" for i in range(len(input_args[-1]))],
        ]
        input_names = self.xh1_hmonnx_compatible(input_names)
        output_names = ["logits"]
        convert_quanted_model_to_hmonnx(quant_model, input_args, dst_file, input_names, output_names)

    def prepare_llm_inputs(self, data: dict = dict()):
        hf_config, config = self.hf_config, self.config
        num_decoder_layers = 1 if config.only_first_block else hf_config.num_hidden_layers
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        kv_cache_shape = [
            1,
            hf_config.num_key_value_heads,
            config.context_length,
            head_dim,
        ]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_decoder_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_decoder_layers)
        ]

        input_ids = data.get("input_ids", None)
        input_embeds = data.get("input_embeds", None)
        past_seq_length = data.get("past_seq_length", None)
        current_input_length = data.get("current_input_length", None)
        if input_embeds is not None:
            assert input_embeds.shape[0] == 1, "only support batch size 1"
            seq_length = input_embeds.shape[1]
            input_embeds = input_embeds.to(torch.device("cpu"))
            assert seq_length <= config.input_sequence_length, (
                f"Input sequence length is too long. max input sequence length is {config.input_sequence_length} but got {seq_length}"
            )
            if config.input_sequence_length > seq_length:
                padding_embedding = torch.zeros(1, config.input_sequence_length - seq_length, input_embeds.shape[-1])
                input_embeds = torch.cat([input_embeds, padding_embedding], dim=1)
        else:
            if input_ids is None:
                if data.get("text", None) is not None:
                    text = data["text"]
                    input_ids = self.tokenizer(text, return_tensors="pt").input_ids
                else:
                    input_ids = torch.randint(0, 1000, (1, config.input_sequence_length), dtype=torch.int32)
            assert input_ids.shape[0] == 1, "only support batch size 1"
            input_ids = input_ids.to(self.token_embedding.weight.device)
            seq_length = input_ids.shape[1]
            assert seq_length <= config.input_sequence_length, (
                f"Input sequence length is too long. max input sequence length is {config.input_sequence_length} but got {seq_length}"
            )
            if seq_length < config.input_sequence_length:
                padding_input_ids = torch.zeros(
                    (1, config.input_sequence_length - seq_length),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)
            input_embeds = self.token_embedding(input_ids)
        past_seq_length = (
            torch.tensor([past_seq_length], dtype=torch.int32, device=input_ids.device).reshape(-1)
            if past_seq_length is not None
            else torch.tensor([0], dtype=torch.int32, device=input_ids.device)
        )
        current_input_length = current_input_length or torch.tensor([seq_length], dtype=torch.int32)
        inputs = (
            input_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        return inputs

    @cached_property
    def native_model(self) -> nn.Module:
        with TimeProfiler("load_hf_model", self.logger) as tp:
            return self.load_hf_model(
                self.hf_model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu",
            )
    
    @cached_property
    def wraped_llm_model(self) -> nn.Module:
        native_model = self.native_model
        with TimeProfiler("convert_to_wrap_module", self.logger) as tp:
            return self.convert_llm_to_wrap_module(native_model)

    @cached_property
    def quanted_llm_model(self) -> QuantGraph:
        wraped_model = self.wraped_llm_model
        input_args = self.prepare_inputs()
        with TimeProfiler("convert_to_quant_module", self.logger) as tp:
            return self.convert_llm_to_quant_module(wraped_model, input_args)

    def export_llm_to_hmonnx(self, output_dir: str = None, generate_golden: bool = False):
        work_dir = Path(output_dir)
        model_name = Path(self.hf_model_path).name
        prefix = f"{model_name}-{self.config.quant_scheme.target_device}"
        meta_info = self.meta_info

        # 1. Export HMONNX
        quanted_model = self.quanted_llm_model
        input_args = self.prepare_inputs()

        # 1.1 Export Prefill HMONNX
        with TimeProfiler("Export Prefill HMONNX", self.logger) as tp:
            prefill_onnx_file = (
                work_dir
                / "hmonnx"
                / "prefill"
                / f"{Path(self.hf_model_path).name}-{self.config.quant_scheme.target_device}-{self.config.context_length // 1024}k-{self.config.quant_scheme.quant_type}_prefill.onnx"
            )
            prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
            self.export_to_hmonnx(quanted_model, input_args, str(prefill_onnx_file))
            meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        # 1.2 Export Decode HMONNX
        with TimeProfiler("Export Decode HMONNX", self.logger) as tp:
            decoder_wrap_cfg = self.wrap_cfg.copy()
            decoder_wrap_cfg.input_sequence_length = 1
            quanted_model.update_cfg(decoder_wrap_cfg)
            decode_input_args = (
                input_args[0][:, :1, :],
                input_args[2],
                torch.ones_like(input_args[2]),
                input_args[3],
                input_args[4],
            )
            decode_onnx_file = (
                work_dir
                / "hmonnx"
                / "decode"
                / f"{Path(self.hf_model_path).name}-{self.config.quant_scheme.target_device}-{self.config.context_length // 1024}k-{self.config.quant_scheme.quant_type}_decoder.onnx"
            )
            decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
            self.export_to_hmonnx(quanted_model, decode_input_args, str(decode_onnx_file))
            meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))

        # 2. Save meta.json and others
        with TimeProfiler("save_meta_and_others", self.logger) as tp:
            token_embedding_file = work_dir / "token_embedding.pt"
            torch.save(self.token_embedding.state_dict(), str(token_embedding_file))
            hf_config_dir = work_dir / "hf_config"
            hf_config_dir.mkdir(exist_ok=True, parents=True)
            hf_config_files = [
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "tokenizer.json",
            ]
            for cfg_file in hf_config_files:
                src_file = Path(self.hf_model_path) / cfg_file
                dst_file = Path(hf_config_dir) / cfg_file
                if src_file.exists():
                    shutil.copyfile(src_file, dst_file)
                else:
                    self.logger.warning(f"{src_file} not exists, skip copy")

            meta_info.update(
                {
                    "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "device": str(self.config.quant_scheme.target_device),
                    "model_name": Path(self.hf_model_path).name,
                    "hf_model_path": self.hf_model_path,
                    "hf_config": str(hf_config_dir.relative_to(work_dir)),
                    "token_embedding_file": str(token_embedding_file.relative_to(output_dir)),
                    "wrap_cfg": self.wrap_cfg.to_dict(),
                    "kv_cache": dict(
                        shape=[
                            1,
                            self.hf_config.num_key_value_heads,
                            self.config.context_length,
                            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
                        ],
                        num_decoder_layers=1 if self.config.only_first_block else self.hf_config.num_hidden_layers,
                    ),
                    "quant_scheme": self.config.quant_scheme.to_dict(),
                    "quant_weight": self.config.quant_weight,
                    "prefill_onnx": str(prefill_onnx_file.relative_to(work_dir)),
                    "decode_onnx": str(decode_onnx_file.relative_to(work_dir)),
                }
            )

            # 3. Generate Golden
            if generate_golden:
                # 3.1 Generate Prefill Golden
                with TimeProfiler("Generate Prefill Golden", self.logger) as tp:
                    prefill_golden_dir = work_dir / "golden" / f"{prefix}-llm-prefill"
                    prefill_golden_dir.mkdir(exist_ok=True, parents=True)
                    prefill_model = HMONNXGoldenInference(prefill_onnx_file)
                    prefill_model.save_golden = True
                    prefill_model.exec_device = torch.device("cuda:0")
                    prefill_model.golden_dir = str(prefill_golden_dir)
                    prefill_model.forward(*input_args)
                    meta_info["prefill_golden_dir"] = str(prefill_golden_dir.relative_to(work_dir))

                # 3.2 Generate Decode Golden
                with TimeProfiler("Generate Decode Golden", self.logger) as tp:
                    decode_golden_dir = work_dir / "golden" / f"{prefix}-llm-decode"
                    decode_golden_dir.mkdir(exist_ok=True, parents=True)
                    decode_model = HMONNXGoldenInference(decode_onnx_file)
                    decode_model.save_golden = True
                    decode_model.exec_device = torch.device("cuda:0")
                    decode_model.golden_dir = str(decode_golden_dir)
                    decode_model.forward(*input_args)
                    meta_info["decode_golden_dir"] = str(decode_golden_dir.relative_to(work_dir))

            json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)
            return meta_info

    def export(self, output_dir: str, generate_golden=False):
        self.llm_demo()
        # if self.config.export_mode == "Tiny":
        #     self.config.input_resolution=512
        # elif self.config.export_mode == "Small":
        #     self.config.input_resolution=640
        # elif self.config.export_mode == "Base":
        #     self.config.input_resolution=1024
        # elif self.config.export_mode == "Large":
        #     self.config.input_resolution=1280
        # elif self.config.export_mode == "Gundam":
        #     self.config.input_resolution=640
        #     self.export_vision_to_hmonnx(output_dir+"_vision_1", generate_golden)
        #     self.export_sam_to_hmonnx(output_dir+"_sam_1", generate_golden)
        #     self.config.input_resolution=1024
        #     self.export_vision_to_hmonnx(output_dir+"_vision_2", generate_golden)
        #     self.export_sam_to_hmonnx(output_dir+"_sam_2", generate_golden)
        #     self.export_llm_to_hmonnx(output_dir+"_llm", generate_golden)
        #     self.export_projector_to_hmonnx(output_dir+"_projector", generate_golden)
        #     return 


        # self.export_llm_to_hmonnx(output_dir+"_llm", generate_golden)
        # self.export_vision_to_hmonnx(output_dir+"_vision", generate_golden)
        # self.export_sam_to_hmonnx(output_dir+"_sam", generate_golden)
        # self.export_projector_to_hmonnx(output_dir+"_projector", generate_golden)

    def xh_llm_model(self, eval_type: EvalModelType):
        from xh_model_zoo.xh_llm.models.deepseek_ocr.deepseekv2_llm_model import DeepseekV2LegacyModel
        xh_model = DeepseekV2LegacyModel(
            hf_model=self.hf_model_path,
            frontend_type="TorchFX",
            wrap_cfg=self.wrap_cfg,
            quant_config=self.quant_cfg,
            export_cfg=ConfigDict(
                dict(
                    input_names=[
                        "inputs_embeds",
                        "past_seq_length",
                        "current_input_length",
                    ],
                    output_names=["logits", "hidden_state"],
                )
            ),
        )

        if eval_type == EvalModelType.WRAPED:
            xh_model._wrap_model = self.wraped_llm_model.eval().cuda()

            xh_model.change_eval_type(eval_type=eval_type)
        # xh_model.to(torch.float16)
        xh_model.to('cuda')

        return xh_model


    def llm_demo(self):
        prompt = "<image>\nFree OCR. "
        # prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        image_file = 'examples/llm/deepseek_ocr/data/img1.png'
        output_path = 'examples/llm/deepseek_ocr/data'
        native_model = self.native_model.eval().cuda()

        input_data: List[torch.Tensor] = []
        original_forward = native_model.model._forward

        def forward_with_hook(*args, **kwargs):
            input_data.append(kwargs['inputs_embeds'])
            raise Exception("Stop Forwarding to get the first llm input")
            return original_forward(*args, **kwargs)

        native_model.model._forward = forward_with_hook
        with torch.no_grad():
            try:
                native_model.infer(self.tokenizer, 
                            prompt=prompt, 
                            image_file=image_file, 
                            output_path = output_path, 
                            base_size = 512, 
                            image_size = 512, 
                            crop_mode=False, 
                            save_results = True, 
                            test_compress = True)
            except:
                print("搜集到了第一个llm输入数据")

        native_model.model._forward = original_forward
        native_model = deepcopy(native_model)


        # wrap_model = self.wraped_llm_model.eval().cuda()
        # del self.native_model

        xh_model = self.xh_llm_model(eval_type=EvalModelType.WRAPED)
        del self.native_model
        from xh_model_zoo.xh_llm.models.deepseek_ocr.deepseekv2_hf_compatible import DeepseekV2HFCompatible
        hf_compatible_model:DeepseekV2HFCompatible = DeepseekV2HFCompatible.to_hf_compatible(native_model,xh_model)
        # hf_compatible_model.cuda().half()
        hf_compatible_model.demo(inputs_embeds=input_data[0], tokenizer=self.tokenizer)


    @classmethod
    def convert_and_export(
        cls, hf_model_path: str, config: DeepseekOCRConverterConfig, output_dir: str, generate_golden=False
    ):
        config = DeepseekOCRConverterConfig.from_dict_or_other(config)
        return cls(hf_model_path, config).export(output_dir, generate_golden)

