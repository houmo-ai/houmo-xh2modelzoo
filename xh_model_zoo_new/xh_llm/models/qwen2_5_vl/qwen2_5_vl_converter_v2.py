import tempfile, time, shutil, json
import torch.nn as nn, torch, onnx
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from PIL import Image
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

from transformers import AutoConfig, AutoTokenizer
from qwen_vl_utils import process_vision_info
from xhquant.api import (
    CacheTensor,
    get_root_logger,
    Config,
    ConfigDict,
    convert_onnx_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    QuantGraph,
    HMONNXGoldenInference,
)

from xhquant.api import ConfigDict
from xh_model_zoo_new.utils import TimeProfiler
from ...base_llm_converter import BaseLLMConverter, BaseLLMConverterConfig
from .processing_qwen2_5_vl import Qwen2_5_VLProcessor
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .data_preprocess import Qwen2_5_VLDataPreprocess


@dataclass
class VisionConfig:
    image_max_size_h: int = 364
    image_max_size_w: int = 644
    image_max_size_t: int = 2
    patch_size: int = 14
    temporal_patch_size: int = 2

    sample_image_path: str = field(default_factory=str)


@dataclass
class Qwen25VLConverterConfig(BaseLLMConverterConfig):
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    gptqmodel_cfg: str = field(default_factory=str)
    quant_weight: str = field(default_factory=str)
    max_pe_length: int = 32768

    def to_wrap_quant_cfg(self) -> Tuple[ConfigDict, ConfigDict]:
        """
        在 BaseHFLLMConverterConfig 的 wrap_cfg 基础上，补充视觉相关配置。
        """
        base_wrap_cfg, quant_cfg = super().to_wrap_quant_cfg()
        vision_cfg = ConfigDict(
            dict(
                max_size_h=self.vision_config.image_max_size_h,
                max_size_w=self.vision_config.image_max_size_w,
                max_size_t=self.vision_config.image_max_size_t,
                temporal_patch_size=self.vision_config.temporal_patch_size,
                patch_size=self.vision_config.patch_size,
            )
        )
        # Base 的 wrap_cfg 是 ConfigDict，直接挂一个 vision字段
        base_wrap_cfg.vision = vision_cfg
        # 额外记录 pe 长度
        base_wrap_cfg.max_pe_length = self.max_pe_length
        return base_wrap_cfg, quant_cfg


@dataclass
class VisionInputs:
    """
    供视觉子模型导出使用的输入。
    """

    hm_pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor  # (t,h,w)
    window_index: Optional[torch.Tensor] = None  # for rerank
    attention_bias: Optional[torch.Tensor] = None


@dataclass
class LLMInputs:
    """
    供 LLM wrap 模型 / 量化使用的输入。
    """

    inputs_embeds: torch.Tensor
    time_position_ids: torch.Tensor
    height_position_ids: torch.Tensor
    width_position_ids: torch.Tensor
    past_seq_length: torch.Tensor
    current_input_length: torch.Tensor
    image_embeds: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]
    past_key_caches: List[CacheTensor]
    past_value_caches: List[CacheTensor]


@dataclass
class Qwen25VLInputs:
    """
    Qwen2.5-VL 的统一输入包装，方便扩展。
    """

    vision: VisionInputs
    llm_prefill: LLMInputs


class Qwen25VLConverter(BaseLLMConverter):
    MODEL_KEYS: List[str] = ["Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VL", "qwen2_5_vl"]
    config: Qwen25VLConverterConfig
    processor: Qwen2_5_VLProcessor

    def __init__(self, hf_model_path: str, config: Qwen25VLConverterConfig):
        super().__init__(hf_model_path, config)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(hf_model_path)

        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_end_token_id = 151653
        self.vision_token_id = 151654
        self.eos_token_id = [151645, 151643]
        self.spatial_merge_size = 2
        self.window_size = 112

    # def _register_wrap_module(self, native_model: nn.Module = None):
    #     from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls
    #     from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls

    #     vision_register_wrap_cls(native_model)
    #     llm_register_wrap_cls(native_model)

    def _build_default_messages(self) -> List[Dict[str, Any]]:
        """
        默认使用一张示例图 + 简单文本，方便快速导出。
        """
        image_path = self.config.vision_config.sample_image_path
        if not image_path:
            # 保持与旧版逻辑一致的默认路径，方便兼容
            image_path = "data/images/qwen2_vl_demo.jpeg"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": "Describe this image.",
                    },
                ],
            }
        ]
        return messages

    def _run_processor(self, messages: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        统一走 Processor，将 messages 映射为 processor 输入张量。
        包含图像 resize 逻辑，确保输入尺寸与 image_max_size_w/h 一致。
        """
        processor = self.processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # 图像 resize 逻辑，确保与 vision_config.image_max_size_w/h 对齐
        image_max_size_h = self.config.vision_config.image_max_size_h
        image_max_size_w = self.config.vision_config.image_max_size_w

        if image_inputs and len(image_inputs) > 0 and isinstance(image_inputs[0], Image.Image):
            w, h = image_inputs[0].size
            scale = image_max_size_w / w, image_max_size_h / h
            scale = min(scale[0], scale[1])
            new_w = int(w * scale + 0.5)
            new_h = int(h * scale + 0.5)
            resized_img = image_inputs[0].resize((new_w, new_h), Image.Resampling.BICUBIC)
            pad_img = Image.new("RGB", (image_max_size_w, image_max_size_h), (122, 116, 104))
            pad_img.paste(resized_img, (0, 0))
            image_inputs[0] = pad_img
            # 更新 processor 配置
            processor.image_processor.max_pixels = max(
                image_max_size_w * image_max_size_h + 1, processor.image_processor.max_pixels
            )

        processor_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return processor_inputs, {"images": image_inputs}, {"videos": video_inputs}

    def _build_kv_caches(self) -> Tuple[List[CacheTensor], List[CacheTensor]]:
        """
        根据 HF config 构造 KV Cache，占位即可，主要用于导出。
        """
        hf_config, config = self.hf_config, self.config
        num_decoder_layers = hf_config.num_hidden_layers
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        kv_cache_shape = [1, hf_config.num_key_value_heads, config.context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_decoder_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_decoder_layers)
        ]

        self.meta_info["kv_cache"] = {
            "shape": kv_cache_shape,
            "num_decoder_layers": num_decoder_layers,
        }
        return past_key_caches, past_value_caches

    def vision_convert_to_wrap_module(self, vision_model, wrap_cfg: ConfigDict):
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls
        from xh_model_zoo_new.xh_llm.models.builder import wrap_llm_model

        vision_register_wrap_cls(vision_model)
        wraped_vision_model = wrap_llm_model(vision_model, wrap_cfg)
        return wraped_vision_model

    def prepare_inputs(self, native_model, data: Dict[str, Any] = dict()) -> Qwen25VLInputs:
        """
        构造 Qwen2.5-VL 所需的 vision + LLM 输入，方便后续导出复用。
        """
        messages: Optional[List[Dict[str, Any]]] = data.get("messages")
        if messages is None:
            messages = self._build_default_messages()

        processor_inputs, _, _ = self._run_processor(messages)

        # 构造 vision 输入（对齐 v1：使用 processor 产出的 pixel_values）
        # hm_pixel_values: [b, 3, h, w]（不包含时间维，视觉导出阶段再 repeat 到 t）
        max_h = self.config.vision_config.image_max_size_h
        max_w = self.config.vision_config.image_max_size_w
        hm_pixel_values = processor_inputs.get("hm_pixel_values")[0]
        if isinstance(hm_pixel_values, torch.Tensor) and hm_pixel_values.ndim == 3:
            hm_pixel_values = hm_pixel_values.unsqueeze(0)
        # 确保尺寸与配置一致（经过 _run_processor 的 resize 后应该已经对齐）
        assert isinstance(hm_pixel_values, torch.Tensor), "pixel_values should be a tensor"
        assert (
            hm_pixel_values.shape[2] == max_h and hm_pixel_values.shape[3] == max_w
        ), f"pixel_values shape {hm_pixel_values.shape} doesn't match config ({max_h}, {max_w})"
        image_grid_thw = processor_inputs["image_grid_thw"]
        hm_pixel_values = hm_pixel_values.unsqueeze(2).repeat(1, 1, self.config.vision_config.image_max_size_t, 1, 1)
        vision_inputs = VisionInputs(
            hm_pixel_values=hm_pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # 通过原生视觉模型先跑一遍，得到 image_embeds，供 LLM 使用
        vision_model = native_model.visual
        execution_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        vision_model.to(execution_device)
        with torch.no_grad():
            image_embeds = vision_model.forward_ori(
                processor_inputs["pixel_values"].to(execution_device),
                grid_thw=processor_inputs["image_grid_thw"].to(execution_device),
            ).cpu()
        vision_model.cpu()

        # 构造 LLM 输入（包含 3D RoPE 位置编码：time/height/width）
        input_ids = processor_inputs["input_ids"]
        input_seq_len = input_ids.shape[-1]
        input_sequence_length = self.config.input_sequence_length
        steps = (input_seq_len + input_sequence_length - 1) // input_sequence_length
        data_prefill = {
            "input_ids": input_ids,
            "image_embeds": image_embeds.to(torch.float16),
            "past_seq_length": 0,
            "image_grid_thw": processor_inputs["image_grid_thw"],
        }
        data_preprocess = Qwen2_5_VLDataPreprocess(self.token_embedding, input_sequence_length * steps)
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
        ) = data_preprocess(data_prefill)

        inputs_embeds = inputs_embeds[:, :input_sequence_length, :]
        time_position_ids = time_position_ids[:input_sequence_length]
        height_position_ids = height_position_ids[:input_sequence_length]
        width_position_ids = width_position_ids[:input_sequence_length]
        current_input_length = torch.tensor([input_sequence_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches, past_value_caches = self._build_kv_caches()

        llm_inputs = LLMInputs(
            inputs_embeds=inputs_embeds,
            time_position_ids=time_position_ids,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            image_embeds=image_embeds.to(torch.float16),
            image_grid_thw=image_grid_thw,
            past_key_caches=past_key_caches,
            past_value_caches=past_value_caches,
        )

        return Qwen25VLInputs(vision=vision_inputs, llm_prefill=llm_inputs)

    def vision_convert_to_quant_module(self, wraped_vision_model, hm_pixel_values: torch.Tensor):
        self.logger.info(f"start export vision model to onnx format............")
        hm_pixel_values = hm_pixel_values.to(dtype=wraped_vision_model.dtype, device=wraped_vision_model.device)
        window_index = wraped_vision_model.window_index
        attention_bias = wraped_vision_model.attention_bias
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_onnx_file = str(Path(tmp_dir) / "vision.onnx")
            torch.onnx.export(
                wraped_vision_model.cpu().float(),
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
        self.logger.info(f"simplify onnx model............")
        onnx_model, check = simplify_large_onnx(onnx_model)
        vision_args = [hm_pixel_values.float().cpu(), window_index.cpu()]

        if self.config.vision_config.image_max_size_w % 112 and self.config.vision_config.image_max_size_h % 112:
            vision_args.append(attention_bias.cpu())

        quant_vision_model = convert_onnx_to_quanted_model(
            onnx_model, vision_args, self.config.quant_scheme.target_device
        )
        return quant_vision_model

    def export_to_hmonnx(self, quant_model, prefill_args, dst_file):
        num_decoder_layers = 1 if self.config.only_first_block else self.hf_config.num_hidden_layers
        input_names = [
            "inputs_embeds",
            "time_position_ids",
            "height_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
            *[f"past_key_cache_{i}" for i in range(num_decoder_layers)],
            *[f"past_value_cache_{i}" for i in range(num_decoder_layers)],
        ]
        output_names = ["logits"]
        convert_quanted_model_to_hmonnx(quant_model, prefill_args, dst_file, input_names, output_names)

    def load_hf_model(self, hf_model_path, **kwargs):
        native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(hf_model_path, **kwargs)
        if native_model.config.tie_word_embeddings:
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
        native_model.eval()
        if hasattr(native_model.config, "quantization_config"):
            native_model = self.dequantize_hf_model(native_model)
        if self.config.quant_weight is not None:
            self.load_quant_weight(self.config.quant_weight, native_model)
        self.token_embedding = native_model.get_input_embeddings()
        return native_model

    @cached_property
    def qwen25_vl_inputs(self) -> Qwen25VLInputs:
        native_model = self.native_model
        with TimeProfiler("Prepare Qwen25VLInputs", self.logger) as tp:
            return self.prepare_inputs(native_model)

    @cached_property
    def native_model(self) -> nn.Module:
        with TimeProfiler("Load Native Model", self.logger) as tp:
            native_model = self.load_hf_model(
                self.hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
            )
            return native_model

    @cached_property
    def wraped_model(self) -> Tuple[nn.Module, nn.Module]:
        native_model = self.native_model
        with TimeProfiler("Convert to Wrap Model", self.logger) as tp:
            wraped_vision_model = self.vision_convert_to_wrap_module(native_model.visual, self.wrap_cfg.vision)
            wraped_model = self.convert_to_wrap_module(native_model)
        return wraped_vision_model, wraped_model

    @cached_property
    def quanted_model(self) -> Tuple[QuantGraph, QuantGraph]:
        wraped_vision_model, wraped_model = self.wraped_model
        qwen2_5_vl_inputs = self.qwen25_vl_inputs
        with TimeProfiler("Convert to Quant Model", self.logger) as tp:
            quant_vision_model = self.vision_convert_to_quant_module(
                wraped_vision_model, qwen2_5_vl_inputs.vision.hm_pixel_values
            )

            prefill_inputs = qwen2_5_vl_inputs.llm_prefill
            prefill_args = (
                prefill_inputs.inputs_embeds,
                prefill_inputs.time_position_ids,
                prefill_inputs.height_position_ids,
                prefill_inputs.width_position_ids,
                prefill_inputs.past_seq_length,
                prefill_inputs.current_input_length,
                # torch.tensor([input_sequence_length], dtype=torch.int32),
                prefill_inputs.past_key_caches,
                prefill_inputs.past_value_caches,
            )
            quant_model = self.convert_to_quant_module(wraped_model, prefill_args)
        return quant_vision_model, quant_model

    @cached_property
    def wraped_llm_model(self) -> nn.Module:
        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls
        qwen25_vl_inputs = self.qwen25_vl_inputs
        native_model = self.native_model
        with TimeProfiler("Wrap LLM Model", self.logger) as tp:
            wraped_llm_model = self.convert_to_wrap_module(native_model)
        return wraped_llm_model

    @cached_property
    def wraped_vision_model(self) -> nn.Module:
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls
        native_model = self.native_model
        with TimeProfiler("Wrap Vision Model", self.logger) as tp:
            wraped_vision_model = self.vision_convert_to_wrap_module(native_model.visual, self.wrap_cfg.vision)
        return wraped_vision_model

    @cached_property
    def quanted_vision_model(self) -> QuantGraph:
        wraped_vision_model = self.wraped_vision_model
        qwen2_5_vl_inputs  = self.qwen25_vl_inputs
        with TimeProfiler("Quant Vision Model", self.logger) as tp:
            quant_vision_model = self.vision_convert_to_quant_module(
                wraped_vision_model, qwen2_5_vl_inputs.vision.hm_pixel_values
            )
        return quant_vision_model

    @cached_property
    def quanted_llm_model(self) -> QuantGraph:
        wraped_llm_model = self.wraped_llm_model
        with TimeProfiler("Quant LLM Model", self.logger) as tp:
            prefill_inputs = self.qwen25_vl_inputs.llm_prefill
            prefill_args = (
                prefill_inputs.inputs_embeds,
                prefill_inputs.time_position_ids,
                prefill_inputs.height_position_ids,
                prefill_inputs.width_position_ids,
                prefill_inputs.past_seq_length,
                prefill_inputs.current_input_length,
                # torch.tensor([input_sequence_length], dtype=torch.int32),
                prefill_inputs.past_key_caches,
                prefill_inputs.past_value_caches,
            )
            quant_llm_model = self.convert_to_quant_module(wraped_llm_model, prefill_args)
        return quant_llm_model

    @torch.no_grad()
    def export(self, output_dir: str, generate_golden: bool = False) -> Dict[str, Any]:
        """Export Qwen2.5-VL model to HMONNX

        Args:
            output_dir (str): The directory to save the exported model
            generate_golden (bool, optional): Whether to generate golden. Defaults to False.

        Returns:
            Dict[str, Any]: The meta information of the exported model
        """
        work_dir = Path(output_dir)
        self.work_dir = output_dir
        model_name = Path(self.hf_model_path).name
        prefix = f"{model_name}-{self.config.quant_scheme.target_device}"
        meta_info = self.meta_info  # For save meta info

        meta_info.update(
            dict(
                device=self.config.quant_scheme.target_device,
                model_name=model_name,
                quant_scheme=self.config.quant_scheme.to_dict(),
                wrap_cfg=self.wrap_cfg.to_dict(),
            )
        )

        # 1. Export HMONNX
        qwen2_5_vl_inputs = self.qwen25_vl_inputs
        # wraped_vision_model, wraped_model = self.wraped_model
        # quanted_vision_model, quanted_model = self.quanted_model
        wraped_vision_model, wraped_model = self.wraped_vision_model, self.wraped_llm_model
        quanted_vision_model, quanted_model = self.quanted_vision_model, self.quanted_llm_model

        # 1.1 Export Vision Model Onnx
        with TimeProfiler("Export Vision Model HMONNX", self.logger) as tp:
            vision_onnx_str = str(work_dir / "hmonnx" / f"{prefix}-vision.onnx")
            convert_quanted_model_to_hmonnx(
                quanted_vision_model,
                (qwen2_5_vl_inputs.vision.hm_pixel_values.half(), wraped_vision_model.window_index),
                vision_onnx_str,
            )
            meta_info["vision_onnx"] = str(Path(vision_onnx_str).relative_to(work_dir))

            # 1.2 Export LLM Model Prefill Onnx
        with TimeProfiler("Export LLM Model Prefill HMONNX", self.logger) as tp:
            prefill_onnx_str = str(work_dir / "hmonnx" / f"{prefix}-llm-prefill.onnx")
            input_sequence_length = self.config.input_sequence_length
            prefill_inputs = qwen2_5_vl_inputs.llm_prefill
            prefill_args = (
                prefill_inputs.inputs_embeds[:, :input_sequence_length, :],
                prefill_inputs.time_position_ids[:input_sequence_length],
                prefill_inputs.height_position_ids[:input_sequence_length],
                prefill_inputs.width_position_ids[:input_sequence_length],
                prefill_inputs.past_seq_length,
                torch.tensor([input_sequence_length], dtype=torch.int32),
                prefill_inputs.past_key_caches,
                prefill_inputs.past_value_caches,
            )
            self.export_to_hmonnx(quanted_model, prefill_args, prefill_onnx_str)
            meta_info["prefill_onnx"] = str(Path(prefill_onnx_str).relative_to(work_dir))

            # 1.3 Export LLM Model Decode Onnx
        with TimeProfiler("Export LLM Model Decode HMONNX", self.logger) as tp:
            self.wrap_cfg.input_sequence_length = 1
            quanted_model.update_cfg(self.wrap_cfg)
            decode_onnx_str = str(work_dir / "hmonnx" / f"{prefix}-llm-decode.onnx")
            decoder_args = (
                prefill_inputs.inputs_embeds[:, :1, :],
                prefill_inputs.time_position_ids[:1],
                prefill_inputs.height_position_ids[:1],
                prefill_inputs.width_position_ids[:1],
                torch.tensor([input_sequence_length], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                prefill_inputs.past_key_caches,
                prefill_inputs.past_value_caches,
            )
            self.export_to_hmonnx(quanted_model, decoder_args, decode_onnx_str)
            meta_info["decode_onnx"] = str(Path(decode_onnx_str).relative_to(work_dir))

        # 2. Save Other Files and Meta Info
        with TimeProfiler("Save Other Files and MetaInfo", self.logger) as tp:
            # 2.1 Save HF files
            hf_config_dir = work_dir / "hf_config"
            hf_config_dir.mkdir(exist_ok=True, parents=True)
            hf_config_files = [
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "vocab.json",
                "tokenizer.json",
            ]
            for cfg_file in hf_config_files:
                source_path = Path(self.hf_model_path) / cfg_file
                if source_path.exists():
                    shutil.copyfile(source_path, Path(hf_config_dir) / cfg_file)
                else:
                    print(f"Warning: {cfg_file} not found in {self.hf_model_path}")
            meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

            chat_template_jinja_path = Path(self.hf_model_path) / "chat_template.jinja"
            chat_template_json_path = Path(self.hf_model_path) / "chat_template.json"
            if chat_template_jinja_path.exists():
                shutil.copyfile(chat_template_jinja_path, Path(hf_config_dir) / "chat_template.jinja")
            elif chat_template_json_path.exists():
                shutil.copyfile(chat_template_json_path, Path(hf_config_dir) / "chat_template.json")
            else:
                print(f"Warning: Neither chat_template.jinja nor chat_template.json found in {self.hf_model_path}")
            meta_info["chat_template"] = str(
                Path(hf_config_dir) / "chat_template.jinja"
                if chat_template_jinja_path.exists()
                else str(Path(hf_config_dir) / "chat_template.json")
            )

            # 2.2 Save Token Embedding
            token_embedding = self.native_model.model.get_input_embeddings()
            token_embedding_file = Path(work_dir) / "token_embedding.pt"
            torch.save(token_embedding.state_dict(), str(token_embedding_file))
            meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        with open(work_dir / "meta.json", "w") as f:
            json.dump(meta_info, f, indent=4)

        if generate_golden:
            assert torch.cuda.is_available(), "CUDA is not available"
            with TimeProfiler("Generate Vision Golden", self.logger) as tp:
                vision_golden_dir = work_dir / "golden" / f"{prefix}-vision"
                vision_golden_dir.mkdir(exist_ok=True, parents=True)
                vision_model = HMONNXGoldenInference(vision_onnx_str)
                vision_model.save_golden = True
                vision_model.golden_dir = vision_golden_dir
                vision_model.exec_device = torch.device("cuda")

                input_args = [
                    qwen2_5_vl_inputs.vision.hm_pixel_values.to(torch.device("cuda")).half(),
                    wraped_vision_model.window_index.to(torch.device("cuda")).int(),
                ]
                if (
                    self.config.vision_config.image_max_size_w % 112
                    and self.config.vision_config.image_max_size_h % 112
                ):
                    input_args.append(wraped_vision_model.attention_bias.cuda().half())
                vision_model.forward(*input_args)
                meta_info["vision_golden_dir"] = str(vision_golden_dir.relative_to(work_dir))

            with TimeProfiler("Generate LLM Prefill Golden", self.logger):
                prefill_golden_dir = work_dir / "golden" / f"{prefix}-llm-prefill"
                prefill_golden_dir.mkdir(exist_ok=True, parents=True)

                prefill_model = HMONNXGoldenInference(prefill_onnx_str)
                prefill_model.save_golden = True
                prefill_model.exec_device = torch.device("cuda:0")
                prefill_model.golden_dir = str(prefill_golden_dir)

                input_args = []
                for arg in prefill_args:
                    if isinstance(arg, (list, tuple)):
                        input_args.extend(arg)
                    else:
                        input_args.append(arg)
                prefill_model.forward(*input_args)
                meta_info["prefill_golden_dir"] = str(prefill_golden_dir.relative_to(work_dir))

            with TimeProfiler("Generate LLM Decode Golden", self.logger):
                decode_golden_dir = work_dir / "golden" / f"{prefix}-llm-decode"
                decode_golden_dir.mkdir(exist_ok=True, parents=True)

                decode_model = HMONNXGoldenInference(decode_onnx_str)
                decode_model.save_golden = True
                decode_model.exec_device = torch.device("cuda:0")
                decode_model.golden_dir = str(decode_golden_dir)

                input_args = []
                for arg in decoder_args:
                    if isinstance(arg, (list, tuple)):
                        input_args.extend(arg)
                    else:
                        input_args.append(arg)
                decode_model.forward(*input_args)
                meta_info["decode_golden_dir"] = str(decode_golden_dir.relative_to(work_dir))

        return meta_info

    @classmethod
    def convert_and_export(
        cls, hf_model_path: str, config: Qwen25VLConverterConfig, output_dir: str, generate_golden: bool = False
    ):
        return cls(hf_model_path, config).export(output_dir, generate_golden)
