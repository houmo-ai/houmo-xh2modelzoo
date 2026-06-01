import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    Qwen3Model,
)

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen3_embedding_convert_config import Qwen3EmbeddingConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    CacheTensor,
    Config,
    DeviceType,
    ConfigDict,
    PrecisionMode,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    ptq_quantize,
)


class Qwen3EmbeddingConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3EmbeddingConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        native_model = AutoModel.from_pretrained(hf_model_dir, **kwargs)
        assert isinstance(native_model, Qwen3Model), (
            f"The model is not Qwen3Model, but {type(native_model)}"
        )
        native_model.eval()
        self.hf_model_path = hf_model_dir
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        device = "cuda"
        native_model = self.load_hf_model(
            hf_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        if config.quant_weight:
            state_dict = load_safetensors_file(config.quant_weight)
            if "post_norm_linear.weight" in state_dict:
                hidden_size = native_model.config.hidden_size
                native_model.post_norm_linear = torch.nn.Linear(
                    hidden_size, hidden_size, bias=False
                )
            self.load_quant_weight(config.quant_weight, native_model)

            has_quant_weight = any(
                k.endswith("quant_weight") for k in state_dict.keys()
            )
            logger.info(f"State dict contains quant_weight: {has_quant_weight}")
            no_quant_weight = []
            for name, module in native_model.named_modules():
                if isinstance(module, torch.nn.Linear) and not hasattr(
                    module, "quant_weight"
                ):
                    if name == "post_norm_linear":
                        continue
                    no_quant_weight.append(name)
            if no_quant_weight:
                logger.warning(
                    f"Linear layers without quant_weight: {no_quant_weight[:20]} "
                    f"(total={len(no_quant_weight)})"
                )
            if hasattr(native_model, "post_norm_linear"):
                logger.info("post_norm_linear is kept as non-quantized linear head")

        work_dir = Path(output_dir)
        self.work_dir = output_dir

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, (
            f"Only support convert to XH2a, but got {target_device}"
        )
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
        if config.quant_weight:
            meta_info["quant_weight"] = config.quant_weight

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

        token_embedding = native_model.get_input_embeddings()
        token_embedding_file = Path(work_dir) / "quant_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["quant_embedding_file"] = str(
            token_embedding_file.relative_to(work_dir)
        )
        # backward compatible key
        meta_info["token_embedding_file"] = meta_info["quant_embedding_file"]

        from ._llm_model_impl import (
            register_wrap_cls as qwen3_embedding_register_wrap_cls,  # noqa: F401
        )

        qwen3_embedding_register_wrap_cls(native_model)

        wrap_cfg = Config(
            dict(
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        num_decoder_layers = native_model.config.num_hidden_layers
        head_dim = (
            native_model.config.head_dim
            if hasattr(native_model.config, "head_dim")
            else native_model.config.hidden_size
            // native_model.config.num_attention_heads
        )
        kv_cache_shape = [
            1,
            native_model.config.num_key_value_heads,
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
            past_key_caches.append(
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            )
            past_value_caches.append(
                CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            )

        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)

        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        input_texts = [
            f"Instruct: {task}\nQuery:What is the capital of China?",
        ]
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, padding_side="left")
        batch = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=input_sequence_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).to(torch.int32)
        inputs_embeds = token_embedding(input_ids)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32, device=device)
        current_input_length_t = attention_mask.sum(dim=1).to(torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )
        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
        ]
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["hidden_states"]

        prefix = work_dir.name
        prefill_onnx_file = work_dir / "prefill" / f"{prefix}_prefill_with_act.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(
            "********************* start export prefill model *********************"
        )
        if not prefill_onnx_file.exists():
            quanted_model = convert_fx_model_to_quanted_model(
                wraped_qwen_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            # PTQ calibration for 0.6B (no Quarot/GPTQ weights)
            if getattr(config, "ptq", False):
                logger.info("*************** Start PTQ Quantize ***************")
                ptq_quantize(quanted_model, [inputs], PrecisionMode.ALIGNED, [device])
                logger.info("*************** Finished PTQ Quantize ***************")
            input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model, inputs, str(prefill_onnx_file), input_names, output_names
            )
            logger.info(f"Export Prefill model to {prefill_onnx_file}")

        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(
        cls, hf_model_path: str, config: Qwen3EmbeddingConvertConfig, output_dir: str
    ):
        Qwen3EmbeddingConverterXH2a(config)._convert(hf_model_path, output_dir)
