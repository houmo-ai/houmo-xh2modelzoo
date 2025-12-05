import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2ForCausalLM

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .qwen2_convert_config import Qwen2ConvertConfig

from xhquant.api import (  # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)


class Qwen2ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen2ConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path = None
        self.output_dir = None

    def load_hf_model(self, hf_model_path, **kwargs) -> Qwen2ForCausalLM:
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        assert not hasattr(config, "quantization_config")
        native_model = AutoModelForCausalLM.from_pretrained(hf_model_path, **kwargs)
        assert not hasattr(native_model, "hf_quantizer")
        assert isinstance(
            native_model, Qwen2ForCausalLM
        ), f"The model is not Qwen2ForCausalLM, but {type(native_model)}"

        if native_model.config.tie_word_embeddings:
            old_torchscript = native_model.config.torchscript
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
            native_model.config.torchscript = old_torchscript

        self.hf_model_path = hf_model_path
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

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
        batch_size = config.batch_size
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
            import shutil

            shutil.copyfile(
                Path(hf_model_path) / cfg_file,
                Path(hf_config_dir) / cfg_file,
            )
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        token_embedding = native_model.model.get_input_embeddings()

        token_embedding_file = Path(work_dir) / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        from ._model import register_wrap_modules as qwen2_register_wrap_modules  # noqa: F403, F401

        qwen2_register_wrap_modules(native_model)

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
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
        head_dim = wraped_qwen_model.model.config.hidden_size // wraped_qwen_model.model.config.num_attention_heads
        num_hidden_layers = wraped_qwen_model.model.config.num_hidden_layers
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            wrap_cfg.batch_size,
            wraped_qwen_model.model.config.num_key_value_heads,
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

        if hasattr(config, "eval_ppl"):
            from xh2_model_zoo.utils.eval_ppl import evaluate  # isort:skip
            wraped_qwen_model.seqlen = input_sequence_length
            evaluate(wraped_qwen_model, model_path=hf_model_path, output_dir=output_dir)

        # 导出Prefill模型
        input_ids = []
        current_input_length = []
        position_ids = []
        for _ in range(batch_size):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = 0
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long)
            current_input_length.append(seq_length)

            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids: Tensor = torch.cat(input_ids, dim=0)
        position_ids: Tensor = torch.cat(position_ids, dim=0)

        inputs_embeds = token_embedding(input_ids)
        past_seq_length: Tensor = torch.tensor([0] * wrap_cfg.batch_size, dtype=torch.int32)
        current_input_length: Tensor = torch.tensor(current_input_length, dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            position_ids,
            past_key_caches,
            past_value_caches,
        )
        input_names = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "position_ids",
        ]
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits"]

        prefix = f"{model_name}-{target_device}-batch_{batch_size}-{context_length//1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / f"{prefix}_prefill.onnx"
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
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        logger.info(f"********************* start export decode model *********************")
        decode_inputs = (
            inputs_embeds[:, :1, :],
            past_seq_length,
            torch.ones_like(current_input_length),
            position_ids[:, :1],
            past_key_caches,
            past_value_caches,
        )

        # 更新与input_sequence_length相关的Module
        wrap_cfg.input_sequence_length = 1
        c

        decode_onnx_file = work_dir / "hmonnx" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(quanted_model, decode_inputs, str(decode_onnx_file), input_names, output_names)

        logger.info(f"Export decode model to {decode_onnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen2ConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        Qwen2ConverterXH2a(config)._convert(hf_model_path, output_dir)
