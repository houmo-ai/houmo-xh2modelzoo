import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import yaml
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM
from transformers.modeling_gguf_pytorch_utils import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    TENSOR_PROCESSORS,
    TensorProcessor,
    _gguf_parse_value,
    get_gguf_hf_weights_map,
    read_field,
)
from xhquant.api import CacheTensor

from ....datasets.preprocess.mix_search_preprocess import ms_data_preprocess
from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from ..lora_layer import apply_lora_to_linear
from .qwen3_convert_config import Qwen3LegacyLoRAConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    PrecisionMode,
)
from transformers.utils import is_torch_available
from transformers.utils.import_utils import is_gguf_available


def load_gguf_checkpoint_for_lora(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `False`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    logger = get_root_logger()
    if is_gguf_available() and is_torch_available():
        from gguf import GGUFReader, dequantize
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    # NOTE: Some GGUF checkpoints may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currently this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        if "t5encoder" in architecture:
            parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
        updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all("output.weight" != tensor.name for tensor in reader.tensors) or architecture in exceptions
    )

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")
            parsed_parameters[gguf_key] = value

    # Gemma3 GGUF checkpoint only contains weights of text backbone
    if parsed_parameters["config"]["model_type"] == "gemma3":
        parsed_parameters["config"]["model_type"] = "gemma3_text"

    # retrieve config vocab_size from tokenizer
    # Please refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    if return_tensors:
        parsed_parameters["tensors"] = {}

        tensor_key_mapping = get_gguf_hf_weights_map(model_to_load)
        config = parsed_parameters.get("config", {})

        ProcessorClass = TENSOR_PROCESSORS.get(architecture, TensorProcessor)
        processor = ProcessorClass(config=config)

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            name = tensor.name
            atoms = name.split(".")
            lora_name = None
            if atoms[-1] in ["lora_a", "lora_b"]:
                name = ".".join(atoms[:-1])
                lora_name = atoms[-1]

            weights = dequantize(tensor.data, tensor.tensor_type)

            result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=parsed_parameters,
            )

            weights = result.weights
            name = result.name

            if name not in tensor_key_mapping:
                continue

            name = tensor_key_mapping[name]
            if lora_name is not None:
                name = f"{name}_{lora_name}"
            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters


class Qwen3LegacyLoRAConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3LegacyLoRAConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None
        assert Path(self.config.lora_checkpoint).exists(), f"LoRA checkpoint {self.config.lora_checkpoint} not exists"

    def load_hf_model(self, hf_model_dir: str, **kwargs) -> nn.Module:
        # config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        # assert not hasattr(config, "quantization_config")
        native_model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **kwargs)
        # assert not hasattr(native_model, "hf_quantizer")
        assert isinstance(native_model, Qwen3ForCausalLM), (
            f"The model is not Qwen2ForCausalLM, but {type(native_model)}"
        )
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

        native_model = self.get_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        )
        resume_from = self.config.quant_weight

        # 融合GPTQ权重
        resume_from = self.config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, native_model)

        # load lora权重
        lora_checkpoint = self.config.lora_checkpoint
        assert lora_checkpoint is not None, "lora_checkpoint must be provided when use_lora is True"
        lora_params = load_gguf_checkpoint_for_lora(lora_checkpoint, return_tensors=True, model_to_load=native_model)
        lora_scale = lora_params.get("adapter.lora.alpha", None)

        for name, param in lora_params["tensors"].items():
            module_path, _, buffer_name = name.rpartition(".")
            m = native_model.get_submodule(module_path)
            m.register_buffer(buffer_name, param)

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
        meta_info["lora"] = {
            "lora_scale": lora_scale,
            "lora_checkpoint": str(lora_checkpoint),
        }

        work_dir = Path(output_dir)
        hf_config_dir = Path(work_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
            "chat_template.jinja",
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

        from ._model import (
            register_wrap_modules as qwen3_register_wrap_modules,  # noqa: F403, F401
        )

        qwen3_register_wrap_modules(native_model)

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                # batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
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
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
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

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        # 添加LoRA变换
        import xhquant.frontend.builder as xh_builder

        class LoRATransform(object):
            def __init__(self, lora_scale: float):
                self.lora_scale = lora_scale

            def __call__(self, graph_module: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
                self.graph_module = graph_module
                if "inputs" in kwargs:
                    inputs = kwargs["inputs"]
                else:
                    inputs = args[0]

                apply_lora_to_linear(graph_module, inputs, self.lora_scale)
                return graph_module

        lora_trans = LoRATransform(lora_scale)
        xh_builder.USER_FX_TRANS.append(lora_trans)

        logger.info("********************* start export prefill model *********************")
        inputs = list(inputs)
        lora_mask = torch.tensor([1.0], dtype=torch.float16)
        inputs.append(lora_mask)  # 增加lora_mask输入
        quanted_model = convert_fx_model_to_quanted_model(
            wraped_qwen_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )
        if self.config.mix_search is not None:
            quanted_model = quanted_model.cuda()
            quanted_model.enable_fast_precision_mode()
            from xhquant.mix_precision.mix_precision import MixPrecisionSearch

            with open(self.config.mix_search, "r") as f:
                ms_cfg = yaml.safe_load(f)
            ms = MixPrecisionSearch(quanted_model, ms_cfg)
            tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
            # data preprocess
            gpu_loader, label_dataloader = ms_data_preprocess(
                wraped_qwen_model, tokenizer, past_key_caches, past_value_caches
            )

            precision_mode = PrecisionMode.FAST
            ms.search(gpu_loader, precision_mode, label_dataloader, wrap_cfg)
            quanted_model.enable_aligned_precision_mode()  # adjust precision mode
            quanted_model = quanted_model.to("cpu")
        # quant_info_onnx_file = str(Path(output_dir) / "quant_info.onnx")
        # quanted_model.dump_quant_info_to_onnx(quant_info_onnx_file)
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        input_names.append("lora_mask")

        convert_quanted_model_to_hmonnx(quanted_model, inputs, str(prefill_onnx_file), input_names, output_names)
        logger.info(f"Export Prefill model to {prefill_onnx_file}")

        xh_builder.USER_FX_TRANS.remove(lora_trans)

        logger.info("********************* start export decode model *********************")
        decode_inputs = (
            inputs_embeds[:, :1, :],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            # position_ids[:, :1],
            past_key_caches,
            past_value_caches,
            lora_mask,
        )

        # 更新与input_sequence_length相关的Module
        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_model,
            decode_inputs,
            str(decode_onnx_file),
            input_names,
            output_names,
        )

        logger.info(f"Export decode model to {decode_onnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3LegacyLoRAConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
            if not hasattr(hf_config, "quantization_config"):
                assert config.quant_weight is not None and Path(config.quant_weight).exists()
        Qwen3LegacyLoRAConverterXH2a(config)._convert(hf_model_path, output_dir)
