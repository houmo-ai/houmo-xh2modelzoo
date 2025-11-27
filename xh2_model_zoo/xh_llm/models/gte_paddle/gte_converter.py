import json
import shutil
import time
from pathlib import Path
from turtle import pos
from typing import Any, Dict, Optional
import torch.nn.functional as F
from matplotlib import axis
import torch
# from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, Qwen3ForCausalLM

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    HMONNXGoldenInference,
)
import os
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

class GteConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        # config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        from modelscope import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
        model = AutoModel.from_pretrained(hf_model_dir, trust_remote_code=True)
        native_model = model  # type: ignore
        native_model.eval()
        self.hf_model_path = hf_model_dir
        return native_model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        device = "cuda"
        native_model = self.load_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
        )

        # # 融合GPTQ权重
        # resume_from = self.config.quant_weight
        # if resume_from is not None:
        #     self.load_quant_weight(resume_from, native_model)
        # lm_head = native_model.lm_head
        # if not hasattr(lm_head, "quant_weight"):
        #     config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"
        work_dir = Path(output_dir)
        self.work_dir = output_dir

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
        # meta_info["quant_weight"] = resume_from

        hf_config_dir = Path(work_dir) / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        hf_config_files = [
            "config.json",
            "configuration.json",
            "sentence_bert_config.json",
            "modules.json",
            "special_tokens_map.json",
            # "generation_config.json",
            "tokenizer_config.json",
            # "vocab.json",
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

        word_embedding = native_model.embeddings.word_embeddings
        word_embedding_file = Path(work_dir) / "word_embeddings.pt"
        torch.save(word_embedding.state_dict(), str(word_embedding_file))
        meta_info["word_embedding_file"] = str(word_embedding_file.relative_to(work_dir))

        token_type_embedding = native_model.embeddings.token_type_embeddings
        token_type_embedding_file = Path(work_dir) / "token_type_embeddings.pt"
        torch.save(token_type_embedding.state_dict(), str(token_type_embedding_file))
        meta_info["token_type_embeddings"] = str(token_type_embedding_file.relative_to(work_dir))

        from ._model import register_wrap_modules as gte_register_wrap_modules  # noqa: F403, F401

        gte_register_wrap_modules()

        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                # batch_size=batch_size,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=False,
                num_logits_to_keep=0,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )

        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)

        # 设置kv cache
        # num_hidden_layers = wraped_qwen_model.model.config.num_hidden_layers
        # head_dim = wraped_qwen_model.model.layers[0].self_attn.head_dim
        # num_decoder_layers = num_hidden_layers
        # kv_cache_shape = [
        #     1,
        #     wraped_qwen_model.model.config.num_key_value_heads,
        #     wrap_cfg.max_sequence_length,
        #     head_dim,
        # ]
        # meta_info["kv_cache"] = dict(
        #     shape=kv_cache_shape,
        #     num_decoder_layers=num_decoder_layers,
        # )

        # past_key_caches = []
        # past_value_caches = []
        # for _ in range(num_decoder_layers):
        #     past_key_caches.append(torch.zeros(kv_cache_shape, dtype=torch.float16, device=device))
        #     past_value_caches.append(torch.zeros(kv_cache_shape, dtype=torch.float16, device=device))

        # 导出Prefill模型
        input_ids = []
        current_input_length = []
        # position_ids = []
        for _ in range(1):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long, device=device)
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
        
        word_embedding.cuda()
        token_type_embedding.cuda()
        wraped_qwen_model.cuda()

        # rope [4, 11, 1, 64]
        token_type_ids = torch.zeros_like(input_ids_t)
        word_embeds = word_embedding(input_ids_t) # [4, 11, 768]
        token_embeds = token_type_embedding(token_type_ids) # [4, 11, 768]
        attention_mask = torch.zeros( (1,1,1,input_sequence_length), device=device)
        position_ids = torch.arange(input_ids_t.shape[-1], device=device).unsqueeze(0)
        # past_seq_length_t = torch.tensor([0], dtype=torch.int32, device=device)
        # current_input_length_t = torch.tensor(current_input_length, dtype=torch.int32, device=device)

        inputs = (
            word_embeds,
            token_embeds,
            attention_mask,
            position_ids,
            # past_seq_length_t,
            # current_input_length_t,
            # position_ids,
            # past_key_caches,
            # past_value_caches,
        )
        input_names = [
            "word_embeds",
            "token_embeds",
            "attention_mask",
            "position_ids",
        ]
        # for layer_idx in range(num_decoder_layers):
        #     input_names.append(f"past_key_cache_{layer_idx}")
        # for layer_idx in range(num_decoder_layers):
        #     input_names.append(f"past_value_cache_{layer_idx}")
        output_names = ["logits"]

        if False:
            input_texts = [
                "what is the capital of China?",
                "how to implement quick sort in python?",
                "北京",
                "快排算法介绍"
            ]
            b_outputs = []
            for text in input_texts:
                data_dict = self.tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                input_ids_t = data_dict['input_ids'].cuda()
                attention_mask = torch.zeros_like( data_dict['attention_mask'] ).cuda()

                word_embeds = word_embedding(input_ids_t) # [4, 11, 768]
                
                token_type_ids = torch.zeros_like(input_ids_t)
                token_embeds = token_type_embedding(token_type_ids) # [4, 11, 768]
                # attention_mask = torch.zeros( (1,1,1,input_ids_t.shape[-1]), device=device)
                position_ids = torch.arange(input_ids_t.shape[-1], device=device).unsqueeze(0)

                inputs_list = (word_embeds, token_embeds, attention_mask, position_ids)

                outputs = wraped_qwen_model(*inputs_list)
                outputs = outputs[:, 0][:768]
                b_outputs.append(outputs)

            out_data  = torch.concat(b_outputs, dim=0)
            embeddings = F.normalize(out_data, p=2, dim=1)
            scores = (embeddings[:1] @ embeddings[1:].T)
            print(scores.tolist())

        # output = wraped_qwen_model(*inputs)

        prefix = f"{model_name}-{target_device}-{context_length//1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info(f"********************* start export prefill model *********************")
        if not os.path.exists(prefill_onnx_file):
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
        

        if True:
            session = HMONNXGoldenInference(prefill_onnx_file)
            session.to(device)
            session.save_golden = False

            input_texts = [
                "what is the capital of China?",
                "how to implement quick sort in python?",
                "北京",
                "快排算法介绍"
            ]
            b_outputs = []
            for text in input_texts:
                data_dict = self.tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                input_ids_t = data_dict['input_ids'].cuda()
                attention_mask = torch.zeros_like( data_dict['attention_mask'] ).cuda()

                word_embeds = word_embedding(input_ids_t) # [4, 11, 768]
                
                token_type_ids = torch.zeros_like(input_ids_t)
                token_embeds = token_type_embedding(token_type_ids) # [4, 11, 768]
                # attention_mask = torch.zeros( (1,1,1,input_ids_t.shape[-1]), device=device)
                position_ids = torch.arange(256, device=device).unsqueeze(0)

                inputs_list = (word_embeds, token_embeds, attention_mask, position_ids)

                outputs = session(*inputs_list)
                outputs = outputs[:, 0][:768]
                b_outputs.append(outputs)

            out_data  = torch.concat(b_outputs, dim=0)
            embeddings = F.normalize(out_data, p=2, dim=1)
            scores = (embeddings[:1] @ embeddings[1:].T)
            print(scores.tolist())        

        ## generate golden ==================================
        inp = [ inputs[0].half(), inputs[1].half(), inputs[2].half(), inputs[3].to(torch.int32)]
        session = HMONNXGoldenInference(prefill_onnx_file)
        session.to(device)
        session.save_golden = True
        session.golden_dir = "work_dirs/paddle_gte-XH2a-batch_1-2k-w4a8h1_sefp/hmonnx/golden"
        session.step = 0

        # GPTQ ========================================
        # session._set_session_env()
        # from xhquant.api import PrecisionMode
        # for name, node in session._session.named_modules():
        #     if hasattr(node, "op_type"):
        #         if node.op_type in ['Linear']:
        #             # print(node.precision_mode)
        #             node.precision_mode = PrecisionMode.GPTQ

        session(*inp)
        logger.info(f"Export decode model to {decode_onnx_file}")
        json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        GteConverterXH2a(config)._convert(hf_model_path, output_dir)
