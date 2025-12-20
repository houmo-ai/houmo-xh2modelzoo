
import json
import os
import tempfile
import time
from pathlib import Path
import token
from typing import Any, Dict, Optional
import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from shapely import length
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoModel, BertModel
from xhquant.api import convert_onnx_to_hmonnx, get_root_logger, xhquant_init
from ..base_converter import HFTransfromersConverter
from .qwen2_ste_convert_config import SteQwen2ConvertConfig
from xhquant.api import (  # type: ignore # isort:skip
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    HMONNXGoldenInference,
    Config,
    CacheTensor,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
)
from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
class Net(nn.Module):
    def __init__(self, ste_model: BertModel, config=None):
        super().__init__()
        self.config = config
        self.ste_model = ste_model
        self.prompts = ste_model.prompts
    def preprocess(self, sentences, prompt_name=None):
        if prompt_name is not None:
            prompt = self.prompts[prompt_name]
        else:
            prompt = None
        if prompt is not None and len(prompt) > 0:
            extra_features = {}
            sentences = [prompt + sentence for sentence in sentences]
            length = self.ste_model._get_prompt_length(prompt)
            extra_features["prompt_length"] = length
        all_embeddings = []
        length_sorted_idx = np.argsort([-self.ste_model._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        feats = []
        for start_index in trange(0, len(sentences), self.config.batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index : start_index + self.config.batch_size]
            features = self.ste_model.tokenize(sentences_batch)
            feats.append(features)
        return feats
    def sen_pool(self, attention_mask, token_embeddings, attention_mask_flip=None):
        output_vectors = []
        bs, seq_len, hidden_dim = token_embeddings.shape
        # values, indices = attention_mask.flip(1).max(1)
        # values, indices = attention_mask_flip.max(1) # 1 0 
        # indices = torch.where(values == 0, seq_len - 1, indices) # 0
        gather_indices = seq_len - 1 # 26
        # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
        gather_indices = gather_indices.unsqueeze(1)
        # assert gather_indices.shape == (bs, 1, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        output_vectors.append(embedding)
        output_vector = torch.cat(output_vectors, 1)
        return output_vector
    def forward(self, 
            input_ids, 
            past_seq_length,
            current_input_length,
            position_ids,
            past_key_caches,
            past_value_caches,     
        ):
        trans_out = self.ste_model[0].auto_model(
            input_ids, 
            past_seq_length,
            current_input_length,
            position_ids,
            past_key_caches,
            past_value_caches,            
        )[0]
        pool_out = trans_out[:, -1, :]# self.sen_pool(attention_mask, trans_out, attention_mask_flip)
        out = F.normalize(pool_out, p=2, dim=1)
        return out
class SteQwen2ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a
    def __init__(self, config: SteQwen2ConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None
    def load_hf_model(self, hf_model_dir: str, **kwargs) -> Any:
        # model: BertModel = AutoModel.from_pretrained(hf_model_dir, device_map="cpu", torch_dtype=torch.float16)
        model = SentenceTransformer(hf_model_dir, trust_remote_code=True)
        model.eval()  # type:ignore
        return model
    def _convert(self, hf_model_path: str, output_dir: str):
        model_name = Path(hf_model_path).name
        target_device = self.config.quant_scheme.target_device
        work_dir = Path(output_dir)
        logger = get_root_logger()
        config = self.config
        native_model = self.load_hf_model(
            hf_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
        )
        device = "cuda"
        wraped_model = Net(native_model, self.config)
        wraped_model.to(device)
        bz = config.batch_size
        context_length = config.context_length
        # input_ids = torch.randint(0, 100, (bz, context_length)).to(device)
        # attention_mask = torch.ones((bz, context_length), dtype=torch.float16).to(device)
        input_sequence_length = config.input_sequence_length
        # queries = [
        #     "how much protein should a female eat",
        #     "summit define",
        # ]
        # inputs = wraped_model.preprocess(queries)
        from ._model import register_wrap_modules as qwen2_register_wrap_modules  # noqa: F403, F401
        qwen2_register_wrap_modules(native_model)
        # 改写原模型, 以适合torch.fx导出
        wrap_cfg = Config(
            dict(
                batch_size=bz,
                max_sequence_length=context_length,  # 最大上下文长度
                input_sequence_length=input_sequence_length,  # prefill时输入的序列长度
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(
                    cache_axis=2,
                ),
            )
        )
        # meta_info["wrap_cfg"] = wrap_cfg.to_dict()
        wraped_qwen_model = wrap_llm_model(native_model, wrap_cfg)
        token_embedding = native_model[0].auto_model.embed_tokens
        # wraped_model.ste_model[0].auto_model = wraped_qwen_model
        # 设置kv cache
        head_dim = wraped_qwen_model[0].auto_model.config.hidden_size // wraped_qwen_model[0].auto_model.config.num_attention_heads
        num_hidden_layers = wraped_qwen_model[0].auto_model.config.num_hidden_layers
        num_decoder_layers = num_hidden_layers
        kv_cache_shape = [
            wrap_cfg.batch_size,
            wraped_qwen_model[0].auto_model.config.num_key_value_heads,
            wrap_cfg.max_sequence_length,
            head_dim,
        ]        
        past_key_caches = []
        past_value_caches = []
        for _ in range(num_decoder_layers):
            past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
            past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
        input_ids = []
        current_input_length = []
        position_ids = []
        for _ in range(bz):
            input_id = torch.randint(0, 1000, (input_sequence_length,), dtype=torch.long, device="cuda")
            seq_length = input_id.shape[0]
            past_seq_length = 0
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long, device="cuda")
            current_input_length.append(seq_length)
            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)
        input_ids: Tensor = torch.cat(input_ids, dim=0)
        position_ids: Tensor = torch.cat(position_ids, dim=0)
        inputs_embeds = token_embedding(input_ids) # 
        past_seq_length: Tensor = torch.tensor([0] * wrap_cfg.batch_size, dtype=torch.int32, device="cuda")
        current_input_length: Tensor = torch.tensor(current_input_length, dtype=torch.int32, device="cuda")
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
        prefix = f"{model_name}-{target_device}-batch_{bz}-{context_length//1024}k-{quant_type}"
        prefill_onnx_file = work_dir / "hmonnx" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))
        logger.info(f"********************* start export prefill model *********************")
        # quant_info_onnx_file = str(Path(output_dir) / "quant_info.onnx")
        # quanted_model.dump_quant_info_to_onnx(quant_info_onnx_file)
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        if not Path(prefill_onnx_file).exists():
            quanted_model = convert_fx_model_to_quanted_model(
                wraped_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            convert_quanted_model_to_hmonnx(
                quanted_model, inputs, str(prefill_onnx_file), input_names, output_names
            )        
        prefill_golden_dir = work_dir / "hmonnx/golden/prefill"
        if not Path(prefill_golden_dir).exists():
            hm_input = [
                inputs_embeds.half(),
                past_seq_length.to(torch.int32),
                current_input_length.to(torch.int32),
                position_ids.to(torch.int32),
            ]
            for k_data in past_key_caches:
                hm_input.append(k_data.half())
            for v_data in past_value_caches:
                hm_input.append(v_data.half())
            session = HMONNXGoldenInference(prefill_onnx_file)
            session.to(device)
            session.save_golden = True
            session.golden_dir = work_dir / "hmonnx/golden/prefill"
            session.step = 0
            session(*hm_input)
        logger.info(f"********************* start export decoder model *********************")
        decoder = f"{model_name}-{target_device}-batch_{bz}-{context_length//1024}k-{quant_type}"
        decoder_onnx_file = work_dir / "hmonnx" / f"{decoder}_decoder.onnx"
        decoder_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        inputs_embeds = token_embedding( input_ids[:,0:1] )
        # inputs[0] = inputs_embeds
        # inputs[2] = torch.tensor([1], dtype=torch.int32, device="cuda")
        # inputs[3] = inputs[3][:,0:1]
        decoder_inputs = (
            inputs_embeds,
            past_seq_length,
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            inputs[3][:,0:1],
            past_key_caches,
            past_value_caches,            
        )
        input_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        if not Path(decoder_onnx_file).exists():
            wrap_cfg.input_sequence_length = 1
            quanted_model.update_cfg(wrap_cfg)
            # quanted_model = convert_fx_model_to_quanted_model(
            #     wraped_model,
            #     decoder_inputs,
            #     target_device,
            #     quant_config=quant_config,
            # )
            convert_quanted_model_to_hmonnx(
                quanted_model, decoder_inputs, str(decoder_onnx_file), input_names, output_names
            )       
            
        decoder_golden_dir = work_dir / "hmonnx/golden/decoder"
        if not Path(decoder_golden_dir).exists():
            hm_input = [
                inputs_embeds.half(),
                past_seq_length.to(torch.int32),
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                inputs[3][:,0:1].to(torch.int32),
            ]
            for k_data in past_key_caches:
                hm_input.append(k_data.half())
            for v_data in past_value_caches:
                hm_input.append(v_data.half())
            
            session = HMONNXGoldenInference(decoder_onnx_file)
            session.to(device)
            session.save_golden = True
            session.golden_dir = decoder_golden_dir
            session.step = 0
            session(*hm_input)
        # logger.info(f"Export model to {hmonnx_file}")
        # json.dump(meta_info, open(work_dir / "meta.json", "w"), indent=4)
    @classmethod
    def convert(cls, hf_model_path: str, config: SteQwen2ConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        is_ssfp = is_ssfp_quant_config(quant_config)
        assert not is_ssfp, f"不支持SSFP量化"
        SteQwen2ConverterXH2a(config)._convert(hf_model_path, output_dir)
