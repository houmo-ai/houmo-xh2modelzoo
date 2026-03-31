from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from xhmodel_merak.xh_llm.llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from xhquant.api import get_xhquant_logger

from ..base_llm_model import BaseLLMModel, XHLLMModelProcessor
from ..kv_cache_mixin import KVCacheMixin
from ..types import KVCacheConfig, LLMModelMeta
from ..utils import unfold_args
from .hmonnx_model import HMONNXBaseModel, HMONNXModel


class BaseLLMHMONNXModel(HMONNXBaseModel):
    LLM_MODEL_CLS: type[BaseLLMModel] = BaseLLMModel

    def __init__(self, meta: LLMModelMeta):
        super().__init__()
        self.meta_info = meta
        self.hf_model_dir = meta.hf_config
        self.hf_compatible_model = None
        self.embed_tokens = self._build_embed_tokens_from_meta(meta)

        self._llm_prefill = True
        self.kvcache_config = (
            meta.kv_cache if isinstance(meta.kv_cache, KVCacheConfig) else KVCacheConfig(**meta.kv_cache)
        )
        self.use_cache = self.kvcache_config.num_layers > 0
        self.prefill_model = HMONNXModel(meta.prefill_hmonnx)
        self.decode_model = HMONNXModel(meta.decode_hmonnx)

        self._data_processor = None
        self._kvcache_mixin = KVCacheMixin(self.kvcache_config)
        self.pad_token_id = self.meta_info.pad_token_id

    @property
    def past_key_caches(self):
        return self._kvcache_mixin.past_key_caches

    @property
    def past_value_caches(self):
        return self._kvcache_mixin.past_value_caches

    def get_kvcache_mixin(self):
        return self._kvcache_mixin

    def get_num_logits_to_keep(self) -> int:
        return self.meta_info.model_config.num_logits_to_keep

    def get_input_sequence_length(self) -> int:
        if self._llm_prefill:
            return self.meta_info.model_config.prefill_chunk_length
        else:
            return 1

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_sequence_length(self, seq_length: int):
        # 该函数在每次前向传播时被调用，用于设置当前输入序列的长度，以便模型能够正确处理不同长度的输入
        # 对于预填充阶段，输入序列长度为预填充块长度；对于解码阶段，输入序列长度为1
        if self._data_processor is not None:
            self._data_processor.input_sequence_length = seq_length

    def get_data_preprocessor(self) -> BaseLLMInputProcessor:
        if self._data_processor is None:
            self._data_processor = self._get_data_preprocessor()
        self._data_processor.to(self.device, self.dtype)
        return self._data_processor

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        preprocessor = BaseLLMInputProcessor(
            BaseInputProcessorConfig(
                embed_tokens=self.get_input_embeddings(),
                input_sequence_length=self.get_input_sequence_length(),
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                pad_token_id=self.pad_token_id,
            )
        )
        return preprocessor

    def _set_device(self, device):
        super()._set_device(device)

        self.embed_tokens.to(device)
        self._kvcache_mixin.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        return self

    def get_tf_processor(self):
        return XHLLMModelProcessor.from_pretraind(self.hf_model_dir)

    @staticmethod
    def _build_embed_tokens_from_meta(meta: LLMModelMeta) -> nn.Embedding:
        """从 meta 中的量化 embedding 文件构建 nn.Embedding 模块。

        Args:
            meta: LLMModelMeta 对象，包含 embedding 文件路径

        Returns:
            加载了量化权重的 nn.Embedding 模块
        """
        # 获取 embedding 文件路径
        quant_embedding_path = Path(meta.quant_embedding)
        state_dict = torch.load(str(quant_embedding_path), map_location="cpu")
        vocab_size, hidden_size = state_dict["weight"].shape
        # 从 HuggingFace 配置获取 vocab_size 和 hidden_size
        config = AutoConfig.from_pretrained(meta.hf_config, trust_remote_code=True)
        if hasattr(config, "text_config"):
            config = config.text_config
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size

        # 创建 nn.Embedding 实例
        embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.float16)

        # 加载从文件中保存的 state_dict
        if quant_embedding_path.exists():
            state_dict = torch.load(str(quant_embedding_path), map_location="cpu")
            embed_tokens.load_state_dict(state_dict)
            logger = get_xhquant_logger()
            logger.info(f"Loaded quantized embedding from {quant_embedding_path}")
        else:
            logger = get_xhquant_logger()
            logger.warning(f"Quantized embedding file not found: {quant_embedding_path}")

        return embed_tokens

    def get_tokenizer(self, **kwargs):
        assert self.hf_model_dir is not None
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, **kwargs)
        return tokenizer

    def is_support_dynamic_input(self) -> bool:
        return False

    def is_prefill(self):
        return self._llm_prefill

    def is_decode(self):
        return not self._llm_prefill

    def set_prefill(self):
        self._llm_prefill = True
        """更新预填充阶段使用的 HMONNX 模型路径，并重新加载模型。"""
        self.prefill_model.to(device=self.device)
        # if self.fast_mode:
        #     self.prefill_model.to_fast()

    def set_decode(self):
        self._llm_prefill = False
        """更新解码阶段使用的 HMONNX 模型路径，并重新加载模型。"""
        self.prefill_model.to(device="cpu")
        self.decode_model.to(device=self.device)
        # if self.fast_mode:
        #     self.decode_model.to_fast()

    def generate(self, *args, **kwargs):
        llm_model_cls = self.LLM_MODEL_CLS
        # hf_model = llm_model_cls.get_empty_hf_model(self.hf_model_dir)
        hf_model = llm_model_cls._get_hf_model_for_compatible(self.hf_model_dir)
        if self.hf_compatible_model is None:
            # 从类中直接获取函数，避免自动绑定 self
            hf_compatible_model = llm_model_cls.build_hf_compatible_model(hf_model, self)
            assert isinstance(hf_compatible_model, llm_model_cls.get_hf_model_cls())
            hf_compatible_model.to(device=self.device, dtype=self.dtype)
            self.hf_compatible_model = hf_compatible_model

        infer_model = self.hf_compatible_model
        assert infer_model is not None
        out = infer_model.generate(*args, **kwargs)
        return out

    def forward(self, *args, **kwargs):
        args = unfold_args(args)
        if self._llm_prefill:
            self.prefill_model.to(device=self.device)
            out = self.prefill_model(*args)
            if self.enable_golden:
                self.prefill_model.update_step()
        else:
            self.decode_model.to(device=self.device)
            out = self.decode_model(*args)
            if self.enable_golden:
                self.decode_model.update_step()
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        return out

    def prepare_for_inference(self):
        pass

    def release_inference_model(self):
        pass
