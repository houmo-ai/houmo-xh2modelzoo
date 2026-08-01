import hashlib
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer

from xhmodel_merak.xh_llm.llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from xhquant.api import get_xhquant_logger
from xhquant.core.hmfp_kv_cache import HMFPPagedKVCache
from xhquant.xhonnxruntime.convert_to_page_attention import convert_to_page_attention
from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2
from xhquant.xhonnxruntime.hmonnx_optimizer import (
    materialize_parallel_linear_fusion,
)
from xhquant.xhonnxruntime.llm_hmonnx_loader import LLMHMONNXLoader
from xhquant.xhonnxruntime.parsers import PageAttention, PageAttentionContext

from ..base_llm_model import BaseLLMModel, XHLLMModelProcessor
from ..kv_cache_mixin import KVCacheMixin
from ..types import KVCacheConfig, LLMModelMeta
from ..utils import unfold_args
from .hmonnx_model import HMONNXBaseModel, HMONNXModel


class BaseLLMHMONNXModel(HMONNXBaseModel):
    LLM_MODEL_CLS: type[BaseLLMModel] = BaseLLMModel

    def __init__(
        self,
        meta: LLMModelMeta,
        enable_cuda_graph=False,
        enable_auto_offload=False,
        enable_golden=False,
        enable_prefill_cuda_graph: bool | None = None,
        enable_decode_cuda_graph: bool | None = None,
        enable_page_attention: bool = False,
        enable_parallel_linear_fusion: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.meta_info = meta
        self.hf_model_dir = meta.hf_config
        self.hf_compatible_model = None
        self.embed_tokens = self._build_embed_tokens_from_meta(meta)

        self._llm_prefill = True
        self.kvcache_config = (
            meta.kv_cache if isinstance(meta.kv_cache, KVCacheConfig) else KVCacheConfig(**meta.kv_cache)
        )
        self.use_cache = self.kvcache_config.num_layers > 0
        self.enable_page_attention = enable_page_attention
        convert_unfused_sliding_kv_cache = (
            str(getattr(meta, "attention_lowering", ""))
            == "full_flash_attention"
        )
        prefill_hmonnx, decode_hmonnx = self._prepare_hmonnx_paths(
            meta.prefill_hmonnx,
            meta.decode_hmonnx,
            enable_parallel_linear_fusion=enable_parallel_linear_fusion,
            enable_page_attention=enable_page_attention,
            convert_unfused_sliding_kv_cache=convert_unfused_sliding_kv_cache,
        )
        prefill_graph = None
        decode_graph = None
        if True:
            llm_loader = LLMHMONNXLoader(prefill_hmonnx, decode_hmonnx)
            prefill_graph = llm_loader.prefill_graph
            decode_graph = llm_loader.decode_graph
        prefill_cuda_graph = enable_cuda_graph if enable_prefill_cuda_graph is None else enable_prefill_cuda_graph
        decode_cuda_graph = enable_cuda_graph if enable_decode_cuda_graph is None else enable_decode_cuda_graph

        self.prefill_model = HMONNXModel(
            prefill_hmonnx,
            onnx_graph=prefill_graph,
            enable_cuda_graph=prefill_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=self._valid_devices,
        )

        layer_infos = None
        if isinstance(self.prefill_model.hmonnx_session, HMONNXInferenceV2):
            layer_infos = self.prefill_model.hmonnx_session.get_layer_infos()

        self.decode_model = HMONNXModel(
            decode_hmonnx,
            onnx_graph=decode_graph,
            enable_golden=enable_golden,
            enable_cuda_graph=decode_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            device_map=self._valid_devices,
            layer_infos=layer_infos,
        )

        self._data_processor = None
        self._kvcache_mixin = KVCacheMixin(self.kvcache_config)
        self._sync_page_attention_mode_to_kvcache()
        self.pad_token_id = self.meta_info.pad_token_id

    @staticmethod
    def _prepare_hmonnx_paths(
        prefill_hmonnx: str | Path,
        decode_hmonnx: str | Path,
        *,
        enable_parallel_linear_fusion: bool,
        enable_page_attention: bool,
        convert_unfused_sliding_kv_cache: bool = False,
    ) -> tuple[str, str]:
        """Apply immutable graph derivations before loading any initializer."""

        prefill_path = str(prefill_hmonnx)
        decode_path = str(decode_hmonnx)
        if enable_parallel_linear_fusion:
            prefill_path = str(materialize_parallel_linear_fusion(prefill_path))
            decode_path = str(materialize_parallel_linear_fusion(decode_path))
        if enable_page_attention:
            # PageAttention must lower the already fused graph so both
            # optimizations are present in the one graph loaded at runtime.
            if convert_unfused_sliding_kv_cache:
                prefill_path = BaseLLMHMONNXModel._convert_to_page_attention_hmonnx(
                    prefill_path,
                    convert_unfused_sliding_kv_cache=True,
                )
                decode_path = BaseLLMHMONNXModel._convert_to_page_attention_hmonnx(
                    decode_path,
                    convert_unfused_sliding_kv_cache=True,
                )
            else:
                prefill_path = BaseLLMHMONNXModel._convert_to_page_attention_hmonnx(prefill_path)
                decode_path = BaseLLMHMONNXModel._convert_to_page_attention_hmonnx(decode_path)
        return prefill_path, decode_path

    def _sync_page_attention_mode_to_kvcache(self) -> None:
        if hasattr(self._kvcache_mixin, "enable_page_attention"):
            self._kvcache_mixin.enable_page_attention = self.enable_page_attention

    @staticmethod
    def _convert_to_page_attention_hmonnx(
        hmonnx_path: str | Path,
        *,
        convert_unfused_sliding_kv_cache: bool = False,
    ) -> str:
        input_path = Path(hmonnx_path)
        digest_source = (
            f"{input_path.resolve(strict=False)}|"
            f"mixed={int(convert_unfused_sliding_kv_cache)}"
        )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        output_name = f"{input_path.stem}_page_attention_{digest}{input_path.suffix}"
        output_path = input_path.with_name(output_name)
        try:
            convert_to_page_attention(
                input_path,
                output_path,
                convert_unfused_sliding_kv_cache=convert_unfused_sliding_kv_cache,
            )
            return str(output_path)
        except PermissionError:
            logger = get_xhquant_logger()
            logger.warning(
                f"No permission to write page-attention HMONNX beside {input_path}; falling back to temp dir."
            )

        output_path = Path(tempfile.gettempdir()) / "xhmodel_merak_page_attention" / digest / output_name
        convert_to_page_attention(
            input_path,
            output_path,
            convert_unfused_sliding_kv_cache=convert_unfused_sliding_kv_cache,
        )
        return str(output_path)

    def _get_page_attention_modules(self, hmonnx_model: HMONNXModel) -> list[PageAttention]:
        session = hmonnx_model.hmonnx_session
        if not hasattr(session, "graph_module"):
            # Legacy eager HMONNX wraps the real graph session and creates it
            # lazily on the first forward. Page-attention context has to be
            # attached before that forward, so initialize and unwrap it here.
            initialize = getattr(session, "initialize", None)
            if callable(initialize):
                initialize()
            session = getattr(session, "_session", session)
        graph_module = getattr(session, "graph_module", None)
        if graph_module is None:
            node_modules = getattr(session, "node_modules", None)
            if node_modules is not None:
                return [module for module in node_modules if isinstance(module, PageAttention)]
            raise RuntimeError(
                f"HMONNX session {type(session).__name__} exposes neither graph_module "
                "nor node_modules for PageAttention context binding"
            )
        page_attention_modules = []
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                m = graph_module.get_submodule(str(node.target))
                if isinstance(m, PageAttention):
                    page_attention_modules.append(m)
        return page_attention_modules

    def set_page_attention_context(
        self, paged_kv_caches: list[HMFPPagedKVCache], block_ids: Tensor, slot_mapping: Tensor, block_size: int
    ):
        if paged_kv_caches is None:
            raise ValueError("paged_kv_caches is required for page attention.")
        if block_ids is None:
            raise ValueError("block_ids is required for page attention.")
        if slot_mapping is None:
            raise ValueError("slot_mapping is required for page attention.")

        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.as_tensor(block_ids, dtype=torch.int64)
        if not isinstance(slot_mapping, torch.Tensor):
            slot_mapping = torch.as_tensor(slot_mapping, dtype=torch.int64)
        block_size = int(block_size)

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        if self._llm_prefill:
            page_attention_modules = self._get_page_attention_modules(self.prefill_model)
        else:
            page_attention_modules = self._get_page_attention_modules(self.decode_model)

        if len(page_attention_modules) != len(paged_kv_caches):
            raise ValueError(
                "PageAttention module/cache count mismatch: "
                f"{len(page_attention_modules)} modules vs {len(paged_kv_caches)} caches."
            )

        contexts_by_device = BaseLLMHMONNXModel._stage_page_attention_context_by_device(
            self,
            paged_kv_caches,
            block_ids,
            slot_mapping,
        )

        for layer_idx, page_attn_m in enumerate(page_attention_modules):
            paged_kv_cache = paged_kv_caches[layer_idx]
            block_ids_tensor, slot_mapping_tensor = contexts_by_device[str(torch.device(paged_kv_cache.device))]
            page_attn_context = PageAttentionContext(
                paged_kv_cache=paged_kv_cache,
                block_ids=block_ids_tensor,
                slot_mapping=slot_mapping_tensor,
                block_size=block_size,
            )
            page_attn_m.set_context(page_attn_context)

    def _stage_page_attention_context_by_device(
        self,
        paged_kv_caches: list[HMFPPagedKVCache],
        block_ids: Tensor,
        slot_mapping: Tensor,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Update model-local, fixed-address metadata buffers on each cache device."""
        stage = "prefill" if self._llm_prefill else "decode"
        buffers = getattr(self, "_page_attention_context_device_buffers", None)
        if buffers is None:
            buffers = {}
            self._page_attention_context_device_buffers = buffers

        capacities_by_device: dict[str, int] = {}
        for paged_kv_cache in paged_kv_caches:
            device_key = str(torch.device(paged_kv_cache.device))
            capacities_by_device[device_key] = max(
                capacities_by_device.get(device_key, 0),
                int(paged_kv_cache.num_blocks),
            )

        staged = {}
        invalidate_active_graph = False
        block_ids_flat = block_ids.reshape(-1)
        slot_mapping_flat = slot_mapping.reshape(-1)
        for device_key, block_capacity in capacities_by_device.items():
            device = torch.device(device_key)
            key = (stage, device_key)
            device_buffers = buffers.get(key)
            if device_buffers is None:
                device_buffers = {}
                buffers[key] = device_buffers

            block_buffer = device_buffers.get("block_ids")
            if not isinstance(block_buffer, Tensor) or block_buffer.numel() < block_capacity:
                invalidate_active_graph |= isinstance(block_buffer, Tensor)
                block_buffer = torch.empty(block_capacity, dtype=torch.int64, device=device)
                device_buffers["block_ids"] = block_buffer

            slot_capacity = int(slot_mapping_flat.numel())
            slot_buffer = device_buffers.get("slot_mapping")
            if not isinstance(slot_buffer, Tensor) or slot_buffer.numel() < slot_capacity:
                invalidate_active_graph |= isinstance(slot_buffer, Tensor)
                slot_buffer = torch.empty(slot_capacity, dtype=torch.int64, device=device)
                device_buffers["slot_mapping"] = slot_buffer

            if block_ids_flat.numel() > block_buffer.numel():
                raise ValueError(
                    f"block_ids length {block_ids_flat.numel()} exceeds cache capacity {block_buffer.numel()} "
                    f"on {device}."
                )

            block_view_length = int(block_ids_flat.numel())
            slot_view_length = int(slot_mapping_flat.numel())
            previous_block_view_length = device_buffers.get("block_ids_view_length")
            previous_slot_view_length = device_buffers.get("slot_mapping_view_length")
            if previous_block_view_length is not None and previous_block_view_length != block_view_length:
                invalidate_active_graph = True
            if previous_slot_view_length is not None and previous_slot_view_length != slot_view_length:
                invalidate_active_graph = True
            device_buffers["block_ids_view_length"] = block_view_length
            device_buffers["slot_mapping_view_length"] = slot_view_length

            if device.type == "cuda":
                with torch.cuda.device(device):
                    block_buffer.zero_()
                    block_buffer[: block_ids_flat.numel()].copy_(block_ids_flat, non_blocking=True)
                    slot_buffer.fill_(-1)
                    slot_buffer[: slot_mapping_flat.numel()].copy_(slot_mapping_flat, non_blocking=True)
            else:
                block_buffer.zero_()
                block_buffer[: block_ids_flat.numel()].copy_(block_ids_flat)
                slot_buffer.fill_(-1)
                slot_buffer[: slot_mapping_flat.numel()].copy_(slot_mapping_flat)
            # Backing buffers keep cache-capacity allocation and stable storage,
            # while PageAttention sees only the active/bucketed metadata range.
            # Passing the full buffer makes a short decode look like a maximum
            # length context and can select the wrong split-KV kernel.
            staged[device_key] = (
                block_buffer[:block_view_length],
                slot_buffer[:slot_view_length],
            )

        if invalidate_active_graph:
            BaseLLMHMONNXModel._clear_active_page_attention_cuda_graph(self)
        return staged

    def _clear_active_page_attention_cuda_graph(self) -> None:
        """Clear only the prefill/decode HMONNX graph whose metadata pointer changed."""
        active_model = self.prefill_model if self._llm_prefill else self.decode_model
        session = getattr(active_model, "hmonnx_session", None)
        interpreter = getattr(session, "interpreter", None)
        clear = getattr(interpreter, "clear", None)
        if callable(clear):
            clear(clear_disabled_reason=True)

    @property
    def past_key_caches(self):
        return self._kvcache_mixin.past_key_caches

    @property
    def past_value_caches(self):
        return self._kvcache_mixin.past_value_caches

    def get_kvcache_mixin(self):
        self._sync_page_attention_mode_to_kvcache()
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
        self._sync_page_attention_context_to_processor()
        self._data_processor.to(self.device, self.dtype)
        return self._data_processor

    def _sync_page_attention_context_to_processor(self) -> None:
        if self._data_processor is not None:
            self._data_processor.enable_page_attention = self.enable_page_attention

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        preprocessor = BaseLLMInputProcessor(
            BaseInputProcessorConfig(
                embed_tokens=self.get_input_embeddings(),
                input_sequence_length=self.get_input_sequence_length(),
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                enable_page_attention=self.enable_page_attention,
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
        try:
            loaded = torch.load(str(quant_embedding_path), map_location="cpu", weights_only=True)
        except Exception:
            loaded = torch.load(str(quant_embedding_path), map_location="cpu", weights_only=False)

        if isinstance(loaded, nn.Embedding):
            return loaded

        state_dict = loaded
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
            state_dict = loaded
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
        self.set_input_sequence_length(self.meta_info.model_config.prefill_chunk_length)
        # if self.fast_mode:
        #     self.prefill_model.to_fast()

    def set_decode(self):
        self._llm_prefill = False
        """更新解码阶段使用的 HMONNX 模型路径，并重新加载模型。"""
        self.prefill_model.to(device="cpu")
        self.decode_model.to(device=self.device)
        self.set_input_sequence_length(self.get_input_sequence_length())
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
        else:
            self.decode_model.to(device=self.device)
            out = self.decode_model(*args)
        if isinstance(out, (tuple, list)) and len(out) == 1:
            out = out[0]
        return out

    def prepare_for_inference(self):
        pass

    def release_inference_model(self):
        pass
