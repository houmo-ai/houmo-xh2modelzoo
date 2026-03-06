from copy import deepcopy
from typing import Optional, Union

import torch.nn as nn
from transformers.models.qwen2 import Qwen2ForCausalLM
from xhquant import nn as xhnn
from xhquant.api import QuantGraph

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen2LegacyModel(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        if "extra_cfg" in self.quant_cfg:
            self.extra_quant_cfg = self.quant_cfg.pop("extra_cfg")
        else:
            self.extra_quant_cfg = None

    # def _set_device(self, device):
    #     self.token_embedding = self.token_embedding.to(device)
    #     return super()._set_device(device)

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def init_wrap_model(self, hf_model=None):
        from ._model import register_wrap_modules as qwen2_register_wrap_modules

        qwen2_register_wrap_modules()

        super().init_wrap_model(hf_model)
        hf_model = self.wrap_model

        # self.token_embedding.weight 和 lm_head.weight 是相同对象
        self.token_embedding = deepcopy(hf_model.model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        # assert self.num_hidden_layers == 28
        # head_dim = hf_model.model.config.hidden_size // hf_model.model.config.num_attention_heads
        head_dim = hf_model.model.layers[0].self_attn.head_dim
        self.pad_token_id = hf_model.config.eos_token_id
        self.head_dim = head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [1, hf_model.model.config.num_key_value_heads, self.cache_length, head_dim],
            )
        #     self.past_key_caches = []
        #     self.past_value_caches = []
        #     num_decoder_layers = self.num_hidden_layers
        #     only_first_block = self.wrap_cfg.get("only_first_block", False)
        #     if only_first_block:
        #         num_decoder_layers = 1
        #     for i in range(num_decoder_layers):
        #         self.past_key_caches.append(
        #             CacheTensor(
        #                 torch.zeros(
        #                     1,
        #                     hf_model.model.config.num_key_value_heads,
        #                     self.cache_length,
        #                     head_dim,
        #                     dtype=torch.float16,
        #                 )
        #             )
        #         )
        #         self.past_value_caches.append(
        #             CacheTensor(
        #                 torch.zeros(
        #                     1,
        #                     hf_model.model.config.num_key_value_heads,
        #                     self.cache_length,
        #                     head_dim,
        #                     dtype=torch.float16,
        #                 )
        #             )
        #         )
        #         # self.register_buffer(
        #         #     f"past_k_cache_{i}",
        #         #     torch.zeros(
        #         #         1, hf_model.model.config.num_key_value_heads, self.cache_length, head_dim, dtype=torch.float16
        #         #     ),
        #         #     persistent=False,
        #         # )
        #         # self.register_buffer(
        #         #     f"past_v_cache_{i}",
        #         #     torch.zeros(
        #         #         1, hf_model.model.config.num_key_value_heads, self.cache_length, head_dim, dtype=torch.float16
        #         #     ),
        #         #     persistent=False,
        #         # )
        #     # 插入KVCache输入的量化配置
        # for layer_idx in range(num_decoder_layers):
        #     self.quant_cfg.inputs[f"past_key_cache_{layer_idx}"] = ConfigDict(
        #         dict(
        #             quantizer=dict(
        #                 qspec=dict(fake_dtype="float16"),
        #             )
        #         )
        #     )
        #     self.quant_cfg.inputs[f"past_value_cache_{layer_idx}"] = ConfigDict(
        #         dict(
        #             quantizer=dict(
        #                 qspec=dict(fake_dtype="float16"),
        #             )
        #         )
        #     )

        # for layer_idx in range(num_decoder_layers):
        #     self.export_cfg.input_names.append(f"past_key_cache_{layer_idx}")
        # for layer_idx in range(num_decoder_layers):
        #     self.export_cfg.input_names.append(f"past_value_cache_{layer_idx}")

        hf_model = None

    def get_hf_model(self, device_map="cpu", **kwargs) -> Qwen2ForCausalLM:
        hf_model: Qwen2ForCausalLM = super().get_hf_model(device_map)
        assert id(hf_model.get_output_embeddings().weight) != id(hf_model.get_input_embeddings().weight)
        return hf_model

    def convert_to_quant_graph(self, target_device: str) -> Optional[QuantGraph]:
        super().convert_to_quant_graph(target_device)
        assert self._quanted_model is not None
        # TODO: 临时解决方案，后续需要修改
        if self.extra_quant_cfg is not None:
            if "attn_weights" in self.extra_quant_cfg:
                attn_weights_cfg = self.extra_quant_cfg["attn_weights"]
                if "act_schema" in attn_weights_cfg:
                    act_scheme = attn_weights_cfg["act_schema"]
                else:
                    act_scheme = attn_weights_cfg["act_scheme"]

                if "act_schema_2" in attn_weights_cfg:
                    weight_scheme = attn_weights_cfg["act_schema_2"]
                else:
                    weight_scheme = attn_weights_cfg["act_scheme_2"]
                for node in self._quanted_model.graph.nodes:
                    if node.op == "call_module":
                        m = self._quanted_model.get_submodule(node.target)
                        if isinstance(m, (xhnn.MaskedSoftmax, xhnn.SoftmaxPlus, nn.Softmax)):
                            i_node = node.args[0]
                            matmul_module = self._quanted_model.get_submodule(i_node.target)
                            assert isinstance(matmul_module, xhnn.MatMul), f"{type(matmul_module)}"
                            act_bit = act_scheme.get("bits")
                            if act_bit is not None:
                                matmul_module.i_cfg.qspec.man_bit = act_bit
                            w_bit = weight_scheme.get("bits")
                            if w_bit is not None:
                                matmul_module.i_cfg_2.qspec.man_bit = w_bit

        return self._quanted_model
