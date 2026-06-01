from copy import deepcopy
from typing import Optional, cast

import torch
from qwen_tts import Qwen3TTSModel

from xhquant.api import QuantGraph

from ..base_llm_model import LLMBaseModel
from ..base_model import BaseModel
from ..builder import LLM_COMPATIBLE_MODULES, MODELS
from ..common.llm_hfcompatible import LLM_HFCompatible
from .qwen3_tts import XHQwen3TTSModel
from .qwen3_tts import (
    XHQwen3TTSTalkerCodePredictorModelForConditionalGeneration as Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)


@MODELS.register_module()
class XHQwen3TTSCodePredictor(LLMBaseModel):
    def get_input_embeddings(self):
        return self.token_embedding

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()
        from ._code_predictor import register_wrap_modules as register_code_predictor_wrap_modules

        register_code_predictor_wrap_modules(hf_model.model.talker.code_predictor)
        super().init_wrap_model(hf_model.model.talker.code_predictor)
        hf_model = self.wrap_model
        if isinstance(hf_model, Qwen3TTSTalkerCodePredictorModelForConditionalGeneration):
            llm_model = hf_model.model
        else:
            raise ValueError(f"{type(hf_model)} is not supported")

        # self.token_embedding.weight 和 lm_head.weight 是相同对象
        self.token_embedding = deepcopy(llm_model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = llm_model.config.num_hidden_layers

        head_dim = llm_model.layers[0].self_attn.head_dim
        self.pad_token_id = hf_model.config.eos_token_id
        self.head_dim = head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            max_layers = -1
            if only_first_block:
                max_layers = 1
            if "max_layers" in self.wrap_cfg:
                max_layers = self.wrap_cfg["max_layers"]

            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers

            self.prepare_kv_cache(
                num_decoder_layers,
                [1, llm_model.config.num_key_value_heads, self.cache_length, head_dim],
            )
        self.export_cfg.input_names.append("generate_steps")

        hf_model = None

    def get_hf_model(self, device_map="cpu", **kwargs) -> XHQwen3TTSModel:
        dtype = kwargs.get("dtype", torch.float16)
        hf_model = XHQwen3TTSModel.from_pretrained(
            self.hf_model_dir,
            device_map=device_map,
            dtype=dtype,
            # attn_implementation="flash_attention_2",
        )
        hf_model = cast(XHQwen3TTSModel, hf_model)
        return hf_model

    def convert_to_quant_graph(self, target_device: str) -> Optional[QuantGraph]:
        # raise NotImplementedError("Qwen3TTSTalker暂不支持量化导出")
        super().convert_to_quant_graph(target_device)
        assert self._quanted_model is not None
        # # TODO: 临时解决方案，后续需要修改
        # if self.extra_quant_cfg is not None:
        #     if "attn_weights" in self.extra_quant_cfg:
        #         attn_weights_cfg = self.extra_quant_cfg["attn_weights"]
        #         if "act_schema" in attn_weights_cfg:
        #             act_scheme = attn_weights_cfg["act_schema"]
        #         else:
        #             act_scheme = attn_weights_cfg["act_scheme"]

        #         if "act_schema_2" in attn_weights_cfg:
        #             weight_scheme = attn_weights_cfg["act_schema_2"]
        #         else:
        #             weight_scheme = attn_weights_cfg["act_scheme_2"]
        #         for node in self._quanted_model.graph.nodes:
        #             if node.op == "call_module":
        #                 m = self._quanted_model.get_submodule(node.target)
        #                 if isinstance(m, (xhnn.MaskedSoftmax, xhnn.SoftmaxPlus, nn.Softmax)):
        #                     i_node = node.args[0]
        #                     matmul_module = self._quanted_model.get_submodule(i_node.target)
        #                     assert isinstance(matmul_module, xhnn.MatMul), f"{type(matmul_module)}"
        #                     act_bit = act_scheme.get("bits")
        #                     if act_bit is not None:
        #                         matmul_module.i_cfg.qspec.man_bit = act_bit
        #                     w_bit = weight_scheme.get("bits")
        #                     if w_bit is not None:
        #                         matmul_module.i_cfg_2.qspec.man_bit = w_bit

        return self._quanted_model

    def prepare_inputs(self, data: dict | tuple | list):
        inputs = super().prepare_inputs(data)
        assert isinstance(data, dict)
        generate_steps = data["generate_steps"]
        if not isinstance(generate_steps, torch.Tensor):
            generate_steps = torch.tensor([generate_steps], dtype=torch.int32).to(inputs[0].device)
        inputs = list(inputs)
        inputs.append(generate_steps)
        inputs = tuple(inputs)
        return inputs

    def prepare_inputs_for_graph(self, data: dict | tuple | list):
        (
            inputs_embeds,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
            generate_steps,
        ) = self.prepare_inputs(data)
        return (
            inputs_embeds,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
            generate_steps,
        )


class Qwen3TTSTalkerCodePredictorModelForConditionalGenerationHFCompatible(LLM_HFCompatible):
    def _setup(self, llm_model: BaseModel):
        super()._setup(llm_model)
        # self.token_embedding = self.model.get_input_embeddings()
        del self.model
        del self.weight_embedding
        # self.model = nn.Module()
        self.model = llm_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self


def build_qwen3_tts_code_predictor_hf_compatible(
    hf_model: Qwen3TTSModel, xh_code_predictor_model: BaseModel
) -> Qwen3TTSModel:
    if Qwen3TTSTalkerCodePredictorModelForConditionalGeneration not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                Qwen3TTSTalkerCodePredictorModelForConditionalGeneration: "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
            },
            Qwen3TTSTalkerCodePredictorModelForConditionalGenerationHFCompatible,
        )

    LLM_COMPATIBLE_MODULES.convert(hf_model.model.talker.code_predictor, llm_model=xh_code_predictor_model)
    return hf_model
