import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, TextStreamer, AutoTokenizer, Cache, DynamicCache
from typing import Union, List, TYPE_CHECKING, Optional, Dict, Any
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights
from abc import abstractmethod
from tqdm import tqdm
from torch import Tensor
from torch.fx import GraphModule

from xhquant.core import CacheTensor
from xhquant.utils.registry.dynamic_module import DynamicModule
from xhquant.xhonnxruntime.hmonnx_inference import HMONNX_PARSERS
from xh_model_zoo_new.core.infer_adapter import InferAdapter
from xhquant.api import QuantGraph, FrontendGraph, HMONNXInference, ConfigDict
from xhquant.xhonnxruntime.hmonnx_inference import DeviceDtypeMixin
from xhquant.xhonnxruntime.hmonnx_graph_inference import HMONNXGrapInference
from xh_model_zoo_new.xh_llm.models.builder import LLM_DYNAMIC_MODULES

if TYPE_CHECKING:
    from xhquant.api import ConfigDict
    from transformers.configuration_utils import PretrainedConfig


class _LLMInfer(nn.Module):

    @property
    @abstractmethod
    def support_dynamic_input(self) -> bool:
        """Whether the model supports dynamic input"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError


# TODO disable_quant, EvalModelTYpe 这些后续再进行支持,先保证基本的实现
class BaseLLMInferQModelImpl(_LLMInfer):

    def __init__(self, model, wrap_cfg, hf_config):
        super().__init__()
        self.model = model
        self.wrap_cfg = ConfigDict(wrap_cfg)
        self.hf_config = hf_config

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        self.set_input_sequence_length(inputs_embeds.shape[-2])
        if isinstance(self.model, GraphModule):
            new_args = (inputs_embeds, past_seq_length, current_input_length, *past_key_caches, *past_value_caches)
        else:
            new_args = (inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches)
        logits = self.model(*new_args)
        return logits

    def get_input_sequence_length(self) -> int:
        return self.wrap_cfg.input_sequence_length

    def set_input_sequence_length(self, input_sequence_length: int):
        self.wrap_cfg.input_sequence_length = input_sequence_length
        self.update_cfg(self.wrap_cfg)

    def update_cfg(self, cfg: "ConfigDict"):
        def apply_fn(module):
            if hasattr(module, "_update_cfg"):
                module._update_cfg(cfg)

        self.apply(apply_fn)

    def get_num_logits_to_keep(self) -> int:
        return self.wrap_cfg.num_logits_to_keep

    @property
    def support_dynamic_input(self) -> bool:
        return True


class BaseLLMInferHMONNXImpl(_LLMInfer, DeviceDtypeMixin):
    def __init__(self, prefill_path, decoder_path, wrap_cfg: ConfigDict):
        super().__init__()
        self.prefill_session = HMONNXGrapInference(prefill_path)
        self.decoder_session = HMONNXGrapInference(decoder_path)
        self.wrap_cfg = ConfigDict(wrap_cfg)

    @property
    def support_dynamic_input(self) -> bool:
        return False

    @property
    def input_sequence_length(self) -> int:
        return self.wrap_cfg.input_sequence_length

    def get_input_sequence_length(self) -> int:
        return self.wrap_cfg.input_sequence_length

    def get_num_logits_to_keep(self) -> int:
        return self.wrap_cfg.num_logits_to_keep

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        token_length = inputs_embeds.shape[-2]
        if token_length == 1:
            return self.decode(inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches)
        else:
            return self.prefill(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
            )

    def prefill(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        seq_length = inputs_embeds.shape[-2]

        steps = (seq_length + self.input_sequence_length - 1) // self.input_sequence_length
        padding_len = steps * self.input_sequence_length - seq_length
        padding_embeds = torch.zeros(
            inputs_embeds.shape[0],
            padding_len,
            inputs_embeds.shape[-1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        if steps > 1:
            for i in tqdm(range(steps), desc="prefilling"):
                start = i * self.input_sequence_length
                end = (i + 1) * self.input_sequence_length
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                out = self.prefill_session(
                    sub_inputs_embeds,
                    sub_past_seq_length,
                    sub_current_input_length,
                    *past_key_caches,
                    *past_value_caches,
                )
        else:
            out = self.prefill_session(
                inputs_embeds, past_seq_length, current_input_length, *past_key_caches, *past_value_caches
            )
        return out

    def decode(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        return self.decoder_session(
            inputs_embeds, past_seq_length, current_input_length, *past_key_caches, *past_value_caches
        )


class BaseLLMHFCompatible(InferAdapter, DynamicModule):
    llm_model: _LLMInfer
    tokenizer: Optional[AutoTokenizer]

    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    @is_generating.setter
    def is_generating(self, value: bool) -> None:
        assert isinstance(value, bool), "is_generating must be a bool"
        self._is_generating = value

    def _setup(
        self,
        llm_model: _LLMInfer,
        embed_tokens: nn.Embedding,
        wrap_cfg: "ConfigDict",
        hf_config: "PretrainedConfig",
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.model._modules.clear()
        self.llm_model = llm_model
        self._prefill = True
        self.embed_tokens = embed_tokens
        self.past_seq_length = 0
        self.tokenizer = tokenizer
        wrap_cfg = ConfigDict(wrap_cfg)

        self._is_generating = False

        num_decoder_layers = 1 if wrap_cfg.only_first_block else hf_config.num_hidden_layers
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        kv_cache_shape = [1, hf_config.num_key_value_heads, wrap_cfg.max_sequence_length, head_dim]
        self.past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_decoder_layers)
        ]
        self.past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_decoder_layers)
        ]
        return self

    @property
    def support_dynamic_input(self) -> bool:
        return self.llm_model.support_dynamic_input

    @property
    def prefill(self) -> bool:
        return self._prefill

    @prefill.setter
    def prefill(self, value: bool) -> None:
        self._prefill = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if past_key_values is None:
            pass
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
        if inputs_embeds is None:
            self.embed_tokens.to(input_ids.device)
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds.shape[0] == 1
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        past_seq_length = torch.tensor([self.past_seq_length if self.is_generating else 0], dtype=torch.int32).to(
            inputs_embeds.device
        )
        # TODO: 需要根据attention mask计算seq_length
        seq_length = input_ids.shape[-1]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        outputs = self.llm_model(
            inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
        )
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        num_logits_to_keep = self.llm_model.get_num_logits_to_keep()
        if num_logits_to_keep != 0:
            logits = logits[:, :seq_length, :]
        else:
            logits = logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.is_generating:
            self.past_seq_length += seq_length
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def generate(self, *args, **kwargs):
        self.prefill = True
        self.past_seq_length = 0
        return super().generate(*args, **kwargs)

    @classmethod
    def from_qmodel(
        cls,
        llm_model: Union[QuantGraph, FrontendGraph, nn.Module],
        hf_config: "PretrainedConfig",
        wrap_cfg: "ConfigDict",
        token_embedding: nn.Embedding,
        native_model_or_path: Optional[Union[AutoModelForCausalLM, str]] = None,  # TODO 用 Config 来避免加载全量权重
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> "BaseLLMHFCompatible":
        """"""
        # 1. Create Infer
        infer_impl = BaseLLMInferQModelImpl(llm_model, wrap_cfg, hf_config)

        # 2. Create HFCompatible Model
        if native_model_or_path is None:
            with no_init_weights(), init_empty_weights():
                native_model = AutoModelForCausalLM.from_config(hf_config)
        elif isinstance(native_model_or_path, str):
            native_model = AutoModelForCausalLM.from_pretrained(native_model_or_path, trust_remote_code=True)
        elif isinstance(native_model_or_path, nn.Module):
            native_model = native_model_or_path
        else:
            raise ValueError(f"Invalid type: {type(native_model_or_path)}")

        if type(native_model) not in LLM_DYNAMIC_MODULES:
            LLM_DYNAMIC_MODULES.register_module(
                {
                    type(native_model): type(native_model).__name__,
                },
                cls,
            )
        return LLM_DYNAMIC_MODULES.convert(
            native_model,
            llm_model=infer_impl,
            embed_tokens=token_embedding,
            hf_config=hf_config,
            wrap_cfg=wrap_cfg,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_hmonnx(
        cls,
        prefill_path: str,
        decoder_path: str,
        hf_config,
        wrap_cfg: ConfigDict,
        token_embedding: nn.Embedding,
        native_model_or_path: Optional[Union[AutoModelForCausalLM, str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> "BaseLLMHFCompatible":
        # 1. CreateInfer
        infer_impl = BaseLLMInferHMONNXImpl(
            prefill_path=prefill_path,
            decoder_path=decoder_path,
            wrap_cfg=wrap_cfg,
        )
        # 2. Create HFCompatible Model
        if native_model_or_path is None:
            from transformers.modeling_utils import no_init_weights

            with no_init_weights():
                native_model = AutoModelForCausalLM.from_config(hf_config)
        elif isinstance(native_model_or_path, str):
            native_model = AutoModelForCausalLM.from_pretrained(native_model_or_path, trust_remote_code=True)
        elif isinstance(native_model_or_path, nn.Module):
            native_model = native_model_or_path
        else:
            raise ValueError(f"Invalid type: {type(native_model_or_path)}")

        if type(native_model) not in LLM_DYNAMIC_MODULES:
            LLM_DYNAMIC_MODULES.register_module(
                {
                    type(native_model): type(native_model).__name__,
                },
                cls,
            )
        return LLM_DYNAMIC_MODULES.convert(
            native_model,
            llm_model=infer_impl,
            embed_tokens=token_embedding,
            hf_config=hf_config,
            wrap_cfg=wrap_cfg,
            tokenizer=tokenizer,
        )

    def demo(self, prompt: str, max_generation_length: int = 32):
        """
        Demo the model with a given prompt.
        Args:
            prompt: The prompt to generate.
            max_generation_length: The maximum length of the generated text.
        Returns:
            The generated text.
        """
        device = self.embed_tokens.weight.device
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        # Create streamer for real-time output
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate with streaming
        generated_ids = self.generate(**model_inputs, max_new_tokens=max_generation_length, streamer=streamer)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
