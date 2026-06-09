from pathlib import Path
from typing import Optional, Union, cast

import torch
import torch.nn as nn
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerOutputWithPast,
)
from torch import Tensor
from tqdm import tqdm
from transformers import Cache, DynamicCache

from xhquant.utils.registry import DynamicModule

from xh_model_zoo.api import get_root_logger

from ..base_model import BaseModel
from ..builder import LLM_COMPATIBLE_MODULES, MODELS
from ..common.hmonnx_model import HMONNXModel
from ..common.llm_hfcompatible import LLM_HFCompatible
from ..llm_onnx_model import LLMONNXModel
from .qwen3_tts import XHQwen3TTSModel
from .qwen3_tts import (
    XHQwen3TTSTalkerCodePredictorModelForConditionalGeneration as Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)
from .qwen3_tts import XHQwen3TTSTalkerForConditionalGeneration as Qwen3TTSTalkerForConditionalGeneration


@MODELS.register_module()
class Qwen3TTSTextProjectionInference(HMONNXModel):
    def forward(self, *inputs) -> Tensor:
        assert self.session is not None, "ONNX session is not initialized!"
        input_name = self.session.get_input_names()[0]
        input_info = self.session.get_input(input_name)
        net_input_seq_len = input_info.shape[1]
        hidden_state = inputs[0]
        input_seq_len = hidden_state.shape[1]
        # steps = (input_seq_len + net_input_seq_len - 1) // net_input_seq_len
        logger = get_root_logger()
        logger.info(f"TextProjectionInference: input_seq_len={input_seq_len}, net_input_seq_len={net_input_seq_len}")
        hidden_state_chunks = hidden_state.split(net_input_seq_len, dim=1)
        outputs = []
        for chunk in hidden_state_chunks:
            outputs.append(super().forward(chunk))
        out = torch.cat(outputs, dim=1)
        assert isinstance(out, Tensor)
        return out


@MODELS.register_module()
class Qwen3TTSCodePredictorInference(LLMONNXModel):
    def __init__(self, model_cfg: str) -> None:
        import json

        with open(model_cfg, "r") as f:
            cfg = json.load(f)
        model_dir = Path(model_cfg).parent
        prefill_hmonnx_file = str(model_dir / cfg["prefill_onnx_file"])
        decode_hmonnx_file = str(model_dir / cfg["decode_onnx_file"])
        prefill = {
            "onnx": prefill_hmonnx_file,
            "input_sequence_length": cfg["wrap_cfg"]["input_sequence_length"],
        }
        decode = {
            "onnx": decode_hmonnx_file,
            "input_sequence_length": cfg["wrap_cfg"]["input_sequence_length"],
        }
        kv_cache_shape = cfg["kv_cache_shape"]
        kv_cache = {
            "num_hidden_layers": cfg["num_hidden_layers"],
            "shape": kv_cache_shape,
        }
        super().__init__(prefill, decode, kv_cache)

        token_embedding_file = model_dir / cfg["token_embedding_file"]
        embeding_state_dict = torch.load(token_embedding_file, map_location="cpu")
        num_embeddings, embedding_dim = embeding_state_dict["0.weight"].shape
        token_embedding = []
        for _ in range(len(embeding_state_dict)):
            token_embedding.append(
                nn.Embedding(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                )
            )

        self.token_embedding = nn.ModuleList(token_embedding)
        self.token_embedding.load_state_dict(embeding_state_dict)
        self.token_embedding.to(torch.float16)

    def get_input_sequence_length(self):
        if self._prefill:
            return self.prefill_input_sequence_length
        else:
            return 1

    def set_input_sequence_length(self, input_sequence_length: int):
        pass

    def forward(self, *args, **kwargs) -> Tensor:
        ## 将输入的List展开
        new_args = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        args = new_args
        if "generation_steps" in kwargs:
            generate_steps = kwargs["generation_steps"]
            generate_steps = generate_steps.to(torch.int32)
            args.append(generate_steps)
        logger = get_root_logger()
        if self._prefill:
            logger.info(
                f"CodePredictor HMONNX Inference: prefill mode, input_sequence_length={self.prefill_input_sequence_length}"
            )
            assert self.prefill_session is not None, "Prefill session is not initialized!"
            return self.prefill_session(*args)
        else:
            logger.info("CodePredictor HMONNX Inference: decode mode")
            assert self.decode_session is not None, "Decode session is not initialized!"
            return self.decode_session(*args)


class Qwen3TTSTalkerCodePredictorModelHMONNXHFCompatible(LLM_HFCompatible):
    def _setup(self, llm_model: LLMONNXModel):
        super()._setup(llm_model)

        del self.model
        del self.weight_embedding
        # self.model = nn.Module()
        self.model = llm_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self


def build_qwen3_tts_code_predictor_hmonnx_hf_compatible(
    code_predictor: Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    llm_model,
) -> Qwen3TTSTalkerCodePredictorModelForConditionalGeneration:
    if Qwen3TTSTalkerCodePredictorModelForConditionalGeneration not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                Qwen3TTSTalkerCodePredictorModelForConditionalGeneration: "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
            },
            Qwen3TTSTalkerCodePredictorModelHMONNXHFCompatible,
        )

    LLM_COMPATIBLE_MODULES.convert(code_predictor, llm_model=llm_model)
    assert isinstance(code_predictor, Qwen3TTSTalkerCodePredictorModelHMONNXHFCompatible)
    return code_predictor


@MODELS.register_module()
class Qwen3TTSTalkerInference(LLMONNXModel):
    def __init__(self, model_cfg: str) -> None:
        import json

        with open(model_cfg, "r") as f:
            cfg = json.load(f)
        model_dir = Path(model_cfg).parent
        prefill_hmonnx_file = str(model_dir / cfg["prefill_onnx_file"])
        decode_hmonnx_file = str(model_dir / cfg["decode_onnx_file"])
        prefill = {
            "onnx": prefill_hmonnx_file,
            "input_sequence_length": cfg["wrap_cfg"]["input_sequence_length"],
        }
        decode = {
            "onnx": decode_hmonnx_file,
            "input_sequence_length": cfg["wrap_cfg"]["input_sequence_length"],
        }
        kv_cache_shape = cfg["kv_cache_shape"]
        kv_cache = {
            "num_hidden_layers": cfg["num_hidden_layers"],
            "shape": kv_cache_shape,
        }
        super().__init__(prefill, decode, kv_cache)

        token_embedding_file = model_dir / cfg["token_embedding_file"]
        text_embedding_file = model_dir / cfg["text_embedding_file"]
        # token_embedding_state_dict = torch.load(token_embedding_file, map_location="cpu")
        # text_embedding_state_dict = torch.load(text_embedding_file, map_location="cpu")
        self.token_embedding = torch.nn.Embedding.from_pretrained(
            torch.load(token_embedding_file, map_location="cpu")["weight"], freeze=True
        )
        self.text_embedding = torch.nn.Embedding.from_pretrained(
            torch.load(text_embedding_file, map_location="cpu")["weight"], freeze=True
        )
        self.token_embedding.to(torch.float16)
        self.text_embedding.to(torch.float16)

    def get_input_sequence_length(self):
        if self._prefill:
            return self.prefill_input_sequence_length
        else:
            return 1

    def set_input_sequence_length(self, input_sequence_length: int):
        pass

    def forward(self, *args, **kwargs) -> Tensor:
        ## 将输入的List展开
        new_args = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        args = new_args

        logger = get_root_logger()
        if self._prefill:
            logger.info(
                f"Talker HMONNX Inference: prefill mode, input_sequence_length={self.prefill_input_sequence_length}"
            )
            assert self.prefill_session is not None, "Prefill session is not initialized!"
            return self.prefill_session(*args)
        else:
            logger.info("Talker HMONNX Inference: decode mode")
            assert self.decode_session is not None, "Decode session is not initialized!"
            return self.decode_session(*args)

    def get_input_embeddings(self):
        return self.token_embedding

    def get_text_embeddings(self):
        return self.text_embedding


class Qwen3TTSTalkerHMONNXHFCompatible(LLM_HFCompatible):
    def _setup(self, llm_model: Qwen3TTSTalkerInference):
        m = super()._setup(llm_model)
        if llm_model is not None:
            assert isinstance(llm_model, Qwen3TTSTalkerInference)
            del m.model
            del m.codec_head
            self.model = llm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return m

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
    ) -> Qwen3TTSTalkerOutputWithPast:
        if past_key_values is None:
            pass

        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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

        # if self.gradient_checkpointing and self.training and use_cache:
        #     # logger.warning_once(
        #     #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        #     # )
        #     use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
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

        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]

        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        if self.dynamic_input and isinstance(self._llm_model, BaseModel):
            self._llm_model.set_input_sequence_length(seq_length)

        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        padding_len = steps * net_input_seq_len - seq_length
        padding_embeds = self.embed_tokens(
            torch.zeros(inputs_embeds.shape[0], padding_len, dtype=torch.long, device=inputs_embeds.device)
        )
        inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        num_logits_to_keep = self._llm_model.get_num_logits_to_keep()
        assert num_logits_to_keep == 1, "Qwen3TTSTalker目前只支持num_logits_to_keep=1的情况"
        logits = None
        past_hidden = None
        if steps > 1:
            for i in tqdm(range(steps)):
                start = i * net_input_seq_len
                end = (i + 1) * net_input_seq_len
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                self._llm_model.set_input_sequence_length(int(sub_current_input_length.item()))
                output = self._llm_model.forward(
                    sub_inputs_embeds, sub_past_seq_length, sub_current_input_length, past_key_caches, past_value_caches
                )
                if isinstance(output, (list, tuple)):
                    logits, past_hidden = output
                else:
                    logits = output.logits
                    past_hidden = output.past_hidden

        else:
            output = self._llm_model.forward(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
            )
            if isinstance(output, (list, tuple)):
                logits, past_hidden = output
            else:
                logits = output["logits"]
                past_hidden = output["past_hidden"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Qwen3TTSTalkerOutputWithPast(
            logits=logits,
            past_hidden=past_hidden,
            past_key_values=past_key_values,
        )

    def _sample_forward(
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
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ):
        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if self._prefill:
            if self.dynamic_input:
                self._llm_model.set_input_sequence_length(seq_length)
        else:
            self._llm_model.set_input_sequence_length(1)

        out = self._xh_orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        self._past_seq_length += seq_length
        if self.prefill:
            self.prefill = False
        return out


def build_qwen3_tts_talker_hmonnx_hf_compatible(
    talker: Qwen3TTSTalkerForConditionalGeneration,
    llm_model,
) -> Qwen3TTSTalkerForConditionalGeneration:
    if Qwen3TTSTalkerForConditionalGeneration not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                Qwen3TTSTalkerForConditionalGeneration: "Qwen3TTSTalkerForConditionalGeneration",
            },
            Qwen3TTSTalkerHMONNXHFCompatible,
        )

    LLM_COMPATIBLE_MODULES.convert(talker, llm_model=llm_model)
    assert isinstance(talker, Qwen3TTSTalkerHMONNXHFCompatible)
    return talker


@MODELS.register_module()
class Qwen3TTSSpeechTokenizerInference(HMONNXModel):
    def __init__(self, model_cfg: str) -> None:
        import json

        with open(model_cfg, "r") as f:
            cfg = json.load(f)
        model_dir = Path(model_cfg).parent
        hmonnx_file = str(model_dir / cfg["hmonnx"])
        decode_padding_shapes_file = model_dir / cfg["decode_padding_shapes"]
        with open(decode_padding_shapes_file, "r") as f:
            self.decode_padding_shapes = json.load(f)
        super().__init__(hmonnx_file)

    def forward(self, *inputs) -> Tensor:
        assert self.session is not None, "ONNX session is not initialized!"
        input_name = self.session.get_input_names()[0]
        input_info = self.session.get_input(input_name)
        net_input_seq_len = input_info.shape[2]
        codes = inputs[0]
        input_seq_len = codes.shape[2]
        logger = get_root_logger()
        logger.info(f"SpeechTokenizerInference: input_seq_len={input_seq_len}, net_input_seq_len={net_input_seq_len}")
        b, c, seq = codes.shape
        output_shape = self.decode_padding_shapes[f"{b}_{c}_{seq}"]
        input_codes = torch.nn.functional.pad(codes, (0, net_input_seq_len - input_seq_len))
        input_codes = input_codes.to(torch.int32)
        out = super().forward(input_codes)
        out = out[:, :, : output_shape[2]]
        return out


class Qwen3TTSSpeechTokenizerHMONNXHFCompatible(DynamicModule):
    def _setup(self, hmonnx_model: LLMONNXModel):
        self.hmonnx_model = hmonnx_model
        del self.pre_transformer
        del self.quantizer
        del self.upsample
        del self.decoder
        return self

    def forward(self, codes):
        return self.hmonnx_model.forward(codes)


def build_qwen3_tts_speech_tokenizer_hmonnx_hf_compatible(
    decoder,
    hmonnx_model,
):
    decoder_type = type(decoder)
    if decoder_type not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                decoder_type: decoder_type.__name__,
            },
            Qwen3TTSSpeechTokenizerHMONNXHFCompatible,
        )

    LLM_COMPATIBLE_MODULES.convert(decoder, hmonnx_model=hmonnx_model)
    assert isinstance(decoder, Qwen3TTSSpeechTokenizerHMONNXHFCompatible)
    return decoder


@MODELS.register_module()
class Qwen3TTSHMONNXInference:
    def __init__(
        self, hf_model: str, text_projection: dict, code_predictor: dict, talker: dict, speech_tokenizer: dict
    ) -> None:
        self.hf_model_dir = hf_model
        self.native_model = self.get_hf_model(device_map="cpu", dtype=torch.float16)
        self.text_projection: Qwen3TTSTextProjectionInference = MODELS.build(text_projection)
        assert isinstance(self.text_projection, Qwen3TTSTextProjectionInference)
        self.text_projection.session.initialize()
        self.native_model.model.talker.text_projection = self.text_projection
        self.code_predictor: Qwen3TTSCodePredictorInference = MODELS.build(code_predictor)

        build_qwen3_tts_code_predictor_hmonnx_hf_compatible(
            self.native_model.model.talker.code_predictor, self.code_predictor
        )
        self.talker = MODELS.build(talker)
        build_qwen3_tts_talker_hmonnx_hf_compatible(self.native_model.model.talker, self.talker)

        self.speech_tokenizer = MODELS.build(speech_tokenizer)
        self.speech_tokenizer.session.initialize()
        build_qwen3_tts_speech_tokenizer_hmonnx_hf_compatible(
            self.native_model.model.speech_tokenizer.model.decoder, self.speech_tokenizer
        )

    def to(self, *args, **kwargs) -> "Qwen3TTSHMONNXInference":
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.

        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self.native_model.to(device)
            self.text_projection.to(device)
            self.code_predictor.to(device)
            self.code_predictor.set_exec_device(device)
            self.talker.to(device)
            self.talker.set_exec_device(device)
            self.speech_tokenizer.to(device)
            return self
        return self

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

    def generate_voice_design(self, text: str, language: str, instruct: str):
        # 获取文本投影的输入
        wavs, sr = self.native_model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
        )

        return wavs, sr

    def generate_custom_voice(self, text: str, language: str, speaker: str, **kwargs):
        wavs, sr = self.native_model.generate_custom_voice(
            text=text, language=language, speaker=speaker, **kwargs
        )
        return wavs, sr

    def generate_voice_clone(self, text: str, language: str, ref_audio: str, ref_text: str, **kwargs):
        wavs, sr = self.native_model.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio, ref_text=ref_text, **kwargs
        )
        return wavs, sr
