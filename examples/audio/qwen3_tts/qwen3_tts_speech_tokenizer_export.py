# Example: export a Qwen3-TTS sub-model to XH2a / HMONNX
import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import cast

import soundfile as sf
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2DecoderTransformerModel
from tqdm import tqdm
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from xhquant.api import (
    Config,
    ConfigDict,
    FrontendType,
    Hook,
    PrecisionMode,
    QTensor,
    convert_onnx_to_hmonnx,
    ptq_quantize,
    set_random_seed,
)
from xhquant.patch import RewriterContext
from xhquant.api import Config
from xhquant.export.onnx.transforms import hmonnx_transforms
import onnx
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models import qwen3_tts as qwen3_tts_models  # noqa: F401  # noqa: F401
from xh_model_zoo.xh_llm.models.qwen3_tts import XHQwen3TTSModel, XHQwen3TTSTalker, build_qwen3_tts_talker_hf_compatible


class XHQwen3TTSTokenizerV2DecoderTransformerModel(Qwen3TTSTokenizerV2DecoderTransformerModel):
    def _setup(self, chunk_size=300):
        self.chunk_size = chunk_size
        cache_position = torch.arange(0, self.chunk_size)
        attention_mask = None
        inputs_embeds = None
        past_key_values = None
        position_ids = cache_position.unsqueeze(0)
        inputs_embeds = torch.randn(1, self.chunk_size, self.config.hidden_size)
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks

        full_attention = create_causal_mask(**mask_kwargs)
        self.register_buffer("full_attention", full_attention, persistent=False)

        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            sliding_attention = create_sliding_window_causal_mask(**mask_kwargs)
            self.register_buffer("sliding_attention", sliding_attention, persistent=False)

        # self.cache_position = torch.arange(0, self.chunk_size)
        # self.position_ids = self.cache_position.unsqueeze(0)
        # self.register_buffer("cache_position", self.cache_position, persistent=False)
        # self.register_buffer("position_ids", self.position_ids, persistent=False)
        hidden_states = torch.randn(1, self.chunk_size, self.config.hidden_size, dtype=torch.float16)
        if torch.cuda.is_available():
            hidden_states = hidden_states.cuda()
        cos, sin = self.rotary_emb(hidden_states, position_ids.to(hidden_states.device))
        cos = cos.cpu()
        sin = sin.cpu()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        # input_ids=None,
        # attention_mask=None,
        # position_ids=None,
        # past_key_values=None,
        inputs_embeds=None,
        # use_cache=None,
        # cache_position=None,
        # **kwargs,
    ) -> BaseModelOutputWithPast:
        # if input_ids is not None:
        #     raise ValueError("input_ids is not expected")
        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.input_proj(inputs_embeds)
        cache_position = None
        position_ids = None
        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # cache_position = torch.arange(
            #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            # )
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask_mapping = {
            "full_attention": self.full_attention,
            "sliding_attention": self.sliding_attention if self.has_sliding_layers else None,
        }
        # # It may already have been prepared by e.g. `generate`
        # if not isinstance(causal_mask_mapping := attention_mask, dict):
        #     # Prepare mask arguments
        #     mask_kwargs = {
        #         "config": self.config,
        #         "input_embeds": inputs_embeds,
        #         "attention_mask": attention_mask,
        #         "cache_position": cache_position,
        #         "past_key_values": past_key_values,
        #         "position_ids": position_ids,
        #     }
        #     # Create the masks
        #     causal_mask_mapping = {
        #         "full_attention": create_causal_mask(**mask_kwargs),
        #     }
        #     # The sliding window alternating layers are not always activated depending on the config
        #     if self.has_sliding_layers:
        #         causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (self.cos_cached, self.sin_cached)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                # past_key_values=past_key_values,
                # use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                # **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            # past_key_values=past_key_values if use_cache else None,
        )




# ---------------------------------------------------------------------------
# Generation helper: pick the generate_* method by cfg.tts_mode
# ---------------------------------------------------------------------------
_DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def _run_generate(hf_model, cfg):
    mode = getattr(cfg, "tts_mode", "custom_voice")
    text = getattr(cfg, "tts_text", _DEFAULT_TEXT)
    if mode == "voice_design":
        return hf_model.generate_voice_design(
            text=text, language="Chinese",
            instruct=getattr(cfg, "tts_instruct", ""),
        )
    elif mode == "voice_clone":
        ref_audio = getattr(cfg, "ref_audio", "/tmp/clone_1.wav")
        assert Path(ref_audio).exists(), f"missing reference audio {ref_audio}"
        return hf_model.generate_voice_clone(
            text=text, language="Chinese",
            ref_audio=ref_audio, ref_text=getattr(cfg, "ref_text", ""),
        )
    else:
        return hf_model.generate_custom_voice(
            text=text, language="Chinese",
            speaker=getattr(cfg, "tts_speaker", "vivian"),
        )

def _export_impl(cfg: Config, args: argparse.Namespace):
    device = cfg.exec_device
    dtype = getattr(torch, cfg.dtype)
    logger = get_root_logger()
    work_dir = cfg.work_dir
    model_dir = cfg.hf_model_dir
    hf_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda",
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    hf_model = cast(Qwen3TTSModel, hf_model)
    config_file = cfg.config_file
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    target_device = cfg.target_device
    meta_info["model_name"] = cfg_name
    meta_info["target_device"] = target_device
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    # register a forward hook to capture the input shape
    # instruct_ids,   # [1, 40, 2048]   # shared across requests if instruct is unchanged
    # [[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]] # only needs one global inference
    # input_id[:, :3]: <|im_start|>assistant\n   # only needs one global inference
    # input_id[:, 3:4]
    # input_id[:, 3:-5]
    # three fixed inputs, two dynamic inputs
    input_shape = None
    input_dtype = None

    def _hook(module, inputs):
        logger.info(f"inputs[0].shape: {inputs[0].shape}")
        nonlocal input_shape
        nonlocal input_dtype
        input_shape = list(inputs[0].shape)
        input_dtype = inputs[0].dtype
        return inputs

    decode_hook = hf_model.model.speech_tokenizer.model.decoder.register_forward_pre_hook(_hook)
    pre_transformer = hf_model.model.speech_tokenizer.model.decoder.pre_transformer
    pre_transformer.config._attn_implementation = "eager"
    wavs, sr = _run_generate(hf_model, cfg)
    out_file = Path(work_dir) / f"output_{getattr(cfg, 'tts_mode', 'custom_voice')}.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")

    decode_hook.remove()

    chunk_size = 300

    gt_shapes = {}
    for seq_len_in in tqdm(range(1, 301), desc="decoder padding shape validation"):
        dummy_input_shape = list(input_shape)
        dummy_input_shape[-1] = seq_len_in
        dummy_input = torch.randint(0, 100, dummy_input_shape, device=device, dtype=input_dtype)

        wav_out = hf_model.model.speech_tokenizer.model.decoder(dummy_input)
        actual_output_shape = list(wav_out.shape)
        input_shape_str = f"{'_'.join(map(str, dummy_input_shape))}"
        gt_shapes[input_shape_str] = actual_output_shape

        # Clear GPU cache to prevent OOM
        del dummy_input, wav_out
        torch.cuda.empty_cache()
    json.dump(gt_shapes, open(str(Path(work_dir) / "decode_padding_shapes.json"), "w"), indent=2)
    meta_info["decode_padding_shapes"] = str(Path("decode_padding_shapes.json"))

    # export speech_tokenizer
    input_shape[-1] = chunk_size
    example_input = torch.randint(0, 100, input_shape, device=device, dtype=input_dtype)
    onnx_dir = Path(work_dir) / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)

    speech_tokenizer_onnx_file = str(Path(work_dir) / "onnx" / "speech_tokenizer.onnx")
    pre_transformer = hf_model.model.speech_tokenizer.model.decoder.pre_transformer
    pre_transformer.__class__ = XHQwen3TTSTokenizerV2DecoderTransformerModel
    pre_transformer = cast(XHQwen3TTSTokenizerV2DecoderTransformerModel, pre_transformer)
    pre_transformer._setup()
    torch.onnx.export(
        hf_model.model.speech_tokenizer.model.decoder.float().cpu(),
        example_input.to(torch.int32).cpu(),
        speech_tokenizer_onnx_file,
        input_names=["codes"],
        output_names=["wav"],
        dynamo=True,
    )

    # speech_tokenizer_preprocess_onnx_file = str(Path(work_dir) / "onnx" / "speech_tokenizer_preprocess.onnx")

    # class DecoderPreprocess(nn.Module):
    #     def __init__(self, decoder):
    #         super().__init__()
    #         self._decoder = decoder

    #     def forward(self, codes):
    #         hidden = self._decoder.quantizer.decode(codes)
    #         hidden = self._decoder.pre_conv(hidden).transpose(1, 2)
    #         return hidden

    # preprocess = DecoderPreprocess(hf_model.model.speech_tokenizer.model.decoder)
    # torch.onnx.export(
    #     preprocess.float().cpu(),
    #     example_input.cpu(),
    #     speech_tokenizer_preprocess_onnx_file,
    #     input_names=["codes"],
    #     output_names=["wav"],
    #     dynamo=True,
    # )

    # class DecoderPostprocess(nn.Module):
    #     def __init__(self, decoder):
    #         super().__init__()
    #         self._decoder = decoder

    #     def forward(self, hidden):
    #         hidden = hidden.permute(0, 2, 1)
    #         for blocks in self._decoder.upsample:
    #             for block in blocks:
    #                 hidden = block(hidden)
    #         wav = hidden
    #         for block in self._decoder.decoder:
    #             wav = block(wav)
    #         return wav.clamp(min=-1, max=1)

    # postprocess = DecoderPostprocess(hf_model.model.speech_tokenizer.model.decoder)
    # example_input = torch.randn(1, 300, 1024)
    # speech_tokenizer_postprocess_onnx_file = str(Path(work_dir) / "onnx" / "speech_tokenizer_postprocess.onnx")
    # torch.onnx.export(
    #     postprocess.float().cpu(),
    #     example_input.float().cpu(),
    #     speech_tokenizer_postprocess_onnx_file,
    #     input_names=["hidden"],
    #     output_names=["wav"],
    #     dynamo=True,
    # )
    # pre_transformer = hf_model.model.speech_tokenizer.model.decoder.pre_transformer
    # pre_transformer.__class__ = XHQwen3TTSTokenizerV2DecoderTransformerModel
    # pre_transformer = cast(XHQwen3TTSTokenizerV2DecoderTransformerModel, pre_transformer)
    # pre_transformer._setup()
    # example_input = torch.randn(1, 300, 1024)
    # pre_transformer.eval()
    # speech_tokenizer_pre_transformer_onnx_file = str(Path(work_dir) / "onnx" / "speech_tokenizer_pre_transformer.onnx")
    # pre_transformer.config._attn_implementation = "eager"
    # torch.onnx.export(
    #     pre_transformer.float().cpu(),
    #     example_input.cpu(),
    #     speech_tokenizer_pre_transformer_onnx_file,
    #     input_names=["hidden"],
    #     output_names=["pre_transformer_hidden"],
    #     dynamo=True,
    # )

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    logger.info(f"speech_tokenizer exported to {speech_tokenizer_onnx_file}")

    # Apply HMONNX transforms to convert unsupported ops like SplitToSequence/SequenceAt
    onnx_model = onnx.load(speech_tokenizer_onnx_file)
    hmonnx_transforms(onnx_model)
    onnx.save(onnx_model, speech_tokenizer_onnx_file)
    logger.info("Applied HMONNX transforms to speech_tokenizer ONNX model")

    hmonnx_dir = Path(work_dir) / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    speech_tokenizer_hmonnx_file = str(hmonnx_dir / f"speech_tokenizer_{target_device}.onnx")
    convert_onnx_to_hmonnx(
        speech_tokenizer_onnx_file,
        [example_input.to(torch.int32).cpu()],
        target_device,
        speech_tokenizer_hmonnx_file,
    )
    if args.golden:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _golden import run_hmonnx_golden

        golden_dir = Path(work_dir) / "golden" / "speech_tokenizer"
        run_hmonnx_golden(
            speech_tokenizer_hmonnx_file,
            golden_dir,
            (example_input.to(torch.int32).cpu(),),
            args.golden_device,
        )
        meta_info["golden_dir"] = str(golden_dir.relative_to(work_dir))

    meta_info["hmonnx"] = str(Path("hmonnx") / f"speech_tokenizer_{target_device}.onnx")
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)
    logger.info(f"speech_tokenizer converted to hmonnx and saved to {speech_tokenizer_hmonnx_file}")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if getattr(args, "variant", None):
        from config.llm._components import apply_variant
        apply_variant(cfg, args.variant)
    if getattr(args, "hf_model_dir", None):
        from config.llm._components import apply_hf_model_dir_override
        apply_hf_model_dir_override(cfg, args.hf_model_dir)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # exec device: data is moved here when running a module/op

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)
    cfg.config_file = config_file

    _export_impl(cfg, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_speech_tokenizer_xh2a.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")
    parser.add_argument("--golden", action="store_true", help="export hmonnx golden")
    parser.add_argument(
        "--golden-device", type=str, default="cuda", help="device for golden inference"
    )
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_customvoice", "1_7B_voicedesign"], default=None,
                        help="TTS variant; injects hf_model/tts_mode into the parsed config")
    parser.add_argument("--hf-model-dir", type=str, default=None,
                        help="override HF model directory, e.g. a Hugging Face cache snapshot")
    parser.add_argument("--name", type=str, default=None,
                        help="explicit work_dir name & product prefix; defaults to config stem")

    args = parser.parse_args()
    cfg_name = args.name if args.name else Path(args.config).stem
    cfg_name = f"{cfg_name}"
    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = args.work_dir
    if Path(work_dir).exists():
        import shutil

        from loguru import logger

        logger.info(f"Work dir {work_dir} already exists, removing it...")
        shutil.rmtree(work_dir)
    main(args)
