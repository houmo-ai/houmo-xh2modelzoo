# Copyright 2025 HOUMO AI
#
# File: funaudiochat_xh2a_hmonnx_test.py
# Description:
#   End-to-end generation validation for FunAudioChat HMONNX replacement.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
from xhquant.api import xhquant_init

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.xh_llm.models.funaudiochat._model import (  # noqa: E402
    FunAudioChatHMONNXForConditionalGeneration,
)
from xh_model_zoo.xh_llm.models.funaudiochat.constant import (  # noqa: E402
    AUDIO_TEMPLATE,
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    SPOKEN_S2M_PROMPT,
)
from xh_model_zoo.xh_llm.models.funaudiochat.cosyvoice_detokenizer import (  # noqa: E402
    get_audio_detokenizer,
    token2wav,
)
from xh_model_zoo.xh_llm.models.funaudiochat.register import register_funaudiochat  # noqa: E402


def load_runtime_meta(work_dir: Path) -> dict:
    export_meta_path = work_dir / "export_meta.json"
    if export_meta_path.exists():
        export_meta = json.loads(export_meta_path.read_text(encoding="utf-8"))
        qwen_meta_path = work_dir / "qwen3" / "meta.json"
        if "qwen3_meta" not in export_meta and qwen_meta_path.exists():
            export_meta["qwen3_meta"] = str(qwen_meta_path.relative_to(work_dir))
        return export_meta

    audio_meta_path = work_dir / "audio_encoder" / "meta.json"
    if not audio_meta_path.exists():
        raise FileNotFoundError(f"missing {export_meta_path} and {audio_meta_path}")

    audio_meta = json.loads(audio_meta_path.read_text(encoding="utf-8"))
    quant_type = audio_meta.get("audio_quant_type", "w8a8h1_sefp")
    return {
        "hf_model_path": "/data01/datasets/Fun-Audio-Chat-8B",
        "audio_encoder_hmonnx": f"audio_encoder/hmonnx/funaudiochat_audio_encoder_{quant_type}.onnx",
        "audio_encoder_meta": "audio_encoder/meta.json",
        "input_sequence_length": 256,
    }


def build_inputs(processor, audio_path: str, system_prompt: str, device: torch.device):
    audio = [librosa.load(audio_path, sr=16000)[0]]
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(device)


def save_waveform(audio_ids: torch.Tensor, output_path: Path, detokenizer_model_path: str):
    detokenizer = get_audio_detokenizer(model_path=detokenizer_model_path)
    token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))
    waveform = token2wav(detokenizer, token_for_cosyvoice, embedding=None, token_hop_len=25 * 30, pre_lookahead_len=3).cpu().squeeze().detach().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, detokenizer.sample_rate)


def decode_text(processor, generated_ids: torch.Tensor, prompt_length: int) -> str:
    trimmed = generated_ids[:, prompt_length:]
    return processor.decode(trimmed[0], skip_special_tokens=True)


@torch.no_grad()
def run_generate_float(
    model,
    processor,
    inputs,
    *,
    max_new_tokens: int,
):
    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs["text_greedy"] = True
    sp_gen_kwargs["only_crq_sampling"] = True
    sp_gen_kwargs["disable_speech"] = False
    model.sp_gen_kwargs.update(sp_gen_kwargs)

    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["do_sample"] = False
    gen_kwargs["max_new_tokens"] = max_new_tokens

    generated_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    generated_text = decode_text(processor, generated_ids, inputs.input_ids.shape[1])
    return {
        "generated_ids": generated_ids.cpu(),
        "audio_ids": audio_ids.cpu(),
        "generated_text": generated_text,
    }


@torch.no_grad()
def run_generate(
    model: FunAudioChatHMONNXForConditionalGeneration,
    processor,
    inputs,
    *,
    use_hmonnx_encoder: bool,
    use_hmonnx_language: bool,
    use_hmonnx_decoder: bool,
    max_new_tokens: int,
):
    model.enable_hmonnx_encoder(use_hmonnx_encoder)
    model.enable_hmonnx_language(use_hmonnx_language)
    model.enable_hmonnx_decoder(use_hmonnx_decoder)

    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs["text_greedy"] = True
    sp_gen_kwargs["only_crq_sampling"] = True
    sp_gen_kwargs["disable_speech"] = False
    model.sp_gen_kwargs.update(sp_gen_kwargs)

    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["do_sample"] = False
    gen_kwargs["max_new_tokens"] = max_new_tokens

    generated_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    generated_text = decode_text(processor, generated_ids, inputs.input_ids.shape[1])
    return {
        "generated_ids": generated_ids.cpu(),
        "audio_ids": audio_ids.cpu(),
        "generated_text": generated_text,
    }


def main(args):
    register_funaudiochat()
    xhquant_init(None, args.debug)
    work_dir = Path(args.work_dir)
    export_meta = load_runtime_meta(work_dir)

    hf_model_path = export_meta["hf_model_path"]
    config = AutoConfig.from_pretrained(hf_model_path)
    processor = AutoProcessor.from_pretrained(hf_model_path)
    config.audio_config.crq_transformer_config["torch_dtype"] = torch.float16
    float_model = AutoModelForSeq2SeqLM.from_pretrained(
        hf_model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    inputs = build_inputs(processor, args.audio, args.system_prompt, torch.device(args.device))

    # float_outputs = run_generate_float(
    #     float_model,
    #     processor,
    #     inputs,
    #     max_new_tokens=args.max_new_tokens,
    # )

    model = FunAudioChatHMONNXForConditionalGeneration.from_float_model(float_model)
    model = model.cuda()
    model.eval()
    model.load_hmonnx_runtime(
        work_dir,
        device=args.device,
        execution_device=args.execution_device,
        input_sequence_length=int(export_meta.get("input_sequence_length", args.input_sequence_length)),
    )
    model.to(torch.device(args.device))

    # pt_outputs = run_generate(
    #     model,
    #     processor,
    #     inputs,
    #     use_hmonnx_encoder=False,
    #     use_hmonnx_language=False,
    #     use_hmonnx_decoder=False,
    #     max_new_tokens=args.max_new_tokens,
    # )
    pt_outputs = None
    hm_encoder_outputs = None
    
    # hm_encoder_outputs = run_generate(
    #     model,
    #     processor,
    #     inputs,
    #     use_hmonnx_encoder=True,
    #     use_hmonnx_language=False,
    #     use_hmonnx_decoder=False,
    #     max_new_tokens=args.max_new_tokens,
    # )
    hm_language_outputs = None
    # if model.hmonnx_language_prefill_session is not None and model.hmonnx_language_decode_session is not None:
    #     hm_language_outputs = run_generate(
    #         model,
    #         processor,
    #         inputs,
    #         use_hmonnx_encoder=True,
    #         use_hmonnx_language=True,
    #         use_hmonnx_decoder=False,
    #         max_new_tokens=args.max_new_tokens,
    #     )
    
    hm_full_outputs = None
    if model.hmonnx_decoder_prefill_session is not None and model.hmonnx_decoder_decode_session is not None:
        hm_full_outputs = run_generate(
            model,
            processor,
            inputs,
            use_hmonnx_encoder=True,
            use_hmonnx_language=True,
            use_hmonnx_decoder=True,
            max_new_tokens=args.max_new_tokens,
        )
    
    if hm_encoder_outputs is not None or hm_language_outputs is not None or hm_full_outputs is not None:
        metrics = {
            "hm_encoder_available": hm_encoder_outputs is not None,
            "hm_language_available": hm_language_outputs is not None,
            "hm_full_available": hm_full_outputs is not None,
        }
        if pt_outputs is not None:
            metrics["pt_text"] = pt_outputs["generated_text"]
            metrics["pt_audio_token_length"] = int(pt_outputs["audio_ids"].shape[-1])
        if hm_encoder_outputs is not None:
            metrics["hm_encoder_text"] = hm_encoder_outputs["generated_text"]
            metrics["hm_encoder_audio_token_length"] = int(hm_encoder_outputs["audio_ids"].shape[-1])
            if pt_outputs is not None:
                metrics["hm_encoder_text_equal"] = pt_outputs["generated_text"] == hm_encoder_outputs["generated_text"]
                metrics["hm_encoder_audio_tokens_equal"] = bool(torch.equal(pt_outputs["audio_ids"], hm_encoder_outputs["audio_ids"]))
        if hm_language_outputs is not None:
            metrics["hm_language_text"] = hm_language_outputs["generated_text"]
            metrics["hm_language_audio_token_length"] = int(hm_language_outputs["audio_ids"].shape[-1])
        if hm_full_outputs is not None:
            metrics["hm_full_text"] = hm_full_outputs["generated_text"]
            metrics["hm_full_audio_token_length"] = int(hm_full_outputs["audio_ids"].shape[-1])
    else:
        metrics = {
            # "float_text": float_outputs["generated_text"],
            # "pt_text": pt_outputs["generated_text"],
            # "pt_matches_float_text": pt_outputs["generated_text"] == float_outputs["generated_text"],
            # "pt_matches_float_audio_tokens": bool(torch.equal(pt_outputs["audio_ids"], float_outputs["audio_ids"])),
            # "float_audio_token_length": int(float_outputs["audio_ids"].shape[-1]),
            # "pt_audio_token_length": int(pt_outputs["audio_ids"].shape[-1]),
        }
    
    if hm_full_outputs is not None and pt_outputs is not None:
        metrics.update(
            {
                "hm_full_text_equal": pt_outputs["generated_text"] == hm_full_outputs["generated_text"],
                "hm_full_audio_tokens_equal": bool(torch.equal(pt_outputs["audio_ids"], hm_full_outputs["audio_ids"])),
            }
        )
        
    if hm_language_outputs is not None:
        metrics.update(
            {
                "hm_language_text": hm_language_outputs["generated_text"],
                # "hm_language_text_equal": pt_outputs["generated_text"] == hm_language_outputs["generated_text"],
                # "hm_language_audio_tokens_equal": bool(
                #     torch.equal(pt_outputs["audio_ids"], hm_language_outputs["audio_ids"])
                # ),
                "hm_language_audio_token_length": int(hm_language_outputs["audio_ids"].shape[-1]),
            }
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.save_waveforms:
        out_dir = work_dir / "generate_validation"
        # save_waveform(float_outputs["audio_ids"], out_dir / "float.wav", args.detokenizer_model_path)
        if pt_outputs is not None:
            save_waveform(pt_outputs["audio_ids"], out_dir / "pt.wav", args.detokenizer_model_path)
        if hm_encoder_outputs is not None:
            save_waveform(hm_encoder_outputs["audio_ids"], out_dir / "hm_encoder.wav", args.detokenizer_model_path)
        if hm_language_outputs is not None:
            save_waveform(
                hm_language_outputs["audio_ids"],
                out_dir / "hm_language.wav",
                args.detokenizer_model_path,
            )
        if hm_full_outputs is not None:
            save_waveform(hm_full_outputs["audio_ids"], out_dir / "hm_full.wav", args.detokenizer_model_path)
        print(json.dumps({"wave_dir": str(out_dir)}, ensure_ascii=False, indent=2))

    if args.require_same_tokens:
        if hm_encoder_outputs is not None and pt_outputs is not None and not metrics["hm_encoder_audio_tokens_equal"]:
            raise SystemExit("validation failed: hm encoder audio tokens differ")
        if args.validate_full and hm_full_outputs is not None and pt_outputs is not None and not metrics["hm_full_audio_tokens_equal"]:
            raise SystemExit("validation failed: hm full audio tokens differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default="work_dirs/funaudiochat_xh2a")
    parser.add_argument(
        "--audio",
        type=str,
        default="/data01/home/xuchen/xh2/xh2modelzoo/examples/audio/fun_audio_chat/asr_example_hotword.wav",
    )
    parser.add_argument("--system-prompt", type=str, default=SPOKEN_S2M_PROMPT)
    parser.add_argument("--input-sequence-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execution-device", type=str, default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--require-same-tokens", action="store_true")
    parser.add_argument("--validate-full", action="store_true")
    parser.add_argument("--save-waveforms", action="store_true")
    parser.add_argument(
        "--detokenizer-model-path",
        type=str,
        default="/data01/datasets/Fun-CosyVoice3-0.5B-2512",
    )
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())
