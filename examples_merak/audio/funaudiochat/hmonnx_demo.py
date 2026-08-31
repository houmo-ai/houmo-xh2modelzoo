from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_other_model.models.funaudiochat.constant import (  # noqa: E402
    AUDIO_TEMPLATE,
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    SPOKEN_S2M_PROMPT,
)
from xhmodel_merak.xh_other_model.models.funaudiochat._model import (  # noqa: E402
    FunAudioChatHMONNXForConditionalGeneration,
)
from xhmodel_merak.xh_other_model.models.funaudiochat.register import register_funaudiochat  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FunAudioChat using exported HMONNX graphs")
    parser.add_argument("--model-dir", required=True, help="Original HF FunAudioChat model directory")
    parser.add_argument("--export-dir", required=True, help="Directory containing export_meta_info.json")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

    register_funaudiochat()
    export_dir = Path(args.export_dir).expanduser().resolve()
    meta_path = export_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(args.model_dir)
    config.audio_config.crq_transformer_config["torch_dtype"] = "float16"
    float_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype="auto",
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    model = FunAudioChatHMONNXForConditionalGeneration.from_float_model(float_model)
    model.load_hmonnx_runtime(
        str(export_dir),
        device=args.device,
        execution_device=args.execution_device,
        input_sequence_length=int(meta.get("input_sequence_length", 256)),
    )
    model.enable_hmonnx_encoder()
    model.enable_hmonnx_decoder()
    model.enable_hmonnx_language()

    processor = AutoProcessor.from_pretrained(args.model_dir)
    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs["text_greedy"] = True
    model.sp_gen_kwargs.update(sp_gen_kwargs)
    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs["max_new_tokens"] = args.max_new_tokens

    audio = [librosa.load(args.audio, sr=16000)[0]]
    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False)
    inputs = inputs.to(model.device)
    generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    text_ids = generate_ids[:, inputs.input_ids.shape[1] :]
    print("generate_text:", processor.decode(text_ids[0], skip_special_tokens=True))
    if audio_ids is not None:
        print("generate_audio_tokens:", int(audio_ids.shape[-1]))


if __name__ == "__main__":
    main()
