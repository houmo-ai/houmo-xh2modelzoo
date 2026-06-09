# Example: export a Qwen3-TTS sub-model to XH2a / HMONNX
import argparse
import json
import time
from pathlib import Path
from typing import cast

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from xhquant.api import (
    Config,
    ConfigDict,
    convert_onnx_to_hmonnx,
    set_random_seed,
)
from xhquant.api import Config
from xh_model_zoo.api import get_root_logger, xhquant_llm_init




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
    meta_info["act"] = type(hf_model.model.talker.text_projection.act_fn).__name__
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
    feature_dim = 2048

    def _hook(module, inputs):
        logger.info(f"inputs[0].shape: {inputs[0].shape}")
        nonlocal feature_dim
        feature_dim = inputs[0].shape[-1]
        return inputs

    text_projection_hook = hf_model.model.talker.text_projection.register_forward_pre_hook(_hook)

    wavs, sr = _run_generate(hf_model, cfg)
    out_file = Path(work_dir) / f"output_{getattr(cfg, 'tts_mode', 'custom_voice')}.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")

    text_projection_hook.remove()
    # export text_projection
    example_input = torch.randn(1, 1, feature_dim, device=device, dtype=dtype)
    onnx_dir = Path(work_dir) / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    text_projection_onnx_file = Path(work_dir) / "onnx" / "text_projection.onnx"
    torch.onnx.export(
        hf_model.model.talker.text_projection.float().cpu(),
        (example_input.float().cpu(),),
        text_projection_onnx_file,
        input_names=["inputs_embeds"],
        output_names=["outputs"],
    )
    logger.info(f"text_projection exported to {text_projection_onnx_file}")

    hmonnx_dir = Path(work_dir) / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    text_projection_hmonnx_file = str(hmonnx_dir / f"text_projection_{target_device}.onnx")
    convert_onnx_to_hmonnx(
        text_projection_onnx_file,
        [example_input.float().cpu()],
        target_device,
        text_projection_hmonnx_file,
    )
    if args.golden:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _golden import run_hmonnx_golden

        golden_dir = Path(work_dir) / "golden" / "text_projection"
        run_hmonnx_golden(
            text_projection_hmonnx_file,
            golden_dir,
            (example_input.float().cpu(),),
            args.golden_device,
        )
        meta_info["golden_dir"] = str(golden_dir.relative_to(work_dir))

    meta_info["hmonnx"] = str(Path(text_projection_hmonnx_file).relative_to(Path(meta_file).parent))
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)
    logger.info(f"text_projection converted to hmonnx and saved to {text_projection_hmonnx_file}")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if getattr(args, "variant", None):
        from config.llm._components import apply_variant
        apply_variant(cfg, args.variant)
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
        default="./config/llm/qwen3_tts_12hz_text_projection_xh2a.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")
    parser.add_argument("--golden", action="store_true", help="export hmonnx golden")
    parser.add_argument(
        "--golden-device", type=str, default="cuda", help="device for golden inference"
    )
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"], default=None,
                        help="TTS variant; injects hf_model/tts_mode into the parsed config")
    parser.add_argument("--name", type=str, default=None,
                        help="explicit work_dir name & product prefix; defaults to config stem")

    args = parser.parse_args()
    cfg_name = args.name if args.name else Path(args.config).stem
    cfg_name = f"{cfg_name}"
    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = args.work_dir
    # from loguru import logger

    # if Path(work_dir).exists():
    #     from loguru import logger

    #     logger.warning(f"{work_dir} already exists, please remove it first")
    #     exit(1)
    main(args)
