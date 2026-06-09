# Unified Qwen3-TTS Talker export script
# Supports the 1.7B-VoiceDesign / 0.6B-CustomVoice / 0.6B-Base variants
# Pick a variant with --variant; component structure comes from --config
import argparse
import json
import time
from pathlib import Path
from typing import cast

import soundfile as sf
import torch

from xhquant.api import Config, ConfigDict, PrecisionMode, ptq_quantize, set_random_seed
from xhquant.utils.time_profiler import time_profiler
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.qwen3_tts import XHQwen3TTSModel, XHQwen3TTSTalker, build_qwen3_tts_talker_hf_compatible


# ---------------------------------------------------------------------------
# Generation helper: pick the generate_* method by cfg.tts_mode
# ---------------------------------------------------------------------------
_DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def _run_generate(hf_model, cfg):
    """Call the generate_* method matching cfg.tts_mode; returns (wavs, sr)."""
    mode = getattr(cfg, "tts_mode", "custom_voice")
    text = getattr(cfg, "tts_text", _DEFAULT_TEXT)
    if mode == "voice_design":
        return hf_model.generate_voice_design(
            text=text, language="Chinese",
            instruct=getattr(cfg, "tts_instruct", ""),
        )
    elif mode == "voice_clone":
        ref_audio = getattr(cfg, "ref_audio", "/tmp/clone_1.wav")
        assert Path(ref_audio).exists(), (
            f"missing reference audio {ref_audio}; download clone_1.wav first (see README)"
        )
        return hf_model.generate_voice_clone(
            text=text, language="Chinese",
            ref_audio=ref_audio,
            ref_text=getattr(cfg, "ref_text", ""),
        )
    else:  # custom_voice (default)
        return hf_model.generate_custom_voice(
            text=text, language="Chinese",
            speaker=getattr(cfg, "tts_speaker", "vivian"),
        )


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    logger,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")
    torch.cuda.empty_cache()
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")
    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _export_impl(cfg: Config, args: argparse.Namespace):
    dtype = getattr(torch, cfg.dtype)
    logger = get_root_logger()
    work_dir = cfg.work_dir
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3TTSTalker)
    exec_device = cfg.exec_device
    config_file = cfg.config_file
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["model_name"] = cfg_name
    meta_info["wrap_cfg"] = cfg.model.wrap_cfg.to_dict()
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    # get the original float model
    hf_model = xh_model.get_hf_model(device_map="cpu", dtype=torch.float16)
    assert isinstance(hf_model, XHQwen3TTSModel)
    hf_model = cast(XHQwen3TTSModel, hf_model)
    feature_dim = None

    def _hook(self, args, kwargs):
        nonlocal feature_dim
        inputs_embeds = kwargs.get("inputs_embeds", None)
        feature_dim = inputs_embeds.shape[-1] if inputs_embeds is not None else None
        logger.info(f"feature_dim: {feature_dim}")
        logger.info(f"inputs_embeds: {inputs_embeds.shape}")
        raise RuntimeError("Stop forward after getting feature_dim for export")

    talker_hook = hf_model.model.talker.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        _run_generate(hf_model, cfg)
    except RuntimeError:
        pass
    talker_hook.remove()
    assert feature_dim is not None, "Failed to get feature_dim from the model, export cannot proceed without it."

    # token/text embeddings must be saved for later inference
    token_embedding = hf_model.model.talker.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    text_embedding = hf_model.model.talker.get_text_embeddings()
    text_embedding_file = Path(cfg.work_dir) / "text_embedding.pt"
    torch.save(text_embedding.state_dict(), str(text_embedding_file))
    meta_info.text_embedding_file = str(text_embedding_file.relative_to(cfg.work_dir))

    xh_model.init_wrap_model(hf_model)
    xh_model.to(dtype=dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    del hf_model

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    data_batch = {
        "inputs_embeds": torch.randn(1, input_sequence_length, feature_dim),
        "past_seq_length": 0,
    }

    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)
    torch.cuda.empty_cache()
    xh_model.change_eval_type(eval_type=EvalModelType.FRONTEND)

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)
    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()

    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (list, tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args
    with time_profiler() as t:
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [exec_device],
            auto_release_unused_parameters=True,
        )
        logger.info(f"PTQ Quantize time: {t():.04f} s")
    logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    logger.info("*************** Start exporting prefill model ***************")
    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_file = xhmodel_export_onnx(
        xh_model, data_batch, str(prefill_onnx_dir), f"{cfg_name}_prefill", logger,
    )
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")

    if args.golden:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _golden import run_hmonnx_golden
        xh_model.reset_kvcache()
        prefill_golden_dir = Path(cfg.work_dir) / "golden" / f"{cfg_name}_prefill"
        run_hmonnx_golden(prefill_onnx_file, prefill_golden_dir,
                          xh_model.prepare_inputs(data_batch), args.golden_device)
        meta_info.prefill_golden_dir = str(prefill_golden_dir.relative_to(cfg.work_dir))
        json.dump(meta_info, open(meta_file, "w"), indent=4)

    # export the decode model
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype)
    data_batch = {"inputs_embeds": torch.randn(1, 1, feature_dim), "past_seq_length": 0}
    torch.cuda.empty_cache()
    xh_model.set_input_sequence_length(1)
    logger.info("*************** Start exporting decode model ***************")
    xh_model.to("cpu")
    decode_onnx_file = xhmodel_export_onnx(
        xh_model, data_batch, str(decode_onnx_dir), f"{cfg_name}_decode", logger,
    )
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"save decode onnx model to {decode_onnx_file}")

    if args.golden:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _golden import run_hmonnx_golden
        xh_model.reset_kvcache()
        decode_golden_dir = Path(cfg.work_dir) / "golden" / f"{cfg_name}_decode"
        run_hmonnx_golden(decode_onnx_file, decode_golden_dir,
                          xh_model.prepare_inputs(data_batch), args.golden_device)
        meta_info.decode_golden_dir = str(decode_golden_dir.relative_to(cfg.work_dir))

    json.dump(meta_info, open(meta_file, "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if getattr(args, "variant", None):
        from config.llm._components import apply_variant
        apply_variant(cfg, args.variant)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem  # noqa: F841  (module-level global read by _export_impl)
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"

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
        default="./config/llm/qwen3_tts_12hz_talker_2k_xh2a.py",
        help="unified talker component config; pick variant with --variant",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--golden", action="store_true", help="export hmonnx golden")
    parser.add_argument("--golden-device", type=str, default="cuda")
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"], default=None,
                        help="TTS variant; injects hf_model/tts_mode into the parsed config")
    parser.add_argument("--name", type=str, default=None,
                        help="explicit work_dir name & product prefix; defaults to config stem")

    args = parser.parse_args()
    cfg_name = args.name if args.name else Path(args.config).stem
    args.work_dir = str(Path("./work_dirs") / cfg_name)

    if Path(args.work_dir).exists():
        import shutil
        from loguru import logger
        logger.info(f"Work dir {args.work_dir} already exists, removing it...")
        shutil.rmtree(args.work_dir)
    main(args)


