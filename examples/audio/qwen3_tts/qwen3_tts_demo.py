# Example: export a Qwen3-TTS sub-model to XH2a / HMONNX
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, cast

import soundfile as sf
import torch
import torchaudio

from xhquant.api import Config, ConfigDict, PrecisionMode, ptq_quantize, set_random_seed
from xhquant.utils.time_profiler import time_profiler
from xhquant.xhonnxruntime import config as xh_ort_config
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.qwen3_tts import Qwen3TTSHMONNXInference, XHQwen3TTSModel
from qwen_tts import VoiceClonePromptItem
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


DEFAULT_FRONTEND_HMONNX_DIR = "./work_dirs/qwen3_tts_12hz_0_6B_base_frontend_xh2a"


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    logger,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")  # move to CPU for export
    torch.cuda.empty_cache()
    # print_gpu_info(logger)
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info("************* Start Exported Graph *************")
    # logger.info(str(xh_model.exported_model.graph))
    logger.info("************* End Exported Graph *************")
    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")  # move to CPU for export
    torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


# ---------------------------------------------------------------------------
# Generation helper: pick the generate_* method by cfg.tts_mode
# ---------------------------------------------------------------------------
_DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def _resolve_frontend_hmonnx_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    frontend_dir = Path(args.frontend_hmonnx_dir)
    encode_hmonnx = Path(args.speech_tokenizer_encode_hmonnx) if args.speech_tokenizer_encode_hmonnx else (
        frontend_dir / "hmonnx" / "speech_tokenizer_encode_XH2a.onnx"
    )
    speaker_hmonnx = Path(args.speaker_encoder_hmonnx) if args.speaker_encoder_hmonnx else (
        frontend_dir / "hmonnx" / "speaker_encoder_XH2a.onnx"
    )
    if not encode_hmonnx.exists():
        raise FileNotFoundError(f"speech_tokenizer.encode HMONNX not found: {encode_hmonnx}")
    if not speaker_hmonnx.exists():
        raise FileNotFoundError(f"speaker_encoder HMONNX not found: {speaker_hmonnx}")
    return encode_hmonnx, speaker_hmonnx


def _pad_or_trim_1d(x: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_len = min(int(x.numel()), target_len)
    out = torch.zeros(target_len, dtype=torch.float32)
    mask = torch.zeros(target_len, dtype=torch.int32)
    if valid_len > 0:
        out[:valid_len] = x[:valid_len].to(torch.float32)
        mask[:valid_len] = 1
    return out.unsqueeze(0), mask.unsqueeze(0)


def _pad_or_trim_mels(mels: torch.Tensor, target_frames: int, target_dim: int) -> torch.Tensor:
    if mels.shape[-1] != target_dim:
        raise ValueError(f"speaker mels dim mismatch: expected {target_dim}, got {mels.shape[-1]}")
    out = torch.zeros((mels.shape[0], target_frames, target_dim), dtype=torch.float32)
    valid_frames = min(int(mels.shape[1]), target_frames)
    if valid_frames > 0:
        out[:, :valid_frames, :] = mels[:, :valid_frames, :].to(torch.float32)
    return out


class VoiceCloneFrontendHMONNX:
    """Run exported speech_tokenizer.encode and speaker_encoder HMONNX modules."""

    def __init__(self, args: argparse.Namespace, device: torch.device, logger):
        from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

        encode_hmonnx, speaker_hmonnx = _resolve_frontend_hmonnx_paths(args)
        logger.info(f"Loading speech_tokenizer.encode HMONNX: {encode_hmonnx}")
        self.encode_session = HMONNXInference(str(encode_hmonnx))
        self.encode_session.to(device)
        logger.info(f"Loading speaker_encoder HMONNX: {speaker_hmonnx}")
        self.speaker_session = HMONNXInference(str(speaker_hmonnx))
        self.speaker_session.to(device)
        self.device = device
        self.logger = logger

        self.audio_samples = int(self.encode_session.inputs[0].shape[1])
        self.mel_frames = int(self.speaker_session.inputs[0].shape[1])
        self.mel_dim = int(self.speaker_session.inputs[0].shape[2])
        self.encode_input_dtype = self.encode_session.inputs[0].dtype
        self.encode_mask_dtype = self.encode_session.inputs[1].dtype
        self.speaker_input_dtype = self.speaker_session.inputs[0].dtype
        self.sample_rate = int(args.frontend_sample_rate)

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.to(torch.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != self.sample_rate:
            wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
        return wav.squeeze(0).contiguous()

    def _speaker_mels(self, wav: torch.Tensor) -> torch.Tensor:
        mels = mel_spectrogram(
            wav.unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=self.sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return _pad_or_trim_mels(mels, self.mel_frames, self.mel_dim)

    @torch.no_grad()
    def build_prompt(self, ref_audio: str, ref_text: str, xvec_only: bool) -> List[VoiceClonePromptItem]:
        wav = self._load_audio(ref_audio)
        if wav.numel() > self.audio_samples:
            self.logger.warning(
                f"Reference audio {ref_audio} has {wav.numel()} samples; "
                f"truncating to {self.audio_samples} for static speech_tokenizer.encode HMONNX"
            )

        input_values, padding_mask = _pad_or_trim_1d(wav, self.audio_samples)
        input_values = input_values.to(device=self.device, dtype=self.encode_input_dtype)
        padding_mask = padding_mask.to(device=self.device, dtype=self.encode_mask_dtype)
        encode_out = self.encode_session(input_values, padding_mask)
        audio_codes, valid_frames = encode_out if isinstance(encode_out, (tuple, list)) else (encode_out, None)
        valid_len = audio_codes.shape[1] if valid_frames is None else int(valid_frames.detach().cpu().reshape(-1)[0])
        ref_code = None if xvec_only else audio_codes[0, :valid_len, :].detach().cpu().to(torch.long)

        mels = self._speaker_mels(wav).to(device=self.device, dtype=self.speaker_input_dtype)
        speaker_out = self.speaker_session(mels)
        if isinstance(speaker_out, (tuple, list)):
            speaker_out = speaker_out[0]
        ref_spk_embedding = speaker_out[0].detach().cpu().to(torch.float32)

        return [
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=bool(xvec_only),
                icl_mode=bool(not xvec_only),
                ref_text=ref_text,
            )
        ]


def _run_generate(xh_model, cfg, args: argparse.Namespace, logger):
    """Call the generate_* method matching cfg.tts_mode; returns (wavs, sr)."""
    mode = getattr(cfg, "tts_mode", "custom_voice")
    text = getattr(cfg, "tts_text", _DEFAULT_TEXT)
    if mode == "voice_design":
        return xh_model.generate_voice_design(
            text=text, language="Chinese",
            instruct=getattr(cfg, "tts_instruct", ""),
        )
    elif mode == "voice_clone":
        ref_audio = getattr(cfg, "ref_audio", "/tmp/clone_1.wav")
        assert Path(ref_audio).exists(), f"missing reference audio {ref_audio}"
        ref_text = getattr(cfg, "ref_text", "")
        voice_clone_frontend = VoiceCloneFrontendHMONNX(args, torch.device(cfg.exec_device), logger)
        voice_clone_prompt = voice_clone_frontend.build_prompt(ref_audio, ref_text, args.xvec_only)
        return xh_model.generate_voice_clone(
            text=text, language="Chinese",
            ref_audio=None,
            ref_text=ref_text,
            voice_clone_prompt=voice_clone_prompt,
        )
    else:  # custom_voice (default)
        return xh_model.generate_custom_voice(
            text=text, language="Chinese",
            speaker=getattr(cfg, "tts_speaker", "vivian"),
        )


def _impl(cfg: Config, args: argparse.Namespace):
    logger = get_root_logger()
    work_dir = cfg.work_dir
    exec_device = cfg.exec_device
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, Qwen3TTSHMONNXInference)
    xh_model.to(exec_device)

    xh_ort_config.disable_progress = False
    wavs, sr = _run_generate(xh_model, cfg, args, logger)
    out_file = Path(work_dir) / f"output_{getattr(cfg, 'tts_mode', 'custom_voice')}.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if getattr(args, "variant", None):
        from config.llm._components import apply_variant_hmonnx
        apply_variant_hmonnx(cfg, args.variant)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    cfg.debug = False
    cfg.exec_device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # exec device: data is moved here when running a module/op

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)
    cfg.config_file = config_file

    _impl(cfg, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py",
        help="HMONNX config (unified; pick variant with --variant)",
    )
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"], default=None,
                        help="TTS variant; injects work_dirs paths + hf_model/tts_mode into the parsed config")
    parser.add_argument("--name", type=str, default=None,
                        help="scratch work_dir name; defaults to '<config stem>_<variant>' to avoid clashes")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--xvec-only", action="store_true",
                        help="voice-clone frontend only uses speaker embedding; disables prompt audio codes")
    parser.add_argument("--frontend-hmonnx-dir", type=str, default=DEFAULT_FRONTEND_HMONNX_DIR,
                        help="work dir containing exported voice-clone frontend HMONNX files")
    parser.add_argument("--speech-tokenizer-encode-hmonnx", type=str, default=None,
                        help="explicit speech_tokenizer.encode HMONNX path; overrides --frontend-hmonnx-dir")
    parser.add_argument("--speaker-encoder-hmonnx", type=str, default=None,
                        help="explicit speaker_encoder HMONNX path; overrides --frontend-hmonnx-dir")
    parser.add_argument("--frontend-sample-rate", type=int, default=24000,
                        help="sample rate expected by exported voice-clone frontend modules")

    args = parser.parse_args()
    _stem = Path(args.config).stem
    if args.name:
        cfg_name = args.name
    elif args.variant:
        cfg_name = f"{_stem}_{args.variant}"
    else:
        cfg_name = _stem
    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = args.work_dir

    if Path(work_dir).exists():
        import shutil

        from loguru import logger

        logger.info(f"Work dir {work_dir} already exists, removing it...")
        shutil.rmtree(work_dir)
    main(args)
