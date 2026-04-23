import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from funasr import AutoModel
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
from funasr.utils.postprocess_utils import sentence_postprocess

from input_utils import build_decoder_inputs_from_arrays, build_encoder_inputs_from_arrays, get_onnx_input_specs, init_decoder_caches

try:
    import onnxruntime
    onnxruntime.preload_dlls()
except (ImportError, OSError):
    pass

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_available_providers, get_device


LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = [0, 10, 5]
DEFAULT_ENCODER_LOOK_BACK = 4
DEFAULT_DECODER_LOOK_BACK = 1


class OrtInferSession:
    def __init__(self, model_file: str, device_id: int = -1, intra_op_num_threads: int = 4):
        session_options = SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.log_severity_level = 4
        session_options.enable_cpu_mem_arena = False
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_provider = "CUDAExecutionProvider"
        cpu_provider = "CPUExecutionProvider"
        providers = []
        if device_id >= 0 and get_device() == "GPU" and cuda_provider in get_available_providers():
            providers.append((cuda_provider, {"device_id": str(device_id)}))
        providers.append((cpu_provider, {}))
        self.session = InferenceSession(model_file, sess_options=session_options, providers=providers)

    def __call__(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        input_dict = dict(zip(self.get_input_names(), inputs))
        return self.session.run(self.get_output_names(), input_dict)

    def get_input_names(self) -> List[str]:
        return [value.name for value in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [value.name for value in self.session.get_outputs()]


def load_model(model_dir: str, model_revision: str, device: str = "cpu"):
    model = AutoModel(model=model_dir, model_revision=model_revision, device=device, disable_update=True)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()
    return model


def create_runtime_state(
    model,
    chunk_size: List[int],
    encoder_chunk_look_back: int,
    decoder_chunk_look_back: int,
) -> Dict:
    runtime_kwargs = dict(model.kwargs)
    runtime_kwargs.update(
        {
            "chunk_size": list(chunk_size),
            "encoder_chunk_look_back": encoder_chunk_look_back,
            "decoder_chunk_look_back": decoder_chunk_look_back,
        }
    )
    cache: Dict = {}
    model.model.init_cache(cache, **runtime_kwargs)
    return {"cache": cache, "runtime_kwargs": runtime_kwargs}


def init_predictor_cache(model, chunk_size: List[int]) -> Dict[str, torch.Tensor]:
    output_size = model.model.encoder.output_size()
    return {
        "chunk_size": list(chunk_size),
        "cif_hidden": torch.zeros((1, 1, output_size), dtype=torch.float32),
        "cif_alphas": torch.zeros((1, 1), dtype=torch.float32),
    }


def run_predictor_chunk(
    model,
    enc: np.ndarray,
    predictor_cache: Dict[str, torch.Tensor],
    is_final: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        enc_tensor = torch.from_numpy(enc.astype(np.float32))
        acoustic_embeds, token_lengths, _, _ = model.model.predictor.forward_chunk(
            enc_tensor,
            predictor_cache,
            is_final=is_final,
        )
        token_lengths = token_lengths.round().long()
    return acoustic_embeds.detach().cpu().numpy().astype(np.float32), token_lengths.detach().cpu().numpy().astype(np.int32)


def load_audio_tensor(audio_path: str, runtime_kwargs: Dict) -> Tuple[torch.Tensor, bool]:
    cfg = {"is_final": False}
    audio_sample = load_audio_text_image_video(
        audio_path,
        fs=runtime_kwargs["frontend"].fs,
        audio_fs=runtime_kwargs.get("fs", 16000),
        data_type=runtime_kwargs.get("data_type", "sound"),
        tokenizer=runtime_kwargs.get("tokenizer"),
        cache=cfg,
    )
    if isinstance(audio_sample, (list, tuple)):
        audio_sample = audio_sample[0]
    if audio_sample.ndim == 0:
        audio_sample = audio_sample.reshape(1)
    if audio_sample.ndim > 1:
        audio_sample = audio_sample.reshape(-1)
    return audio_sample, cfg["is_final"]


def extract_full_features(model, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
    audio_sample, _ = load_audio_tensor(audio_path, model.kwargs)
    speech, speech_lengths = extract_fbank(
        [audio_sample],
        data_type=model.kwargs.get("data_type", "sound"),
        frontend=model.kwargs["frontend"],
    )
    return (
        speech.detach().cpu().numpy().astype(np.float32),
        speech_lengths.detach().cpu().numpy().astype(np.int32),
    )


def iter_feature_chunks(
    _model,
    audio_path: str,
    runtime_state: Dict,
) -> Iterable[Tuple[np.ndarray, np.ndarray, bool]]:
    cache = runtime_state["cache"]
    runtime_kwargs = runtime_state["runtime_kwargs"]
    chunk_size = runtime_kwargs["chunk_size"]
    chunk_stride_samples = int(chunk_size[1] * 960)
    audio_sample, inferred_final = load_audio_tensor(audio_path, runtime_kwargs)
    audio_sample = torch.cat((cache["prev_samples"], audio_sample))

    total_chunks = int(len(audio_sample) // chunk_stride_samples + int(inferred_final))
    remainder = int(len(audio_sample) % chunk_stride_samples * (1 - int(inferred_final)))
    for index in range(total_chunks):
        is_final = inferred_final and index == total_chunks - 1
        audio_chunk = audio_sample[index * chunk_stride_samples : (index + 1) * chunk_stride_samples]
        if is_final and len(audio_chunk) < 960:
            cache["encoder"]["tail_chunk"] = True
            speech = cache["encoder"]["feats"]
            speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.int64, device=speech.device)
        else:
            speech, speech_lengths = extract_fbank(
                [audio_chunk],
                data_type=runtime_kwargs.get("data_type", "sound"),
                frontend=runtime_kwargs["frontend"],
                cache=cache["frontend"],
                is_final=is_final,
            )
        yield speech.detach().cpu().numpy().astype(np.float32), speech_lengths.detach().cpu().numpy().astype(np.int32), is_final

    cache["prev_samples"] = audio_sample[-remainder:] if remainder > 0 else torch.empty(0)


def cif_chunk(
    enc: np.ndarray,
    alphas: np.ndarray,
    cache: Dict[str, np.ndarray],
    chunk_size: List[int],
    is_final: bool,
    tail_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, _, hidden_size = enc.shape
    if alphas.ndim == 3:
        alphas = np.squeeze(alphas, axis=-1)
    alphas = alphas.astype(np.float32, copy=True)
    hidden = enc.astype(np.float32, copy=False)

    alphas[:, : chunk_size[0]] = 0.0
    if not is_final:
        alphas[:, sum(chunk_size[:2]) :] = 0.0

    cached_hidden = cache.get("cif_hidden")
    cached_alphas = cache.get("cif_alphas")
    if cached_hidden is not None and cached_alphas is not None:
        hidden = np.concatenate([cached_hidden, hidden], axis=1)
        alphas = np.concatenate([cached_alphas, alphas], axis=1)

    if is_final:
        tail_hidden = np.zeros((batch_size, 1, hidden_size), dtype=np.float32)
        tail_alphas = np.full((batch_size, 1), tail_threshold, dtype=np.float32)
        hidden = np.concatenate([hidden, tail_hidden], axis=1)
        alphas = np.concatenate([alphas, tail_alphas], axis=1)

    token_lengths: List[int] = []
    frame_batches: List[np.ndarray] = []
    next_cache_hidden: List[np.ndarray] = []
    next_cache_alphas: List[float] = []

    for batch_index in range(batch_size):
        integrate = 0.0
        frames = np.zeros((hidden_size,), dtype=np.float32)
        frame_list: List[np.ndarray] = []
        for time_index in range(alphas.shape[1]):
            alpha = float(alphas[batch_index, time_index])
            if alpha + integrate < 1.0:
                integrate += alpha
                frames = frames + alpha * hidden[batch_index, time_index]
            else:
                frames = frames + (1.0 - integrate) * hidden[batch_index, time_index]
                frame_list.append(frames.copy())
                integrate += alpha
                integrate -= 1.0
                frames = integrate * hidden[batch_index, time_index]

        next_cache_alphas.append(integrate)
        if integrate > 0.0:
            next_cache_hidden.append(frames / integrate)
        else:
            next_cache_hidden.append(frames)

        token_lengths.append(len(frame_list))
        if frame_list:
            frame_batches.append(np.stack(frame_list, axis=0))
        else:
            frame_batches.append(np.zeros((0, hidden_size), dtype=np.float32))

    cache["cif_hidden"] = np.stack(next_cache_hidden, axis=0)[:, None, :].astype(np.float32)
    cache["cif_alphas"] = np.array(next_cache_alphas, dtype=np.float32)[:, None]

    max_token_len = max(token_lengths) if token_lengths else 0
    if max_token_len == 0:
        return np.zeros((batch_size, 0, hidden_size), dtype=np.float32), np.array(token_lengths, dtype=np.int32)

    padded_batches = []
    for frames, token_len in zip(frame_batches, token_lengths):
        if token_len < max_token_len:
            pad = np.zeros((max_token_len - token_len, hidden_size), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
        padded_batches.append(frames)
    return np.stack(padded_batches, axis=0).astype(np.float32), np.array(token_lengths, dtype=np.int32)


def decode_ids(model, sample_ids: np.ndarray, valid_token_num: int) -> List[str]:
    token_ids = sample_ids[:valid_token_num].tolist()
    token_ids = [token_id for token_id in token_ids if token_id not in {model.model.blank_id, model.model.sos, model.model.eos}]
    return model.kwargs["tokenizer"].ids2tokens(token_ids)


def tokens_to_text(tokens: List[str]) -> str:
    return sentence_postprocess(tokens)[0]


def build_decoder_inputs_from_encoder_outputs(
    model,
    decoder_onnx: Path,
    enc: np.ndarray,
    enc_len: np.ndarray,
    alphas: np.ndarray,
) -> Dict[str, np.ndarray]:
    predictor_cache = {
        "cif_hidden": np.zeros((int(enc.shape[0]), 1, int(enc.shape[-1])), dtype=np.float32),
        "cif_alphas": np.zeros((int(enc.shape[0]), 1), dtype=np.float32),
    }
    acoustic_embeds, acoustic_embeds_len = cif_chunk(
        enc,
        alphas,
        predictor_cache,
        DEFAULT_CHUNK_SIZE,
        True,
        model.model.predictor.tail_threshold,
    )
    decoder_inputs = build_decoder_inputs_from_arrays(
        decoder_onnx,
        enc,
        enc_len.astype(np.int32),
        acoustic_embeds,
        acoustic_embeds_len.astype(np.int32),
        caches=init_decoder_caches(decoder_onnx, batch_size=int(enc.shape[0])),
    )
    return {
        "enc": enc.astype(np.float32),
        "enc_len": enc_len.astype(np.int32),
        "alphas": alphas.astype(np.float32),
        "acoustic_embeds": acoustic_embeds.astype(np.float32),
        "acoustic_embeds_len": acoustic_embeds_len.astype(np.int32),
        "decoder_inputs": decoder_inputs,
    }


def build_full_utterance_inputs(
    model,
    audio_path: str,
    encoder_onnx: Path,
    decoder_onnx: Path,
    device_id: int,
    threads: int,
) -> Dict[str, np.ndarray]:
    speech, speech_lengths = extract_full_features(model, audio_path)
    encoder_session = OrtInferSession(str(encoder_onnx), device_id=device_id, intra_op_num_threads=threads)
    encoder_specs = get_onnx_input_specs(encoder_onnx)
    encoder_inputs = build_encoder_inputs_from_arrays(
        encoder_onnx,
        speech,
        speech_lengths,
    )
    encoder_outputs = encoder_session([encoder_inputs[name] for name, _ in encoder_specs])
    enc, enc_len, alphas = encoder_outputs[:3]
    prepared = build_decoder_inputs_from_encoder_outputs(model, decoder_onnx, enc, enc_len, alphas)
    return {
        "speech": speech.astype(np.float32),
        "speech_lengths": speech_lengths.astype(np.int32),
        **prepared,
    }


def build_first_chunk_inputs(
    model,
    audio_path: str,
    encoder_onnx: Path,
    decoder_onnx: Path,
    device_id: int,
    threads: int,
) -> Dict[str, np.ndarray]:
    LOGGER.warning("build_first_chunk_inputs uses chunk semantics and does not match the default Paraformer ONNX export")
    runtime_state = create_runtime_state(
        model,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_ENCODER_LOOK_BACK,
        DEFAULT_DECODER_LOOK_BACK,
    )
    encoder_session = OrtInferSession(str(encoder_onnx), device_id=device_id, intra_op_num_threads=threads)
    encoder_specs = get_onnx_input_specs(encoder_onnx)
    predictor_cache = init_predictor_cache(model, DEFAULT_CHUNK_SIZE)
    for speech, speech_lengths, is_final in iter_feature_chunks(model, audio_path, runtime_state):
        encoder_inputs = build_encoder_inputs_from_arrays(
            encoder_onnx,
            speech.astype(np.float32),
            speech_lengths.astype(np.int32),
        )
        encoder_outputs = encoder_session([encoder_inputs[name] for name, _ in encoder_specs])
        enc, enc_len, alphas = encoder_outputs[:3]
        acoustic_embeds, acoustic_embeds_len = run_predictor_chunk(model, enc, predictor_cache, is_final)
        decoder_inputs = build_decoder_inputs_from_arrays(
            decoder_onnx,
            enc,
            enc_len.astype(np.int32),
            acoustic_embeds,
            acoustic_embeds_len.astype(np.int32),
            caches=init_decoder_caches(decoder_onnx, batch_size=int(enc.shape[0])),
        )
        return {
            "speech": speech.astype(np.float32),
            "speech_lengths": speech_lengths.astype(np.int32),
            "enc": enc.astype(np.float32),
            "enc_len": enc_len.astype(np.int32),
            "alphas": alphas.astype(np.float32),
            "acoustic_embeds": acoustic_embeds.astype(np.float32),
            "acoustic_embeds_len": acoustic_embeds_len.astype(np.int32),
            "decoder_inputs": decoder_inputs,
        }
    raise RuntimeError("no chunks produced from audio")