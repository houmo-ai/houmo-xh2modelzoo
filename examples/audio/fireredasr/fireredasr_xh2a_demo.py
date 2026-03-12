"""
FireRedASR xh2a standalone HMONNX real-audio demo.

This script does NOT load FireRedASR native checkpoints.
It only uses:
1. Audio Encoder HMONNX
2. LLM Prefill/Decode HMONNX export directory
3. cmvn.ark

Pipeline:
  wav -> fbank+cmvn -> audio_encoder_hmonnx -> merge <speech> embedding ->
  llm_hmonnx prefill/decode -> transcript text
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Ensure local repo package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhquant.api import HMONNXInference
from xh_model_zoo.xh_llm.models.llm_onnx_model import LLMONNXModel

try:
    from xh_model_zoo.xh_llm.models.llm_onnx_model import LLMLoRAONNXModel
except ImportError:
    LLMLoRAONNXModel = LLMONNXModel

# Add FireRedASR code path (for fbank+cmvn implementation and speech token const).
FIREREDASR_ROOT = REPO_ROOT / ".." / "FireRedASR"
sys.path.insert(0, str(FIREREDASR_ROOT))
from fireredasr.data.asr_feat import ASRFeatExtractor  # noqa: E402
from fireredasr.tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN  # noqa: E402


DEFAULT_MAX_AUDIO_SECONDS = 30.0
FBANK_FRAME_SHIFT_MS = 10
FBANK_DIM = 80
CONTEXT_PAD = 6
ATTN_MASK_PAD_VALUE = -65504.0

DECODE_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}"
    "{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\\n' }}{% endif %}"
    "{% endfor %}"
)


def compute_conv_valid_length(input_length: int) -> int:
    length = (input_length - 3) // 2 + 1
    length = (length - 3) // 2 + 1
    return int(length)


def compute_conv_total_length(fbank_padded_length: int) -> int:
    length = fbank_padded_length + CONTEXT_PAD
    length = (length - 3) // 2 + 1
    length = (length - 3) // 2 + 1
    return int(length)


def compute_adapter_output_length(conv_len: int, downsample_rate: int = 2) -> int:
    return int(conv_len // downsample_rate)


def audio_seconds_to_fbank_frames(audio_seconds: float) -> int:
    if audio_seconds <= 0:
        raise ValueError(f"audio_seconds must be > 0, got {audio_seconds}")
    return max(1, int(audio_seconds * 1000 / FBANK_FRAME_SHIFT_MS))


def pad_fbank_to_fixed(fbank: torch.Tensor, t_max: int) -> torch.Tensor:
    batch, t_cur, dim = fbank.shape
    if t_cur >= t_max:
        return fbank[:, :t_max, :]
    padded = torch.zeros(batch, t_max, dim, dtype=fbank.dtype, device=fbank.device)
    padded[:, :t_cur, :] = fbank
    return padded


def build_masks_from_length(
    fbank_length: int, fbank_padded_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_conv_len = compute_conv_total_length(fbank_padded_length)
    valid_conv_len = compute_conv_valid_length(fbank_length)

    conv_mask = torch.zeros(1, 1, total_conv_len, dtype=torch.float32)
    conv_mask[:, :, :valid_conv_len] = 1.0

    attn_mask = torch.full(
        (1, 1, 1, total_conv_len), ATTN_MASK_PAD_VALUE, dtype=torch.float32
    )
    attn_mask[:, :, :, :valid_conv_len] = 0.0
    return attn_mask, conv_mask


def collect_wav_paths(wav_path_args: List[str]) -> List[str]:
    wav_paths: List[str] = []
    for pattern in wav_path_args:
        if any(ch in pattern for ch in ["*", "?", "["]):
            wav_paths.extend(sorted(glob.glob(pattern)))
            continue
        p = Path(pattern)
        if p.is_dir():
            wav_paths.extend(sorted(str(x) for x in p.glob("*.wav")))
        elif p.exists():
            wav_paths.append(str(p))
    return wav_paths


def load_ref_texts(ref_file: str) -> Dict[str, str]:
    if not ref_file or not Path(ref_file).exists():
        return {}
    ref_texts: Dict[str, str] = {}
    with open(ref_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                ref_texts[parts[0]] = parts[1].replace(" ", "")
    return ref_texts


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[n][m]


class AudioEncoderHMONNX:
    def __init__(self, hmonnx_path: str, device: torch.device):
        self.device = device
        self.session = HMONNXInference(hmonnx_path)
        self.session.exec_device = str(device)
        self.session.to(str(device))

    @torch.no_grad()
    def __call__(
        self, fbank_features: torch.Tensor, attn_mask: torch.Tensor, conv_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.session(
            fbank_features.half().to(self.device),
            attn_mask.half().to(self.device),
            conv_mask.half().to(self.device),
        )
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


class FireRedASRStandaloneHMONNX:
    def __init__(
        self,
        audio_hmonnx_path: str,
        llm_hmonnx_dir: str,
        cmvn_path: str,
        device: torch.device,
        max_audio_seconds: float,
    ):
        self.device = device
        self.max_audio_seconds = float(max_audio_seconds)
        self.t_fbank_max = audio_seconds_to_fbank_frames(self.max_audio_seconds)
        self.audio_encoder = AudioEncoderHMONNX(audio_hmonnx_path, device)
        self.feat_extractor = ASRFeatExtractor(cmvn_path)

        meta_path = Path(llm_hmonnx_dir) / "export_meta_info.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"export_meta_info.json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_info = json.load(f)

        hf_config_dir = Path(llm_hmonnx_dir) / self.meta_info["hf_config"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(hf_config_dir))
        self.tokenizer.padding_side = "right"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
        )

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.eos_token_id is None or self.eos_token_id < 0:
            self.eos_token_id = self.tokenizer.eos_token_id
        self.speech_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        if self.speech_token_id is None or self.speech_token_id < 0:
            raise RuntimeError(f"Cannot resolve speech token id for token: {DEFAULT_SPEECH_TOKEN}")

        lora_mode = self.meta_info.get("lora_mode", "merge_lora")
        onnx_cls = LLMLoRAONNXModel if lora_mode == "keep_lora" else LLMONNXModel
        use_lora_mask = bool(self.meta_info.get("keep_lora_use_mask", True))
        prefill_seq_len = int(
            self.meta_info.get("wrap_cfg", {}).get("input_sequence_length", 256)
        )
        llm_kwargs = dict(
            prefill=dict(
                onnx=str(Path(llm_hmonnx_dir) / self.meta_info["prefill_onnx_file"]),
                input_sequence_length=prefill_seq_len,
            ),
            decode=dict(
                onnx=str(Path(llm_hmonnx_dir) / self.meta_info["decode_onnx_file"]),
            ),
            kv_cache=dict(
                num_hidden_layers=self.meta_info["num_hidden_layers"],
                shape=self.meta_info["kv_cache_shape"],
            ),
        )
        if lora_mode == "keep_lora":
            llm_kwargs["use_lora_mask"] = use_lora_mask
        self.llm = onnx_cls(**llm_kwargs)
        token_embedding_state_dict = torch.load(
            str(Path(llm_hmonnx_dir) / self.meta_info["token_embedding_file"]),
            map_location="cpu",
            weights_only=True,
        )
        token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        token_embedding.load_state_dict(token_embedding_state_dict)
        self.llm.set_input_embeddings(token_embedding)
        self.llm.pad_token_id = int(self.pad_token_id)
        self.llm.set_exec_device(self.device)
        self.llm.to(self.device)
        self.llm.to(torch.float16)

    def _build_prompt_input_ids(self, prompt: str, max_prompt_len: int = 128) -> torch.Tensor:
        messages = [
            {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}{prompt}"},
            {"role": "assistant", "content": ""},
        ]
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            chat_template=DECODE_TEMPLATE,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_prompt_len,
        )
        if not isinstance(token_ids, list):
            raise RuntimeError("Unexpected tokenizer output for chat template.")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return input_ids

    def _merge_speech_with_prompt(
        self, input_ids: torch.Tensor, speech_features: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.llm, "get_input_embeddings"):
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        else:
            inputs_embeds = self.llm.token_embedding(input_ids)
        speech_pos = torch.where(input_ids[0] == self.speech_token_id)[0]
        if speech_pos.numel() != 1:
            raise RuntimeError(
                f"Expected exactly one {DEFAULT_SPEECH_TOKEN} in prompt, got {speech_pos.numel()}."
            )
        pos = int(speech_pos.item())

        speech_features = speech_features.to(inputs_embeds.dtype)
        merged = torch.cat(
            [
                inputs_embeds[:, :pos, :],
                speech_features,
                inputs_embeds[:, pos + 1 :, :],
            ],
            dim=1,
        )
        return merged

    @torch.no_grad()
    def _generate_from_inputs_embeds(
        self,
        merged_inputs_embeds: torch.Tensor,
        max_new_tokens: int,
        decode_min_len: int = 0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
    ) -> List[int]:
        for cache in self.llm.past_key_caches:
            cache.reset()
        for cache in self.llm.past_value_caches:
            cache.reset()

        seq_len = int(merged_inputs_embeds.shape[1])
        step_input_len = int(self.llm.prefill_input_sequence_length)
        if step_input_len <= 0:
            step_input_len = seq_len

        logits = None
        last_chunk_valid_len = 0
        for start in range(0, seq_len, step_input_len):
            end = min(start + step_input_len, seq_len)
            sub_embeds = merged_inputs_embeds[:, start:end, :]
            last_chunk_valid_len = int(end - start)
            prefill_data = {
                "inputs_embeds": sub_embeds,
                "past_seq_length": start,
                "input_sequence_length": step_input_len,
            }
            (
                prefill_inputs_embeds,
                prefill_past_seq_length,
                prefill_seq_length,
                prefill_k_caches,
                prefill_v_caches,
                *prefill_extra_args,
            ) = self.llm.prepare_inputs(prefill_data)
            prefill_inputs = [
                prefill_inputs_embeds,
                prefill_past_seq_length,
                prefill_seq_length,
            ]
            prefill_inputs += prefill_k_caches
            prefill_inputs += prefill_v_caches
            prefill_inputs += prefill_extra_args
            logits = self.llm.prefill_session(*prefill_inputs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
        if logits is None:
            raise RuntimeError("Prefill failed: no logits generated.")
        if logits.shape[1] > 1:
            last_idx = min(max(last_chunk_valid_len - 1, 0), logits.shape[1] - 1)
            logits = logits[:, last_idx : last_idx + 1, :]

        generated_token_ids: List[int] = []
        past_seq_length = merged_inputs_embeds.shape[1]

        for step in range(max_new_tokens):
            next_token_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0 and generated_token_ids:
                for token_id in set(generated_token_ids):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
            if (
                next_token_id == self.eos_token_id
                and step + 1 < decode_min_len
                and self.eos_token_id is not None
                and self.eos_token_id >= 0
            ):
                next_token_logits[0, self.eos_token_id] = -1e9
                next_token_id = int(torch.argmax(next_token_logits, dim=-1).item())
            if next_token_id == self.eos_token_id and step + 1 >= decode_min_len:
                break

            generated_token_ids.append(next_token_id)

            decode_data = {
                "input_ids": torch.tensor([[next_token_id]], dtype=torch.long, device=self.device),
                "past_seq_length": past_seq_length,
            }
            logits = self.llm.decode(decode_data)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            past_seq_length += 1

        return generated_token_ids

    @torch.no_grad()
    def transcribe_one(
        self,
        wav_path: str,
        prompt: str = "请转写音频为文字",
        decode_max_len: int = 0,
        decode_min_len: int = 0,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
    ) -> Dict[str, str]:
        uttid = Path(wav_path).stem
        feats, lengths, durs = self.feat_extractor([wav_path])
        fbank_len = int(lengths[0].item())
        fbank_len = min(fbank_len, self.t_fbank_max)

        fbank_padded = pad_fbank_to_fixed(feats, self.t_fbank_max)
        attn_mask, conv_mask = build_masks_from_length(fbank_len, self.t_fbank_max)

        start_time = time.time()
        speech_features = self.audio_encoder(fbank_padded, attn_mask, conv_mask)
        speech_conv_len = compute_conv_valid_length(fbank_len)
        speech_len = compute_adapter_output_length(speech_conv_len)
        speech_features = speech_features[:, :speech_len, :]

        input_ids = self._build_prompt_input_ids(prompt=prompt)
        merged_inputs_embeds = self._merge_speech_with_prompt(input_ids, speech_features)

        max_new_tokens = speech_len if decode_max_len < 1 else int(decode_max_len)
        max_new_tokens = max(1, max_new_tokens)
        generated_ids = self._generate_from_inputs_embeds(
            merged_inputs_embeds=merged_inputs_embeds,
            max_new_tokens=max_new_tokens,
            decode_min_len=decode_min_len,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        elapsed = time.time() - start_time
        duration = float(durs[0]) if durs else 0.0
        rtf = elapsed / duration if duration > 0 else 0.0
        return {
            "uttid": uttid,
            "wav": wav_path,
            "text": text,
            "rtf": f"{rtf:.4f}",
        }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FireRedASR standalone HMONNX demo (no native model weights).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hmonnx",
        choices=["hmonnx", "standalone"],
        help="compat mode arg (only hmonnx/standalone is supported)",
    )
    parser.add_argument("--audio_hmonnx_path", type=str, required=True, help="Audio Encoder HMONNX path")
    parser.add_argument("--llm_hmonnx_dir", type=str, required=True, help="LLM HMONNX export work dir")
    parser.add_argument("--cmvn_path", type=str, default=None, help="Kaldi cmvn.ark path")
    parser.add_argument("--model_dir", type=str, default=None, help="deprecated: only used to fallback cmvn path")
    parser.add_argument(
        "--audio_seconds",
        type=float,
        default=None,
        help="最大语音长度（秒）；默认优先从 audio_encoder_export_meta.json 自动读取，缺失时回退 30s",
    )
    parser.add_argument("--wav_path", type=str, nargs="+", required=True, help="wav path(s), glob, or directory")
    parser.add_argument("--prompt", type=str, default="请转写音频为文字", help="ASR prompt text (without <speech>)")
    parser.add_argument("--decode_max_len", type=int, default=0, help="max decode tokens (0 = use speech length)")
    parser.add_argument("--decode_min_len", type=int, default=0, help="min decode tokens")
    parser.add_argument("--beam_size", type=int, default=1, help="deprecated: greedy only in standalone mode")
    parser.add_argument("--llm_length_penalty", type=float, default=0.0, help="deprecated: standalone mode ignores")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--exec_device", type=str, default="cuda:0", help="execution device")
    parser.add_argument("--use_gpu", action="store_true", help="compat arg: prefer gpu")
    parser.add_argument("--no_gpu", action="store_true", help="force cpu")
    parser.add_argument("--audio_onnx_path", type=str, default=None, help="deprecated: standalone mode ignores")
    parser.add_argument("--rotated_adapter_path", type=str, default=None, help="deprecated: standalone mode ignores")
    parser.add_argument("--ref_file", type=str, default=None, help="optional reference text file: uttid text")
    parser.add_argument("--out_json", type=str, default=None, help="optional output json path")
    return parser


def _load_audio_export_meta(audio_hmonnx_path: str):
    meta_path = Path(audio_hmonnx_path).parent / "audio_encoder_export_meta.json"
    if not meta_path.exists():
        return None, meta_path
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f), meta_path


def main():
    args = parse_arguments().parse_args()
    if args.mode not in ["hmonnx", "standalone"]:
        raise ValueError(f"Unsupported mode: {args.mode}")

    wav_paths = collect_wav_paths(args.wav_path)
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found from --wav_path: {args.wav_path}")

    cmvn_path = args.cmvn_path
    if cmvn_path is None and args.model_dir:
        fallback_cmvn = Path(args.model_dir) / "cmvn.ark"
        if fallback_cmvn.exists():
            cmvn_path = str(fallback_cmvn)

    if not Path(args.audio_hmonnx_path).exists():
        raise FileNotFoundError(f"audio_hmonnx_path not found: {args.audio_hmonnx_path}")
    if not Path(args.llm_hmonnx_dir).exists():
        raise FileNotFoundError(f"llm_hmonnx_dir not found: {args.llm_hmonnx_dir}")
    if cmvn_path is None or not Path(cmvn_path).exists():
        raise FileNotFoundError(f"cmvn_path not found: {cmvn_path}")

    audio_meta, audio_meta_path = _load_audio_export_meta(args.audio_hmonnx_path)
    resolved_audio_seconds = args.audio_seconds
    if resolved_audio_seconds is None and audio_meta is not None:
        resolved_audio_seconds = float(
            audio_meta.get("audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        )
    if resolved_audio_seconds is None:
        resolved_audio_seconds = DEFAULT_MAX_AUDIO_SECONDS
    if resolved_audio_seconds <= 0:
        raise ValueError(f"audio_seconds must be > 0, got {resolved_audio_seconds}")

    use_gpu = (args.use_gpu or args.exec_device.startswith("cuda")) and not args.no_gpu
    if not use_gpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(args.exec_device)

    print("=" * 80)
    print("FireRedASR standalone HMONNX demo")
    print(f"audio_hmonnx: {args.audio_hmonnx_path}")
    print(f"llm_hmonnx_dir: {args.llm_hmonnx_dir}")
    print(f"cmvn_path: {cmvn_path}")
    print(f"num_wavs: {len(wav_paths)}")
    print(f"max_audio: {resolved_audio_seconds:g}s")
    if audio_meta is not None:
        print(f"audio_meta: {audio_meta_path}")
    print(f"device: {device}")
    print("=" * 80)

    pipeline = FireRedASRStandaloneHMONNX(
        audio_hmonnx_path=args.audio_hmonnx_path,
        llm_hmonnx_dir=args.llm_hmonnx_dir,
        cmvn_path=cmvn_path,
        device=device,
        max_audio_seconds=resolved_audio_seconds,
    )

    ref_texts = load_ref_texts(args.ref_file)
    results = []
    total_ref_chars = 0
    total_ref_edit = 0
    for wav_path in wav_paths:
        one = pipeline.transcribe_one(
            wav_path=wav_path,
            prompt=args.prompt,
            decode_max_len=args.decode_max_len,
            decode_min_len=args.decode_min_len,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
        )
        uttid = one["uttid"]
        text = one["text"]
        print(f"{uttid}\t{text}\trtf={one['rtf']}")

        ref_text = ref_texts.get(uttid, None)
        if ref_text is not None and len(ref_text) > 0:
            edit = edit_distance(list(ref_text), list(text))
            cer = edit / len(ref_text)
            one["ref_text"] = ref_text
            one["cer"] = cer
            total_ref_chars += len(ref_text)
            total_ref_edit += edit
        results.append(one)

    summary = {
        "num_utts": len(results),
        "cer": (total_ref_edit / total_ref_chars) if total_ref_chars > 0 else None,
        "results": results,
    }
    if summary["cer"] is not None:
        print(f"CER={summary['cer']:.6f}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
