import argparse
import json
import re
from pathlib import Path


SAMPLE_RATE = 16000


def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def _first_tensor(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def _build_audio_chunks(audio, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(audio) <= chunk_size:
        return [audio]
    return [audio[start : start + chunk_size] for start in range(0, len(audio), chunk_size)]


def _is_cjk_char(ch: str) -> bool:
    return "\u3400" <= ch <= "\u4dbf" or "\u4e00" <= ch <= "\u9fff" or "\uf900" <= ch <= "\ufaff"


def _join_merged_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if left[-1] in "，；：、“‘「『《" or right[0] in ",.;:?!，。！？；：、”’」』》":
        return f"{left}{right}"
    if _is_cjk_char(left[-1]) or _is_cjk_char(right[0]):
        return f"{left}{right}"
    return f"{left} {right}"


def _make_text_prefix(tokenizer, text: str, rollback_tokens: int) -> str:
    if not text:
        return ""
    ids = tokenizer.encode(text)
    rollback = max(0, int(rollback_tokens))
    while True:
        end_idx = max(0, len(ids) - rollback)
        prefix = tokenizer.decode(ids[:end_idx]) if end_idx > 0 else ""
        if "\ufffd" not in prefix:
            return prefix
        if end_idx == 0:
            return ""
        rollback += 1


def _clean_forced_asr_text(raw: str) -> str:
    text = "" if raw is None else str(raw).strip()
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[-1]
    return re.sub(r"(?:^|\s*)language\s+[A-Za-z]+", "", text).strip()


def _merge_stream_text(prefix_text: str, generated: str) -> str:
    generated_text = _clean_forced_asr_text(generated)
    prefix_text = (prefix_text or "").strip()
    if not prefix_text:
        return generated_text
    if generated_text.startswith(prefix_text):
        return generated_text
    return _join_merged_text(prefix_text, generated_text)


def _is_cjk_text(text: str) -> bool:
    return any(_is_cjk_char(ch) for ch in text)


def _looks_like_non_chinese_hallucination(text: str) -> bool:
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = sum(1 for ch in text if _is_cjk_char(ch))
    return latin >= 12 and latin > cjk


def _load_artifacts(work_dir: Path) -> dict:
    meta_path = work_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "meta": meta,
        "encoder": work_dir / meta["encoder"]["hmonnx_file"],
        "prefill": work_dir / meta["prefill_onnx_file"],
        "decode": work_dir / meta["decode_onnx_file"],
        "config": work_dir / meta["hf_config"],
        "embedding": work_dir / meta["token_embedding_file"],
    }


class HMONNXQwen3ASRPrefixRunner:
    def __init__(
        self,
        *,
        work_dir: Path,
        device: str | None,
        max_audio_length: int | None,
        max_new_tokens: int,
        cache_len: int,
    ):
        import torch
        import torch.nn as nn
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        from transformers import AutoConfig, AutoTokenizer
        from xhquant.api import HMONNXInference as InferenceEngine
        from xhquant.core import CacheTensor

        self.torch = torch
        self.cache_tensor_cls = CacheTensor
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_new_tokens = int(max_new_tokens)
        self.cache_len = int(cache_len)

        artifacts = _load_artifacts(work_dir)
        meta = artifacts["meta"]
        self.max_audio_length = int(max_audio_length or meta["encoder"]["model_cfg"]["fixed_max_audio_length"])
        self.max_prefill = int(
            meta.get(
                "prefill_input_sequence_length",
                int(_get_feat_extract_output_lengths(self.max_audio_length)) + 21,
            )
        )

        for key in ["encoder", "prefill", "decode", "config", "embedding"]:
            if not artifacts[key].exists():
                raise FileNotFoundError(artifacts[key])

        self.encoder_sess = InferenceEngine(str(artifacts["encoder"]))
        self.encoder_sess.to(str(self.device))
        self.prefill_sess = InferenceEngine(str(artifacts["prefill"]))
        self.prefill_sess.to(str(self.device))
        self.decode_sess = InferenceEngine(str(artifacts["decode"]))
        self.decode_sess.to(str(self.device))

        cfg_dir = str(artifacts["config"])
        self.processor = Qwen3ASRProcessor.from_pretrained(cfg_dir, fix_mistral_regex=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg_dir, trust_remote_code=True, use_fast=True)
        self.config = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)

        weights = torch.load(artifacts["embedding"], map_location="cpu")["weight"]
        self.embed_tokens = nn.Embedding(*weights.shape).to(self.device, dtype=torch.float16).eval()
        self.embed_tokens.weight.data.copy_(weights.to(device=self.device, dtype=torch.float16))

        text_config = self.config.thinker_config.text_config
        self.num_layers = text_config.num_hidden_layers
        self.num_kv_heads = text_config.num_key_value_heads
        self.hidden_size = text_config.hidden_size
        self.head_dim = text_config.head_dim

        proc_tokenizer = self.processor.tokenizer
        if "<|audio_pad|>" in proc_tokenizer.get_vocab():
            self.audio_pad_id = proc_tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        else:
            self.audio_pad_id = proc_tokenizer.encode("<|audio_pad|>", add_special_tokens=False)[0]

    def _new_cache(self):
        torch = self.torch
        shape = (1, self.num_kv_heads, self.cache_len, self.head_dim)
        kcache = [self.cache_tensor_cls(torch.zeros(shape, dtype=torch.float16, device=self.device)) for _ in range(self.num_layers)]
        vcache = [self.cache_tensor_cls(torch.zeros(shape, dtype=torch.float16, device=self.device)) for _ in range(self.num_layers)]
        return kcache, vcache

    def encode_audio(self, audio_array):
        torch = self.torch
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "audio", "audio": "placeholder"}]},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, audio=audio_array, return_tensors="pt", padding=True).to(self.device)
        inputs["input_features"] = inputs["input_features"].float()

        feat_len = int(inputs["input_features"].shape[2])
        feature_lens = inputs["feature_attention_mask"].sum(dim=-1).to(torch.int32)
        if feat_len > self.max_audio_length:
            print(f"audio features truncated: feat_len={feat_len}, max_audio_length={self.max_audio_length}")
            inputs["input_features"] = inputs["input_features"][:, :, : self.max_audio_length]
            inputs["feature_attention_mask"] = inputs["feature_attention_mask"][:, : self.max_audio_length]
            feature_lens = torch.tensor([self.max_audio_length], dtype=torch.int32, device=self.device)
            feat_len = self.max_audio_length
        if feat_len < self.max_audio_length:
            pad_width = (0, self.max_audio_length - feat_len)
            inputs["input_features"] = torch.nn.functional.pad(inputs["input_features"], pad_width, value=0.0)
            inputs["feature_attention_mask"] = torch.nn.functional.pad(inputs["feature_attention_mask"], pad_width, value=0)

        output = self.encoder_sess.run(
            {
                "input_features": inputs["input_features"].to(torch.float16),
                "feature_lens": feature_lens,
            }
        )
        audio_embeds = _first_tensor(output).to(self.device)
        t_out = int(_get_feat_extract_output_lengths(feature_lens).item())
        audio_embeds = audio_embeds[:, :t_out, :]
        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)
        return audio_embeds

    def _build_inputs_embeds(self, audio_embeds_list, prefix_text: str = "", language: str | None = None):
        torch = self.torch
        audio_content = [{"type": "audio", "audio": f"placeholder_{i}"} for i in range(len(audio_embeds_list))]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": audio_content},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            prompt += f"language {language}<asr_text>"
        if prefix_text:
            prompt += prefix_text

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        text_input_ids = encoded["input_ids"].to(self.device)
        text_embeds = self.embed_tokens(text_input_ids)
        pad_indices = (text_input_ids == self.audio_pad_id).nonzero(as_tuple=True)[1]
        if len(pad_indices) != len(audio_embeds_list):
            raise RuntimeError(f"audio pad count mismatch: pads={len(pad_indices)}, embeds={len(audio_embeds_list)}")

        parts = []
        cursor = 0
        for pad_idx, audio_embeds in zip(pad_indices.tolist(), audio_embeds_list):
            parts.append(text_embeds[:, cursor:pad_idx, :])
            parts.append(audio_embeds.to(self.device).to(text_embeds.dtype))
            cursor = pad_idx + 1
        parts.append(text_embeds[:, cursor:, :])
        return torch.cat(parts, dim=1)

    def decode_from_embeds(self, final_inputs_embeds) -> str:
        torch = self.torch
        seq_len = final_inputs_embeds.shape[1]
        if seq_len > self.max_prefill:
            print(f"prefill input truncated: seq_len={seq_len}, max_prefill={self.max_prefill}")
        length = min(seq_len, self.max_prefill)
        prefill_embeds = torch.zeros((1, self.max_prefill, self.hidden_size), dtype=torch.float16, device=self.device)
        prefill_embeds[:, :length, :] = final_inputs_embeds[:, :length, :].to(torch.float16)

        valid_length = torch.tensor([0], dtype=torch.int32, device=self.device)
        current_length = torch.tensor([length], dtype=torch.int32, device=self.device)
        kcache, vcache = self._new_cache()
        prefill_inputs = {"input_embeds": prefill_embeds, "valid_length": valid_length, "current_length": current_length}
        prefill_names = self.prefill_sess.get_input_names()
        for i in range(self.num_layers):
            k_key = f"model_layers_{i}_self_attn_kcache_input"
            v_key = f"model_layers_{i}_self_attn_vcache_input"
            if k_key in prefill_names:
                prefill_inputs[k_key] = kcache[i]
            if v_key in prefill_names:
                prefill_inputs[v_key] = vcache[i]

        next_token_id = torch.argmax(_first_tensor(self.prefill_sess.run(prefill_inputs)), dim=-1).item()
        generated_ids = [next_token_id]
        valid_length = torch.tensor([length], dtype=torch.int32, device=self.device)
        current_length = torch.tensor([1], dtype=torch.int32, device=self.device)
        decode_names = self.decode_sess.get_input_names()

        for _ in range(self.max_new_tokens):
            token_tensor = torch.tensor([[generated_ids[-1]]], device=self.device)
            decode_inputs = {
                "input_embeds": self.embed_tokens(token_tensor).to(torch.float16),
                "valid_length": valid_length,
                "current_length": current_length,
            }
            for i in range(self.num_layers):
                k_key = f"model_layers_{i}_self_attn_kcache_input"
                v_key = f"model_layers_{i}_self_attn_vcache_input"
                if k_key in decode_names:
                    decode_inputs[k_key] = kcache[i]
                if v_key in decode_names:
                    decode_inputs[v_key] = vcache[i]
            next_id = torch.argmax(_first_tensor(self.decode_sess.run(decode_inputs)), dim=-1).item()
            generated_ids.append(next_id)
            valid_length = valid_length + 1
            if next_id == self.processor.tokenizer.eos_token_id:
                break
        return self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def transcribe_with_embeddings(self, audio_embeds_list, prefix_text: str = "", language: str | None = None) -> str:
        return self.decode_from_embeds(
            self._build_inputs_embeds(audio_embeds_list=audio_embeds_list, prefix_text=prefix_text, language=language)
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR HMONNX cumulative prefix inference.")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-audio-length", "--max_audio_length", dest="max_audio_length", type=int, default=None)
    parser.add_argument("--cache-len", "--cache_len", dest="cache_len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128)
    parser.add_argument("--chunk-seconds", "--chunk_seconds", dest="chunk_seconds", type=float, default=20.0)
    parser.add_argument("--unfixed-chunk-num", "--unfixed_chunk_num", dest="unfixed_chunk_num", type=int, default=2)
    parser.add_argument("--rollback-tokens", "--rollback_tokens", dest="rollback_tokens", type=int, default=5)
    parser.add_argument("--language", default="Chinese")
    parser.set_defaults(reject_non_chinese_hallucination=True)
    parser.add_argument("--reject-non-chinese-hallucination", dest="reject_non_chinese_hallucination", action="store_true")
    parser.add_argument("--no-reject-non-chinese-hallucination", dest="reject_non_chinese_hallucination", action="store_false")
    return parser.parse_args(argv)


def main(args=None) -> None:
    import librosa
    import numpy as np
    from qwen_asr import parse_asr_output

    args = parse_args() if args is None else args
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    runner = HMONNXQwen3ASRPrefixRunner(
        work_dir=Path(args.work_dir),
        device=args.device,
        max_audio_length=args.max_audio_length,
        max_new_tokens=args.max_new_tokens,
        cache_len=args.cache_len,
    )
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    chunks = _build_audio_chunks(audio, int(sr * float(args.chunk_seconds)))

    text = ""
    audio_accum = np.zeros((0,), dtype=np.float32)
    for idx, chunk in enumerate(chunks):
        audio_accum = np.concatenate([audio_accum, chunk.astype(np.float32, copy=False)], axis=0)
        prefix_text = "" if idx < args.unfixed_chunk_num else _make_text_prefix(runner.tokenizer, text, args.rollback_tokens)
        audio_embeds = runner.encode_audio(audio_accum)
        print(
            f">>> chunk {idx + 1}/{len(chunks)}, samples={len(chunk)}, "
            f"audio_accum_samples={len(audio_accum)}, prefix_chars={len(prefix_text)}"
        )
        generated = runner.transcribe_with_embeddings([audio_embeds], prefix_text=prefix_text, language=args.language)
        candidate_text = _merge_stream_text(prefix_text, generated)
        language = args.language or parse_asr_output(generated)[0]
        chunk_text = candidate_text[len(prefix_text) :].strip() if prefix_text and candidate_text.startswith(prefix_text) else candidate_text

        accepted = True
        if (
            args.reject_non_chinese_hallucination
            and args.language == "Chinese"
            and not _is_cjk_text(chunk_text)
            and _looks_like_non_chinese_hallucination(chunk_text)
        ):
            accepted = False
        if accepted:
            text = candidate_text
        print(f"[chunk {idx + 1}] language={language!r} accepted={accepted} chunk_text={chunk_text!r} text={text!r}")

    print("transcription:", text)


if __name__ == "__main__":
    main()
