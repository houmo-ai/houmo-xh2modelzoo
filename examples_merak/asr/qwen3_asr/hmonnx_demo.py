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


class HMONNXQwen3ASRChunkRunner:
    def __init__(
        self,
        *,
        work_dir: Path,
        device: str | None,
        max_audio_length: int | None,
        cache_len: int,
        max_new_tokens: int,
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
        self.cache_len = int(cache_len)
        self.max_new_tokens = int(max_new_tokens)

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

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "audio", "audio": "placeholder"}]},
        ]
        self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _new_cache(self):
        torch = self.torch
        shape = (1, self.num_kv_heads, self.cache_len, self.head_dim)
        kcache = [self.cache_tensor_cls(torch.zeros(shape, dtype=torch.float16, device=self.device)) for _ in range(self.num_layers)]
        vcache = [self.cache_tensor_cls(torch.zeros(shape, dtype=torch.float16, device=self.device)) for _ in range(self.num_layers)]
        return kcache, vcache

    def _encode_audio(self, audio_array):
        torch = self.torch
        inputs = self.processor(text=self.prompt, audio=audio_array, return_tensors="pt", padding=True).to(self.device)
        inputs["input_features"] = inputs["input_features"].float()

        feat_len = int(inputs["input_features"].shape[2])
        feature_lens = inputs["feature_attention_mask"].sum(dim=-1).to(torch.int32)
        if feat_len > self.max_audio_length:
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
        return inputs["input_ids"], audio_embeds

    def _build_inputs_embeds(self, text_input_ids, audio_embeds):
        text_embeds = self.embed_tokens(text_input_ids)
        pad_indices = (text_input_ids == self.audio_pad_id).nonzero(as_tuple=True)[1]
        if len(pad_indices) == 0:
            return text_embeds
        start_idx = pad_indices[0].item()
        end_idx = pad_indices[-1].item()
        return self.torch.cat(
            [
                text_embeds[:, :start_idx, :],
                audio_embeds.to(text_embeds.dtype),
                text_embeds[:, end_idx + 1 :, :],
            ],
            dim=1,
        )

    def transcribe_chunk(self, audio_array) -> str:
        torch = self.torch
        text_input_ids, audio_embeds = self._encode_audio(audio_array)
        final_inputs_embeds = self._build_inputs_embeds(text_input_ids, audio_embeds)

        length = min(final_inputs_embeds.shape[1], self.max_prefill)
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

        result = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        match = re.search(r"(?<=<asr_text>)[\s\S]*", result)
        return match.group().strip() if match else result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR HMONNX chunk inference.")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-audio-length", "--max_audio_length", dest="max_audio_length", type=int, default=None)
    parser.add_argument("--cache-len", "--cache_len", dest="cache_len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=128)
    return parser.parse_args(argv)


def main(args=None) -> None:
    import librosa
    import numpy as np

    args = parse_args() if args is None else args
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    runner = HMONNXQwen3ASRChunkRunner(
        work_dir=Path(args.work_dir),
        device=args.device,
        max_audio_length=args.max_audio_length,
        cache_len=args.cache_len,
        max_new_tokens=args.max_new_tokens,
    )
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    chunk_seconds = runner.max_audio_length / 100.0
    chunk_size = int(sr * chunk_seconds)
    n_chunks = max(1, (len(audio) + chunk_size - 1) // chunk_size)

    results = []
    for idx in range(n_chunks):
        chunk = audio[idx * chunk_size : (idx + 1) * chunk_size].astype(np.float32, copy=False)
        print(f">>> chunk {idx + 1}/{n_chunks}, samples={len(chunk)}")
        results.append(runner.transcribe_chunk(chunk))
    print("transcription:", " ".join(filter(None, results)))


if __name__ == "__main__":
    main()
