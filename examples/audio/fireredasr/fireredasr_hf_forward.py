"""
FireRedASR HF Forward 替换模块

将 wrap/quant 后的 xh2a 模型（Audio Encoder 和 LLM）接入到 FireRedASR 的
原始 forward 和 eval 流程中，实现无缝替换。

用法:
  1. 导出 Audio Encoder 为 ONNX/HMONNX
  2. 导出 LLM 为 HMONNX
  3. 使用本模块构建替换模型，接入 FireRedASR 的 transcribe 流程

支持三种 Audio Encoder 后端:
  - PyTorch 原始模型
  - ONNX Runtime
  - HMONNX (xh2a)

支持两种 LLM 后端:
  - HuggingFace 原始模型 (with/without merged LoRA)
  - HMONNX wrapped (通过 Qwen2_HFCompatible)
"""

import sys
import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

# Ensure local repo package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 添加 FireRedASR 路径
FIREREDASR_ROOT = REPO_ROOT / ".." / "FireRedASR"
sys.path.insert(0, str(FIREREDASR_ROOT))


# ======================== Audio Encoder 替换 ========================

class FireRedASRAudioEncoderWrapper(nn.Module):
    """将 ONNX/HMONNX Audio Encoder 包装为与原始 encoder+adapter 兼容的接口。
    
    原始 FireRedASR 的 transcribe 调用:
        encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
    
    本 wrapper 将 encoder+adapter 合并为一次调用，返回与 speech_features 兼容的输出。
    """

    def __init__(self, audio_backend, max_fbank_frames: int = 3000):
        """
        Args:
            audio_backend: ONNXAudioEncoder 或 HMONNXAudioEncoder 实例
            max_fbank_frames: 最大 fbank 帧数（对应30s音频=3000帧）
        """
        super().__init__()
        self.audio_backend = audio_backend
        self.max_fbank_frames = max_fbank_frames

    def _compute_conv_output_length(self, input_length: int) -> int:
        """计算 Conv2dSubsampling 后输出长度（含6帧 context pad）。"""
        L = input_length + 6  # context pad
        L = (L - 3) // 2 + 1
        L = (L - 3) // 2 + 1
        return L

    def forward(self, padded_feat, feat_lengths):
        """
        Args:
            padded_feat: [B, T, 80] - 原始 Fbank 特征（可变长度）
            feat_lengths: [B] - 每个样本的有效帧数
        
        Returns:
            speech_features: [B, T_out, llm_dim]
            speech_lens: [B] - 每个样本的有效输出帧数
        """
        batch_size = padded_feat.size(0)
        device = padded_feat.device
        dtype = padded_feat.dtype

        # 1. Pad fbank 到固定长度
        T_actual = padded_feat.size(1)
        if T_actual < self.max_fbank_frames:
            pad_tensor = torch.zeros(
                batch_size, self.max_fbank_frames - T_actual, 80,
                device=device, dtype=dtype
            )
            padded_feat = torch.cat([padded_feat, pad_tensor], dim=1)
        elif T_actual > self.max_fbank_frames:
            padded_feat = padded_feat[:, :self.max_fbank_frames, :]
            feat_lengths = feat_lengths.clamp(max=self.max_fbank_frames)

        # 2. 生成 conv_mask / attn_mask（与 audio_encoder_xh2a_export 一致）
        total_conv_len = self._compute_conv_output_length(self.max_fbank_frames)
        conv_mask = torch.zeros(batch_size, 1, total_conv_len, dtype=dtype, device=device)
        attn_mask = torch.full(
            (batch_size, 1, 1, total_conv_len),
            fill_value=-65504.0,
            dtype=dtype,
            device=device,
        )
        for i in range(batch_size):
            valid_len = self._compute_conv_output_length(feat_lengths[i].item())
            conv_mask[i, :, :valid_len] = 1.0
            attn_mask[i, :, :, :valid_len] = 0.0

        # 3. 调用 audio backend
        # New interface: (fbank, attn_mask, conv_mask). Keep compatibility with
        # legacy wrappers that only accept (fbank, conv_mask).
        try:
            speech_features = self.audio_backend(padded_feat, attn_mask, conv_mask)
        except TypeError:
            speech_features = self.audio_backend(padded_feat, conv_mask)

        # 4. 计算输出长度
        speech_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            conv_len = self._compute_conv_output_length(feat_lengths[i].item())
            # Adapter 2x 下采样
            speech_lens[i] = conv_len // 2

        return speech_features, speech_lens


class FireRedASRModelReplacement:
    """将 xh2a 导出的模型替换到 FireRedASR 的推理流程中。
    
    Usage:
        from fireredasr.models.fireredasr import FireRedAsr
        
        # 1. 加载原始模型
        asr_model = FireRedAsr.from_pretrained("llm", model_dir)
        
        # 2. 替换 audio encoder
        replacement = FireRedASRModelReplacement()
        replacement.replace_audio_encoder(
            asr_model.model,
            audio_backend="onnx",  # or "hmonnx"
            onnx_path="path/to/audio_encoder.onnx",
        )
        
        # 3. 替换 LLM (可选)
        replacement.replace_llm(
            asr_model.model,
            llm_backend="hmonnx",
            hmonnx_work_dir="path/to/work_dir",
        )
        
        # 4. 正常调用
        results = asr_model.transcribe(...)
    """

    @staticmethod
    def replace_audio_encoder(
        firered_model,
        audio_backend: str = "onnx",
        onnx_path: Optional[str] = None,
        hmonnx_path: Optional[str] = None,
        device: str = "cpu",
        max_fbank_frames: int = 3000,
    ):
        """替换 FireRedASR 的 audio encoder + adapter。
        
        Args:
            firered_model: FireRedAsrLlm 实例
            audio_backend: "onnx" 或 "hmonnx"
            onnx_path: ONNX 模型路径
            hmonnx_path: HMONNX 模型路径
            device: 推理设备
            max_fbank_frames: 最大 fbank 帧数
        """
        if audio_backend == "onnx":
            assert onnx_path is not None, "onnx_path required for onnx backend"
            try:
                from examples.audio.fireredasr.audio_encoder_xh2a_export import ONNXAudioEncoder
            except ModuleNotFoundError:
                from audio_encoder_xh2a_export import ONNXAudioEncoder
            backend = ONNXAudioEncoder(onnx_path, device=device)
        elif audio_backend == "hmonnx":
            assert hmonnx_path is not None, "hmonnx_path required for hmonnx backend"
            try:
                from examples.audio.fireredasr.audio_encoder_xh2a_export import HMONNXAudioEncoder
            except ModuleNotFoundError:
                from audio_encoder_xh2a_export import HMONNXAudioEncoder
            backend = HMONNXAudioEncoder(hmonnx_path, device=device)
        else:
            raise ValueError(f"Unknown audio backend: {audio_backend}")

        wrapper = FireRedASRAudioEncoderWrapper(backend, max_fbank_frames)

        # 修改 transcribe 的调用方式: 替换 encoder 和 adapter
        firered_model._original_encoder = firered_model.encoder
        firered_model._original_adapter = firered_model.encoder_projector
        firered_model._audio_wrapper = wrapper

        # Monkey-patch transcribe
        original_transcribe = firered_model.transcribe.__func__

        def patched_transcribe(self, padded_feat, feat_lengths, padded_input_ids, attention_mask,
                               beam_size=1, decode_max_len=0, decode_min_len=0,
                               repetition_penalty=1.0, llm_length_penalty=1.0, temperature=1.0):
            # 使用替换后的 audio encoder
            speech_features, speech_lens = self._audio_wrapper(padded_feat, feat_lengths)
            inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

            inputs_embeds, attention_mask, _ = \
                self._merge_input_ids_with_speech_features(
                    speech_features.to(inputs_embeds.dtype), inputs_embeds,
                    padded_input_ids, attention_mask, speech_lens=speech_lens
                )

            max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
            max_new_tokens = max(1, max_new_tokens)

            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                num_beams=beam_size,
                do_sample=False,
                min_length=decode_min_len,
                top_p=1.0,
                repetition_penalty=repetition_penalty,
                length_penalty=llm_length_penalty,
                temperature=temperature,
                bos_token_id=self.llm.config.bos_token_id,
                eos_token_id=self.llm.config.eos_token_id,
                pad_token_id=self.llm.config.pad_token_id,
            )
            return generated_ids

        import types
        firered_model.transcribe = types.MethodType(patched_transcribe, firered_model)
        print(f"Audio encoder replaced with {audio_backend} backend")

    @staticmethod
    def replace_llm_with_hf_compatible(
        firered_model,
        xh_model,
        device: str = "cuda:0",
    ):
        """替换 FireRedASR 的 LLM 为 xh2a wrap/quant 后的 HF compatible 模型。
        
        Args:
            firered_model: FireRedAsrLlm 实例
            xh_model: XHQwen2LegacyModel 实例（已完成 wrap 和量化）
            device: 推理设备
        """
        from xh_model_zoo.xh_llm.models.qwen2_legacy import Qwen2_HFCompatible

        # 获取原始 HF model（用于结构）
        original_llm = firered_model.llm
        hf_model_dir = str(Path(firered_model.llm.config._name_or_path))

        # 将 xh_model 转换为 HF compatible
        hf_compatible = Qwen2_HFCompatible.to_hf_compatible(
            original_llm, xh_model
        )

        # 替换 LLM
        firered_model.llm = hf_compatible
        firered_model.llm.to(device)
        print("LLM replaced with HF compatible xh2a model")

    @staticmethod
    def replace_llm_with_hmonnx(
        firered_model,
        hmonnx_work_dir: str,
        hf_model_dir: str,
        device: str = "cuda:0",
        rotated_adapter_path: Optional[str] = None,
    ):
        """替换 FireRedASR 的 LLM 为 HMONNX 推理模型。
        
        Args:
            firered_model: FireRedAsrLlm 实例
            hmonnx_work_dir: HMONNX 导出的工作目录（包含 meta info）
            hf_model_dir: HF 模型目录（用于 tokenizer 和 config）
            device: 推理设备
        """
        import json
        import torch
        import torch.nn as nn
        from xh_model_zoo.xh_llm.models.llm_onnx_model import LLMONNXModel
        try:
            from xh_model_zoo.xh_llm.models.llm_onnx_model import LLMLoRAONNXModel
        except ImportError:
            LLMLoRAONNXModel = LLMONNXModel

        meta_info_path = Path(hmonnx_work_dir) / "export_meta_info.json"
        with open(meta_info_path) as f:
            meta_info = json.load(f)

        # 构建 HMONNX LLM 模型
        lora_mode = meta_info.get("lora_mode", "merge_lora")
        onnx_cls = LLMLoRAONNXModel if lora_mode == "keep_lora" else LLMONNXModel
        use_lora_mask = bool(meta_info.get("keep_lora_use_mask", True))
        prefill_seq_len = int(meta_info.get("wrap_cfg", {}).get("input_sequence_length", 256))
        llm_kwargs = dict(
            prefill=dict(
                onnx=str(Path(hmonnx_work_dir) / meta_info["prefill_onnx_file"]),
                input_sequence_length=prefill_seq_len,
            ),
            decode=dict(
                onnx=str(Path(hmonnx_work_dir) / meta_info["decode_onnx_file"]),
            ),
            kv_cache=dict(
                num_hidden_layers=meta_info["num_hidden_layers"],
                shape=meta_info["kv_cache_shape"],
            ),
        )
        if lora_mode == "keep_lora":
            llm_kwargs["use_lora_mask"] = use_lora_mask
        llm_onnx_model = onnx_cls(**llm_kwargs)

        token_embedding_state_dict = torch.load(
            str(Path(hmonnx_work_dir) / meta_info["token_embedding_file"]),
            map_location="cpu",
            weights_only=True,
        )
        token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        token_embedding.load_state_dict(token_embedding_state_dict)
        llm_onnx_model.set_input_embeddings(token_embedding)
        llm_onnx_model.set_exec_device(device)
        llm_onnx_model.to(device)
        llm_onnx_model.to(torch.float16)

        # 包装为支持 generate 的接口
        firered_model.llm = HMONNXLLMWrapper(llm_onnx_model, firered_model.llm.config, device)
        if rotated_adapter_path is not None and Path(rotated_adapter_path).exists():
            rotated_adapter_sd = load_safetensors_file(str(rotated_adapter_path))
            missing, unexpected = firered_model.encoder_projector.load_state_dict(rotated_adapter_sd, strict=False)
            print(
                f"Loaded rotated audio projector: {rotated_adapter_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        print(f"LLM replaced with HMONNX model (lora_mode={lora_mode})")


class HMONNXLLMWrapper(nn.Module):
    """将 HMONNX LLM 模型包装为兼容 FireRedASR transcribe 调用的接口。
    
    FireRedASR 的 transcribe 调用:
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)
        generated_ids = self.llm.generate(inputs_embeds=inputs_embeds, ...)
    """

    def __init__(self, hmonnx_model, config, device: str = "cpu"):
        super().__init__()
        self.hmonnx_model = hmonnx_model
        self.config = config
        self._device = device
        self._embed_tokens = None

    def get_input_embeddings(self):
        """返回 token embedding 层。"""
        if self._embed_tokens is None:
            self._embed_tokens = self.hmonnx_model.token_embedding
        return self._embed_tokens

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds=None,
        max_new_tokens=100,
        num_beams=1,
        do_sample=False,
        min_length=0,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        **kwargs,
    ):
        """使用 HMONNX 模型进行 auto-regressive 生成。"""
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required for HMONNXLLMWrapper.generate")
        device = inputs_embeds.device

        # Reset kv cache before prefill.
        for cache in self.hmonnx_model.past_key_caches:
            cache.reset()
        for cache in self.hmonnx_model.past_value_caches:
            cache.reset()

        # Prefill can exceed static prefill_seq_len (e.g. long speech features).
        # Run prefill in chunks and keep kv-cache continuity with past_seq_length.
        seq_total = int(inputs_embeds.size(1))
        prefill_chunk = int(self.hmonnx_model.prefill_input_sequence_length)
        if prefill_chunk <= 0:
            prefill_chunk = seq_total

        logits = None
        last_chunk_valid_len = 0
        for chunk_start in range(0, seq_total, prefill_chunk):
            chunk_end = min(chunk_start + prefill_chunk, seq_total)
            chunk_inputs = inputs_embeds[:, chunk_start:chunk_end, :]
            last_chunk_valid_len = int(chunk_end - chunk_start)
            prefill_data = {
                "inputs_embeds": chunk_inputs,
                "past_seq_length": chunk_start,
                "input_sequence_length": prefill_chunk,
            }
            (
                prefill_inputs_embeds,
                prefill_past_seq_length,
                prefill_seq_length,
                prefill_k_caches,
                prefill_v_caches,
                *prefill_extra_args,
            ) = self.hmonnx_model.prepare_inputs(prefill_data)
            prefill_inputs = [prefill_inputs_embeds, prefill_past_seq_length, prefill_seq_length]
            prefill_inputs += prefill_k_caches
            prefill_inputs += prefill_v_caches
            prefill_inputs += prefill_extra_args
            logits = self.hmonnx_model.prefill_session(*prefill_inputs)

        if logits is None:
            return torch.empty((1, 0), dtype=torch.long, device=device)
        if logits.shape[1] > 1:
            last_idx = min(max(last_chunk_valid_len - 1, 0), logits.shape[1] - 1)
            logits = logits[:, last_idx : last_idx + 1, :]

        generated_ids = []
        past_seq_len = seq_total
        for step in range(max_new_tokens):
            # 取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]

            # 应用 repetition penalty
            if repetition_penalty != 1.0 and len(generated_ids) > 0:
                prev_tokens = torch.tensor(generated_ids, device=device).unsqueeze(0)
                for token_id in prev_tokens[0]:
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

            # 应用 temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            # 检查 EOS
            if next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)

            # Decode step
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            next_embeds = self.get_input_embeddings()(next_token_tensor)
            decode_data = {
                "inputs_embeds": next_embeds,
                "past_seq_length": past_seq_len,
                "input_sequence_length": 1,
            }
            (
                decode_inputs_embeds,
                decode_past_seq_length,
                decode_seq_length,
                decode_k_caches,
                decode_v_caches,
                *decode_extra_args,
            ) = self.hmonnx_model.prepare_inputs(decode_data)
            decode_inputs = [decode_inputs_embeds, decode_past_seq_length, decode_seq_length]
            decode_inputs += decode_k_caches
            decode_inputs += decode_v_caches
            decode_inputs += decode_extra_args
            logits = self.hmonnx_model.decode_session(*decode_inputs)
            past_seq_len += 1

        return torch.tensor([generated_ids], device=device)


def build_fireredasr_with_xh2a(
    fireredasr_model_dir: str,
    audio_onnx_path: Optional[str] = None,
    audio_hmonnx_path: Optional[str] = None,
    llm_hmonnx_work_dir: Optional[str] = None,
    rotated_adapter_path: Optional[str] = None,
    device: str = "cuda:0",
):
    """一站式构建使用 xh2a 后端的 FireRedASR 模型。
    
    Args:
        fireredasr_model_dir: FireRedASR 模型目录
        audio_onnx_path: Audio Encoder ONNX 路径 (优先使用 HMONNX)
        audio_hmonnx_path: Audio Encoder HMONNX 路径
        llm_hmonnx_work_dir: LLM HMONNX 工作目录
        device: 推理设备
    
    Returns:
        FireRedAsr 实例（已替换后端）
    """
    from fireredasr.models.fireredasr import FireRedAsr

    # 加载原始模型
    asr_model = FireRedAsr.from_pretrained("llm", fireredasr_model_dir)

    replacement = FireRedASRModelReplacement()

    # 替换 Audio Encoder
    if audio_hmonnx_path:
        replacement.replace_audio_encoder(
            asr_model.model,
            audio_backend="hmonnx",
            hmonnx_path=audio_hmonnx_path,
            device=device,
        )
    elif audio_onnx_path:
        replacement.replace_audio_encoder(
            asr_model.model,
            audio_backend="onnx",
            onnx_path=audio_onnx_path,
            device=device,
        )

    # 替换 LLM
    if llm_hmonnx_work_dir:
        hf_model_dir = str(Path(fireredasr_model_dir) / "Qwen2-7B-Instruct")
        replacement.replace_llm_with_hmonnx(
            asr_model.model,
            hmonnx_work_dir=llm_hmonnx_work_dir,
            hf_model_dir=hf_model_dir,
            device=device,
            rotated_adapter_path=rotated_adapter_path,
        )

    return asr_model


def _load_ref_texts(ref_file: Optional[str]) -> Dict[str, str]:
    if ref_file is None or not Path(ref_file).exists():
        return {}
    ref_texts = {}
    with open(ref_file) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                ref_texts[parts[0]] = parts[1].replace(" ", "")
    return ref_texts


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    n, m = len(ref), len(hyp)
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


def _collect_wav_paths(wav_path_args: List[str]) -> List[str]:
    paths: List[str] = []
    for pattern in wav_path_args:
        if any(ch in pattern for ch in ["*", "?", "["]):
            paths.extend(sorted(glob.glob(pattern)))
        else:
            p = Path(pattern)
            if p.is_dir():
                paths.extend(sorted(str(x) for x in p.glob("*.wav")))
            elif p.exists():
                paths.append(str(p))
    return paths


def _run_cli():
    from fireredasr.models.fireredasr import FireRedAsr

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, choices=["hf", "hmonnx"], default="hf")
    parser.add_argument("--model_dir", type=str, required=True, help="FireRedASR model dir")
    parser.add_argument("--wav_path", type=str, nargs="+", required=True, help="wav path(s), glob, or directory")
    parser.add_argument("--llm_hmonnx_dir", type=str, default=None, help="HMONNX LLM export work dir")
    parser.add_argument("--audio_onnx_path", type=str, default=None, help="optional audio encoder onnx path")
    parser.add_argument("--audio_hmonnx_path", type=str, default=None, help="optional audio encoder hmonnx path")
    parser.add_argument("--rotated_adapter_path", type=str, default=None, help="optional rotated audio projector")
    parser.add_argument("--ref_file", type=str, default=None, help="optional reference text: uttid text")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=0)
    parser.add_argument("--decode_min_len", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--llm_length_penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out_json", type=str, default=None, help="optional output json path")
    args = parser.parse_args()

    wav_paths = _collect_wav_paths(args.wav_path)
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found from --wav_path: {args.wav_path}")
    uttids = [Path(p).stem for p in wav_paths]

    device = "cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu"
    infer_args = {
        "use_gpu": args.use_gpu and torch.cuda.is_available(),
        "beam_size": args.beam_size,
        "decode_max_len": args.decode_max_len,
        "decode_min_len": args.decode_min_len,
        "repetition_penalty": args.repetition_penalty,
        "llm_length_penalty": args.llm_length_penalty,
        "temperature": args.temperature,
    }

    if args.mode == "hf":
        asr_model = FireRedAsr.from_pretrained("llm", args.model_dir)
    else:
        if args.llm_hmonnx_dir is None:
            raise ValueError("--llm_hmonnx_dir is required in hmonnx mode")
        asr_model = build_fireredasr_with_xh2a(
            fireredasr_model_dir=args.model_dir,
            audio_onnx_path=args.audio_onnx_path,
            audio_hmonnx_path=args.audio_hmonnx_path,
            llm_hmonnx_work_dir=args.llm_hmonnx_dir,
            rotated_adapter_path=args.rotated_adapter_path,
            device=device,
        )
    if infer_args["use_gpu"]:
        asr_model.model.cuda()

    ref_texts = _load_ref_texts(args.ref_file)
    results = []
    total_ref_chars = 0
    total_ref_edit = 0
    for uttid, wav_path in zip(uttids, wav_paths):
        one = asr_model.transcribe([uttid], [wav_path], infer_args)[0]
        text = one["text"]
        item = {"uttid": uttid, "wav_path": wav_path, "text": text}
        ref = ref_texts.get(uttid, None)
        if ref is not None and len(ref) > 0:
            cer = _edit_distance(list(ref), list(text)) / len(ref)
            item["ref_text"] = ref
            item["cer"] = cer
            total_ref_chars += len(ref)
            total_ref_edit += _edit_distance(list(ref), list(text))
        print(f"{uttid}\t{text}")
        results.append(item)

    summary = {
        "mode": args.mode,
        "num_utts": len(results),
        "cer": (total_ref_edit / total_ref_chars) if total_ref_chars > 0 else None,
        "results": results,
    }
    if args.out_json is not None:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved results to {args.out_json}")
    if summary["cer"] is not None:
        print(f"CER={summary['cer']:.6f}")


if __name__ == "__main__":
    _run_cli()
