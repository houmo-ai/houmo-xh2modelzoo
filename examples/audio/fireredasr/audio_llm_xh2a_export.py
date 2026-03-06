"""
FireRedASR LLM (Qwen2-7B + LoRA) 导出脚本

支持两种 LoRA 处理方案:
  方案1 (--merge_lora): 融合 LoRA 权重到 Qwen2 基础模型中，导出标准 Qwen2 模型
  方案2 (--keep_lora):  保留 LoRA 分支，导出内联 LoRA 的模型

基于 qwen2_legacy 的 LLM 导出流程，适配 FireRedASR 的权重加载方式。

输入（LLM 推理时）:
  - inputs_embeds: [B, S, 3584]  包含语音特征的 embedding 序列
  - past_seq_length: int          KV Cache 已有长度

输出:
  - logits: [B, 1, vocab_size]   LLM 的 logits 输出
"""

import argparse
import gc
import glob
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import xhquant.utils.suppress_printing
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from xhquant.api import (
    ConfigDict,
    FrontendType,
    Hook,
    PrecisionMode,
    QTensor,
    ptq_quantize,
    set_random_seed,
)

# Ensure local repo package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh_model_zoo.api import (
    Config,
    decode_next_token,
    get_root_logger,
    xhquant_llm_init,
)
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
import xh_model_zoo.xh_llm.models.qwen2_legacy  # noqa: F401
from xh_model_zoo.utils.time_profiler import time_profiler

# 添加 FireRedASR 路径
FIREREDASR_ROOT = REPO_ROOT / ".." / "FireRedASR"
sys.path.insert(0, str(FIREREDASR_ROOT))


def _is_valid_hf_model_dir(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").exists()
    has_tokenizer = (model_dir / "tokenizer_config.json").exists() or (model_dir / "tokenizer.json").exists()
    return has_config and has_tokenizer


def _resolve_hf_model_dir(cfg_hf_model_dir: str, fireredasr_model_dir: str, cli_hf_model_dir: Optional[str] = None) -> str:
    """Resolve a usable HF model directory for FireRedASR Qwen2 base model."""
    candidates: List[Path] = []

    fr_root = Path(fireredasr_model_dir).expanduser()
    firered_internal = fr_root / "Qwen2-7B-Instruct"

    # Priority rule:
    # 1) explicit --hf_model_dir
    # 2) FireRedASR bundled Qwen2 (when --hf_model_dir is not set)
    # 3) config/default fallbacks
    if cli_hf_model_dir:
        candidates.append(Path(cli_hf_model_dir).expanduser())
    else:
        candidates.append(firered_internal)

    if cfg_hf_model_dir:
        candidates.append(Path(cfg_hf_model_dir).expanduser())

    candidates.extend(
        [
            firered_internal,
            Path("weights") / "Qwen2-7B-Instruct",
            Path("/data01/datasets/FireRedASR2-LLM/Qwen2-7B-Instruct"),
        ]
    )

    seen = set()
    checked: List[str] = []
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if _is_valid_hf_model_dir(resolved):
            chosen = path.expanduser()
            if chosen.exists():
                return str(chosen)
            return str(resolved)

    raise FileNotFoundError(
        "Cannot find a valid Qwen2 HF model directory. Checked:\n"
        + "\n".join(f"  - {p}" for p in checked)
        + "\nPlease pass --hf_model_dir explicitly."
    )


def to_device(inputs, device):
    if isinstance(inputs, Tensor):
        return inputs.to(device)
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)([to_device(x, device) for x in inputs])
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, QTensor):
        return inputs.to(device)
    else:
        return inputs


def flatten_inputs(args):
    new_args = []
    for arg in args:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    return new_args


def compute_tensor_diff_metrics(ref: Tensor, cur: Tensor) -> Dict[str, float]:
    ref_fp32 = ref.float().reshape(1, -1)
    cur_fp32 = cur.float().reshape(1, -1)
    diff = torch.abs(ref_fp32 - cur_fp32)
    cosine = F.cosine_similarity(ref_fp32, cur_fp32, dim=1).item()
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "cosine_sim": cosine,
    }


def log_logits_compare(logger, name: str, ref_logits: Tensor, cur_logits: Tensor, meta_info: ConfigDict):
    metrics = compute_tensor_diff_metrics(ref_logits, cur_logits)
    logger.info(
        f"{name} logits diff: max={metrics['max_diff']:.6e}, "
        f"mean={metrics['mean_diff']:.6e}, cosine={metrics['cosine_sim']:.6f}"
    )
    meta_info[name] = metrics


def _cast_quant_weight_tensor(v: Tensor) -> Tensor:
    if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
        return v.to(torch.int8)
    if v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
        return v.to(torch.int16)
    return v.to(torch.float32)


def _load_resume_state_dict_with_quant_weight(
    native_model: nn.Module,
    archive_file: str,
    logger,
    strict: bool = True,
) -> None:
    """Load checkpoint and explicitly inject *.quant_weight into corresponding Linear modules."""
    logger.info(f"Load previously saved checkpoint from: {archive_file}")
    is_safetensors = archive_file.endswith(".safetensors")
    if is_safetensors:
        state_dict = load_safetensors_file(archive_file, device="cpu")
    else:
        state_dict = torch.load(archive_file, weights_only=True, map_location="cpu")

    model_state_dict = native_model.state_dict()
    unexpected_keys = [k for k in state_dict.keys() if k not in model_state_dict]
    injected_quant_weight = 0
    for key in unexpected_keys:
        if not key.endswith(".quant_weight"):
            logger.warning(f"ignore unexpect state dict: {key}")
            state_dict.pop(key)
            continue

        submodule_name = key[: -len(".quant_weight")]
        submodule = native_model.get_submodule(submodule_name)
        q_weight = _cast_quant_weight_tensor(state_dict[key])
        if "quant_weight" in submodule._buffers:
            submodule._buffers["quant_weight"] = q_weight
        else:
            submodule.register_buffer("quant_weight", q_weight, persistent=False)
        injected_quant_weight += 1
        state_dict.pop(key)

    missing_keys, load_unexpected_keys = native_model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded quant_weight tensors into native linear modules: {injected_quant_weight}")
    if missing_keys:
        logger.warning(f"load_state_dict missing keys: {len(missing_keys)}; first 5: {missing_keys[:5]}")
    if load_unexpected_keys:
        logger.warning(
            f"load_state_dict unexpected keys: {len(load_unexpected_keys)}; first 5: {load_unexpected_keys[:5]}"
        )
    if strict and (missing_keys or load_unexpected_keys):
        raise RuntimeError(
            f"Strict load failed for checkpoint {archive_file}: "
            f"missing={len(missing_keys)}, unexpected={len(load_unexpected_keys)}"
        )
    del state_dict


def _attach_keep_lora_runtime(
    xh_model: LLMBaseModel,
    data_batch: Dict[str, Any],
    lora_alpha: float,
):
    """Inject keep_lora branch into frontend graph with fixed enable scale."""
    from xh_model_zoo.xh_llm.models.lora_layer import apply_lora_to_linear

    frontend_inputs = xh_model.prepare_inputs_for_graph(data_batch)
    frontend_inputs = flatten_inputs(frontend_inputs)
    # keep_lora mode always enables LoRA branch in runtime for FireRedASR export.
    apply_lora_to_linear(
        xh_model.frontend_model,
        frontend_inputs,
        lora_scale=lora_alpha,
        runtime_mask=False,
    )


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
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


def _find_wav_dir() -> Optional[str]:
    for candidate in ["data/wav", "data/wavdata/wav", str(FIREREDASR_ROOT / "examples" / "wav")]:
        p = Path(candidate)
        if p.exists() and list(p.glob("*.wav")):
            return str(p)
    return None


def _find_ref_text(wav_dir: Optional[str]) -> Optional[str]:
    if wav_dir is None:
        return None
    text_file = Path(wav_dir) / "text"
    if text_file.exists():
        return str(text_file)
    return None


@torch.no_grad()
def validate_asr_hmonnx_llm(
    fireredasr_model_dir: str,
    hmonnx_work_dir: str,
    wav_dir: Optional[str],
    ref_text_file: Optional[str],
    use_gpu: bool,
    logger,
    rotated_adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    from fireredasr.models.fireredasr import FireRedAsr
    try:
        from examples.audio.fireredasr.fireredasr_hf_forward import FireRedASRModelReplacement
    except ModuleNotFoundError:
        # Some environments install an `examples` package from other repos.
        # Fall back to local sibling script to avoid import shadowing.
        forward_file = Path(__file__).resolve().parent / "fireredasr_hf_forward.py"
        spec = importlib.util.spec_from_file_location("fireredasr_hf_forward_local", forward_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load FireRedASRModelReplacement from {forward_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        FireRedASRModelReplacement = module.FireRedASRModelReplacement

    wav_dir = wav_dir or _find_wav_dir()
    if wav_dir is None:
        logger.warning("No wav_dir found, skip ASR validation.")
        return {"skipped": True, "reason": "no_wav_dir"}

    wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_paths:
        logger.warning(f"No wav files found in {wav_dir}, skip ASR validation.")
        return {"skipped": True, "reason": "no_wav_files"}

    ref_text_file = ref_text_file or _find_ref_text(wav_dir)
    ref_texts = _load_ref_texts(ref_text_file)

    infer_args = {
        "use_gpu": use_gpu,
        "beam_size": 1,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 1.0,
        "llm_length_penalty": 0.0,
        "temperature": 1.0,
    }
    uttids = [Path(p).stem for p in wav_paths]

    native_model = FireRedAsr.from_pretrained("llm", fireredasr_model_dir)
    if use_gpu and torch.cuda.is_available():
        native_model.model.cuda()
    # Keep native/hmonnx path consistent: run per-utterance transcribe for both.
    native_results = []
    for uttid, wav_path in zip(uttids, wav_paths):
        native_results.extend(native_model.transcribe([uttid], [wav_path], infer_args))
    native_texts = {item["uttid"]: item["text"] for item in native_results}
    if use_gpu and torch.cuda.is_available():
        native_model.model.cpu()
        torch.cuda.empty_cache()
    del native_model
    gc.collect()

    hmonnx_model = FireRedAsr.from_pretrained("llm", fireredasr_model_dir)
    replacement = FireRedASRModelReplacement()
    hf_model_dir = str(Path(fireredasr_model_dir) / "Qwen2-7B-Instruct")
    replacement.replace_llm_with_hmonnx(
        hmonnx_model.model,
        hmonnx_work_dir=hmonnx_work_dir,
        hf_model_dir=hf_model_dir,
        device="cuda:0" if use_gpu and torch.cuda.is_available() else "cpu",
    )
    if rotated_adapter_path is not None and Path(rotated_adapter_path).exists():
        rotated_adapter_sd = load_safetensors_file(str(rotated_adapter_path))
        missing, unexpected = hmonnx_model.model.encoder_projector.load_state_dict(rotated_adapter_sd, strict=False)
        logger.info(
            f"Loaded rotated audio projector from {rotated_adapter_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    if use_gpu and torch.cuda.is_available():
        hmonnx_model.model.cuda()
    hmonnx_results = []
    for uttid, wav_path in zip(uttids, wav_paths):
        single_result = hmonnx_model.transcribe([uttid], [wav_path], infer_args)
        hmonnx_results.extend(single_result)
    hmonnx_texts = {item["uttid"]: item["text"] for item in hmonnx_results}

    details = []
    total_chars = 0
    total_edit = 0
    total_ref_chars = 0
    total_ref_edit = 0
    for uttid in uttids:
        native_text = native_texts.get(uttid, "")
        hmonnx_text = hmonnx_texts.get(uttid, "")
        is_match = native_text == hmonnx_text

        cer_nat_hmx = 0.0
        if len(native_text) > 0:
            edit = _edit_distance(list(native_text), list(hmonnx_text))
            total_edit += edit
            total_chars += len(native_text)
            cer_nat_hmx = edit / len(native_text)

        cer_ref_native = None
        cer_ref_hmonnx = None
        ref_text = ref_texts.get(uttid, None)
        if ref_text is not None and len(ref_text) > 0:
            edit_native = _edit_distance(list(ref_text), list(native_text))
            edit_hmonnx = _edit_distance(list(ref_text), list(hmonnx_text))
            cer_ref_native = edit_native / len(ref_text)
            cer_ref_hmonnx = edit_hmonnx / len(ref_text)
            total_ref_chars += len(ref_text)
            total_ref_edit += edit_hmonnx

        logger.info(
            f"[ASR-CMP] {uttid}: match={is_match}, cer(native->hmonnx)={cer_nat_hmx:.4f}, "
            f"native='{native_text}', hmonnx='{hmonnx_text}'"
        )
        details.append(
            dict(
                uttid=uttid,
                native_text=native_text,
                hmonnx_text=hmonnx_text,
                ref_text=ref_text,
                match=is_match,
                cer_native_vs_hmonnx=cer_nat_hmx,
                cer_ref_native=cer_ref_native,
                cer_ref_hmonnx=cer_ref_hmonnx,
            )
        )

    summary = {
        "num_utts": len(uttids),
        "num_match": sum(1 for item in details if item["match"]),
        "match_rate": sum(1 for item in details if item["match"]) / max(len(details), 1),
        "cer_native_vs_hmonnx": (total_edit / total_chars) if total_chars > 0 else None,
        "cer_ref_hmonnx": (total_ref_edit / total_ref_chars) if total_ref_chars > 0 else None,
        "details": details,
    }
    logger.info(
        f"ASR compare summary: match_rate={summary['match_rate']:.4f}, "
        f"cer(native->hmonnx)={summary['cer_native_vs_hmonnx']}"
    )
    return summary


def _generate_golden(cfg, input_ids: torch.Tensor, tokenizer, prefill_onnx_file: str, decode_onnx_file: str, logger):
    from xhquant.api import HMONNXGoldenInference

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_sequence_length = cfg.model.wrap_cfg.input_sequence_length
    valid_len = input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dtype = getattr(torch, cfg.dtype)

    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    token_embedding_sd = torch.load(str(token_embedding_file), map_location="cpu", weights_only=True)
    vocab_size, embed_dim = token_embedding_sd["weight"].shape
    token_embedding = nn.Embedding(vocab_size, embed_dim)
    token_embedding.load_state_dict(token_embedding_sd)
    token_embedding = token_embedding.to(device).to(dtype).eval()

    if valid_len < input_sequence_length:
        input_ids_pad = torch.cat(
            [input_ids, torch.full((input_ids.shape[0], input_sequence_length - valid_len), pad_token_id, dtype=torch.long)],
            dim=-1,
        )
    else:
        input_ids_pad = input_ids
    prefill_inputs_embeds = token_embedding(input_ids_pad.to(device))

    prefill_golden_dir = Path(prefill_onnx_file).parent / "golden" / Path(prefill_onnx_file).stem
    prefill_golden_dir.mkdir(exist_ok=True, parents=True)
    prefill_model = HMONNXGoldenInference(prefill_onnx_file)
    prefill_model.save_golden = True
    prefill_model.exec_device = str(device)
    prefill_model.golden_dir = str(prefill_golden_dir)
    prefill_model.to(str(device))
    prefill_model.initialize()

    prefill_input_feed: Dict[str, torch.Tensor] = {}
    prefill_cache_feed: Dict[str, torch.Tensor] = {}
    for name in prefill_model.get_input_names():
        info = prefill_model.get_input(name)
        if name in ("inputs_embeds", "input_1"):
            prefill_input_feed[name] = prefill_inputs_embeds
        elif name in ("past_seq_length", "valid_length"):
            prefill_input_feed[name] = torch.tensor([0], dtype=info.dtype, device=device)
        elif name in ("current_input_length", "current_length"):
            prefill_input_feed[name] = torch.tensor([valid_len], dtype=info.dtype, device=device)
        else:
            t = torch.zeros(info.shape, dtype=info.dtype, device=device)
            prefill_input_feed[name] = t
            prefill_cache_feed[name] = t
    prefill_out = prefill_model.forward(*[prefill_input_feed[n] for n in prefill_model.get_input_names()])
    if isinstance(prefill_out, (list, tuple)):
        prefill_logits = prefill_out[0]
    else:
        prefill_logits = prefill_out

    decode_token_id = prefill_logits[:, -1:, :].argmax(dim=-1)
    decode_inputs_embeds = token_embedding(decode_token_id.to(device))

    decode_golden_dir = Path(decode_onnx_file).parent / "golden" / Path(decode_onnx_file).stem
    decode_golden_dir.mkdir(exist_ok=True, parents=True)
    decoder_model = HMONNXGoldenInference(decode_onnx_file)
    decoder_model.save_golden = True
    decoder_model.exec_device = str(device)
    decoder_model.golden_dir = str(decode_golden_dir)
    decoder_model.to(str(device))
    decoder_model.initialize()

    decode_input_feed: Dict[str, torch.Tensor] = {}
    for name in decoder_model.get_input_names():
        info = decoder_model.get_input(name)
        if name in ("inputs_embeds", "input_1"):
            decode_input_feed[name] = decode_inputs_embeds
        elif name in ("past_seq_length", "valid_length"):
            decode_input_feed[name] = torch.tensor([valid_len], dtype=info.dtype, device=device)
        elif name in ("current_input_length", "current_length"):
            decode_input_feed[name] = torch.tensor([1], dtype=info.dtype, device=device)
        elif name in prefill_cache_feed:
            decode_input_feed[name] = prefill_cache_feed[name]
        else:
            decode_input_feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=device)
    decoder_model.forward(*[decode_input_feed[n] for n in decoder_model.get_input_names()])
    logger.info(f"Export prefill/decode golden to {prefill_golden_dir} and {decode_golden_dir}")


def _pack_golden(onnx_file: str, logger):
    golden_dir = Path(onnx_file).parent / "golden" / Path(onnx_file).stem
    if not golden_dir.exists():
        logger.warning(f"Golden directory not found: {golden_dir}")
        return
    tar_file = golden_dir.parent / f"{golden_dir.name}.tar.gz"
    with tarfile.open(str(tar_file), "w:gz") as tar:
        tar.add(str(golden_dir), arcname=golden_dir.name)
    logger.info(f"Packed golden: {tar_file}")


class PingPangGPUHook(Hook):
    """CPU/GPU 乒乓方式执行，用于大模型导出时节省显存。"""

    def __init__(self, name, device) -> None:
        super().__init__()
        self.name = name
        self.device = device
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.8, device)
            torch.cuda.max_split_size_mb = 128

    def after_node_run(self, graph_module, graph, node, output, args, kwargs):
        module = None
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            module.to("cpu")
        if module is not None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        return output

    def before_node_run(self, graph_module, graph, node, args, kwargs):
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            module.to(self.device)
        if node.op not in ["placeholder", "getattr"]:
            args = to_device(args, self.device)
            kwargs = to_device(kwargs, self.device)
        return args, kwargs


def load_fireredasr_lora_weights(model_dir: str):
    """加载 FireRedASR 的 LoRA 权重和配置。
    
    Returns:
        lora_state_dict: dict - LoRA 权重 (keys 含 lora_A/lora_B)
        lora_config: dict - LoRA 配置信息
    """
    model_path = Path(model_dir) / "model.pth.tar"
    package = torch.load(str(model_path), map_location="cpu", weights_only=False)

    # 从 checkpoint 中提取 LoRA 相关权重
    state_dict = package["model_state_dict"]
    lora_state_dict = {}
    encoder_state_dict = {}
    adapter_state_dict = {}

    for key, value in state_dict.items():
        if "lora_" in key:
            # FireRedASR 的 LoRA key 格式: llm.base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
            # 需要转换为 model.layers.X.self_attn.q_proj.weight_lora_a/weight_lora_b 格式
            lora_state_dict[key] = value
        elif key.startswith("encoder.") or key.startswith("encoder_projector."):
            # encoder 和 adapter 权重，分开存储
            if key.startswith("encoder."):
                encoder_state_dict[key[len("encoder."):]] = value
            else:
                adapter_state_dict[key[len("encoder_projector."):]] = value

    lora_config = {
        "r": 64,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
        "lora_dropout": 0.05,
    }

    args = package.get("args", None)

    return lora_state_dict, lora_config, encoder_state_dict, adapter_state_dict, args


def _normalize_lora_module_path(module_path: str) -> str:
    if module_path.startswith("llm.base_model.model."):
        module_path = module_path[len("llm.base_model.model."):]
    elif module_path.startswith("llm."):
        module_path = module_path[len("llm."):]
    return module_path


def _build_lora_pairs(lora_state_dict: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
    lora_pairs: Dict[str, Dict[str, Tensor]] = {}
    for key, value in lora_state_dict.items():
        if "lora_A" in key:
            parts = key.split(".")
            lora_a_idx = parts.index("lora_A")
            module_path = _normalize_lora_module_path(".".join(parts[:lora_a_idx]))
            lora_pairs.setdefault(module_path, {})["lora_A"] = value
        elif "lora_B" in key:
            parts = key.split(".")
            lora_b_idx = parts.index("lora_B")
            module_path = _normalize_lora_module_path(".".join(parts[:lora_b_idx]))
            lora_pairs.setdefault(module_path, {})["lora_B"] = value
    return lora_pairs


def _get_lora_scaling(lora_config: Dict[str, Any]) -> float:
    return float(lora_config["lora_alpha"]) / float(lora_config["r"])


@torch.no_grad()
def _forward_with_temp_merged_lora(
    native_model: nn.Module,
    input_ids: Tensor,
    lora_state_dict: Dict[str, Tensor],
    lora_config: Dict[str, Any],
) -> Tensor:
    """Run one HF forward with temporary LoRA merge, then restore original weights."""
    scaling = _get_lora_scaling(lora_config)
    lora_pairs = _build_lora_pairs(lora_state_dict)
    applied_deltas: List[Tuple[nn.Linear, Tensor]] = []

    for module_path, lora_weights in lora_pairs.items():
        if "lora_A" not in lora_weights or "lora_B" not in lora_weights:
            continue
        try:
            module = native_model.get_submodule(module_path)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear):
            continue

        lora_A = lora_weights["lora_A"]
        lora_B = lora_weights["lora_B"]
        if lora_A.shape[0] != lora_B.shape[1]:
            lora_A = lora_A.T
            lora_B = lora_B.T

        calc_dtype = module.weight.dtype
        if module.weight.device.type == "cpu" and calc_dtype in (torch.float16, torch.bfloat16):
            calc_dtype = torch.float32
        lora_A = lora_A.to(device=module.weight.device, dtype=calc_dtype)
        lora_B = lora_B.to(device=module.weight.device, dtype=calc_dtype)

        expected_shape = tuple(module.weight.shape)
        delta_shape = (lora_B.shape[0], lora_A.shape[1])
        if delta_shape != expected_shape:
            continue

        delta = (lora_B @ lora_A) * scaling
        delta = delta.to(dtype=module.weight.dtype)
        module.weight.data.add_(delta)
        applied_deltas.append((module, delta))

    try:
        outputs = native_model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, -1:, :].detach()
    finally:
        for module, delta in reversed(applied_deltas):
            module.weight.data.sub_(delta)

    return logits


def merge_lora_into_base_model(native_model, lora_state_dict, lora_config):
    """将 LoRA 权重融合到基础模型的 Linear 层中。
    
    FireRedASR 的 LoRA key 格式:
        llm.base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight
        llm.base_model.model.model.layers.X.self_attn.q_proj.lora_B.default.weight
    
    目标模型的 key 格式:
        model.layers.X.self_attn.q_proj.weight
    """
    scaling = _get_lora_scaling(lora_config)
    lora_pairs = _build_lora_pairs(lora_state_dict)

    # 融合 LoRA 到基础模型
    merged_count = 0
    total_pairs = len(lora_pairs)
    for idx, (module_path, lora_weights) in enumerate(lora_pairs.items(), start=1):
        if "lora_A" not in lora_weights or "lora_B" not in lora_weights:
            print(f"  WARNING: Incomplete LoRA pair for {module_path}, skipping")
            continue

        try:
            module = native_model.get_submodule(module_path)
        except AttributeError:
            print(f"  WARNING: Module {module_path} not found in model, skipping")
            continue

        if not isinstance(module, nn.Linear):
            print(f"  WARNING: Module {module_path} is not Linear ({type(module)}), skipping")
            continue

        lora_A_raw = lora_weights["lora_A"]
        lora_B_raw = lora_weights["lora_B"]
        if lora_A_raw.shape[0] != lora_B_raw.shape[1]:
            lora_A_raw = lora_A_raw.T
            lora_B_raw = lora_B_raw.T

        # CPU float16 matmul is very slow; compute delta in fp32 then cast back.
        calc_dtype = module.weight.dtype
        if module.weight.device.type == "cpu" and calc_dtype in (torch.float16, torch.bfloat16):
            calc_dtype = torch.float32

        lora_A = lora_A_raw.to(device=module.weight.device, dtype=calc_dtype)
        lora_B = lora_B_raw.to(device=module.weight.device, dtype=calc_dtype)

        expected_shape = tuple(module.weight.shape)
        delta_shape = (lora_B.shape[0], lora_A.shape[1])
        if delta_shape != expected_shape:
            print(
                f"  WARNING: Shape mismatch for {module_path}: "
                f"delta={delta_shape}, weight={expected_shape}, skipping"
            )
            continue

        delta_weight = (lora_B @ lora_A) * scaling
        module.weight.data.add_(delta_weight.to(dtype=module.weight.dtype))
        merged_count += 1
        if idx % 32 == 0 or idx == total_pairs:
            print(f"  Merge progress: {idx}/{total_pairs}")

    print(f"  Merged {merged_count} LoRA pairs into base model")
    return native_model


def register_lora_buffers(native_model, lora_state_dict, lora_config):
    """将 LoRA 权重注册为 buffer（用于保留 LoRA 分支的方案）。
    
    将 lora_A/lora_B 权重注册到对应 Linear 模块的 buffer 中，
    后续在 FX graph 阶段通过 apply_lora_to_linear 实现运行时 LoRA 分支。
    """
    lora_pairs = _build_lora_pairs(lora_state_dict)

    registered_count = 0
    for module_path, lora_weights in lora_pairs.items():
        if "lora_A" not in lora_weights or "lora_B" not in lora_weights:
            continue

        try:
            module = native_model.get_submodule(module_path)
        except AttributeError:
            print(f"  WARNING: Module {module_path} not found in model, skipping")
            continue

        lora_A = lora_weights["lora_A"]
        lora_B = lora_weights["lora_B"]
        if lora_A.shape[0] != lora_B.shape[1]:
            lora_A = lora_A.T
            lora_B = lora_B.T
        module.register_buffer("weight_lora_a", lora_A)
        module.register_buffer("weight_lora_b", lora_B)
        registered_count += 1

    print(f"  Registered {registered_count} LoRA buffer pairs")
    return native_model


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    cfg_name: str,
    device,
    dtype,
    logger,
    valid: bool = True,
):
    """导出 HMONNX 模型。"""
    logger.info("Start exporting...")
    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(device)
        xh_model.to(dtype)
        data_batch["input_ids"] = data_batch["input_ids"].to(device)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            exported_logits = outs.logits.detach()
            next_tokens, next_token_str = decode_next_token(tokenizer, exported_logits)
        logger.info(f"Exported model next token: {next_tokens} {next_token_str}")

    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting ONNX ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _export_impl(cfg, args):
    """导出实现主函数。"""
    logger = get_root_logger()
    if getattr(args, "golden_only", False):
        logger.info("Golden-only mode: skip export, generate golden from existing ONNX")
        xh_model = MODELS.build(cfg.model)
        tokenizer = xh_model.get_tokenizer()
        del xh_model

        messages = [{"role": "user", "content": args.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([text], return_tensors="pt").input_ids

        prefill_onnx_file = str(Path(cfg.work_dir) / "prefill_onnx" / f"{cfg.cfg_name}_prefill.onnx")
        decode_onnx_file = str(Path(cfg.work_dir) / "decode_onnx" / f"{cfg.cfg_name}_decode.onnx")
        if not Path(prefill_onnx_file).exists():
            raise FileNotFoundError(f"Prefill ONNX not found: {prefill_onnx_file}")
        if not Path(decode_onnx_file).exists():
            raise FileNotFoundError(f"Decode ONNX not found: {decode_onnx_file}")

        _generate_golden(cfg, input_ids, tokenizer, prefill_onnx_file, decode_onnx_file, logger)
        if not getattr(args, "golden_skip_pack", False):
            _pack_golden(prefill_onnx_file, logger)
            _pack_golden(decode_onnx_file, logger)
        else:
            logger.info("Skip golden tar packing by --golden_skip_pack")
        logger.info("Golden-only mode done.")
        return

    only_export = not args.valid

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    xhquant.utils.suppress_printing.disable_printing = True

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
            lora_mode=args.lora_mode,
        )
    )

    cfg_name = cfg.cfg_name
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir
    meta_info["model_name"] = cfg_name
    meta_info["wrap_cfg"] = cfg.model.wrap_cfg.to_dict()

    # ============== 构建 xh_model ==============
    xh_model = MODELS.build(cfg.model)

    tokenizer = xh_model.get_tokenizer()
    native_model = xh_model.get_hf_model("cpu")

    # ============== 加载 FireRedASR LoRA 权重 ==============
    fireredasr_model_dir = args.fireredasr_model_dir
    logger.info(f"Loading FireRedASR LoRA weights from {fireredasr_model_dir}...")
    lora_state_dict, lora_config, encoder_sd, adapter_sd, firered_args = \
        load_fireredasr_lora_weights(fireredasr_model_dir)
    logger.info(f"  LoRA config: r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
    logger.info(f"  LoRA weights: {len(lora_state_dict)} keys")

    if args.lora_mode == "merge_lora":
        logger.info("Mode: Merge LoRA into base model")
        native_model = merge_lora_into_base_model(native_model, lora_state_dict, lora_config)
    elif args.lora_mode == "keep_lora":
        logger.info("Mode: Keep LoRA branch (runtime LoRA, fixed enabled)")
        native_model = register_lora_buffers(native_model, lora_state_dict, lora_config)
        meta_info.keep_lora_use_mask = False
    else:
        raise ValueError(f"Unknown lora_mode: {args.lora_mode}")

    # ============== 复制 HF 配置文件 ==============
    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "vocab.json", "tokenizer.json", "chat_template.jinja", "added_tokens.json",
    ]
    for cfg_file in hf_config_files:
        src_file = Path(hf_model_dir) / cfg_file
        dst_file = Path(hf_config_dir) / cfg_file
        if src_file.exists():
            shutil.copyfile(src_file, dst_file)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    # ============== 准备输入 ==============
    # 对于 LLM，使用一个简单的 prompt 进行校准和导出
    prompt = args.prompt
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        archive_file = str(resume_from)
        assert Path(archive_file).exists(), f"resume_from {archive_file} not exists"
        resume_strict = args.lora_mode != "keep_lora"
        _load_resume_state_dict_with_quant_weight(
            native_model=native_model,
            archive_file=archive_file,
            logger=logger,
            strict=resume_strict,
        )
        if not resume_strict:
            logger.info("Loaded resume checkpoint with strict=False for keep_lora mode")
        meta_info.resume_from = archive_file

    hf_logits = None
    hf_logits_for_quant = None
    if not only_export:
        native_model.eval().to(device).to(dtype)
        with torch.no_grad():
            hf_outputs = native_model(input_ids=input_ids, use_cache=False)
            hf_logits = hf_outputs.logits[:, -1:, :].detach()
            hf_logits_for_quant = hf_logits
            if args.lora_mode == "keep_lora":
                hf_logits_for_quant = _forward_with_temp_merged_lora(
                    native_model=native_model,
                    input_ids=input_ids,
                    lora_state_dict=lora_state_dict,
                    lora_config=lora_config,
                )

        next_token_id_base, next_token_text_base = decode_next_token(tokenizer, hf_logits)
        logger.info(f"hf_base next token: {next_token_id_base} {next_token_text_base}")
        if args.lora_mode == "keep_lora" and hf_logits_for_quant is not None:
            next_token_id_lora, next_token_text_lora = decode_next_token(tokenizer, hf_logits_for_quant)
            logger.info(f"hf_with_temp_merged_lora next token: {next_token_id_lora} {next_token_text_lora}")
        meta_info.hf_reference_mode = "hf_base"
        meta_info.quant_reference_mode = (
            "hf_with_temp_merged_lora" if args.lora_mode == "keep_lora" else "hf_base"
        )
        native_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============== 初始化 wrap model ==============
    xh_model.init_wrap_model(native_model)
    native_model = None

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    xh_model.to(device)
    xh_model.to(dtype)
    assert xh_model.use_cache, "xh_model must use cache"

    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": 0,
    }

    if (not only_export) and hf_logits is not None:
        with torch.no_grad():
            wrapped_logits = xh_model.test_step(data_batch).logits.detach()
        log_logits_compare(logger, "valid_wrap_vs_hf", hf_logits, wrapped_logits, meta_info)

    # ============== Frontend Graph ==============
    xh_model.interactive_mode = True
    logger.info("************* Convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)
    if args.lora_mode == "keep_lora":
        _attach_keep_lora_runtime(xh_model, data_batch, lora_alpha=lora_config["lora_alpha"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============== Quant Graph ==============
    logger.info("************* Convert to quant graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    # ============== PTQ 量化 ==============
    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    calib_data = flatten_inputs(calib_data)

    with time_profiler() as t:
        ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])
        logger.info(f"PTQ Quantize time: {t():.04f} s")
    logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    if not only_export:
        with torch.no_grad():
            with time_profiler() as t:
                outs = xh_model.test_step(data_batch)
            logger.info(f"QUANTED_ALIGNED: {t():.04f}")
            quanted_aligned_logits = outs.logits.detach()

        prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, quanted_aligned_logits)
        logger.info(f"Prefill Quanted next token: {prefill_next_token_id} {prefill_next_token_text}")
        if hf_logits_for_quant is not None:
            log_logits_compare(logger, "valid_quant_vs_hf", hf_logits_for_quant, quanted_aligned_logits, meta_info)
    else:
        prefill_next_token_id = None

    # ============== 导出 Prefill 模型 ==============
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    logger.info("*************** Start exporting prefill model ***************")

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model, tokenizer, data_batch, str(prefill_onnx_dir),
        f"{cfg_name}_prefill", device, dtype, logger, not only_export,
    )
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"Prefill ONNX saved to {prefill_onnx_file}")

    # ============== 导出 Decode 模型 ==============
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)
    data_batch["input_ids"] = data_batch["input_ids"].to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.set_input_sequence_length(1)

    past_seq_len = input_ids.shape[-1]
    if prefill_next_token_id is None:
        prefill_next_token_id = input_ids[:, :1]
    input_ids_decode = prefill_next_token_id

    data_batch = {
        "input_ids": input_ids_decode.to(device),
        "past_seq_length": past_seq_len,
    }

    if not only_export:
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            decode_logits = outs.logits.detach()
        decode_token_id, decode_token_text = decode_next_token(tokenizer, decode_logits)
        logger.info(f"Decode Quanted next token: {decode_token_id} {decode_token_text}")

    logger.info("*************** Start exporting decode model ***************")
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")

    decode_onnx_file = xhmodel_export_onnx(
        xh_model, tokenizer, data_batch, str(decode_onnx_dir),
        f"{cfg_name}_decode", device, dtype, logger, not only_export,
    )
    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"Decode ONNX saved to {decode_onnx_file}")

    rotated_adapter_path = args.rotated_adapter_path
    if rotated_adapter_path is None and args.resume_from is not None:
        inferred_rotated = Path(args.resume_from).parent / "audio_projector_rotated.safetensors"
        if inferred_rotated.exists():
            rotated_adapter_path = str(inferred_rotated)
            logger.info(f"Auto-detected rotated audio projector: {rotated_adapter_path}")

    # valid_asr 会通过 fireredasr_hf_forward 读取 export_meta_info.json，这里先落盘一次。
    export_meta_path = Path(cfg.work_dir) / "export_meta_info.json"
    with open(export_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved interim export meta to {export_meta_path}")

    if args.valid_asr:
        asr_summary = validate_asr_hmonnx_llm(
            fireredasr_model_dir=args.fireredasr_model_dir,
            hmonnx_work_dir=cfg.work_dir,
            wav_dir=args.wav_dir,
            ref_text_file=args.ref_text,
            use_gpu=args.use_gpu,
            logger=logger,
            rotated_adapter_path=rotated_adapter_path,
        )
        meta_info["valid_asr"] = asr_summary
    else:
        meta_info["valid_asr"] = {"skipped": True, "reason": "valid_asr_flag_disabled"}

    if getattr(args, "golden", False):
        _generate_golden(
            cfg=cfg,
            input_ids=input_ids,
            tokenizer=tokenizer,
            prefill_onnx_file=str(prefill_onnx_file),
            decode_onnx_file=str(decode_onnx_file),
            logger=logger,
        )
        if not getattr(args, "golden_skip_pack", False):
            _pack_golden(str(prefill_onnx_file), logger)
            _pack_golden(str(decode_onnx_file), logger)
        else:
            logger.info("Skip golden tar packing by --golden_skip_pack")

    # ============== 保存 meta 信息 ==============
    with open(export_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4, ensure_ascii=False)
    logger.info("*************** Export completed! ***************")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FireRedASR LLM (Qwen2 + LoRA) 导出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fireredasr/fireredasr_llm_xh2a_4k.py",
        help="Qwen2 模型配置文件路径",
    )
    parser.add_argument(
        "--fireredasr_model_dir",
        type=str,
        default="/data01/datasets/FireRedASR-LLM-L",
        help="FireRedASR 模型目录（包含 model.pth.tar）",
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default=None,
        help="Qwen2 基座模型目录（可选，默认自动从配置和 FireRedASR 目录推断）",
    )
    parser.add_argument(
        "--lora_mode",
        type=str,
        choices=["merge_lora", "keep_lora"],
        default="merge_lora",
        help="LoRA 处理方式: merge_lora=融合到基座, keep_lora=保留分支",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="导出后进行推理验证")
    parser.add_argument("--prompt", type=str, default="请转写音频为文字")
    parser.add_argument("--resume_from", type=str, default=None, help="加载quarot/gptq量化权重")
    parser.add_argument("--valid_asr", action="store_true", help="导出后使用真实语音做ASR对比验证")
    parser.add_argument("--wav_dir", type=str, default=None, help="真实语音目录，默认自动检测")
    parser.add_argument("--ref_text", type=str, default=None, help="参考文本文件，可选")
    parser.add_argument("--use_gpu", action="store_true", help="ASR验证阶段使用GPU")
    parser.add_argument(
        "--rotated_adapter_path",
        type=str,
        default=None,
        help="可选：audio_projector_rotated.safetensors 路径（quarot 后建议传入）",
    )
    parser.add_argument("--golden", action="store_true", help="导出后生成 HMONNX golden data")
    parser.add_argument("--golden_only", action="store_true", help="仅生成 golden（不重新导出 ONNX）")
    parser.add_argument("--golden_skip_pack", action="store_true", help="仅生成 golden 目录，不打包 tar.gz")
    return parser


def _normalize_name_tag(value: str) -> str:
    """Convert arbitrary string to stable lowercase token for cfg/work_dir naming."""
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return tag or "unknown"


def _build_mode_suffix_and_cfg_name(args, cfg) -> Tuple[str, str]:
    cfg_stem = _normalize_name_tag(Path(args.config).stem)

    mode_parts: List[str] = ["merge_lora" if args.lora_mode == "merge_lora" else "keep_lora"]

    if args.resume_from:
        mode_parts.append("resume")
        resume_hint = f"{Path(args.resume_from).parent.name}_{Path(args.resume_from).name}".lower()
        if "gptq" in resume_hint:
            mode_parts.append("gptq")
        if "quarot" in resume_hint:
            mode_parts.append("quarot")
        if "4bit" in resume_hint:
            mode_parts.append("4bit")
        elif "8bit" in resume_hint:
            mode_parts.append("8bit")
    else:
        if bool(cfg.get("gptq", False)):
            mode_parts.append("gptq")
        if bool(cfg.get("quarot", False)):
            mode_parts.append("quarot")
        if len(mode_parts) == 1:
            mode_parts.append("w8a8")

    mode_suffix = "_".join(mode_parts)
    cfg_name = f"{cfg_stem}_{mode_suffix}"
    return mode_suffix, cfg_name


def main(args):
    begin_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cfg = Config.fromfile(args.config)
    mode_suffix, cfg_name = _build_mode_suffix_and_cfg_name(args, cfg)
    cfg.cfg_name = cfg_name
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)

    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if "quarot" not in cfg:
        cfg.quarot = False
    if "gptq" not in cfg:
        cfg.gptq = False

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()
    logger.info(f"Naming: mode_suffix={mode_suffix}, cfg_name={cfg_name}")
    resolved_hf_model_dir = _resolve_hf_model_dir(
        cfg_hf_model_dir=str(cfg.hf_model_dir),
        fireredasr_model_dir=args.fireredasr_model_dir,
        cli_hf_model_dir=args.hf_model_dir,
    )
    cfg.hf_model_dir = resolved_hf_model_dir
    cfg.model.hf_model = resolved_hf_model_dir
    logger.info(f"Resolved hf_model_dir: {resolved_hf_model_dir}")

    xhquant.utils.suppress_printing.disable_printing = True
    _export_impl(cfg, args)

    end_time = time.time()
    logger.info(f"Total export time: {end_time - begin_time:.2f}s")
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()
        logger.info(f"Peak GPU memory: {peak_mem / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
