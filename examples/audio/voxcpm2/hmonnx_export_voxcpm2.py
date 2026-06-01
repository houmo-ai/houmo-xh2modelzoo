"""
VoxCPM2 base_lm 和 residual_lm 的 HMONNX 导出脚本(prefill/decode)。

python hmonnx_export_voxcpm2.py --model ~/models/VoxCPM2 --prefill_length 216 --cache_length 1024 --gen_golden
"""

from __future__ import annotations

import argparse
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn

from xhquant.api import (
    Config,
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    PrecisionMode,
    QuantScheme,
    create_quant_config,
    get_root_logger,
    ptq_quantize,
)

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType

from xh_model_zoo.xh_llm.models.voxcpm2 import (
    XHVoxCPM2BaseLMModel,
    XHVoxCPM2ResidualLMModel,
)
from xh_model_zoo.xh_llm.models.voxcpm2.voxcpm2_llm_model_impl import register_wrap_cls  # noqa: F401

from voxcpm import VoxCPM2Model

try:
    from .utils import (
        copy_hf_config_files,
        validate_cache_length,
        validate_prefill_length,
        write_json_file,
    )
except ImportError:
    from utils import (
        copy_hf_config_files,
        validate_cache_length,
        validate_prefill_length,
        write_json_file,
    )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def xhmodel_export_onnx(xh_model, data_batch, onnx_output_dir: str, cfg_name: str, logger):
    """和 qwen3_asr 的 export 流程一致,抽出来复用。"""
    logger.info("Start exporting %s ...", cfg_name)
    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish converting to export graph.")

    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting onnx ***************")
    return xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]


def _flatten_prepare_inputs(inputs):
    flat = []
    for arg in inputs:
        if isinstance(arg, (list, tuple)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _first_output(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _calc_parity_metrics(ref: torch.Tensor, got: torch.Tensor):
    ref_f = ref.detach().float().reshape(-1).cpu()
    got_f = got.detach().float().reshape(-1).cpu()
    if ref_f.shape != got_f.shape:
        raise RuntimeError(f"Parity shape mismatch: ref={tuple(ref_f.shape)} got={tuple(got_f.shape)}")
    abs_diff = (got_f - ref_f).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    cosine = float(torch.nn.functional.cosine_similarity(got_f.unsqueeze(0), ref_f.unsqueeze(0), dim=-1).item())
    return dict(max_abs=max_abs, mean_abs=mean_abs, cosine=cosine)


# ---------------------------------------------------------------------------
# 构造 calibration 输入(复刻 VoxCPM2Model._inference 的前半段)
# ---------------------------------------------------------------------------

def _build_calibration_embeddings(
    voxcpm2: VoxCPM2Model,
    prefill_length: int,
    device: torch.device,
    dtype: torch.dtype,
    logger,
):
    """构造一组校准输入:模拟 zero-shot 模式下的 prompt。

    Returns:
        text_embeds_padded: [1, N=prefill_length, H]  —— base_lm 输入
        combined_embed:     [1, N, H]                 —— 带 audio mask 的混合输入
        feat_embed_padded:  [1, N, H]                 —— 用于构造 residual_lm 输入
        text_mask:          [1, N]                    —— 1=text, 0=audio
        audio_mask:         [1, N]                    —— 0=text, 1=audio
    """
    # --- 1. 构造 text token ---
    target_text = "The quick brown fox jumps over the lazy dog."
    text_token = torch.LongTensor(voxcpm2.text_tokenizer(target_text))
    text_token = torch.cat([
        text_token,
        torch.tensor([voxcpm2.audio_start_token], dtype=torch.int64),
    ])
    text_length = text_token.shape[0]
    if text_length > prefill_length:
        raise ValueError(
            f"Calibration text tokens {text_length} exceeds prefill_length {prefill_length}. "
            f"请减短 target_text 或增大 --prefill_length。"
        )

    # --- 2. zero-shot 模式:全部是 text,没有 audio ---
    audio_feat_placeholder = torch.zeros(
        (text_length, voxcpm2.patch_size, voxcpm2.audio_vae.latent_dim),
        dtype=torch.float32,
    )
    text_mask_raw = torch.ones(text_length, dtype=torch.int32)
    audio_mask_raw = torch.zeros(text_length, dtype=torch.int32)

    # 加 batch 维
    text_token = text_token.unsqueeze(0).to(device)
    audio_feat = audio_feat_placeholder.unsqueeze(0).to(device).to(dtype)
    text_mask = text_mask_raw.unsqueeze(0).to(device)
    audio_mask = audio_mask_raw.unsqueeze(0).to(device)

    # --- 3. 模拟 _inference 前半段计算 embeddings ---
    scale_emb = voxcpm2.config.lm_config.scale_emb if voxcpm2.config.lm_config.use_mup else 1.0
    # 对 VoxCPM2 config: use_mup=False, scale_emb=12 —— 实际走 else 分支,scale_emb=1.0
    # 参考 voxcpm2.py 的 _inference,这里直接复用其逻辑

    with torch.no_grad():
        feat_embed = voxcpm2.feat_encoder(audio_feat)
        feat_embed = voxcpm2.enc_to_lm_proj(feat_embed)
        text_embed = voxcpm2.base_lm.embed_tokens(text_token) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed

    cur_len = combined_embed.shape[1]

    # --- 4. pad 到 prefill_length ---
    if cur_len < prefill_length:
        pad = prefill_length - cur_len
        pad_zero_embed = torch.zeros(
            (1, pad, combined_embed.shape[-1]), dtype=combined_embed.dtype, device=device,
        )
        combined_embed = torch.cat([combined_embed, pad_zero_embed], dim=1)
        feat_embed_padded = torch.cat(
            [feat_embed, torch.zeros_like(pad_zero_embed)], dim=1,
        )
        text_mask = torch.cat([text_mask, torch.zeros((1, pad), dtype=text_mask.dtype, device=device)], dim=1)
        audio_mask = torch.cat([audio_mask, torch.zeros((1, pad), dtype=audio_mask.dtype, device=device)], dim=1)
    else:
        feat_embed_padded = feat_embed

    logger.info("Calibration combined_embed.shape: %s", tuple(combined_embed.shape))
    logger.info("Calibration text_length (valid): %d, prefill_length: %d", cur_len, prefill_length)

    return {
        "combined_embed": combined_embed.to(dtype=dtype),
        "feat_embed_padded": feat_embed_padded.to(dtype=dtype),
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "valid_length": cur_len,
    }


# ---------------------------------------------------------------------------
# 单个 LM(base_lm 或 residual_lm)的导出流程
# ---------------------------------------------------------------------------

def _export_single_lm(
    *,
    lm_name: str,                 # "BaseLM" 或 "ResidualLM"
    hf_module: nn.Module,         # MiniCPMModel 实例
    wrap_model_type: str,         # "XHVoxCPM2BaseLMModel" / "XHVoxCPM2ResidualLMModel"
    prefill_length: int,
    cache_length: int,
    quant_type: str,
    work_dir: Path,
    device: torch.device,
    exec_device: torch.device,
    dtype: torch.dtype,
    prefill_data_batch: dict,
    decode_data_batch: dict,
    gen_golden: bool,
    logger,
    enable_rope: bool = True,
    verify: bool = True,
    verify_max_abs_tol: float = 1.0,
    verify_mean_abs_tol: float = 0.1,
    verify_cosine_tol: float = 0.95,
    verify_fail_on_mismatch: bool = False,
) -> dict:
    """对单个 MiniCPMModel 做 wrap/PTQ/export,产出 prefill 和 decode 两张图。

    Returns:
        meta: 包含 prefill_onnx / decode_onnx / kv_cache_shape / input_names 等信息的 dict
    """
    logger.info("=" * 60)
    logger.info("Exporting %s", lm_name)
    logger.info("=" * 60)

    # --- 参考 PyTorch 输出(模块级对齐基准,必须在 wrap 前计算) ---
    prefill_data_batch = dict(prefill_data_batch)
    prefill_data_batch.setdefault("past_seq_length", [0])
    decode_data_batch = dict(decode_data_batch)
    decode_data_batch.setdefault("past_seq_length", [prefill_length])
    with torch.no_grad():
        prefill_ref, _ = hf_module(
            inputs_embeds=prefill_data_batch["input_embeds"],
            is_causal=True,
        )
        decode_ref_full, _ = hf_module(
            inputs_embeds=torch.cat(
                [prefill_data_batch["input_embeds"], decode_data_batch["input_embeds"]],
                dim=1,
            ),
            is_causal=True,
        )
    prefill_ref = prefill_ref.detach()
    decode_ref = decode_ref_full[:, -1:, :].detach()

    # --- 构造 cfg ---
    from copy import deepcopy

    wrap_cfg = ConfigDict(dict(
        use_cache=True,
        input_sequence_length=prefill_length,
        max_sequence_length=prefill_length + cache_length,
        kv_cache=dict(cache_axis=2),
        num_logits_to_keep=0,         # prefill 输出全部 hidden
        enable_rope=enable_rope,
        only_first_block=False,
        batch_size=1,
    ))

    export_cfg = ConfigDict(dict(
        input_names=[],
        output_names=["hidden"],
    ))

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    # --- 实例化 wrap model ---
    xh_model = MODELS.build(dict(
        type=wrap_model_type,
        hf_module=hf_module,
        wrap_cfg=wrap_cfg,
        quant_config=quant_config,
        export_cfg=export_cfg,
    ))

    # 设置 export_cfg 的固定输入名(除 KV cache 外的前三个)
    base_input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
    xh_model.export_cfg.input_names = list(base_input_names)
    # init_wrap_model 里的 prepare_kv_cache 会把 past_key_cache_{i}/past_value_cache_{i} 追加进来
    xh_model.init_wrap_model()

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(device)
    xh_model.to(dtype)

    # --- 构造 data_batch ---
    # --- test_step & convert 流程(和 qwen3_asr 一致) ---
    xh_model.set_input_sequence_length(prefill_length)
    with torch.no_grad():
        _ = xh_model.test_step(prefill_data_batch)

    xh_model.interactive_mode = True
    logger.info("[%s] convert to frontend graph", lm_name)
    xh_model.convert_to_fronted_graph(prefill_data_batch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("[%s] convert to quanted graph", lm_name)
    xh_model.convert_to_quant_graph("XH2a")

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    logger.info("[%s] PTQ quantize", lm_name)
    calib_data = _flatten_prepare_inputs(xh_model.prepare_inputs(prefill_data_batch))
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])
    logger.info("[%s] PTQ done", lm_name)

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model = xh_model.to("cpu")

    # --- 导出 prefill 图 ---
    prefill_dir = work_dir / f"{lm_name}_Prefill"
    prefill_dir.mkdir(exist_ok=True, parents=True)
    prefill_golden_dir = prefill_dir / "hmonnx" / "golden"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cfg_name_prefill = f"voxcpm2_{lm_name.lower()}_prefill_xh2a"
    prefill_onnx = xhmodel_export_onnx(
        xh_model, prefill_data_batch, str(prefill_dir), cfg_name_prefill, logger,
    )
    logger.info("[%s] prefill onnx: %s", lm_name, prefill_onnx)

    if gen_golden:
        if prefill_golden_dir.exists():
            shutil.rmtree(prefill_golden_dir)
        session = HMONNXGoldenInference(prefill_onnx)
        session.to(exec_device)
        session.save_golden = True
        session.golden_dir = str(prefill_golden_dir)
        session.step = 0
        session(*calib_data)
        logger.info("[%s] prefill golden saved: %s", lm_name, prefill_golden_dir)

    xh_model.release_exported_model()

    # --- 切换到 decode 图(input_sequence_length=1,num_logits_to_keep=1) ---
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 更新 cfg:decode 只吃 1 个 token,并且只输出最后一个 hidden
    xh_model.set_input_sequence_length(1)
    # num_logits_to_keep 在 _MiniCPMModel._setup 里注册了,通过 _update_cfg 触发
    # xhquant 内部会遍历所有 DynamicModule 调用 _update_cfg,这里更新 wrap_cfg 即可
    xh_model.wrap_cfg.num_logits_to_keep = 1
    xh_model.wrap_cfg.input_sequence_length = 1

    # 构造 decode data_batch
    # decode 图的 past_seq_length 传入应该是 "当前已填充到的位置"。
    # 对固定长度 prefill 图,这个值仍应使用真实有效长度(valid_length),而不是 pad 后总长度。
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model = xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    decode_dir = work_dir / f"{lm_name}_Decode"
    decode_dir.mkdir(exist_ok=True, parents=True)
    decode_golden_dir = decode_dir / "hmonnx" / "golden"

    cfg_name_decode = f"voxcpm2_{lm_name.lower()}_decode_xh2a"
    decode_onnx = xhmodel_export_onnx(
        xh_model, decode_data_batch, str(decode_dir), cfg_name_decode, logger,
    )
    logger.info("[%s] decode onnx: %s", lm_name, decode_onnx)

    if gen_golden:
        if decode_golden_dir.exists():
            shutil.rmtree(decode_golden_dir)
        decode_calib_data = _flatten_prepare_inputs(xh_model.prepare_inputs(decode_data_batch))
        session = HMONNXGoldenInference(decode_onnx)
        session.to(exec_device)
        session.save_golden = True
        session.golden_dir = str(decode_golden_dir)
        session.step = 0
        session(*decode_calib_data)
        logger.info("[%s] decode golden saved: %s", lm_name, decode_golden_dir)

    parity = {}
    if verify:
        # prefill parity
        prefill_session = HMONNXGoldenInference(prefill_onnx)
        prefill_session.to(exec_device)
        prefill_out = _first_output(prefill_session(*calib_data))
        m_prefill = _calc_parity_metrics(prefill_ref, prefill_out)
        parity["prefill"] = m_prefill
        logger.info(
            "[%s] Parity(Prefill): max_abs=%.6f mean_abs=%.6f cosine=%.6f",
            lm_name,
            m_prefill["max_abs"],
            m_prefill["mean_abs"],
            m_prefill["cosine"],
        )

        # decode parity
        decode_calib_data = _flatten_prepare_inputs(xh_model.prepare_inputs(decode_data_batch))
        decode_session = HMONNXGoldenInference(decode_onnx)
        decode_session.to(exec_device)
        decode_out = _first_output(decode_session(*decode_calib_data))
        m_decode = _calc_parity_metrics(decode_ref, decode_out)
        parity["decode"] = m_decode
        logger.info(
            "[%s] Parity(Decode): max_abs=%.6f mean_abs=%.6f cosine=%.6f",
            lm_name,
            m_decode["max_abs"],
            m_decode["mean_abs"],
            m_decode["cosine"],
        )

        fails = []
        for k, m in parity.items():
            if not (
                m["max_abs"] <= verify_max_abs_tol
                and m["mean_abs"] <= verify_mean_abs_tol
                and m["cosine"] >= verify_cosine_tol
            ):
                fails.append(
                    f"{k}: max_abs={m['max_abs']:.6f}(tol={verify_max_abs_tol}), "
                    f"mean_abs={m['mean_abs']:.6f}(tol={verify_mean_abs_tol}), "
                    f"cosine={m['cosine']:.6f}(tol>={verify_cosine_tol})"
                )
        if fails:
            msg = f"[{lm_name}] parity check failed: " + " | ".join(fails)
            if verify_fail_on_mismatch:
                raise AssertionError(msg)
            logger.warning(msg)
        else:
            logger.info("[%s] parity check passed.", lm_name)

    # --- 返回元信息 ---
    kv_cache_shape = [
        1,
        xh_model.num_key_value_heads,
        cache_length,
        xh_model.head_dim,
    ]

    return dict(
        lm_name=lm_name,
        prefill_onnx=str(Path(prefill_onnx).relative_to(work_dir)),
        decode_onnx=str(Path(decode_onnx).relative_to(work_dir)),
        prefill_golden=str(prefill_golden_dir.relative_to(work_dir)) if gen_golden else None,
        decode_golden=str(decode_golden_dir.relative_to(work_dir)) if gen_golden else None,
        kv_cache_shape=kv_cache_shape,
        num_hidden_layers=xh_model.num_hidden_layers,
        num_key_value_heads=xh_model.num_key_value_heads,
        num_attention_heads=xh_model.num_attention_heads,
        hidden_size=xh_model.hidden_size,
        head_dim=xh_model.head_dim,
        input_names=["inputs_embeds", "past_seq_length", "current_input_length"]
        + [f"past_key_cache_{i}" for i in range(xh_model.num_hidden_layers)]
        + [f"past_value_cache_{i}" for i in range(xh_model.num_hidden_layers)],
        output_names=["hidden"],
        parity=parity if verify else None,
    )


# ---------------------------------------------------------------------------
# 保存 host 侧需要的小模块(投影层、fsq、stop_head 等)
# ---------------------------------------------------------------------------

def _save_host_modules(voxcpm2: VoxCPM2Model, work_dir: Path, logger):
    """把不进 HMONNX 的小模块保存为 .pt,host pipeline 加载使用。"""
    host_dir = work_dir / "host_modules"
    host_dir.mkdir(exist_ok=True, parents=True)

    modules_to_save = {
        "token_embedding": voxcpm2.base_lm.embed_tokens,
        "enc_to_lm_proj": voxcpm2.enc_to_lm_proj,
        "lm_to_dit_proj": voxcpm2.lm_to_dit_proj,
        "res_to_dit_proj": voxcpm2.res_to_dit_proj,
        "fusion_concat_proj": voxcpm2.fusion_concat_proj,
        "fsq_layer": voxcpm2.fsq_layer,
        "stop_proj": voxcpm2.stop_proj,
        "stop_head": voxcpm2.stop_head,
    }
    saved = {}
    for name, module in modules_to_save.items():
        state_file = host_dir / f"{name}.pt"
        torch.save(module.state_dict(), str(state_file))
        saved[name] = str(state_file.relative_to(work_dir))
        logger.info("host module saved: %s -> %s", name, state_file)
    return saved


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    validate_prefill_length(args.prefill_length)
    validate_cache_length(args.cache_length)

    model_path = str(Path(args.model).expanduser().resolve())
    model_name = Path(model_path).name
    target_device = "XH2a"

    script_dir = Path(__file__).resolve().parent
    work_root = script_dir / "work_dirs"
    work_dir = work_root / f"{model_name}_{target_device}"
    work_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exec_device = device
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info("Loading VoxCPM2 from %s", model_path)
    voxcpm2 = VoxCPM2Model.from_local(model_path, optimize=False, training=False)
    voxcpm2.to(device=device, dtype=dtype)
    voxcpm2.eval()

    # --- 1. 构造 calibration 输入 ---
    logger.info("Building calibration embeddings (prefill_length=%d)", args.prefill_length)
    calib = _build_calibration_embeddings(voxcpm2, args.prefill_length, device, dtype, logger)

    # --- 2. 准备 base_lm 输入 ---
    # base_lm prefill 输入 = combined_embed(text 和 audio 已经混合过)
    base_prefill_batch = {
        "input_embeds": calib["combined_embed"],
        "past_seq_length": [0],
    }
    # base_lm decode 输入 = 单 token embed(取最后一个有效位置)
    last_idx = calib["valid_length"] - 1
    base_decode_batch = {
        "input_embeds": calib["combined_embed"][:, last_idx:last_idx + 1, :],
        "past_seq_length": [calib["valid_length"]],
    }

    # --- 3. 构造 residual_lm 的 calibration 输入(必须在 base_lm wrap 前完成) ---
    # 先用原生 base_lm 跑一遍,拿到 enc_outputs,再走 fusion_concat_proj
    logger.info("Building residual_lm calibration inputs")
    with torch.no_grad():
        enc_outputs, _ = voxcpm2.base_lm(
            inputs_embeds=calib["combined_embed"], is_causal=True,
        )
        enc_outputs = enc_outputs.to(dtype)
        enc_outputs = voxcpm2.fsq_layer(enc_outputs) * calib["audio_mask"].unsqueeze(-1) \
            + enc_outputs * calib["text_mask"].unsqueeze(-1)
        residual_inputs = voxcpm2.fusion_concat_proj(
            torch.cat(
                (enc_outputs, calib["audio_mask"].unsqueeze(-1) * calib["feat_embed_padded"]),
                dim=-1,
            )
        )

    residual_prefill_batch = {
        "input_embeds": residual_inputs,
        "past_seq_length": [0],
    }
    residual_decode_batch = {
        "input_embeds": residual_inputs[:, last_idx:last_idx + 1, :],
        "past_seq_length": [calib["valid_length"]],
    }

    # --- 4. 导出 base_lm ---
    base_meta = _export_single_lm(
        lm_name="BaseLM",
        hf_module=voxcpm2.base_lm,
        wrap_model_type="XHVoxCPM2BaseLMModel",
        prefill_length=args.prefill_length,
        cache_length=args.cache_length,
        quant_type=args.quant_type,
        work_dir=work_dir,
        device=device,
        exec_device=exec_device,
        dtype=dtype,
        prefill_data_batch=base_prefill_batch,
        decode_data_batch=base_decode_batch,
        gen_golden=args.gen_golden,
        logger=logger,
        enable_rope=True,
        verify=not args.skip_verify,
        verify_max_abs_tol=args.verify_max_abs_tol,
        verify_mean_abs_tol=args.verify_mean_abs_tol,
        verify_cosine_tol=args.verify_cosine_tol,
        verify_fail_on_mismatch=args.verify_fail_on_mismatch,
    )

    # --- 5. 导出 residual_lm ---
    residual_meta = _export_single_lm(
        lm_name="ResidualLM",
        hf_module=voxcpm2.residual_lm,
        wrap_model_type="XHVoxCPM2ResidualLMModel",
        prefill_length=args.prefill_length,
        cache_length=args.cache_length,
        quant_type=args.quant_type,
        work_dir=work_dir,
        device=device,
        exec_device=exec_device,
        dtype=dtype,
        prefill_data_batch=residual_prefill_batch,
        decode_data_batch=residual_decode_batch,
        gen_golden=args.gen_golden,
        logger=logger,
        enable_rope=False,  # residual_lm no_rope=True
        verify=not args.skip_verify,
        verify_max_abs_tol=args.verify_max_abs_tol,
        verify_mean_abs_tol=args.verify_mean_abs_tol,
        verify_cosine_tol=args.verify_cosine_tol,
        verify_fail_on_mismatch=args.verify_fail_on_mismatch,
    )

    # --- 6. 保存 host 侧小模块 ---
    host_modules = _save_host_modules(voxcpm2, work_dir, logger)

    # --- 7. 复制 HF 配置文件 ---
    hf_config_dir = work_dir / "ConfigFiles"
    copied = copy_hf_config_files(
        Path(model_path), hf_config_dir, logger=logger,
        filenames=["config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json"],
    )

    # --- 8. 写 meta_info ---
    meta = dict(
        create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        model_name=model_name,
        target_device=target_device,
        hf_model=model_path,
        hf_config=str(hf_config_dir.relative_to(work_dir)),
        input_dtype=str(dtype).replace("torch.", ""),
        prefill_length=args.prefill_length,
        cache_length=args.cache_length,
        quant_type=args.quant_type,
        base_lm=base_meta,
        residual_lm=residual_meta,
        host_modules=host_modules,
        copied_hf_config_files=[p.name for p in copied],
    )
    meta_file = work_dir / "lm_export_meta_info.json"
    write_json_file(meta_file, meta)
    logger.info("Export meta written: %s", meta_file)
    logger.info("=" * 60)
    logger.info("All LM exports done. work_dir=%s", work_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        help="VoxCPM2 本地模型目录(含 config.json / model.safetensors 等)",
    )
    parser.add_argument(
        "--prefill_length", type=int, default=216,
        help="prefill 图的固定输入长度 N(default: 216)",
    )
    parser.add_argument(
        "--cache_length", type=int, default=1024,
        help="base_lm / residual_lm 的 KV cache 长度(default: 1024)",
    )
    parser.add_argument(
        "--quant_type", type=str, default="w8a8_sefp",
        help="量化类型(default: w8a8_sefp)",
    )
    parser.add_argument(
        "--gen_golden", action="store_true",
        help="生成 golden 数据供端侧对拍",
    )
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--verify_max_abs_tol", type=float, default=1.0)
    parser.add_argument("--verify_mean_abs_tol", type=float, default=0.1)
    parser.add_argument("--verify_cosine_tol", type=float, default=0.95)
    parser.add_argument("--verify_fail_on_mismatch", action="store_true")
    args = parser.parse_args()
    main(args)
