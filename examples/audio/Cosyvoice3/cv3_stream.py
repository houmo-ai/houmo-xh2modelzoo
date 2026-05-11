"""
CosyVoice 3 hmonnx 流式合成 demo (cv3_stream.py)
================================================

复用 cv3_eval.py 的 CosyVoiceFrontEnd / load_wav / make_pad_mask / parse_arguments，
把 llm_inference 与 token2wav 中的「重资源加载」一次性提到 init_resources()，
循环里只做 forward。每合成完一段立刻把波形写到 out_dir/chunks/chunk_NNN.wav，
结束后再合并写 final.wav。

------------------------------------------------------------
两种推理模式
------------------------------------------------------------
模式 A：句子级流式（默认，不加 --token_level_stream）
    text ──split_paragraph──> [句1, 句2, ...]
    每句一次 full token2wav，整句出完才 yield；
    切点必在标点，拼接处靠 fade-in/out 防跳变；
    适合短句 / demo / 离线批处理。

模式 B：句内 25-token 流式（加 --token_level_stream）
    句内按 token_hop_len (=25) 滑窗，
    每 chunk 把累计前缀 [0:offset+25+3] 喂给定长 token2wav，
    用 speech_offset 切出新增样点；
    首段比模式 A 早出现，但每 chunk 都跑满定长 onnx，
    总耗时 > 模式 A。当前 hift 未导出 finalize=False，
    拼接处仍靠 fade 压跳变。

------------------------------------------------------------
ONNX 后端两档（同时由 init_resources 自动读 input shape 适配）
------------------------------------------------------------
  hmonnx/      (默认)  pre_la=1024  decoder=2048  hift=1024
  hmonnx_512/         pre_la=512   decoder=1024  hift=512   ← 单次 forward 快 ~2-4x

切换方式：CLI 传 --pre_lookahead_layer / --flow_decoder / --hift。

------------------------------------------------------------
用法示例
------------------------------------------------------------

[1] 模式 A，hmonnx 默认 1024/2048（最稳）
    python cv3_stream.py \\
        --text "第一段。第二段稍微长一点点的内容。第三段又是另一句话。" \\
        --prompt_text "希望你以后能够做得比我还好呦。" \\
        --prompt_wav /data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/zero_shot_prompt.wav \\
        --out_dir stream_out --gpu 0

[2] 模式 A，hmonnx_512（推荐，~4x 提速）
    python cv3_stream.py \\
        --text "清晨的阳光透过窗帘洒进房间，空气里带着一丝淡淡的花香。" \\
        --prompt_text "希望你以后能够做得比我还好呦。" \\
        --prompt_wav /data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/zero_shot_prompt.wav \\
        --out_dir stream_out_512 --gpu 0 \\
        --pre_lookahead_layer hmonnx_512/pre_lookahead_layer/prefill/hmquant_xh2_pre_lookahead_layer_w8a16_512_20260509.onnx \\
        --flow_decoder        hmonnx_512/decoder/prefill/hmquant_xh2_decoder_w8a16_1024_20260509.onnx \\
        --hift                hmonnx_512/hift/prefill/hmquant_xh2_hift_w8a16_512_20260509.onnx

[3] 模式 B，hmonnx_512 + fade=5ms（句内 token 级流式 + 边界归零）
    python cv3_stream.py \\
        --text "清晨的阳光透过窗帘洒进房间，空气里带着一丝淡淡的花香。" \\
        --prompt_text "希望你以后能够做得比我还好呦。" \\
        --prompt_wav /data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/zero_shot_prompt.wav \\
        --out_dir stream_out_512_token --gpu 0 \\
        --pre_lookahead_layer hmonnx_512/pre_lookahead_layer/prefill/hmquant_xh2_pre_lookahead_layer_w8a16_512_20260509.onnx \\
        --flow_decoder        hmonnx_512/decoder/prefill/hmquant_xh2_decoder_w8a16_1024_20260509.onnx \\
        --hift                hmonnx_512/hift/prefill/hmquant_xh2_hift_w8a16_512_20260509.onnx \\
        --token_level_stream --token_max_n 80 --token_min_n 60 --fade_ms 5

[4] 切细句子让首段更快（默认 30/20，可激进到 20/15）
    python cv3_stream.py ... --token_max_n 20 --token_min_n 15

[5] 关 fade 看原始拼接（调试用）
    python cv3_stream.py ... --fade_ms 0

------------------------------------------------------------
环境要求
------------------------------------------------------------
  conda activate xhquant
  export PYTHONPATH=/data01/home/she.gao/xh2modelzoo

------------------------------------------------------------
输出目录结构
------------------------------------------------------------
  out_dir/
    chunks/
      chunk_000.wav, chunk_001.wav, ...   # 按生成顺序逐个落盘
    final.wav                              # 所有 chunk 拼接

每个 chunk 写盘时 stdout 同步打印路径 (flush=True)，
另一终端可用 `aplay chunk_xxx.wav` 实时播放。

------------------------------------------------------------
关键 CLI 参数速查
------------------------------------------------------------
  必填:  --text  --prompt_text  --prompt_wav
  常用:  --out_dir  --gpu  --token_level_stream  --fade_ms
  切分:  --token_max_n (30)  --token_min_n (20)  --merge_len (10)  --comma_split
  滑窗:  --token_hop_len (25)  --pre_lookahead_len (3)   # 仅模式 B
  onnx:  --pre_lookahead_layer  --flow_decoder  --hift   # 切 hmonnx 档位
"""

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Any, List

# 让 import cv3_eval 能在任意 cwd 下工作；cv3_eval 顶层会注入 matcha 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from hyperpyyaml import load_hyperpyyaml
from transformers import Qwen2ForCausalLM

from xhquant.api import HMONNXInference, Config
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.cosyvoice3 import (
    Qwen2_HFCompatible,
    XHQwen2HMONNXModel,
)

from cv3_eval import (
    CosyVoiceFrontEnd,
    load_wav,
    make_pad_mask,
    parse_arguments,
)
from frontend_utils import contains_chinese, split_paragraph, is_only_punctuation
from functools import partial

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


# ============================================================
# 资源容器
# ============================================================
@dataclass
class Resources:
    device: torch.device
    # LLM
    hf_model: Any
    sos_eos_emb: torch.Tensor
    task_id_emb: torch.Tensor
    # token2wav
    input_embedding: nn.Embedding
    flow_encoder: HMONNXInference
    spk_aff: HMONNXInference
    decoder: HMONNXInference
    hift: HMONNXInference
    # onnx 定长容量（运行时读 onnx 输入 shape，避免 1024/2048 硬编码）
    pre_la_cap: int = 1024     # pre_lookahead 输入 token 数上限
    decoder_cap: int = 2048    # decoder 输入 mel 帧数上限
    hift_cap: int = 1024       # hift 输入 mel 帧数上限


def init_resources(args, device: torch.device) -> Resources:
    """一次性加载 HF 模型 + 4 个 hmonnx session + input_embedding + sos/task emb。"""
    t0 = time.time()

    # --- LLM ---
    cfg = Config.fromfile(args.llm_config)
    qwen2_model: XHQwen2HMONNXModel = MODELS.build(cfg.model.llm)
    hf_native = Qwen2ForCausalLM.from_pretrained(args.qwen2_hf_model)
    hf_model = Qwen2_HFCompatible.to_hf_compatible(hf_native, qwen2_model)
    hf_model._llm_model._set_device(device)

    sos_eos_emb = torch.load(args.sos_eos_emb, map_location=device)
    task_id_emb = torch.load(args.task_id_emb, map_location=device)

    # --- token2wav: input embedding ---
    in_emb_w = torch.load(args.input_embedding, map_location="cpu", weights_only=True)
    input_embedding = nn.Embedding(in_emb_w.shape[0], in_emb_w.shape[1])
    input_embedding.load_state_dict({"weight": in_emb_w})
    input_embedding.to(device)

    # --- token2wav: 4 个 hmonnx session ---
    def _mk(path):
        sess = HMONNXInference(path)
        sess.save_golden = False
        sess.to(device)
        return sess

    flow_encoder = _mk(args.pre_lookahead_layer)
    spk_aff = _mk(args.spk_embed_affine_layer)
    decoder = _mk(args.flow_decoder)
    hift = _mk(args.hift)

    # 读 onnx 输入 shape，自动适配 hmonnx / hmonnx_512 等不同档位
    def _seq_len(onnx_path: str, axis: int) -> int:
        import onnx as _onnx
        m = _onnx.load(onnx_path, load_external_data=False)
        dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
        return int(dims[axis])

    pre_la_cap = _seq_len(args.pre_lookahead_layer, axis=1)  # [1, S, 80]
    decoder_cap = _seq_len(args.flow_decoder, axis=2)        # [2, 80, S]
    hift_cap = _seq_len(args.hift, axis=2)                   # [1, 80, S]
    logging.info(
        f"onnx caps: pre_la={pre_la_cap}, decoder={decoder_cap}, hift={hift_cap}"
    )

    logging.info(f"init_resources done in {time.time() - t0:.2f}s")
    return Resources(
        device=device,
        hf_model=hf_model,
        sos_eos_emb=sos_eos_emb,
        task_id_emb=task_id_emb,
        input_embedding=input_embedding,
        flow_encoder=flow_encoder,
        spk_aff=spk_aff,
        decoder=decoder,
        hift=hift,
        pre_la_cap=pre_la_cap,
        decoder_cap=decoder_cap,
        hift_cap=hift_cap,
    )


# ============================================================
# LLM 单句推理（搬自 cv3_eval.llm_inference，去掉模型构建）
# ============================================================
def llm_step(res: Resources, model_input: dict) -> List[int]:
    device = res.device
    sampling = 25  # 与 cv3_eval 保持一致（仅占位，未直接使用）
    max_token_text_ratio = 20
    min_token_text_ratio = 2
    cur_silent, max_silent = 0, 5
    silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    text = model_input["text"].to(device)
    text_len = model_input["text_len"]

    if "prompt_text" not in model_input:
        prompt_text = torch.zeros(1, 0, dtype=torch.int32, device=device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32, device=device)
    else:
        prompt_text = model_input["prompt_text"].to(device)
        prompt_text_len = model_input.get(
            "prompt_text_len",
            torch.tensor([prompt_text.shape[1]], dtype=torch.int32, device=device),
        )

    prompt_speech_token = model_input.get(
        "llm_prompt_speech_token",
        torch.zeros(1, 0, dtype=torch.int32, device=device),
    ).to(device)

    text = torch.concat([prompt_text, text], dim=1)
    text_len = text_len + prompt_text_len
    text_emb = res.hf_model._llm_model.token_embedding(text).to(device)
    prompt_speech_token_emb = res.hf_model._llm_model.speech_embedding(
        prompt_speech_token
    ).to(device)

    lm_input = torch.concat(
        [res.sos_eos_emb, text_emb, res.task_id_emb, prompt_speech_token_emb], dim=1
    ).to(torch.float16)

    min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
    max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

    out: List[int] = []
    for tok in res.hf_model.generate(
        min_len=min_len, max_len=max_len, inputs_embeds=lm_input
    ):
        if tok in silent_tokens:
            cur_silent += 1
            if cur_silent > max_silent:
                continue
        else:
            cur_silent = 0
        out.append(tok)
    return out


# ============================================================
# LLM 流式生成器：与 llm_step 同逻辑，但 yield 每个 token
# ============================================================
def llm_token_generator(res: Resources, model_input: dict):
    device = res.device
    max_token_text_ratio = 20
    min_token_text_ratio = 2
    cur_silent, max_silent = 0, 5
    silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

    text = model_input["text"].to(device)
    text_len = model_input["text_len"]
    if "prompt_text" not in model_input:
        prompt_text = torch.zeros(1, 0, dtype=torch.int32, device=device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32, device=device)
    else:
        prompt_text = model_input["prompt_text"].to(device)
        prompt_text_len = model_input.get(
            "prompt_text_len",
            torch.tensor([prompt_text.shape[1]], dtype=torch.int32, device=device),
        )
    prompt_speech_token = model_input.get(
        "llm_prompt_speech_token",
        torch.zeros(1, 0, dtype=torch.int32, device=device),
    ).to(device)

    text = torch.concat([prompt_text, text], dim=1)
    text_len = text_len + prompt_text_len
    text_emb = res.hf_model._llm_model.token_embedding(text).to(device)
    prompt_speech_token_emb = res.hf_model._llm_model.speech_embedding(
        prompt_speech_token
    ).to(device)
    lm_input = torch.concat(
        [res.sos_eos_emb, text_emb, res.task_id_emb, prompt_speech_token_emb], dim=1
    ).to(torch.float16)
    min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
    max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

    for tok in res.hf_model.generate(
        min_len=min_len, max_len=max_len, inputs_embeds=lm_input
    ):
        if tok in silent_tokens:
            cur_silent += 1
            if cur_silent > max_silent:
                continue
        else:
            cur_silent = 0
        yield tok


# ============================================================
# token -> wav（搬自 cv3_eval.token2wav，去掉 session 构建）
# ============================================================
def token2wav_step(
    res: Resources,
    token: torch.Tensor,
    prompt_token: torch.Tensor,
    prompt_feat: torch.Tensor,
    embedding: torch.Tensor,
    speed: float = 1.0,
) -> torch.Tensor:
    assert token.shape[0] == 1
    device = res.device
    inference_cfg_rate = 0.7
    token_mel_ratio = 2

    token = token.to(device)
    prompt_token = prompt_token.to(device)
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
    prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32)
    token = torch.concat([prompt_token, token], dim=1)
    token_len = prompt_token_len + token_len

    token = res.input_embedding(token)
    token = F.pad(token, (0, 0, 0, res.pre_la_cap - token.shape[1]), value=0).to(torch.float16)

    # encoder
    n = res.flow_encoder.get_input_names()
    h = res.flow_encoder.run({n[0]: token}).repeat_interleave(token_mel_ratio, dim=1)

    # speaker affine
    embedding = F.normalize(embedding, dim=1)
    n = res.spk_aff.get_input_names()
    embedding = res.spk_aff.run({n[0]: embedding})

    # CFM solver
    mel_len1 = prompt_feat.shape[1]
    mel_len2 = token_len * 2 - prompt_feat.shape[1]
    conds = torch.zeros([1, res.decoder_cap, 80], device=device).to(h.dtype)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)
    mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]), res.decoder_cap)).to(h)
    mask = mask.unsqueeze(1)
    mu = h.transpose(1, 2).contiguous()
    rand_noise = torch.randn([1, 80, 50 * 300])
    x = rand_noise[:, :, : mu.size(2)].to(mu.device).to(mu.dtype) * 1.0
    t_span = torch.linspace(0, 1, 10 + 1, device=mu.device, dtype=mu.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(dim=0)

    x_in = torch.zeros([2, 80, x.size(2)], device=device, dtype=x.dtype)
    mask_in = torch.zeros([2, 1, x.size(2)], device=device, dtype=x.dtype)
    mu_in = torch.zeros([2, 80, x.size(2)], device=device, dtype=x.dtype)
    t_in = torch.zeros([2], device=device, dtype=x.dtype)
    cond_in = torch.zeros([2, 80, x.size(2)], device=device, dtype=x.dtype)
    spks_in = torch.zeros([2, 80], device=device, dtype=x.dtype)

    nd = res.decoder.get_input_names()
    sol = []
    for step in range(1, len(t_span)):
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t.unsqueeze(0)
        spks_in[0] = embedding
        cond_in[0] = conds
        d = res.decoder.run(
            {
                nd[0]: x_in,
                nd[1]: mask_in,
                nd[2]: mu_in,
                nd[3]: t_in,
                nd[4]: spks_in,
                nd[5]: cond_in,
            }
        )
        d, cfg_d = torch.split(d, [x.size(0), x.size(0)], dim=0)
        d = (1.0 + inference_cfg_rate) * d - inference_cfg_rate * cfg_d
        x = x + dt * d
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    feat = sol[-1].float()
    feat = feat[:, :, mel_len1 : mel_len1 + mel_len2]

    # hift
    tts_mel = feat[:, :, 0:]
    needed = res.hift_cap - tts_mel.size(2)
    if needed > 0:
        tts_mel = F.pad(tts_mel, (0, needed), value=0)
    else:
        tts_mel = tts_mel[:, :, : res.hift_cap]
    if speed != 1.0:
        tts_mel = F.interpolate(
            tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
        )
    nh = res.hift.get_input_names()
    tts_speech = res.hift.run({nh[0]: tts_mel.to(torch.float16)})
    tts_speech = tts_speech[:, : 480 * mel_len2]
    return tts_speech


# ============================================================
# 单句合成：frontend -> llm -> token2wav
# ============================================================
def synthesize_one(
    res: Resources,
    frontend: CosyVoiceFrontEnd,
    sentence: str,
    prompt_text_full: str,
    prompt_wav_path: str,
    sample_rate: int,
) -> torch.Tensor:
    model_input = frontend.frontend_zero_shot(
        sentence, prompt_text_full, prompt_wav_path, sample_rate
    )
    tokens = llm_step(res, model_input)
    tokens_t = torch.tensor(tokens).unsqueeze(0).to(res.device)
    wav = token2wav_step(
        res,
        token=tokens_t,
        prompt_token=model_input["flow_prompt_speech_token"].to(res.device),
        prompt_feat=model_input["prompt_speech_feat"].to(res.device),
        embedding=model_input["flow_embedding"].to(res.device),
    )
    return wav.to(torch.float32).to("cpu")


# ============================================================
# 单句内部的 25-token 流式（在 hmonnx 定长 onnx 上模拟）
# 注意：每个 chunk 都把"累计前缀"喂给定长 token2wav，故总耗时一定 > 整句单次。
# 这是为了观察 token 级流式的首段延迟 / 听感，不是性能优化。
# ============================================================
def synthesize_one_token_stream(
    res: Resources,
    frontend: CosyVoiceFrontEnd,
    sentence: str,
    prompt_text_full: str,
    prompt_wav_path: str,
    sample_rate: int,
    token_hop_len: int = 25,
    pre_lookahead_len: int = 3,
):
    """生成器：逐个 yield (chunk_wav: torch.Tensor[1,N], is_first: bool, is_final: bool)。"""
    device = res.device
    model_input = frontend.frontend_zero_shot(
        sentence, prompt_text_full, prompt_wav_path, sample_rate
    )
    prompt_token = model_input["flow_prompt_speech_token"].to(device)
    prompt_feat = model_input["prompt_speech_feat"].to(device)
    embedding = model_input["flow_embedding"].to(device)

    prompt_token_len = prompt_token.shape[1]
    # prompt 对齐到 hop 的整数倍：首 chunk 多吃这段 pad
    if prompt_token_len % token_hop_len == 0:
        prompt_token_pad = 0
    else:
        prompt_token_pad = (
            (prompt_token_len + token_hop_len - 1) // token_hop_len
        ) * token_hop_len - prompt_token_len

    # onnx 上限：input_embedding 输入是 (prompt_token + speech_token)，pad 到 pre_la_cap
    # 留 pre_lookahead 余量，避免溢出
    onnx_token_cap = res.pre_la_cap - pre_lookahead_len - 4

    speech_tokens: List[int] = []
    token_offset = 0
    speech_offset = 0
    is_first_yield = True
    capped = False

    def _emit(token_offset_curr: int, finalize: bool):
        nonlocal speech_offset, is_first_yield
        cur_speech_tokens = torch.tensor(
            speech_tokens[: token_offset_curr], dtype=torch.long
        ).unsqueeze(0).to(device)
        wav_full = token2wav_step(
            res,
            token=cur_speech_tokens,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )  # [1, N_total]
        wav_full = wav_full.to(torch.float32).to("cpu")
        new_wav = wav_full[:, speech_offset:]
        speech_offset = wav_full.shape[1]
        is_f = is_first_yield
        is_first_yield = False
        return new_wav, is_f, finalize

    for tok in llm_token_generator(res, model_input):
        speech_tokens.append(int(tok))
        # onnx 容量保护
        if prompt_token_len + len(speech_tokens) >= onnx_token_cap:
            capped = True
            break
        # 触发 chunk
        this_hop = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
        if len(speech_tokens) - token_offset >= this_hop + pre_lookahead_len:
            new_offset = token_offset + this_hop
            new_wav, is_f, _ = _emit(new_offset + pre_lookahead_len, finalize=False)
            token_offset = new_offset
            yield new_wav, is_f, False

    # 收尾：把剩余 token 全跑一次 finalize
    if len(speech_tokens) > token_offset:
        new_wav, is_f, _ = _emit(len(speech_tokens), finalize=True)
        yield new_wav, is_f, True
    if capped:
        logging.warning(
            f"sentence hit onnx token cap ({onnx_token_cap}), truncated tail"
        )


# ============================================================
# 主流程：句子级流式
# ============================================================
def stream_tts(args):
    if not os.path.exists(args.config_yaml):
        raise FileNotFoundError(args.config_yaml)
    if not os.path.exists(args.prompt_wav):
        raise FileNotFoundError(args.prompt_wav)

    with open(args.config_yaml, "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(
                    args.cosyvoice_path, "CosyVoice-BlankEN"
                )
            },
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"using device {device}")

    frontend = CosyVoiceFrontEnd(
        args,
        configs["get_tokenizer"],
        configs["feat_extractor"],
        configs["allowed_special"],
    )
    res = init_resources(args, device)

    sample_rate = configs["sample_rate"]

    # ---- 文本预处理：prompt 不切，target 用可调阈值切 ----
    prompt_text_full = "You are a helpful assistant.<|endofprompt|>" + args.prompt_text
    prompt_text_full = frontend.text_normalize(
        prompt_text_full, split=False, text_frontend=True
    )

    # 先做 normalize 拿到规范化后字符串，再用 split_paragraph 自己控制阈值
    text_norm = frontend.text_normalize(args.text, split=False, text_frontend=True)
    if isinstance(text_norm, list):
        # text_normalize 在 split=False 时偶尔返回 list（generator path），扁平化
        text_norm = "".join(t for t in text_norm if isinstance(t, str))
    if not text_norm:
        logging.error("empty text after normalize, abort")
        return
    lang = "zh" if contains_chinese(text_norm) else "en"
    tokenize = partial(
        frontend.tokenizer.encode, allowed_special=frontend.allowed_special
    )
    sentences = list(
        split_paragraph(
            text_norm,
            tokenize,
            lang,
            token_max_n=args.token_max_n,
            token_min_n=args.token_min_n,
            merge_len=args.merge_len,
            comma_split=args.comma_split,
        )
    )
    sentences = [
        s for s in sentences if isinstance(s, str) and s.strip() and not is_only_punctuation(s)
    ]
    if not sentences:
        logging.error("no valid sentence after split_paragraph, abort")
        return
    logging.info(
        f"split into {len(sentences)} sentence(s) "
        f"(token_max_n={args.token_max_n}, token_min_n={args.token_min_n}, "
        f"merge_len={args.merge_len}, comma_split={args.comma_split})"
    )

    # ---- 输出目录 ----
    chunks_dir = os.path.join(args.out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # ---- 流式循环 ----
    all_wav: List[torch.Tensor] = []
    total_t0 = time.time()
    global_chunk_idx = 0  # token-level 模式下用于全局编号
    for idx, sent in enumerate(sentences):
        if len(sent) < 0.5 * len(prompt_text_full):
            logging.warning(
                f"sentence {idx} too short vs prompt, may degrade quality: {sent!r}"
            )
        t0 = time.time()
        if args.token_level_stream:
            # 句内 25-token 流式
            sent_chunk_idx = 0
            for new_wav, is_first, is_final in synthesize_one_token_stream(
                res,
                frontend,
                sent,
                prompt_text_full,
                args.prompt_wav,
                sample_rate,
                token_hop_len=args.token_hop_len,
                pre_lookahead_len=args.pre_lookahead_len,
            ):
                if args.fade_ms > 0:
                    fade_n = int(args.fade_ms / 1000.0 * sample_rate)
                    if fade_n > 0 and new_wav.shape[1] > 2 * fade_n:
                        ramp_in = torch.linspace(0, 1, fade_n).unsqueeze(0)
                        ramp_out = torch.linspace(1, 0, fade_n).unsqueeze(0)
                        new_wav[:, :fade_n] *= ramp_in
                        new_wav[:, -fade_n:] *= ramp_out
                chunk_path = os.path.join(
                    chunks_dir, f"chunk_{global_chunk_idx:03d}.wav"
                )
                torchaudio.save(chunk_path, new_wav, sample_rate)
                all_wav.append(new_wav)
                dt = time.time() - t0
                tag = "FIRST" if is_first else ("LAST " if is_final else "     ")
                print(
                    f"[sent {idx + 1}/{len(sentences)} chunk {sent_chunk_idx} {tag}] "
                    f"wall={dt:.2f}s len={new_wav.shape[1] / sample_rate:.2f}s "
                    f"-> {chunk_path}",
                    flush=True,
                )
                sent_chunk_idx += 1
                global_chunk_idx += 1
        else:
            wav = synthesize_one(
                res, frontend, sent, prompt_text_full, args.prompt_wav, sample_rate
            )
            # ---- fade-in / fade-out 防拼接爆音 ----
            if args.fade_ms > 0:
                fade_n = int(args.fade_ms / 1000.0 * sample_rate)
                if fade_n > 0 and wav.shape[1] > 2 * fade_n:
                    ramp_in = torch.linspace(0, 1, fade_n).unsqueeze(0)
                    ramp_out = torch.linspace(1, 0, fade_n).unsqueeze(0)
                    wav[:, :fade_n] *= ramp_in
                    wav[:, -fade_n:] *= ramp_out
            chunk_path = os.path.join(chunks_dir, f"chunk_{idx:03d}.wav")
            torchaudio.save(chunk_path, wav, sample_rate)
            all_wav.append(wav)
            dt = time.time() - t0
            print(
                f"[{idx + 1}/{len(sentences)}] {dt:.2f}s -> {chunk_path}  "
                f"text={sent!r}",
                flush=True,
            )

    # ---- 合并 final.wav ----
    if all_wav:
        final = torch.cat(all_wav, dim=1)
        final_path = os.path.join(args.out_dir, "final.wav")
        torchaudio.save(final_path, final, sample_rate)
        total = time.time() - total_t0
        print(
            f"[done] {len(all_wav)} chunks, total {total:.2f}s -> {final_path}",
            flush=True,
        )


def build_parser():
    parser = parse_arguments()
    parser.add_argument("--text", type=str, required=True, help="要合成的整段文本")
    parser.add_argument("--prompt_text", type=str, required=True, help="prompt 文本")
    parser.add_argument(
        "--prompt_wav",
        type=str,
        required=True,
        help="prompt 16k wav 路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="stream_out",
        help="输出目录，会在其下生成 chunks/chunk_NNN.wav 与 final.wav",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument(
        "--token_max_n",
        type=int,
        default=30,
        help="split_paragraph 的单段最大 token 数（默认 30，越小首段越快）",
    )
    parser.add_argument(
        "--token_min_n",
        type=int,
        default=20,
        help="split_paragraph 的单段最小 token 数",
    )
    parser.add_argument(
        "--merge_len",
        type=int,
        default=10,
        help="split_paragraph 短段合并阈值",
    )
    parser.add_argument(
        "--comma_split",
        action="store_true",
        help="是否把逗号也作为切分点（默认关；开启会让 chunk 更小但可能在逗号处出现轻微爆音）",
    )
    parser.add_argument(
        "--fade_ms",
        type=float,
        default=5.0,
        help="每个 chunk 头尾的 fade 时长（毫秒），用于防止拼接处爆音；0 关闭",
    )
    parser.add_argument(
        "--token_level_stream",
        action="store_true",
        help="开启句内 25-token 流式（实验性，每个 chunk 都跑满定长 onnx 故性能更差）",
    )
    parser.add_argument(
        "--token_hop_len",
        type=int,
        default=25,
        help="句内 token 流式的步长（仅在 --token_level_stream 时生效）",
    )
    parser.add_argument(
        "--pre_lookahead_len",
        type=int,
        default=3,
        help="每次 chunk 的 pre_lookahead 余量（仅在 --token_level_stream 时生效）",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    stream_tts(args)
