"""VoxCPM2 AudioVAE Encoder HMONNX 导出脚本。

用法:
    python export_audiovae_encoder.py --model /data01/home/binghu.ji/models/VoxCPM2
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import onnx
import onnxsim
import torch
import torch.nn as nn
from onnx import TensorProto

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
)
from xhquant.patch.core import RewriterContext
from voxcpm import VoxCPM2Model
from utils import (
    compute_audiovae_encoder_input_length,
    load_and_pad_audio,
    remove_weight_norm_recursively,
    write_json_file,
)


GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)

_ONNX_DTYPE_TO_NAME = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.BFLOAT16: "bfloat16",
}

def _first_output(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _read_onnx_input_dtype(onnx_path: str, input_name: str) -> str | None:
    """读取 HMONNX/ONNX 模型指定 input 的真实 dtype 名字。"""
    try:
        model = onnx.load(onnx_path)
    except Exception:
        return None
    for graph_input in model.graph.input:
        if graph_input.name != input_name:
            continue
        elem_type = graph_input.type.tensor_type.elem_type
        return _ONNX_DTYPE_TO_NAME.get(elem_type)
    return None


class AudioVAEEncoderExportWrapper(nn.Module):
    """只保留 encoder 的 mu 分支,绕开 logvar 卷积。

    原生 CausalEncoder.forward 返回 {"hidden_state", "mu", "logvar"}。
    直接 `return self.encoder(audio)["mu"]` 会让 torch.onnx 把 fc_logvar
    也 trace 进图——虽然输出上不用,但 `fc_logvar` 的权重(一个 Conv1d)
    会占据最终 HMONNX 产物的体积,也增大量化误差面。

    这里手动拿到 block 和 fc_mu 两部分,只保留推理路径上真正用到的算子。
    """

    def __init__(self, vae_encoder: nn.Module):
        super().__init__()
        self.block = vae_encoder.block    # 下采样主体(nn.Sequential)
        self.fc_mu = vae_encoder.fc_mu     # 只保留 mu 分支的最后 Conv1d

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [1, 1, L_samp],L_samp 必须是 chunk_size 的倍数
        Returns:
            mu: [1, latent_dim, T_latent]
        """
        h = self.block(audio)
        return self.fc_mu(h)


def _build_calib_audio(
    *,
    audio_path: str | None,
    target_samples: int,
    sample_rate: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if audio_path:
        p = Path(audio_path).expanduser().resolve()
        audio_np = load_and_pad_audio(p, target_samples, sample_rate=sample_rate)
        audio = torch.from_numpy(audio_np).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    else:
        torch.manual_seed(0)
        audio = torch.randn((1, 1, target_samples), device=device, dtype=dtype) * 0.05
    return audio


def _log_parity(logger, tag: str, a: torch.Tensor, b: torch.Tensor):
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    if a.shape != b.shape:
        logger.warning("%s: shape mismatch a=%s b=%s", tag, tuple(a.shape), tuple(b.shape))
        return
    diff = (a - b).abs()
    cosine = float(
        torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    )
    logger.info(
        "Parity(%s): max_abs=%.6f mean_abs=%.6f cosine=%.6f",
        tag,
        float(diff.max().item()),
        float(diff.mean().item()),
        cosine,
    )


def _verify_hmonnx_vs_pytorch(
    *,
    hmonnx_file: Path,
    device: torch.device,
    audio_cal: torch.Tensor,
    voxcpm2_audio_vae: nn.Module,
    encode_sample_rate: int,
    wrapper: nn.Module,
    graph_input_dtype_name: str | None,
    logger,
    max_abs_tol: float,
    mean_abs_tol: float,
    cosine_tol: float,
):
    """对比 HMONNX 和 PyTorch 的数值差异。

    同时做两组对比:
        (1) HMONNX  vs  手写 wrapper(block + fc_mu)    —— 验证导出图正确
        (2) HMONNX  vs  原生 audio_vae.encode()         —— 验证接口等价

    因为当前 L_samp 是 chunk_size 倍数,preprocess 不会再 pad,
    两组结果应该一致。分开打印便于定位问题。
    """
    # 根据 HMONNX 真实 input dtype 决定送入的张量类型
    target_dtype = torch.float32
    if graph_input_dtype_name == "float16":
        target_dtype = torch.float16
    elif graph_input_dtype_name == "bfloat16":
        target_dtype = torch.bfloat16

    audio_for_hmonnx = audio_cal.to(dtype=target_dtype)

    session = HMONNXInference(str(hmonnx_file))
    session.to(str(device))
    hmonnx_out = _first_output(session.run({"audio": audio_for_hmonnx}))

    # 参考 1:手写 wrapper(和 HMONNX 图结构一对一)
    with torch.no_grad():
        ref_wrapper = wrapper(audio_cal.to(dtype=next(wrapper.parameters()).dtype))
    _log_parity(logger, "AudioVAE-Encoder vs wrapper", hmonnx_out, ref_wrapper)

    # 参考 2:原生 AudioVAE.encode(会过 preprocess,此处 L_samp 对齐因此行为相同)
    with torch.no_grad():
        ref_native = voxcpm2_audio_vae.encode(
            audio_cal.squeeze(1).to(device=device, dtype=torch.float32),
            sample_rate=encode_sample_rate,
        )
    _log_parity(logger, "AudioVAE-Encoder vs audio_vae.encode", hmonnx_out, ref_native)

    # 用参考 1 做判定
    ref = ref_wrapper.detach().float().reshape(-1).cpu()
    got = hmonnx_out.detach().float().reshape(-1).cpu()
    abs_diff = (got - ref).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            got.unsqueeze(0), ref.unsqueeze(0), dim=-1
        ).item()
    )
    passed = bool(max_abs <= max_abs_tol and mean_abs <= mean_abs_tol and cosine >= cosine_tol)
    return {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "graph_input_dtype": graph_input_dtype_name,
    }


def main(args):
    model_path = str(Path(args.model).expanduser().resolve())
    model_name = Path(model_path).name
    target_device = "XH2a"

    num_patches = int(args.num_patches)

    script_dir = Path(__file__).resolve().parent
    work_root = script_dir / "work_dirs"
    # 和 decoder 端(AudioVAE_Decoder_np{N}/)对齐,支持并存多份不同 num_patches 的导出
    work_dir = work_root / f"{model_name}_{target_device}" / f"AudioVAE_Encoder_np{num_patches}"
    work_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    logger.info("从 %s 中加载 VoxCPM2 模型", model_path)
    voxcpm2 = VoxCPM2Model.from_local(model_path, optimize=False, training=False)
    voxcpm2.audio_vae.to(device=device, dtype=dtype)
    voxcpm2.audio_vae.eval()

    # weight_norm 是可学习的 parametrization,推理时等价于一个固定权重——导出
    # 前把它展开,避免 ONNX trace 把 weight_g / weight_v 这对参数引入图。
    logger.info("从图中删除 weight_norm。")
    remove_weight_norm_recursively(voxcpm2.audio_vae.encoder)

    wrapper = AudioVAEEncoderExportWrapper(voxcpm2.audio_vae.encoder)
    wrapper = wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    patch_size = int(voxcpm2.patch_size) # 每个 patch 的采样点数
    latent_dim = int(voxcpm2.audio_vae.latent_dim) # 每个 latent 时间步的通道维度
    chunk_size = int(voxcpm2.audio_vae.chunk_size) # 原始音频多少采样点，压成 1 个 latent 时间步
    encode_sample_rate = int(voxcpm2.audio_vae.sample_rate)
    L_samp = compute_audiovae_encoder_input_length(num_patches, patch_size, chunk_size) # 计算 AudioVAE encoder 定长输入长度
    T_latent = num_patches * patch_size

    logger.info(
        "固定维度: num_patches=%d patch_size=%d chunk_size=%d L_samp=%d T_latent=%d",
        num_patches,
        patch_size,
        chunk_size,
        L_samp,
        T_latent,
    )

    audio_cal = _build_calib_audio(
        audio_path=args.audio,
        target_samples=L_samp,
        sample_rate=encode_sample_rate,
        device=device,
        dtype=dtype,
    )
    logger.info("校验音频维度：%s", tuple(audio_cal.shape))

    with torch.no_grad():
        mu_ref = wrapper(audio_cal)
    logger.info("Encoder 前向正确, mu 维度为：%s", tuple(mu_ref.shape))
    assert mu_ref.shape == (1, latent_dim, T_latent), (
        f"Expected mu shape (1, {latent_dim}, {T_latent}), got {tuple(mu_ref.shape)}"
    )

    # --- 导出 ONNX ---
    onnx_file = work_dir / "voxcpm2_audiovae_encoder.onnx"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx = str(Path(tmp_dir) / onnx_file.name)
        with RewriterContext(None, backend="onnxruntime"):
            torch.onnx.export(
                wrapper,
                (audio_cal,),
                tmp_onnx,
                input_names=["audio"],
                output_names=["mu"],
                opset_version=17,
            )
            logger.info("ONNX 格式模型导出成功。")
            onnx_model = onnx.load(tmp_onnx)
            size = onnx_model.ByteSize()
            logger.info("ONNX size: %.3f GB", size / GB)
            if size <= _LARGE_MODEL_SIZE_THRESHOLD:
                simplified, ok = onnxsim.simplify(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )
            else:
                from xhquant.utils.onnxsim_large_model import simplify_large_onnx
                simplified, ok = simplify_large_onnx(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )
            if ok:
                onnx_model = simplified

    onnx.save(
        onnx_model,
        onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{onnx_file.stem}_external_data",
    )
    logger.info("ONNX 格式模型导出到: %s", onnx_file)

    # --- 转 HMONNX ---
    hmonnx_file = (
        work_dir
        / "hmonnx"
        / f"voxcpm2_audiovae_encoder_np{num_patches}_xh2a_{args.quant_type}.onnx"
    )
    hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    logger.info("Converting to HMONNX")
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [audio_cal],
        DeviceType.XH2a,
        hmonnx_file,
        quant_config=quant_config,
        input_names=["audio"],
        output_names=["mu"],
    )
    logger.info("HMONNX 格式保存在: %s", hmonnx_file)

    # 读取 HMONNX 真实 input dtype,避免 session 推理时 dtype 错配
    graph_input_dtype = _read_onnx_input_dtype(str(hmonnx_file), "audio")
    logger.info("HMONNX 图中音频类型 %s", graph_input_dtype)

    # --- parity 检查 ---
    if not args.skip_verify:
        parity = _verify_hmonnx_vs_pytorch(
            hmonnx_file=hmonnx_file,
            device=device,
            audio_cal=audio_cal,
            voxcpm2_audio_vae=voxcpm2.audio_vae,
            encode_sample_rate=encode_sample_rate,
            wrapper=wrapper,
            graph_input_dtype_name=graph_input_dtype,
            logger=logger,
            max_abs_tol=args.verify_max_abs_tol,
            mean_abs_tol=args.verify_mean_abs_tol,
            cosine_tol=args.verify_cosine_tol,
        )
        if parity["passed"]:
            logger.info("AudioVAE Encoder parity check passed.")
        else:
            msg = (
                "AudioVAE Encoder parity check failed: "
                f"max_abs={parity['max_abs']:.6f}(tol={args.verify_max_abs_tol}), "
                f"mean_abs={parity['mean_abs']:.6f}(tol={args.verify_mean_abs_tol}), "
                f"cosine={parity['cosine']:.6f}(tol>={args.verify_cosine_tol})"
            )
            if args.verify_fail_on_mismatch:
                raise AssertionError(msg)
            logger.warning(msg)

    # --- golden ---
    golden_dir = work_dir / "hmonnx" / "golden"
    if args.gen_golden:
        if golden_dir.exists():
            shutil.rmtree(golden_dir)
        session = HMONNXGoldenInference(hmonnx_file)
        session.to(device)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session.step = 0
        # golden 用 HMONNX 真实接受的 dtype 送入,否则 session 内部 assert 会报错
        audio_for_golden = audio_cal
        if graph_input_dtype == "float16":
            audio_for_golden = audio_cal.to(dtype=torch.float16)
        elif graph_input_dtype == "bfloat16":
            audio_for_golden = audio_cal.to(dtype=torch.bfloat16)
        session(audio_for_golden)
        logger.info("golden saved: %s", golden_dir)

    # --- meta ---
    meta = dict(
        create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        model_name=model_name,
        target_device=target_device,
        module="AudioVAE-Encoder",
        onnx_file=str(onnx_file.relative_to(work_dir.parent)),
        hmonnx_file=str(hmonnx_file.relative_to(work_dir.parent)),
        golden_dir=str(golden_dir.relative_to(work_dir.parent)) if args.gen_golden else None,
        quant_type=args.quant_type,
        input_dtype="float32",  # wrapper / calibration 侧 dtype
        graph_input_dtype=graph_input_dtype,  # HMONNX 图真实接受的 dtype
        input_names=["audio"],
        output_names=["mu"],
        input_shapes=dict(audio=[1, 1, L_samp]),
        output_shapes=dict(mu=[1, latent_dim, T_latent]),
        num_patches=num_patches,
        patch_size=patch_size,
        chunk_size=chunk_size,
        latent_dim=latent_dim,
        sample_rate=encode_sample_rate,
        encoder_rates=[int(x) for x in voxcpm2.audio_vae.encoder_rates],
        # host 侧契约:传入 audio 长度必须等于 L_samp
        # 超过截断、不够右 pad 零,和原生 AudioVAE.preprocess 行为等价
        host_call_contract=dict(
            audio_length_must_equal=L_samp,
            chunk_size=chunk_size,
            note="preprocess pad is baked away because L_samp is a multiple of chunk_size.",
        ),
    )
    meta_file = work_dir / "audiovae_encoder_meta_info.json"
    write_json_file(meta_file, meta)
    logger.info("Meta saved: %s", meta_file)
    logger.info("=" * 60)
    logger.info("AudioVAE Encoder 导出结束.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--num_patches",
        type=int,
        default=128,
        help="encoder 定长图可覆盖的 patch 数 (默认: 128,约 20.48s @16k)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="可选:用于 calibration 的音频路径;不传则用随机音频输入",
    )
    parser.add_argument("--quant_type", type=str, default="w16a16_sefp")
    parser.add_argument("--gen_golden", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--verify_max_abs_tol", type=float, default=0.5)
    parser.add_argument("--verify_mean_abs_tol", type=float, default=0.05)
    parser.add_argument("--verify_cosine_tol", type=float, default=0.99)
    parser.add_argument("--verify_fail_on_mismatch", action="store_true")
    args = parser.parse_args()
    main(args)