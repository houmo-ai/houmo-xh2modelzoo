import os
import sys
import json

import librosa
import argparse
import tempfile
import onnx
import onnxsim
import importlib.util

import torch
import torch.nn as nn

from pathlib import Path

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    ptq_quantize,
    to_frontend_graph,
    to_quant_graph,
    get_root_logger
)

from xhquant.api.ptq_export_hmonnx import (
    convert_quanted_model_to_hmonnx,
)

from xhquant.common.types import PrecisionMode
from xhquant.core.datatype_mapping import TORCH_DTYPE_TO_FAKE_DTYPE
from xhquant.frontend.convert import to_frontend_graph
from xhquant.patch.core import RewriterContext
from xhquant.utils.config import ConfigDict
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model

from xh_model_zoo.xh_llm.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration
)

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRProcessor
)

GB = int(2**30)

_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)


def main(args):
    target_device = "XH2a"
    model_dir = os.path.normpath(args.model)
    model_name = os.path.basename(model_dir)
    
    model = Qwen3ASRForConditionalGeneration.from_pretrained(model_dir)
    cfg = Qwen3ASRConfig.from_pretrained(model_dir)
    processor = Qwen3ASRProcessor.from_pretrained(model_dir)
    
    model.eval()
    model.thinker.audio_tower.eval()
    # DEVICE = torch.device("cpu")
    # model.to(DEVICE)
    
    logger = get_root_logger()

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    # model_name = Path(model_dir).stem
    cfg_name = f"{model_name}_{target_device}"

    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    
    head_dim = cfg.thinker_config.text_config.head_dim
    num_heads = cfg.thinker_config.text_config.num_attention_heads
    num_key_value_heads = cfg.thinker_config.text_config.num_key_value_heads
    embed_dim = cfg.thinker_config.text_config.hidden_size
    num_decode_layers = cfg.thinker_config.text_config.num_hidden_layers

    max_source_positions = cfg.thinker_config.audio_config.max_source_positions
    # 手动设置的固定音频长度，用于在导出 ONNX/HMONNX 时固定 Encoder 的输入时间维度 T
    max_audio_length = int(args.max_audio_length)
    model.thinker.audio_tower.config.fixed_max_audio_length = max_audio_length
    try:
        model.config.thinker_config.audio_config.fixed_max_audio_length = max_audio_length
    except Exception:
        pass
    try:
        cfg.thinker_config.audio_config.fixed_max_audio_length = max_audio_length
    except Exception:
        pass
    
    meta_info = {}  
    meta_info_file = work_dir / "meta_info.json"
    if meta_info_file.exists():
        with open(meta_info_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    meta_info["hf_model"] = model_dir
    meta_info["model_cfg"] = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "embed_dim": embed_dim,
        "max_source_positions": max_source_positions,
        "num_decode_layers": num_decode_layers,
    }
    
    # encoder 处理过程 =======================================================================================
    name = "Encoder"
    encoder_work_dir = work_dir / name
    encoder_work_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = encoder_work_dir / f"{model_name}_{name}.onnx"
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = (
        encoder_work_dir / "hmonnx" / f"{model_name}_{name}_xh2a_{quant_type}.onnx"
    )
    golden_path = encoder_work_dir / "hmonnx/golden"
    meta_info["encoder"] = str(hmonnx_file.relative_to(work_dir))
    num_mel_bins = model.config.thinker_config.audio_config.num_mel_bins
    # 记录导出所用的关键维度参数，便于下游核对与复现
    meta_info["model_cfg"]["num_mel_bins"] = num_mel_bins
    meta_info["model_cfg"]["fixed_max_audio_length"] = max_audio_length
    
    # 使用手动指定的 max_audio_length 固定 Encoder 输入的时间维度 T，同时使用配置中的 mel 维度
    input_features = torch.randn(1, num_mel_bins, max_audio_length).to(model.device).to(model.dtype)
    # 对应的长度张量需要与 T 保持一致，以确保导出后的图输入形状固定
    feature_lens = torch.tensor([max_audio_length], dtype=torch.int32).to(model.device)
    
    # 1. 导出onnx
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    model.thinker.audio_tower,
                    (input_features, feature_lens),  # 传入 input_features 和 feature_lens
                    temp_onnx_file,
                    input_names=["input_features", "feature_lens"],
                    output_names=["hidden_state"],
                    # dynamo=True,
                )
                onnx_model = onnx.load(temp_onnx_file)
                model_byte_size = onnx_model.ByteSize()
                if model_byte_size <= _LARGE_MODEL_SIZE_THRESHOLD:
                    onnx_model_sim, checked = onnxsim.simplify(
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
                    onnx_model_sim, checked = simplify_large_onnx(
                        onnx_model,
                        skipped_optimizers=[
                            "fuse_pad_into_conv",
                            "fuse_consecutive_slices",
                            "eliminate_common_subexpression",
                            "fuse_qkv",
                        ],
                    )
                if checked:
                    onnx_model = onnx_model_sim
    else:
        onnx_model = onnx.load(onnx_file)
    if not os.path.exists(onnx_file):
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(onnx_file).stem}_external_data",
        )
    print(f"✅ ONNX 模型已保存: {onnx_file}, 大小: {onnx_model.ByteSize() / GB:.2f} GB")
        
    # 2. 构造输入
    output_names = []
    output_names.append("hidden_state")
    # 3. 转换
    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features, feature_lens],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features", "feature_lens"],
            output_names=output_names,
        )

    # 生成golden
    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(encoder_work_dir / "hmonnx/golden")
        session.step = 0
        session(input_features.half().to("cuda"), feature_lens.to("cuda"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/Qwen/Qwen3-ASR-0.6B/"))
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--quant-type", default="w8a8_sefp", help="quant type, default is w8a8"
    )
    parser.add_argument(
        "--gen_golden", action="store_true", help="generate golden data"
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        default=1500,
        help="手动固定 Encoder 输入的时间维度 T"
    )
    args = parser.parse_args()
    main(args)

# python hmonnx_export_encoder.py --model ~/models/Qwen/Qwen3-ASR-0.6B/ --quant-type w8a8_sefp --max_audio_length 1500
