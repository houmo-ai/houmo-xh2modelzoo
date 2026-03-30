import os
import sys
import json
import stat

import librosa
import argparse
import tempfile
import numpy as np

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

from onnx import TensorProto, helper, numpy_helper

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

# ONNX TensorProto 类型 → numpy dtype 映射
_ONNX_DTYPE_TO_NUMPY = {
    TensorProto.FLOAT:   np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE:  np.float64,
    TensorProto.INT8:    np.int8,
    TensorProto.INT16:   np.int16,
    TensorProto.INT32:   np.int32,
    TensorProto.INT64:   np.int64,
    TensorProto.UINT8:   np.uint8,
    TensorProto.UINT16:  np.uint16,
    TensorProto.UINT32:  np.uint32,
    TensorProto.UINT64:  np.uint64,
    TensorProto.BOOL:    np.bool_,
}

def change_onnx_initializer_type(
    input_model_path: str,
    output_model_path: str,
    target_initializer_name: str,
    new_data_type: int = TensorProto.FLOAT16
):
    input_model_path = os.path.abspath(input_model_path)
    output_model_path = os.path.abspath(output_model_path)
    print(f"📌 规范化路径：")
    print(f"   输入模型：{input_model_path}")
    print(f"   输出模型：{output_model_path}")
    
    # 1. 检查输入文件存在性 + 强制赋予读权限
    if not os.path.exists(input_model_path):
        raise FileNotFoundError(f"输入模型不存在：{input_model_path}")
    
    # 强制添加读权限（针对当前用户）
    try:
        os.chmod(input_model_path, os.stat(input_model_path).st_mode | stat.S_IRUSR)
        print(f"✅ 已赋予输入文件读权限：{input_model_path}")
    except Exception as e:
        print(f"⚠️ 赋予读权限失败（可能需要sudo）：{e}")
    
    # 2. 检查输出目录 + 强制赋予写权限
    output_dir = os.path.dirname(output_model_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True, mode=0o755) 
        print(f"✅ 已创建输出目录：{output_dir}")
    
    # 强制添加输出目录写权限
    try:
        os.chmod(output_dir, os.stat(output_dir).st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        print(f"✅ 已赋予输出目录读写执行权限：{output_dir}")
    except Exception as e:
        print(f"⚠️ 赋予输出目录权限失败（可能需要sudo）：{e}")
    
    try:
        model = onnx.load(input_model_path)
        print("✅ 原始模型加载成功")
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{e}")

    target_np_dtype = _ONNX_DTYPE_TO_NUMPY.get(new_data_type)
    if target_np_dtype is None:
        raise ValueError(f"不支持的目标 ONNX 类型：{new_data_type}")

    modified = False
    for i, init in enumerate(model.graph.initializer):
        if init.name == target_initializer_name:
            old_type = init.data_type
            np_array = numpy_helper.to_array(init)
            np_array_converted = np_array.astype(target_np_dtype)
            new_init = numpy_helper.from_array(np_array_converted, name=init.name)
            model.graph.initializer[i].CopyFrom(new_init)
            print(f"✅ 修改 initializer [{target_initializer_name}]：")
            print(f"   原始类型：{TensorProto.DataType.Name(old_type)} ({np_array.dtype})"
                  f" → 新类型：{TensorProto.DataType.Name(new_data_type)} ({target_np_dtype.__name__})")
            print(f"   shape: {np_array.shape}")
            modified = True
            break

    if not modified:
        print(f"❌ 未找到 initializer：{target_initializer_name}")
        print("📋 模型中前20个 initializer 名称：")
        for idx, init in enumerate(model.graph.initializer[:20]):
            print(f"   {idx+1}. {init.name}  dtype={TensorProto.DataType.Name(init.data_type)}")
        raise ValueError(f"未找到指定的 initializer：{target_initializer_name}")

    for inp in model.graph.input:
        if inp.name == target_initializer_name:
            inp.type.tensor_type.elem_type = new_data_type
            print(f"✅ 同步更新 graph.input [{target_initializer_name}] 的类型声明")
            break

    for vi in model.graph.value_info:
        if vi.name == target_initializer_name:
            vi.type.tensor_type.elem_type = new_data_type
            print(f"✅ 同步更新 graph.value_info [{target_initializer_name}] 的类型声明")
            break

    try:
        onnx.checker.check_model(model)
        print("✅ 修改后模型结构验证通过")
    except onnx.checker.ValidationError as e:
        print(f"⚠️ 模型验证警告（可忽略 hmonnx 自定义算子警告）：{e}")

    output_stem = os.path.splitext(os.path.basename(output_model_path))[0]
    model_byte_size = model.ByteSize()
    use_external_data = model_byte_size > _LARGE_MODEL_SIZE_THRESHOLD
    print(f"📌 模型大小：{model_byte_size / (1 << 30):.3f} GB，"
          f"{'使用' if use_external_data else '不使用'} external data 格式保存")
    try:
        if use_external_data:
            onnx.save(
                model,
                output_model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{output_stem}_external_data",
            )
        else:
            onnx.save(model, output_model_path)
        print(f"✅ 模型已保存至：{output_model_path}")
        saved_model = onnx.load(output_model_path)
        print(f"✅ 回读验证通过，initializer 数量：{len(saved_model.graph.initializer)}")
    except Exception as e:
        temp_output = os.path.join("/tmp", "modified_model_temp.onnx")
        try:
            onnx.save(model, temp_output)
            os.rename(temp_output, output_model_path)
            print(f"✅ 临时目录保存后移动至目标路径：{output_model_path}")
        except Exception as e2:
            raise RuntimeError(f"保存模型失败：\n主方案：{e}\n临时目录方案：{e2}")


def find_less_int32_initializers_to_fp16(
    model_path: str,
    node_name_hint: str = "node_less_2",
):
    model_path = os.path.abspath(model_path)
    model = onnx.load(model_path)

    init_dtype_map = {init.name: init.data_type for init in model.graph.initializer}

    type_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        try:
            type_map[vi.name] = vi.type.tensor_type.elem_type
        except Exception:
            continue

    candidates = []
    for node in model.graph.node:
        if node.op_type == node_name_hint or node.name == node_name_hint or node_name_hint in (node.name or ""):
            candidates.append(node)

    if not candidates:
        for node in model.graph.node:
            if node.op_type == "Less":
                candidates.append(node)

    targets = set()
    for node in candidates:
        if len(node.input) < 2:
            continue
        for idx, in_name in enumerate(node.input):
            if init_dtype_map.get(in_name) != TensorProto.INT32:
                continue
            other_name = None
            for j, n in enumerate(node.input):
                if j != idx:
                    other_name = n
                    break
            other_dtype = init_dtype_map.get(other_name, type_map.get(other_name))
            if other_dtype in (TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE) or other_dtype is None:
                targets.add(in_name)
    print(f"找到 {len(targets)} 个需要转换的 initializer：{targets}，分别是 {', '.join(targets)}")
    return sorted(targets)


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

        target_inits = find_less_int32_initializers_to_fp16(
            str(hmonnx_file),
            node_name_hint="node_less_2",
        )
        if len(target_inits) == 0:
            print("⚠️ 未找到需要转换为 FP16 的 node_less_2/Less INT32 initializer，跳过 dtype 修复")
        else:
            for init_name in target_inits:
                change_onnx_initializer_type(
                    input_model_path=hmonnx_file,
                    output_model_path=hmonnx_file,
                    target_initializer_name=init_name,
                    new_data_type=TensorProto.FLOAT16,
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
