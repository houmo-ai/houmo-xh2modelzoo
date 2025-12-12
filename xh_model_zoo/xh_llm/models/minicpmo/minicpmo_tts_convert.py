import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoTokenizer
import numpy as np
from copy import deepcopy
from xh_model_zoo.utils import MemoryTracker, TimeProfiler


from ..base_converter import HFTransfromersConverter
from .minicpmo_tts_convert_config import MinicpmoTTSConvertConfig
from ..llm_base_model import LLMBaseModel

from xhquant.api import (  # type: ignore # isort:skip
    ConfigDict,
    DeviceType,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)
from xhquant.api import PrecisionMode, ptq_quantize, set_random_seed
from xhquant.utils import set_random_seed
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.minicpmo import MiniCPMO_HFCompatible


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            print(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        # sr采样率（默认22050，但是有重采样的功能）
        # mono 设置为true是单通道，否则是双通道
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])

    return contents


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    execution_device,
    dtype,
    logger,
    valid: bool = True,
):
    logger = get_root_logger()

    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    torch.cuda.empty_cache()

    # memory_tracker = MemoryTracker(execution_device)
    # memory_tracker.log_memory("Before exporting graph", logger)
    xh_model.set_input_sequence_length(data_batch["current_input_length"].cpu().item())
    logger.info("Start exporting graph.............")
    with TimeProfiler("export graph"):
        xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exported graph.")

    # memory_tracker.log_memory("after exporting graph", logger)

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(execution_device)
        xh_model.to(dtype)
        xh_model.set_exec_device(execution_device)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            logits = outs.logits.detach()

        xh_model.to("cpu")  # 切换到cpu上进行模型导出
        torch.cuda.empty_cache()

    # memory_tracker.log_memory("before exporting onnx", logger)
    logger.info("*************** Start exporting onnx ***************")
    with TimeProfiler("export onnx"):
        onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    # memory_tracker.log_memory("after exporting onnx", logger)
    return onnx_file


class MinicpmoTTSConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoTTSConvertConfig):
        super().__init__()
        self.config = config

    def _convert(self, hf_model_path: str, output_dir: str):
        cfg = self.config
        cfg.target_device = "XH2a"
        cfg.hf_model_dir = hf_model_path
        cfg_name = Path(hf_model_path).name
        if cfg.debug:
            cfg_name = f"{cfg_name}_debug"
        work_dir = output_dir
        cfg.work_dir = str(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        is_valid = cfg.valid

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.dtype = "float16"

        seed = 1024
        set_random_seed(seed)

        logger = get_root_logger()

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "tts"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / "tts_config.json")

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

        from .minicpmo_tts_model import XHMiniCPMOTTSModel

        xh_model = XHMiniCPMOTTSModel(
            hf_model=cfg.hf_model_dir,
            frontend_type="TorchFX",
            wrap_cfg=ConfigDict(
                batch_size=1,
                max_sequence_length=cfg.context_length,
                input_sequence_length=cfg.input_sequence_length,
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(cache_axis=2),
                image_slice_max_size=cfg.image_slice_max_size,
            ),
            quant_config=ConfigDict(),
            export_cfg=ConfigDict(
                dict(
                    input_names=[
                        "inputs_embeds",
                        "past_seq_length",
                        "current_input_length",
                        "attention_mask",
                    ],
                    output_names=["logits"],
                )
            ),
        )
        native_model = xh_model.get_hf_model()

        xh_model.init_wrap_model()
        xh_model.wrap_processor(native_model)
        meta_info = ConfigDict(
            dict(
                create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                config=str(Path(config_file).relative_to(out_model_dir)),
            )
        )
        meta_info["hf_model"] = xh_model.hf_model_dir
        meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
        meta_info["patch_size"] = xh_model.patch_size
        meta_info["num_patches_per_side"] = xh_model.num_patches_per_side

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

        video_path = cfg.video
        # if use voice clone prompt, please set ref_audio
        ref_audio_path = cfg.audio
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        sys_msg = native_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")

        contents = get_video_chunk_content(video_path)
        msg = {"role": "user", "content": contents}
        msgs = [sys_msg, msg]

        # please set generate_audio=True and output_audio_path to save the tts result
        generate_audio = True
        output_audio_path = "output.wav"

        xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
        xh_model.to(dtype)
        xh_model.to(device)

        native_model.to(dtype)
        native_model.to(device)

        native_model.init_tts()

        MiniCPMO_HFCompatible.to_hf_compatible(native_model, tts_llama_model=xh_model)

        original_forward = native_model.tts._tts_llama_model._forward

        input_data = list()
        def forward_with_hook(*args, **kwargs):
            p_input_data = dict()
            for k, v in kwargs.items():
                p_input_data[k] = v
            input_data.append(p_input_data)
            return original_forward(*args, **kwargs)
        
        native_model.tts._tts_llama_model._forward = forward_with_hook

        with torch.no_grad():
            res = native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,  # please set omni_input=True when omni inference
                use_tts_template=True,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
            logger.info(res)
        
        native_model.tts._tts_llama_model._forward = original_forward

        data_prefill = input_data[0]
        logger.info("************* convert to frontend graph *************")
        xh_model.convert_to_fronted_graph(data_prefill)
        logger.info("************* convert to quanted graph *************")
        xh_model.convert_to_quant_graph(cfg.target_device)

        ## 进行PTQ量化
        logger.info("*************** Start PTQ Quantize ***************")
        calib_data = xh_model.prepare_inputs_for_graph(data_prefill)
        ## 将输入的List展开
        new_args = []
        for arg in calib_data:
            if isinstance(arg, (List, Tuple)):
                new_args.extend(arg)
            else:
                new_args.append(arg)
        calib_data = new_args
        ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [device])
        logger.info("*************** Finished PTQ Quantize ***************")
        
        xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
        xh_model.to(dtype)

        with torch.no_grad():
            res = native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,  # please set omni_input=True when omni inference
                use_tts_template=True,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
            logger.info(res)

        print("a")
        
        prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
        decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
        prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
        decode_onnx_dir.mkdir(exist_ok=True, parents=True)
        
        xh_model.reset_kvcache()
        if True:
            logger.info("*************** Start exporting prefill model ***************")
            prefill_onnx_file = xhmodel_export_onnx(
                xh_model,
                data_prefill,
                str(prefill_onnx_dir),
                f"{cfg_name}_prefill",
                device,
                dtype,
                logger,
                is_valid,
            )
            xh_model.release_exported_model()
            xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
            meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
            logger.info(f"save prefill onnx model to {prefill_onnx_file}")
            logger.info("*************** Finished export prefill model ***************")
            cleanup_memory()

            from xhquant.api import HMONNXGoldenInference
            hm_model = HMONNXGoldenInference(prefill_onnx_file)
            hm_model.save_golden = True
            hm_model.exec_device = device

            golden_dir = Path(cfg.work_dir) / "golden" / f"{Path(prefill_onnx_file).stem}"
            golden_dir.mkdir(exist_ok=True, parents=True)
            hm_model.golden_dir = str(golden_dir)

            with torch.no_grad():
                calib_data[1] = calib_data[1].to(torch.int32)
                calib_data[2] = calib_data[2].to(torch.int32)
                calib_data = [p_data.to("cpu") for p_data in calib_data]
                hm_model.forward(*calib_data)
        
        if True:
            logger.info("*************** Start exporting decode model ***************")
            data_decode = input_data[1]
            decode_onnx_file = xhmodel_export_onnx(
                xh_model,
                data_decode,
                str(decode_onnx_dir),
                f"{cfg_name}_decode",
                device,
                dtype,
                logger,
                is_valid,
            )
            xh_model.release_exported_model()
            xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
            meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
            logger.info(f"save decode onnx model to {decode_onnx_file}")
            logger.info("*************** Finished export decode model ***************")

            from xhquant.api import HMONNXGoldenInference
            hm_model = HMONNXGoldenInference(decode_onnx_file)
            hm_model.save_golden = True
            hm_model.exec_device = device

            golden_dir = Path(cfg.work_dir) / "golden" / f"{Path(decode_onnx_file).stem}"
            golden_dir.mkdir(exist_ok=True, parents=True)
            hm_model.golden_dir = str(golden_dir)

            with torch.no_grad():
                data_decode = list(xh_model.prepare_inputs_for_graph(data_decode))

                new_args = []
                for arg in data_decode:
                    if isinstance(arg, (List, Tuple)):
                        new_args.extend(arg)
                    else:
                        new_args.append(arg)
                data_decode = new_args
                
                data_decode[1] = data_decode[1].to(torch.int32)
                data_decode[2] = data_decode[2].to(torch.int32)
                
                data_decode = [p_data.to("cpu") for p_data in data_decode]
                hm_model.forward(*data_decode)
        
        meta_file = str(Path(cfg.work_dir) / "export_meta_info.json")
        json.dump(meta_info, open(meta_file, "w"), indent=4)
        logger.info(f"Save meta info to {meta_file}")

    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoTTSConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        if is_ssfp_quant_config(quant_config):
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoTTSConverterXH2a(config)._convert(hf_model_path, output_dir)

