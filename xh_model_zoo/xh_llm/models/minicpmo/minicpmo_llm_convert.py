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

from ..base_converter import HFTransfromersConverter
from .minicpmo_llm_convert_config import MinicpmoLLMConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    ConfigDict,
    PrecisionMode,
    DeviceType,
    ptq_quantize,
    convert_fx_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)
import torch.nn as nn
from xhquant.utils import set_random_seed
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_hf_compatible import MiniCPMO_HFCompatible
from xh_model_zoo.utils.time_profiler import TimeProfiler
from ..llm_base_model import LLMBaseModel
from xh_model_zoo.xh_llm.utils import decode_next_token


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
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

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
    tokenizer,
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

    logger.info("Start exporting graph.............")
    with TimeProfiler("export graph"):
        xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exported graph.")

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(execution_device)
        xh_model.to(dtype)
        xh_model.set_exec_device(execution_device)
        with torch.no_grad():
            logits, hidden_states = xh_model.test_step(data_batch)
            next_tokens, next_token_str = decode_next_token(tokenizer, logits)
        logger.info(f"Exported model next token: {next_tokens} {next_token_str}")

        xh_model.to("cpu")  # 切换到cpu上进行模型导出
        torch.cuda.empty_cache()

    # memory_tracker.log_memory("before exporting onnx", logger)
    logger.info("*************** Start exporting onnx ***************")
    with TimeProfiler("export onnx"):
        onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    # memory_tracker.log_memory("after exporting onnx", logger)
    return onnx_file


class MinicpmoLLMConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoLLMConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

    def _convert(self, hf_model_path: str, output_dir: str):
        cfg = self.config
        is_valid = cfg.valid
        cfg.target_device = "XH2a"
        cfg.hf_model_dir = hf_model_path
        cfg_name = Path(hf_model_path).name
        if cfg.debug:
            cfg_name = f"{cfg_name}_debug"
        work_dir = output_dir
        cfg.work_dir = str(work_dir)
        os.makedirs(work_dir, exist_ok=True)

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.dtype = "float16"

        seed = 1024
        set_random_seed(seed)

        logger = get_root_logger()

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "llm"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / "llm_config.json")
        # cfg.dump(config_file)

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        from .minicpmo_llm_model import XHMiniCPMOLLMModel

        xh_model = XHMiniCPMOLLMModel(
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
                    ],
                    output_names=["logits", "hidden_state"],
                )
            ),
        )
        native_model = xh_model.get_hf_model()

        token_embedding = native_model.llm.get_input_embeddings()
        token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
        torch.save(token_embedding, str(token_embedding_file))
    
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

        MiniCPMO_HFCompatible.to_hf_compatible(native_model, llm_model=xh_model)

        # 添加 pre hook 仅捕获 inputs_embeds 并存到 input_data
        original_generate = native_model.llm.generate

        input_data = dict()


        def generate_with_hook(*args, **kwargs):
            # 每次调用只记录 inputs_embeds（保持原始张量，不做拷贝/迁移）
            if "inputs_embeds" in kwargs:
                input_data.clear()
                input_data["inputs_embeds"] = kwargs["inputs_embeds"]
            return original_generate(*args, **kwargs)

        native_model.llm.generate = generate_with_hook

        if is_valid:
            logger.info("************************ valid wraped model ************************")
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


        native_model.to("cpu")
        cleanup_memory()

        xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
        xh_model.to(dtype)
        xh_model.to(device)

        data_batch = {
            "inputs_embeds": input_data["inputs_embeds"].cpu(),
            "past_seq_length": 0,
        }

        data_prefill = xh_model.prepare_inputs_for_graph(data_batch)

        if is_valid:
            xh_model.reset_kvcache()
            logist, hidden_states = xh_model.test_step(data_batch)
            prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, logist)
            logger.info(f"Prefill Wraped Model next token: {prefill_next_token_id} {prefill_next_token_text}")

        logger.info("************* convert to frontend graph *************")
        xh_model._wrap_model.export_projector = True
        xh_model.convert_to_fronted_graph(data_batch)
        logger.info("************* convert to quanted graph *************")
        xh_model.convert_to_quant_graph(cfg.target_device)

        ## 进行PTQ量化
        logger.info("*************** Start PTQ Quantize ***************")
        calib_data = data_prefill
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

        if is_valid:
            xh_model.to(device)
            with torch.no_grad():
                with TimeProfiler("QUANTED_ALIGNED", logger):
                    outs = xh_model.test_step(data_batch)
                quanted_aligned_logits = outs[0].detach()
            prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, quanted_aligned_logits)
            logger.info(f"Prefill Quanted Aligned Model next token: {prefill_next_token_id} {prefill_next_token_text}")

        prefill_onnx_dir = out_model_dir / "prefill_onnx"
        prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
        decode_onnx_dir = out_model_dir / f"decode_onnx"
        decode_onnx_dir.mkdir(exist_ok=True, parents=True)
        
        if True:
            logger.info("*************** Start exporting decode model ***************")
            prefill_onnx_file = xhmodel_export_onnx(
                xh_model,
                tokenizer,
                data_batch,
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
            del hm_model
            cleanup_memory()

        if True:
            xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
            xh_model.set_input_sequence_length(1)
            if is_valid:
                data_decode = {
                    "inputs_embeds": token_embedding(prefill_next_token_id.cpu()).detach(),
                    "past_seq_length": data_batch["inputs_embeds"].shape[-2],
                }
            else:
                data_decode = {
                "inputs_embeds": token_embedding(torch.randint(0, 1000, (1, 1))).detach(),
                "past_seq_length": 256,
            }
            torch.cuda.empty_cache()
            logger.info("*************** Start exporting decode model ***************")
            with TimeProfiler("export decode onnx", logger):
                decode_onnx_file = xhmodel_export_onnx(
                    xh_model,
                    tokenizer,
                    data_decode,
                    str(decode_onnx_dir),
                    f"{cfg_name}_decode",
                    device,
                    dtype,
                    logger,
                    is_valid,
                )
            xh_model.release_exported_model()
            meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
            logger.info(f"save decode onnx model to {decode_onnx_file}")
            logger.info("*************** Finished exporting decode model ***************")

            cleanup_memory()

            from xhquant.api import HMONNXGoldenInference
            hm_model = HMONNXGoldenInference(decode_onnx_file)
            hm_model.save_golden = True
            hm_model.exec_device = device

            golden_dir = Path(cfg.work_dir) / "golden" / f"{Path(decode_onnx_file).stem}"
            golden_dir.mkdir(exist_ok=True, parents=True)
            hm_model.golden_dir = str(golden_dir)

            calib_data = xh_model.prepare_inputs_for_graph(data_decode)
            new_args = []
            for arg in calib_data:
                if isinstance(arg, (List, Tuple)):
                    new_args.extend(arg)
                else:
                    new_args.append(arg)
            calib_data = new_args
            with torch.no_grad():
                calib_data[1] = calib_data[1].to(torch.int32)
                calib_data[2] = calib_data[2].to(torch.int32)
                calib_data = [p_data.to("cpu") for p_data in calib_data]
                hm_model.forward(*calib_data)
            del hm_model
            cleanup_memory()

        meta_file = str(Path(cfg.work_dir) / "export_meta_info.json")
        json.dump(meta_info, open(meta_file, "w"), indent=4)
        logger.info(f"Save meta info to {meta_file}")


    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoLLMConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoLLMConverterXH2a(config)._convert(hf_model_path, output_dir)

