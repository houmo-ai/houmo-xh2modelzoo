import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    DeviceType,
    convert_fx_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)
import torch.nn as nn
from xhquant.utils import set_random_seed
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_hf_compatible import MiniCPMO_HFCompatible


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


class MinicpmoLLMConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoLLMConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

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

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.dtype = "float16"

        seed = 1024
        set_random_seed(seed)

        logger = get_root_logger()

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "llm"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / "llm_config.json")

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

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

        video_path = cfg.video
        ref_audio_path = cfg.audio
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        sys_msg = native_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")

        contents = get_video_chunk_content(video_path)
        msg = {"role": "user", "content": contents}
        msgs = [sys_msg, msg]

        llm_hf_args: List[Any] = []
        llm_hf_kwargs: Dict[str, Any] = {}

        def _get_hf_llm_inputs(module, args, kwargs):
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    llm_hf_args.append(torch.empty_like(arg))
                else:
                    llm_hf_args.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    llm_hf_kwargs[k] = torch.empty_like(v)
                else:
                    llm_hf_kwargs[k] = v
            assert False

        native_model.to(device)
        native_model.to(dtype)

        handle = None
        try:
            handle = native_model.llm.register_forward_pre_hook(_get_hf_llm_inputs, with_kwargs=True)
            with torch.no_grad():
                native_model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.5,
                    max_new_tokens=256,
                    omni_input=True,
                    use_tts_template=True,
                    generate_audio=False,
                    output_audio_path=None,
                    max_slice_nums=1,
                    use_image_id=False,
                    return_dict=True,
                )
        except Exception:
            pass
        finally:
            if handle is not None:
                del native_model.llm._forward_pre_hooks[handle.id]
                del native_model.llm._forward_pre_hooks_with_kwargs[handle.id]

        input_shapes = [list(x.shape) for x in llm_hf_args if isinstance(x, torch.Tensor)]
        kwarg_shapes = {k: list(v.shape) for k, v in llm_hf_kwargs.items() if isinstance(v, torch.Tensor)}
        logger.info("************************ native model profile ************************")
        logger.info(f"input_shapes: {input_shapes}")
        logger.info(f"kwarg_shapes: {kwarg_shapes}")

        meta_info = ConfigDict(
            dict(
                create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                config=str(Path(config_file).relative_to(out_model_dir)),
            )
        )
        meta_info["hf_model"] = xh_model.hf_model_dir
        meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()

        json.dump(meta_info, open(out_model_dir / "meta_info.json", "w"), indent=4)

        xh_model.init_wrap_model()
        xh_model.wrap_processor(native_model)
        # ensure export_cfg (with kv cache inputs) is ready
        export_input_names = list(xh_model.export_cfg.get("input_names", []))
        export_output_names = list(xh_model.export_cfg.get("output_names", ["logits", "hidden_state"]))

        # ensure kv caches are ready
        xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
        xh_model.to(dtype)
        xh_model.to(device)

        native_model.to(dtype)
        native_model.to(device)

        native_model.init_tts()

        MiniCPMO_HFCompatible.to_hf_compatible(native_model, llm_model=xh_model)

        if cfg.valid:
            logger.info("************************ valid wraped model ************************")
            with torch.no_grad():
                res = native_model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.5,
                    max_new_tokens=256,
                    omni_input=True,
                    use_tts_template=True,
                    generate_audio=False,
                    output_audio_path=None,
                    max_slice_nums=1,
                    use_image_id=False,
                    return_dict=True,
                )
            logger.info(res)

        net_inputs: List[Optional[torch.Tensor]] = [None] * 3
        past_key_caches: List[torch.Tensor] = []
        past_value_caches: List[torch.Tensor] = []

        def get_llm_inputs(module, args, kwargs):
            if len(args) >= 3:
                net_inputs[0] = args[0]
                net_inputs[1] = args[1]
                net_inputs[2] = args[2]
            if "inputs_embeds" in kwargs:
                net_inputs[0] = kwargs["inputs_embeds"]
            if "past_seq_length" in kwargs:
                net_inputs[1] = kwargs["past_seq_length"]
            if "current_input_length" in kwargs:
                net_inputs[2] = kwargs["current_input_length"]
            if len(args) >= 4:
                past_key_caches.extend(args[3])
            if len(args) >= 5:
                past_value_caches.extend(args[4])
            if "past_key_caches" in kwargs:
                past_key_caches.extend(kwargs["past_key_caches"])
            if "past_value_caches" in kwargs:
                past_value_caches.extend(kwargs["past_value_caches"])
            assert False

        handle = None
        try:
            handle = xh_model.register_forward_pre_hook(get_llm_inputs, with_kwargs=True)
            with torch.no_grad():
                native_model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.5,
                    max_new_tokens=256,
                    omni_input=True,
                    use_tts_template=True,
                    generate_audio=False,
                    output_audio_path=None,
                    max_slice_nums=1,
                    use_image_id=False,
                    return_dict=True,
                )
        except Exception:
            pass
        finally:
            if handle is not None:
                del xh_model._forward_pre_hooks[handle.id]
                del xh_model._forward_pre_hooks_with_kwargs[handle.id]

        flat_inputs: List[torch.Tensor] = []
        # prepare tensors and cast
        for idx, value in enumerate(net_inputs):
            if value is None:
                continue
            if value.dtype in (torch.int64, torch.int32):
                value = value.to(device).to(torch.int32)
            else:
                value = value.to(device).to(dtype)
            net_inputs[idx] = value
        flat_inputs.extend([t for t in net_inputs if t is not None])

        # caches
        kv_tensors: List[torch.Tensor] = []
        for cache_list in (past_key_caches, past_value_caches):
            for cache in cache_list:
                cache_tensor = cache.data if hasattr(cache, "data") else cache
                if cache_tensor.dtype in (torch.int64, torch.int32):
                    cache_tensor = cache_tensor.to(device).to(torch.int32)
                else:
                    cache_tensor = cache_tensor.to(device).to(dtype)
                kv_tensors.append(cache_tensor)
        flat_inputs.extend(kv_tensors)

        logger.info(f"inputs_embeds shape: {net_inputs[0].shape if net_inputs[0] is not None else None}")
        logger.info(f"past_seq_length shape: {net_inputs[1].shape if net_inputs[1] is not None else None}")
        logger.info(
            f"current_input_length shape: {net_inputs[2].shape if net_inputs[2] is not None else None}"
        )
        logger.info(f"past_key_caches: {len(past_key_caches)} past_value_caches: {len(past_value_caches)}")

        xh_model.to(dtype)

        onnx_dir = out_model_dir
        onnx_file = onnx_dir / f"{cfg_name}.onnx"

        # build a thin wrapper so the frontend sees a flat signature matching input_names
        num_layers = len(past_key_caches)

        def _build_wrapper(model: nn.Module, num_layers: int) -> nn.Module:
            # Dynamically build a wrapper with explicit kv args so TorchFX sees all placeholders.
            kv_args = [f"kv{i}" for i in range(num_layers * 2)]
            arg_list = ["self", "inputs_embeds", "past_seq_length", "current_input_length"] + kv_args
            arg_str = ", ".join(arg_list)

            body_lines = []
            body_lines.append("    past_keys = [")
            for i in range(num_layers):
                body_lines.append(f"        kv{i},")
            body_lines.append("    ]")
            body_lines.append("    past_values = [")
            for i in range(num_layers, num_layers * 2):
                body_lines.append(f"        kv{i},")
            body_lines.append("    ]")
            body_lines.append(
                "    return self.model(inputs_embeds, past_seq_length, current_input_length, past_keys, past_values)"
            )

            src = [f"def forward({arg_str}):"]
            src.extend(body_lines)
            src_code = "\n".join(src)

            local_vars: Dict[str, Any] = {}
            exec(src_code, globals(), local_vars)
            forward_fn = local_vars["forward"]

            class _LLMWrapper(nn.Module):
                def __init__(self, model: nn.Module):
                    super().__init__()
                    self.model = model

                forward = forward_fn

            return _LLMWrapper(model)

        wrapper = _build_wrapper(xh_model._wrap_model, num_layers)

        # prefer export_cfg input/output names if present, else fallback
        if export_input_names:
            input_names = export_input_names
        else:
            input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
            input_names.extend([f"past_key_cache_{i}" for i in range(num_layers)])
            input_names.extend([f"past_value_cache_{i}" for i in range(num_layers)])
        output_names = export_output_names if export_output_names else ["logits", "hidden_state"]

        convert_fx_model_to_hmonnx(
            wrapper,
            flat_inputs,
            cfg.target_device,
            onnx_file,
            quant_config=cfg.quant_config,
            input_names=input_names,
            output_names=output_names,
        )
        logger.info(f"Export onnx to {onnx_file}")

        meta_info.llm_hmonnx = str(Path(onnx_file).relative_to(onnx_dir))
        json.dump(meta_info, open(onnx_dir / "meta_info.json", "w"), indent=4)

        from xhquant.api import HMONNXGoldenInference

        hm_model = HMONNXGoldenInference(onnx_file)
        hm_model.save_golden = True
        hm_model.exec_device = device

        golden_dir = Path(cfg.work_dir) / "golden" / f"{Path(onnx_file).stem}"
        golden_dir.mkdir(exist_ok=True, parents=True)
        hm_model.golden_dir = str(golden_dir)

        with torch.no_grad():
            hm_model.forward(*flat_inputs)

    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoLLMConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoLLMConverterXH2a(config)._convert(hf_model_path, output_dir)

