import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import xhquant.utils.suppress_printing
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from xhquant.api import (
    Config,
    ConfigDict,
    PrecisionMode,
    QTensor,
    ptq_quantize,
    set_random_seed,
)

try:
    from .common import decode_next_token, xhquant_llm_init, get_root_logger
except ImportError:
    from common import decode_next_token, xhquant_llm_init, get_root_logger  # pyright: ignore[reportMissingImports]
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.paddleocr_vl import PaddleOCRVL_HFCompatible, PaddleOCRVLProcessor, XHPaddleOCRVLLLMModel
from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler


def _register_xh_tile_decomposition(logger=None):
    try:
        from torch._decomp import register_decomposition

        if hasattr(torch.ops, "xh") and hasattr(torch.ops.xh, "Tile"):

            @register_decomposition(torch.ops.xh.Tile.default)
            def _xh_tile_decomp(input: Tensor, repeats):
                return torch.tile(input, repeats)

            if logger is not None:
                logger.info("Register decomposition: xh.Tile -> torch.tile")
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Skip xh.Tile decomposition registration: {exc}")


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


def load_quarot_gptq_state_dict(native_model, args, logger):
    from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

    rotation_utils.fuse_layer_norms(native_model)
    state_dict = load_safetensors_file(args.quarot_gptq_path)

    model_state_dict = native_model.state_dict()
    unexpect_state_dict = []
    for k, v in state_dict.items():
        if k not in model_state_dict:
            unexpect_state_dict.append(k)

    for k in unexpect_state_dict:
        paths = k.split(".")
        if paths[-1] == "quant_weight":
            submodule_name = ".".join(paths[:-1])
            submodule = native_model.get_submodule(submodule_name)
            # submodule = get_submodule(native_model, k)
            v = state_dict[k]
            if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                v = v.to(torch.int8)
            elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                v = v.to(torch.int16)
            else:
                v = v.to(torch.float32)
            submodule.register_buffer("quant_weight", v, persistent=False)
            logger.debug(f"add quant_weight to {submodule_name}")
        else:
            logger.warning(f"ignore unexpect state dict: {k}")
        state_dict.pop(k)

    native_model.load_state_dict(state_dict)
    del state_dict


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl/paddleocr_vl_llm_config.py",
    )
    parser.add_argument(
        "--hf_model_dir", type=str, default="/data01/datasets/PaddleOCR-VL"
    )
    parser.add_argument("--use_gptq_model", action="store_true", default=False)
    # parser.add_argument("--model_type", type=str, default="2B")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--valid", action="store_true", help="evaluate the model")
    # parser.add_argument("--image_path", type=str, default="data/test/image-2025-09-15-14-09-19-086.png")
    # parser.add_argument("--prompt", type=str, default="请描述这张图片的内容")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--quarot_gptq_path", type=str, default=None)
    script_dir = Path(__file__).parent.parent.parent  # 返回到 xh2modelzoo 目录
    default_image_path = str(script_dir / "data" / "images" / "ocr_img.png")
    parser.add_argument("--image_path", type=str, default=default_image_path)
    parser.add_argument(
        "--task", type=str, default="ocr"
    )  # Options: 'ocr' | 'table' | 'chart' | 'formula'
    return parser


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

    # Ensure execution device is CPU for input preparation
    xh_model.set_exec_device(torch.device("cpu"))

    # Ensure inputs are on CPU to match exported graph device
    data_batch = to_device(data_batch, torch.device("cpu"))
    # Ensure KV cache tensors are on CPU to match exported graph device
    if hasattr(xh_model, "kv_cache_to"):
        xh_model.kv_cache_to(torch.device("cpu"))
    # Ensure rope_deltas on CPU for decode path position_ids computation
    if hasattr(xh_model, "rope_deltas") and xh_model.rope_deltas is not None:
        xh_model.rope_deltas = xh_model.rope_deltas.to(torch.device("cpu"))

    # memory_tracker = MemoryTracker(execution_device)
    # memory_tracker.log_memory("Before exporting graph", logger)

    logger.info("Start exporting graph.............")
    with TimeProfiler("export graph"):
        xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exported graph.")

    # memory_tracker.log_memory("after exporting graph", logger)

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    # if valid:
    #     # Exported graph is CPU; validate on CPU to avoid device mismatch
    #     data_batch_exec = to_device(data_batch, torch.device("cpu"))
    #     xh_model.to("cpu")
    #     xh_model.to(dtype)
    #     xh_model.set_exec_device(torch.device("cpu"))
    #     if hasattr(xh_model, "kv_cache_to"):
    #         xh_model.kv_cache_to(torch.device("cpu"))
    #     with torch.no_grad():
    #         outs = xh_model.test_step(data_batch_exec)
    #         logits = outs.logits.detach()
    #         next_tokens, next_token_str = decode_next_token(tokenizer, logits)
    #     logger.info(f"Exported model next token: {next_tokens} {next_token_str}")

    # torch.cuda.empty_cache()

    # xh_model.to("cpu")  # 切换到cpu上进行模型导出
    # torch.cuda.empty_cache()

    # memory_tracker.log_memory("before exporting onnx", logger)
    logger.info("*************** Start exporting onnx ***************")
    with TimeProfiler("export onnx"):
        onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    # memory_tracker.log_memory("after exporting onnx", logger)
    calib_data = xh_model.prepare_inputs_for_graph(data_batch)
    ## 将输入的List展开
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args
    from xhquant.api import HMONNXGoldenInference

    hm_model = HMONNXGoldenInference(onnx_file)
    hm_model.save_golden = True
    hm_model.exec_device = execution_device
    hm_model.golden_dir = Path(onnx_output_dir) / "golden" / f"{Path(onnx_file).stem}"
    hm_model.golden_dir.mkdir(exist_ok=True, parents=True)
    # Align current_input_length with exported inputs_embeds length
    if isinstance(calib_data[0], Tensor):
        seq_len = calib_data[0].shape[1]
        calib_data[5] = torch.tensor([seq_len], dtype=torch.int32).to(
            calib_data[0].device
        )
    calib_data[1] = calib_data[1].to(torch.float16)
    calib_data[2] = calib_data[2].to(torch.float16)
    calib_data[3] = calib_data[3].to(torch.float16)
    hm_model.forward(*calib_data)
    return onnx_file


def _export_impl(cfg, args):
    logger = get_root_logger()
    is_valid_model = args.valid
    config_file = cfg.config_file
    cfg_name = cfg.cfg_name

    device = torch.device(cfg.device)
    execution_device = torch.device(cfg.execution_device)
    # hf_model_device = "balanced_low_0"
    dtype = getattr(torch, cfg.dtype)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(cfg.work_dir)),
        )
    )

    cfg.hf_model_dir = args.hf_model_dir
    cfg.model.hf_model = args.hf_model_dir
    cfg.model.is_gptqmodel = args.use_gptq_model
    cfg.model.wrap_cfg.max_sequence_length = args.max_sequence_length

    paddleocr_vl_llm_model: XHPaddleOCRVLLLMModel = MODELS.build(cfg.model)
    native_model = paddleocr_vl_llm_model.get_hf_model()
    if args.quarot_gptq_path is not None:
        load_quarot_gptq_state_dict(native_model, args, logger)
        logger.info(f"Load state_dict from {args.quarot_gptq_path}")

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(cfg.work_dir)),
        )
    )

    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_file_candidates = [
        ["chat_template.json", "chat_template.jinja"],  # 支持两种模版文件命名
        ["config.json"],
        ["generation_config.json"],
        ["preprocessor_config.json"],
        ["tokenizer_config.json"],
        ["vocab.json"],
        ["tokenizer.json"],
    ]
    for candidates in hf_config_file_candidates:
        copied = False
        for cfg_file in candidates:
            src_file = Path(hf_model_dir) / cfg_file
            if src_file.exists():
                shutil.copyfile(src_file, Path(hf_config_dir) / cfg_file)
                copied = True
                break
        if not copied:
            logger.warning(
                f"Skip copying hf config files, missing {candidates} in {hf_model_dir}"
            )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = native_model.model.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding, str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    hf_model_dir = paddleocr_vl_llm_model.hf_model_dir
    processor = PaddleOCRVLProcessor.from_pretrained(
        hf_model_dir, trust_remote_code=True
    )
    if args.min_pixels is not None:
        processor.image_processor.min_pixels = args.min_pixels
    if args.max_pixels is not None:
        processor.image_processor.max_pixels = args.max_pixels

    image = Image.open(args.image_path).convert("RGB")
    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[args.task]},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    #image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    tokenizer = processor.tokenizer
    native_model.to(execution_device)
    inputs.to(execution_device)

    if is_valid_model:
        with torch.no_grad():
            inputs.pop("hm_pixel_values", None)  # remove the hm_pixel_values if exists
            generated_ids = native_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logger.info("***************** native model output *****************")
        logger.info(output_text[0])

    visual = native_model.visual

    visual.to(execution_device)
    visual.to(dtype)

    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(execution_device).to(dtype)
        image_grid_thw = inputs["image_grid_thw"].to(execution_device)

        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)

        siglip_position_ids = []
        image_grid_hws = []
        sample_indices = []
        cu_seqlens = [0]

        for idx, thw in enumerate(image_grid_thw):
            thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
            numel = np.prod(thw_tuple)
            image_grid_hws.append(thw_tuple)
            image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
            siglip_position_ids.append(image_position_ids)
            sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
            cu_seqlens.append(cu_seqlens[-1] + numel)

        siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
            pixel_values.device
        )
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(pixel_values.device)
        sample_indices = torch.concat(sample_indices, dim=0).to(pixel_values.device)

        vision_outputs = visual(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_hws,
            position_ids=siglip_position_ids,
            vision_return_embed_list=True,
            interpolate_pos_encoding=True,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=False,
            use_rope=True,
            window_size=-1,
        )
        image_embeds = vision_outputs.last_hidden_state
        image_embeds = native_model.mlp_AR(image_embeds, image_grid_thw)
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)

    del visual
    native_model.cpu()

    paddleocr_vl_llm_model.init_wrap_model(native_model)
    del native_model
    paddleocr_vl_llm_model.change_eval_type(EvalModelType.WRAPED)
    paddleocr_vl_llm_model.token_embedding = torch.load(
        token_embedding_file, weights_only=False
    )
    if is_valid_model:
        native_model = paddleocr_vl_llm_model.get_hf_model()
        if args.quarot_gptq_path is not None:
            load_quarot_gptq_state_dict(native_model, args, logger)
            logger.info(f"Load state_dict from {args.quarot_gptq_path}")
        xh2a_hf_compatible_model = PaddleOCRVL_HFCompatible.to_hf_compatible(
            native_model, paddleocr_vl_llm_model
        )
        native_model.to(execution_device)
        native_model.to(dtype)
        paddleocr_vl_llm_model.to(execution_device)
        paddleocr_vl_llm_model.to(dtype)
        with torch.no_grad():
            inputs.pop("hm_pixel_values", None)  # remove the hm_pixel_values if exists
            generated_ids = xh2a_hf_compatible_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logger.info("***************** wrappd model output *****************")
        # logger.info(output_text[0])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if (
        paddleocr_vl_llm_model.past_key_caches is not None
        and len(paddleocr_vl_llm_model.past_key_caches) > 0
    ):
        meta_info.use_cache = True
        meta_info.kv_cache_shape = paddleocr_vl_llm_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(paddleocr_vl_llm_model.past_key_caches)

    data_prefill = {
        "input_ids": inputs["input_ids"],
        "image_embeds": image_embeds,
        "past_seq_length": 0,
        "image_grid_thw": inputs["image_grid_thw"],
    }

    paddleocr_vl_llm_model.change_eval_type(EvalModelType.WRAPED)
    paddleocr_vl_llm_model.to(dtype)
    paddleocr_vl_llm_model.to(device)
    if is_valid_model:
        outs = paddleocr_vl_llm_model.test_step(data_prefill)
        logits = outs.logits.detach()
        prefill_next_token_id, prefill_next_token_text = decode_next_token(
            tokenizer, logits
        )
        logger.info(
            f"Prefill Wraped Model next token: {prefill_next_token_id} {prefill_next_token_text}"
        )

    paddleocr_vl_llm_model.cpu()
    logger.info("************* convert to frontend graph *************")
    paddleocr_vl_llm_model.convert_to_fronted_graph(
        data_prefill, release_wraped_model=False
    )
    logger.info("************* convert to quanted graph *************")
    paddleocr_vl_llm_model.convert_to_quant_graph(cfg.target_device)

    ## 进行PTQ量化
    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = paddleocr_vl_llm_model.prepare_inputs_for_graph(data_prefill)
    ## 将输入的List展开
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args
    # with TimeProfiler("PTQ Quantize", logger), MemoryTracker(execution_device, "ptq", logger):
    ptq_quantize(
        paddleocr_vl_llm_model.quanted_model,
        [calib_data],
        PrecisionMode.ALIGNED,
        [execution_device],
    )
    logger.info("*************** Finished PTQ Quantize ***************")

    paddleocr_vl_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    paddleocr_vl_llm_model.to(dtype)

    if is_valid_model:
        paddleocr_vl_llm_model.to(execution_device)
        paddleocr_vl_llm_model.to(dtype)
        with torch.no_grad():
            inputs.pop("hm_pixel_values", None)  # remove the hm_pixel_values if exists
            generated_ids = xh2a_hf_compatible_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logger.info("***************** wraped model output *****************")
        logger.info(output_text[0])

    if is_valid_model:
        paddleocr_vl_llm_model.to(execution_device)
        with torch.no_grad():
            with TimeProfiler("QUANTED_ALIGNED", logger):
                outs = paddleocr_vl_llm_model.test_step(data_prefill)
            quanted_aligned_logits = outs.logits.detach()
        prefill_next_token_id, prefill_next_token_text = decode_next_token(
            tokenizer, quanted_aligned_logits
        )
        logger.info(
            f"Prefill Quanted Aligned Model next token: {prefill_next_token_id} {prefill_next_token_text}"
        )

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    # 导出Prefill 模型
    if True:
        logger.info("*************** Start exporting prefill model ***************")
        prefill_onnx_file = xhmodel_export_onnx(
            paddleocr_vl_llm_model,
            tokenizer,
            data_prefill,
            str(prefill_onnx_dir),
            f"paddleocr_vl_llm_prefill",
            execution_device,
            dtype,
            logger,
            is_valid_model,
        )
        paddleocr_vl_llm_model.release_exported_model()
        paddleocr_vl_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
        meta_info.prefill_onnx_file = str(
            Path(prefill_onnx_file).relative_to(cfg.work_dir)
        )
        logger.info(f"save prefill onnx model to {prefill_onnx_file}")
        logger.info("*************** Finished export prefill model ***************")

    if True:
        paddleocr_vl_llm_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
        paddleocr_vl_llm_model.set_input_sequence_length(1)
        if is_valid_model:
            data_decode = {
                "input_ids": prefill_next_token_id,
                "past_seq_length": data_prefill["input_ids"].shape[-1],
            }
        else:
            data_decode = {
                "input_ids": torch.randint(0, 1000, (1, 1)),
                "past_seq_length": 256,
            }
        if is_valid_model:
            paddleocr_vl_llm_model.to(execution_device)
            paddleocr_vl_llm_model.to(dtype)
            paddleocr_vl_llm_model.set_exec_device(execution_device)
            if hasattr(paddleocr_vl_llm_model, "kv_cache_to"):
                paddleocr_vl_llm_model.kv_cache_to(execution_device)
            if (
                hasattr(paddleocr_vl_llm_model, "rope_deltas")
                and paddleocr_vl_llm_model.rope_deltas is not None
            ):
                paddleocr_vl_llm_model.rope_deltas = (
                    paddleocr_vl_llm_model.rope_deltas.to(execution_device)
                )
            if isinstance(data_decode.get("input_ids", None), Tensor):
                data_decode["input_ids"] = data_decode["input_ids"].to(execution_device)
            outs = paddleocr_vl_llm_model.test_step(data_decode)
            logits = outs.logits.detach()
            next_token_id, next_token_text = decode_next_token(tokenizer, logits)
            logger.info(
                f"Decode Quanted Model next token: {next_token_id} {next_token_text}"
            )

        torch.cuda.empty_cache()
        logger.info("*************** Start exporting decode model ***************")
        with TimeProfiler("export decode onnx", logger):
            decode_onnx_file = xhmodel_export_onnx(
                paddleocr_vl_llm_model,
                tokenizer,
                data_decode,
                str(decode_onnx_dir),
                "paddleocr_vl_instruct_llm_decode",
                execution_device,
                dtype,
                logger,
                is_valid_model,
            )
        paddleocr_vl_llm_model.release_exported_model()
        meta_info.decode_onnx_file = str(
            Path(decode_onnx_file).relative_to(cfg.work_dir)
        )

        logger.info(f"save decode onnx model to {decode_onnx_file}")
        logger.info("*************** Finished export decode model ***************")

    meta_file = str(Path(cfg.work_dir) / "export_meta_info.json")
    json.dump(meta_info, open(meta_file, "w"), indent=4)
    logger.info(f"Save meta info to {meta_file}")


def main(args):
    cfg = Config.fromfile(args.config)

    cfg_name = Path(args.config).stem
    cfg.work_dir = str(
        Path("./work_dirs")
        / f"paddleocr_vl"
        / f"{cfg_name}_{Path(args.hf_model_dir).stem}"
    )

    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    # cfg.device = "cpu"
    cfg.device = "cuda:0"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.execution_device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # 执行设备，执行某个Module或者op时，再将数据搬到这个设备上

    debug_output_dir = Path(cfg.work_dir) / "debug"
    debug_output_dir.mkdir(exist_ok=True, parents=True)

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    _register_xh_tile_decomposition(logger)

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.config_file = str(config_file)
    cfg.cfg_name = cfg_name
    cfg.dump(config_file)

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息
    with (
        TimeProfiler(f"{cfg_name} export", logger),
        MemoryTracker(0, "export", logger) as memory_tracker,
    ):
        _export_impl(cfg, args)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()

    main(args)
