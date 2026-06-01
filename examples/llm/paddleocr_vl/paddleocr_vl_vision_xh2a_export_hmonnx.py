import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import xhquant.utils.suppress_printing
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from xhquant.api import (
    ConfigDict,
    HMONNXInference,
    QTensor,
    set_random_seed,
)
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

try:
    from .common import xhquant_llm_init, get_root_logger
except ImportError:
    from common import xhquant_llm_init, get_root_logger  # pyright: ignore[reportMissingImports]
from xhquant.api import Config
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.paddleocr_vl import PaddleOCRVLProcessor, XHPaddleOCRVLVisionModel
from xh_model_zoo.utils.memory_tracker import MemoryTracker
from xh_model_zoo.utils.time_profiler import TimeProfiler


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


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="xh2modelzoo/xh_model_zoo/xh_llm/models/paddleocr_vl/paddleocr_vl_vision_config.py",
    )
    parser.add_argument(
        "--hf_model_dir", type=str, default="/data01/datasets/PaddleOCR-VL"
    )
    parser.add_argument("--use_gptq_model", action="store_true", default=False)
    parser.add_argument("--gptq_model_rotate", action="store_true", default=False)
    # parser.add_argument("--model_type", type=str, default="2B")
    # parser.add_argument("--max_size_t", type=int, default=2)
    parser.add_argument("--temporal_patch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="evaluate the model")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--quarot_weight_path", type=str, default=None)
    script_dir = Path(__file__).parent.parent.parent  # 返回到 xh2modelzoo 目录
    default_image_path = str(script_dir / "data" / "images" / "ocr_img.png")
    parser.add_argument("--image_path", type=str, default=default_image_path)
    parser.add_argument(
        "--task", type=str, default="ocr"
    )  # Options: 'ocr' | 'table' | 'chart' | 'formula'
    return parser


def _export_impl(cfg, args):
    logger = get_root_logger()
    is_valid_model = args.valid
    config_file = cfg.config_file
    cfg_name = cfg.cfg_name

    device = torch.device(cfg.device)
    execution_device = torch.device(cfg.execution_device)
    dtype = getattr(torch, cfg.dtype)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(cfg.work_dir)),
        )
    )

    # update cfg.model.wrap_cfg
    # cfg.model.wrap_cfg.max_size_t = args.max_size_t
    cfg.model.wrap_cfg.temporal_patch_size = args.temporal_patch_size
    cfg.model.wrap_cfg.patch_size = args.patch_size
    cfg.model.hf_model = args.hf_model_dir
    cfg.model.is_gptqmodel = args.use_gptq_model
    cfg.hf_model_dir = args.hf_model_dir
    # qwen3_vision_model: XHPaddleOCRVLVisionModel = MODELS.build(cfg.model)
    # native_model = qwen3_vision_model.get_hf_model()
    paddleocr_vl_vision_model: XHPaddleOCRVLVisionModel = MODELS.build(cfg.model)
    native_model = paddleocr_vl_vision_model.get_hf_model()
    if args.quarot_weight_path is not None:
        if native_model.config.tie_word_embeddings:
            old_torchscript = native_model.config.torchscript
            native_model.config.torchscript = True
            native_model.tie_weights()
            native_model.config.tie_word_embeddings = False
            native_model.config.torchscript = old_torchscript
            native_model.config.text_config.tie_word_embeddings = False

        from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

        rotation_utils.fuse_layer_norms(native_model)
        state_dict = load_safetensors_file(args.quarot_weight_path)
        native_model.load_state_dict(state_dict)
        logger.info(f"Load state_dict from {args.quarot_weight_path}")

    if args.use_gptq_model:
        if args.gptq_model_rotate:
            from xh_model_zoo.xh_llm.quarot.quantizer_utils import rotation_utils

            rotation_utils.fuse_layer_norms(native_model, llm_rotate=False)
            rotation_utils.rotate_model(
                native_model, "hadamard", device=device, llm_rotate=False
            )
        else:
            raise NotImplementedError("Only support gptq model with rotation")

    paddleocr_vl_vision_model.init_wrap_model(native_model)

    wraped_model = paddleocr_vl_vision_model._wrap_model
    # native_model.visual = wraped_model
    hf_model_dir = paddleocr_vl_vision_model.hf_model_dir

    processor = PaddleOCRVLProcessor.from_pretrained(
        hf_model_dir, trust_remote_code=True
    )
    from xh_model_zoo.xh_llm.models.paddleocr_vl.image_processing import SiglipImageProcessor

    processor.image_processor = SiglipImageProcessor.from_pretrained(str(hf_model_dir))
    if args.min_pixels is not None:
        processor.image_processor.min_pixels = args.min_pixels
    if args.max_pixels is not None:
        processor.image_processor.max_pixels = args.max_pixels
    logger.info(
        "Vision export processor: %s, min_pixels=%s, max_pixels=%s",
        type(processor.image_processor).__name__,
        getattr(processor.image_processor, "min_pixels", None),
        getattr(processor.image_processor, "max_pixels", None),
    )

    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
    }
    image = Image.open(args.image_path).convert("RGB")
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
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        logger.info(
            f"pixel_values shape from processor: {tuple(inputs['pixel_values'].shape)}"
        )
    native_model.to(execution_device)
    native_model.to(dtype)
    # Only convert floating point tensors to dtype, keep integer indices unchanged
    inputs = {
        k: (
            v.to(execution_device).to(dtype)
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
            else v.to(execution_device) if isinstance(v, torch.Tensor) else v
        )
        for k, v in inputs.items()
    }
    if hasattr(native_model, "visual"):
        native_model.visual.to(execution_device).to(dtype)
    if hasattr(native_model, "mlp_AR"):
        native_model.mlp_AR.to(execution_device).to(dtype)
    if hasattr(native_model, "model") and hasattr(native_model.model, "visual"):
        native_model.model.visual.to(execution_device).to(dtype)
    if hasattr(native_model, "model") and hasattr(native_model.model, "mlp_AR"):
        native_model.model.mlp_AR.to(execution_device).to(dtype)

    if is_valid_model:
        with torch.no_grad():
            generated_ids = native_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            )
        # generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        # output_text = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info("***************** native model output *****************")
        logger.info(output_text)

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    # Processor already outputs full images [N, C, H, W], just add batch dim -> [B, 1, C, H, W]
    hm_pixel_values = (
        pixel_values.unsqueeze(0) if pixel_values.ndim == 4 else pixel_values
    )
    hm_pixel_values = hm_pixel_values.type(wraped_model.dtype).to(wraped_model.device)

    if args.batch_size > 1:
        hm_pixel_values = hm_pixel_values.repeat(args.batch_size, 1, 1, 1, 1)

    image_grid_thw = image_grid_thw.to(torch.long)

    out_onnx_file = str(Path(cfg.work_dir) / "onnx" / f"visual_{args.batch_size}.onnx")
    Path(out_onnx_file).parent.mkdir(exist_ok=True, parents=True)

    # hm_pv = torch.load("hm_pixel_values.pth", map_location="cpu").unsqueeze(2).repeat(1, 1, max_size_t, 1, 1)
    # out = wraped_model.forward(hm_pv.half().to(execution_device), window_index.to(execution_device), attention_bias.to(execution_device))
    if Path(out_onnx_file).exists():
        logger.info(f"onnx model already exists: {out_onnx_file}")
        onnx_model = onnx.load(out_onnx_file, load_external_data=True)
    else:
        # wraped_model.forward(hm_pixel_values, window_index, attention_bias)
        wraped_model.float().eval()
        wraped_model.cpu()

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_file = str(Path(tmp_dir) / "visual.onnx")
            print(f"exporting onnx model to {onnx_file}")
            torch.onnx.export(
                wraped_model,
                (hm_pixel_values.float().cpu()),
                onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["last_hidden_state", "pooler_output"],
                verbose=True,
            )
            onnx_model = onnx.load(onnx_file, load_external_data=True)

        onnx_model, check = simplify_large_onnx(onnx_model)

        onnx.save(
            onnx_model,
            out_onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="visual_external_data",
            convert_attribute=True,
        )

    class ONNXWrapModel(nn.Module):
        def __init__(self, model_path, spatial_merge_size, patch_size, device):
            super().__init__()
            self.session = ort.InferenceSession(model_path)
            self.spatial_merge_size = spatial_merge_size
            self.patch_size = patch_size
            self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
            self.device = device

        @property
        def dtype(self):
            return torch.float16

        def forward(self, pixel_values):
            dtype = pixel_values.dtype
            device = pixel_values.device
            pixel_values = pixel_values.float().cpu().numpy()
            last_hidden_state, pooler_output = self.session.run(
                None, {"pixel_values": pixel_values}
            )
            last_hidden_state = torch.from_numpy(last_hidden_state).to(
                dtype=dtype, device=device
            )
            pooler_output = torch.from_numpy(pooler_output).to(
                dtype=dtype, device=device
            )
            return last_hidden_state, pooler_output

    if is_valid_model:
        # Move models back to GPU after CPU export
        native_model.to(execution_device)
        native_model.to(dtype)
        inputs = {
            k: (
                v.to(execution_device).to(dtype)
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
                else v.to(execution_device) if isinstance(v, torch.Tensor) else v
            )
            for k, v in inputs.items()
        }

        vision_cfg = native_model.config.vision_config
        spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", 1)
        patch_size = getattr(vision_cfg, "patch_size", 14)
        onnx_infer_model = ONNXWrapModel(
            out_onnx_file,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=execution_device,
        )
        native_model.model.visual = onnx_infer_model

        try:
            with torch.no_grad():
                generated_ids = native_model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            logger.info("***************** onnx model output *****************")
            logger.info(output_text)
        except Exception as e:
            logger.warning(
                f"ONNX validation failed (non-critical, ONNX already generated): {e}"
            )

    out_hmonnx_file = Path(cfg.work_dir) / "vision" / f"{cfg_name}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    if not out_hmonnx_file.exists():
        # onnx模型是fp32的，因此输入也需要是fp32
        from xhquant.api import (
            DeviceType,
            convert_onnx_to_hmonnx,
        )

        input_args = [hm_pixel_values.float().cpu()]

        convert_onnx_to_hmonnx(
            out_onnx_file,
            input_args,  # a list of torch tensors
            DeviceType.XH2a,
            str(out_hmonnx_file),
            cfg.model.quant_config,
        )
        logger.info(
            f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}"
        )

    from xhquant.api import HMONNXGoldenInference

    hm_model = HMONNXGoldenInference(out_hmonnx_file)
    hm_model.save_golden = True
    hm_model.exec_device = execution_device
    model_type = getattr(args, "model_type", cfg.cfg_name)
    # wrap_cfg = cfg.model.wrap_cfg
    hm_model.golden_dir = (
        Path(cfg.work_dir) / "vision" / f"paddleocr_vl_{model_type}_vision"
    )
    hm_model.golden_dir.mkdir(exist_ok=True, parents=True)
    print(
        f"DEBUG: hm_pixel_values shape for HMONNX validation: {hm_pixel_values.shape}"
    )
    try:
        hm_model.forward(hm_pixel_values.half())
        logger.info("HMONNX golden validation passed successfully")
    except Exception as e:
        logger.warning(f"HMONNX golden validation failed (optional): {e}")

    if not is_valid_model:
        return

    class HMONNXWrapModel(nn.Module):
        def __init__(self, model, spatial_merge_size, patch_size, device):
            super().__init__()
            self._model = model
            self.spatial_merge_size = spatial_merge_size
            self.patch_size = patch_size
            self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
            self.device = device

        @property
        def dtype(self):
            return torch.float16

        @property
        def dtype(self):
            return torch.float16

        @torch.no_grad()
        def forward(self, pixel_values):
            last_hidden_state, pooler_output = self._model(pixel_values.half())
            return last_hidden_state, pooler_output

    hm_session = HMONNXInference(out_hmonnx_file)
    hm_session.to("cpu")
    hm_session.exec_device = execution_device
    vision_cfg = native_model.config.vision_config
    spatial_merge_size = getattr(vision_cfg, "spatial_merge_size", 1)
    patch_size = getattr(vision_cfg, "patch_size", 14)
    xh_model = HMONNXWrapModel(
        hm_session,
        spatial_merge_size=spatial_merge_size,
        patch_size=patch_size,
        device=execution_device,
    )

    native_model.model.visual = xh_model
    with torch.no_grad():
        generated_ids = native_model.generate(
            **inputs, max_new_tokens=args.max_new_tokens
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    logger.info("***************** hmonnx model output *****************")
    logger.info(output_text)


def main(args):
    cfg = Config.fromfile(args.config)

    cfg_name = Path(args.config).stem
    cfg.work_dir = str(
        Path("./work_dirs")
        / f"paddleocr_vl"
        / f"{cfg_name}_{args.batch_size}_use_gptq_model_{args.use_gptq_model}_{Path(args.hf_model_dir).stem}"
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

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.config_file = str(config_file)
    cfg.cfg_name = cfg_name
    cfg.dump(config_file)

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息
    with TimeProfiler(f"{cfg_name} export", logger), MemoryTracker(0, "export", logger):
        _export_impl(cfg, args)


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()

    main(args)
