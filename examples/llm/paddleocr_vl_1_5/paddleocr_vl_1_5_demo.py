import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import xhquant.utils.suppress_printing

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xhquant.api import Config, ConfigDict
try:
    from .common import xhquant_llm_init, get_root_logger
except ImportError:
    from common import xhquant_llm_init, get_root_logger  # pyright: ignore[reportMissingImports]
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.paddleocr_vl_1_5 import (
    PaddleOCRVLONNXModel,
    PaddleOCRVLProcessor,
)

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--hf_model_config_dir", type=str)
    parser.add_argument("--visual_onnx_path", type=str)
    parser.add_argument("--prefill_onnx_path", type=str)
    parser.add_argument("--decode_onnx_path", type=str)
    parser.add_argument(
        "--meta_info", type=str, default=None, help="export_meta_info.json path"
    )
    parser.add_argument("--cache_len", type=int, default=8192)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    script_dir = Path(__file__).parent.parent.parent  # 返回到 xh2modelzoo 目录
    default_image_path = str(script_dir / "data" / "images" / "test.png")
    parser.add_argument("--image", type=str, default=default_image_path)
    # parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="ocr",
        choices=["ocr", "table", "chart", "formula", "spotting", "seal"],
    )
    parser.add_argument("--images_json", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true", help="do sample")
    parser.add_argument("--output_path", type=str, default="output.txt")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=None,
        help="Override image processor min_pixels to match ONNX export size",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Override image processor max_pixels to match ONNX export size",
    )
    # parser.add_argument("--max_size_w", type=int, default=224,
    #                     help="Max width for image resize, to match vision export max_size_w")
    # parser.add_argument("--max_size_h", type=int, default=224,
    #                     help="Max height for image resize, to match vision export max_size_h")
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    cfg = Config()
    hf_model_dir = args.hf_model_config_dir
    meta = None
    if args.meta_info is not None:
        with open(args.meta_info, "r") as f:
            meta = json.load(f)
        hf_model_dir = meta.get("hf_model", hf_model_dir)
        num_hidden_layers = meta.get("num_hidden_layers")
        kv_cache_shape = meta.get("kv_cache_shape")
        if kv_cache_shape is not None:
            kv_cache_shape = [int(x) for x in kv_cache_shape]
    else:
        num_hidden_layers = args.num_hidden_layers
        if args.num_key_value_heads is None or args.head_dim is None:
            raise ValueError(
                "Please provide --num_key_value_heads and --head_dim when meta_info is not set"
            )
        kv_cache_shape = [1, args.num_key_value_heads, args.cache_len, args.head_dim]

    # Prefer hf_config co-located with exported ONNX/token_embedding.
    hf_config_dir = None
    candidates = [
        Path(args.hf_model_config_dir) / "hf_config",
        Path(hf_model_dir) / "hf_config",
        Path(args.hf_model_config_dir),
        Path(hf_model_dir),
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            hf_config_dir = candidate
            break
    if hf_config_dir is None:
        hf_config_dir = Path(args.hf_model_config_dir) / "hf_config"

    token_embedding_path = Path(args.hf_model_config_dir) / "token_embedding.pt"
    if not token_embedding_path.exists() and args.meta_info is not None:
        token_embedding_path = Path(args.meta_info).parent / "token_embedding.pt"

    cfg.hf_model_config_dir = str(hf_config_dir)
    cfg.embed_tokens = str(token_embedding_path)

    if num_hidden_layers is None or kv_cache_shape is None:
        raise ValueError(
            "num_hidden_layers/kv_cache_shape is required (use --meta_info or explicit args)"
        )

    cfg.model = ConfigDict()
    cfg.model.type = "PaddleOCRVLONNXModel"
    cfg.model.hf_model_dir = hf_model_dir
    cfg.model.image_feature = ConfigDict()
    cfg.model.image_feature.onnx = args.visual_onnx_path
    cfg.model.prefill = ConfigDict()
    cfg.model.prefill.onnx = args.prefill_onnx_path
    cfg.model.prefill.input_sequence_length = 256
    cfg.model.decode = ConfigDict()
    cfg.model.decode.onnx = args.decode_onnx_path
    cfg.model.kv_cache = ConfigDict()
    cfg.model.kv_cache.num_hidden_layers = num_hidden_layers
    cfg.model.kv_cache.shape = kv_cache_shape
    cfg.model.cache_len = args.cache_len

    cfg.work_dir = str(Path("./work_dirs") / "paddleocr_vl_1_5" / "demo")
    cfg_name = "paddleocr_vl_1_5_demo"
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"

    xhquant_llm_init(log_file, False)
    logger = get_root_logger()
    cfg.dump(Path(cfg.work_dir) / f"{cfg_name}.py")

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息

    exec_device = cfg.exec_device
    hf_model_config_dir = cfg.hf_model_config_dir

    # 消除Embedding的安全检查
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding = torch.load(
        cfg.embed_tokens, weights_only=False, map_location="cpu"
    )
    torch.serialization.clear_safe_globals()

    processor_dir = Path(hf_model_config_dir)
    # If the export bundle lacks dynamic modules, fall back to the original HF model dir.
    if not (processor_dir / "image_processing.py").exists():
        processor_dir = Path(hf_model_dir)
    processor = PaddleOCRVLProcessor.from_pretrained(
        str(processor_dir), trust_remote_code=True
    )

    # Replace HF-bundled image_processor with our modified version (skip patchify)
    from xh_model_zoo.xh_llm.models.paddleocr_vl.image_processing import SiglipImageProcessor

    processor.image_processor = SiglipImageProcessor.from_pretrained(str(processor_dir))

    # Override image processor min/max pixels to match the visual ONNX export size.
    # The visual ONNX has baked-in position embeddings for a specific image size
    # determined by min_pixels/max_pixels at export time. If the processor
    # generates different dimensions, position embeddings will be wrong.
    if args.min_pixels is not None:
        processor.image_processor.min_pixels = args.min_pixels
        logger.info(f"Override image_processor.min_pixels = {args.min_pixels}")
    if args.max_pixels is not None:
        processor.image_processor.max_pixels = args.max_pixels
        logger.info(f"Override image_processor.max_pixels = {args.max_pixels}")
    # Override image processor size to match the visual export max_size_w/h
    # processor.image_processor.size = {"height": args.max_size_h, "width": args.max_size_w}
    logger.info(f"Override image_processor.size = {processor.image_processor.size}")

    xh_model: PaddleOCRVLONNXModel = MODELS.build(cfg.model)
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)
    xh_model.to(exec_device)

    prompt = args.prompt if args.prompt is not None else PROMPTS[args.task]
    # print(f"****************************Using prompt: {prompt}")

    if args.images_json is not None:
        with open(args.output_path, "w") as f_txt:
            with open(args.images_json, "r") as f:
                images_json = json.load(f)
            i = 0
            for image_name, image_info in images_json.items():
                i += 1
                image_path = image_info["url"]
                prompt = image_info["prompt"]
                print(f"Processing {i}: {image_path}")
                out = xh_model.chat(
                    prompt,
                    image_path,
                    processor,
                    logger,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                )
                f_txt.write(f"{image_name}\n{out}\n")
    else:
        xh_model.chat(
            prompt,
            args.image,
            processor,
            logger,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )


if __name__ == "__main__":
    main()
