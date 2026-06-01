import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
import xhquant.xhonnxruntime.config as xhonnxruntime_config

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import xhquant_llm_init, get_root_logger
from xh_model_zoo.xh_llm.models.glm_ocr import GlmOcrONNXModel, GlmOcrProcessor


def _resolve_path(path_str: str | None, workspace_root: Path) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return str(cwd_path)

    workspace_path = (workspace_root / path).resolve()
    return str(workspace_path)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, default="work_dirs/glm_ocr_llm_xh2a_2k_export",
                        help="LLM export directory containing hf_config/ and token_embedding.pt")
    parser.add_argument("--vision_export_dir", type=str, default="work_dirs/glm_ocr_vision_xh2a_export_hmonnx",
                        help="Vision export directory")
    parser.add_argument("--visual_onnx", type=str, default=None,
                        help="vision onnx path; default: <vision_export_dir>/vision/<vision_dir_name>.onnx")
    parser.add_argument("--prefill_onnx", type=str, default=None,
                        help="prefill onnx path; default: <model_dir>/prefill_onnx/<model_dir_name>_prefill.onnx")
    parser.add_argument("--decode_onnx", type=str, default=None,
                        help="decode onnx path; default: <model_dir>/decode_onnx/<model_dir_name>_decode.onnx")
    parser.add_argument("--image", type=str, default='examples/llm/glm_ocr/data/img3.png')
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--images_json", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--use_fast", default=True, action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--cache_len", type=int, default=2048)
    parser.add_argument("--input_sequence_length", type=int, default=256)
    parser.add_argument("--image_size_w", type=int, default=672)
    parser.add_argument("--image_size_h", type=int, default=672)
    parser.add_argument("--eos_token_id", type=int, nargs="+", default=[151329])
    parser.add_argument(
        "--vision_mode",
        type=str,
        default="unquantized",
        choices=["quantized", "unquantized"],
        help="select vision model mode: quantized hmonnx-converted onnx or original unquantized onnx",
    )
    parser.add_argument(
        "--vision_onnx_path",
        type=str,
        default=None,
        help="optional manual override for vision ONNX path; has highest priority",
    )
    return parser


def _resolve_vision_onnx_path(args, default_visual_onnx, workspace_root: Path) -> tuple[str, str]:
    if args.vision_onnx_path is not None:
        path = _resolve_path(args.vision_onnx_path, workspace_root)
        return path, "manual"

    use_unquantized = args.vision_mode == "unquantized"

    if not use_unquantized:
        path = _resolve_path(default_visual_onnx, workspace_root)
        return path, "quantized"

    # Try to find unquantized onnx in the vision export dir
    candidate_paths = []
    vision_export_dir = Path(args.vision_export_dir)
    candidate_paths.append(vision_export_dir / "onnx" / "visual_1.onnx")

    quantized_path = Path(default_visual_onnx)
    if "vision" in quantized_path.parts:
        idx = quantized_path.parts.index("vision")
        root = Path(*quantized_path.parts[:idx])
        candidate_paths.append(root / "onnx" / "visual_1.onnx")

    for candidate in candidate_paths:
        resolved = _resolve_path(str(candidate), workspace_root)
        if resolved is not None and Path(resolved).exists():
            return resolved, "unquantized"

    raise FileNotFoundError(
        "Cannot find unquantized vision ONNX. Please pass --vision_onnx_path explicitly, "
        "or ensure <vision_export_dir>/onnx/visual_1.onnx exists."
    )


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    workspace_root = Path(__file__).resolve().parents[3]

    model_dir = Path(args.model_dir)
    vision_export_dir = Path(args.vision_export_dir)
    model_dir_name = model_dir.name
    vision_dir_name = vision_export_dir.name

    # Derive default ONNX paths
    default_visual_onnx = args.visual_onnx or str(vision_export_dir / "vision" / f"{vision_dir_name}.onnx")
    prefill_onnx = args.prefill_onnx or str(model_dir / "prefill_onnx" / f"{model_dir_name}_prefill.onnx")
    decode_onnx = args.decode_onnx or str(model_dir / "decode_onnx" / f"{model_dir_name}_decode.onnx")
    hf_model_config_dir = str(model_dir / "hf_config")
    embed_tokens_path = str(model_dir / "token_embedding.pt")

    work_dir = str(Path("./work_dirs") / "glm_ocr_xh2a_hmonnx_demo")
    log_file = Path(work_dir) / "demo_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)
    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xhquant_llm_init(log_file, False)
    logger = get_root_logger()
    logger.info(f"Args: {args}")

    xhquant.utils.suppress_printing.disable_printing = True
    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False

    # Load token embedding
    torch.serialization.add_safe_globals([nn.Embedding])
    token_embedding_path = _resolve_path(embed_tokens_path, workspace_root)
    token_embedding = torch.load(token_embedding_path, weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    hf_model_config_dir = _resolve_path(hf_model_config_dir, workspace_root)
    processor = GlmOcrProcessor.from_pretrained(hf_model_config_dir)

    # Resolve vision onnx path (quantized vs unquantized)
    vision_onnx_path, vision_mode = _resolve_vision_onnx_path(args, default_visual_onnx, workspace_root)
    logger.info(f"Vision mode: {vision_mode}, onnx={vision_onnx_path}")

    # Build model directly (matching qwen2_5_vl pattern: SimpleNamespace + direct construction)
    image_feature_cfg = SimpleNamespace(onnx=vision_onnx_path)
    prefill_cfg = SimpleNamespace(
        onnx=_resolve_path(prefill_onnx, workspace_root),
        input_sequence_length=args.input_sequence_length,
    )
    decode_cfg = SimpleNamespace(onnx=_resolve_path(decode_onnx, workspace_root))
    kv_cache_cfg = SimpleNamespace(
        num_hidden_layers=args.num_hidden_layers,
        shape=[1, args.num_kv_heads, args.cache_len, args.head_dim],
    )

    xh_model: GlmOcrONNXModel = GlmOcrONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        cache_len=args.cache_len,
        image_size_w=args.image_size_w,
        image_size_h=args.image_size_h,
        presence_penalty=args.presence_penalty,
        eos_token_id=args.eos_token_id,
    )
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)
    xh_model.to(exec_device)

    output_path = args.output_path
    if output_path is None:
        output_path = str(Path(work_dir) / "demo_output.txt")
    output_path = _resolve_path(output_path, workspace_root)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    default_prompt = "Text Recognition:"

    def run_single(image_path: str, prompt: str) -> tuple[str, float]:
        image_path = _resolve_path(image_path, workspace_root)
        start = time.perf_counter()
        out = xh_model.chat(
            prompt,
            image_path,
            processor,
            logger,
            use_fast=args.use_fast,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - start
        return out, elapsed

    if args.images_json is not None:
        images_json_path = _resolve_path(args.images_json, workspace_root)
        with open(output_path, "w", encoding="utf-8") as f_txt:
            with open(images_json_path, "r", encoding="utf-8") as f:
                images_json = json.load(f)
            if isinstance(images_json, dict):
                items = list(images_json.items())
            elif isinstance(images_json, list):
                items = [(str(i), item) for i, item in enumerate(images_json)]
            else:
                raise TypeError(f"Unsupported images_json format: {type(images_json)}")

            for i, (image_name, image_info) in enumerate(items, 1):
                image_path = image_info.get("url") or image_info.get("image")
                prompt = image_info.get("prompt", default_prompt)
                if image_path is None:
                    logger.warning(f"skip sample {image_name}: missing 'url'/'image' field")
                    continue

                print(f"Processing {i}: {image_path}", flush=True)
                out, elapsed = run_single(image_path, prompt)
                f_txt.write(f"{image_name}\n{out}\n")
                f_txt.flush()
                print(f"[{i}] {image_name}", flush=True)
                print(f"elapsed: {elapsed:.3f}s", flush=True)
                print(out, flush=True)
    else:
        image_path = args.image if args.image is not None else "examples/llm/glm_ocr/data/img3.png"
        prompt = args.prompt if args.prompt is not None else default_prompt
        out, elapsed = run_single(image_path, prompt)
        with open(output_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(out + "\n")
        print(f"elapsed: {elapsed:.3f}s", flush=True)
        print(out, flush=True)


if __name__ == "__main__":
    main()
