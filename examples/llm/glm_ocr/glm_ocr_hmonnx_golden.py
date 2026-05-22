from pathlib import Path
import json
import numpy as np
import sys
import torch
import torch.nn as nn
import xhquant.utils.suppress_printing
from PIL import Image, ImageOps
from transformers import PPDocLayoutV3ImageProcessor
from xhquant.api import HMONNXGoldenInference

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import decode_next_token, xhquant_llm_init, get_root_logger
from xh_model_zoo.xh_llm.models.glm_ocr import GlmOcrONNXModel, GlmOcrProcessor
from xh_model_zoo.xh_llm.models.glm_ocr.utils import build_inputs, build_messages


def load_and_process_image(image_path: str, target_w: int, target_h: int) -> Image.Image:
    """Resize and pad image to target dimensions to match exported model's static input shape."""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if (orig_w, orig_h) != (target_w, target_h):
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(114, 114, 114))
    return image


def _load_layout_image(image_path: str, pdf_page: int) -> Image.Image:
    source = Path(image_path)
    if source.suffix.lower() == ".pdf":
        import fitz

        doc = fitz.open(str(source))
        if len(doc) == 0:
            raise ValueError(f"PDF has no pages: {source}")
        page_index = max(0, min(int(pdf_page) - 1, len(doc) - 1))
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return Image.open(source).convert("RGB")


def _generate_layout_golden(args, golden_output_dir: Path, workspace_root: Path, exec_device: str, logger) -> dict:
    layout_hmonnx = Path(_resolve_path(args.layout_hmonnx, workspace_root))
    if not layout_hmonnx.exists():
        raise FileNotFoundError(f"PP-DocLayoutV3 HMONNX not found: {layout_hmonnx}")

    layout_model_dir = _resolve_path(args.layout_model_dir, workspace_root)
    layout_image_path = _resolve_path(args.layout_image, workspace_root)
    layout_golden_dir = golden_output_dir / "layout"
    layout_golden_dir.mkdir(exist_ok=True, parents=True)

    processor = PPDocLayoutV3ImageProcessor.from_pretrained(layout_model_dir)
    image = _load_layout_image(layout_image_path, args.layout_pdf_page)
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    session = HMONNXGoldenInference(str(layout_hmonnx))
    session.to("cpu")
    session.exec_device = exec_device
    session.save_golden = True
    session.golden_dir = str(layout_golden_dir)
    session.step = 0

    with torch.no_grad():
        outputs = session(pixel_values.half().to(exec_device))

    if not isinstance(outputs, (tuple, list)):
        outputs = [outputs]

    np.save(layout_golden_dir / "pixel_values.npy", pixel_values.detach().cpu().numpy())
    output_shapes = [list(output.shape) for output in outputs if isinstance(output, torch.Tensor)]
    summary = {
        "layout_model_dir": layout_model_dir,
        "layout_hmonnx": str(layout_hmonnx),
        "layout_image": layout_image_path,
        "layout_pdf_page": args.layout_pdf_page,
        "layout_golden_dir": str(layout_golden_dir),
        "input_shape": list(pixel_values.shape),
        "output_shapes": output_shapes,
    }
    with (layout_golden_dir / "layout_golden_summary.json").open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    logger.info(f"PP-DocLayoutV3 golden generated at: {layout_golden_dir}")
    logger.info("PP-DocLayoutV3 output shapes: " + json.dumps(output_shapes))
    return summary


def _resolve_path(path_str: str, workspace_root: Path) -> str:
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
    parser.add_argument("--image", type=str, default="examples/llm/glm_ocr/data/img3.png")
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--resume", action="store_true", help="resume export golden")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_decode_steps", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--cache_len", type=int, default=2048)
    parser.add_argument("--input_sequence_length", type=int, default=256)
    parser.add_argument("--image_size_w", type=int, default=672)
    parser.add_argument("--image_size_h", type=int, default=672)
    parser.add_argument("--eos_token_id", type=int, nargs="+", default=[151329])
    parser.add_argument("--skip_layout_golden", action="store_true", help="skip PP-DocLayoutV3 HMONNX golden generation")
    parser.add_argument("--layout_model_dir", type=str, default="/data01/datasets/ppdoclayoutv3_safetensors",
                        help="PP-DocLayoutV3 safetensors directory")
    parser.add_argument("--layout_hmonnx", type=str,
                        default="work_dirs/ppdoclayoutv3_xh2a_export_hmonnx/hmonnx/ppdoclayoutv3_w16a16_sefp_XH2a.onnx",
                        help="PP-DocLayoutV3 HMONNX file")
    parser.add_argument("--layout_image", type=str, default="examples/llm/glm_ocr/data/18UF.pdf",
                        help="image or PDF used for PP-DocLayoutV3 golden input")
    parser.add_argument("--layout_pdf_page", type=int, default=1, help="1-based PDF page for PP-DocLayoutV3 golden input")
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    workspace_root = Path(__file__).resolve().parents[3]

    model_dir = Path(args.model_dir)
    vision_export_dir = Path(args.vision_export_dir)
    model_dir_name = model_dir.name
    vision_dir_name = vision_export_dir.name

    # Derive default ONNX paths (matching qwen2_5_vl inline config pattern)
    visual_onnx = args.visual_onnx or str(vision_export_dir / "vision" / f"{vision_dir_name}.onnx")
    prefill_onnx = args.prefill_onnx or str(model_dir / "prefill_onnx" / f"{model_dir_name}_prefill.onnx")
    decode_onnx = args.decode_onnx or str(model_dir / "decode_onnx" / f"{model_dir_name}_decode.onnx")
    hf_model_config_dir = str(model_dir / "hf_config")
    embed_tokens_path = str(model_dir / "token_embedding.pt")

    work_dir = str(Path("./work_dirs") / "glm_ocr_xh2a_hmonnx_golden")
    log_file = Path(work_dir) / "golden_debug.log"
    Path(work_dir).mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    xhquant_llm_init(log_file, False)
    logger = get_root_logger()
    logger.info(f"Args: {args}")

    golden_output_dir = Path(work_dir) / "golden"
    golden_output_dir.mkdir(exist_ok=True, parents=True)

    vision_golden_dir = golden_output_dir / "vision"
    prefill_golden_dir = golden_output_dir / "prefill"
    decode_golden_dir = golden_output_dir / "decode"

    layout_summary = None
    if not args.skip_layout_golden:
        layout_summary = _generate_layout_golden(args, golden_output_dir, workspace_root, exec_device, logger)

    if not args.resume and prefill_golden_dir.exists():
        logger.error(f"{prefill_golden_dir} already exists, please remove it first")
        exit(1)
    if not args.resume and decode_golden_dir.exists():
        logger.error(f"{decode_golden_dir} already exists, please remove it first")
        exit(1)

    exec_device = exec_device
    device = device
    dtype = getattr(torch, "float16")

    hf_model_config_dir = _resolve_path(hf_model_config_dir, workspace_root)
    processor = GlmOcrProcessor.from_pretrained(hf_model_config_dir)
    tokenizer = processor.tokenizer

    torch.serialization.add_safe_globals([nn.Embedding])
    embed_tokens_path = _resolve_path(embed_tokens_path, workspace_root)
    token_embedding = torch.load(embed_tokens_path, weights_only=False, map_location="cpu")
    torch.serialization.clear_safe_globals()

    # Build model directly (matching qwen2_5_vl pattern: SimpleNamespace + direct construction)
    image_feature_cfg = {"onnx": _resolve_path(visual_onnx, workspace_root)}
    prefill_cfg = {
        "onnx": _resolve_path(prefill_onnx, workspace_root),
        "input_sequence_length": args.input_sequence_length,
    }
    decode_cfg = {"onnx": _resolve_path(decode_onnx, workspace_root)}
    kv_cache_cfg = {
        "num_hidden_layers": args.num_hidden_layers,
        "shape": [1, args.num_kv_heads, args.cache_len, args.head_dim],
    }

    xh_model: GlmOcrONNXModel = GlmOcrONNXModel(
        image_feature=image_feature_cfg,
        prefill=prefill_cfg,
        decode=decode_cfg,
        kv_cache=kv_cache_cfg,
        cache_len=args.cache_len,
        image_size_w=args.image_size_w,
        image_size_h=args.image_size_h,
        eos_token_id=args.eos_token_id,
    )
    xh_model.set_input_embeddings(token_embedding)
    xh_model.set_exec_device(exec_device)

    prompt = args.prompt
    image_path = args.image
    image_path = _resolve_path(image_path, workspace_root)
    max_decode_steps = args.max_decode_steps

    # Resize image to match the static input shape of the exported HMONNX vision model
    target_w = args.image_size_w
    target_h = args.image_size_h
    image = load_and_process_image(image_path, target_w=target_w, target_h=target_h)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = build_inputs(processor, messages, device=torch.device(device))

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]

    if args.batch_size > 1:
        logger.warning("GLM OCR vision currently uses flattened patch tokens; batch_size>1 may be unsupported.")

    # ---- Vision ----
    xh_model.init_image_feature()
    xh_model.to(torch.device(device))
    xh_model.set_exec_device(exec_device)

    vision_golden_dir.mkdir(exist_ok=True, parents=True)
    xh_model.save_image_feature_golden(str(vision_golden_dir))
    image_embeds_npy = vision_golden_dir / "image_embeds.npy"
    if image_embeds_npy.exists() and args.resume:
        image_embeds = torch.from_numpy(np.load(str(image_embeds_npy))).to(torch.device(device))
    else:
        image_embeds = xh_model.extract_image_features(pixel_values, image_grid_thw)
        np.save(str(image_embeds_npy), image_embeds.detach().cpu().numpy())
    np.save(vision_golden_dir / "pixel_values.npy", pixel_values.detach().cpu().numpy())
    np.save(vision_golden_dir / "image_grid_thw.npy", image_grid_thw.detach().cpu().numpy())
    xh_model.release_image_feature()

    input_ids = input_ids.to(torch.device(device))
    image_embeds = image_embeds.to(torch.device(device), dtype=torch.float16)

    # ---- Prefill ----
    xh_model.init_prefill()
    xh_model.to(torch.device(device))
    xh_model.set_exec_device(exec_device)

    xh_model.save_prefill_golden(str(prefill_golden_dir))
    prefill_golden_dir.mkdir(exist_ok=True, parents=True)

    data_prefill = {
        "input_ids": input_ids,
        "image_embeds": image_embeds,
        "past_seq_length": 0,
        "image_grid_thw": image_grid_thw,
    }
    prefill_logits = xh_model.prefill(data_prefill, save_golden=True)

    next_token_id, next_token_text = decode_next_token(tokenizer, prefill_logits)
    logger.info(f"Prefill next token: {next_token_id} {next_token_text}")

    np.save(prefill_golden_dir / "logits.npy", prefill_logits.detach().cpu().numpy())
    np.save(prefill_golden_dir / "input_ids.npy", input_ids.detach().cpu().numpy())

    xh_model.release_prefill_session()

    # ---- Decode ----
    xh_model.init_decode()
    xh_model.to(torch.device(device))
    xh_model.set_exec_device(exec_device)

    decode_records = []
    current_length = input_ids.shape[-1]

    for step in range(max_decode_steps):
        step_decode_golden_dir = decode_golden_dir / f"decode_{step}"
        step_decode_golden_dir.mkdir(exist_ok=True, parents=True)
        xh_model.save_decode_golden(str(step_decode_golden_dir))

        data_decode = {
            "input_ids": next_token_id,
            "past_seq_length": current_length + step,
        }
        decode_logits = xh_model.decode(data_decode)

        pred_token_id, pred_token_text = decode_next_token(tokenizer, decode_logits)
        logger.info(f"Decode step {step} next token: {pred_token_id} {pred_token_text}")

        np.save(step_decode_golden_dir / "logits.npy", decode_logits.detach().cpu().numpy())

        try:
            pred_token_scalar = int(pred_token_id.view(-1)[0].item())
        except Exception:
            pred_token_scalar = int(torch.argmax(decode_logits[:, -1, :], dim=-1)[0].item())

        decode_records.append(
            {
                "step": step,
                "pred_token_id": pred_token_scalar,
                "pred_token_text": tokenizer.decode([pred_token_scalar], skip_special_tokens=False),
            }
        )

        if pred_token_scalar in xh_model.eos_token_id:
            break

        next_token_id = torch.tensor([[pred_token_scalar]], dtype=torch.long, device=torch.device(device))

    xh_model.release_decode_session()

    summary = {
        "model_dir": str(model_dir),
        "vision_export_dir": str(vision_export_dir),
        "image": image_path,
        "prompt": prompt,
        "vision_output": str(image_embeds_npy),
        "layout": layout_summary,
        "prefill_next_token_id": int(next_token_id.view(-1)[0].item()),
        "prefill_next_token_text": tokenizer.decode([int(next_token_id.view(-1)[0].item())], skip_special_tokens=False),
        "decode": decode_records,
    }
    with (golden_output_dir / "golden_summary.json").open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)

    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
