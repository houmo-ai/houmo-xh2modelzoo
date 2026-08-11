"""Batch Unlimited-OCR inference for OmniDocBench markdown evaluation.

This script only produces prediction markdown files. Run OmniDocBench's
``pdf_validation.py`` in the official OmniDocBench environment to compute the
metrics from these files.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMHONNXModel, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhmodel_merak.xh_llm.models.unlimited_ocr.unlimited_ocr_model import XHUnlimitedOCRModel
from xhquant.api import Config, xhquant_init


DEFAULT_PROMPT = "<image>\\n<|grounding|>Convert the document to markdown. "
FREE_OCR_PROMPT = "<image>\\nFree OCR. "
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def re_match(text: str):
    ref_pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(ref_pattern, text, re.DOTALL)

    det_pattern = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"
    for full_match, label, box in re.findall(det_pattern, text, re.DOTALL):
        matches.append((full_match, label, box))

    matches_image = []
    matches_other = []
    for match in matches:
        if match[1].strip() == "image" or "<|ref|>image<|/ref|>" in match[0]:
            matches_image.append(match[0])
        else:
            matches_other.append(match[0])
    return matches, matches_image, matches_other


def strip_stop_tokens(text: str) -> str:
    for stop in ("<｜end▁of▁sentence｜>", "<|end▁of▁sentence|>", "<|endoftext|>"):
        if text.endswith(stop):
            text = text[: -len(stop)]
    return text.strip()


def postprocess_to_markdown(raw_output: str) -> str:
    output = strip_stop_tokens(raw_output)
    _, image_matches, other_matches = re_match(output)
    for index, image_match in enumerate(image_matches):
        output = output.replace(image_match, f"![](images/{index}.jpg)\n")
    for other_match in other_matches:
        output = output.replace(other_match, "")
    output = output.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    return output.strip() + ("\n" if output.strip() else "")


def read_annotation_images(annotation_path: Path, dataset_dir: Path | None) -> list[Path]:
    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected OmniDocBench annotation list, got {type(data)!r}")

    image_paths: list[Path] = []
    base_dir = dataset_dir or annotation_path.parent
    for record in data:
        rel_path = record.get("page_info", {}).get("image_path")
        if not rel_path:
            continue
        image_paths.append(base_dir / str(rel_path))
    return image_paths


def read_image_list(image_list_path: Path) -> list[Path]:
    images: list[Path] = []
    for line in image_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        images.append(Path(line))
    return images


def collect_images(args: argparse.Namespace) -> list[Path]:
    if args.annotation:
        images = read_annotation_images(Path(args.annotation), Path(args.dataset_dir) if args.dataset_dir else None)
    elif args.image_list:
        images = read_image_list(Path(args.image_list))
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    else:
        raise ValueError("One of --annotation, --image-list, or --image-dir must be provided.")

    if args.limit is not None:
        images = images[: args.limit]
    return images


def build_hf_model(args: argparse.Namespace):
    model = XHUnlimitedOCRModel.get_hf_model(args.hf_model_dir, dtype=torch.bfloat16)
    ensure_generation_support(model, args.hf_model_dir)
    model = model.eval().to(args.device)
    tokenizer = model.get_tokenizer() if hasattr(model, "get_tokenizer") else None
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, trust_remote_code=True)
    return model, tokenizer


def ensure_generation_support(model, hf_model_dir: str) -> None:
    from transformers import GenerationConfig, GenerationMixin

    model_cls = type(model)
    if not issubclass(model_cls, GenerationMixin):
        patched_cls = type(
            f"{model_cls.__name__}WithGeneration",
            (model_cls, GenerationMixin),
            {"__module__": model_cls.__module__},
        )
        model.__class__ = patched_cls
    if getattr(model, "generation_config", None) is None:
        try:
            model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
        except OSError:
            model.generation_config = GenerationConfig.from_model_config(model.config)


def build_wrap_model(args: argparse.Namespace):
    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model = AutoLLMModel.from_pretrained(config=model_cfg)
    model.set_state(LLMModelState.WRAP)
    model.visual.set_state(LLMModelState.WRAP)
    model._set_device(args.device)
    model.visual._set_device(args.device)
    if args.wrap_disable_fast_moe:
        disable_fast_moe(model.wrap_model)
        # ``model.infer`` runs under ``torch.autocast(bfloat16)``; match the wrap
        # weights to bf16 so the reference MoE ``index_add_`` sees consistent
        # scalar types (fast Triton MoE is disabled to avoid grouped-gemm build).
        model._set_dtype(torch.bfloat16)
        model.visual._set_dtype(torch.bfloat16)
    model.prepare_for_inference()
    if model.hf_compatible_model is None:
        hf_native = model.get_compatible_native_model()
        model.hf_compatible_model = type(model).build_hf_compatible_model(hf_native, model)
    hf_model = model.hf_compatible_model.eval().to(args.device)
    tokenizer = hf_model.get_tokenizer() if hasattr(hf_model, "get_tokenizer") else None
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model, trust_remote_code=True)
    return model, hf_model, tokenizer


def disable_fast_moe(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        if type(submodule).__name__ == "MoeBlock":
            original_forward = submodule.forward

            def reference_forward(
                hidden_states,
                routing_weights,
                selected_experts=None,
                *,
                _forward=original_forward,
                _block=submodule,
            ):
                if routing_weights.dim() == 2:
                    routing_weights = routing_weights.view(hidden_states.shape[0], hidden_states.shape[1], -1)
                # ``model.infer`` wraps generation in ``torch.autocast(bfloat16)``,
                # which makes expert matmul outputs bf16 while the residual stream
                # (and ``final_hidden_states``) stays fp32, breaking the reference
                # ``index_add_``. Disable autocast and pin every MoE input to the
                # expert weight dtype so all ops share one scalar type.
                weight_dtype = _block.expert_gate_proj_weight.dtype
                with torch.autocast(device_type="cuda", enabled=False):
                    out = _forward(
                        hidden_states.to(weight_dtype),
                        routing_weights.to(weight_dtype),
                        selected_experts=selected_experts,
                        fast_mode=False,
                    )
                return out.to(hidden_states.dtype)

            submodule.forward = reference_forward


def infer_with_native_entry(model, tokenizer, image_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_omnidocbench_") as temp_dir:
        raw_output = model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(image_path),
            output_path=temp_dir,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=False,
            save_results=False,
            eval_mode=True,
            max_length=args.max_length,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            ngram_window=args.ngram_window,
            temperature=args.temperature,
        )
        raw = raw_output or ""
        return postprocess_to_markdown(raw), raw


def build_hmonnx_infer(args: argparse.Namespace) -> Callable[[Path], str]:
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.hmonnx_config)
    hmonnx_model = hmonnx_model.to(args.device)
    if args.fast:
        hmonnx_model.to_fast()
    if args.auto_offload:
        hmonnx_model.enable_auto_offload = True

    processor = hmonnx_model.get_tf_processor()
    tokenizer = processor.tokenizer

    def infer(image_path: Path) -> str:
        model_inputs = processor.process(args.prompt, str(image_path), device=args.device)
        with torch.no_grad(), LLMInferenceContextManager(hmonnx_model):
            generated_ids = hmonnx_model.generate(
                input_ids=model_inputs["input_ids"],
                images_ori=model_inputs["images_ori"],
                images_crop=model_inputs.get("images_crop"),
                images_seq_mask=model_inputs["images_seq_mask"],
                images_spatial_crop=model_inputs["images_spatial_crop"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids, strict=False)
        ]
        raw_output = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return postprocess_to_markdown(raw_output), raw_output

    return infer


def build_infer_fn(args: argparse.Namespace) -> Callable[[Path], tuple[str, str]]:
    if args.mode == "hmonnx":
        return build_hmonnx_infer(args)

    if args.mode == "hf":
        model, tokenizer = build_hf_model(args)
    elif args.mode == "wrap":
        xh_model, model, tokenizer = build_wrap_model(args)

        def infer(image_path: Path) -> tuple[str, str]:
            with torch.no_grad(), LLMInferenceContextManager(xh_model, devices=[args.device]):
                return infer_with_native_entry(model, tokenizer, image_path, args)

        return infer
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    def infer(image_path: Path) -> tuple[str, str]:
        with torch.no_grad():
            return infer_with_native_entry(model, tokenizer, image_path, args)

    return infer


def write_failure(failures_path: Path, image_path: Path, exc: BaseException) -> None:
    with failures_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {"image": str(image_path), "error": repr(exc), "traceback": traceback.format_exc()},
                ensure_ascii=False,
            )
            + "\n"
        )


def output_path_for(output_dir: Path, image_path: Path) -> Path:
    return output_dir / f"{image_path.stem}.md"


def run(args: argparse.Namespace) -> None:
    xhquant_init(None, args.debug)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "failures.jsonl"
    if failures_path.exists() and not args.resume:
        failures_path.unlink()

    images = collect_images(args)
    if not images:
        raise ValueError("No images found for OmniDocBench inference.")
    infer = build_infer_fn(args)

    manifest = []
    iterator: Iterable[Path] = images if args.no_tqdm else tqdm(images, desc=f"Unlimited-OCR {args.mode}")
    for image_path in iterator:
        image_path = image_path.resolve()
        md_path = output_path_for(output_dir, image_path)
        if args.resume and md_path.exists():
            manifest.append({"image": str(image_path), "markdown": str(md_path), "status": "skipped"})
            continue
        start = time.perf_counter()
        try:
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            markdown, raw = infer(image_path)
            md_path.write_text(markdown, encoding="utf-8")
            if args.save_raw:
                (output_dir / f"{image_path.stem}.raw.txt").write_text(raw, encoding="utf-8")
            elapsed = time.perf_counter() - start
            manifest.append(
                {
                    "image": str(image_path),
                    "markdown": str(md_path),
                    "status": "ok",
                    "time": elapsed,
                    "raw_len": len(raw.strip()),
                    "md_len": len(markdown.strip()),
                }
            )
        except Exception as exc:
            md_path.write_text("", encoding="utf-8")
            write_failure(failures_path, image_path, exc)
            elapsed = time.perf_counter() - start
            manifest.append({"image": str(image_path), "markdown": str(md_path), "status": "failed", "time": elapsed})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok_count = sum(1 for item in manifest if item["status"] in {"ok", "skipped"})
    failed_count = sum(1 for item in manifest if item["status"] == "failed")
    print(f"Wrote {ok_count} markdown file(s) to {output_dir}; failed={failed_count}; manifest={manifest_path}")
    if failed_count:
        print(f"Failures saved to {failures_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Unlimited-OCR over OmniDocBench images and write one markdown file per page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["hf", "wrap", "hmonnx"], required=True)
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--image-list", type=str, default="")
    parser.add_argument("--annotation", type=str, default="", help="Path to OmniDocBench.json")
    parser.add_argument("--dataset-dir", type=str, default="", help="Base directory containing annotation-relative images")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw model output as <stem>.raw.txt for diagnostics")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--config", type=str, default="configs_merak/xh2a/llm_models/unlimited_ocr/_unlimited_ocr_xh2a_calib.py")
    parser.add_argument("--hf-model-dir", type=str, default="data/models/Unlimited-OCR")
    parser.add_argument(
        "--hmonnx-config",
        type=str,
        default="work_dirs/_unlimited_ocr_xh2a_calib_debug/hmquant_xh2_unlimited_ocr_calib_w8a8_256_32k_20260708/golden_meta_info.json",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["markdown", "free-ocr"],
        default="markdown",
        help="Preset prompt to use when --prompt is not provided.",
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=32768, help="HF/wrap generation max_length")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="HMONNX generation max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--ngram-window", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="Use fast HMONNX kernels where available")
    parser.add_argument("--auto-offload", action="store_true")
    parser.add_argument(
        "--wrap-disable-fast-moe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Triton fast MoE in wrap mode; slower but avoids local Triton compile issues.",
    )
    args = parser.parse_args()
    if not args.prompt:
        args.prompt = DEFAULT_PROMPT if args.prompt_mode == "markdown" else FREE_OCR_PROMPT
    return args


if __name__ == "__main__":
    run(parse_args())