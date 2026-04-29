from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

from examples_merak.llm.gemma4_moe.gemma4_moe_visual_preprocess import (
    extract_valid_patch_tokens,
    prepare_visual_input_image,
)
from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import (
    Gemma4MoeVisionWrapper,
    XHGemma4MoeVisualProcessor,
)
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_meta(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_meta_path(meta_path: str | Path, referenced_path: str | Path) -> Path:
    resolved_path = Path(referenced_path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(meta_path).resolve().parent / resolved_path


def load_hf_config(llm_meta: dict, llm_meta_path: str | Path) -> dict:
    hf_config = llm_meta.get("hf_config")
    if not hf_config:
        return {}
    hf_config_path = resolve_meta_path(llm_meta_path, hf_config)
    if hf_config_path.is_dir():
        hf_config_path = hf_config_path / "config.json"
    if not hf_config_path.exists():
        return {}
    return load_meta(hf_config_path)


def resolve_llm_runtime_meta(llm_meta_path: str | Path) -> tuple[Path, dict]:
    resolved_meta_path = Path(llm_meta_path).resolve()
    llm_meta = load_meta(resolved_meta_path)
    if llm_meta.get("model_config", {}).get("model_type"):
        return resolved_meta_path, llm_meta

    exported_dir = llm_meta.get("exported_dir")
    if exported_dir:
        golden_meta_path = resolve_meta_path(resolved_meta_path, Path(exported_dir) / "golden_meta_info.json")
        if golden_meta_path.exists():
            return golden_meta_path, load_meta(golden_meta_path)

    raise ValueError(
        "Full inference requires a Merak golden_meta_info.json. "
        "Pass it directly or use an export_meta_info.json that points to exported_dir."
    )


def resolve_image_token_id(image_token_id: int | None, llm_meta: dict, llm_meta_path: str | Path) -> int:
    if image_token_id is not None:
        return image_token_id

    for candidate in (llm_meta, load_hf_config(llm_meta, llm_meta_path)):
        token_id = candidate.get("image_token_id")
        if token_id is not None:
            return int(token_id)

    raise ValueError(
        "Unable to resolve image_token_id. Pass --image-token-id or ensure the exported hf_config contains it."
    )


def build_mm_token_type_ids(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    return (input_ids == image_token_id).to(torch.int32)


def build_vision_bidirectional_mask(
    mm_token_type_ids: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    vision_positions = mm_token_type_ids.bool()
    batch, seq_len = mm_token_type_ids.shape
    mask = torch.zeros((batch, 1, seq_len, seq_len), dtype=dtype, device=mm_token_type_ids.device)
    causal_block = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=mm_token_type_ids.device),
        diagonal=1,
    )
    mask = mask.masked_fill(causal_block.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)
    for b in range(batch):
        idx = torch.where(vision_positions[b])[0]
        if idx.numel() > 0:
            mask[b, 0, idx[:, None], idx[None, :]] = 0
    return mask


def resolve_pad_token_id(tokenizer) -> int:
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        return int(eos_token_id[0])
    if eos_token_id is not None:
        return int(eos_token_id)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    raise ValueError("Unable to resolve pad_token_id from tokenizer.")


def render_chat_prompt(tokenizer, prompt: str, include_image: bool, enable_thinking: bool) -> str:
    content = prompt
    if include_image:
        content = f"\n\n<|image|>\n\n{prompt}"
    messages = [{"role": "user", "content": content}]
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        chat_template_kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    except TypeError:
        chat_template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **chat_template_kwargs)


def expand_image_placeholders(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape (1, seq), but got {tuple(input_ids.shape)}")

    expanded_ids: list[int] = []
    mm_token_type_ids: list[int] = []
    placeholder_count = 0
    for token_id in input_ids[0].tolist():
        if token_id == image_token_id:
            placeholder_count += 1
            expanded_ids.extend([image_token_id] * num_image_tokens)
            mm_token_type_ids.extend([1] * num_image_tokens)
        else:
            expanded_ids.append(token_id)
            mm_token_type_ids.append(0)

    if placeholder_count == 0:
        raise ValueError("The rendered prompt did not contain any <|image|> placeholder token.")

    return (
        torch.tensor([expanded_ids], dtype=torch.long),
        torch.tensor([mm_token_type_ids], dtype=torch.long),
    )


VISION_TOWER_PREFIX = "model.vision_tower."
EMBED_VISION_PREFIX = "model.embed_vision."


def _is_native_vision_weight(weight_name: str) -> bool:
    return weight_name.startswith(VISION_TOWER_PREFIX) or weight_name.startswith(EMBED_VISION_PREFIX)


def resolve_native_vision_weight_map(hf_model_dir: str | Path) -> dict[Path, list[str]]:
    hf_model_dir = Path(hf_model_dir)
    index_path = hf_model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = load_meta(index_path).get("weight_map", {})
        grouped: dict[Path, list[str]] = {}
        for weight_name, shard_name in weight_map.items():
            if not _is_native_vision_weight(weight_name):
                continue
            grouped.setdefault(hf_model_dir / shard_name, []).append(weight_name)
        if grouped:
            return grouped

    for candidate_name in ("model.safetensors", "model-00001-of-00001.safetensors"):
        candidate_path = hf_model_dir / candidate_name
        if not candidate_path.exists():
            continue
        with safe_open(str(candidate_path), framework="pt", device="cpu") as shard:
            weight_names = [weight_name for weight_name in shard.keys() if _is_native_vision_weight(weight_name)]
        if weight_names:
            return {candidate_path: weight_names}

    raise FileNotFoundError(
        f"Unable to locate Gemma4 native vision weights under {hf_model_dir}. "
        "Expected model.safetensors index entries for model.vision_tower/model.embed_vision."
    )


def load_native_vision_wrapper(
    hf_model_dir: str | Path,
    device: str,
    upsample_token: bool,
) -> Gemma4MoeVisionWrapper:
    hf_model_dir = Path(hf_model_dir)
    config = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(config)

    vision_tower = model.model.vision_tower
    embed_vision = model.model.embed_vision
    del model

    vision_tower.to_empty(device=device)
    embed_vision.to_empty(device=device)

    vision_state: dict[str, torch.Tensor] = {}
    embed_state: dict[str, torch.Tensor] = {}
    for shard_path, weight_names in resolve_native_vision_weight_map(hf_model_dir).items():
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for weight_name in weight_names:
                tensor = shard.get_tensor(weight_name)
                if weight_name.startswith(VISION_TOWER_PREFIX):
                    vision_state[weight_name.removeprefix(VISION_TOWER_PREFIX)] = tensor
                else:
                    embed_state[weight_name.removeprefix(EMBED_VISION_PREFIX)] = tensor

    vision_tower.load_state_dict(vision_state, strict=True, assign=True)
    embed_vision.load_state_dict(embed_state, strict=True, assign=True)
    vision_tower.eval()
    embed_vision.eval()
    vision_tower.config._attn_implementation = "eager"
    vision_tower.config.pooling_kernel_size = 3 if upsample_token else 1

    wrapper = Gemma4MoeVisionWrapper(vision_tower, embed_vision, vision_tower.config)
    wrapper.to(device)
    wrapper.eval()
    return wrapper


class VisionEncoderRunner:
    def __init__(self, model_path: Path, device: str):
        self.model_path = model_path
        self.device = device
        self._session = HMONNXModel(str(model_path))
        self._session.to(device)

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self._session(pixel_values.to(device=self.device, dtype=torch.float16))
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]
        return image_embeds


def run_native_vision_encoder(
    *,
    hf_model_dir: str | Path,
    pixel_values: torch.Tensor,
    image_position_ids: torch.Tensor,
    pixel_values_valid: torch.Tensor,
    device: str,
    upsample_token: bool,
) -> torch.Tensor:
    vision_wrapper = load_native_vision_wrapper(hf_model_dir, device, upsample_token)
    pixel_values = pixel_values.to(device)
    image_position_ids = image_position_ids.to(device)
    pixel_values_valid = pixel_values_valid.to(device)
    with torch.no_grad():
        vision_wrapper.precompute_constants(pixel_values, image_position_ids)
        return vision_wrapper(pixel_values_valid)


def resolve_vision_model_path(vision_meta_path: str | Path, vision_meta: dict) -> Path | None:
    vision_hmonnx = vision_meta.get("vision_hmonnx")
    if vision_hmonnx:
        resolved = resolve_meta_path(vision_meta_path, vision_hmonnx)
        if resolved.exists():
            return resolved
    return None


def build_image_embeds(
    *,
    image_path: str | Path,
    llm_runtime_meta: dict,
    vision_meta: dict,
    vision_meta_path: str | Path,
    device: str,
) -> torch.Tensor:
    target_image_size = tuple(vision_meta.get("image_preprocess", {}).get("target_image_size", [448, 448]))
    upsample_token = bool(vision_meta.get("upsample_token", False))
    processed_image, _ = prepare_visual_input_image(
        image_path,
        upsample_token=upsample_token,
        target_image_size=target_image_size,
    )

    hf_model_dir = vision_meta.get("hf_model") or llm_runtime_meta["model_config"]["hf_model"]
    processor = XHGemma4MoeVisualProcessor.from_pretrained(hf_model_dir, upsample_token=upsample_token)
    vision_inputs = processor(images=processed_image, return_tensors="pt")
    pixel_values = vision_inputs["pixel_values"]
    image_position_ids = vision_inputs["image_position_ids"]
    pixel_values_valid, _, valid_mask = extract_valid_patch_tokens(pixel_values, image_position_ids)

    expected_valid = int(torch.tensor(vision_meta["valid_mask"], dtype=torch.bool).sum().item())
    actual_valid = int(valid_mask.sum().item())
    if actual_valid != expected_valid:
        raise ValueError(
            f"Vision valid patch count mismatch: expected {expected_valid}, got {actual_valid}. "
            "Use the same image size and preprocess settings as the exported vision model."
        )

    vision_model_path = resolve_vision_model_path(vision_meta_path, vision_meta)
    if vision_model_path is not None:
        image_embeds = VisionEncoderRunner(vision_model_path, device)(pixel_values_valid)
    else:
        image_embeds = run_native_vision_encoder(
            hf_model_dir=hf_model_dir,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            pixel_values_valid=pixel_values_valid,
            device=device,
            upsample_token=upsample_token,
        )
    if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
        image_embeds = image_embeds[0]
    return image_embeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma4 MoE VLM HMONNX generate with mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default="work_dirs/gemma4_moe_with_mask_26b_a4b_it_xh2a_w8a8_256_2k/export_meta_info.json",
        help="Merak golden_meta_info.json or source export_meta_info.json that points to exported_dir.",
    )
    parser.add_argument(
        "--vision-config",
        type=str,
        default="work_dirs/gemma4_moe_26b_a4b_it_vision_xh2a_no_upsample_token_448x448/export_meta_info.json",
    )
    parser.add_argument("--image", type=str, default="data/images/bee.jpg")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument(
        "--image-token-id",
        type=int,
        default=None,
        help="Override image token id. Default resolves from llm export meta -> hf_config/config.json.",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--fast", action="store_true", help="Run the text HMONNX runtime in fast mode.")
    parser.add_argument("--golden", action="store_true", help="Save HMONNX golden outputs while generating.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation.")
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable thinking mode for chat template.",
    )
    parser.add_argument(
        "--streaming-out",
        dest="streaming_out",
        action="store_true",
        help="Stream output tokens during generation.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()

    runtime_meta_path, llm_runtime_meta = resolve_llm_runtime_meta(args.llm_config)
    runtime_model = AutoLLMHONNXModel.from_pretrained(str(runtime_meta_path))

    if type(runtime_model).__name__ != "XHGemma4MoeWithMaskHMONNXModel":
        raise TypeError(
            f"Expected XHGemma4MoeWithMaskHMONNXModel, got {type(runtime_model).__name__} from {runtime_meta_path}"
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        device = "cpu"

    runtime_model.to(device)
    if args.fast and not args.golden:
        runtime_model.to_fast()
    if args.golden:
        runtime_model.enable_golden = True
        if args.fast:
            logger.warning("Golden generation should run in aligned mode; ignoring --fast.")

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_meta_path(runtime_meta_path, llm_runtime_meta["hf_config"])),
        trust_remote_code=True,
    )
    image_token_id = resolve_image_token_id(args.image_token_id, llm_runtime_meta, runtime_meta_path)

    image_embeds = None
    if args.image:
        vision_meta = load_meta(args.vision_config)
        image_embeds = build_image_embeds(
            image_path=args.image,
            llm_runtime_meta=llm_runtime_meta,
            vision_meta=vision_meta,
            vision_meta_path=args.vision_config,
            device=device,
        )
        n_image_tokens = int(image_embeds.shape[0])
    else:
        n_image_tokens = 0

    prompt_text = render_chat_prompt(
        tokenizer,
        prompt=args.prompt,
        include_image=image_embeds is not None,
        enable_thinking=args.enable_thinking,
    )
    prompt_input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    if image_embeds is not None:
        input_ids, mm_token_type_ids = expand_image_placeholders(prompt_input_ids, image_token_id, n_image_tokens)
    else:
        input_ids = prompt_input_ids
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

    bidirectional_mask = build_vision_bidirectional_mask(mm_token_type_ids)
    if image_embeds is not None and int(mm_token_type_ids.sum().item()) != n_image_tokens:
        raise ValueError(
            f"Expanded image token count mismatch: expected {n_image_tokens}, got {int(mm_token_type_ids.sum().item())}"
        )
    if bidirectional_mask.shape[-1] != input_ids.shape[-1]:
        raise ValueError("Vision bidirectional mask shape does not match input_ids length.")

    if args.smoke_test:
        print(
            "Task10 VLM smoke OK",
            "prompt_tokens=",
            int(input_ids.shape[-1]),
            "image_tokens=",
            int(mm_token_type_ids.sum().item()),
            "llm_sliding=",
            llm_runtime_meta.get("sliding_window_cfg", {}).get("sliding_window"),
        )
        return

    streamer = TextStreamer(tokenizer, skip_prompt=True) if args.streaming_out else None
    pad_token_id = resolve_pad_token_id(tokenizer)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    generation_kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "mm_token_type_ids": mm_token_type_ids.to(device),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": pad_token_id,
    }
    if image_embeds is not None:
        generation_kwargs["image_embeds"] = image_embeds.to(device=device, dtype=runtime_model.dtype)
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    contexts = [
        TimeProfiler("gemma4_moe_vlm_generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(runtime_model),
    ]
    with ContextManagers(contexts):
        generated_ids = runtime_model.generate(**generation_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.to(device), generated_ids, strict=False)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    logger.info(f"{'-' * 20} Gemma4 MoE VLM Output {'-' * 20}")
    logger.info(output_text[0] if output_text else "")


if __name__ == "__main__":
    main()
