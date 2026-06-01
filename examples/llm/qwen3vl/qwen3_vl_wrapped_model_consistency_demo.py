import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from torchvision.io import write_png
from transformers.masking_utils import create_causal_mask
from transformers.video_utils import VideoMetadata
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration

from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xh_model_zoo.xh_llm.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from xh_model_zoo.xh_llm.models.qwen3_vl.data_preprocess import Qwen3_VLDataPreprocess
from xh_model_zoo.xh_llm.models.qwen3_vl._llm_model_impl import register_wrap_cls as llm_register_wrap_cls
from xh_model_zoo.xh_llm.models.qwen3_vl._vision_model_impl import register_wrap_cls as vision_register_wrap_cls
from xh_model_zoo.xh_llm.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--image-path", type=str, default="data/test/image-2025-09-15-14-09-19-086.png")
    parser.add_argument("--video-path", type=str, default="data/test/test.mp4")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--image-size-h", type=int, default=448)
    parser.add_argument("--image-size-w", type=int, default=448)
    parser.add_argument("--max-size-t", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--temporal-patch-size", type=int, default=2)
    parser.add_argument("--input-sequence-length", type=int, default=512)
    parser.add_argument("--cache-len", type=int, default=2048)
    parser.add_argument("--max-pe-length", type=int, default=32768)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--disable-wrapped-llm-cache", action="store_true")
    parser.add_argument("--text-attention-layers", type=str, default="0,1,2")
    parser.add_argument("--mask-export-dir", type=str, default=None)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def get_effective_max_size_t(args) -> int:
    return 2 if args.mode == "image" else args.max_size_t


def build_sampled_video_metadata(video_tensor: torch.Tensor, sample_fps: float):
    num_frames = int(video_tensor.shape[0])
    duration = None if sample_fps <= 0 else num_frames / sample_fps
    return [
        VideoMetadata(
            total_num_frames=num_frames,
            fps=sample_fps,
            width=int(video_tensor.shape[-1]),
            height=int(video_tensor.shape[-2]),
            duration=duration,
            video_backend="sampled_clip",
            frames_indices=list(range(num_frames)),
        )
    ]


def build_video_raw_clip(video_tensor: torch.Tensor, target_t: int, target_h: int, target_w: int) -> torch.Tensor:
    if video_tensor.dim() != 4:
        raise ValueError(f"Expected sampled video tensor with shape [T, C, H, W], but got {tuple(video_tensor.shape)}")

    video_tensor = video_tensor.float()
    if video_tensor.shape[0] != target_t:
        if video_tensor.shape[0] > target_t:
            indices = torch.linspace(0, video_tensor.shape[0] - 1, target_t).round().long()
            video_tensor = video_tensor.index_select(0, indices)
        else:
            pad_count = target_t - video_tensor.shape[0]
            pad_frames = video_tensor[-1:].repeat(pad_count, 1, 1, 1)
            video_tensor = torch.cat([video_tensor, pad_frames], dim=0)

    if video_tensor.shape[-2:] != (target_h, target_w):
        video_tensor = F.interpolate(video_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)

    return video_tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def build_media_inputs(args, processor):
    prompt = args.prompt
    if prompt is None:
        prompt = "Describe this image." if args.mode == "image" else "Describe this video."

    if args.mode == "image":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": args.image_path,
                        "resized_height": args.image_size_h,
                        "resized_width": args.image_size_w,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=args.patch_size)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        return inputs

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video_path,
                    "nframes": args.max_size_t,
                    "resized_height": args.image_size_h,
                    "resized_width": args.image_size_w,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    sampled_video = video_inputs[0]
    sampled_metadata = build_sampled_video_metadata(sampled_video, float(video_kwargs["fps"][0]))
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        videos_kwargs={"video_metadata": sampled_metadata, "return_metadata": True},
    )
    inputs["hm_pixel_values"] = [
        build_video_raw_clip(
            sampled_video,
            target_t=args.max_size_t,
            target_h=args.image_size_h,
            target_w=args.image_size_w,
        )
    ]
    return inputs


def build_standard_inputs(inputs):
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    }
    return {key: value for key, value in inputs.items() if key in allowed}


def compare_tensors(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    diff = (lhs - rhs).abs()
    max_abs_diff = diff.max().item() if lhs.numel() else 0.0
    mean_abs_diff = diff.mean().item() if lhs.numel() else 0.0
    # cosine similarity on flattened vectors
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(lhs_flat.unsqueeze(0), rhs_flat.unsqueeze(0)).item() if lhs.numel() else 0.0
    print(f"{name}: shape={tuple(lhs.shape)}, max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}, cos_sim={cos_sim:.6f}")
    return max_abs_diff


def slice_to_match(lhs: torch.Tensor, rhs: torch.Tensor):
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    if lhs.ndim != rhs.ndim:
        return lhs, rhs
    slices = []
    for lhs_dim, rhs_dim in zip(lhs.shape, rhs.shape):
        if lhs_dim == rhs_dim:
            slices.append(slice(None))
        elif lhs_dim > rhs_dim:
            slices.append(slice(0, rhs_dim))
        else:
            return lhs, rhs
    return lhs[tuple(slices)], rhs


def compare_aligned_tensors(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    aligned_lhs, aligned_rhs = slice_to_match(lhs, rhs)
    return compare_tensors(name, aligned_lhs, aligned_rhs)


def print_mask_stats(name: str, official_mask: torch.Tensor, local_mask: torch.Tensor):
    # official_mask: bool tensor (True=CAN attend) from create_causal_mask
    # local_mask:    additive float tensor (0.0=CAN attend, large_negative=masked)
    official_can = official_mask.bool().detach().cpu()                  # True = CAN attend
    local_can = (local_mask.detach().float().cpu() >= -1.0)             # True = CAN attend
    if official_can.ndim == 4:
        official_can = official_can[0, 0]
    if local_can.ndim == 4:
        local_can = local_can[0, 0]
    min_r = min(official_can.shape[0], local_can.shape[0])
    min_c = min(official_can.shape[1], local_can.shape[1])
    official_can = official_can[:min_r, :min_c]
    local_can = local_can[:min_r, :min_c]
    both_allow = (official_can & local_can).sum().item()
    only_local = (local_can & ~official_can).sum().item()       # causal but not same-packed
    only_official = (official_can & ~local_can).sum().item()    # should be 0 for AND mask
    print(
        f"{name}: both_allow={both_allow} only_local_allows={only_local} only_official_allows={only_official}"
    )


def print_probability_leak_stats(name: str, probs: torch.Tensor, official_mask: torch.Tensor):
    probs, official_mask = slice_to_match(probs, official_mask)
    probs = probs.detach().float().cpu()
    official_valid = (~official_mask.bool()).detach().cpu().expand_as(probs)
    valid_mass = (probs * official_valid).sum(dim=-1)
    invalid_mass = (probs * (~official_valid)).sum(dim=-1)
    print(
        f"{name}: valid_mass_range=({valid_mass.min().item():.6g},{valid_mass.max().item():.6g}) "
        f"invalid_mass_range=({invalid_mass.min().item():.6g},{invalid_mass.max().item():.6g})"
    )


def print_position_jump_stats(position_ids: torch.Tensor, max_items: int = 10):
    position_ids = position_ids.detach().cpu()
    diff = position_ids[1:] - position_ids[:-1]
    jump_indices = (diff != 1).nonzero().flatten().tolist()
    jump_pairs = [
        (index, int(position_ids[index].item()), int(position_ids[index + 1].item()))
        for index in jump_indices[:max_items]
    ]
    print(f"text_position_jump_count: {len(jump_indices)}")
    print(f"text_position_jump_examples: {jump_pairs}")


def summarize_valid_runs(valid_row: torch.Tensor, max_runs: int = 8):
    valid_indices = valid_row.nonzero(as_tuple=False).flatten().tolist()
    if not valid_indices:
        return []
    runs = []
    start = valid_indices[0]
    end = start
    for index in valid_indices[1:]:
        if index == end + 1:
            end = index
        else:
            runs.append((start, end))
            start = index
            end = index
    runs.append((start, end))
    if len(runs) > max_runs:
        return runs[:max_runs] + [("...", "...")]
    return runs


def print_mask_row_examples(
    name: str,
    official_mask: torch.Tensor,
    local_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    max_rows: int = 10,
):
    local_mask, official_mask = slice_to_match(local_mask, official_mask)
    official_valid = (~official_mask.bool()).detach().cpu()[0, 0]
    local_valid = (~local_mask.bool()).detach().cpu()[0, 0]
    mismatch_rows = [row for row in range(official_valid.shape[0]) if not torch.equal(official_valid[row], local_valid[row])]
    row_stats = [
        (row, int(official_valid[row].sum().item()), int(local_valid[row].sum().item()))
        for row in mismatch_rows[:max_rows]
    ]
    print(f"{name}_mismatch_row_count: {len(mismatch_rows)}")
    print(f"{name}_mismatch_row_examples: {row_stats}")
    if position_ids is not None:
        position_ids = position_ids.detach().cpu().tolist()

    for row in mismatch_rows[:max_rows]:
        official_runs = summarize_valid_runs(official_valid[row])
        local_runs = summarize_valid_runs(local_valid[row])
        row_text_pos = position_ids[row] if position_ids is not None and row < len(position_ids) else None
        print(
            f"{name}_row_detail: row={row} text_pos={row_text_pos} "
            f"official_valid_count={int(official_valid[row].sum().item())} official_valid_runs={official_runs} "
            f"local_valid_count={int(local_valid[row].sum().item())} local_valid_runs={local_runs}"
        )


def export_mask_images(
    output_dir: str,
    prefix: str,
    official_mask: torch.Tensor,
    local_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
):
    """
    official_mask: torch.bool tensor from create_causal_mask (True = CAN attend, False = masked)
    local_mask:    additive float tensor (0.0 = CAN attend, large_negative = masked)
    Both are stored as: white pixel (255) = CAN attend, black pixel (0) = masked
    diff image: white = both allow, red = only local allows, blue = never allowed by either
    """
    os.makedirs(output_dir, exist_ok=True)
    local_mask = local_mask.detach().float().cpu()
    official_mask = official_mask.detach().cpu()

    # official_mask is bool tensor: True=attend.  After slicing to same seq-len:
    # shape is [1, 1, q, k] for official, [1, 1, q, k] for local (or [q, k] for local)
    if official_mask.ndim == 4:
        official_can_attend = official_mask[0, 0].bool()            # True = CAN attend
    else:
        official_can_attend = official_mask.bool()

    if local_mask.ndim == 4:
        local_can_attend = local_mask[0, 0] >= -1.0                 # 0.0 = CAN attend
    else:
        local_can_attend = local_mask >= -1.0

    # Align shapes: official is [actual_seq, actual_seq], local may be [padded_seq, padded_seq]
    min_rows = min(official_can_attend.shape[0], local_can_attend.shape[0])
    min_cols = min(official_can_attend.shape[1], local_can_attend.shape[1])
    official_can_attend = official_can_attend[:min_rows, :min_cols]
    local_can_attend = local_can_attend[:min_rows, :min_cols]

    # Save individual masks: white = CAN attend, black = masked
    write_png(
        official_can_attend.to(torch.uint8).unsqueeze(0) * 255,
        os.path.join(output_dir, f"{prefix}_official_mask.png"),
    )
    write_png(
        local_can_attend.to(torch.uint8).unsqueeze(0) * 255,
        os.path.join(output_dir, f"{prefix}_local_mask.png"),
    )

    # Diff image:
    #   white  = both allow (AND intersection = official's valid set since official ⊆ local)
    #   red    = only local allows (causal lower-triangle but different packed group)
    #   blue   = only official allows (should be empty for proper AND mask)
    #   black  = both mask (upper triangle)
    both = official_can_attend & local_can_attend
    local_only = local_can_attend & ~official_can_attend
    official_only = official_can_attend & ~local_can_attend
    diff = torch.zeros((3, min_rows, min_cols), dtype=torch.uint8)
    diff[:, both] = 255               # white
    diff[0, local_only] = 255         # red channel → red
    diff[2, official_only] = 255      # blue channel → blue
    write_png(diff, os.path.join(output_dir, f"{prefix}_mask_diff.png"))

    official_count = int(official_can_attend.sum().item())
    local_count = int(local_can_attend.sum().item())
    summary_lines = [
        f"shape=({min_rows}, {min_cols})",
        f"official_can_attend_count={official_count}  ({100.0*official_count/min_rows/min_cols:.2f}%)",
        f"local_can_attend_count={local_count}  ({100.0*local_count/min_rows/min_cols:.2f}%)",
        f"both_allow={int(both.sum())}",
        f"only_local_allows={int(local_only.sum())}  (lower-triangle but different packed group)",
        f"only_official_allows={int(official_only.sum())}  (should be 0 for AND mask)",
        "diff_legend=white:both_allow(AND_intersection)  red:local_only(causal_but_not_packed)  blue:official_only  black:both_mask",
    ]
    if position_ids is not None:
        position_ids = position_ids.detach().cpu()
        vals, counts = torch.unique(position_ids, return_counts=True)
        summary_lines.append(
            "time_position_id_histogram=" + str(list(zip(vals.tolist(), counts.tolist())))
        )
    summary_path = os.path.join(output_dir, f"{prefix}_mask_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def parse_layer_indices(spec: str):
    if not spec.strip():
        return []
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def reconstruct_processor_patches_from_hm(
    hm_pixel_values: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
):
    raw = hm_pixel_values.float() / 255.0
    raw = (raw - 0.5) / 0.5
    batch, channels, frames, height, width = raw.shape
    raw = raw.reshape(
        batch,
        frames // temporal_patch_size,
        temporal_patch_size,
        channels,
        height // patch_size // merge_size,
        merge_size,
        patch_size,
        width // patch_size // merge_size,
        merge_size,
        patch_size,
    )
    raw = raw.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    raw = raw.reshape(
        batch,
        (frames // temporal_patch_size) * (height // patch_size) * (width // patch_size),
        channels * temporal_patch_size * patch_size * patch_size,
    )
    return raw.squeeze(0).contiguous()


def compare_patch_inputs_and_outputs(args, standard_inputs, hm_pixel_values, dtype):
    if args.mode == "image":
        reference_patches = standard_inputs["pixel_values"].float().cpu()
    else:
        reference_patches = standard_inputs["pixel_values_videos"].float().cpu()

    reconstructed_patches = reconstruct_processor_patches_from_hm(
        hm_pixel_values.cpu(),
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        merge_size=2,
    )
    reconstruct_diff = compare_tensors("reconstructed_processor_patches", reconstructed_patches, reference_patches)

    reference_model = HFQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    reference_model = reference_model.to(args.device)
    with torch.no_grad():
        official_patch = reference_model.visual.patch_embed(reference_patches.to(args.device, dtype=dtype)).float().cpu()
    reference_model = reference_model.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    local_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    visual = local_model.visual
    visual.eval().cpu()
    vision_register_wrap_cls(local_model)
    wrapped_vision = wrap_llm_model(
        visual,
        {
            "max_size_w": args.image_size_w,
            "max_size_h": args.image_size_h,
            "max_size_t": get_effective_max_size_t(args),
            "patch_size": args.patch_size,
            "temporal_patch_size": args.temporal_patch_size,
        },
    )
    wrapped_vision = wrapped_vision.to(args.device, dtype=dtype).eval()
    with torch.no_grad():
        wrapped_patch = wrapped_vision.patch_embed(hm_pixel_values.to(args.device, dtype=dtype)).float().cpu()
    wrapped_vision = wrapped_vision.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    patch_diff = compare_tensors("patch_embed_output", wrapped_patch, official_patch)
    return reconstruct_diff, patch_diff


def move_batch(data, device, dtype=None):
    moved = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            if dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def run_reference(args, standard_inputs, dtype):
    model = HFQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    model = model.to(args.device)
    model_inputs = move_batch(standard_inputs, args.device, dtype)
    with torch.no_grad():
        if args.mode == "image":
            visual_chunks, deepstack_out = model.get_image_features(
                model_inputs["pixel_values"],
                model_inputs["image_grid_thw"],
            )
            visual_out = visual_chunks[0]
        else:
            visual_chunks, deepstack_out = model.get_video_features(
                model_inputs["pixel_values_videos"],
                model_inputs["video_grid_thw"],
            )
            visual_out = visual_chunks[0]
        logits = model(**model_inputs).logits.detach().float().cpu()
    visual_out = visual_out.detach().float().cpu()
    deepstack_out = [tensor.detach().float().cpu() for tensor in deepstack_out]
    model = model.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return visual_out, deepstack_out, logits


def run_wrapped_vision(args, hm_pixel_values, dtype):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    visual = model.visual
    visual.eval().cpu()
    vision_register_wrap_cls(model)
    wrapped_vision = wrap_llm_model(
        visual,
        {
            "max_size_w": args.image_size_w,
            "max_size_h": args.image_size_h,
            "max_size_t": get_effective_max_size_t(args),
            "patch_size": args.patch_size,
            "temporal_patch_size": args.temporal_patch_size,
        },
    )
    wrapped_vision = wrapped_vision.to(args.device, dtype=dtype).eval()
    with torch.no_grad():
        visual_out, deepstack_out = wrapped_vision(hm_pixel_values.to(args.device, dtype=dtype))
    visual_out = visual_out.squeeze(0).detach().float().cpu()
    deepstack_out = [tensor.squeeze(0).detach().float().cpu() for tensor in deepstack_out]
    wrapped_vision = wrapped_vision.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return visual_out, deepstack_out


def pad_input_ids(input_ids: torch.Tensor, padded_seq_len: int, pad_token_id: int = 0) -> torch.Tensor:
    if input_ids.shape[-1] == padded_seq_len:
        return input_ids
    padding = torch.full(
        (input_ids.shape[0], padded_seq_len - input_ids.shape[-1]),
        fill_value=pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat([input_ids, padding], dim=-1)


def prepare_prefill_inputs(args, inputs, wrapped_visual_out, wrapped_deepstack_out, token_embedding, target_seq_len=None):
    data_prefill = {
        "input_ids": inputs["input_ids"],
        "past_seq_length": 0,
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
    }
    if args.mode == "image":
        data_prefill["image_embeds"] = wrapped_visual_out.to(torch.float16)
        data_prefill["deepstack_image_embeds"] = [tensor.to(torch.float16) for tensor in wrapped_deepstack_out]
    else:
        data_prefill["video_embeds"] = wrapped_visual_out.to(torch.float16)
        data_prefill["deepstack_video_embeds"] = [tensor.to(torch.float16) for tensor in wrapped_deepstack_out]

    actual_seq_len = int(inputs["input_ids"].shape[-1])
    padded_seq_len = (
        int(target_seq_len)
        if target_seq_len is not None
        else int(math.ceil(actual_seq_len / args.input_sequence_length) * args.input_sequence_length)
    )

    data_preprocess = Qwen3_VLDataPreprocess(token_embedding, padded_seq_len)
    data_input = data_preprocess(data_prefill)
    (
        inputs_embeds,
        time_position_ids,
        height_position_ids,
        width_position_ids,
        past_seq_length,
        current_input_length,
        deepstack_embed_0,
        deepstack_embed_1,
        deepstack_embed_2,
    ) = data_input

    padded_input_ids = pad_input_ids(inputs["input_ids"], padded_seq_len)
    visual_pos_masks = (padded_input_ids == 151655) | (padded_input_ids == 151656)
    position_ids = torch.stack([time_position_ids, height_position_ids, width_position_ids], dim=0).unsqueeze(1)
    deepstack_visual_embeds = [deepstack_embed_0, deepstack_embed_1, deepstack_embed_2]
    return {
        "actual_seq_len": actual_seq_len,
        "padded_seq_len": padded_seq_len,
        "runtime_current_input_length": torch.tensor([actual_seq_len], dtype=torch.int32),
        "padded_input_ids": padded_input_ids,
        "inputs_embeds": inputs_embeds,
        "time_position_ids": time_position_ids,
        "height_position_ids": height_position_ids,
        "width_position_ids": width_position_ids,
        "position_ids": position_ids,
        "past_seq_length": past_seq_length,
        "current_input_length": current_input_length,
        "deepstack_visual_embeds": deepstack_visual_embeds,
        "visual_pos_masks": visual_pos_masks,
    }


def compare_prefill_input_alignment(reference_prepared_inputs, wrapped_prepared_inputs):
    actual_seq_len = reference_prepared_inputs["actual_seq_len"]
    print(
        "prefill_length_alignment: "
        f"reference_actual_seq_len={reference_prepared_inputs['actual_seq_len']} "
        f"wrapped_padded_seq_len={wrapped_prepared_inputs['padded_seq_len']} "
        f"wrapped_runtime_current_input_length={int(wrapped_prepared_inputs['runtime_current_input_length'][0].item())} "
        f"wrapped_preprocess_current_input_length={int(wrapped_prepared_inputs['current_input_length'][0].item())}"
    )
    compare_tensors(
        "prefill_inputs_embeds_valid",
        wrapped_prepared_inputs["inputs_embeds"][:, :actual_seq_len, :],
        reference_prepared_inputs["inputs_embeds"],
    )
    compare_tensors(
        "prefill_time_position_ids_valid",
        wrapped_prepared_inputs["time_position_ids"][:actual_seq_len],
        reference_prepared_inputs["time_position_ids"],
    )
    compare_tensors(
        "prefill_height_position_ids_valid",
        wrapped_prepared_inputs["height_position_ids"][:actual_seq_len],
        reference_prepared_inputs["height_position_ids"],
    )
    compare_tensors(
        "prefill_width_position_ids_valid",
        wrapped_prepared_inputs["width_position_ids"][:actual_seq_len],
        reference_prepared_inputs["width_position_ids"],
    )
    for layer_index, reference_tensor in enumerate(reference_prepared_inputs["deepstack_visual_embeds"]):
        compare_tensors(
            f"prefill_deepstack_{layer_index}_valid",
            wrapped_prepared_inputs["deepstack_visual_embeds"][layer_index][:, :actual_seq_len, :],
            reference_tensor,
        )


def build_wrapped_position_embeddings(language_model, prepared_inputs):
    cos = language_model.rotary_emb.cos_cached
    sin = language_model.rotary_emb.sin_cached

    time_cos = cos[prepared_inputs["time_position_ids"]]
    time_sin = sin[prepared_inputs["time_position_ids"]]
    height_cos = cos[prepared_inputs["height_position_ids"]]
    height_sin = sin[prepared_inputs["height_position_ids"]]
    width_cos = cos[prepared_inputs["width_position_ids"]]
    width_sin = sin[prepared_inputs["width_position_ids"]]

    time_cos = time_cos * language_model.time_mask.to(time_cos.device, time_cos.dtype)
    time_sin = time_sin * language_model.time_mask.to(time_sin.device, time_sin.dtype)
    height_cos = height_cos * language_model.hight_mask.to(height_cos.device, height_cos.dtype)
    height_sin = height_sin * language_model.hight_mask.to(height_sin.device, height_sin.dtype)
    width_cos = width_cos * language_model.width_mask.to(width_cos.device, width_cos.dtype)
    width_sin = width_sin * language_model.width_mask.to(width_sin.device, width_sin.dtype)

    cos = (time_cos + height_cos + width_cos).squeeze(1).unsqueeze(0).unsqueeze(0)
    sin = (time_sin + height_sin + width_sin).squeeze(1).unsqueeze(0).unsqueeze(0)
    return cos, sin


def run_reference_prefill_llm_layerwise(args, prepared_inputs, dtype):
    model = HFQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    ).eval()
    model = model.to(args.device)
    language_model = model.model.language_model

    inputs_embeds = prepared_inputs["inputs_embeds"].to(args.device, dtype=dtype)
    position_ids = prepared_inputs["position_ids"].to(args.device)
    visual_pos_masks = prepared_inputs["visual_pos_masks"].to(args.device)
    deepstack_visual_embeds = []
    for tensor in prepared_inputs["deepstack_visual_embeds"]:
        dense_tensor = tensor[:, : prepared_inputs["padded_seq_len"], :].to(args.device, dtype=dtype)
        deepstack_visual_embeds.append(dense_tensor[visual_pos_masks].contiguous())
    cache_position = torch.arange(prepared_inputs["padded_seq_len"], device=args.device)
    attention_mask = create_causal_mask(
        config=language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=torch.ones(1, prepared_inputs["padded_seq_len"], dtype=torch.long, device=args.device),
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids[0],
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds
    layer_pre = []
    layer_post = []
    layer_inputs = []
    layer_details = []
    attention_step_layers = set(parse_layer_indices(args.text_attention_layers))
    attention_steps = {}
    with torch.no_grad():
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            layer_inputs.append(hidden_states.detach().float().cpu())
            if layer_idx in attention_step_layers:
                attention_steps[layer_idx] = {
                    key: value.detach().float().cpu()
                    for key, value in compute_official_text_attention_steps(
                        decoder_layer.self_attn,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                    ).items()
                }
            residual = hidden_states
            normed_hidden_states = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids[0],
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + attn_output
            residual = hidden_states
            mlp_input = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_output = decoder_layer.mlp(mlp_input)
            hidden_states = residual + mlp_output
            layer_pre.append(hidden_states.detach().float().cpu())
            if layer_idx < 3:
                layer_details.append(
                    {
                        "normed_hidden_states": normed_hidden_states.detach().float().cpu(),
                        "attn_output": attn_output.detach().float().cpu(),
                        "mlp_input": mlp_input.detach().float().cpu(),
                        "mlp_output": mlp_output.detach().float().cpu(),
                    }
                )
            if layer_idx < len(deepstack_visual_embeds):
                hidden_states = language_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
            layer_post.append(hidden_states.detach().float().cpu())
        norm_hidden_states = language_model.norm(hidden_states)
        logits = model.lm_head(norm_hidden_states).detach().float().cpu()

    result = {
        "position_cos": position_embeddings[0].detach().float().cpu(),
        "position_sin": position_embeddings[1].detach().float().cpu(),
        "attention_mask": attention_mask.detach().float().cpu(),
        "layer_inputs": layer_inputs,
        "attention_steps": attention_steps,
        "layer_pre": layer_pre,
        "layer_post": layer_post,
        "layer_details": layer_details,
        "norm_hidden_states": norm_hidden_states.detach().float().cpu(),
        "logits": logits,
    }
    model = model.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_wrapped_llm_layerwise(args, prepared_inputs, dtype):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation="eager",
    ).eval()

    if hasattr(model.model, "visual"):
        model.model.visual = nn.Identity()

    llm_register_wrap_cls(model)
    wrapped_llm = wrap_llm_model(
        model,
        {
            "batch_size": 1,
            "max_sequence_length": args.cache_len,
            "max_pe_length": args.max_pe_length,
            "input_sequence_length": prepared_inputs["padded_seq_len"],
            "use_cache": not args.disable_wrapped_llm_cache,
            "num_logits_to_keep": 0,
            "kv_cache": {"cache_axis": 2},
        },
    )
    wrapped_llm = wrapped_llm.to(args.device, dtype=dtype).eval()
    language_model = wrapped_llm.model.language_model

    num_layers = model.language_model.config.num_hidden_layers
    num_kv_heads = model.language_model.config.num_key_value_heads
    head_dim = model.language_model.config.head_dim
    cache_shape = (1, num_kv_heads, args.cache_len, head_dim)
    past_key_caches = [torch.zeros(cache_shape, dtype=dtype, device=args.device) for _ in range(num_layers)]
    past_value_caches = [torch.zeros(cache_shape, dtype=dtype, device=args.device) for _ in range(num_layers)]

    inputs_embeds = prepared_inputs["inputs_embeds"].to(args.device, dtype=dtype)
    position_ids = prepared_inputs["position_ids"].to(args.device)
    deepstack_embed_0, deepstack_embed_1, deepstack_embed_2 = [
        tensor[:, : prepared_inputs["padded_seq_len"], :].to(args.device, dtype=dtype)
        for tensor in prepared_inputs["deepstack_visual_embeds"]
    ]
    cache_position = torch.arange(prepared_inputs["padded_seq_len"], device=args.device)
    attention_mask = create_causal_mask(
        config=language_model.config,
        input_embeds=inputs_embeds,
        attention_mask=torch.ones(1, prepared_inputs["padded_seq_len"], dtype=torch.long, device=args.device),
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids[0],
    )
    position_embeddings = build_wrapped_position_embeddings(language_model, prepared_inputs)
    layer_pre = []
    layer_post = []
    layer_inputs = []
    layer_details = []
    attention_step_layers = set(parse_layer_indices(args.text_attention_layers))
    attention_steps = {}
    with torch.no_grad():
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            layer_inputs.append(hidden_states.detach().float().cpu())
            if layer_idx in attention_step_layers:
                attention_steps[layer_idx] = {
                    key: value.detach().float().cpu()
                    for key, value in compute_wrapped_text_attention_steps(
                        decoder_layer.self_attn,
                        hidden_states,
                        position_embeddings,
                        prepared_inputs["past_seq_length"].to(args.device),
                        prepared_inputs["current_input_length"].to(args.device),
                        None if args.disable_wrapped_llm_cache else past_key_caches[layer_idx],
                        None if args.disable_wrapped_llm_cache else past_value_caches[layer_idx],
                        attention_mask,
                    ).items()
                }
            residual = hidden_states
            normed_hidden_states = decoder_layer.input_layernorm(hidden_states)
            attn_output, _, _ = decoder_layer.self_attn(
                hidden_states=normed_hidden_states,
                past_seq_length=prepared_inputs["past_seq_length"].to(args.device),
                current_input_length=prepared_inputs["current_input_length"].to(args.device),
                past_k_cache=None if args.disable_wrapped_llm_cache else past_key_caches[layer_idx],
                past_v_cache=None if args.disable_wrapped_llm_cache else past_value_caches[layer_idx],
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + attn_output
            residual = hidden_states
            mlp_input = decoder_layer.post_attention_layernorm(hidden_states)
            mlp_output = decoder_layer.mlp(mlp_input)
            hidden_states = residual + mlp_output
            layer_pre.append(hidden_states.detach().float().cpu())
            if layer_idx < 3:
                layer_details.append(
                    {
                        "normed_hidden_states": normed_hidden_states.detach().float().cpu(),
                        "attn_output": attn_output.detach().float().cpu(),
                        "mlp_input": mlp_input.detach().float().cpu(),
                        "mlp_output": mlp_output.detach().float().cpu(),
                    }
                )
            if layer_idx == 0:
                hidden_states = hidden_states + deepstack_embed_0
            if layer_idx == 1:
                hidden_states = hidden_states + deepstack_embed_1
            if layer_idx == 2:
                hidden_states = hidden_states + deepstack_embed_2
            layer_post.append(hidden_states.detach().float().cpu())
        norm_hidden_states = language_model.norm(hidden_states)
        logits = wrapped_llm.lm_head(norm_hidden_states)
    result = {
        "position_cos": position_embeddings[0].detach().float().cpu(),
        "position_sin": position_embeddings[1].detach().float().cpu(),
        "layer_inputs": layer_inputs,
        "attention_steps": attention_steps,
        "layer_pre": layer_pre,
        "layer_post": layer_post,
        "layer_details": layer_details,
        "norm_hidden_states": norm_hidden_states.detach().float().cpu(),
        "logits": logits[:, : prepared_inputs["actual_seq_len"], :].detach().float().cpu(),
    }
    wrapped_llm = wrapped_llm.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def compute_official_text_attention_steps(attn_module, hidden_states, position_embeddings, attention_mask):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)

    q_proj = attn_module.q_proj(hidden_states)
    k_proj = attn_module.k_proj(hidden_states)
    v_proj = attn_module.v_proj(hidden_states)

    q_norm = attn_module.q_norm(q_proj.view(hidden_shape))
    k_norm = attn_module.k_norm(k_proj.view(hidden_shape))
    query_states = q_norm.transpose(1, 2)
    key_states = k_norm.transpose(1, 2)
    value_states = v_proj.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_rope, key_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_scaled = query_rope * attn_module.scaling
    repeated_key = torch.repeat_interleave(key_rope, attn_module.num_key_value_groups, dim=1)
    repeated_value = torch.repeat_interleave(value_states, attn_module.num_key_value_groups, dim=1)
    attn_logits = torch.matmul(query_scaled, repeated_key.transpose(2, 3))
    masked_logits = attn_logits + attention_mask[:, :, :, : repeated_key.shape[-2]]
    attn_probs = torch.softmax(masked_logits, dim=-1, dtype=torch.float32).to(query_scaled.dtype)
    attn_output_heads = torch.matmul(attn_probs, repeated_value)
    attn_output_merged = attn_output_heads.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = attn_module.o_proj(attn_output_merged)

    return {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "query_states": query_states,
        "key_states": key_states,
        "value_states": value_states,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "query_scaled": query_scaled,
        "repeated_key": repeated_key,
        "repeated_key_for_matmul": repeated_key.transpose(2, 3),
        "repeated_value": repeated_value,
        "attn_logits": attn_logits,
        "masked_logits": masked_logits,
        "attn_probs": attn_probs,
        "attn_output_heads": attn_output_heads,
        "attn_output_merged": attn_output_merged,
        "attn_output": attn_output,
    }


def compute_wrapped_text_attention_steps(
    attn_module,
    hidden_states,
    position_embeddings,
    past_seq_length,
    current_input_length,
    past_k_cache=None,
    past_v_cache=None,
    attention_mask=None,
):
    bsz, q_len, _ = hidden_states.size()

    q_proj = attn_module.q_proj(hidden_states)
    k_proj = attn_module.k_proj(hidden_states)
    v_proj = attn_module.v_proj(hidden_states)

    q_norm = attn_module.q_norm(q_proj.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim))
    k_norm = attn_module.k_norm(k_proj.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim))
    query_states = q_norm.transpose(1, 2)
    key_states = k_norm.transpose(1, 2)
    value_states = v_proj.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_rope, key_rope = attn_module.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

    cached_key = key_rope
    cached_value = value_states
    if attn_module.use_cache:
        cached_key = attn_module.k_cache(key_rope, past_seq_length, current_input_length, past_k_cache)
        cached_value = attn_module.v_cache(value_states, past_seq_length, current_input_length, past_v_cache)

    query_scaled = query_rope * attn_module.kv_scale
    repeated_key = torch.repeat_interleave(cached_key.transpose(2, 3), attn_module.num_key_value_groups, dim=1)
    repeated_value = torch.repeat_interleave(cached_value, attn_module.num_key_value_groups, dim=1)
    attn_logits = torch.matmul(query_scaled, repeated_key)
    attn_probs = attn_module.masked_softmax(attn_logits, past_seq_length)
    attn_output_heads = torch.matmul(attn_probs, repeated_value)
    attn_output_merged = attn_output_heads.transpose(1, 2).reshape(bsz, q_len, attn_module.config.num_attention_heads * attn_module.head_dim)
    attn_output = attn_module.o_proj(attn_output_merged)

    current_repeated_key = repeated_key[:, :, :, :q_len]
    current_repeated_value = repeated_value[:, :, :q_len, :]
    explicit_attn_logits = torch.matmul(query_scaled, current_repeated_key)
    explicit_causal_mask = torch.full(
        (q_len, q_len),
        fill_value=torch.finfo(explicit_attn_logits.dtype).min,
        device=explicit_attn_logits.device,
        dtype=explicit_attn_logits.dtype,
    )
    explicit_causal_mask = torch.triu(explicit_causal_mask, diagonal=1).view(1, 1, q_len, q_len)
    explicit_masked_logits = explicit_attn_logits + explicit_causal_mask
    explicit_attn_probs = torch.softmax(explicit_masked_logits, dim=-1, dtype=torch.float32).to(query_scaled.dtype)
    explicit_attn_output_heads = torch.matmul(explicit_attn_probs, current_repeated_value)
    explicit_attn_output_merged = explicit_attn_output_heads.transpose(1, 2).reshape(
        bsz, q_len, attn_module.config.num_attention_heads * attn_module.head_dim
    )
    explicit_attn_output = attn_module.o_proj(explicit_attn_output_merged)

    nocache_repeated_key = torch.repeat_interleave(key_rope.transpose(2, 3), attn_module.num_key_value_groups, dim=1)
    nocache_repeated_value = torch.repeat_interleave(value_states, attn_module.num_key_value_groups, dim=1)
    nocache_attn_logits = torch.matmul(query_scaled, nocache_repeated_key)
    nocache_masked_logits = nocache_attn_logits + explicit_causal_mask
    nocache_attn_probs = torch.softmax(nocache_masked_logits, dim=-1, dtype=torch.float32).to(query_scaled.dtype)
    nocache_attn_output_heads = torch.matmul(nocache_attn_probs, nocache_repeated_value)
    nocache_attn_output_merged = nocache_attn_output_heads.transpose(1, 2).reshape(
        bsz, q_len, attn_module.config.num_attention_heads * attn_module.head_dim
    )
    nocache_attn_output = attn_module.o_proj(nocache_attn_output_merged)

    official_masked_logits = None
    official_mask_attn_probs = None
    official_mask_attn_output_heads = None
    official_mask_attn_output_merged = None
    official_mask_attn_output = None
    if attention_mask is not None:
        official_masked_logits = nocache_attn_logits + attention_mask[:, :, :, : nocache_repeated_key.shape[-1]]
        official_mask_attn_probs = torch.softmax(official_masked_logits, dim=-1, dtype=torch.float32).to(query_scaled.dtype)
        official_mask_attn_output_heads = torch.matmul(official_mask_attn_probs, nocache_repeated_value)
        official_mask_attn_output_merged = official_mask_attn_output_heads.transpose(1, 2).reshape(
            bsz, q_len, attn_module.config.num_attention_heads * attn_module.head_dim
        )
        official_mask_attn_output = attn_module.o_proj(official_mask_attn_output_merged)

    return {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "query_states": query_states,
        "key_states": key_states,
        "value_states": value_states,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "cached_key": cached_key,
        "cached_value": cached_value,
        "query_scaled": query_scaled,
        "repeated_key": repeated_key,
        "repeated_value": repeated_value,
        "attn_logits": attn_logits,
        "attn_probs": attn_probs,
        "attn_output_heads": attn_output_heads,
        "attn_output_merged": attn_output_merged,
        "attn_output": attn_output,
        "explicit_causal_mask": explicit_causal_mask,
        "explicit_attn_logits": explicit_attn_logits,
        "explicit_masked_logits": explicit_masked_logits,
        "explicit_attn_probs": explicit_attn_probs,
        "explicit_attn_output_heads": explicit_attn_output_heads,
        "explicit_attn_output_merged": explicit_attn_output_merged,
        "explicit_attn_output": explicit_attn_output,
        "nocache_repeated_key": nocache_repeated_key,
        "nocache_repeated_value": nocache_repeated_value,
        "nocache_attn_logits": nocache_attn_logits,
        "nocache_masked_logits": nocache_masked_logits,
        "nocache_attn_probs": nocache_attn_probs,
        "nocache_attn_output_heads": nocache_attn_output_heads,
        "nocache_attn_output_merged": nocache_attn_output_merged,
        "nocache_attn_output": nocache_attn_output,
        "official_masked_logits": official_masked_logits,
        "official_mask_attn_probs": official_mask_attn_probs,
        "official_mask_attn_output_heads": official_mask_attn_output_heads,
        "official_mask_attn_output_merged": official_mask_attn_output_merged,
        "official_mask_attn_output": official_mask_attn_output,
    }


def compare_text_attention_step_errors(args, prepared_inputs, reference_llm, wrapped_llm):
    layer_indices = parse_layer_indices(args.text_attention_layers)
    if not layer_indices:
        return

    for layer_idx in layer_indices:
        hidden_states = reference_llm["layer_inputs"][layer_idx]
        official_steps = reference_llm["attention_steps"][layer_idx]
        wrapped_steps = wrapped_llm["attention_steps"][layer_idx]
        if layer_idx == layer_indices[0]:
            print_position_jump_stats(prepared_inputs["time_position_ids"][: hidden_states.shape[1]])
            compare_aligned_tensors(
                "text_attention_mask_explicit_vs_official",
                wrapped_steps["explicit_causal_mask"],
                reference_llm["attention_mask"],
            )
            if args.mask_export_dir:
                export_mask_images(
                    args.mask_export_dir,
                    f"text_attn_layer_{layer_idx}",
                    reference_llm["attention_mask"],
                    wrapped_steps["explicit_causal_mask"],
                    prepared_inputs["time_position_ids"][: hidden_states.shape[1]],
                )
            print_mask_stats(
                "text_attention_valid_mask_diff",
                reference_llm["attention_mask"],
                wrapped_steps["explicit_causal_mask"],
            )
            print_mask_row_examples(
                "text_attention_valid_mask_diff",
                reference_llm["attention_mask"],
                wrapped_steps["explicit_causal_mask"],
                prepared_inputs["time_position_ids"][: hidden_states.shape[1]],
                max_rows=25,
            )
            print_probability_leak_stats(
                "text_attn_probs_leak",
                wrapped_steps["attn_probs"],
                reference_llm["attention_mask"],
            )
            print_probability_leak_stats(
                "text_nocache_attn_probs_leak",
                wrapped_steps["nocache_attn_probs"],
                reference_llm["attention_mask"],
            )
            print_probability_leak_stats(
                "text_official_mask_attn_probs_leak",
                wrapped_steps["official_mask_attn_probs"],
                reference_llm["attention_mask"],
            )
            print_probability_leak_stats(
                "text_reference_attn_probs_leak",
                official_steps["attn_probs"],
                reference_llm["attention_mask"],
            )

        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_hidden_input",
            wrapped_llm["layer_inputs"][layer_idx],
            hidden_states,
        )
        step_pairs = [
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("q_norm", "q_norm"),
            ("k_norm", "k_norm"),
            ("query_states", "query_states"),
            ("key_states", "key_states"),
            ("value_states", "value_states"),
            ("query_rope", "query_rope"),
            ("key_rope", "key_rope"),
            ("query_scaled", "query_scaled"),
            ("attn_output_heads", "attn_output_heads"),
            ("attn_output_merged", "attn_output_merged"),
            ("attn_output", "attn_output"),
        ]
        for wrapped_key, official_key in step_pairs:
            compare_aligned_tensors(
                f"text_attn_layer_{layer_idx}_{wrapped_key}",
                wrapped_steps[wrapped_key],
                official_steps[official_key],
            )

        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_repeated_key",
            wrapped_steps["repeated_key"],
            official_steps["repeated_key_for_matmul"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_repeated_value",
            wrapped_steps["repeated_value"],
            official_steps["repeated_value"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_attn_logits",
            wrapped_steps["attn_logits"],
            official_steps["attn_logits"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_attn_probs",
            wrapped_steps["attn_probs"],
            official_steps["attn_probs"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_explicit_attn_logits",
            wrapped_steps["explicit_attn_logits"],
            official_steps["attn_logits"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_explicit_masked_logits",
            wrapped_steps["explicit_masked_logits"],
            official_steps["masked_logits"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_explicit_attn_probs",
            wrapped_steps["explicit_attn_probs"],
            official_steps["attn_probs"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_explicit_attn_output_heads",
            wrapped_steps["explicit_attn_output_heads"],
            official_steps["attn_output_heads"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_explicit_attn_output",
            wrapped_steps["explicit_attn_output"],
            official_steps["attn_output"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_repeated_key",
            wrapped_steps["nocache_repeated_key"],
            official_steps["repeated_key_for_matmul"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_repeated_value",
            wrapped_steps["nocache_repeated_value"],
            official_steps["repeated_value"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_attn_logits",
            wrapped_steps["nocache_attn_logits"],
            official_steps["attn_logits"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_masked_logits",
            wrapped_steps["nocache_masked_logits"],
            official_steps["masked_logits"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_attn_probs",
            wrapped_steps["nocache_attn_probs"],
            official_steps["attn_probs"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_attn_output_heads",
            wrapped_steps["nocache_attn_output_heads"],
            official_steps["attn_output_heads"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_nocache_attn_output",
            wrapped_steps["nocache_attn_output"],
            official_steps["attn_output"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_official_masked_logits",
            wrapped_steps["official_masked_logits"],
            official_steps["masked_logits"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_official_mask_attn_probs",
            wrapped_steps["official_mask_attn_probs"],
            official_steps["attn_probs"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_official_mask_attn_output_heads",
            wrapped_steps["official_mask_attn_output_heads"],
            official_steps["attn_output_heads"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_official_mask_attn_output",
            wrapped_steps["official_mask_attn_output"],
            official_steps["attn_output"],
        )

        if args.disable_wrapped_llm_cache:
            continue
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_cached_key_vs_key_rope",
            wrapped_steps["cached_key"],
            official_steps["key_rope"],
        )
        compare_aligned_tensors(
            f"text_attn_layer_{layer_idx}_cached_value_vs_value_states",
            wrapped_steps["cached_value"],
            official_steps["value_states"],
        )


def main():
    args = parse_args()
    dtype = get_dtype(args.dtype)
    processor = Qwen3VLProcessor.from_pretrained(args.model_path)
    inputs = build_media_inputs(args, processor)
    standard_inputs = build_standard_inputs(inputs)

    hm_pixel_values = inputs["hm_pixel_values"][0]
    target_t = get_effective_max_size_t(args)
    if hm_pixel_values.dim() == 4:
        hm_pixel_values = hm_pixel_values.unsqueeze(2)
    current_t = hm_pixel_values.shape[2]
    if current_t != target_t:
        if target_t % current_t != 0:
            raise ValueError(f"Cannot align hm_pixel_values temporal length from {current_t} to target {target_t}")
        hm_pixel_values = hm_pixel_values.repeat(1, 1, target_t // current_t, 1, 1)

    reconstruct_diff, patch_diff = compare_patch_inputs_and_outputs(args, standard_inputs, hm_pixel_values, dtype)

    reference_visual_out, reference_deepstack_out, reference_logits = run_reference(args, standard_inputs, dtype)
    wrapped_visual_out, wrapped_deepstack_out = run_wrapped_vision(args, hm_pixel_values, dtype)
    llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    ).eval()
    reference_prepared_inputs = prepare_prefill_inputs(
        args,
        standard_inputs,
        wrapped_visual_out,
        wrapped_deepstack_out,
        llm_model.model.get_input_embeddings(),
        target_seq_len=int(standard_inputs["input_ids"].shape[-1]),
    )
    prepared_inputs = prepare_prefill_inputs(
        args,
        standard_inputs,
        wrapped_visual_out,
        wrapped_deepstack_out,
        llm_model.model.get_input_embeddings(),
    )
    llm_model = llm_model.cpu()
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    compare_prefill_input_alignment(reference_prepared_inputs, prepared_inputs)

    reference_llm = run_reference_prefill_llm_layerwise(args, reference_prepared_inputs, dtype)
    wrapped_llm = run_wrapped_llm_layerwise(args, prepared_inputs, dtype)
    wrapped_logits = wrapped_llm["logits"]

    vision_diff = compare_tensors("vision_embeds", wrapped_visual_out, reference_visual_out)
    for index, (wrapped_tensor, reference_tensor) in enumerate(zip(wrapped_deepstack_out, reference_deepstack_out)):
        compare_tensors(f"deepstack_{index}", wrapped_tensor, reference_tensor)

    print(
        f"text_position_cos_shapes: wrapped={tuple(wrapped_llm['position_cos'].shape)} reference={tuple(reference_llm['position_cos'].shape)}"
    )
    print(
        f"text_position_sin_shapes: wrapped={tuple(wrapped_llm['position_sin'].shape)} reference={tuple(reference_llm['position_sin'].shape)}"
    )
    compare_text_attention_step_errors(args, prepared_inputs, reference_llm, wrapped_llm)
    for index, (wrapped_tensor, reference_tensor) in enumerate(zip(wrapped_llm["layer_pre"], reference_llm["layer_pre"])):
        compare_aligned_tensors(f"llm_layer_{index}_pre_deepstack", wrapped_tensor, reference_tensor)
    for index, (wrapped_tensor, reference_tensor) in enumerate(zip(wrapped_llm["layer_post"], reference_llm["layer_post"])):
        compare_aligned_tensors(f"llm_layer_{index}_post_deepstack", wrapped_tensor, reference_tensor)
    for layer_index, (wrapped_detail, reference_detail) in enumerate(zip(wrapped_llm["layer_details"], reference_llm["layer_details"])):
        compare_aligned_tensors(
            f"llm_layer_{layer_index}_normed_hidden_states",
            wrapped_detail["normed_hidden_states"],
            reference_detail["normed_hidden_states"],
        )
        compare_aligned_tensors(
            f"llm_layer_{layer_index}_attn_output",
            wrapped_detail["attn_output"],
            reference_detail["attn_output"],
        )
        compare_aligned_tensors(
            f"llm_layer_{layer_index}_mlp_input",
            wrapped_detail["mlp_input"],
            reference_detail["mlp_input"],
        )
        compare_aligned_tensors(
            f"llm_layer_{layer_index}_mlp_output",
            wrapped_detail["mlp_output"],
            reference_detail["mlp_output"],
        )
    compare_aligned_tensors("llm_norm_hidden_states", wrapped_llm["norm_hidden_states"], reference_llm["norm_hidden_states"])

    logits_diff = compare_aligned_tensors("prefill_logits", wrapped_logits, reference_logits)
    wrapped_next_token = int(wrapped_logits[:, -1, :].argmax(dim=-1).item())
    reference_next_token = int(reference_logits[:, -1, :].argmax(dim=-1).item())
    next_token_equal = wrapped_next_token == reference_next_token

    # Top-k token consistency check
    for k in [1, 5, 10, 20, 50]:
        w_topk = wrapped_logits[:, -1, :].topk(k, dim=-1).indices.squeeze(0)
        r_topk = reference_logits[:, -1, :].topk(k, dim=-1).indices.squeeze(0)
        w_set = set(w_topk.tolist())
        r_set = set(r_topk.tolist())
        overlap = len(w_set & r_set)
        print(f"top-{k}: overlap={overlap}/{k}, wrapped={w_topk.tolist()}, reference={r_topk.tolist()}")

    print(f"wrapped_next_token={wrapped_next_token}")
    print(f"reference_next_token={reference_next_token}")
    print(f"next_token_equal={next_token_equal}")
    print(f"reconstructed_patches_passed={reconstruct_diff <= args.atol}")
    print(f"patch_embed_passed={patch_diff <= args.atol}")
    print(f"vision_passed={vision_diff <= args.atol}")
    print(f"logits_passed={logits_diff <= args.atol}")


if __name__ == "__main__":
    main()