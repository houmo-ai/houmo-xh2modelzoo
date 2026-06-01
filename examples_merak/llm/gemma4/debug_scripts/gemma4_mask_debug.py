import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping

from xhmodel_merak.xh_llm.models.gemma4.data_preprocess import Gemma4DataPreprocess


def _vision_spans(mm_token_type_ids: torch.Tensor):
    mm = mm_token_type_ids[0].tolist()
    spans = []
    start = None
    for idx, value in enumerate(mm):
        is_vision = value in (1, 2)
        if is_vision and start is None:
            start = idx
        if start is not None and (idx == len(mm) - 1 or mm[idx + 1] not in (1, 2)):
            spans.append((start, idx + 1))
            start = None
    return spans


def main(args):
    model_dir = args.model_dir
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    messages = [{"role": "user", "content": [{"type": "image", "image": str(Path(args.image_path).resolve())}, {"type": "text", "text": args.prompt}]}]
    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    model_inputs = model_inputs.to(model.device)
    print('input_ids shape', tuple(model_inputs.input_ids.shape))
    print('mm_token_type_ids shape', tuple(model_inputs.mm_token_type_ids.shape))
    spans = _vision_spans(model_inputs.mm_token_type_ids)
    print('vision spans', spans)

    image_features = model.model.get_image_features(
        model_inputs.pixel_values,
        model_inputs.image_position_ids,
        return_dict=True,
    ).pooler_output
    image_features = image_features.to(model.device, torch.bfloat16)

    input_ids = model_inputs.input_ids.clone()
    input_ids[input_ids == model.config.image_token_id] = model.config.text_config.pad_token_id
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = (model_inputs.input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    merged_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

    mask_mapping = create_causal_mask_mapping(
        model.config,
        merged_embeds,
        model_inputs.attention_mask,
        past_key_values=None,
        position_ids=None,
        mm_token_type_ids=model_inputs.mm_token_type_ids,
        pixel_values=model_inputs.pixel_values,
        is_first_iteration=True,
    )
    full_mask = mask_mapping['full_attention']
    sliding_mask = mask_mapping['sliding_attention']
    print('hf full mask', tuple(full_mask.shape), full_mask.dtype)
    print('hf sliding mask', tuple(sliding_mask.shape), sliding_mask.dtype)

    data_processor = Gemma4DataPreprocess(
        token_embedding=model.get_input_embeddings(),
        input_sequence_length=model_inputs.input_ids.shape[1],
        context_length=model_inputs.input_ids.shape[1],
        past_key_caches=[],
        past_value_caches=[],
        pad_token_id=model.config.text_config.pad_token_id,
        image_token_id=model.config.image_token_id,
        sliding_window=model.config.text_config.sliding_window,
    )
    data_processor.to(model.device, torch.bfloat16)
    _, _, _, xh_full_mask, xh_sliding_mask, _, _ = data_processor(
        {
            'input_ids': model_inputs.input_ids,
            'image_embeds': image_features,
            'mm_token_type_ids': model_inputs.mm_token_type_ids,
            'past_seq_length': 0,
        }
    )
    print('xh full mask', tuple(xh_full_mask.shape), xh_full_mask.dtype)
    print('xh sliding mask', tuple(xh_sliding_mask.shape), xh_sliding_mask.dtype)

    if spans:
        s, e = spans[0]
        print('hf full mask vision block sample', full_mask[0, 0, s, s:e].detach().float().cpu().tolist()[:8])
        print('hf sliding mask vision block sample', sliding_mask[0, 0, s, s:e].detach().float().cpu().tolist()[:8])
        print('xh full mask vision block sample', xh_full_mask[0, 0, s, s:e].detach().float().cpu().tolist()[:8])
        print('xh sliding mask vision block sample', xh_sliding_mask[0, 0, s, s:e].detach().float().cpu().tolist()[:8])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./weights/gemma-4-31B-it')
    parser.add_argument('--image-path', type=str, default='data/images/qwen2_vl_demo.jpeg')
    parser.add_argument('--prompt', type=str, default='Describe this image.')
    parser.add_argument('--device-map', type=str, default='auto')
    args = parser.parse_args()
    main(args)
