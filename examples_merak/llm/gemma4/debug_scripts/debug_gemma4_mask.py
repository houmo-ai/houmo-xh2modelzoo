import argparse
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping


def first_vision_span(mm_token_type_ids: torch.Tensor):
    mm = mm_token_type_ids[0]
    is_vision = (mm == 1) | (mm == 2)
    idx = torch.nonzero(is_vision, as_tuple=False).flatten()
    if idx.numel() == 0:
        return None
    start = idx[0].item()
    end = start
    while end + 1 < mm.numel() and bool(is_vision[end + 1]):
        end += 1
    return start, end + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='./weights/gemma-4-31B-it')
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': Image.new('RGB', (224, 224), color='white')},
                {'type': 'text', 'text': 'Describe this image in one short sentence.'},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )
    hidden_size = config.text_config.hidden_size if isinstance(config.text_config, dict) else config.text_config.hidden_size
    inputs_embeds = torch.zeros((inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], hidden_size), dtype=torch.float32)
    position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0)
    masks = create_causal_mask_mapping(
        config,
        inputs_embeds,
        inputs['attention_mask'],
        None,
        position_ids,
        inputs['mm_token_type_ids'],
        inputs.get('pixel_values'),
        is_training=False,
    )
    span = first_vision_span(inputs['mm_token_type_ids'])
    print('mask keys:', list(masks.keys()))
    print('input_ids:', tuple(inputs['input_ids'].shape))
    print('mm_token_type_ids unique:', torch.unique(inputs['mm_token_type_ids']).tolist())
    print('vision span:', span)
    for name, mask in masks.items():
        if mask is None:
            print(name, None)
            continue
        print(name, tuple(mask.shape), mask.dtype, float(mask.max()), float(mask.min()))
        if span is not None:
            s, e = span
            block = mask[0, 0, s:e, s:e]
            visible = int((block == 0).sum().item())
            total = block.numel()
            print(f'{name} vision_block_visible={visible}/{total}')

if __name__ == '__main__':
    main()
