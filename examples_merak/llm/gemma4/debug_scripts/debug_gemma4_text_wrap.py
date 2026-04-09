import argparse
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText

from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhmodel_merak.xh_llm.models.gemma4.gemma4_processor import XHGemma4Processor


def build_image_embeds(hf_model, inputs):
    with torch.no_grad():
        image_features = hf_model.model.get_image_features(
            inputs['pixel_values'],
            inputs['image_position_ids'],
            return_dict=True,
        ).pooler_output
    return image_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(model_cfg)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.hf_model,
        trust_remote_code=True,
        dtype=torch.bfloat16 if 'cuda' in device else torch.float32,
        device_map={'': device},
    ).eval()
    processor = XHGemma4Processor.from_pretrained(model_cfg.hf_model)
    # For text wrap test, use text-only mode to keep sequence length reasonable
    inputs = processor.apply_chat_template([
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Hello world, this is a test prompt.'},
        ]}
    ])
    inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    # Create dummy image embeds for text wrap testing
    # Since we're testing text wrap, we can skip actual image processing
    # Create dummy image_embeds with shape [batch, num_image_tokens, hidden_dim]
    image_embeds = torch.zeros(inputs['input_ids'].shape[0], 0, 3072, dtype=torch.bfloat16, device=device)
    
    # Run HF forward BEFORE wrapping to get reference outputs
    print("Running HF model forward...")
    with torch.no_grad():
        hf_out = hf_model(**inputs, use_cache=False, logits_to_keep=1)
    
    # Now initialize wrap model
    xh_model.init_wrap_model(hf_model)
    xh_model._device = device
    xh_model._dtype = torch.bfloat16

    xh_model.get_kvcache_mixin().prepare_kv_cache(dtype=torch.bfloat16)
    data = {
        'input_ids': inputs['input_ids'],
        'mm_token_type_ids': inputs.get('mm_token_type_ids'),
        'image_embeds': image_embeds,
        'past_seq_length': 0,
    }
    wrap_inputs = xh_model.get_data_preprocessor()(data)
    with torch.no_grad():
        wrap_logits = xh_model.wrap_model(*wrap_inputs)
    print('hf logits', tuple(hf_out.logits.shape))
    print('wrap logits', tuple(wrap_logits.shape))
    hf_last = hf_out.logits[:, -1, :].float()
    wrap_last = wrap_logits[:, data['input_ids'].shape[1] - 1, :].float()
    cosine = torch.nn.functional.cosine_similarity(hf_last.flatten(), wrap_last.flatten(), dim=0).item()
    print('last-token cosine', cosine)


if __name__ == '__main__':
    main()
