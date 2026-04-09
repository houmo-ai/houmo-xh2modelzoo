import argparse
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText

from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhmodel_merak.xh_llm.models.gemma4.gemma4_processor import XHGemma4Processor


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
        torch_dtype=torch.bfloat16 if 'cuda' in device else torch.float32,
        device_map={'': device},
    ).eval()
    xh_model.visual.init_wrap_model(hf_model)
    processor = XHGemma4Processor.from_pretrained(model_cfg.hf_model)
    inputs = processor.apply_chat_template([
        {'role': 'user', 'content': [
            {'type': 'image', 'image': Image.new('RGB', (224, 224), color='white')},
            {'type': 'text', 'text': 'Describe this image.'},
        ]}
    ])
    pixel_values = inputs['pixel_values'].to(device)
    image_position_ids = inputs['image_position_ids'].to(device)
    with torch.no_grad():
        hf_features = hf_model.model.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
        wrap_features = xh_model.visual.wrap_model(pixel_values, image_position_ids)
    print('hf_features', tuple(hf_features.shape))
    print('wrap_features', tuple(wrap_features.shape))
    print('cosine', torch.nn.functional.cosine_similarity(hf_features.flatten(), wrap_features.flatten(), dim=0).item())


if __name__ == '__main__':
    main()
