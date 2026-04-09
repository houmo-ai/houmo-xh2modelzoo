import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4 import XHGemma4Model, XHGemma4ModelConfig


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path('./work_dirs') / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / f'llm_{args.eval_type}.log'), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    if args.only_first_block:
        cfg.model.only_first_block = True
    cfg.dump(work_dir / Path(args.config).name)

    model_cfg: XHGemma4ModelConfig = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model: XHGemma4Model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

    processor = AutoProcessor.from_pretrained(model_cfg.hf_model, trust_remote_code=True)
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_cfg.hf_model,
        dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    ).eval()

    messages = [{"role": "user", "content": [{"type": "image", "image": str(Path(args.image_path).resolve())}, {"type": "text", "text": args.prompt}]}]
    model_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
        enable_thinking=False,
    )
    model_inputs = model_inputs.to(hf_model.device)

    with torch.no_grad():
        image_features = hf_model.model.get_image_features(
            model_inputs.pixel_values,
            model_inputs.image_position_ids,
            return_dict=True,
        ).pooler_output
    debug_batch = {
        'input_ids': model_inputs.input_ids,
        'mm_token_type_ids': model_inputs.mm_token_type_ids,
        'image_embeds': image_features,
        'past_seq_length': 0,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    contexts = [TimeProfiler('llm_debug', logger), MemoryTracker(device=device, name='llm_debug', logger=logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=torch.float16)
        data_processor = xh_model.get_data_preprocessor()
        processed_inputs = data_processor(debug_batch)
        logits = xh_model(*processed_inputs)
    logger.info(f'processed input count: {len(processed_inputs)}')
    logger.info(f'logits shape: {tuple(logits.shape)}')
    logger.info(f'logits dtype: {logits.dtype}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py')
    parser.add_argument('--eval-type', type=str, default='wrap', choices=LLMModelState.get_all_values())
    parser.add_argument('--image-path', type=str, default='data/images/qwen2_vl_demo.jpeg')
    parser.add_argument('--prompt', type=str, default='Describe this image.')
    parser.add_argument('--only-first-block', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
