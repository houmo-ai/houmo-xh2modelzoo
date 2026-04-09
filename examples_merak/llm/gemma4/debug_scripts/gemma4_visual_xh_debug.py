import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, LLMInferenceContextManager, LLMModelState
from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.gemma4 import XHGemma4VisionModel, XHGemma4VisualConfig


def main(args):
    cfg_name = Path(args.config).stem
    work_dir = Path('./work_dirs') / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / f'visual_{args.eval_type}.log'), args.debug)
    seed = 1024
    set_random_seed(seed)
    logger = get_xhquant_logger()

    cfg = Config.fromfile(args.config)
    cfg.seed = seed
    cfg.dump(work_dir / Path(args.config).name)

    model_cfg: XHGemma4VisualConfig = AutoLLMConfig.from_pretrained(cfg.model)
    model_cfg.work_dir = str(work_dir)
    xh_model: XHGemma4VisionModel = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.set_state(LLMModelState.from_string(args.eval_type))

    messages = [{"role": "user", "content": [{"type": "image", "image": str(Path(args.image_path).resolve())}, {"type": "text", "text": args.prompt}]}]
    processor = xh_model.get_tf_processor()
    model_inputs = processor.apply_chat_template(messages)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    contexts = [TimeProfiler('vision_debug', logger), MemoryTracker(device=device, name='vision_debug', logger=logger), LLMInferenceContextManager(xh_model), torch.no_grad()]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        image_embeds = xh_model(
            model_inputs['pixel_values'].to(device=device, dtype=dtype),
            model_inputs['image_position_ids'].to(device),
        )
    if isinstance(image_embeds, (tuple, list)):
        image_embeds = image_embeds[0]
    logger.info(f'image_embeds shape: {tuple(image_embeds.shape)}')
    logger.info(f'image_embeds dtype: {image_embeds.dtype}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_visual_xh2a_2k.py')
    parser.add_argument('--eval-type', type=str, default='wrap', choices=LLMModelState.get_all_values())
    parser.add_argument('--image-path', type=str, default='data/images/qwen2_vl_demo.jpeg')
    parser.add_argument('--prompt', type=str, default='Describe this image.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
