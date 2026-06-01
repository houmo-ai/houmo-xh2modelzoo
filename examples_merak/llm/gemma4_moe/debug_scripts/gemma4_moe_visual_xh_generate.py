from __future__ import annotations

import argparse
from PIL import Image

from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
from xhquant.api import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-type", choices=["wrap", "fronted", "quanted_fast"], default="wrap")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    model = AutoLLMModel.from_pretrained(model_cfg)
    dummy = model.get_dummy_inputs()
    processor = model.get_tf_processor()
    processed = processor(images=Image.new("RGB", (model.config.max_size_w, model.config.max_size_h)))
    assert dummy["image"].shape == processed["pixel_values"].shape
    print("Task12 visual debug OK", args.eval_type, dummy["image"].shape)


if __name__ == "__main__":
    main()