import argparse
import os
from pathlib import Path

from xhquant.api import DeviceType
from xhquant.api import QuantScheme
from xhquant.api import xhquant_init
from xhquant.utils import set_random_seed

from xh_model_zoo.xh_aigc.models.sd3_custom_a import SD3CustomAConvertConfig
from xh_model_zoo.xh_aigc.models.sd3_custom_a import SD3CustomAConverter


def main(args):
    model = args.model
    model_dir = os.path.normpath(model)
    custom_a_model_dir = args.custom_a_model
    custom_a_model_dir = os.path.normpath(custom_a_model_dir)
    model_name = Path(custom_a_model_dir).name
    target_device = DeviceType.XH2a
    height = args.height
    width = args.width
    cfg_name = f"{model_name}_{target_device.name}_{width}x{height}"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}.log"
    xhquant_init(log_file, debug=False)

    target_device = DeviceType.XH2a
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    convert_config = SD3CustomAConvertConfig(
        quant_scheme=quant_scheme,
        guidance_scale=args.guidance_scale,
        num_inference_steps=8,
        height=height,
        width=width,
    )

    SD3CustomAConverter.from_pretrained(model_dir, convert_config, str(work_dir), custom_a_model=custom_a_model_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--custom-a-model", type=str, default="data/models/sd3_2b_custom_a")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.5,
        help="Seed for the random number generator",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed, deterministic=False)
    main(args)
