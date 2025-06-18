import argparse
import os
from pathlib import Path

from xhquant.api import DeviceType, QuantScheme, xhquant_init
from xhquant.utils import set_random_seed

from xh_model_zoo.xh_aigc.models.sd3_lenovo import SD3LenovoConvertConfig, SD3LenovoConverter


def main(args):
    model = args.model
    model_dir = os.path.normpath(model)
    lenovo_model_dir = args.lenovo_model
    lenovo_model_dir = os.path.normpath(lenovo_model_dir)
    model_name = Path(lenovo_model_dir).name
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
    convert_config = SD3LenovoConvertConfig(
        quant_scheme=quant_scheme,
        guidance_scale=args.guidance_scale,
        height=height,
        width=width,
    )

    SD3LenovoConverter.from_pretrained(model_dir, convert_config, str(work_dir), lenovo_model=lenovo_model_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--lenovo-model", type=str, default="data/models/sd3_2b_lenovo")
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
