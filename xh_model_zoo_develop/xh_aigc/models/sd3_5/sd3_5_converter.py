import transformers
from xhquant.utils import digit_version

from ..sd3 import SD3ConvertConfig, SD3Converter


class SD3_5_Converter(SD3Converter):
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, convert_config: SD3ConvertConfig, work_dir: str, **kwargs):
        assert digit_version(transformers.__version__) >= digit_version(
            "4.47.0"
        ), "transformers version must be >= 4.47.0"
        converter = SD3Converter(pretrained_model_path, convert_config)
        converter._convert(work_dir)
