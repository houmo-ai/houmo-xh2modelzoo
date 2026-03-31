from collections import OrderedDict

# from .scan_model_types import extract_register_llm_model_args, is_register_llm_model_call
from .scan_model_types import get_support_all_model_types


## model type to module mapping, used for auto loading model class
MODEL_TYPE_MAPPING_MODULES: OrderedDict[str, str] = get_support_all_model_types()
