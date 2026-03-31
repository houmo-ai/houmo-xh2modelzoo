from collections import OrderedDict

from .scan_model_types import get_support_master_model_types


## model type to module mapping, used for auto loading model class
support_llm_model_types: OrderedDict[str, str] = get_support_master_model_types()
