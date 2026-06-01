from xhmodel_merak.xh_llm.llm_data_processor import BaseVisualProcessor

from .base_model import XHSubModel
from .kv_cache_mixin import EmptyKVCacheMixin


class BaseVisionModel(XHSubModel):
    def _to_wrap(self, hf_model):
        # 包装模型，准备进行转换
        self.init_wrap_model(hf_model)

    def _get_data_preprocessor(self) -> BaseVisualProcessor:
        preprocessor = BaseVisualProcessor()
        return preprocessor

    def get_kvcache_mixin(self):
        return EmptyKVCacheMixin()
