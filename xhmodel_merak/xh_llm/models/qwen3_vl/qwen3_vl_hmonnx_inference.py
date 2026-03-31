import torch

from xhmodel_merak.xh_llm.utils import unfold_args

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import LLMModelMeta
from .data_preprocess import Qwen3VLDataPreprocess
from .qwen3_vl_processor import XHQwen3VLProcessor


class VisualHMONNXModel(HMONNXModel):
    def forward(self, *args):
        out = super().forward(*args)
        image_embeds_i, *deepstack_image_embeds = out
        out = (image_embeds_i, deepstack_image_embeds)
        return out


class XHQwen3VLHMONNXModel(VisonLLMHMONNXModel):
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        self.visual = VisualHMONNXModel(self.visual_meta.hmonnx)

    def _set_device(self, device):
        super()._set_device(device)
        self.visual.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def to_fast(self):
        self.visual.to_fast()
        super().to_fast()
        return self

    def get_tf_processor(self):
        processor = XHQwen3VLProcessor.from_pretrained(self.hf_model_dir)
        meta_info = self.meta_info.model_config
        processor.config.patch_size = meta_info.visual_config.patch_size
        processor.config.max_size_h = meta_info.visual_config.max_size_h
        processor.config.max_size_w = meta_info.visual_config.max_size_w
        return processor

    def forward(self, *args):
        args = unfold_args(args)
        args = [arg.to(torch.int32) if arg.dtype == torch.int64 else arg for arg in args]
        return super().forward(*args)

    def _get_data_preprocessor(self) -> Qwen3VLDataPreprocess:
        input_sequence_length = self.get_input_sequence_length()
        data_preprocess = Qwen3VLDataPreprocess(
            token_embedding=self.get_input_embeddings(),
            input_sequence_length=input_sequence_length,
            image_size_w=self.visual_meta.max_size_w,
            image_size_h=self.visual_meta.max_size_h,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            patch_size=self.visual_meta.patch_size,
            image_token_id=self.meta_info.image_token_id,
            video_token_id=self.meta_info.video_token_id,
            vision_start_token_id=self.meta_info.vision_start_token_id,
            vision_end_token_id=self.meta_info.vision_end_token_id,
            spatial_merge_size=self.meta_info.spatial_merge_size,
        )
        data_preprocess.to(self._device, self._dtype)
        return data_preprocess
