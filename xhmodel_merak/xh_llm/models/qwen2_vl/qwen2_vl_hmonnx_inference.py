import torch

from xhmodel_merak.xh_llm.utils import unfold_args

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import LLMModelMeta
from .data_preprocess import Qwen2VLDataPreprocess
from .qwen2_vl_processor import XHQwen2VLProcessor


class XHQwen2VLHMONNXModel(VisonLLMHMONNXModel):
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        # LLM and visual HMONNX files are exported separately, but inference
        # presents them as one VLM so generation can call visual.forward first.
        self.visual_meta = meta_info.visual_config
        self.visual = HMONNXModel(self.visual_meta.hmonnx)

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
        processor = XHQwen2VLProcessor.from_pretrained(self.hf_model_dir)
        visual_config = self.meta_info.model_config.visual_config
        processor.config.patch_size = visual_config.patch_size
        processor.config.max_size_h = visual_config.max_size_h
        processor.config.max_size_w = visual_config.max_size_w
        return processor

    def forward(self, *args):
        args = unfold_args(args)
        # HMONNX kernels expect int32 shape/index tensors; HF processors often
        # produce int64 ids and grids by default.
        args = [arg.to(torch.int32) if arg.dtype == torch.int64 else arg for arg in args]
        return super().forward(*args)

    def _get_data_preprocessor(self) -> Qwen2VLDataPreprocess:
        data_preprocess = Qwen2VLDataPreprocess(
            token_embedding=self.get_input_embeddings(),
            input_sequence_length=self.get_input_sequence_length(),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            image_token_id=self.meta_info.image_token_id,
            video_token_id=self.meta_info.video_token_id,
            vision_start_token_id=self.meta_info.vision_start_token_id,
            vision_end_token_id=self.meta_info.vision_end_token_id,
            spatial_merge_size=self.meta_info.spatial_merge_size,
        )
        data_preprocess.to(self._device, self._dtype)
        return data_preprocess
