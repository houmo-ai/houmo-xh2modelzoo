from functools import partial
from types import MethodType
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModel, AutoProcessor

from xh_model_zoo.utils.image import impad, imrescale
from ..base_model import BaseModel
from ..builder import MODELS, wrap_llm_model


def _get_sliced_images(self, image, max_slice_nums=None, rescale_size=None):
    slice_images = self._old_get_sliced_images(image, max_slice_nums)
    img_max_w, img_max_h = rescale_size
    if rescale_size is not None:
        for i, image in enumerate(slice_images):

            img = np.array(image)
            img, scale_factor = imrescale(
                img,
                (img_max_w, img_max_h),
                interpolation="bilinear",
                return_scale=True,
                backend="cv2",
            )
            pad_img = impad(img, shape=(img_max_h, img_max_w), pad_val=0)
            image = Image.fromarray(pad_img)
            slice_images[i] = image

    return slice_images


def _audio_feature_extract_fixed_length(self, *args, **kwargs):
    audio_features, audio_feature_lens_list, audio_ph_list = self._old_audio_feature_extract(*args, **kwargs)
    # self.feature_extractor.nb_max_frames = 3000
    audio_features = torch.nn.functional.pad(
        audio_features,
        (0, self.feature_extractor.nb_max_frames - audio_features.shape[-1]),
        mode="constant",
        value=0,
    )
    return audio_features, audio_feature_lens_list, audio_ph_list


class XHMiniCPMOBaseModel(BaseModel):
    def get_hf_model(self, device_map="cpu", **kwargs):
        assert self.hf_model_dir is not None
        hf_model = AutoModel.from_pretrained(
            self.hf_model_dir,
            trust_remote_code=True,
            attn_implementation="sdpa",  # sdpa or flash_attention_2
            torch_dtype=torch.float16,
            init_vision=True,
            init_audio=True,
            init_tts=True,
            device_map=device_map,
        ).eval()
        self.patch_size = hf_model.config.patch_size
        processor = AutoProcessor.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        hf_model.processor = processor

        vpm = hf_model.vpm
        self.patch_size = vpm.embeddings.patch_size
        self.num_patches_per_side = vpm.embeddings.num_patches_per_side

        return hf_model

    def wrap_processor(self, hf_model):
        image_slice_max_size = self.wrap_cfg.image_slice_max_size
        image_processor = hf_model.processor.image_processor
        image_processor._old_get_sliced_images = image_processor.get_sliced_images
        image_processor.get_sliced_images = MethodType(
            partial(
                _get_sliced_images,
                rescale_size=[image_slice_max_size[0] * self.patch_size, image_slice_max_size[1] * self.patch_size],
            ),
            image_processor,
        )
        processor = hf_model.processor
        processor._old_audio_feature_extract = processor.audio_feature_extract
        processor.audio_feature_extract = MethodType(_audio_feature_extract_fixed_length, processor)
        return hf_model

    def unwrap_processor(self, hf_model):
        if hasattr(hf_model.processor.image_processor, "_old_get_sliced_images"):
            image_processor = hf_model.processor.image_processor
            image_processor.get_sliced_images = image_processor._old_get_sliced_images
            del image_processor._old_get_sliced_images

        if hasattr(hf_model.processor, "_old_audio_feature_extract"):
            processor = hf_model.processor
            processor.audio_feature_extract = processor._old_audio_feature_extract
            del processor._old_audio_feature_extract

        return hf_model
