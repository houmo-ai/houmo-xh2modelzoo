from functools import partial
from types import MethodType

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor

from xhmodel_merak.xh_other_model.base_model import BaseModel


def imrescale(img, size, interpolation="bilinear", return_scale=False, backend="cv2"):
    import cv2

    del backend
    height, width = img.shape[:2]
    scale = min(size[0] / width, size[1] / height)
    resized = cv2.resize(img, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_LINEAR)
    return (resized, scale) if return_scale else resized


def impad(img, shape, pad_val=0):
    padded = np.full((*shape, img.shape[2]), pad_val, dtype=img.dtype)
    padded[: img.shape[0], : img.shape[1]] = img
    return padded


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
    feature_extractor = getattr(self, "feature_extractor", None)
    if feature_extractor is None:
        feature_extractor = getattr(self, "audio_feature_extractor", None)
    if feature_extractor is None or not hasattr(feature_extractor, "nb_max_frames"):
        return audio_features, audio_feature_lens_list, audio_ph_list

    pad_width = int(feature_extractor.nb_max_frames) - int(audio_features.shape[-1])
    if pad_width > 0:
        audio_features = torch.nn.functional.pad(
            audio_features,
            (0, pad_width),
            mode="constant",
            value=0,
        )
    elif pad_width < 0:
        audio_features = audio_features[..., : int(feature_extractor.nb_max_frames)]

    return audio_features, audio_feature_lens_list, audio_ph_list


class XHMiniCPMOBaseModel(BaseModel):
    def _init_wrap_model_with_llm_registry(self, hf_model: nn.Module) -> nn.Module:
        """Wrap a component through the xh_llm trace registry.

        The component classes retain the legacy lifecycle implementation for
        now, but their wrappers are registered in the shared
        ``XHLLM_TRACEABLE_MODULES`` registry.  Calling the canonical xh_llm
        helper explicitly keeps the registry used for conversion and the
        registry used by the wrappers in sync without changing the shared
        other-model implementation.
        """
        from ...wrap_model import wrap_llm_model

        self._wrap_model = wrap_llm_model(hf_model, self.wrap_cfg)
        return self._wrap_model

    def get_hf_model(self, device_map="cpu", **kwargs):
        assert self.hf_model_dir is not None
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        from .hf_compatible import TTS_SAMPLING_DEFAULTS

        for key, value in TTS_SAMPLING_DEFAULTS.items():
            if not hasattr(config.tts_config, key):
                setattr(config.tts_config, key, value)
        from transformers import PreTrainedModel

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

            def _get_tied_keys(self):
                return self.__dict__.get(
                    "_xh_all_tied_weights_keys",
                    {key: key for key in (getattr(self, "_tied_weights_keys", None) or ())},
                )

            def _set_tied_keys(self, value):
                self.__dict__["_xh_all_tied_weights_keys"] = value

            PreTrainedModel.all_tied_weights_keys = property(_get_tied_keys, _set_tied_keys)
        hf_model = AutoModel.from_pretrained(
            self.hf_model_dir,
            config=config,
            trust_remote_code=True,
            attn_implementation="sdpa",  # sdpa or flash_attention_2
            torch_dtype=torch.float16,
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
