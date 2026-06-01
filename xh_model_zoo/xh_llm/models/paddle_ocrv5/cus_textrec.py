# Copyright 2025 HOUMO AI
#
# File: cus_textrec.py
# Description:
#   Custom PaddleOCRv5 text recognition predictor wrapper for xh2modelzoo inference.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from paddlex.inference.models.text_recognition.predictor import TextRecPredictor


class Cus_TextRecPredictor(TextRecPredictor):
    entities = []

    def __init__(self, *args, **kwargs):
        pass

    def __setup__(self, rec_model):
        """"""
        """
        初始化模型
        """
        self.rec_model = rec_model
        return self

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        rec_model = None,
        fixed_shape = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        cls.fixed_shape = fixed_shape
        hf_model.__class__ = cls
        hf_model.__setup__(rec_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return hf_model

    def process(self, batch_data, return_word_box=False):
        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        
        # 计算每张图片的宽高比
        width_list = []
        for img in batch_raw_imgs:
            width_list.append(img.shape[1] / float(img.shape[0]))
        
        # 根据宽高比分组并设置目标宽度
        if self.fixed_shape is not None:
            # 使用固定尺寸
            original_input_shape = self.pre_tfs["ReisizeNorm"].input_shape
            self.pre_tfs["ReisizeNorm"].input_shape = self.fixed_shape
        else:
            # 根据宽高比动态调整尺寸
            max_ratio = max(width_list)
            if max_ratio <= 3:
                target_width = 320
            elif max_ratio <= 6:
                target_width = 640
            elif max_ratio <= 10:
                target_width = 960
            else:
                target_width = 1280
            
            # 临时修改 rec_image_shape
            original_rec_image_shape = self.pre_tfs["ReisizeNorm"].rec_image_shape
            self.pre_tfs["ReisizeNorm"].rec_image_shape = [3, 48, target_width]
        
        indices = np.argsort(np.array(width_list))
        batch_imgs = self.pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        
        # 恢复原始配置
        if self.fixed_shape is not None:
            self.pre_tfs["ReisizeNorm"].input_shape = original_input_shape
        else:
            self.pre_tfs["ReisizeNorm"].rec_image_shape = original_rec_image_shape
        
        if self._use_static_model:
            if x[0].shape[-1] != 320:
                batch_preds = self.infer(x=x)
            else:
                inp = torch.from_numpy(x[0]).half().cuda()
                batch_preds = self.rec_model(inp)
                batch_preds = batch_preds.cpu().numpy().astype(np.float32)
                batch_preds = [batch_preds]
        else:
            with TemporaryDeviceChanger(self.device):
                batch_preds = self.infer(x=x)
        batch_num = self.batch_sampler.batch_size
        img_num = len(batch_raw_imgs)
        rec_image_shape = next(
            op["RecResizeImg"]["image_shape"]
            for op in self.config["PreProcess"]["transform_ops"]
            if "RecResizeImg" in op
        )
        imgC, imgH, imgW = rec_image_shape[:3]
        max_wh_ratio = imgW / imgH
        end_img_no = min(img_num, batch_num)
        wh_ratio_list = []
        for ino in range(0, end_img_no):
            h, w = batch_raw_imgs[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            wh_ratio_list.append(wh_ratio)
        texts, scores = self.post_op(
            batch_preds,
            return_word_box=return_word_box or self.return_word_box,
            wh_ratio_list=wh_ratio_list,
            max_wh_ratio=max_wh_ratio,
        )
        if self.model_name in (
            "arabic_PP-OCRv3_mobile_rec",
            "arabic_PP-OCRv5_mobile_rec",
        ):
            texts = [get_display(s) for s in texts]
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "rec_text": texts,
            "rec_score": scores,
            "vis_font": [self.vis_font] * len(batch_raw_imgs),
        }
