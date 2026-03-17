# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union
import numpy as np
import torch
from paddlex.inference.models.text_detection.predictor import TextDetPredictor


class Cus_TextDetPredictor(TextDetPredictor):
    entities = []
    def __init__(self, *args, **kwargs):
        pass

    def __setup__(self, det_model):
        """"""
        """
        初始化模型
        """
        self.det_model = det_model
        return self

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        det_model = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        hf_model.__class__ = cls
        hf_model.__setup__(det_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return hf_model
    
    def process(
        self,
        batch_data: List[Union[str, np.ndarray]],
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        unclip_ratio: Union[float, None] = None,
        max_side_limit: Union[int, None] = None,
    ):

        batch_raw_imgs = self.pre_tfs["Read"](imgs=batch_data.instances)
        batch_imgs, batch_shapes = self.pre_tfs["Resize"](
            imgs=batch_raw_imgs,
            limit_side_len=limit_side_len or self.limit_side_len,
            limit_type=limit_type or self.limit_type,
            max_side_limit=(
                max_side_limit if max_side_limit is not None else self.max_side_limit
            ),
        )
        batch_imgs = self.pre_tfs["Normalize"](imgs=batch_imgs)
        batch_imgs = self.pre_tfs["ToCHW"](imgs=batch_imgs)
        x = self.pre_tfs["ToBatch"](imgs=batch_imgs)
        
        if self._use_static_model:
            # batch_preds = self.infer(x=x) # 1, 3, 512, 896
            inp = torch.from_numpy(x[0]).half().cuda()
            batch_preds = self.det_model(inp)
            batch_preds = batch_preds.cpu().numpy().astype(np.float32)
            batch_preds = [batch_preds]
        else:
            with TemporaryDeviceChanger(self.device):
                batch_preds = self.infer(x=x)
        polys, scores = self.post_op(
            batch_preds,
            batch_shapes,
            thresh=thresh or self.thresh,
            box_thresh=box_thresh or self.box_thresh,
            unclip_ratio=unclip_ratio or self.unclip_ratio,
        )
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "dt_polys": polys,
            "dt_scores": scores,
        }    
