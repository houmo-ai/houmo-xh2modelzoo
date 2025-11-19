from pathlib import Path
from typing import List

import numpy as np
import torch

from ..registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class SDXLVAECalibDataset(BaseDataset):
    def __init__(self, data_root, pipeline=None, test_mode=True, **kwargs):
        super().__init__(
            ann_file=None, metainfo=None, data_root=data_root, pipeline=pipeline, test_mode=test_mode, **kwargs
        )

    def load_data_list(self) -> List[dict]:
        all_files = Path(self.data_root).glob("*.npy")
        data_list = []
        for index, data_path in enumerate(all_files):
            clip_data = torch.from_numpy(np.load(data_path)).to(torch.float32)
            clip_data = clip_data.squeeze(0)
            data_list.append(
                {
                    "input": clip_data,
                }
            )
        return data_list
