from pathlib import Path
from typing import List

import torch

from ..registry import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class SDXLUNetCalibDataset(BaseDataset):
    METAINFO = {
        "classes": [],
        "palette": [],
    }

    def __init__(self, data_root, pipeline=None, test_mode=True, **kwargs):
        super().__init__(
            ann_file=None, metainfo=None, data_root=data_root, pipeline=pipeline, test_mode=test_mode, **kwargs
        )
        # all_files = Path(data_root).glob("*.*")
        # unet_datas = list()
        # for index, file in enumerate(all_files):
        #     unet_data = torch.load(file, "cpu", weights_only=True)

    def load_data_list(self) -> List[dict]:
        all_files = Path(self.data_root).glob("*.*")
        unet_datas = list()
        data_list = []
        for index, file in enumerate(all_files):
            unet_data = torch.load(file, "cpu", weights_only=True)
            data_list.append(
                {
                    "input": unet_data[0].to(torch.float32).squeeze(0),
                    "input_9": unet_data[1].to(torch.float32).squeeze(0),
                    "encoder_hidden_states": unet_data[2].to(torch.float32).squeeze(0),
                }
            )
        return data_list
