import random
from pathlib import Path
from typing import List

from xhquant.utils import fileio
from xhquant.utils.fileio import get_file_backend

from ..registry import DATASETS
from .base_dataset import BaseDataset


def get_img_id_to_img_path(annotations):
    img_id_to_img_path = {}
    for img_info in annotations["images"]:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_id_to_img_path[img_id] = file_name

    return img_id_to_img_path


def get_img_id_to_captions(annotations):
    img_id_to_captions = {}
    for caption_info in annotations["annotations"]:
        img_id = caption_info["image_id"]
        if img_id not in img_id_to_captions:
            img_id_to_captions[img_id] = []

        caption = caption_info["caption"]
        img_id_to_captions[img_id].append(caption)

    return img_id_to_captions


@DATASETS.register_module()
class COCOCaption(BaseDataset):
    """COCO Caption dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``..
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix["img_path"]
        annotations = fileio.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)
        img_id_to_filename = get_img_id_to_img_path(annotations)
        img_id_to_captions = get_img_id_to_captions(annotations)
        img_ids = list(img_id_to_filename.keys())

        data_list = []

        for img_id in img_ids:
            img_filename = img_id_to_filename[img_id]
            img_path = file_backend.join_path(img_prefix, img_filename)
            captions = img_id_to_captions[img_id]
            data_info = {
                "image_id": img_id,
                "img_path": str(img_path),
                "gt_caption": captions,
            }
            data_list.append(data_info)

        return data_list
