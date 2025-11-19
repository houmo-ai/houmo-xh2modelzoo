import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import torchvision.datasets as dset
from PIL import Image

from xh2_model_zoo.datasets.Coco.evaluate import COCO80_NAMES

from ..data_path import (
    COCO_DATASET_TRAIN_PATH,
    COCO_DATASET_VAL_PATH,
    COCO_JSON_TRAIN_PATH,
    COCO_JSON_VAL_PATH,
)
from ..Imagenet.classic_classification import LoaderGenerator
from .evaluate import Coco

dev_path = str(Path(__file__).parent.resolve().parent.parent) + "/"


class CocoLoaderGenerator(LoaderGenerator):
    def __init__(self, *args, root=None, annFile=None, **kwargs):
        # 目前只是适用于yolov3
        super().__init__(*args, **kwargs)
        self.root = root if root is not None else COCO_DATASET_VAL_PATH
        self.annFile = annFile if annFile is not None else COCO_JSON_VAL_PATH

    def load(self):
        if self.calib_transform is None:
            self.calib_transform = None
        if self.test_transform is None:
            self.test_transform = None

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = dset.CocoDetection(root=self.root, annFile=self.annFile)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            if self.with_orishape:
                self._test_set = CocoWithShape(
                    root=self.root,
                    annFile=self.annFile,
                    transform=self.test_transform,
                )
            else:
                self._test_set = dset.CocoDetection(
                    root=self.root,
                    annFile=self.annFile,
                    transform=self.test_transform,
                )
        return self._test_set

    @property
    def calib_set(self):
        if self._calib_set is None:
            dataset = dset.CocoDetection(
                root=self.root,
                annFile=self.annFile,
                transform=self.calib_transform,
            )
            if self.class_balance:
                self._calib_set = CocoClassBalancedDataset(dataset, 0.1)
            else:
                self._calib_set = dataset
        return self._calib_set

    def evaluate(self, batches_pred, image_ids, resType="bbox"):
        assert self.test_dataset is not None
        evaluator = Coco(self.annFile, resType=resType)
        res = evaluator.run(batches_pred, imgIds=image_ids)
        return res


class FixedCocoDetection(dset.CocoDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return os.path.join(self.root, path)

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if len(target) == 0:  # add image_id if there is no box in the iamge
            target.append(dict(image_id=id))

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class CustomCocoDetection(dset.CocoDetection):
    def __init__(self, *args, im_file_paths, labels, **kwargs):
        # 目前只是适用于yolov3
        super().__init__(*args, **kwargs)
        self.im_files = im_file_paths
        self.labels = labels
        self.ids = [i for i in range(5000)]

    def _load_image(self, id: int):
        img_path = self.im_files[id]
        return Image.open(os.path.join(img_path)).convert("RGB")

    def _load_target(self, id: int):
        return self.labels[id]

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target, self.im_files[id]


class Cus_CocoLoaderGenerator(CocoLoaderGenerator):
    def __init__(self, *args, im_file_paths, labels, **kwargs):
        # 目前只是适用于yolov3
        super().__init__(*args, **kwargs)
        self.im_files = im_file_paths
        self.labels = labels

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = CustomCocoDetection(
                root=COCO_DATASET_VAL_PATH,
                annFile=COCO_JSON_VAL_PATH,
                im_file_paths=self.im_files,
                labels=self.labels,
            )
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = CustomCocoDetection(
                root=COCO_DATASET_VAL_PATH,
                annFile=COCO_JSON_VAL_PATH,
                transform=self.test_transform,
                im_file_paths=self.im_files,
                labels=self.labels,
            )
        return self._test_set

    @property
    def calib_set(self):
        if self._calib_set is None:
            self._calib_set = CustomCocoDetection(
                root=COCO_DATASET_VAL_PATH,
                annFile=COCO_JSON_VAL_PATH,
                transform=self.calib_transform,
                im_file_paths=self.im_files,
                labels=self.labels,
            )
        return self._calib_set


def eval_coco(batches_pred, image_ids, resType="bbox", val_json_path=None):
    evaluator = Coco(COCO_JSON_VAL_PATH if val_json_path is None else val_json_path, resType=resType)
    res = evaluator.run(batches_pred, imgIds=image_ids)
    return res


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        target["orig_size"] = [int(h), int(w)]
        return image, target


class CocoWithShape(dset.CocoDetection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare = ConvertCocoPolysToMask()

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        image = self._load_image(image_id)
        target = self._load_target(image_id)

        target = {"image_id": image_id, "annotations": target}
        image, target = self.prepare(image, target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


class CocoClassBalancedDataset:
    def __init__(self, dataset: dset.CocoDetection, oversample_thr: float = 0.1):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.cat_names = set(COCO80_NAMES)
        # Get repeat factors for each image.
        repeat_factors = self._get_repeat_factors(self.oversample_thr)
        # Repeat dataset's indices according to repeat_factors. For example,
        # if `repeat_factors = [1, 2, 3]`, and the `len(dataset) == 3`,
        # the repeated indices will be [1, 2, 2, 3, 3, 3].
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

    def get_cat_id_from_image_id(self, image_id):
        annots = self.dataset.coco.imgToAnns[image_id]
        cat_ids = set([])
        for annot in annots:
            cat_id = annot["category_id"]
            cat_obj = self.dataset.coco.cats[cat_id]
            if cat_obj["name"] in self.cat_names:
                cat_ids.add(cat_id)
        return cat_ids

    def _get_repeat_factors(self, repeat_thr: float) -> List[float]:
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq: defaultdict = defaultdict(float)
        num_images = len(self.dataset.ids)
        for idx in range(num_images):
            image_id = self.dataset.ids[idx]
            cat_ids = self.get_cat_id_from_image_id(image_id)
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f"caterogy {k} does not contain any images"
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq)) for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            # the length of `repeat_factors` need equal to the length of
            # dataset. Hence, if the `cat_ids` is empty,
            # the repeat_factor should be 1.
            repeat_factor: float = 1.0
            cat_ids = self.get_cat_id_from_image_id(self.dataset.ids[idx])  # set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) != 0:
                repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)
        return repeat_factors

    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            int: Local index of data.
        """
        return self.repeat_indices[idx]

    def __getitem__(self, idx):
        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)
