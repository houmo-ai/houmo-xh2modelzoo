import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from xhquant.utils import fileio

from ...image import imcrop, imfrombytes, imresize
from ...registry import TRANSFORMS
from .base import BaseTransform


@TRANSFORMS.register_module()
class ResizeEdge(BaseTransform):
    """Resize images along the specified edge.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    **Added Keys:**

    - scale
    - scale_factor

    Args:
        scale (int): The edge scale to resizing.
        edge (str): The edge to resize. Defaults to 'short'.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results.
            Defaults to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
            Defaults to 'bilinear'.
    """

    def __init__(
        self,
        scale: int,
        edge: str = "short",
        backend: str = "cv2",
        interpolation: str = "bilinear",
    ) -> None:
        allow_edges = ["short", "long", "width", "height"]
        assert edge in allow_edges, f'Invalid edge "{edge}", please specify from {allow_edges}.'
        self.edge = edge
        self.scale = scale
        self.backend = backend
        self.interpolation = interpolation

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        img, w_scale, h_scale = imresize(
            results["img"],
            results["scale"],
            interpolation=self.interpolation,
            return_scale=True,
            backend=self.backend,
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["scale"] = img.shape[:2][::-1]
        results["scale_factor"] = (w_scale, h_scale)

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'scale', 'scale_factor',
            'img_shape' keys are updated in result dict.
        """
        assert "img" in results, "No `img` field in the input."

        h, w = results["img"].shape[:2]
        if any(
            [
                # conditions to resize the width
                self.edge == "short" and w < h,
                self.edge == "long" and w > h,
                self.edge == "width",
            ]
        ):
            width = self.scale
            height = int(self.scale * h / w)
        else:
            height = self.scale
            width = int(self.scale * w / h)
        results["scale"] = (width, height)

        self._resize_img(results)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"edge={self.edge}, "
        repr_str += f"backend={self.backend}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@TRANSFORMS.register_module()
class CenterCrop(BaseTransform):
    """Crop the center of the image, segmentation masks, bounding boxes and key
    points. If the crop area exceeds the original image and ``auto_pad`` is
    True, the original image will be padded before cropping.

    Required Keys:

    - img
    - gt_seg_map (optional)
    - gt_bboxes (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional)
    - gt_bboxes (optional)
    - gt_keypoints (optional)

    Added Key:

    - pad_shape


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (w, h). If set to an integer, then cropping
            width and height are equal to this integer.
        auto_pad (bool): Whether to pad the image if it's smaller than the
            ``crop_size``. Defaults to False.
        pad_cfg (dict): Base config for padding. Refer to ``Pad`` for
            detail. Defaults to ``dict(type='Pad')``.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the
            gt bboxes are allowed to cross the border of images. Therefore,
            we don't need to clip the gt bboxes in these cases.
            Defaults to True.
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        auto_pad: bool = False,
        pad_cfg: dict = dict(type="Pad"),
        clip_object_border: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), "The expected crop_size is an integer, or a tuple containing two "
        "intergers"

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.auto_pad = auto_pad

        self.pad_cfg = pad_cfg.copy()
        # size will be overwritten
        if "size" in self.pad_cfg and auto_pad:
            warnings.warn(
                "``size`` is set in ``pad_cfg``,"
                "however this argument will be overwritten"
                " according to crop size and image size"
            )

        self.clip_object_border = clip_object_border

    def _crop_img(self, results: dict, bboxes: np.ndarray) -> None:
        """Crop image.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if results.get("img", None) is not None:
            img = imcrop(results["img"], bboxes=bboxes)
            img_shape = img.shape[:2]  # type: ignore
            results["img"] = img
            results["img_shape"] = img_shape
            results["pad_shape"] = img_shape

    def _crop_seg_map(self, results: dict, bboxes: np.ndarray) -> None:
        """Crop semantic segmentation map.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if results.get("gt_seg_map", None) is not None:
            img = imcrop(results["gt_seg_map"], bboxes=bboxes)
            results["gt_seg_map"] = img

    def _crop_bboxes(self, results: dict, bboxes: np.ndarray) -> None:
        """Update bounding boxes according to CenterCrop.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if "gt_bboxes" in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h])
            # gt_bboxes has shape (num_gts, 4) in (tl_x, tl_y, br_x, br_y)
            # order.
            gt_bboxes = results["gt_bboxes"] - bbox_offset
            if self.clip_object_border:
                gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, results["img"].shape[1])
                gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, results["img"].shape[0])
            results["gt_bboxes"] = gt_bboxes

    def _crop_keypoints(self, results: dict, bboxes: np.ndarray) -> None:
        """Update key points according to CenterCrop. Keypoints that not in the
        cropped image will be set invisible.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if "gt_keypoints" in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            keypoints_offset = np.array([offset_w, offset_h, 0])
            # gt_keypoints has shape (N, NK, 3) in (x, y, visibility) order,
            # NK = number of points per object
            gt_keypoints = results["gt_keypoints"] - keypoints_offset
            # set gt_kepoints out of the result image invisible
            height, width = results["img"].shape[:2]
            valid_pos = (
                (gt_keypoints[:, :, 0] >= 0)
                * (gt_keypoints[:, :, 0] < width)
                * (gt_keypoints[:, :, 1] >= 0)
                * (gt_keypoints[:, :, 1] < height)
            )
            gt_keypoints[:, :, 2] = np.where(valid_pos, gt_keypoints[:, :, 2], 0)
            gt_keypoints[:, :, 0] = np.clip(gt_keypoints[:, :, 0], 0, results["img"].shape[1])
            gt_keypoints[:, :, 1] = np.clip(gt_keypoints[:, :, 1], 0, results["img"].shape[0])
            results["gt_keypoints"] = gt_keypoints

    def transform(self, results: dict) -> dict:
        """Apply center crop on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Results with CenterCropped image and semantic segmentation
            map.
        """
        crop_width, crop_height = self.crop_size[0], self.crop_size[1]

        assert "img" in results, "`img` is not found in results"
        img = results["img"]
        # img.shape has length 2 for grayscale, length 3 for color
        img_height, img_width = img.shape[:2]

        if crop_height > img_height or crop_width > img_width:
            if self.auto_pad:
                # pad the area
                img_height = max(img_height, crop_height)
                img_width = max(img_width, crop_width)
                pad_size = (img_width, img_height)
                _pad_cfg = self.pad_cfg.copy()
                _pad_cfg.update(dict(size=pad_size))
                pad_transform = TRANSFORMS.build(_pad_cfg)
                results = pad_transform(results)
            else:
                crop_height = min(crop_height, img_height)
                crop_width = min(crop_width, img_width)

        y1 = max(0, int(round((img_height - crop_height) / 2.0)))
        x1 = max(0, int(round((img_width - crop_width) / 2.0)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        # crop the image
        self._crop_img(results, bboxes)
        # crop the gt_seg_map
        self._crop_seg_map(results, bboxes)
        # crop the bounding box
        self._crop_bboxes(results, bboxes)
        # crop the keypoints
        self._crop_keypoints(results, bboxes)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size = {self.crop_size}"
        repr_str += f", auto_pad={self.auto_pad}"
        repr_str += f", pad_cfg={self.pad_cfg}"
        repr_str += f",clip_object_border = {self.clip_object_border})"
        return repr_str
