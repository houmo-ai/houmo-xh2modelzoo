import os.path as osp
import random
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pycocotools.mask as maskUtils
import torch
from transformers import CLIPTokenizer
from xhquant.utils import fileio

from ...image import imfrombytes
from ...registry import TRANSFORMS
from ...structures.bbox import get_box_type
from ...structures.mask import BitmapMasks, PolygonMasks
from .base import BaseTransform


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`imfrombytes`.
            See :func:`imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        file_client_args: Optional[dict] = None,
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError('"file_client_args" and "backend_args" cannot be set ' "at the same time.")

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results["img_path"]
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(filename, backend_args=self.backend_args)
            img = imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f"failed to load image: {filename}"
        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"ignore_empty={self.ignore_empty}, "
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
        )

        if self.file_client_args is not None:
            repr_str += f"file_client_args={self.file_client_args})"
        else:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


class BaseLoadAnnotations(BaseTransform):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in key point detection.
                # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
                # means the visibility of this keypoint. n must be equal to the
                # number of keypoint categories.
                'keypoints': [x1, y1, v1, ..., xn, yn, vn]
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4)
             # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # with (x, y, v) order, in np.float32 type.
            'gt_keypoints': np.ndarray(N, NK, 3)
        }

    Required Keys:

    - instances

      - bbox (optional)
      - bbox_label
      - keypoints (optional)

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_seg_map (np.uint8)
    - gt_keypoints (np.float32)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Defaults to False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`imfrombytes`.
            See :func:`imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_keypoints: bool = False,
        imdecode_backend: str = "cv2",
        file_client_args: Optional[dict] = None,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError('"file_client_args" and "backend_args" cannot be set ' "at the same time.")

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results["instances"]:
            gt_bboxes.append(instance["bbox"])
        results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from
                :class:`dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results["instances"]:
            gt_bboxes_labels.append(instance["bbox_label"])
        results["gt_bboxes_labels"] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from
                :class:`dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(self.file_client_args, results["seg_map_path"])
            img_bytes = file_client.get(results["seg_map_path"])
        else:
            img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)

        results["gt_seg_map"] = imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend).squeeze()

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results["instances"]:
            gt_keypoints.append(instance["keypoints"])
        results["gt_keypoints"] = np.array(gt_keypoints, np.float32).reshape((len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label}, "
        repr_str += f"with_seg={self.with_seg}, "
        repr_str += f"with_keypoints={self.with_keypoints}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f"file_client_args={self.file_client_args})"
        else:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class LoadDetAnnotations(BaseLoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in instance/panoptic segmentation. The segmentation mask
                # of the instance or the information of segments.
                # 1. If list[list[float]], it represents a list of polygons,
                # one for each connected component of the object. Each
                # list[float] is one simple polygon in the format of
                # [x1, y1, ..., xn, yn] (n >= 3). The Xs and Ys are absolute
                # coordinates in unit of pixels.
                # 2. If dict, it represents the per-pixel segmentation mask in
                # COCO's compressed RLE format. The dict should have keys
                # “size” and “counts”.  Can be loaded by pycocotools
                'mask': list[list[float]] or dict,

                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    Required Keys:

    - height
    - width
    - instances

      - bbox (optional)
      - bbox_label
      - mask (optional)
      - ignore_flag

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        poly2mask (bool): Whether to convert mask to bitmap. Default: True.
        box_type (str): The box type used to wrap the bboxes. If ``box_type``
            is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
        reduce_zero_label (bool): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to False.
        ignore_index (int): The label index to be ignored.
            Valid only if reduce_zero_label is true. Defaults is 255.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        with_mask: bool = False,
        poly2mask: bool = True,
        box_type: str = "hbox",
        # use for semseg
        reduce_zero_label: bool = False,
        ignore_index: int = 255,
        **kwargs,
    ) -> None:
        super(LoadDetAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get("instances", []):
            gt_bboxes.append(instance["bbox"])
            gt_ignore_flags.append(instance["ignore_flag"])
        if self.box_type is None:
            results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results["gt_bboxes"] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results["gt_ignore_flags"] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get("instances", []):
            gt_bboxes_labels.append(instance["bbox_label"])
        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results["gt_bboxes_labels"] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int, img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get("instances", []):
            gt_mask = instance["mask"]
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [np.array(polygon) for polygon in gt_mask if len(polygon) % 2 == 0 and len(polygon) >= 6]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance["ignore_flag"] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance["ignore_flag"] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and not (
                gt_mask.get("counts") is not None
                and gt_mask.get("size") is not None
                and isinstance(gt_mask["counts"], (list, str))
            ):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance["ignore_flag"] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance["ignore_flag"])
        results["gt_ignore_flags"] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``BaseDataset``.
        """
        h, w = results["ori_shape"]
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks([self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results["gt_masks"] = gt_masks

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if results.get("seg_map_path", None) is None:
            return

        img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)
        gt_semantic_seg = imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend).squeeze()

        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = self.ignore_index
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == self.ignore_index - 1] = self.ignore_index

        # modify if custom classes
        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results["gt_seg_map"] = gt_semantic_seg
        results["ignore_index"] = self.ignore_index

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label}, "
        repr_str += f"with_mask={self.with_mask}, "
        repr_str += f"with_seg={self.with_seg}, "
        repr_str += f"poly2mask={self.poly2mask}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str


@TRANSFORMS.register_module()
class LoadSegAnnotations(BaseLoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend="pillow",
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
        )
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn(
                "`reduce_zero_label` will be deprecated, "
                "if you would like to ignore the zero label, please "
                "set `reduce_zero_label=True` when dataset "
                "initialized"
            )
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(results["seg_map_path"], backend_args=self.backend_args)
        gt_semantic_seg = (
            imfrombytes(img_bytes, flag="unchanged", backend=self.imdecode_backend).squeeze().astype(np.uint8)
        )

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results["reduce_zero_label"]
        assert self.reduce_zero_label == results["reduce_zero_label"], (
            "Initialize dataset with `reduce_zero_label` as "
            f'{results["reduce_zero_label"]} but when load annotation '
            f"the `reduce_zero_label` is {self.reduce_zero_label}"
        )
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get("label_map", None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results["label_map"].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results["gt_seg_map"] = gt_semantic_seg
        results["seg_fields"].append("gt_seg_map")

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(reduce_zero_label={self.reduce_zero_label}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str


@TRANSFORMS.register_module()
class LoadClipTokens(BaseTransform):
    """Load caption tokens."""

    def __init__(self, clip_token_model) -> None:
        self.clip_token_model = clip_token_model
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_token_model)

    def transform(self, results: dict) -> dict:
        gt_captions = results["gt_caption"]
        input_ids = self.tokenizer(
            gt_captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        results["token_ids"] = input_ids
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__


@TRANSFORMS.register_module()
class RandomLoadClipTokens(BaseTransform):
    """Load caption tokens."""

    def __init__(self, clip_token_model) -> None:
        self.clip_token_model = clip_token_model
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_token_model)

    def transform(self, results: dict) -> dict:
        gt_captions = results["gt_caption"]
        if isinstance(gt_captions, Sequence):
            text = random.choice(gt_captions)
        else:
            text = gt_captions
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        results["token_ids"] = input_ids[0]
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__
