import copy
import functools
import os.path as osp
from os import PathLike
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

from torch.utils.data import Dataset
from xhquant.utils import is_abs
from xhquant.utils.config import Config
from xhquant.utils.fileio import join_path, list_from_file
from xhquant.utils.logger import get_root_logger

from ..registry import TRANSFORMS


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f"transform should be a callable object, " f"but got {type(transform)}")
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be a callable object or dict, " f"but got {type(transform)}")

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def force_full_init(old_func: Callable) -> Any:
    """Those methods decorated by ``force_full_init`` will be forced to call
    ``full_init`` if the instance has not been fully initiated.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``full_init`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, "full_init"):
            raise AttributeError(f"{type(obj)} does not have full_init " "method.")
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, "_fully_initialized", False):
            logger = get_root_logger()
            logger.warning(
                f"Attribute `_fully_initialized` is not defined in "
                f"{type(obj)} or `type(obj)._fully_initialized is "
                "False, `full_init` will be called and "
                f"{type(obj)}._fully_initialized will be set to True",
            )
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper


def expanduser(path):
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset):
    METAINFO: dict = dict()
    _fully_initialized = False

    def __init__(
        self,
        ann_file: Optional[str] = "",
        metainfo: Union[Mapping, Config, None] = None,
        data_root: Optional[str] = "",
        data_prefix: dict = dict(img_path=""),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = True,
        max_refetch: int = 1000,
    ):
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        ann_file = expanduser(ann_file)

        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_list: List[dict] = []
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        # Join paths.
        self._join_prefix()

        if not lazy_init:
            self.full_init()
        self.max_refetch = max_refetch

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    @property
    def img_prefix(self):
        return self.data_prefix["img_path"]

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self._metainfo.get("classes", None)

    @classmethod
    def _load_metainfo(cls, metainfo: Union[Mapping, Config, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError("metainfo should be a Mapping or Config, " f"but got {type(metainfo)}")
        logger = get_root_logger()
        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    logger.warning(
                        f"{v} is not a meta file, simply parsed as meta " "information",
                    )
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError("prefix should be a string, but got " f"{type(prefix)}")
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def __getitem__(self, idx: int) -> dict:
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception("Test time pipline should not get `None` " "data_sample")
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(
            f"Cannot find valid image after {self.max_refetch}! " "Please check your image path and pipeline"
        )

        return {}

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self):
        return len(self.data_list)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info["sample_idx"] = idx
        else:
            data_info["sample_idx"] = len(self) + idx

        return data_info

    def load_data_list(self) -> List[dict]:
        raise NotImplementedError

    def full_init(self):
        if self._fully_initialized:
            return
        logger = get_root_logger()
        logger.info(f"Initializing {self.__class__.__name__}...")
        # load data information
        self.data_list = self.load_data_list()
        # # filter illegal data, such as data that has no annotations.
        # self.data_list = self.filter_data()
        # # Get subset data according to indices.

        # # serialize data_list
        # if self.serialize_data:
        #     self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = "Dataset " + self.__class__.__name__
        body = []
        if self._fully_initialized:
            body.append(f"Number of samples: \t{self.__len__()}")
        else:
            body.append("Haven't been initialized")

        body.extend(self.extra_repr())

        if len(self.pipeline.transforms) > 0:
            body.append("With transforms:")
            for t in self.pipeline.transforms:
                body.append(f"    {t}")

        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)
