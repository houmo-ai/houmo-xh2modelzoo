import collections
import copy
from typing import List, Optional, Sequence, Tuple, Union

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from ..registry import DATASETS, TRANSFORMS
from .base_dataset import BaseDataset, force_full_init


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset`` and support lazy_init.

    Note:
        ``ConcatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``ConcatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        datasets (Sequence[BaseDataset] or Sequence[dict]): A list of datasets
            which will be concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        ignore_keys (List[str] or str): Ignore the keys that can be
            unequal in `dataset.metainfo`. Defaults to None.
            `New in version 0.3.0.`
    """

    def __init__(
        self,
        datasets: Sequence[Union[BaseDataset, dict]],
        lazy_init: bool = False,
        ignore_keys: Union[str, List[str], None] = None,
    ):
        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    "elements in datasets sequence should be config or "
                    f"`BaseDataset` instance, but got {type(dataset)}"
                )
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError("ignore_keys should be a list or str, " f"but got {type(ignore_keys)}")

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    raise ValueError(f"{key} does not in the meta information of " f"the {i}-th dataset")
                first_type = type(self._metainfo[key])
                cur_type = type(dataset.metainfo[key])
                if first_type is not cur_type:  # type: ignore
                    raise TypeError(
                        f"The type {cur_type} of {key} in the {i}-th dataset "
                        "should be the same with the first dataset "
                        f"{first_type}"
                    )
                if (
                    isinstance(self._metainfo[key], np.ndarray)
                    and not np.array_equal(self._metainfo[key], dataset.metainfo[key])
                    or (
                        not isinstance(self._metainfo[key], np.ndarray) and self._metainfo[key] != dataset.metainfo[key]
                    )
                ):
                    raise ValueError(
                        f"The meta information of the {i}-th dataset does not "
                        "match meta information of the first dataset"
                    )

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the first dataset in ``self.datasets``.

        Returns:
            dict: Meta information of first dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for d in self.datasets:
            d.full_init()
        # Get the cumulative sizes of `self.datasets`. For example, the length
        # of `self.datasets` is [2, 3, 4], the cumulative sizes is [2, 5, 9]
        super().__init__(self.datasets)
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(f"absolute value of index({idx}) should not exceed dataset" f"length({len(self)}).")
            idx = len(self) + idx
        # Get `dataset_idx` to tell idx belongs to which dataset.
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # Get the inner index of single dataset.
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)

    @force_full_init
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                "Please call `full_init` method manually to " "accelerate the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ConcatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            "`ConcatDataset` dose not support `get_subset` and "
            "`get_subset_` interfaces because this will lead to ambiguous "
            "implementation of some methods. If you want to use `get_subset` "
            "or `get_subset_` interfaces, please use them in the wrapped "
            "dataset first and then use `ConcatDataset`."
        )

    def get_subset(self, indices: Union[List[int], int]) -> "BaseDataset":
        """Not supported in ``ConcatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            "`ConcatDataset` dose not support `get_subset` and "
            "`get_subset_` interfaces because this will lead to ambiguous "
            "implementation of some methods. If you want to use `get_subset` "
            "or `get_subset_` interfaces, please use them in the wrapped "
            "dataset first and then use `ConcatDataset`."
        )


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Note:
        ``RepeatDataset`` should not inherit from ``BaseDataset`` since
        ``get_subset`` and ``get_subset_`` could produce ambiguous meaning
        sub-dataset which conflicts with original dataset. If you want to use
        a sub-dataset of ``RepeatDataset``, you should set ``indices``
        arguments for wrapped dataset which inherit from ``BaseDataset``.

    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        times (int): Repeat times.
        lazy_init (bool): Whether to load annotation during
            instantiation. Defaults to False.
    """

    def __init__(self, dataset: Union[BaseDataset, dict], times: int, lazy_init: bool = False):
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                "elements in datasets sequence should be config or " f"`BaseDataset` instance, but got {type(dataset)}"
            )
        self.times = times
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx: Global index of ``RepeatDataset``.

        Returns:
            idx (int): Local index of data.
        """
        return idx % self._ori_len

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                "Please call `full_init` method manually to accelerate the " "speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset[sample_idx]

    @force_full_init
    def __len__(self):
        return self.times * self._ori_len

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            "`RepeatDataset` dose not support `get_subset` and "
            "`get_subset_` interfaces because this will lead to ambiguous "
            "implementation of some methods. If you want to use `get_subset` "
            "or `get_subset_` interfaces, please use them in the wrapped "
            "dataset first and then use `RepeatDataset`."
        )

    def get_subset(self, indices: Union[List[int], int]) -> "BaseDataset":
        """Not supported in ``RepeatDataset`` for the ambiguous meaning of sub-
        dataset."""
        raise NotImplementedError(
            "`RepeatDataset` dose not support `get_subset` and "
            "`get_subset_` interfaces because this will lead to ambiguous "
            "implementation of some methods. If you want to use `get_subset` "
            "or `get_subset_` interfaces, please use them in the wrapped "
            "dataset first and then use `RepeatDataset`."
        )


@DATASETS.register_module()
class ClassBalancedDataset:
    """A wrapper of class balanced dataset.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :meth:`get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Note:
        ``ClassBalancedDataset`` should not inherit from ``BaseDataset``
        since ``get_subset`` and ``get_subset_`` could  produce ambiguous
        meaning sub-dataset which conflicts with original dataset. If you
        want to use a sub-dataset of ``ClassBalancedDataset``, you should set
        ``indices`` arguments for wrapped dataset which inherit from
        ``BaseDataset``.

    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def __init__(
        self,
        dataset: Union[BaseDataset, dict],
        oversample_thr: float,
        lazy_init: bool = False,
    ):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                "elements in datasets sequence should be config or " f"`BaseDataset` instance, but got {type(dataset)}"
            )
        self.oversample_thr = oversample_thr
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        # Get repeat factors for each image.
        repeat_factors = self._get_repeat_factors(self.dataset, self.oversample_thr)
        # Repeat dataset's indices according to repeat_factors. For example,
        # if `repeat_factors = [1, 2, 3]`, and the `len(dataset) == 3`,
        # the repeated indices will be [1, 2, 2, 3, 3, 3].
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        self._fully_initialized = True

    def _get_repeat_factors(self, dataset: BaseDataset, repeat_thr: float) -> List[float]:
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (BaseDataset): The dataset.
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            List[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq: defaultdict = defaultdict(float)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
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
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) != 0:
                repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            int: Local index of data.
        """
        return self.repeat_indices[idx]

    @force_full_init
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                "Please call `full_init` method manually to accelerate " "the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    @force_full_init
    def __len__(self):
        return len(self.repeat_indices)

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            "`ClassBalancedDataset` dose not support `get_subset` and "
            "`get_subset_` interfaces because this will lead to ambiguous "
            "implementation of some methods. If you want to use `get_subset` "
            "or `get_subset_` interfaces, please use them in the wrapped "
            "dataset first and then use `ClassBalancedDataset`."
        )

    def get_subset(self, indices: Union[List[int], int]) -> "BaseDataset":
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            "`ClassBalancedDataset` dose not support `get_subset` and "
            "`get_subset_` interfaces because this will lead to ambiguous "
            "implementation of some methods. If you want to use `get_subset` "
            "or `get_subset_` interfaces, please use them in the wrapped "
            "dataset first and then use `ClassBalancedDataset`."
        )


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Args:
        dataset (ConcatDataset or dict): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(
        self,
        dataset: Union[ConcatDataset, dict],
        pipeline: Sequence[dict],
        skip_type_keys: Optional[List[str]] = None,
        lazy_init: bool = False,
    ) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                "elements in datasets sequence should be config or "
                f"`ConcatDataset` instance, but got {type(dataset)}"
            )

        if skip_type_keys is not None:
            assert all([isinstance(skip_type_key, str) for skip_type_key in skip_type_keys])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform["type"])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError("pipeline must be a dict")

        self._metainfo = self.dataset.metainfo
        self.num_samples = len(self.dataset)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for transform, transform_type in zip(self.pipeline, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, "get_indices"):
                indices = transform.get_indices(self.dataset)
                if not isinstance(indices, collections.abc.Sequence):
                    indices = [indices]
                mix_results = [copy.deepcopy(self.dataset[index]) for index in indices]
                results["mix_results"] = mix_results

            results = transform(results)

            if "mix_results" in results:
                results.pop("mix_results")

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([isinstance(skip_type_key, str) for skip_type_key in skip_type_keys])
        self._skip_type_keys = skip_type_keys
