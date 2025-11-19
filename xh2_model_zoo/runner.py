import copy
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from xhquant.api import digit_version, get_rank, get_root_logger, get_world_size

from .datasets.utils import worker_init_fn as default_worker_init_fn
from .evaluation import Evaluator
from .registry import DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS

TORCH_VERSION = torch.__version__


def _get_batch_size(dataloader: Union[Dict, DataLoader, Any]):
    if isinstance(dataloader, dict):
        if "batch_size" in dataloader:
            return dataloader["batch_size"]
        elif "batch_sampler" in dataloader and "batch_size" in dataloader["batch_sampler"]:
            return dataloader["batch_sampler"]["batch_size"]
        else:
            raise ValueError("Please set batch_size in `Dataloader` or " "`batch_sampler`")
    elif isinstance(dataloader, DataLoader):
        assert dataloader.batch_sampler is not None
        assert isinstance(dataloader.batch_sampler, BatchSampler)
        return dataloader.batch_sampler.batch_size
    else:
        raise ValueError("dataloader should be a dict or a Dataloader " f"instance, but got {type(dataloader)}")


class _SlicedDataset:

    def __init__(self, dataset, length) -> None:
        self._dataset = dataset
        self._length = length

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return self._length


from .datasets import *  # noqa F401 #isort:skip
from .datasets.transforms import *  # noqa F401 #isort:skip


class Runner:
    @staticmethod
    def build_evaluator(evaluator: Union[Dict, List, Evaluator]) -> Evaluator:
        if isinstance(evaluator, Evaluator):
            return evaluator
        elif isinstance(evaluator, dict):
            # if `metrics` in dict keys, it means to build customized evalutor
            if "metrics" in evaluator:
                evaluator.setdefault("type", "Evaluator")
                return EVALUATOR.build(evaluator)
            # otherwise, default evalutor will be built
            else:
                return Evaluator(evaluator)  # type: ignore
        elif isinstance(evaluator, list):
            # use the default `Evaluator`
            return Evaluator(evaluator)  # type: ignore
        else:
            raise TypeError("evaluator should be one of dict, list of dict, and Evaluator" f", but got {evaluator}")

    @staticmethod
    def build_dataloader(
        dataloader: Union[DataLoader, Dict],
        seed: Optional[int] = None,
        diff_rank_seed: bool = False,
    ) -> DataLoader:
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)
        # build dataset
        dataset_cfg = dataloader_cfg.pop("dataset")
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            # if hasattr(dataset, "full_init"):
            #     dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        assert isinstance(dataset, Dataset)
        num_batch_per_epoch = dataloader_cfg.pop("num_batch_per_epoch", None)
        if num_batch_per_epoch is not None:
            world_size = get_world_size()
            num_samples = num_batch_per_epoch * _get_batch_size(dataloader_cfg) * world_size
            dataset = _SlicedDataset(dataset, num_samples)

        # build sampler
        sampler_cfg = dataloader_cfg.pop("sampler")
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(sampler_cfg, default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg
        batch_sampler = None
        init_fn = None

        if "worker_init_fn" in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop("worker_init_fn")
            worker_init_fn_type = worker_init_fn_cfg.pop("type")
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    "type of worker_init_fn should be string or callable "
                    f"object, but got {type(worker_init_fn_type)}"
                )
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn, **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop("disable_subprocess_warning", False)
                assert isinstance(disable_subprocess_warning, bool), (
                    "disable_subprocess_warning should be a bool, but got " f"{type(disable_subprocess_warning)}"
                )
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get("num_workers", 0),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning,
                )
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if "persistent_workers" in dataloader_cfg and digit_version(TORCH_VERSION) < digit_version("1.7.0"):
            logger = get_root_logger()
            logger.warning(
                "`persistent_workers` is only available when " "pytorch version >= 1.7",
            )
            dataloader_cfg.pop("persistent_workers")

        # TODO: support multi gpu
        collate_fn = NotImplementedError
        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop("collate_fn", dict(type="pseudo_collate"))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop("type")
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError("collate_fn should be a dict or callable object, but got " f"{collate_fn_cfg}")

        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg,
        )
        return data_loader
