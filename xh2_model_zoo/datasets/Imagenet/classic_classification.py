"""
Reuse version v4
Author: Hahn Yuan
"""

import argparse
import copy
import os
import random
import re
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def calculate_n_correct(outputs, targets):
    _, predicted = outputs.max(1)
    n_correct = predicted.eq(targets).sum().item()
    return n_correct


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    # if target.size(0) != 256:
    #     print(target.size(0))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0))
    res.append(batch_size)
    return res


class SetSplittor:
    def __init__(self, fraction=0.2):
        self.fraction = fraction

    def split(self, dataset):
        pass


class LoaderGenerator:
    """ """

    def __init__(
        self,
        root=None,
        dataset_name="Custom_dataset",
        train_batch_size=64,
        test_batch_size=1,
        calib_batch_size=None,
        num_workers=0,
        subset: int = -1,
        device: str = "cuda:0",
        verbose: bool = False,
        calib_transform=None,
        test_transform=None,
        train_transform=None,
        with_orishape=False,
        **kwargs,
    ):
        self.root = root
        self.dataset_name = str.lower(dataset_name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.calib_batch_size = calib_batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.items = []
        self._train_set = None
        self._test_set = None
        self._calib_set = None
        self.train_loader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": kwargs.get("pin_memory", True),
            "drop_last": kwargs.get("drop_last", False),
        }
        self.test_loader_kwargs = self.train_loader_kwargs.copy()
        self.calib_transform = calib_transform
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.with_orishape = with_orishape

        self.subset = subset
        self.device = device
        self.verbose = verbose
        self.test_dataset = None  # save test_loader for evaluate
        self.class_balance = kwargs.get("class_balance", False)
        self.load()

    @property
    def train_set(self):
        pass

    @property
    def test_set(self):
        pass

    @property
    def calib_set(self):
        pass

    def load(self):
        pass

    def validate(self):
        pass

    def train_loader(self, collate_fn=None):
        assert self.train_set is not None
        if self.subset > 0:
            self.train_set = Subset(self.train_set, indices=[_ for _ in range(0, self.subset)])
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **self.train_loader_kwargs,
        )

    def test_loader(self, shuffle=False, batch_size=None, subset=None, collate_fn=None):
        assert self.test_set is not None

        if subset:
            assert subset <= len(self.test_set)

            if shuffle:
                inds = np.random.permutation(len(self.test_set))[:subset]
            else:
                inds = [_ for _ in range(0, subset)]

            self._test_set = Subset(self._test_set, indices=inds)

        if batch_size is None:
            batch_size = self.test_batch_size
            if subset > 0:
                assert subset >= batch_size

        if "coco" in self.dataset_name:
            if collate_fn is not None:
                collate_fn_coco = collate_fn
            else:
                if "yolov3" in self.dataset_name:

                    def collate_fn_coco(batch):
                        img, tgt, imgpath = tuple(zip(*batch))
                        return img[0], tgt[0], imgpath

                else:

                    def collate_fn_coco(batch):
                        return tuple(zip(*batch))

            sampler = torch.utils.data.SequentialSampler(self._test_set)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)

            self.test_dataset = torch.utils.data.DataLoader(
                self._test_set,
                batch_sampler=batch_sampler,
                shuffle=shuffle,
                collate_fn=collate_fn_coco,
                **self.test_loader_kwargs,
            )

            return self.test_dataset

        self.test_dataset = torch.utils.data.DataLoader(
            self._test_set,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **self.test_loader_kwargs,
        )

        return self.test_dataset

    def val_loader(self, collate_fn=None):
        assert self.val_set is not None
        if self.subset:
            self.val_set = Subset(self.val_set, indices=[_ for _ in range(0, self.subset)])

        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **self.test_loader_kwargs,
        )

    def trainval_loader(self, collate_fn=None):
        assert self.trainval_set is not None
        if self.subset > 0:
            self.trainval_set = Subset(self.trainval_set, indices=[_ for _ in range(0, self.subset)])
        return torch.utils.data.DataLoader(
            self.trainval_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **self.train_loader_kwargs,
        )

    def calib_loader(self, calib_num=2, seed=3, shuffle=False, collate_fn=None):

        assert self.calib_set is not None

        if self.calib_batch_size is None:
            batch_size = 1
        else:
            batch_size = self.calib_batch_size

        assert calib_num >= batch_size and calib_num <= len(self._calib_set)

        if shuffle:
            inds = np.random.permutation(len(self.train_set))[:calib_num].tolist()
        else:
            inds = [_ for _ in range(0, calib_num)]

        if self.calib_set is None:
            warnings.warn("calbset has not set use random data make up")
            np.random.seed(seed)
            inds = np.random.permutation(len(self.train_set))[:calib_num]
            self._calib_set = torch.utils.data.Subset(copy.deepcopy(self.train_set), inds)
        else:
            self._calib_set = Subset(self._calib_set, indices=inds)

        if "coco" in self.dataset_name:
            if collate_fn is None:
                if "yolov3" in self.dataset_name:

                    def collate_fn_coco(batch):
                        img, tgt, imgpath = tuple(zip(*batch))
                        return img[0], tgt[0]

                else:

                    def collate_fn_coco(batch):
                        img, tgt = tuple(zip(*batch))
                        return img

            else:
                collate_fn_coco = collate_fn

            sampler = torch.utils.data.SequentialSampler(self._calib_set)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)

            self.calib_dataset = torch.utils.data.DataLoader(
                self._calib_set,
                batch_sampler=batch_sampler,
                shuffle=False,
                collate_fn=collate_fn_coco,
                **self.test_loader_kwargs,
            )

            return self.calib_dataset

        return torch.utils.data.DataLoader(
            self._calib_set,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **self.train_loader_kwargs,
        )


class CIFARLoaderGenerator(LoaderGenerator):
    def load(self):
        if self.dataset_name == "cifar100":
            self.dataset_fn = datasets.CIFAR100
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        elif self.dataset_name == "cifar10":
            self.dataset_fn = datasets.CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        else:
            raise NotImplementedError
        self.calib_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = self.dataset_fn(self.root, train=True, download=True, transform=self.calib_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = self.dataset_fn(self.root, train=False, download=True, transform=self.test_transform)
        return self._test_set


class COCOLoaderGenerator(LoaderGenerator):
    # fix error unmerged in quant
    def load(self, augmentation_detection_tansforms=None, detection_tansforms=None):
        # download from https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh
        self.train_set = DetectionListDataset(
            os.path.join(self.root, "trainvalno5k.txt"),
            transform=augmentation_detection_tansforms,
        )
        self.test_set = DetectionListDataset(
            os.path.join(self.root, "5k.txt"),
            transform=detection_tansforms,
            multiscale=False,
        )
        self.train_loader_kwargs = {"collate_fn": self.train_set.collate_fn}
        self.test_loader_kwargs = {"collate_fn": self.test_set.collate_fn}


class DetectionListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = [path for path in file.readlines()]
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print(f"Could not apply transform.")
                return
        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1
        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack(
            [F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0) for img in imgs]
        )
        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


# from tmp.pil_resize import resize,cvresize,quantresize
class ImageNetLoaderGenerator(LoaderGenerator):
    def load(self):
        if self.calib_transform is None:
            self.calib_transform = transforms.Compose([transforms.ToTensor()])
        if self.test_transform is None:
            self.test_transform = transforms.Compose([transforms.ToTensor()])

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = ImageFolder(os.path.join(self.root, "train"), self.train_transform)
        return self._train_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = ImageFolder(os.path.join(self.root, "val"), self.test_transform)
        return self._test_set

    @property
    def calib_set(self):
        if self._calib_set is None:
            self._calib_set = ImageFolder(os.path.join(self.root, "train"), self.calib_transform)
        return self._calib_set

    def evaluate(self, batches_pred):
        """
        一套十分标准的imagenet测试逻辑
        """
        assert self.test_dataset is not None

        recorder = {
            "top1_accuracy": [],
            "top5_accuracy": [],
        }
        img_number = []

        for batch_idx, (batch_input, batch_label) in tqdm(
            enumerate(self.test_dataset),
            desc="Evaluating Model...",
            total=len(self.test_dataset),
        ):

            batch_pred = batches_pred[batch_idx].to(self.device)
            batch_label = torch.tensor(batch_label).to(self.device)

            if isinstance(batch_pred, list):
                batch_pred = torch.tensor(batch_pred)

            prec1, prec5, batch_size = accuracy(batch_pred, batch_label, topk=(1, 5))
            recorder["top1_accuracy"].append(prec1.item())
            recorder["top5_accuracy"].append(prec5.item())
            img_number.append(batch_size)

            if batch_idx % 100 == 0 and self.verbose:
                print(
                    "Test: [{0} / {1}]\t"
                    "Top1_accuracy {top1:.3f} ({top1:.3f})\t"
                    "Top5_accuracy {top5:.3f} ({top5:.3f})".format(
                        batch_idx,
                        len(self.test_dataset),
                        top1=sum(recorder["top1_accuracy"]) / len(recorder["top1_accuracy"]),
                        top5=sum(recorder["top5_accuracy"]) / len(recorder["top5_accuracy"]),
                    )
                )

            if batch_idx + 1 == len(batches_pred):
                break

        print(
            " * Top1_accuracy {top1:.3f} Top5_accuracy {top5:.3f}".format(
                top1=sum(recorder["top1_accuracy"]) / sum(img_number),
                top5=sum(recorder["top5_accuracy"]) / sum(img_number),
            )
        )

        res_top1 = np.array(sum(recorder["top1_accuracy"]) / sum(img_number))
        return res_top1


class DebugLoaderGenerator(LoaderGenerator):
    def load(self):
        version = re.findall("\d+", self.dataset_name)[0]

        class DebugSet(torch.utils.data.Dataset):
            def __getitem__(self, idx):
                if version == "0":
                    return torch.ones([1, 4, 4]), 0
                if version == "1":
                    return torch.ones([1, 8, 8]), 0
                if version == "2":
                    return torch.ones([1, 1, 1]), 0
                if version == "3":
                    return torch.ones([1, 3, 3]), 0
                else:
                    raise NotImplementedError(f"version {version} of Debug dataset is not supported")

            def __len__(self):
                return 1

        self.train_set = DebugSet()
        self.test_set = DebugSet()


def get_dataset(args: argparse.Namespace):
    """Preparing Datasets, args:
    dataset (required): MNIST, cifar10/100, ImageNet, coco
    dataset_root: str, default='./datasets'
    num_workers: int
    batch_size: int
    test_batch_size: int
    val_fraction: float, default=0

    """
    dataset_name = str.lower(args.dataset)
    dataset_root = getattr(args, "dataset_root", "./datasets")
    num_workers = args.num_workers if hasattr(args, "num_workers") else 16
    batch_size = args.batch_size if hasattr(args, "batch_size") else 64
    test_batch_size = args.test_batch_size if hasattr(args, "test_batch_size") else batch_size
    val_fraction = args.val_fraction if hasattr(args, "val_fraction") else 0
    if "cifar" in dataset_name:
        # Data loading code
        g = CIFARLoaderGenerator(dataset_root, args.dataset, batch_size, test_batch_size, num_workers)
    elif "coco" in dataset_name:
        g = COCOLoaderGenerator(dataset_root, args.dataset, batch_size, test_batch_size, num_workers)
    elif "debug" in dataset_name:
        g = DebugLoaderGenerator(dataset_root, args.dataset, batch_size, test_batch_size, num_workers)
    elif args.dataset == "ImageNet":
        g = ImageNetLoaderGenerator(dataset_root, args.dataset, batch_size, test_batch_size, num_workers)
    else:
        raise NotImplementedError
    return g.train_loader(), g.test_loader()


if __name__ == "__main__":
    # Preparing the inp_data for analysis
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CIFAR10", type=str, help="the location of the dataset")
    parser.add_argument(
        "--inp_data_size",
        default=1,
        type=int,
        help="batchsize of input data for testing",
    )
    parser.add_argument("--output_dir", default="./", type=str, help="output path of the inp_data.pth")
    args = parser.parse_args()

    train_loader, test_loader = get_dataset(args)
    inp_data = [
        test_loader.dataset[np.random.randint(len(test_loader.dataset))][0].unsqueeze(0)
        for _ in range(args.inp_data_size)
    ]
    save_file = f"{args.output_dir}/{args.dataset}_inp_data.pth"
    print(f"saving to {save_file}")
    torch.save(torch.cat(inp_data, 0), save_file)
