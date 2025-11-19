import os
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision
import torchvision.transforms as transforms

from xh2_model_zoo.datasets.data_path import IMAGNET_DATASET_PATH, n_workers
from xh2_model_zoo.datasets.Imagenet.classic_classification import ImageNetLoaderGenerator
from xh2_model_zoo.datasets.preprocess.transform import RGB2YUV, ToTensorNotNormal
from xh2_model_zoo.utils.helpers import get_onnx_from_url
from xh2_model_zoo.utils.onnx.onnx_run import OnnxRuntimeDetector

__all__ = ["Vit"]


class Vit:
    def __init__(self, model_name=None, input_shape=None, device="cpu"):
        self.device = device
        self._dataset = None
        self.model = self._load_model(model_name)
        self.model_name = model_name
        self.input_shape = input_shape

    def _load_model(self, model_name):
        self.model_path = model_name

        if model_name.endswith(".onnx"):
            return onnx.load(model_name)
        elif model_name == "vit_small_patch16_224":
            self.input_shape = (224, 224)
            url = "http://10.10.1.53:8081/artifactory/model_zoo2/houmo/vit/vit_small_patch16_224.onnx"
            model, self.model_path = get_onnx_from_url(url, return_path=True, company="houmo/vit")
            return model
        else:
            raise NotImplementedError(f"Unknown model name of" % (model_name))

    def forward(self, data):
        # data: torch.Tensor with shape (n, c, h, w)
        if self.model_path.endswith(".onnx"):
            data = data.numpy()
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            data = (((data / 255.0) - mean) / std).astype(np.float32)

            nn_out = self._forward(data)
            return torch.tensor(nn_out)

    def pre_process(self, input):
        return input

    def post_process(self, output):
        return output

    def evaluate(self, result):
        top1_acc = self._dataset.evaluate(result)

        return top1_acc

    def dataset(
        self,
        modes=["processcalib", "processtest"],
        calib_num=2,
        test_batch_size=256,
        shuffle=False,
        input_shape=None,
        subset=None,
        dataset_root=None,
        **kwargs,
    ):
        test_transform = train_transform = calib_transform = None
        for mode in modes:
            assert mode in ["calib", "test", "train", "processcalib", "processtest", "processtrain", "trt"]
            if mode == "processcalib":
                calib_transform = transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(224), ToTensorNotNormal()]
                )
            elif mode == "processtest":
                test_transform = transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(224), ToTensorNotNormal()]
                )
            elif mode == "processtrain":
                train_transform = transforms.Compose(
                    [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), ToTensorNotNormal()]
                )

            if mode == "trt":
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                calib_transform = transforms.Compose(
                    [
                        transforms.Resize(self.input_shape),
                        transforms.ToTensor(),
                        # transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize(mean, std),
                    ]
                )
                test_transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(self.input_shape),
                        transforms.ToTensor(),
                        transforms.ConvertImageDtype(torch.float),
                        transforms.Normalize(mean, std),
                    ]
                )
                train_transform = transforms.Compose([train_transform, transforms.Resize(self.input_shape)])
            else:
                if input_shape:
                    self.input_shape = input_shape
                if self.input_shape:
                    calib_transform = transforms.Compose([calib_transform, transforms.Resize(self.input_shape)])
                    test_transform = transforms.Compose([test_transform, transforms.Resize(self.input_shape)])
                    train_transform = transforms.Compose([train_transform, transforms.Resize(self.input_shape)])

        if self._dataset is None:
            self._dataset = ImageNetLoaderGenerator(
                dataset_root if dataset_root is not None else IMAGNET_DATASET_PATH,
                "imagenet",
                test_batch_size=test_batch_size,
                num_workers=n_workers,
                device=self.device,
                calib_transform=calib_transform,
                test_transform=test_transform,
                train_transform=train_transform,
                **kwargs,
            )
        ret_loaders = list()

        for mode in modes:
            if mode in ["calib", "processcalib"]:
                ret_loaders.append(self._dataset.calib_loader(calib_num=calib_num, shuffle=shuffle))
            elif mode in ["test", "processtest"]:
                ret_loaders.append(self._dataset.test_loader(batch_size=test_batch_size, subset=subset))
            elif mode in ["train", "processtrain"]:
                ret_loaders.append(self._dataset.train_loader())
            elif mode in ["trt"]:
                calib_set = self._dataset.calib_loader(calib_num=calib_num, shuffle=shuffle)
                calib = calib_set
                ret_loaders.append(calib)
                ret_loaders.append(self._dataset.test_loader(batch_size=test_batch_size, subset=subset))
        return ret_loaders
