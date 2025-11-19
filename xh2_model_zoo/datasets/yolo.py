import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from ..registry import DATASETS

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."

IMG_FORMATS = {
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
}

FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\n"


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return np.array(boxes)  # cls, x1y1x2y2


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = (
        f"{os.sep}images{os.sep}",
        f"{os.sep}labels{os.sep}",
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class ImageTransform:
    def __init__(self, input_shape, to_tensor=False):
        self.input_shape = input_shape
        self.to_tensor = to_tensor

    def __call__(self, img, label=None):
        img = np.array(img)
        img_h, img_w = img.shape[:2]
        # Padded resize
        img, ratio, (dw, dh) = letterbox(img, self.input_shape, stride=32, auto=False)
        if label is not None:
            label = label.copy()
            label[:, 1:] = label[:, 1:] * np.array([img_w, img_h, img_w, img_h]) * np.array(
                [ratio[0], ratio[1], ratio[0], ratio[1]]
            ) + np.array([dw, dh, dw, dh])

        """
        cls shape is n x 5, with cls, cx, cy, w, h
        """
        # Convert
        img = img.transpose((2, 0, 1))  # [::-1]   # HWC to CHW
        img = np.ascontiguousarray(img)
        input = torch.from_numpy(img)  # .unsqueeze(0)
        # assert not self.to_tensor
        if self.to_tensor:
            input = input / 255.0
        else:
            input = input.float()  # .unsqueeze(0) # .numpy()
        if label is not None:
            return input, label
        return input


@DATASETS.register_module()
class YOLODataset(data.Dataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(
        self,
        data_file,
        return_label=False,
        input_shape=(640, 640),
        task="detect",
        bgr=False,
    ):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        assert task == "detect", "Currently only detect is supported!"
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.nkpt = 0
        self.ndim = 0
        self.num_cls = 80
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        self.im_files = self.get_img_files(data_file)
        self.labels = self.get_labels()
        assert len(self.im_files) == len(self.label_files)
        self.transform = ImageTransform(input_shape, True)
        self.bgr = bgr
        self.return_label = return_label

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        return im_files

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)

        labels = []
        for lbf in self.label_files:
            if os.path.isfile(lbf):
                with open(lbf) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb) and (not self.use_keypoints):  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xyxy)
                    lb = np.array(lb, dtype=np.float32)
                    nl = len(lb)
                    if nl:
                        if self.use_keypoints:
                            assert lb.shape[1] == (
                                5 + self.nkpt * self.ndim
                            ), f"labels require {(5 + self.nkpt * self.ndim)} columns each"
                            points = lb[:, 5:].reshape(-1, self.ndim)[:, :2]
                        else:
                            assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                            points = lb[:, 1:]
                        assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                        assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                        # All labels
                        max_cls = lb[:, 0].max()  # max label count
                        assert max_cls <= self.num_cls, (
                            f"Label class {int(max_cls)} exceeds dataset class count {self.num_cls}. "
                            f"Possible class labels are 0-{self.num_cls - 1}"
                        )
                        _, i = np.unique(lb, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            lb = lb[i]  # remove duplicates
                            if segments:
                                segments = [segments[x] for x in i]
                    else:
                        lb = np.zeros(
                            (
                                0,
                                ((5 + self.nkpt * self.ndim) if self.use_keypoints else 5),
                            ),
                            dtype=np.float32,
                        )
            else:
                lb = np.zeros(
                    (0, (5 + self.nkpt * self.ndim) if self.use_keypoints else 5),
                    dtype=np.float32,
                )
            if self.use_keypoints:
                keypoints = lb[:, 5:].reshape(-1, self.nkpt, self.ndim)
                if self.ndim == 2:
                    kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                    keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
            lb = lb[:, :5]
            labels.append(
                {
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xyxy",
                }
            )
        return labels

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        im = cv2.imread(self.im_files[index])
        if not self.bgr:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        lbl = np.hstack((self.labels[index]["cls"], self.labels[index]["bboxes"]))
        img, label = self.transform(im, lbl)
        if self.return_label:
            return img, label
        return img


def show(img, label):
    im = (img.permute((1, 2, 0)).contiguous() * 255).numpy().astype(np.uint8)
    for i in range(label.shape[0]):
        x1, y1, x2, y2 = label[i, 1:]
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (233, 0, 244), 1)
    cv2.imwrite("tmp.jpg", im)


if __name__ == "__main__":
    ds = YOLODataset("/data01/datasets/coco/coco2017/val2017.txt", bgr=True)
    x = ds[0]
    show(x[0], x[1])
