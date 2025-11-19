import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
import math
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import accimage
except ImportError:
    accimage = None

from xh2_model_zoo.utils.box_utils import letterbox


class RGB2RGBSP:
    def __call__(self, img):
        c, h, w = img.shape
        r, g, b = torch.split(img, [1, 1, 1], 0)
        gb = torch.stack((g, b), -1).view(2, h, w)
        out = torch.cat((r, gb), 0).view(h, w, c)
        return out


# YUV = RGB * M[0:3] + M[3]
M_RGB2YUV = {
    # 'BT601': [
    #     [0.299, -0.168735892,  0.5],
    #     [0.587, -0.331264108, -0.418687589],
    #     [0.114,  0.5,         -0.081312411],
    #     [0, 128, 128] # bias
    # ],
    # Below is a lite version of above
    "BT601": [
        [0.299, -0.169, 0.5],
        [0.587, -0.331, -0.419],
        [0.114, 0.5, -0.081],
        [0, 128, 128],  # bias
    ],
}

# RBG = YUV * M[0:3] + M[3]
M_YUV2RGB = {
    # From Nvidia PVA
    "BT601": [[1, 1, 1], [0, -0.344, 1.772], [1.402, -0.714, 0], [-179.456, 135.459, -226.816]],
}


M_BGR2YUV = {
    # Below is a lite version of above
    "BT601": [
        [0.114, 0.5, -0.081],
        [0.587, -0.331, -0.419],
        [0.299, -0.169, 0.5],
        [0, 128, 128],  # bias
    ],
}

# RBG = YUV * M[0:3] + M[3]
M_YUV2BGR = {
    # From Nvidia PVA
    "BT601": [[1, 1, 1], [1.772, -0.344, 0], [0, -0.714, 1.402], [-226.816, 135.459, -179.456]],
}


def _batch_uv_to_444(y, u, v, fmt="422"):
    _MAP = {"420": (2, 2), "422": (2, 1)}
    assert u.shape == v.shape
    new_u = torch.zeros_like(y)
    new_v = torch.zeros_like(y)
    div_w, div_h = _MAP[fmt]
    if div_w == 2 and div_h == 1:
        new_u[:, :, 0::2] = u  # 1, h, w
        new_u[:, :, 1::2] = u
        new_v[:, :, 0::2] = v
        new_v[:, :, 1::2] = v
    elif div_w == 2 and div_h == 2:
        new_u[:, 0::2, 0::2] = u  # 1, h, w
        new_u[:, 0::2, 1::2] = u  # 1, h, w
        new_u[:, 1::2, 0::2] = u
        new_u[:, 1::2, 1::2] = u
        new_v[:, 0::2, 0::2] = v
        new_v[:, 0::2, 1::2] = v
        new_v[:, 1::2, 0::2] = v
        new_v[:, 1::2, 1::2] = v
    return new_u, new_v


class ColorConverter:
    def __init__(self, fmt="RGB2YUV", version="BT601"):
        if fmt == "RGB2YUV":
            Mb = M_RGB2YUV
        elif fmt == "YUV2RGB":
            Mb = M_YUV2RGB
        else:
            raise NotImplementedError(fmt + " is unknown")

        self.M = Mb[version][0:3]
        self.b = Mb[version][3]

    def __call__(self, img_hwc: np.ndarray):
        """
        @img_hwc: store pattern is [h, w, c]
        """
        res = np.matmul(img_hwc, self.M) + self.b
        return np.round(np.clip(res, 0, 255)).astype(np.uint8)


class YUVFormat:
    def __init__(self, fmt="422", interpolation=False) -> None:
        self.fmt = fmt
        self._MAP = {"420": (2, 2), "422": (2, 1), "YUV420": (2, 2), "YUV422": (2, 1)}
        self.interpolation = interpolation

    def __call__(self, img: torch.Tensor):
        _, img_h, img_w = img.size()
        # breakpoint()
        y, u, v = torch.split(img, 1, dim=0)
        if self.fmt == "444":
            uv = torch.stack([u, v], dim=-1)
            y = y.view(-1)
            uv = uv.view(-1)
            yuv = torch.cat((y, uv), 0)
            return yuv.view((img_h, img_w, 3))
        div_w, div_h = self._MAP[self.fmt]
        if self.interpolation:
            uv_resize = transforms.Resize((img_h // div_h, img_w // div_w))
            u = uv_resize(u)

            v = uv_resize(v)
            # Convert u and v to uint8 with clipping and rounding:
            u = u.clip(0, 255).round()
            v = v.clip(0, 255).round()
        else:
            u = u[:, 0::div_w]
            u = u[0::div_h, :]
            v = v[:, 0::div_w]
            v = v[0::div_h, :]
        uv = torch.stack([u, v], dim=-1)
        y = y.view(-1)
        uv = uv.view(-1)
        yuv = torch.cat((y, uv), 0)
        result = torch.zeros(img_h * img_w * 3)
        result[: yuv.shape[0]] = yuv
        return result.view((img_h, img_w, 3))


class RGB2YUV:
    def __init__(self, version="BT601", fmt="422", interpolation=True) -> None:
        """
        layout hwc or chw
        """
        Mb = M_RGB2YUV[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3]).T
        self.b = self.b.view(3, 1, 1)
        self.formatter = YUVFormat(fmt, interpolation)

    def __call__(self, img: torch.Tensor) -> Any:
        self.M = self.M.to(img.device)
        self.b = self.b.to(img.device)
        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()

        # Change YUV store format
        result = self.formatter(result)
        return result


class BGR2YUV:
    def __init__(self, version="BT601", fmt="422", interpolation=True) -> None:
        """
        layout hwc or chw
        """
        Mb = M_BGR2YUV[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3]).T
        self.b = self.b.view(3, 1, 1)
        self.formatter = YUVFormat(fmt, interpolation)

    def __call__(self, img: torch.Tensor) -> Any:
        self.M.to(img.device)
        self.b.to(img.device)
        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()

        # Change YUV store format
        result = self.formatter(result)
        return result


class YUV2RGB:
    def __init__(self, version="BT601") -> None:
        Mb = M_YUV2RGB[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3]).T
        self.b = self.b.view(3, 1, 1)

    def __call__(self, img: torch.Tensor) -> Any:
        self.M.to(img.device)
        self.b.to(img.device)

        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()
        return result


class YUV2BGR:
    def __init__(self, version="BT601") -> None:
        Mb = M_YUV2BGR[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3]).T
        self.b = self.b.view(3, 1, 1)

    def __call__(self, img: torch.Tensor) -> Any:
        self.M.to(img.device)
        self.b.to(img.device)

        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()
        return result


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


class ToTensorNotNormal:
    def __init__(self, need_sq=False) -> None:
        self.need_sq = need_sq

    def __call__(self, pic):

        if not (F.F_pil._is_pil_image(pic) or _is_numpy(pic)):
            raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

        if _is_numpy(pic) and not _is_numpy_image(pic):
            raise ValueError("pic should be 2/3 dimensional. Got {} dimensions.".format(pic.ndim))

        default_float_dtype = torch.get_default_dtype()

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype)
            else:
                return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic).to(dtype=default_float_dtype)

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

        if pic.mode == "1":
            img = 255 * img
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if self.need_sq:
            img = img.unsqueeze(0)

        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ResizeHWC:
    def __init__(self, dst_h, dst_w) -> None:
        self.h = dst_h
        self.w = dst_w

    def __call__(self, img: np.array) -> np.array:
        return cv2.resize(img, (self.w, self.h))


class PIL2Numpy:
    def __call__(self, img) -> np.array:
        return np.array(img)


class PilLoader:
    def __call__(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class Identity:
    def __call__(self, input_data: torch.Tensor) -> Any:
        return input_data


class CocoInvalidTargetFilter:
    def __call__(self, input, target) -> Any:
        if len(target) == 0:
            return
        return input, target


class Getorishape(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        target["orig_size"] = [int(h), int(w)]
        return image, target


class YoloLoadImage:
    def __init__(self, img_size: Union[Tuple, List] = (416, 416), model_name="yolov3", device="cpu") -> None:
        self.img_size = img_size
        self.model_name = model_name
        self.device = device

    def pre_process(self, img):
        img = np.array(img)

        # Padded resize
        img = letterbox(img, self.img_size, stride=32, auto=False)[0]

        # Convert
        img = img.transpose((2, 0, 1))  # [::-1]   # HWC to CHW
        img = np.ascontiguousarray(img)

        input = torch.from_numpy(img).unsqueeze(0)
        input = input.to(self.device)

        return input

    def pre_process_yolox(self, img):
        img = np.array(img)[:, :, ::-1]
        padded_img = np.ones((self.img_size[0], self.img_size[1], 3), dtype=np.uint8) * 114
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        input = torch.from_numpy(padded_img).unsqueeze(0)
        input = input.to(self.device)
        return input

    def pre_process_rtdetr(self, img):
        img = np.array(img)

        img = letterbox(img, self.img_size, auto=False, scaleFill=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))  # [::-1]   # HWC to CHW
        img = np.ascontiguousarray(img)

        input = torch.Tensor(img).unsqueeze(0)
        input = input.to(self.device)

        return input

    def pre_process_v4(self, img):
        img = np.array(img.resize(self.img_size))

        # Padded resize
        # img = letterbox(img, self.img_size, stride=32, auto=False)[0]

        # Convert
        img = img.transpose((2, 0, 1))  # [::-1]   # HWC to CHW
        img = np.ascontiguousarray(img)

        input = torch.Tensor(img).unsqueeze(0)
        input = input.to(self.device)

        return input

    def pre_process_yolov3(self, img):
        img = np.array(img)
        imh, imw, imc = img.shape  # original shape

        assert self.img_size[0] == self.img_size[1]
        r = self.img_size[0] / max(imh, imw)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
            img = cv2.resize(img, (math.ceil(imw * r), math.ceil(imh * r)), interpolation=interp)

            h, w = img.shape[:2]

        # Padded resize
        img, ratio, pad = letterbox(img, self.img_size[0], stride=32, auto=False, scaleup=False)

        img = img.transpose((2, 0, 1))  # [::-1]   # HWC to CHW
        img = np.ascontiguousarray(img)

        input = torch.Tensor(img).unsqueeze(0)
        input = input.to(self.device)

        return input

    def __call__(self, img) -> Any:
        if self.model_name in ["yolov5", "yolov7", "yolov8", "yoloworld"]:
            trans_img = self.pre_process(img)
        elif self.model_name in ["yolov3"]:
            trans_img = self.pre_process_yolov3(img)
        elif self.model_name in ["yolov4"]:
            trans_img = self.pre_process_v4(img)
        elif self.model_name in ["rtdetr"]:
            trans_img = self.pre_process_rtdetr(img)
        elif self.model_name in ["yolox"]:
            trans_img = self.pre_process_yolox(img)
        else:
            raise NotImplementedError

        return trans_img

    def __repr__(self):
        return self.__class__.__name__ + "()"


if __name__ == "__main__":
    img = cv2.imread("demo/data/dog.jpg", 1)
    data = torch.from_numpy(img).permute((2, 0, 1)).float()

    data_1 = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

    yuv_data0 = BGR2YUV()(data)

    yuv_data1 = RGB2YUV()(data_1)

    diff = yuv_data0 - yuv_data1
    print(diff.abs().max())
