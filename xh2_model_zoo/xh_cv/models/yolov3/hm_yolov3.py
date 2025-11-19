import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
import torchvision.transforms as transforms

from xh2_model_zoo.datasets.Coco.coco import Cus_CocoLoaderGenerator, eval_coco
from xh2_model_zoo.datasets.Coco.evaluate import COCO80_TO_COCO91
from xh2_model_zoo.datasets.data_path import COCO_DATASET_VAL_PATH, COCO_JSON_VAL_PATH, n_workers
from xh2_model_zoo.datasets.preprocess.transform import YoloLoadImage
from xh2_model_zoo.utils.box_utils import (
    letterbox,
    non_max_suppression,
    scale_coords,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywh,
    xyxy2xywhn,
)
from xh2_model_zoo.utils.helpers import get_onnx_from_url
from xh2_model_zoo.utils.onnx.onnx_run import OnnxRuntimeDetector

__all__ = ["HMYolov3"]

cache_path = os.path.join(str(Path(__file__).parent), "val2017.cache")


class HMYolov3:
    def __init__(
        self, model_name=None, device="cpu", onnx_without_postprocess=True, input_shape=None, method="hquant"
    ) -> None:
        self.device = device
        self.input_shape = input_shape
        self._dataset = None
        self.onnx_without_postprocess = onnx_without_postprocess
        self.model = self._load_model(model_name)
        self._forward = self._init_forward(self.model_path)
        self.img_shape = []
        self.jdict = []
        self.im_files = []
        self.images_ids = []
        self.class_map = COCO80_TO_COCO91

        # 为训练特殊的加载数据方式预留
        self.shapes = None
        self.batch_shapes = None
        self.iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

        self.method = method
        self.model_name = model_name

    def _init_forward(self, model):
        return OnnxRuntimeDetector(model, self.device).forward_single_input

    def _load_model(self, model_name):
        self.model_path = model_name

        if model_name.endswith(".onnx"):
            return onnx.load(model_name)
        elif model_name == "yolov3_640x640":
            self.input_shape = (640, 640)
            if not self.onnx_without_postprocess:
                url = "http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolov3/yolov3.onnx"
                model, self.model_path = get_onnx_from_url(url, return_path=True, company="houmo/yolov3")
            else:
                url = "http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolov3/yolov3_without_ptprocess.onnx"
                model, self.model_path = get_onnx_from_url(url, return_path=True, company="houmo/yolov3")
            return model
        elif model_name == "yolov3_416x416":
            self.input_shape = (416, 416)
            url = "http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolov3/yolov3_416x416.onnx"
            model, self.model_path = get_onnx_from_url(url, return_path=True, company="houmo/yolov3")
            return model
        else:
            raise NotImplementedError(f"Unknown model name of" % (model_name))

    def forward(self, data):
        # data: torch.Tensor with shape (n, c, h, w)
        if self.model_path.endswith(".onnx"):
            data = (data.float() / 255.0).cpu().numpy().astype(np.float32)
            nn_out = self._forward(data)
            return torch.tensor(nn_out)

    def get_batch_shape(self, batch_size=1, net_size=640, stride=32, pad=0.5):
        cache, exists = np.load(cache_path, allow_pickle=True).item(), True
        nf, nm, ne, nc, n = cache.pop("results")
        [cache.pop(k) for k in ("hash", "version", "msgs")]
        labels, shapes, segments = zip(*cache.values())

        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.im_files = [COCO_DATASET_VAL_PATH + i[-17:] for i in self.im_files]
        # self.label_files = img2label_paths(cache.keys())  # update

        self.shapes = np.array(shapes)

        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        if False:  # 对图片按H/W重新排序,每个batch拥有不同shape
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            # self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            # self.segments = [self.segments[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * net_size / stride + pad).astype(int) * stride

    def pre_process(self, img, labels):
        img = np.array(img)
        self.img_shape = []
        imh, imw, imc = img.shape  # original shape

        r = self.input_shape[0] / max(imh, imw)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
            img = cv2.resize(img, (math.ceil(imw * r), math.ceil(imh * r)), interpolation=interp)

        h, w = img.shape[:2]

        # shape = self.batch_shapes[self.batch[index]] #if self.rect else self.input_shape
        # Padded resize
        img, ratio, pad = letterbox(img, self.input_shape[0], stride=32, auto=False, scaleup=False)
        self.img_shape.append([(imh, imw), ((h / imh, w / imw), pad)])

        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nl = len(labels)  # number labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))  # [::-1]  # BHWC to BCHW
        img = np.ascontiguousarray(img)

        input = torch.Tensor(img).unsqueeze(0)
        # input = input.to(torch.float32)-128
        input = input.to(self.device)
        return input, labels_out

    def post_process(
        self,
        pre_out,
        batch_nn_out,
        batch_target,
        num_classes=80,
        thresh=0.001,
        nms_thresh=0.6,
        path=None,
        draw=False,
    ):
        nb, _, height, width = pre_out.shape
        batch_target = batch_target.to(self.device)
        batch_target[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # to pixels

        if self.onnx_without_postprocess:
            batch_nn_out = [self.post_box_process(batch_nn_out)]

        preds = non_max_suppression(
            batch_nn_out[0],
            thresh,
            nms_thresh,
            labels=[],
            multi_label=True,
        )

        for si, pred in enumerate(preds):
            labels = batch_target[batch_target[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(path[si]), self.img_shape[si][0]
            predn = pred.clone()
            scale_coords(pre_out[si].shape[1:], predn[:, :4], shape, self.img_shape[si][1])  # native-space pred

            image_id = int(Path(path).stem)
            self.images_ids.append(image_id)

            if npr == 0:
                continue
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(pre_out[si].shape[1:], tbox, shape, self.img_shape[si][1])  # native-space labels
                # labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                # correct = process_batch(predn, labelsn, self.iouv)

            if draw:
                import cv2

                img = cv2.imread(str(path))
                for p_pred in predn:
                    x1, y1, x2, y2, score, cls = p_pred
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.imwrite("demo.jpg", img)
            self.save_one_json(predn, self.jdict, path, self.class_map)

    def evaluate(self):
        # images_ids = [int(Path(x).stem) for x in self.im_files]
        res = eval_coco(self.jdict, self.images_ids)[0]
        return res

    def dataset(
        self, calib_transform=None, test_transform=None, calib_num=2, test_batch_size=1, shuffle=False, subset=None
    ):
        self.get_batch_shape(batch_size=test_batch_size)  # only support batchsize =1

        if calib_transform == None:
            calib_transform = transforms.Compose(
                [
                    YoloLoadImage(model_name="yolov3", img_size=self.input_shape),
                    # lambda  x: torch.tensor(x).to(torch.float32)-128
                ]
            )

        if self._dataset is None:
            self._dataset = Cus_CocoLoaderGenerator(
                dataset_name="coco_yolov3",
                test_batch_size=test_batch_size,
                num_workers=n_workers,
                device=self.device,
                calib_transform=calib_transform,
                test_transform=test_transform,
                im_file_paths=self.im_files,
                labels=self.labels,
            )

        calib = self._dataset.calib_loader(calib_num=calib_num, shuffle=shuffle)
        test_set = self._dataset.test_loader(batch_size=test_batch_size, subset=subset)
        if self.method == "tensorrt":
            calib_set = []
            for _, input in enumerate(calib):
                calib_set.append(input[0] / 255)
        else:
            calib_set = calib
        return calib_set, test_set

    def save_one_json(self, predn, jdict, path, class_map):
        # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            jdict.append(
                {
                    "image_id": image_id,
                    "category_id": class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def post_box_process(self, in_features):
        # in.shape = out.shape: 1x3x80x80x85 1x3x40x40x85 1x3x20x20x85
        output = []

        for i in range(3):
            assert len(in_features[i].shape) == 5

            bs, channel, ny, nx, no = in_features[i].shape
            grid, anchor_grid = self._make_grid(nx, ny, i)

            in_features[i][..., 0:2] = (in_features[i][..., 0:2] * 2 - 0.5 + grid) * self.stride[i]  # xy
            in_features[i][..., 2:4] = (in_features[i][..., 2:4] * 2) ** 2 * anchor_grid  # wh

            output.append(in_features[i].reshape(bs, -1, no))

        return torch.concat(output, dim=1)

    def _make_grid(self, nx=20, ny=20, i=0):
        anchors = torch.tensor(
            [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        )

        self.stride = torch.tensor([8, 16, 32]).view(-1, 1, 1).to(self.device)
        anchors = anchors.view(3, 3, 2).to(self.device)
        anchors = anchors / self.stride

        yv, xv = torch.meshgrid([torch.arange(ny, device=self.device), torch.arange(nx, device=self.device)])
        grid = torch.stack((xv, yv), 2).expand((1, 3, ny, nx, 2)).float()
        anchor_grid = (anchors[i] * self.stride[i]).view((1, 3, 1, 1, 2)).expand((1, 3, ny, nx, 2)).float()
        return grid, anchor_grid
