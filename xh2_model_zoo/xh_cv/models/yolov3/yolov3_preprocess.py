import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from benchmark.utils.config import Config

logger = logging.getLogger(__name__)


class OnnxYolov3:
    def __init__(self, use_inv_map: bool, image_size: Tuple[int, int] = (416, 416)) -> None:
        self._use_inv_map = use_inv_map
        config = Config()
        dataset_root_path = config.get("DATASETS_PATH")
        data_path = os.path.join(dataset_root_path, "COCO")
        self._annotation_file = os.path.join(
            data_path,
            "annotations/instances_val2017.json",
        )
        if self._use_inv_map:
            # for pytorch
            label_map = {}
            with open(self._annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            self.inv_map = {v: k for k, v in label_map.items()}
        self.yolo_layer = OnnxYolov3.Predictor(image_size)

    class Predictor:
        def __init__(self, image_size: Tuple[int, int]):
            self._image_size = image_size

        def get_tvm_fmt_out(self, output_data: List[np.array], classes: int = 80):
            def sigmoid(x):
                return 1.0 / (1 + np.exp(-x))

            biases = [
                10,
                13,
                16,
                30,
                33,
                23,
                30,
                61,
                62,
                45,
                59,
                119,
                116,
                90,
                156,
                198,
                373,
                326,
            ]
            mask = [
                [6, 7, 8],
                [3, 4, 5],
                [0, 1, 2],
            ]
            total = 9
            tvm_fmt_out = []
            for i in range(3):
                """
                Three kind of resolutions: (13, 13), (26, 26), (52, 52)
                """
                layer_out = {}
                layer_out["type"] = "Yolo"
                out_data = output_data[i]
                # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
                n, out_c, out_h, out_w = out_data.shape
                layer_out["biases"] = biases
                layer_out["mask"] = mask[i]
                out_data_nchw = out_data
                n = 3
                layer_attr = (n, out_c, out_h, out_w, classes, total)
                out_shape = (
                    layer_attr[0],
                    layer_attr[1] // layer_attr[0],
                    layer_attr[2],
                    layer_attr[3],
                )
                out_data_nchw = np.reshape(out_data_nchw, out_shape)
                out_data_nchw_fp = out_data_nchw.astype(np.float32)
                out_data_nchw_fp[:, 0:2, :, :] = sigmoid(
                    out_data_nchw_fp[:, 0:2, :, :],
                )
                out_data_nchw_fp[:, 4:85, :, :] = sigmoid(
                    out_data_nchw_fp[:, 4:85, :, :],
                )
                layer_out["output"] = out_data_nchw_fp
                layer_out["classes"] = layer_attr[4]
                tvm_fmt_out.append(layer_out)
            return tvm_fmt_out

        def post_process(
            self,
            nn_out: List[np.array],
            image_size: Tuple[int, int],
            num_classes: int = 80,
            thresh: float = 0.5,
            nms_thresh: float = 0.45,
        ) -> List[Any]:
            from tvm.relay.testing import yolo_detection

            netw, neth = self._image_size
            im_w, im_h = image_size
            dets = yolo_detection.fill_network_boxes(
                (netw, neth),
                (im_w, im_h),
                thresh,
                1,
                nn_out,
            )
            yolo_detection.do_nms_sort(dets, num_classes, nms_thresh)
            results = []
            for det in dets:
                if (det["prob"] == 0).all() == False:
                    category_id = det["prob"].argmax()
                    score = det["prob"].max()
                    cx, cy, dw, dh = det["bbox"]
                    l = (cx - dw / 2.0) * im_w
                    t = (cy - dh / 2.0) * im_h
                    w = dw * im_w
                    h = dh * im_h
                    if l < 0:
                        l = 0.0
                    if t < 0:
                        t = 0.0
                    bbox = [l, t, w, h]
                    results.append(
                        dict(
                            category_id=category_id,
                            score=score,
                            bbox=bbox,
                        ),
                    )
            return results

    def __call__(self, result: Dict[str, Any], out_datas: List[np.array], labels: List[Dict[str, Any]]) -> None:
        # results come as:
        #   darknet yolov3: detection_classes, detection_scores, (detection_boxes)
        count = 0
        processed_results = []
        image_ids = []
        for out_data, label in zip([out_datas], labels):
            data_out = [out_data[2], out_data[1], out_data[0]]
            net_out = self.yolo_layer.get_tvm_fmt_out(data_out)
            dets = self.yolo_layer.post_process(
                net_out,
                (label["width"], label["height"]),
            )
            detections = []
            # import cv2
            # img_org = cv2.imread(label['file_name'])
            for det in dets:
                detections.append(
                    [
                        float(label["id"]),
                        det["bbox"][0],
                        det["bbox"][1],
                        det["bbox"][2],
                        det["bbox"][3],
                        det["score"],
                        float(det["category_id"] + 1),
                    ]
                )
            #    cv2.rectangle(img_org, (int(det[2][0]), int(det[2][1])), (int(det[2][0]+det[2][2]), int(det[2][1]+det[2][3])), (0, 0, 255), 1)
            #    cv2.putText(img_org, str(det[0]), (int(det[2][0]), int(det[2][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.imwrite('/home/xiongwei/gerrit/test/test.jpg', img_org)
            logger.info(
                "label: %s, predict: %s",
                str(
                    label,
                ),
                str(detections),
            )
            image_ids.append(label["id"])
            processed_results.append(detections)
            count += 1
        result["lock"].acquire()
        result["data"] += processed_results
        result["image_ids"] += image_ids
        result["lock"].release()

    def reset(self, result: Dict[str, Any]) -> None:
        result["lock"].acquire()
        result["data"] = []
        result["image_ids"] = []
        result["lock"].release()

    def summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        detections = []
        image_indices = []
        output_det = result["data"]
        image_indices = list(result["image_ids"])
        for batch in range(0, len(output_det)):
            for idx in range(0, len(output_det[batch])):
                detection = output_det[batch][idx]
                if self._use_inv_map:
                    # Covert the catagory_id from index of categories list to id of item in categories list
                    detection[6] = float(self.inv_map[int(detection[6])])
                detections.append(np.array(detection))
        detections = np.array(detections)
        image_indices.sort(reverse=True)
        # map indices to coco image id's
        cocoGt = COCO(self._annotation_file)
        cocoDt = cocoGt.loadRes(detections)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
        cocoEval.params.imgIds = image_indices
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        accuracy = {"meanap": cocoEval.stats[1]}
        return accuracy
