import os

COCO80_ID2NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_NAME_MAP = {
    "0": 0,  # person
    "1": 1,  # bicycle
    "2": 2,  # car
    "3": 3,  # motorcycle
    "4": 4,  # airplane
    "5": 5,  # bus
    "6": 6,  # train
    "7": 7,  # truck
    "8": 8,  # boat
    "9": 9,  # traffic light
    "10": 10,  # fire hydrant
    "11": 12,  # stop sign
    "12": 13,  # parking meter
    "13": 14,  # bench
    "14": 15,  # bird
    "15": 16,  # cat
    "16": 17,  # dog
    "17": 18,  # horse
    "18": 19,  # sheep
    "19": 20,  # cow
    "20": 21,  # elephant
    "21": 22,  # bear
    "22": 23,  # zebra
    "23": 24,  # giraffe
    "24": 26,  # backpack
    "25": 27,  # umbrella
    "26": 30,  # handbag
    "27": 31,  # tie
    "28": 32,  # suitcase
    "29": 33,  # frisbee
    "30": 34,  # skis
    "31": 35,  # snowboard
    "32": 36,  # sports ball
    "33": 37,  # kite
    "34": 38,  # baseball bat
    "35": 39,  # baseball glove
    "36": 40,  # skateboard
    "37": 41,  # surfboard
    "38": 42,  # tennis racket
    "39": 43,  # bottle
    "40": 45,  # wine glass
    "41": 46,  # cup
    "42": 47,  # fork
    "43": 48,  # knife
    "44": 49,  # spoon
    "45": 50,  # bowl
    "46": 51,  # banana
    "47": 52,  # apple
    "48": 53,  # sandwich
    "49": 54,  # orange
    "50": 55,  # broccoli
    "51": 56,  # carrot
    "52": 57,  # hot dog
    "53": 58,  # pizza
    "54": 59,  # donut
    "55": 60,  # cake
    "56": 61,  # chair
    "57": 62,  # couch
    "58": 63,  # potted plant
    "59": 64,  # bed
    "60": 66,  # dining table
    "61": 69,  # toilet
    "62": 71,  # tv
    "63": 72,  # laptop
    "64": 73,  # mouse
    "65": 74,  # remote
    "66": 75,  # keyboard
    "67": 76,  # cell phone
    "68": 77,  # microwave
    "69": 78,  # oven
    "70": 79,  # toaster
    "71": 80,  # sink
    "72": 81,  # refrigerator
    "73": 83,  # book
    "74": 84,  # clock
    "75": 85,  # vase
    "76": 86,  # scissors
    "77": 87,  # teddy bear
    "78": 88,  # hair drier
    "79": 89,  # toothbrush
}

# converts 80-index (val2014) to 91-index (paper)
# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
# b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
# x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
# x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
COCO80_TO_COCO91 = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]

COCO91 = [i for i in range(91)]


class Evaluator:
    pass


class Coco(Evaluator):
    """
    COCO Data format: https://cocodataset.org/#format-data
    """

    def __init__(self, annFile, resType="bbox"):
        """
        Params:
          @dataDir, eg., '/datasets/coco/coco2017'
          @annFile, annotation file path
          @resType, one of ['segm', 'bbox', 'keypoints']
        """
        self.resType = resType

        # initialize COCO ground truth api
        from pycocotools.coco import COCO

        self.GT = COCO(annFile)

    def run(self, results, imgIds=None):
        """
        Params:
          @results, eg.:
            '''
              [{'image_id': 289343,
                'bbox': [473.07, 395.93, 38.65, 28.67],
                'category_id': 18,
                'score': 0.99},
               {...}
               ......]
            '''
          @imgIds, its type can be:
            1. slice, eg., 0:2 ---- take the first two elements
            2. list, eg., [image-id1, imgage-id2, ...]
        """
        # if (type(results) == str) and (results[-4:] == '.txt'):
        #     TODO
        self.DT = self.GT.loadRes(results)
        from pycocotools.cocoeval import COCOeval

        cocoEval = COCOeval(self.GT, self.DT, self.resType)

        if imgIds == None:
            cocoEval.params.imgIds = self.GT.getImgIds()
        elif type(imgIds) == slice:
            cocoEval.params.imgIds = self.GT.getImgIds()[imgIds]
        elif type(imgIds) == list:
            cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats


def build_evaluator(cfg):
    dataset = cfg.dataset
    if dataset["name"] == "Coco":
        return Coco(dataset["val_anno"], cfg.evaluation["metric"]["type"])
