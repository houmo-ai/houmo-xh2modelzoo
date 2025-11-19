import os


def check_path(datset_name: str, defult_path: str):
    if os.environ.get(datset_name):
        return os.environ.get(datset_name)
    else:
        return defult_path


IMAGNET_DATASET_PATH = check_path("IMAGNET_DATASET_PATH", "data/datasets/imagenet")
IMAGNET_DATASET_MEAN = [0.485, 0.456, 0.406]
IMAGNET_DATASET_STD = [0.229, 0.224, 0.225]

COCO_DATASET_VAL_PATH = check_path("COCO_DATASET_VAL_PATH", "data/datasets/coco/coco2017/val2017")
COCO_JSON_VAL_PATH = check_path("COCO_JSON_VAL_PATH", "data/datasets/coco/coco2017/annotations/instances_val2017.json")

COCO_DATASET_TRAIN_PATH = check_path("COCO_DATASET_TRAIN_PATH", "data/datasets/coco/coco2017/train2017")
COCO_JSON_TRAIN_PATH = check_path(
    "COCO_JSON_TRAIN_PATH", "data/datasets/coco/coco2017/annotations/instances_train2017.json"
)

KITTI_DATASET_PATH = check_path("KITTI_DATASET_PATH", "data/datasets/kitti2015/")
KITTI_DATASET_VAL_INFOS = check_path("KITTI_DATASET_VAL_INFOS", "data/datasets/kitti2015/kitti_infos_val.pkl")

NUSCENE_DATASET_PATH = check_path("NUSCENE_DATASET_PATH", "data/datasets/nuScenes")
NUSCENE_DATASET_VAL_INFOS = check_path(
    "NUSCENE_DATASET_VAL_INFOS", "data/datasets/nuscenes/nuscenes_infos_val_mono3d.coco.json"
)

NUSENCE_BEVDET_DATASET_PATH = check_path("NUSCENE_DATASET_PATH", "data/datasets/nuScenes_bevdet")
NUSENCE_BEVDET_V2_DATASET_VAL_INFOS = check_path(
    "NUSCENE_DATASET_VAL_INFOS", "data/datasets/nuScenes_bevdet/bevdetv2-nuscenes_infos_val.pkl"
)

NUSENCE_BEVDET_V1_DATASET_PATH = check_path("NUSCENE_DATASET_PATH", "data/nuscenes/")
NUSENCE_BEVDET_V1_DATASET_VAL_INFOS = check_path("NUSCENE_DATASET_VAL_INFOS", "data/nuscenes/nuscenes_infos_val.pkl")

NUSCENE_MAPTR_DATASET_PATH = check_path("NUSCENE_DATASET_PATH", "data/datasets/nuScenes")
NUSCENE_MAPTR_DATASET_VAL_INFOS = check_path(
    "NUSCENE_DATASET_VAL_INFOS", "data/datasets/nuScenes/nuscenes_infos_temporal_val.pkl"
)
NUSCENE_MAPTR_DATASET_MAP_INFOS = check_path(
    "NUSCENE_MAPTR_DATASET_MAP_INFOS", "data/datasets/nuScenes/nuscenes_map_anns_val.json"
)

CITYSCAPES_DATASET_VAL_PATH = check_path("CITYSCAPES_DATASET_VAL_PATH", "data/datasets/cityscapes/leftImg8bit/val")
CITYSCAPES_DATASET_VAL_GT_PATH = check_path("CITYSCAPES_DATASET_VAL_GT_PATH", "data/datasets/cityscapes/gtFine/val")


TRACKING_DATA_PATH = check_path("TRACKING_DATA_PATH", "data/datasets/tracking")

CRUISE_GO_DATA_PATH = check_path("CRUISE_GO_DATA_PATH", "data/datasets/cruise/cruise_go_vehicle_model")
CRUISE_CUTIN_DATA_PATH = check_path("CRUISE_CUTIN_DATA_PATH", "data/datasets/cruise/cruise_cutin_vehicle_model")

POSENET_DATA_PATH = check_path("POSENET_DATA_PATH", "data/datasets/posenet_data")

BDD100k_PATH = check_path("BDD100k_PATH", "data/datasets/bdd100k")

ADT_DATA_PATH = check_path("ADT_DATA_PATH", "/data01/datasets/obstacle_3d_benchmark/")

LFW_DATA_PATH = check_path("LFW_DATA_PATH", "data/datasets/lfw")

n = 8 if os.cpu_count() * 2 >= 8 else os.cpu_count() * 2
n_workers = check_path("number_workers", n)


def update_imagenet_dataset_path(new_path: str):
    global IMAGNET_DATASET_PATH  # 声明全局变量
    IMAGNET_DATASET_PATH = new_path
    print(f"Dataset path updated to: {IMAGNET_DATASET_PATH}")


def update_coco_dataset_path(new_path: str, data_path: str, json_path: str):
    global COCO_DATASET_VAL_PATH  # 声明全局变量
    global COCO_JSON_VAL_PATH  # 声明全局变量
    COCO_DATASET_VAL_PATH = new_path
    COCO_JSON_VAL_PATH = json_path
    print(f"Dataset path updated to: {COCO_DATASET_VAL_PATH}")
