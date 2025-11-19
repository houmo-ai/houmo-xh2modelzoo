from xhquant.utils.registry import Registry

MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry(
    "transform",
)
DATA_SAMPLERS = Registry("data sampler")

EVALUATOR = Registry("evaluator")
METRICS = Registry("metrics")
FUNCTIONS = Registry("functions")
INFERENCE_ENGINES = Registry("inference engine")


def build_model_from_cfg(cfg):
    return MODELS.build(cfg)


def build_dataset_from_cfg(cfg):
    return DATASETS.build(cfg)
