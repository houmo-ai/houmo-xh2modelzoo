from . import register


def register_custom_model(model_dir):
    register.CUSTOM_MODELS.append(model_dir)
