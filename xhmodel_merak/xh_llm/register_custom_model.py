import importlib
import os
import sys


def register_custom_model(model_dir):
    model_dir = os.path.abspath(model_dir)
    parent_dir = os.path.dirname(model_dir)
    module_name = os.path.basename(model_dir)

    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    importlib.import_module(module_name)
