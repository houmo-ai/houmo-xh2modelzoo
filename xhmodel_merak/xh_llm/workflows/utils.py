import os


def same_abs_path(path1: str, path2: str) -> bool:
    return os.path.abspath(os.path.normpath(str(path1))) == os.path.abspath(os.path.normpath(str(path2)))
