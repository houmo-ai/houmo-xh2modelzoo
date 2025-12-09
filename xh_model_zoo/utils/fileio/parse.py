import warnings
from io import StringIO

from .file_client import FileClient
from .io import get_text


def list_from_file(
    filename,
    prefix="",
    offset=0,
    max_num=0,
    encoding="utf-8",
    file_client_args=None,
    backend_args=None,
):
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError('"file_client_args" and "backend_args" cannot be set at the ' "same time.")
    cnt = 0
    item_list = []

    if file_client_args is not None:
        file_client = FileClient.infer_client(file_client_args, filename)
        text = file_client.get_text(filename, encoding)
    else:
        text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip("\n\r"))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str, encoding="utf-8", file_client_args=None, backend_args=None):

    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError('"file_client_args" and "backend_args" cannot be set at the ' "same time.")

    mapping = {}

    if file_client_args is not None:
        file_client = FileClient.infer_client(file_client_args, filename)
        text = file_client.get_text(filename, encoding)
    else:
        text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for line in f:
            items = line.rstrip("\n").split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
