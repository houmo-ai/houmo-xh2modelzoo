import hashlib
from enum import Enum
from pathlib import Path
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class CaseInsensitiveEnum(StrEnum):
    """
    一个支持大小写不敏感成员查找的枚举类。
    """

    @classmethod
    def _missing_(cls, value):
        """
        重写 _missing_ 方法以实现大小写不敏感的查找。
        """
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return super()._missing_(value)

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return self.name.lower()


def calculate_file_md5(file_path: str | Path) -> str:
    """计算文件的 MD5 哈希值。

    Args:
        file_path: 文件路径

    Returns:
        文件的 MD5 哈希值（16 进制字符串）
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def find_non_jsonable_paths(obj: Any, path: str = "root") -> list[tuple[str, str, str]]:
    bad = []

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return bad

    if isinstance(obj, dict):
        for k, v in obj.items():
            # 检查 key
            if not isinstance(k, str):
                bad.append((f"{path}.<key {repr(k)}>", type(k).__name__, repr(k)))
            bad.extend(find_non_jsonable_paths(v, f"{path}[{repr(k)}]"))
        return bad

    if isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            bad.extend(find_non_jsonable_paths(item, f"{path}[{i}]"))
        if isinstance(obj, set):
            bad.append((path, "set", repr(obj)))
        return bad

    if isinstance(obj, bytes):
        bad.append((path, "bytes", repr(obj)))
        return bad

    # try:
    #     json.dumps(obj)
    # except TypeError:
    #     bad.append((path, type(obj).__name__, repr(obj)))

    return bad
