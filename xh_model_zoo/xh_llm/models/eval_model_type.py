from enum import Enum, auto


class EvalModelType(Enum):
    NONE = auto()
    WRAPED = auto()  # 由改写后的模型
    FRONTEND = auto()  # 由原始模型得到的计算图, 在nn.Module中运行
    QUANTED_DISABLED = auto()  # 在QModule中运行,但禁用量化计算，浮点计算，
    QUANTED_ALIGNED = auto()  # 运行真正量化计算，在QModule中运行，可做Golden
    QUANTED_FAST = auto()  # 在QModule中运行, 准确度介于Fake和Aligned之间
    EXPORTED = auto()  # 用于导出的计算图，做golden data
    CALIBRATION = auto  # 标定QModule

    def __str__(self):
        # return self.name.capitalize()
        return self.name

    def __repr__(self) -> str:
        return self.value
