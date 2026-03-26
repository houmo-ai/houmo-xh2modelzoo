import onnx
import numpy as np
from onnx import TensorProto, helper

def modify_onnx_initializer_dtype(onnx_path, save_path, target_initializer_name):
    """
    修改ONNX模型中指定Initializer的类型从float16改为int32，并保存新模型
    :param onnx_path: 原始ONNX模型路径
    :param save_path: 修改后模型的保存路径
    :param target_initializer_name: 目标Initializer的名称（如"model.sub.scalar"）
    """
    # 1. 加载ONNX模型（设置load_external_data=False避免外部数据依赖）
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph  # 获取模型计算图

    # 2. 定位目标Initializer
    target_initializer = None
    for init in graph.initializer:
        if init.name == target_initializer_name:
            target_initializer = init
            break

    if target_initializer is None:
        raise ValueError(f"未找到名称为 {target_initializer_name} 的Initializer")

    # 3. 验证原始数据类型是否为float16
    if target_initializer.data_type != TensorProto.FLOAT16:
        raise TypeError(f"目标Initializer的原始类型不是float16，当前类型为 {TensorProto.DataType.Name(target_initializer.data_type)}")

    # 4. 提取float16数据并转换为int32
    # ---- 4.1 读取float16张量数据 ----
    # ONNX的FLOAT16数据存储在float16_data中，转换为numpy数组
    float16_data = np.frombuffer(target_initializer.raw_data, dtype=np.float16)
    # 注意：float16转int32需根据业务需求处理（如四舍五入/取整/直接转换）
    int32_data = float16_data.astype(np.int32)  # 直接转换（也可使用np.round/floor等）

    # ---- 4.2 清空原始数据并设置新的int32数据 ----
    # 清空原始float16数据
    # target_initializer.float16_data.clear()

    target_initializer.raw_data = b""  # 清空二进制字节流
    # target_initializer.float16_data.clear()  # 清空float16专属字段
    # target_initializer.int32_data.clear()    # 清空int32专属字段（防止冲突）
    # 6.2 设置新的数据类型为INT32
    target_initializer.data_type = TensorProto.INT32
    # 6.3 将int32数组序列化为二进制字节流，写入raw_data
    target_initializer.raw_data = int32_data.tobytes()
    try:
        onnx.checker.check_model(model)
        print("模型修改后校验通过，无语法错误")
    except onnx.checker.ValidationError as e:
        print(f"模型修改后校验失败：{e}")
        return

    # 7. 保存修改后的模型
    onnx.save(model, save_path)
    print(f"模型已保存至 {save_path}，成功将 {target_initializer_name} 的类型从float16改为int32")

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换为你的ONNX路径、保存路径和目标Initializer名称
    ONNX_INPUT_PATH = "/data01/home/xuchen/xh2/xhquant_llm/work_dirs/kimi_a3b_30b_instruct_lagacy_xh2a_2k_hmonnx/golden/prefill/hmquant_kimi_a3b_30b_instruct_legacy_xh2a_2k_batch_eval_prefill_with_act.onnx"
    ONNX_OUTPUT_PATH = "/data01/home/xuchen/xh2/xhquant_llm/work_dirs/kimi_a3b_30b_instruct_lagacy_xh2a_2k_hmonnx/golden/prefill/hmquant_kimi_a3b_30b_instruct_legacy_xh2a_2k_batch_eval_prefill_with_act.onnx"
    TARGET_INIT_NAME = "model.sub.scalar"  # 目标Initializer名称

    modify_onnx_initializer_dtype(
        onnx_path=ONNX_INPUT_PATH,
        save_path=ONNX_OUTPUT_PATH,
        target_initializer_name=TARGET_INIT_NAME
    )