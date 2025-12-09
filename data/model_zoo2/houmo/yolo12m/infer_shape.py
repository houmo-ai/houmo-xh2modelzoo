import onnx

def remove_all_shape_info(onnx_model_path, output_path):
    # 加载ONNX模型
    model = onnx.load(onnx_model_path)
    
    # 移除输入张量的shape信息
    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.HasField('shape'):
            # 清除shape字段
            input_tensor.type.tensor_type.ClearField('shape')
    
    # 移除输出张量的shape信息
    for output_tensor in model.graph.output:
        if output_tensor.type.tensor_type.HasField('shape'):
            output_tensor.type.tensor_type.ClearField('shape')
    
    # 移除中间张量的shape信息
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.HasField('shape'):
            value_info.type.tensor_type.ClearField('shape')
    
    # 保存修改后的模型
    onnx.save(model, output_path)
    print(f"已移除所有shape信息，保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    input_model = "data/model_zoo2/houmo/yolo12m/yolo12m_batch8.onnx"   # 输入模型路径
    output_model = "data/model_zoo2/houmo/yolo12m/yolo12m_batch8.onnx" # 输出模型路径
    remove_all_shape_info(input_model, output_model)
