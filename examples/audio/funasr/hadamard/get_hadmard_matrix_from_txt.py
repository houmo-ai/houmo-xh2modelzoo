import torch

def get_had180_from_file(file_path="had.180.pal.txt"):
    """
    从指定的文本文件中读取 180x180 Hadamard 矩阵数据并转换为 PyTorch tensor。

    参数:
    file_path (str): 包含 Hadamard 矩阵数据的文本文件路径。

    返回:
    torch.FloatTensor: 形状为 (180, 180) 的 PyTorch tensor。
    """
    matrix = []
    expected_size = 180
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                # 去除行首尾的空白字符（包括换行符）
                cleaned_line = line.strip()

                # 将当前行转换为数值列表
                row = []
                for char in cleaned_line:
                    if char == '+':
                        row.append(1.0)
                    elif char == '-':
                        row.append(-1.0)
                
                # 只有当行包含有效数据时才添加到矩阵中
                if row:
                    # 检查行长度是否符合预期
                    if len(row) != expected_size:
                        raise ValueError(f"在文件 {file_path} 的第 {line_num + 1} 行，数据长度不正确。期望 {expected_size}，得到 {len(row)}。")
                    matrix.append(row)
            
        # 检查最终矩阵的行数是否符合预期
        if len(matrix) != expected_size:
            raise ValueError(f"从文件 {file_path} 读取的数据行数不正确。期望 {expected_size} 行，得到 {len(matrix)} 行。")
            
        return torch.FloatTensor(matrix)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {file_path}")
    except IOError as e:
        raise IOError(f"读取文件 {file_path} 时发生错误: {e}")


if __name__ == "__main__":
    try:
        # 请确保 'had.180.pal.txt' 文件与该脚本在同一目录下，
        # 或者提供文件的完整路径。
        hadamard_tensor = get_had180_from_file("had.180.pal.txt")
        print(f"Hadamard 矩阵的形状: {hadamard_tensor.shape}")
        print(f"数据类型: {hadamard_tensor.dtype}")
        print(f"前几行前几列:\n{hadamard_tensor[:5, :5]}")
        
        # 可选：验证这是否是一个 Hadamard 矩阵 (H * H^T = 180 * I)
        # 注意：这会消耗大量内存和计算时间，仅用于验证
        product = torch.mm(hadamard_tensor, hadamard_tensor.t())
        identity_scaled = 180 * torch.eye(180)
        is_hadamard = torch.allclose(product, identity_scaled, atol=1e-6)
        print(f"验证是否为 Hadamard 矩阵 (H*H^T == 180*I): {is_hadamard}")

    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"错误: {e}")
