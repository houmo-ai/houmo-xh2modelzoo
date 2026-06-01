# Copyright 2025 HOUMO AI
#
# File: test_rope2.py
# Description:
#   Example script: llm/zimage/test_rope2.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

# 固定随机种子（保证结果可复现）
torch.manual_seed(42)

token_len = 32

# 模拟输入
x_in = torch.randn(1, token_len, 30, 128)  
roper = torch.randn(1, token_len, 1, 64)  # cos，最后一维64
ropei = torch.randn(1, token_len, 1, 64)  # sin，最后一维64

def get_correct_rearrange_matrix(feature_dim=128):
    """
    生成和原函数完全等价的置换矩阵
    关键修正：调整矩阵的行/列顺序，匹配原函数的重排逻辑
    """
    half_dim = feature_dim // 2
    # 1. 构造重排后的目标索引（和原函数完全一致）
    # 目标顺序：[0,2,4,...,126, 1,3,...,127]
    odd_indices = torch.arange(0, feature_dim, 2)  # [0,2,...,126] (64个)
    even_indices = torch.arange(1, feature_dim, 2) # [1,3,...,127] (64个)
    target_idx = torch.cat([odd_indices, even_indices], dim=0)  # shape [128]
    
    # 2. 生成正确的置换矩阵（关键修正：按行索引重排，而非列）
    # 置换矩阵P的定义：P[i,j] = 1 当且仅当 输出的第i维 = 输入的第j维
    perm_matrix = torch.zeros((feature_dim, feature_dim))
    for i in range(feature_dim):
        # 输出的第i维 对应 输入的第target_idx[i]维
        perm_matrix[i, target_idx[i]] = 1.0
    
    return perm_matrix

P = get_correct_rearrange_matrix(128)

# ==================== 关键步骤：重排x_in的维度顺序 ====================
def rearrange_x_for_original_rotate(x):
    """
    重排x的最后一维，让原始_rotate_half匹配复数旋转逻辑
    输入x.shape: (1, 32, 30, 128)
    重排逻辑：把最后一维从 [1,2,3,4,...,127,128] → [1,3,...,127, 2,4,...,128]
    """
    # 步骤1：拆分最后一维为奇数位（实部）和偶数位（虚部）
    # 奇数索引：0,2,4,...126 → 对应实部，共64维
    x_odd = x[..., ::2]  # shape (1,32,30,64)
    # 偶数索引：1,3,5,...127 → 对应虚部，共64维
    x_even = x[..., 1::2]  # shape (1,32,30,64)
    
    # 步骤2：拼接成新的x → 前64维是奇数位（实部），后64维是偶数位（虚部）
    x_rearranged = torch.cat([x_odd, x_even], dim=-1)  # shape不变，仍为128维
    return x_rearranged

# 对输入x进行重排
# x_rearranged = rearrange_x_for_original_rotate(x_in)

x_rearranged = torch.matmul(P, x_in.unsqueeze(-1)).squeeze(-1) # [128,128]

# ==================== 第一段代码：完全使用原始逻辑（无修改） ====================
def _rotate_half(x):
    # 原始算子：对半拆分，旋转拼接
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

# 原始rope扩展方式
cos2 = roper.repeat([1,1,1,2])
sin2 = ropei.repeat([1,1,1,2])

# 注意：这里用重排后的x计算，最后再还原顺序
out1_rearranged = x_rearranged * cos2 + _rotate_half(x_rearranged) * sin2

# 还原x的顺序：把前64维（奇数位）和后64维（偶数位）还原为原始的交错顺序
def restore_x_order(x_rearranged):
    """
    把重排后的x还原为原始顺序
    输入：[1,3,...,127, 2,4,...,128] → 输出：[1,2,3,4,...,127,128]
    """
    x_odd = x_rearranged[..., :64]  # 前64维：奇数位
    x_even = x_rearranged[..., 64:] # 后64维：偶数位
    
    # 交错拼接：创建新维度，再展平
    x_restored = torch.stack([x_odd, x_even], dim=-1).flatten(-2)
    return x_restored

def get_restore_matrix(feature_dim=128):
    """
    生成和restore_x_order完全等价的还原置换矩阵
    Returns:
        restore_matrix: 形状为[feature_dim, feature_dim]的置换矩阵
    """
    half_dim = feature_dim // 2
    # 1. 构造还原的目标索引（逆操作）
    # 输入重排后的维度：[0,1,2,...,63, 64,65,...,127] → 对应原始的[0,2,4,...,126, 1,3,...,127]
    # 还原目标：输出维度i → 若i为偶数，取输入的i//2；若i为奇数，取输入的half_dim + (i//2)
    restore_idx = torch.zeros(feature_dim, dtype=torch.long)
    for i in range(feature_dim):
        if i % 2 == 0:
            # 偶数位（0,2,4...）→ 取重排后前64维的第i//2位
            restore_idx[i] = i // 2
        else:
            # 奇数位（1,3,5...）→ 取重排后后64维的第i//2位
            restore_idx[i] = half_dim + (i // 2)
    
    # 2. 生成还原置换矩阵（显式定义：输出i对应输入restore_idx[i]）
    restore_matrix = torch.zeros((feature_dim, feature_dim))
    for i in range(feature_dim):
        restore_matrix[i, restore_idx[i]] = 1.0
    
    return restore_matrix

restore_matrix = get_restore_matrix(128) 

# 还原顺序得到最终的out1
# out1 = restore_x_order(out1_rearranged)
out1 = torch.matmul(restore_matrix, out1_rearranged.unsqueeze(-1)).squeeze(-1)

# ==================== 第二段代码：原始复数旋转逻辑（无修改） ====================
x_reshaped = x_in.reshape(1, token_len, 30,-1,2)
x_real = x_reshaped[...,0]
x_imag = x_reshaped[...,1]
out_real = x_real * roper - x_imag * ropei
out_imag = x_imag * roper + x_real * ropei 
out2 = torch.stack([out_real, out_imag], dim=-1).flatten(3)

# ==================== 验证结果一致性 ====================
mean_error = (out1 - out2).abs().mean()
is_close = torch.allclose(out1, out2, atol=1e-6)

print(f"均值误差: {mean_error:.10f}")  # ≈1e-15（纯浮点误差）
print(f"是否一致: {is_close}")  # 输出True