import torch
import torch.nn as nn

# ===================== 核心参数（含29维度，固定不变） =====================
batch_size = 1
seq_len = 29  # 重点关注的序列长度维度
input_dim = 2048
total_output_dim = 3072  # 16*192=3072
group_num = 16
split_dim1 = 128  # 最后一维拆分1
split_dim2 = 64   # 最后一维拆分2
sub_output_dim1 = group_num * split_dim1  # 16*128=2048
sub_output_dim2 = group_num * split_dim2  # 16*64=1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# 强制校验维度关系（避免计算错误）
assert sub_output_dim1 + sub_output_dim2 == total_output_dim, "总输出维度不匹配"
assert split_dim1 + split_dim2 == 192, "单组维度不匹配"
assert seq_len == 29, "序列长度维度被意外修改"

# ===================== 1. 定义层（强制连续+设备/dtype对齐） =====================
# 原始大Linear
linear_big = nn.Linear(input_dim, total_output_dim, bias=True).to(device, dtype)
# 拆分小Linear
linear1 = nn.Linear(input_dim, sub_output_dim1, bias=True).to(device, dtype)
linear2 = nn.Linear(input_dim, sub_output_dim2, bias=True).to(device, dtype)

# ===================== 2. 严格拆分参数（带29维度无关，但必须正确） =====================
with torch.no_grad():
    # 权重拆分
    W_big = linear_big.weight.data.contiguous()  # 强制连续
    linear1.weight.data = W_big[:sub_output_dim1, :].clone().contiguous()
    linear2.weight.data = W_big[sub_output_dim1:, :].clone().contiguous()
    # 偏置拆分
    b_big = linear_big.bias.data.contiguous()
    linear1.bias.data = b_big[:sub_output_dim1].clone().contiguous()
    linear2.bias.data = b_big[sub_output_dim1:].clone().contiguous()

# 参数校验
print("=== 【参数校验】===")
print(f"linear1权重形状: {linear1.weight.shape} (预期: [2048,2048])")
print(f"linear2权重形状: {linear2.weight.shape} (预期: [1024,2048])")
print(f"参数拆分是否一致: {torch.allclose(linear1.weight.data, W_big[:2048,:], atol=1e-6)}")

# ===================== 3. 构造输入（固定29维度） =====================
x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype).contiguous()
print(f"\n=== 【输入校验】===")
print(f"输入形状: {x.shape} (预期: [1,29,2048])")
print(f"输入seq_len维度长度: {x.shape[1]} (预期:29)")

# ===================== 4. 原始流程（强制连续+29维度专项校验） =====================
def original_process(x, linear):
    # 大Linear前向
    out_big = linear(x).contiguous()  # 强制连续
    print(f"\n=== 【原始流程-大Linear】===")
    print(f"out_big形状: {out_big.shape} (预期: [1,29,3072])")
    print(f"out_big seq_len维度长度: {out_big.shape[1]} (预期:29)")
    
    # view重塑：重点保证29在第二个维度
    out_view = out_big.view(batch_size, seq_len, group_num, split_dim1 + split_dim2).contiguous()
    print(f"\n=== 【原始流程-view】===")
    print(f"out_view形状: {out_view.shape} (预期: [1,29,16,192])")
    print(f"out_view seq_len维度索引1的长度: {out_view.shape[1]} (预期:29)")
    print(f"out_view group_num维度索引2的长度: {out_view.shape[2]} (预期:16)")
    
    # transpose：仅交换1（29）和2（16）维度
    out_trans = out_view.transpose(1, 2).contiguous()
    print(f"\n=== 【原始流程-transpose】===")
    print(f"out_trans形状: {out_trans.shape} (预期: [1,16,29,192])")
    print(f"out_trans 交换后seq_len维度索引2的长度: {out_trans.shape[2]} (预期:29)")
    
    # split拆分
    out1_ori, out2_ori = torch.split(out_trans, [split_dim1, split_dim2], dim=-1)
    return out1_ori, out2_ori, out_big, out_view, out_trans

out1_ori, out2_ori, out_big, out_view, out_trans = original_process(x, linear_big)

# ===================== 5. 拆分流程（强制连续+29维度专项校验） =====================
def split_process(x, linear1, linear2):
    # 小Linear前向
    out1_linear = linear1(x).contiguous()
    out2_linear = linear2(x).contiguous()
    print(f"\n=== 【拆分流程-小Linear】===")
    print(f"out1_linear形状: {out1_linear.shape} (预期: [1,29,2048])")
    print(f"out2_linear形状: {out2_linear.shape} (预期: [1,29,1024])")
    print(f"out1_linear seq_len维度长度: {out1_linear.shape[1]} (预期:29)")
    
    # view重塑：严格对齐原始流程的维度顺序（batch,29,16,128）
    out1_view = out1_linear.view(batch_size, seq_len, group_num, split_dim1).contiguous()
    out2_view = out2_linear.view(batch_size, seq_len, group_num, split_dim2).contiguous()
    print(f"\n=== 【拆分流程-view】===")
    print(f"out1_view形状: {out1_view.shape} (预期: [1,29,16,128])")
    print(f"out2_view形状: {out2_view.shape} (预期: [1,29,16,64])")
    print(f"out1_view seq_len维度索引1的长度: {out1_view.shape[1]} (预期:29)")
    
    # transpose：仅交换1（29）和2（16）维度，与原始流程一致
    out1_split = out1_view.transpose(1, 2).contiguous()
    out2_split = out2_view.transpose(1, 2).contiguous()
    print(f"\n=== 【拆分流程-transpose】===")
    print(f"out1_split形状: {out1_split.shape} (预期: [1,16,29,128])")
    print(f"out2_split形状: {out2_split.shape} (预期: [1,16,29,64])")
    print(f"out1_split 交换后seq_len维度索引2的长度: {out1_split.shape[2]} (预期:29)")
    
    return out1_split, out2_split, out1_linear, out2_linear

out1_split, out2_split, out1_linear, out2_linear = split_process(x, linear1, linear2)

# ===================== 6. 最终校验（含29维度逐元素校验） =====================
print(f"\n=== 【最终结果校验】===")
# 整体结果校验
print(f"128维部分形状是否一致: {out1_ori.shape == out1_split.shape} (预期:True)")
print(f"64维部分形状是否一致: {out2_ori.shape == out2_split.shape} (预期:True)")
print(f"128维部分数值是否一致: {torch.allclose(out1_ori, out1_split, atol=1e-6)}")
print(f"64维部分数值是否一致: {torch.allclose(out2_ori, out2_split, atol=1e-6)}")

# 29维度专项校验：随机取一个seq_len位置（如第10个），校验元素是否一致
random_seq_idx = 10  # 随机选29中的一个位置（0~28）
print(f"\n=== 【29维度专项校验】- 取第{random_seq_idx}个序列位置 ===")
print(f"原始128维该位置数值最大: {out1_ori[0, :, random_seq_idx, :].max().item():.6f}")
print(f"拆分128维该位置数值最大: {out1_split[0, :, random_seq_idx, :].max().item():.6f}")
print(f"原始64维该位置数值最大: {out2_ori[0, :, random_seq_idx, :].max().item():.6f}")
print(f"拆分64维该位置数值最大: {out2_split[0, :, random_seq_idx, :].max().item():.6f}")
print(f"128维该位置数值是否一致: {torch.allclose(out1_ori[0, :, random_seq_idx, :], out1_split[0, :, random_seq_idx, :], atol=1e-6)}")
print(f"64维该位置数值是否一致: {torch.allclose(out2_ori[0, :, random_seq_idx, :], out2_split[0, :, random_seq_idx, :], atol=1e-6)}")

# 最大误差校验
print(f"\n=== 【误差校验】===")
print(f"128维部分最大误差: {(out1_ori - out1_split).abs().max().item():.8f}")
print(f"64维部分最大误差: {(out2_ori - out2_split).abs().max().item():.8f}")