import torch

def scatter_reduce_mean(dim, index, src, out_shape):
    # 创建一个和输出形状相同的零张量来存储累加结果
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # 创建一个和输出形状相同的零张量来存储索引计数
    count = torch.zeros(out_shape, dtype=torch.long, device=src.device)

    # 使用scatter_add_累加索引位置的值
    out.scatter_add_(dim, index, src)

    # 使用scatter_add_累加索引位置出现的次数
    count.scatter_add_(dim, index, torch.ones_like(src, dtype=torch.long))

    # 避免除以零，将计数为零的位置设置为1
    count[count < 1] = 1

    # 计算平均值
    out = out / count.float()  # 确保除法是浮点数除法

    return out

@torch.jit.script_if_tracing
def scatter_reduce_sum(ldesc, lines_junc_idx, prob):
    # 初始化结果张量
    denom = torch.zeros_like(ldesc[:, 0], dtype=prob.dtype)

    # 累加: 遍历每个元素，进行索引累加
    for i in range(lines_junc_idx.size(0)):
        for j in range(lines_junc_idx.size(1)):
            if lines_junc_idx[i, j] != i:  # 排除自身
                denom[lines_junc_idx[i, j]] += prob[i, j]

    return denom