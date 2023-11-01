import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt


# 定义一个三角形的遮罩，用于自注意力机制
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # 创建一个上三角矩阵，主对角线以下的元素为False
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        # 初始化AnomalyAttention模块
        super(AnomalyAttention, self).__init__()

        # 缩放参数，用于调整注意力分数的大小
        self.scale = scale

        # 控制是否使用遮罩（mask）的标志
        self.mask_flag = mask_flag

        # 控制是否输出注意力信息的标志
        self.output_attention = output_attention

        # 用于随机失活（dropout）的层
        self.dropout = nn.Dropout(attention_dropout)

        # 定义窗口大小（window_size）
        window_size = win_size

        # 创建一个用于存储距离信息的矩阵，初始化为全零
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                # 计算绝对距离并填充到self.distances中
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        # 获取输入张量的形状信息
        B, L, H, E = queries.shape  # B：batch大小，L：序列长度，H：头数，E：嵌入维度
        _, S, _, D = values.shape  # S：序列长度，D：嵌入维度

        # 缩放参数，用于调整注意力分数的大小
        scale = self.scale or 1. / sqrt(E)

        # 计算注意力分数（scores）：queries与keys之间的点积
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 如果mask_flag为True且没有提供attn_mask，则使用TriangularCausalMask
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 缩放注意力分数，得到注意力矩阵（attn）
        attn = scale * scores

        # 将sigma转置，改变其维度顺序，以匹配后续计算
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L

        # 根据sigma计算权重，并对其进行平滑处理
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1

        # 将sigma扩展维度，以与注意力矩阵（attn）的形状匹配
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L

        # 计算先验信息（prior），用于调整注意力权重
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        # 对注意力分数进行随机失活（dropout），并应用softmax函数得到注意力权重
        series = self.dropout(torch.softmax(attn, dim=-1))

        # 使用注意力权重对values进行加权求和
        V = torch.einsum("bhls,bshd->blhd", series, values)

        # 根据output_attention标志，决定是否返回注意力信息
        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        # 初始化AttentionLayer模块
        super(AttentionLayer, self).__init()

        # 如果未提供d_keys和d_values，则计算它们的默认值
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # 使用LayerNorm层对输入进行归一化
        self.norm = nn.LayerNorm(d_model)

        # 保存内部的注意力机制（AnomalyAttention）模块
        self.inner_attention = attention

        # 创建线性投影层，将输入queries映射到查询向量空间
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)

        # 创建线性投影层，将输入keys映射到键向量空间
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)

        # 创建线性投影层，将输入values映射到值向量空间
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # 创建线性投影层，用于计算sigma（标准差）的值
        self.sigma_projection = nn.Linear(d_model, n_heads)

        # 创建线性投影层，将加权和的值映射回原始维度
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        # 保存头的数量
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # 获取输入张量的形状信息
        B, L, _ = queries.shape  # B：batch大小，L：序列长度
        _, S, _ = keys.shape  # S：序列长度

        # 获取头的数量
        H = self.n_heads

        # 复制输入queries，用于计算sigma
        x = queries

        # 使用线性投影层将输入queries映射到查询向量空间，并重新组织维度
        queries = self.query_projection(queries).view(B, L, H, -1)

        # 使用线性投影层将输入keys映射到键向量空间，并重新组织维度
        keys = self.key_projection(keys).view(B, S, H, -1)

        # 使用线性投影层将输入values映射到值向量空间，并重新组织维度
        values = self.value_projection(values).view(B, S, H, -1)

        # 使用线性投影层计算sigma的值，并重新组织维度
        sigma = self.sigma_projection(x).view(B, L, H)

        # 使用内部的AnomalyAttention计算注意力权重和加权和
        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )

        # 重新组织维度以得到最终的输出
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
