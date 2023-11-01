import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding

"""
这个模块定义了编码器层，它包含了自注意力层和卷积层。
自注意力层用于捕捉输入序列的依赖关系，而卷积层用于进行局部特征的处理。注意力权重、掩码、和sigma值由自注意力层输出，
然后将其与原始输入相加并应用Layer Normalization来更新输入。
随后，通过卷积层和激活函数处理输入，再次将其与原始输入相加并应用Layer Normalization来获得最终的输出。 Dropout层用于降低过拟合风险。
"""
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()

        # 设置Feed-Forward网络的维度，默认为4倍的目标维度d_model
        d_ff = d_ff or 4 * d_model

        # 定义自注意力层（attention）模块
        self.attention = attention

        # 定义第一个卷积层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)

        # 定义第二个卷积层
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # 定义Layer Normalization层，用于标准化输入数据
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 创建一个dropout层，用于在模型训练过程中进行随机失活
        self.dropout = nn.Dropout(dropout)

        # 设置激活函数，默认为ReLU或GELU
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 使用自注意力层计算注意力权重、注意力掩码、和sigma值
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        # 更新输入张量x，将自注意力层的输出与原始输入相加，并应用dropout
        x = x + self.dropout(new_x)

        # 复制一份输入x到y，并通过卷积和激活函数进行处理
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 最终更新输入x，将卷积层的输出与原始输入相加，并应用Layer Normalization
        return self.norm2(x + y), attn, mask, sigma


"""
这个模块定义了一个编码器（Encoder），它由一系列编码器层组成。
每个编码器层通过自注意力机制对输入数据进行编码，同时记录了注意力系列信息、先验信息和sigma值。
最终的编码器输出可以经过可选的Layer Normalization层进行标准化。
"""
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()

        # 定义一系列编码器层（attn_layers）作为模型的主要组件
        self.attn_layers = nn.ModuleList(attn_layers)

        # 定义可选的Layer Normalization层，用于标准化编码器层的输出
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]

        # 用于存储每个编码器层输出的注意力系列信息、先验信息和sigma值
        series_list = []
        prior_list = []
        sigma_list = []

        # 遍历每个编码器层，计算输出，并保存注意力信息
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        # 如果指定了Layer Normalization层，对最终输出进行标准化
        if self.norm is not None:
            x = self.norm(x)

        # 返回最终的编码器输出，以及每个编码器层的注意力系列信息、先验信息和sigma值
        return x, series_list, prior_list, sigma_list


"""
这个模块定义了一个AnomalyTransformer模型，它包括了数据嵌入、编码器和输出投影层。
数据嵌入将输入数据进行值嵌入和位置嵌入，编码器通过多个编码器层处理嵌入数据，最后通过投影层映射到目标维度。模型可以选择是否输出注意力信息。
"""
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()

        # 设置是否输出注意力信息
        self.output_attention = output_attention

        # 创建数据嵌入（DataEmbedding）模块，用于将输入数据进行值嵌入和位置嵌入
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # 创建编码器（Encoder）模块，由多个编码器层（EncoderLayer）组成
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 创建输出投影层，将编码器的输出映射到目标维度c_out
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        # 通过数据嵌入模块处理输入数据，得到值嵌入和位置嵌入
        enc_out = self.embedding(x)

        # 通过编码器处理值嵌入和位置嵌入，得到编码器的输出
        enc_out, series, prior, sigmas = self.encoder(enc_out)

        # 通过投影层将编码器的输出映射到目标维度c_out
        enc_out = self.projection(enc_out)

        # 如果设置为输出注意力信息，返回编码器输出、注意力系列信息、先验信息和sigma值
        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            # 否则，只返回编码器输出
            return enc_out  # [B, L, D]
