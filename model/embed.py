import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


"""
这个代码定义了一个用于生成位置编码的模块，它将位置信息添加到输入序列中，以帮助模型处理序列中不同位置的信息。
"""
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # 创建一个位置编码张量，用于为输入序列中的每个位置添加位置信息
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 创建一个表示位置的序列，从0到max_len，然后添加维度以便进行后续计算
        position = torch.arange(0, max_len).float().unsqueeze(1)

        # 计算位置编码的分母项（div_term），其中涉及对数空间的计算
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 使用正弦函数和余弦函数计算位置编码
        # 通过对位置乘以分母项并使用不同的偏移（0::2 和 1::2）来生成正弦和余弦值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将位置编码张量扩展一个维度，并将其作为不可训练的模型参数进行注册
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 返回位置编码张量，截取前x.size(1)个位置的编码
        return self.pe[:, :x.size(1)]


"""
这个代码定义了一个TokenEmbedding模块，用于将输入的词嵌入向量转换为目标维度d_model的表示。
它使用卷积操作来捕捉局部特征，并将输入的维度进行适当的变换，以满足模型的输入要求。
"""
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        # 定义卷积层，用于将输入的词嵌入（token embedding）转换为目标维度d_model
        # 卷积核大小为3，用于捕捉局部特征
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        # 对所有模块进行权重初始化，使用Kaiming初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # 对输入x进行处理，首先将其维度排列为(批大小, 序列长度, 词嵌入维度)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

        # 返回转换后的张量，形状为(批大小, 序列长度, 目标维度d_model)
        return x


"""
这个模块将输入数据进行值嵌入和位置嵌入的处理，最终输出一个包含值嵌入和位置嵌入信息的张量。 Dropout 层有助于降低过拟合风险。
"""
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        # 创建值嵌入（value embedding）模块，将输入数据映射到目标维度d_model
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # 创建位置嵌入（positional embedding）模块，用于添加位置信息到数据
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # 创建一个dropout层，用于在模型训练过程中进行随机失活以防止过拟合
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 通过值嵌入模块将输入x转换为目标维度，并通过位置嵌入模块添加位置信息
        x = self.value_embedding(x) + self.position_embedding(x)

        # 应用dropout操作以减少过拟合风险
        x = self.dropout(x)

        # 返回处理后的张量，其中包含了值嵌入和位置嵌入的信息
        return x

