import torch
from conformer import ConformerConvModule, ConformerBlock, Conformer


# 实例化一个 ConformerConvModule 模块
layer = ConformerConvModule(
    dim = 512,                  # 输入和输出的维度大小，设置为512
    causal = False,             # 是否为因果卷积 - 如果为True，则1D卷积将通过填充变为因果卷积；此处设置为False
    expansion_factor = 2,       # 扩展因子，用于扩展深度可分离卷积的维度，设置为2
    kernel_size = 31,           # 卷积核大小，设置为31。根据研究，17到31被认为是最佳范围
    dropout = 0.                # 在卷积模块末尾应用的Dropout概率，设置为0，即不使用Dropout
)

# 创建一个随机张量，形状为 (批次大小=1, 序列长度=1024, 特征维度=512)
x = torch.randn(1, 1024, 512)

# 将卷积模块的输出与原始输入张量相加，实现残差连接
x = layer(x) + x


# 实例化一个 ConformerBlock 模块
block = ConformerBlock(
    dim = 512,                   # 输入和输出的维度大小，设置为512
    dim_head = 64,               # 每个注意力头的维度大小，设置为64
    heads = 8,                   # 多头注意力的头数，设置为8
    ff_mult = 4,                 # 前馈神经网络内部维度相对于输入维度的倍数，设置为4
    conv_expansion_factor = 2,   # 卷积模块的扩展因子，设置为2
    conv_kernel_size = 31,       # 卷积核大小，设置为31
    attn_dropout = 0.,           # 多头自注意力机制的Dropout概率，设置为0
    ff_dropout = 0.,             # 前馈神经网络的Dropout概率，设置为0
    conv_dropout = 0.            # 卷积模块的Dropout概率，设置为0
)

# 创建一个随机张量，形状为 (批次大小=1, 序列长度=1024, 特征维度=512)
x = torch.randn(1, 1024, 512)

# 将输入张量 `x` 输入到 ConformerBlock 模块中，处理后的输出形状仍为 (1, 1024, 512)
block(x) # (1, 1024, 512)


# 实例化一个完整的 Conformer 模型
conformer = Conformer(
    dim = 512,                   # 输入和输出的维度大小，设置为512
    depth = 12,                  # 堆叠的 ConformerBlock 模块数量，设置为12
    dim_head = 64,               # 每个注意力头的维度大小，设置为64
    heads = 8,                   # 多头注意力的头数，设置为8
    ff_mult = 4,                 # 前馈神经网络内部维度相对于输入维度的倍数，设置为4
    conv_expansion_factor = 2,   # 卷积模块的扩展因子，设置为2
    conv_kernel_size = 31,       # 卷积核大小，设置为31
    attn_dropout = 0.,           # 多头自注意力机制的Dropout概率，设置为0
    ff_dropout = 0.,             # 前馈神经网络的Dropout概率，设置为0
    conv_dropout = 0.            # 卷积模块的Dropout概率，设置为0
)

# 创建一个随机张量，形状为 (批次大小=1, 序列长度=1024, 特征维度=512)
x = torch.randn(1, 1024, 512)

# 将输入张量 `x` 输入到 Conformer 模型中，处理后的输出形状仍为 (1, 1024, 512)
conformer(x) # (1, 1024, 512)
