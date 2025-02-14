import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def exists(val):
    """
    检查一个值是否存在（即不为None）。

    Args:
        val: 任意类型的值。

    Returns:
        bool: 如果值不为None，则返回True；否则返回False。
    """
    return val is not None


def default(val, d):
    """
    如果值存在（即不为None），则返回该值；否则，返回默认值。

    Args:
        val: 任意类型的值。
        d: 默认值。

    Returns:
        任意类型: 如果val不为None，则返回val；否则返回d。
    """
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    """
    计算用于保持输入输出尺寸相同的填充大小。

    在卷积操作中，为了保持输入和输出的空间尺寸相同，
    需要在输入的边缘进行适当的填充。

    Args:
        kernel_size (int): 卷积核的大小。

    Returns:
        tuple: 包含填充大小的元组，格式为 (pad_left, pad_right)。
    """
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    """
    Swish激活函数类。

    Swish是一种自门控激活函数，定义为：f(x) = x * sigmoid(x)。
    它在某些深度学习任务中比ReLU表现更好。

    Args:
        None

    Returns:
        torch.Tensor: 经过Swish激活函数处理后的张量。
    """
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    """
    GLU（Gated Linear Unit）激活函数类。

    GLU是一种门控机制，将输入张量分成两部分，一部分作为门控信号，另一部分作为输出信号。

    Args:
        dim (int): 分割维度的索引。

    Returns:
        torch.Tensor: 经过GLU激活函数处理后的张量。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 将输入张量沿指定维度分割成两部分
        out, gate = x.chunk(2, dim=self.dim)
        # 应用门控机制：输出 = out * sigmoid(gate)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    """
    深度可分离卷积1D类。

    深度可分离卷积将标准卷积分解为深度卷积和逐点卷积，
    以减少计算量和参数量。

    Args:
        chan_in (int): 输入通道数。
        chan_out (int): 输出通道数。
        kernel_size (int): 卷积核的大小。
        padding (int 或 tuple): 填充大小。

    Returns:
        torch.Tensor: 经过深度可分离卷积处理后的张量。
    """
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        # 定义深度可分离卷积层，groups=chan_in表示每个输入通道有独立的卷积核
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        # 对输入张量进行填充
        x = F.pad(x, self.padding)
        # 应用深度可分离卷积
        return self.conv(x)


class Scale(nn.Module):
    """
    缩放层类。

    该层对输入张量应用一个缩放因子。

    Args:
        scale (float): 缩放因子。
        fn (nn.Module): 要应用的神经网络模块。

    Returns:
        torch.Tensor: 经过缩放处理后的张量。
    """
    def __init__(self, scale, fn):
        super().__init__()
        # 要应用的神经网络模块
        self.fn = fn
        # 缩放因子
        self.scale = scale

    def forward(self, x, **kwargs):
        # 应用神经网络模块，并乘以缩放因子
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    """
    预归一化层类。

    该层在应用某个神经网络模块之前，对输入张量进行层归一化。

    Args:
        dim (int): 归一化的维度。
        fn (nn.Module): 要应用的神经网络模块。

    Returns:
        torch.Tensor: 经过预归一化处理后的张量。
    """
    def __init__(self, dim, fn):
        super().__init__()
        # 要应用的神经网络模块
        self.fn = fn
        # 层归一化层
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        # 对输入张量进行层归一化
        x = self.norm(x)
        # 应用神经网络模块
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    """
    自注意力机制（Self-Attention）实现。

    该类实现了多头自注意力机制（Multi-Head Self-Attention），并支持相对位置编码。
    它可以用于各种Transformer模型中，以捕捉输入序列中不同位置之间的关系。

    Args:
        dim (int): 输入和输出的维度大小。
        heads (int, optional): 多头注意力的头数，默认为8。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        dropout (float, optional): Dropout概率，默认为0。
        max_pos_emb (int, optional): 最大位置嵌入尺寸，默认为512。
    """
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        # 计算内部维度，用于线性变换
        inner_dim = dim_head * heads
        # 多头注意力的头数
        self.heads= heads
        # 缩放因子，用于缩放注意力得分
        self.scale = dim_head ** -0.5
        # 线性变换层，用于计算查询（Q）
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 线性变换层，用于计算键（K）和值（V）
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        # 线性变换层，用于输出投影
        self.to_out = nn.Linear(inner_dim, dim)

        # 最大位置嵌入尺寸
        self.max_pos_emb = max_pos_emb
        # 相对位置嵌入层，用于计算相对位置编码
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        """
        前向传播方法，执行多头自注意力计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
            context (torch.Tensor, optional): 上下文张量，用于跨注意力机制。如果为None，则使用x作为上下文。
            mask (torch.Tensor, optional): 输入张量的掩码，用于屏蔽某些位置。
            context_mask (torch.Tensor, optional): 上下文张量的掩码，用于屏蔽某些位置。

        Returns:
            torch.Tensor: 经过多头自注意力处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        # 如果没有提供上下文张量，则使用输入张量作为上下文
        context = default(context, x)

        # 计算查询（Q）、键（K）和值（V）
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # 重塑张量以适应多头注意力机制，形状变为 (batch_size, heads, sequence_length, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 计算注意力得分（未缩放的）
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Shaw的相对位置编码实现
        seq = torch.arange(n, device = device) # 生成序列索引
        # 计算距离矩阵，形状为 (sequence_length, sequence_length)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        # 将距离限制在 [-max_pos_emb, max_pos_emb] 之间，并加上偏移量
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        # 获取相对位置嵌入，形状为 (2 * max_pos_emb + 1, dim_head)
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        # 计算相对位置注意力得分，形状为 (batch_size, heads, sequence_length, sequence_length)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        # 将相对位置注意力得分加到原始注意力得分上
        dots = dots + pos_attn

        # 处理掩码
        if exists(mask) or exists(context_mask):
            # 如果没有提供输入掩码，则使用全1掩码
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            # 如果没有提供上下文掩码，则使用全1掩码或输入掩码
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            # 获取掩码填充值（最小浮点数）
            mask_value = -torch.finfo(dots.dtype).max
            # 重塑掩码以匹配注意力得分的形状
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            # 对注意力得分进行掩码填充，掩码为0的位置填充为mask_value
            dots.masked_fill_(~mask, mask_value)

        # 对注意力得分应用softmax函数，得到注意力权重
        attn = dots.softmax(dim = -1)

        # 通过注意力权重对值进行加权求和，得到输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # 重塑输出张量的形状为 (batch_size, sequence_length, inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过线性变换层进行输出投影
        out = self.to_out(out)
        # 应用Dropout正则化
        return self.dropout(out)


class FeedForward(nn.Module):
    """
    前馈神经网络（Feed-Forward Network, FFN）类。

    该网络通常用于Transformer模型中，作为多头注意力机制后的位置前馈网络。
    它由线性变换、激活函数和Dropout层组成。

    Args:
        dim (int): 输入和输出的维度大小。
        mult (int, optional): 内部维度相对于输入维度的倍数，默认为4。
        dropout (float, optional): Dropout概率，默认为0。

    Returns:
        torch.Tensor: 经过前馈神经网络处理后的张量。
    """
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        # 定义前馈神经网络的序列结构
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), # 线性变换，扩展维度
            Swish(), # 应用Swish激活函数
            nn.Dropout(dropout), # 应用Dropout正则化
            nn.Linear(dim * mult, dim), # 线性变换，恢复原始维度
            nn.Dropout(dropout) # 应用Dropout正则化
        )

    def forward(self, x):
        """
        前向传播方法，执行前馈神经网络的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        Returns:
            torch.Tensor: 经过前馈神经网络处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        return self.net(x)


# ===========================
# Conformer 卷积模块类（ConformerConvModule）
# ===========================

class ConformerConvModule(nn.Module):
    """
    Conformer模型的卷积模块类。

    该模块实现了Conformer模型中的卷积部分，用于捕捉局部特征。
    它结合了层归一化、1D卷积、门控线性单元（GLU）、深度可分离卷积、批量归一化、Swish激活函数等。

    Args:
        dim (int): 输入和输出的维度大小。
        causal (bool, optional): 是否为因果卷积，默认为False。
        expansion_factor (int, optional): 扩展因子，用于扩展内部维度，默认为2。
        kernel_size (int, optional): 卷积核的大小，默认为31。
        dropout (float, optional): Dropout概率，默认为0。

    Returns:
        torch.Tensor: 经过卷积模块处理后的张量。
    """
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()
        # 计算内部维度
        inner_dim = dim * expansion_factor
        # 计算填充大小，如果是因果卷积，则只填充左侧；否则进行相同填充
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        # 定义卷积模块的序列结构
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 对输入进行层归一化
            Rearrange('b n c -> b c n'),  # 重排张量形状以适应1D卷积
            nn.Conv1d(dim, inner_dim * 2, 1),  # 1D卷积，扩展内部维度
            GLU(dim=1),  # 应用GLU激活函数
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),  # 深度可分离卷积
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),  # 如果不是因果卷积，则应用批量归一化；否则使用恒等映射
            Swish(),  # 应用Swish激活函数
            nn.Conv1d(inner_dim, dim, 1),  # 1D卷积，恢复原始维度
            Rearrange('b c n -> b n c'),  # 重排张量形状以恢复原始形状
            nn.Dropout(dropout)  # 应用Dropout正则化
        )

    def forward(self, x):
        """
        前向传播方法，执行卷积模块的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        Returns:
            torch.Tensor: 经过卷积模块处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        return self.net(x)


class ConformerBlock(nn.Module):
    """
    Conformer模块块。

    该模块块结合了前馈神经网络（FFN）、多头自注意力机制（Multi-Head Self-Attention）和卷积模块（ConformerConvModule），
    并通过层归一化和残差连接来增强模型的表达能力。

    Args:
        dim (int): 输入和输出的维度大小。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为8。
        ff_mult (int, optional): 前馈神经网络内部维度相对于输入维度的倍数，默认为4。
        conv_expansion_factor (int, optional): 卷积模块的扩展因子，默认为2。
        conv_kernel_size (int, optional): 卷积核的大小，默认为31。
        attn_dropout (float, optional): 多头自注意力机制的Dropout概率，默认为0。
        ff_dropout (float, optional): 前馈神经网络的Dropout概率，默认为0。
        conv_dropout (float, optional): 卷积模块的Dropout概率，默认为0。
        conv_causal (bool, optional): 是否为因果卷积，默认为False。

    Returns:
        torch.Tensor: 经过Conformer模块块处理后的张量。
    """
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        # 定义第一个前馈神经网络（FFN）
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # 定义多头自注意力机制
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # 定义卷积模块
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # 定义第二个前馈神经网络（FFN）
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # 使用预归一化（PreNorm）对注意力机制和前馈神经网络进行归一化处理
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        # 定义层归一化层，用于最后的输出归一化
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        """
        前向传播方法，执行Conformer模块块的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
            mask (torch.Tensor, optional): 输入张量的掩码，用于屏蔽某些位置。

        Returns:
            torch.Tensor: 经过Conformer模块块处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        # 应用第一个前馈神经网络，并添加残差连接
        x = self.ff1(x) + x
        # 应用多头自注意力机制，并添加残差连接
        x = self.attn(x, mask = mask) + x
        # 应用卷积模块，并添加残差连接
        x = self.conv(x) + x
        # 应用第二个前馈神经网络，并添加残差连接
        x = self.ff2(x) + x
        # 应用层归一化
        x = self.post_norm(x)
        return x


class Conformer(nn.Module):
    """
    Conformer模型类。

    该模型由多个Conformer模块块堆叠而成，能够处理序列数据，如语音识别、自然语言处理等任务。

    Args:
        dim (int): 输入和输出的维度大小。
        depth (int): Conformer模块块的堆叠深度。
        dim_head (int, optional): 每个注意力头的维度大小，默认为64。
        heads (int, optional): 多头注意力的头数，默认为8。
        ff_mult (int, optional): 前馈神经网络内部维度相对于输入维度的倍数，默认为4。
        conv_expansion_factor (int, optional): 卷积模块的扩展因子，默认为2。
        conv_kernel_size (int, optional): 卷积核的大小，默认为31。
        attn_dropout (float, optional): 多头自注意力机制的Dropout概率，默认为0。
        ff_dropout (float, optional): 前馈神经网络的Dropout概率，默认为0。
        conv_dropout (float, optional): 卷积模块的Dropout概率，默认为0。
        conv_causal (bool, optional): 是否为因果卷积，默认为False。

    Returns:
        torch.Tensor: 经过Conformer模型处理后的张量。
    """
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        # 设置维度大小
        self.dim = dim
        # 初始化模块列表，用于存储多个Conformer模块块
        self.layers = nn.ModuleList([])

        # 堆叠多个Conformer模块块
        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal

            ))

    def forward(self, x):
        """
        前向传播方法，执行Conformer模型的计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。

        Returns:
            torch.Tensor: 经过Conformer模型处理后的输出张量，形状为 (batch_size, sequence_length, dim)。
        """
        # 遍历所有Conformer模块块，并应用它们到输入张量
        for block in self.layers:
            x = block(x)

        return x
