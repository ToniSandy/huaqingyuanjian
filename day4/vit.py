import torch
from torch import nn

from einops import rearrange, repeat  # 用于方便地操作张量维度
from einops.layers.torch import Rearrange  # 用于构建PyTorch层的维度重排

# 辅助函数

def pair(t):
    """将输入转换为元组形式。如果输入已经是元组则直接返回，否则返回(input, input)"""
    return t if isinstance(t, tuple) else (t, t)

# 定义各个模块类

class FeedForward(nn.Module):
    """前馈神经网络模块"""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """
        参数:
            dim: 输入维度
            hidden_dim: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, hidden_dim),  # 线性变换
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, dim),  # 线性变换回原始维度
            nn.Dropout(dropout)  # Dropout层
        )

    def forward(self, x):
        """前向传播"""
        return self.net(x)

class Attention(nn.Module):
    """自注意力机制模块"""
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """
        参数:
            dim: 输入维度
            heads: 注意力头的数量
            dim_head: 每个注意力头的维度
            dropout: dropout概率
        """
        super().__init__()
        inner_dim = dim_head * heads  # 计算内部维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要投影输出

        self.heads = heads  # 注意力头数量
        self.scale = dim_head ** -0.5  # 缩放因子，用于缩放点积注意力

        self.norm = nn.LayerNorm(dim)  # 层归一化

        self.attend = nn.Softmax(dim = -1)  # softmax层，用于计算注意力权重
        self.dropout = nn.Dropout(dropout)  # dropout层

        # 将输入转换为查询(Q)、键(K)、值(V)的线性层
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 输出投影层，如果需要投影则使用线性层+dropout，否则使用恒等映射
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """前向传播"""
        x = self.norm(x)  # 层归一化

        # 生成Q、K、V，并分割成三部分
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 重排维度: [batch, seq_len, (heads * dim_head)] -> [batch, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # 计算点积注意力得分
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 计算注意力权重
        attn = self.attend(dots)
        attn = self.dropout(attn)  # 应用dropout

        # 应用注意力权重到V上
        out = torch.matmul(attn, v)
        # 重排维度: [batch, heads, seq_len, dim_head] -> [batch, seq_len, heads * dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # 投影输出

class Transformer(nn.Module):
    """Transformer模块，包含多个注意力层和前馈层"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """
        参数:
            dim: 输入维度
            depth: Transformer的层数
            heads: 注意力头的数量
            dim_head: 每个注意力头的维度
            mlp_dim: 前馈网络的隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 最终的层归一化
        self.layers = nn.ModuleList([])  # 存储各层的列表
        for _ in range(depth):
            # 每层包含一个注意力模块和一个前馈模块
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        """前向传播，使用残差连接"""
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力层 + 残差连接
            x = ff(x) + x  # 前馈层 + 残差连接

        return self.norm(x)  # 返回归一化后的结果

class ViT(nn.Module):
    """Vision Transformer (ViT)模型"""
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        """
        参数:
            image_size: 图像尺寸(高度,宽度)或单一尺寸
            patch_size: 分块尺寸(高度,宽度)或单一尺寸
            num_classes: 分类类别数
            dim: 嵌入维度
            depth: Transformer层数
            heads: 注意力头数量
            mlp_dim: 前馈网络隐藏层维度
            pool: 池化方式，'cls'或'mean'
            channels: 图像通道数
            dim_head: 每个注意力头的维度
            dropout: dropout概率
            emb_dropout: 嵌入层的dropout概率
        """
        super().__init__()
        # 处理图像尺寸和分块尺寸
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 验证图像尺寸能否被分块尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算分块数量和每个分块的维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 将图像转换为分块嵌入的序列
        self.to_patch_embedding = nn.Sequential(
            # 重排维度: [batch, channels, height, width] -> [batch, num_patches, patch_dim]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),  # 层归一化
            nn.Linear(patch_dim, dim),  # 线性投影到嵌入维度
            nn.LayerNorm(dim),  # 层归一化
        )

        # 位置嵌入和类别标记
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 可学习的位置嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别标记
        self.dropout = nn.Dropout(emb_dropout)  # 嵌入层的dropout

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool  # 池化方式
        self.to_latent = nn.Identity()  # 恒等映射(占位符，可用于添加更多处理)

        # MLP分类头
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """前向传播"""
        # 将图像转换为分块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # 获取batch大小和分块数量

        # 重复类别标记以匹配batch大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # 将类别标记连接到分块嵌入前面
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)  # 应用dropout

        # 通过Transformer编码器
        x = self.transformer(x)

        # 池化: 均值池化或使用类别标记
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)  # 恒等映射
        return self.mlp_head(x)  # 分类预测
