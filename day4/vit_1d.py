

# classes

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack  # 用于张量维度操作
from einops.layers.torch import Rearrange  # 用于构建PyTorch层的维度重排


# 定义神经网络模块类

class FeedForward(nn.Module):
    """前馈神经网络模块（MLP）"""

    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        参数:
            dim: 输入/输出维度
            hidden_dim: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),  # 层归一化
            nn.Linear(dim, hidden_dim),  # 扩展维度
            nn.GELU(),  # GELU激活函数
            nn.Dropout(dropout),  # 随机失活
            nn.Linear(hidden_dim, dim),  # 降回原始维度
            nn.Dropout(dropout)  # 随机失活
        )

    def forward(self, x):
        """前向传播：输入x形状为(batch_size, seq_len, dim)"""
        return self.net(x)


class Attention(nn.Module):
    """多头自注意力机制模块"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        参数:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个头的维度
            dropout: dropout概率
        """
        super().__init__()
        inner_dim = dim_head * heads  # 计算内部总维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影

        self.heads = heads  # 注意力头数
        self.scale = dim_head ** -0.5  # 缩放因子(1/√d_k)

        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.attend = nn.Softmax(dim=-1)  # 注意力权重计算
        self.dropout = nn.Dropout(dropout)  # 注意力dropout

        # 生成Q,K,V的线性变换(合并计算效率更高)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 输出投影层(如果需要)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """前向传播"""
        x = self.norm(x)  # 先进行层归一化

        # 生成Q,K,V并分割 [3, batch, heads, seq_len, dim_head]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 重排维度: [batch, seq_len, heads*dim_head] -> [batch, heads, seq_len, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算点积注意力得分
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 计算注意力权重
        attn = self.attend(dots)
        attn = self.dropout(attn)  # 应用dropout

        # 应用注意力权重到V上
        out = torch.matmul(attn, v)
        # 合并多头输出: [batch, heads, seq_len, dim_head] -> [batch, seq_len, heads*dim_head]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # 输出投影


class Transformer(nn.Module):
    """Transformer编码器(堆叠多个注意力+前馈层)"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        参数:
            dim: 特征维度
            depth: Transformer层数
            heads: 注意力头数
            dim_head: 每个头的维度
            mlp_dim: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.layers = nn.ModuleList([])  # 存储各Transformer层
        for _ in range(depth):
            # 每层包含注意力+前馈网络(带残差连接)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """前向传播(带残差连接)"""
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力+残差
            x = ff(x) + x  # 前馈网络+残差
        return x


class ViT(nn.Module):
    """用于时间序列的Vision Transformer模型"""

    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
        参数:
            seq_len: 输入序列长度
            patch_size: 每个patch的长度
            num_classes: 分类类别数
            dim: 嵌入维度
            depth: Transformer层数
            heads: 注意力头数
            mlp_dim: 前馈网络隐藏层维度
            channels: 输入通道数(默认为3)
            dim_head: 每个注意力头的维度
            dropout: Transformer内部dropout
            emb_dropout: 嵌入层dropout
        """
        super().__init__()
        # 验证序列长度能被patch大小整除
        assert (seq_len % patch_size) == 0, '序列长度必须能被patch大小整除'

        num_patches = seq_len // patch_size  # 计算patch数量
        patch_dim = channels * patch_size  # 每个patch的维度

        # 将序列分割为patch并嵌入
        self.to_patch_embedding = nn.Sequential(
            # 重排维度: [batch, channels, seq] -> [batch, num_patches, patch_dim]
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),  # 层归一化
            nn.Linear(patch_dim, dim),  # 线性投影到嵌入维度
            nn.LayerNorm(dim),  # 层归一化
        )

        # 位置嵌入(可学习参数)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 类别标记(可学习参数)
        self.cls_token = nn.Parameter(torch.randn(dim))  # 形状为(dim,)
        self.dropout = nn.Dropout(emb_dropout)  # 嵌入层dropout

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 最终层归一化
            nn.Linear(dim, num_classes)  # 线性分类层
        )

    def forward(self, series):
        """前向传播"""
        # 将输入序列转换为patch嵌入 [batch, num_patches, dim]
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape  # 获取batch大小和patch数量

        # 为每个样本重复cls_token [batch, dim] -> [batch, 1, dim]
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)

        # 将cls_token与patch嵌入拼接 [batch, num_patches+1, dim]
        x, ps = pack([cls_tokens, x], 'b * d')

        # 添加位置嵌入
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)  # 应用dropout

        # 通过Transformer编码器
        x = self.transformer(x)

        # 解包获取cls_token [batch, dim]
        cls_tokens, _ = unpack(x, ps, 'b * d')

        # 通过分类头得到预测结果
        return self.mlp_head(cls_tokens)


if __name__ == '__main__':
    # 示例用法
    v = ViT(
        seq_len=256,  # 输入序列长度
        patch_size=16,  # 每个patch的长度
        num_classes=1000,  # 分类类别数
        dim=1024,  # 嵌入维度
        depth=6,  # Transformer层数
        heads=8,  # 注意力头数
        mlp_dim=2048,  # MLP隐藏层维度
        dropout=0.1,  # Transformer内部dropout
        emb_dropout=0.1  # 嵌入层dropout
    )

    # 模拟输入数据 [batch_size, channels, sequence_length]
    time_series = torch.randn(4, 3, 256)
    # 前向传播得到分类logits [batch_size, num_classes]
    logits = v(time_series)  # (4, 1000)