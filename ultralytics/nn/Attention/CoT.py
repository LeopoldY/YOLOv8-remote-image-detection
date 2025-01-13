import torch
from torch import flatten, nn
from torch.nn import functional as F
 
 
import torch
from torch import flatten, nn
from torch.nn import functional as F

class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.kernel_size = kernel_size  # 卷积核大小
        
        # 关键信息嵌入层，使用分组卷积提取特征
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),  # 归一化层
            nn.ReLU()  # 激活函数
        )
        # 值信息嵌入层，使用1x1卷积进行特征转换
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)  # 归一化层
        )
        
        # 注意力机制嵌入层，先降维后升维，最终输出与卷积核大小和通道数相匹配的特征
        factor = 4  # 降维比例
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),  # 归一化层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)  # 升维匹配卷积核形状
        )
 
    def forward(self, x):
        bs, c, h, w = x.shape  # 输入特征的尺寸
        k1 = self.key_embed(x)  # 应用关键信息嵌入
        v = self.value_embed(x).view(bs, c, -1)  # 应用值信息嵌入，并展平
        
        y = torch.cat([k1, x], dim=1)  # 将关键信息和原始输入在通道维度上拼接
        att = self.attention_embed(y)  # 应用注意力机制嵌入层
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # 计算平均后展平
        
        k2 = F.softmax(att, dim=-1) * v  # 应用softmax进行标准化，并与值信息相乘
        k2 = k2.view(bs, c, h, w)  # 重塑形状与输入相同
        
        return k1 + k2  # 将两部分信息相加并返回

class CoT_Increased(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim  # 输入通道数
        self.kernel_size = kernel_size  # 卷积核大小
        
        # 关键信息嵌入层，使用分组卷积提取特征
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),  # 归一化层
            nn.ReLU()  # 激活函数
        )
        # 值信息嵌入层，使用1x1卷积进行特征转换
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)  # 归一化层
        )
        
        # 注意力机制嵌入层，先降维后升维，最终输出与卷积核大小和通道数相匹配的特征
        factor = 4  # 降维比例
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),  # 归一化层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)  # 升维匹配卷积核形状
        )

        self.attention_embed2 = nn.Sequential(
            nn.Conv2d(dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),  # 归一化层
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dim // factor, kernel_size * kernel_size * dim, 1)  # 升维匹配卷积核形状
        )

    def forward(self, x):
        bs, c, h, w = x.shape  # 输入特征的尺寸

        k1 = self.key_embed(x)  # 应用关键信息嵌入 shape [bs, c, h, w]
        v = self.value_embed(x).view(bs, c, -1)  # 应用值信息嵌入，并展平 shape [bs, c, h*w]
        
        y = torch.cat([k1, x], dim=1)  # 将关键信息和原始输入在通道维度上拼接
        
        att = self.attention_embed(y)  # 应用注意力机制嵌入层
        k1_att = self.attention_embed2(k1)  # 应用注意力机制嵌入层2 用于和att融合

        att = k1_att + att # 局部残差连接

        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # 计算平均后展平

        k2 = F.softmax(att, dim=-1) * v  # 应用softmax进行标准化，并与值信息相乘
        k2 = k2.view(bs, c, h, w)  # 重塑形状与输入相同
        out = k2 + x # 全局残差连接
        return out  # 将两部分信息相加并返回