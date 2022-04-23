import torch.nn as nn
import torch
from torch.autograd import Function

#域自适应
# 梯度反转层代码的实现
"""
最小化分类损失用于准确分类，最大化域分类损失用于混淆目标域数据与源域数据
grl该层用于特征提取网络与域分类网络之间，反向传播过程中实现梯度取反，进而构造出类似于GAN的对抗损失，又通过该层避免了GAN的两个阶段的训练过程
GAN从某种意义上实现了域与域之间的像素级别自适应，而GRL则实现了域与域之间的特征级别自适应

https://zhuanlan.zhihu.com/p/109051269
"""
class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度翻转
        output = grad_output.neg() * ctx.alpha

        return output, None

class DA_LSTM(nn.Module):

    def __init__(self, emb_dim, hid_dim, output_dim, n_layers, dropout, bias):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False, batch_first=True,
                           bias=bias)
        #辨别器 用来做二分类任务  分辨样本是source还是target  最终的y值是domain标签0或者是1
        self.discriminator = nn.Sequential(
            nn.Linear(hid_dim, 64),
            nn.Linear(64, output_dim)
        )

    def forward(self, input, alpha):
        # output的值是source和target分别对应的输出
        output, (hidden, cell) = self.rnn(input)
        # 通过对两个域的输入求均值，获取两个域共同的特征
        y = GRL.apply(torch.mean(output, dim=1), alpha)
        y = self.discriminator(y)
        # dim 1 代表列
        # 返回一个只经过LSTM没有经过线性层处理的输出
        # y是经过梯度反转得到的结果，实现域的自适应
        return torch.mean(output, dim=1), y